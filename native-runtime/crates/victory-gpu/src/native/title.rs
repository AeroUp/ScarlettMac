//! What the title-facing objects (`queue.rs`, `fence.rs`, `list.rs`) need from the native host,
//! generic over the backend: queues and fences by the title's object address, title lists cut at
//! memory-value waits, and `hold.rs`'s per-queue FIFO of held work in front of
//! [`NativeHost::execute`]. The macOS glue (`glue.rs`) instantiates it once with Metal.
//!
//! Ordering. Work on one title queue is one backend queue, so submissions run in the order they
//! were made. A list segment that waits for a memory value another queue's list writes stays on
//! the CPU side until the value is there (`hold.rs`; the GPU's own writes into title memory are
//! visible to the CPU as they land, there is no copy); everything else goes to the GPU at once and
//! is never waited for. Fence signals that need no CPU decision are encoded on the GPU
//! (`MTLSharedEvent`); behind held work they are held too, so that they stay in order. Fence
//! waits stay on the CPU until the backend reports completion. A committed producer can be
//! delayed or fail asynchronously; submission is not proof that an event will be signalled.

use super::host::NativeHost;
use super::list::{Binding, ComputePso, NativeList};
use super::memory::PoolTable;
use crate::hold::{Held, Head, Item, ValueWait, World};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use victory_backend::resources::{Format, GraphicsBackend, PresentBackend};

/// A wait for a memory value (`Wait32BitValueX`, `Wait64BitValueX`): the value at `address`
/// (`bytes` 4 or 8, masked) compared with `reference` by a D3D12 comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub address: u64,
    pub bytes: u8,
    pub comparison: u32,
    pub reference: u64,
    pub mask: u64,
}

impl From<Wait> for ValueWait {
    fn from(w: Wait) -> ValueWait {
        ValueWait {
            address: w.address,
            bytes: w.bytes,
            comparison: w.comparison,
            reference: w.reference,
            mask: w.mask,
        }
    }
}

/// A title list: the recording [`NativeList`] plus the segments cut off it at waits recorded after
/// commands, each with the waits at its start.
pub struct TitleList<B: GraphicsBackend> {
    done: Vec<(NativeList<B>, Vec<Wait>)>,
    /// The list the calls record into, and the waits at its start.
    pub list: NativeList<B>,
    waits: Vec<Wait>,
    /// Waits recorded: the ones that cut a segment.
    pub cuts: u32,
    /// The title pipeline object `SetPipelineState` named last (0: none), and the primitive
    /// topology set (the console's `D3D_PRIMITIVE_TOPOLOGY`).
    pub pipeline: usize,
    pub topology: u32,
    /// The formats of the targets `OMSetRenderTargets` bound last.
    pub targets: TargetFormats,
    /// Whether every target `OMSetRenderTargets` bound last has a host texture. When one has
    /// none the binding is refused (a partial one would renumber the slots) and draws are
    /// dropped until the title binds a set the host can make.
    pub targets_ready: bool,
    /// The strides of the bound vertex buffers by slot, which a guessed pipeline is made from at
    /// its draw, and the title resources bound as targets (for the target trace).
    pub strides: Vec<u32>,
    pub target_keys: Vec<usize>,
}

/// The formats of a draw's bound colour and depth targets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TargetFormats {
    pub color: Vec<Format>,
    pub depth: Option<Format>,
}

impl<B: GraphicsBackend> Default for TitleList<B> {
    fn default() -> Self {
        TitleList {
            done: Vec::new(),
            list: NativeList::default(),
            waits: Vec::new(),
            cuts: 0,
            pipeline: 0,
            topology: 0,
            targets: TargetFormats::default(),
            targets_ready: true,
            strides: Vec::new(),
            target_keys: Vec::new(),
        }
    }
}

impl<B: GraphicsBackend> TitleList<B> {
    /// `Reset`.
    pub fn reset(&mut self) {
        *self = TitleList::default();
    }

    /// A wait recorded at the current position: at the start of the segment if it has no commands
    /// yet, else the segment ends here and the wait starts the next one (state carries over).
    pub fn wait(&mut self, w: Wait) {
        if !self.list.is_empty() {
            let segment = self.list.cut();
            self.done
                .push((segment, std::mem::replace(&mut self.waits, vec![w])));
            self.cuts += 1;
        } else {
            self.waits.push(w);
        }
    }

    /// The segments a submission runs, in order (copies: the title may reset and re-record).
    /// The copies clone `Rc`s that the title's recording threads clone and drop under the title's
    /// lock, and `Rc` counts are not atomic: `_locked` is there so that only a holder of the title
    /// (inside `glue::with_title`) can take them.
    pub fn segments<P: PresentBackend>(&self, _locked: &Title<P>) -> Vec<(NativeList<B>, Vec<Wait>)> {
        self.done
            .iter()
            .map(|(l, w)| (l.snapshot(), w.clone()))
            .chain([(self.list.snapshot(), self.waits.clone())])
            .collect()
    }

    /// [`Self::segments`] that hands the lists over instead of copying them: the caller owns what
    /// it runs and the title's next recording (after its `Reset`) starts from empty. D3D12 makes
    /// `ExecuteCommandLists` the last call on a list before `Reset`, so nothing recorded after it
    /// belongs to this submission. Copies made a 30,000-command world frame's submission spend its
    /// host time in `malloc` (`sample` of the render thread, runs/20261128-110159-0DwmVl) and held
    /// the title's lock while every title thread waited for it.
    pub fn take_segments<P: PresentBackend>(&mut self, _locked: &Title<P>) -> Vec<(NativeList<B>, Vec<Wait>)> {
        let mut out: Vec<(NativeList<B>, Vec<Wait>)> = std::mem::take(&mut self.done);
        out.push((std::mem::take(&mut self.list), std::mem::take(&mut self.waits)));
        out
    }
}

/// Counters for the frame line and the tests.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    /// Segments queued by `ExecuteCommandLists`.
    pub segments: u64,
    /// Groups of segments handed to the GPU (a piece is what `hold` runs together).
    pub pieces: u64,
    /// Pieces the host refused (an address outside the mapped pools, ...): logged, their fences
    /// are signalled all the same.
    pub refused: u64,
    /// Segments that waited for a memory value before they ran.
    pub waited: u64,
    /// Segment heads the safety valve ran without their waits (`VALVE`): a wait nothing wrote.
    pub forced: u64,
    /// Empty segments left out with `VICTORY_SKIP_EMPTY_LISTS`: each would have cost a command
    /// buffer and its fence, and done nothing.
    pub skipped: u64,
    /// Title textures detiled (again) because their memory changed.
    pub detiles: u64,
    /// Textures whose layout the host cannot detile (counted once per content).
    pub undetilable: u64,
}

/// What a title pipeline object became on the host.
pub enum Pipe<B: GraphicsBackend> {
    Compute(Rc<ComputePso<B>>),
    /// With the argument layout of its translated shaders (none: the title's root as it is).
    Render(Rc<B::RenderPipeline>, Option<Rc<Binding>>),
    /// A guessed graphics pipeline (`VICTORY_GUESS_GRAPHICS`): made, per target formats and
    /// vertex strides, at its draws.
    Guess(Rc<Guessed<B>>),
    /// A guessed graphics slot without a vertex producer.
    #[cfg(target_os = "macos")]
    Skipped,
    /// The host refused it (the reason is logged and counted once); work using it is dropped.
    Refused,
}

impl<B: GraphicsBackend> Clone for Pipe<B> {
    fn clone(&self) -> Self {
        match self {
            Pipe::Compute(p) => Pipe::Compute(p.clone()),
            Pipe::Render(p, b) => Pipe::Render(p.clone(), b.clone()),
            Pipe::Guess(g) => Pipe::Guess(g.clone()),
            #[cfg(target_os = "macos")]
            Pipe::Skipped => Pipe::Skipped,
            Pipe::Refused => Pipe::Refused,
        }
    }
}

/// The shaders of a guessed pipeline and the host pipelines made from them so far.
pub struct Guessed<B: GraphicsBackend> {
    pub vs: Arc<victory_shader::emit::HostShader>,
    pub ps: Option<Arc<victory_shader::emit::HostShader>>,
    pub name: String,
    /// The pipeline's state with the targets of its draw ([`Graphics::for_draw`] completes it).
    pub state: super::state::Graphics,
    /// The vertex shader's container, translated again for rectangle lists at the first such draw.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) vs_source: Arc<crate::pipeline::Shader>,
    /// By the formats and strides they were made for.
    pub made: std::cell::RefCell<HashMap<String, Pipe<B>>>,
}

/// An operation that must keep its place between list submissions in the held FIFO.
enum QueuedAction<B: PresentBackend> {
    Signal(Rc<B::Queue>, Rc<B::Fence>, u64),
    /// A present: what to show, and when it was queued (a present may not hold the queue for the
    /// display; see [`Title::pump`]).
    Present(Rc<B::Queue>, Rc<B::Surface>, Vec<Rc<B::Texture>>, std::time::Instant),
}

type PendingEpochs<F> = VecDeque<(u64, Vec<(Rc<F>, u64)>)>;

/// How long a present may wait for a drawable before the frame is dropped: a display that has
/// stopped consuming frames (an occluded window, a sleeping display) must not hold the queue, and
/// a visible one hands a drawable over in a frame or two at most.
const PRESENT_WAIT: std::time::Duration = std::time::Duration::from_millis(500);

/// What a present gets once the display has already starved one: enough for a drawable that is
/// already on its way, nothing for one that is not.
const PRESENT_STARVED_WAIT: std::time::Duration = std::time::Duration::from_millis(16);

/// Whether the present queued at `since` may be dropped: the display was starved by an earlier
/// present and this one has had its short window, or it has waited past [`PRESENT_WAIT`].
fn present_gives_up(starved: bool, since: std::time::Instant, now: std::time::Instant) -> bool {
    let wait = if starved { PRESENT_STARVED_WAIT } else { PRESENT_WAIT };
    now.saturating_duration_since(since) >= wait
}

pub struct Title<B: PresentBackend> {
    pub host: NativeHost<B>,
    /// Host pipelines by title pipeline object and the strides their vertex streams step by
    /// ([`Self::pipeline_streams`]; 0 for a pipeline without stage-in inputs).
    pub pipelines: HashMap<(usize, u64), Pipe<B>>,
    /// Host pipelines of the console's pipelines (whose state names no target formats) by title
    /// pipeline object, the targets of the draw ([`super::console_state`]) and the strides its
    /// streams step by.
    pub console_pipes: HashMap<(usize, TargetFormats, u64), Pipe<B>>,
    /// Each described pipeline's input streams as (slot, where its elements end, four-byte
    /// aligned), recorded when its host pipeline is first made. D3D12 steps a stream by the
    /// vertex view's `StrideInBytes`, which the title may bind differently from one draw to the
    /// next of the same pipeline, and Metal's vertex descriptor holds the stride: the signature
    /// of those strides is part of the host pipeline's cache key, so each stride gets a pipeline
    /// whose descriptor was made for it.
    pub pipeline_streams: HashMap<usize, Vec<(u32, u32)>>,
    /// Host pipelines of compute pipelines dispatched with thread counts (`DispatchX`), whose
    /// shader is translated for that, by title pipeline object.
    pub thread_pipes: HashMap<usize, Pipe<B>>,
    /// What the shader path did, and refused.
    pub shaders: super::shaders::ShaderStats,
    /// How long a new pipeline may hold the frame while Metal compiles it (`glue::compile_wait`):
    /// the archive's window, never a frame's translation budget, because the wait runs on a title
    /// thread under the host's lock. Past it the pipeline is put off (`ShaderStats::put_off`).
    pub compile_wait: std::time::Duration,
    /// `VICTORY_SKIP_EMPTY_LISTS`: leave out segments with no commands and no waits at submission
    /// (they would cost a command buffer and its fence and do nothing). A segment that waits is
    /// kept even when it has no commands: its waits still have to hold the queue.
    pub skip_empty_lists: bool,
    /// Sample textures straight from title memory where the layout allows it
    /// ([`super::textures::Title::zero_copy_texture`]): on unless `VICTORY_TEXTURE_ZERO_COPY=0`,
    /// and off in tests unless one turns it on (they fill title memory with solid bytes, which a
    /// tiled or decoded copy has to read as that colour).
    pub zero_copy: bool,
    /// Give a decoded copy's title pages back to the OS once nothing has used it for a while
    /// (`super::textures::release_pages_on`): on unless `VICTORY_TEXTURE_RELEASE_PAGES=0`, and off
    /// in tests unless one turns it on (a title access to a released page is not caught yet).
    pub release_pages: bool,
    /// Give a *streamed* copy's title pages back as soon as it is uploaded
    /// (`super::textures::release_streamed_on`): the title's streamer put those bytes there for the
    /// GPU, the Metal texture is the only copy anything reads, and the title's own access faults
    /// them back (`Title::release_pages`). The duplicate copy a streamed texture keeps in title
    /// memory is what fills an 18 GB Mac's swap.
    pub release_streamed: bool,
    pub(super) libraries: super::shaders::Libraries<B>,
    pub(super) textures: super::textures::Textures<B>,
    /// The byte budget of host texture copies ([`Self::keep_to_budget`]).
    pub(super) budget: super::textures::Budget,
    /// GPU-written textures waiting to be copied back into title memory.
    pub(super) back: super::textures::Back<B>,
    /// Sampled textures realized but not yet uploaded once.
    pub(super) unloaded: std::collections::HashSet<usize>,
    /// Submission sync serial advanced by `Title::sync_jobs_within`, used to expire `hot_keys`
    /// after `HOT` submissions even when a round-robin probe budget cannot visit all of them.
    pub(super) sync_serial: u64,
    /// Keys of textures currently in the fast per-submission probe set (`hot_until > 0`).
    pub(super) hot_keys: Vec<usize>,
    /// Round-robin cursor into `hot_keys` when the time budget ends before all are probed.
    pub(super) hot_cursor: usize,
    /// Keys of textures whose last upload was blank, probed after `hot_keys` until written.
    pub(super) blank_keys: Vec<usize>,
    pub(super) blank_cursor: usize,
    /// Reusable scratch buffer for hot-texture probes (avoids per-submission allocations).
    pub(super) probe_buf: Vec<u8>,
    /// Where `Present` shows frames, if the app gave a layer.
    pub(super) surface: Option<Rc<B::Surface>>,
    /// A present was dropped for want of a drawable and none has been shown since: later presents
    /// are dropped at once instead of waiting for the display ([`Self::pump`]).
    pub(super) present_starved: bool,
    /// Sampled textures left pending because the copy budget was spent when their descriptors were
    /// written: made at the submission that names them ([`Self::keep_named`]), not before.
    pub(super) deferred: std::collections::HashSet<usize>,
    /// Decoded copies whose title pages were given back to the OS (`Title::release_pages`), with
    /// the bytes released: the Metal texture holds the only copy while they are out.
    pub(super) released: std::collections::HashMap<usize, u64>,
    /// Copies the budget (or the idle sweep) gave up, by `textures::memory_identity` and the
    /// submission that gave them up: one made again over the same bytes within
    /// `textures::named_keep()` submissions is counted as churn (the decode and upload the budget
    /// caused again).
    pub(super) evicted: std::collections::HashMap<usize, (u64, u64)>,
    /// Title textures no view has named yet.
    pub(super) pending: HashMap<usize, super::textures::TextureInfo>,
    /// Textures placed in title memory (no resource object), by address: their key and pool.
    pub(super) placed: std::collections::BTreeMap<u64, (usize, usize)>,
    /// The largest placed texture's size: how far before an address a placed texture over it can
    /// start ([`Self::texture_placed`]).
    pub(super) placed_reach: u64,
    /// Placed textures indexed by physical `(pool_id, pool_offset)`: `(key, chunk_len)` entries,
    /// so VA aliases over the same pool pages detect overlaps across addresses too.
    pub(super) placed_pools: HashMap<usize, std::collections::BTreeMap<u64, Vec<(usize, u64)>>>,
    /// Physical pool chunks indexed in [`Self::placed_pools`] per texture key, for O(1) removal.
    pub(super) placed_chunks: HashMap<usize, Vec<(usize, u64, u64)>>,
    /// The largest physical chunk length in [`Self::placed_pools`].
    pub(super) placed_pool_reach: u64,
    /// Title textures whose host texture is another's, laid out identically over the same memory
    /// ([`Self::realize`]): that other key (in `textures`), and their own description.
    pub(super) copies_of: HashMap<usize, (usize, super::textures::TextureInfo)>,
    /// Keys in `textures` by the title memory they start at.
    pub(super) at_memory: HashMap<usize, Vec<usize>>,
    /// Placed copies another placement covered, mapped to the `gpu_epoch` when covered, given up
    /// once nothing uses them (or under budget once all work from that epoch has finished on the GPU)
    /// ([`Self::give_up_covered`]).
    pub(super) covered: HashMap<usize, u64>,
    /// Covered placed copies evicted under a budget (`give_up_covered`), so if a descriptor
    /// mirror or table scan realizes one again while a newer overlapping placement is still live,
    /// it is marked covered again instead of pinning duplicate memory permanently.
    pub(super) covered_evicted: std::collections::HashSet<usize>,
    /// Live sampled textures whose leading unmapped fine mips were trimmed on creation, plus the
    /// [`PoolTable::generation`] last checked for newly mapped fine mips (`upgrade_trimmed_textures`).
    pub(super) trimmed: std::collections::HashSet<usize>,
    pub(super) trimmed_pool_gen: u64,
    /// Monotonic GPU submission epoch advanced on each executed piece, and the highest epoch whose
    /// in-flight work has completed across all queues.
    pub(super) gpu_epoch: u64,
    pub(super) retired_epoch: u64,
    pub(super) queue_fences: HashMap<usize, (Rc<B::Fence>, u64)>,
    pub(super) pending_epochs: PendingEpochs<B::Fence>,
    /// Resource objects whose host texture placed views name too: by object key, the address and
    /// key of each placed view sharing it.
    pub(super) shared: HashMap<usize, Vec<(u64, usize)>>,
    queues: HashMap<usize, Rc<B::Queue>>,
    fences: HashMap<usize, Rc<B::Fence>>,
    pub(super) held: Held<Rc<NativeList<B>>>,
    /// Operations held behind work, indexed by the token in a FIFO Signal item.
    actions: HashMap<usize, QueuedAction<B>>,
    next_action: usize,
    /// The highest value each fence was told to reach, and when that changed (diagnostics).
    requested: HashMap<usize, (u64, std::time::Instant)>,
    /// Unique wait tokens retain fence instances independently of title address reuse.
    waiting: HashMap<usize, Rc<B::Fence>>,
    pub stats: Stats,
}

impl<B: PresentBackend> Title<B> {
    pub fn new(backend: B, memory: Arc<PoolTable>) -> Self {
        let host = NativeHost::with_memory(backend, memory);
        host.collect_written(std::env::var("VICTORY_TEXTURE_BACK").as_deref() == Ok("1"));
        Title {
            host,
            pipelines: HashMap::new(),
            console_pipes: HashMap::new(),
            pipeline_streams: HashMap::new(),
            thread_pipes: HashMap::new(),
            shaders: Default::default(),
            compile_wait: std::time::Duration::MAX,
            skip_empty_lists: crate::switch::on("VICTORY_SKIP_EMPTY_LISTS"),
            zero_copy: cfg!(not(test)) && crate::native::textures::zero_copy_on(),
            release_pages: cfg!(not(test)) && crate::native::textures::release_pages_on(),
            release_streamed: cfg!(not(test)) && crate::native::textures::release_streamed_on(),
            libraries: Default::default(),
            textures: Default::default(),
            budget: Default::default(),
            back: Default::default(),
            unloaded: std::collections::HashSet::new(),
            sync_serial: 0,
            hot_keys: Vec::new(),
            hot_cursor: 0,
            blank_keys: Vec::new(),
            blank_cursor: 0,
            probe_buf: Vec::new(),
            surface: None,
            present_starved: false,
            deferred: Default::default(),
            released: Default::default(),
            evicted: Default::default(),
            pending: HashMap::new(),
            placed: Default::default(),
            shared: HashMap::new(),
            placed_reach: 0,
            placed_pools: HashMap::new(),
            placed_chunks: HashMap::new(),
            placed_pool_reach: 0,
            copies_of: HashMap::new(),
            at_memory: HashMap::new(),
            covered: Default::default(),
            covered_evicted: Default::default(),
            trimmed: Default::default(),
            trimmed_pool_gen: 0,
            gpu_epoch: 0,
            retired_epoch: 0,
            queue_fences: HashMap::new(),
            pending_epochs: VecDeque::new(),
            queues: HashMap::new(),
            fences: HashMap::new(),
            requested: HashMap::new(),
            waiting: HashMap::new(),
            held: Held::new(),
            actions: HashMap::new(),
            next_action: 0,
            stats: Stats::default(),
        }
    }

    fn queue(&mut self, key: usize) -> Result<Rc<B::Queue>, String> {
        if let Some(q) = self.queues.get(&key) {
            return Ok(q.clone());
        }
        let q = Rc::new(self.host.backend.create_queue()?);
        self.queues.insert(key, q.clone());
        Ok(q)
    }

    /// The backend fence of title fence `key`, made with `initial` the first time.
    pub fn fence(&mut self, key: usize, initial: u64) -> Result<Rc<B::Fence>, String> {
        if let Some(f) = self.fences.get(&key) {
            return Ok(f.clone());
        }
        let f = Rc::new(self.host.backend.create_fence(initial)?);
        self.fences.insert(key, f.clone());
        Ok(f)
    }

    /// The title released fence `key`.
    pub fn forget_fence(&mut self, key: usize) {
        self.fences.remove(&key);
        self.requested.remove(&key);
    }

    /// The title released queue `key`; held work stays held (the title does not do it).
    pub fn forget_queue(&mut self, key: usize) {
        self.queues.remove(&key);
    }

    /// Every title fence with a backend fence and the value it has completed: what the title's own
    /// copy of the value (read from memory) is raised to.
    pub fn completed(&mut self) -> Vec<(usize, u64)> {
        let values = self
            .fences
            .iter()
            .map(|(k, f)| (*k, self.host.backend.completed(f)))
            .collect();
        // After the values are read: the work they cover is done, so what it wrote is in title
        // memory before the title sees them.
        self.copy_back();
        values
    }

    /// Fence `key`'s completed value (made with `initial` if it has no backend fence yet).
    pub fn fence_value(&mut self, key: usize, initial: u64) -> u64 {
        let value = match self.fence(key, initial) {
            Ok(f) => self.host.backend.completed(&f).max(initial),
            Err(_) => initial,
        };
        // As in `completed`: read first, then copy back.
        self.copy_back();
        value
    }

    /// `ID3D12Fence::Signal`: the value is reached now; queues waiting for it go on.
    pub fn signal_cpu(&mut self, key: usize, initial: u64, value: u64) {
        if let Ok(f) = self.fence(key, initial) {
            self.host.backend.signal(&f, value);
        }
        self.pump();
    }

    /// Whether fence `key` reached `value`; pumps held work first, so that a wait for a fence
    /// signalled behind a memory-value wait can make progress when the caller polls.
    pub fn try_wait(&mut self, key: usize, initial: u64, value: u64) -> bool {
        self.pump();
        self.fence_value(key, initial) >= value
    }

    /// `ExecuteCommandLists` on title queue `queue`: the lists' segments join what the queue holds
    /// and whatever may run goes to the GPU.
    pub fn submit(&mut self, queue: usize, segments: Vec<(NativeList<B>, Vec<Wait>)>) {
        let named = self.keep_named(segments.iter().map(|(l, _)| l));
        self.sync_dirty();
        self.submit_synced(queue, segments);
        self.let_named_go(named);
    }

    /// [`Self::submit`] once the textures are in step with title memory ([`Self::sync_dirty`]).
    pub fn submit_synced(&mut self, queue: usize, segments: Vec<(NativeList<B>, Vec<Wait>)>) {
        for (list, waits) in segments {
            self.stats.segments += 1;
            if !keeps_segment(self.skip_empty_lists, &list, &waits) {
                self.stats.skipped += 1;
                continue;
            }
            let waits: Vec<ValueWait> = waits.into_iter().map(Into::into).collect();
            self.held.push(queue, Item::Segment(Rc::new(list), waits));
        }
        self.pump();
    }

    /// `ID3D12CommandQueue::Signal`: behind work the queue still holds, else on the GPU at once
    /// (after the work already submitted to its queue).
    pub fn signal_queue(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let r = self.requested.entry(fence).or_insert((0, std::time::Instant::now()));
        if value > r.0 {
            *r = (value, std::time::Instant::now());
        }
        if self.held.holds_work(queue) {
            let n = self.next_action;
            self.next_action += 1;
            let (Ok(q), Ok(f)) = (self.queue(queue), self.fence(fence, initial)) else { return };
            self.actions.insert(n, QueuedAction::Signal(q, f, value));
            self.held.push(queue, Item::Signal(n, 0));
        } else {
            self.signal_gpu(queue, fence, initial, value);
        }
        self.pump();
    }

    fn signal_gpu(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let (Ok(q), Ok(f)) = (self.queue(queue), self.fence(fence, initial)) else {
            return;
        };
        if let Err(e) = self.host.execute(&q, &[], &[], &[(&*f, value)]) {
            tracing::warn!(error = %e, "a fence signal was not queued");
        }
    }

    /// Queue waits remain on the CPU until completion, including committed producers.
    pub fn wait_queue(&mut self, queue: usize, fence: usize, initial: u64, value: u64) {
        let Ok(f) = self.fence(fence, initial) else { return };
        if self.held.holds_work(queue) || self.host.backend.completed(&f) < value {
            let token = self.next_action;
            self.next_action += 1;
            self.waiting.insert(token, f);
            self.held.push(queue, Item::FenceWait(token, value));
        }
        self.pump();
    }

    /// Runs the heads of held queues that may run, until none can.
    pub fn pump(&mut self) {
        if self.held.is_empty() {
            return;
        }
        let Title {
            host,
            queues,
            waiting,
            held,
            actions,
            stats,
            back,
            budget,
            gpu_epoch,
            queue_fences,
            present_starved: starved,
            ..
        } = self;
        let memory = host.memory();
        let read = |address: u64, bytes: u8| read_title(&memory, address, bytes);
        let fence = |f: usize| {
            waiting
                .get(&f)
                .map_or(0, |fence| host.backend.completed(fence))
        };
        let mut run = |queue: usize, lists: Vec<Rc<NativeList<B>>>| {
            stats.pieces += 1;
            let Some(q) = queues.get(&queue).cloned() else {
                return;
            };
            let refs: Vec<&NativeList<B>> = lists.iter().map(|l| &**l).collect();
            let ours = back.next(host, queue);
            let epoch_sig = if budget.limit > 0 {
                let entry = match queue_fences.entry(queue) {
                    std::collections::hash_map::Entry::Occupied(e) => Some(e.into_mut()),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        host.backend.create_fence(0).ok().map(|f| e.insert((Rc::new(f), 0)))
                    }
                };
                entry.map(|(f, v)| {
                    *v += 1;
                    (f.clone(), *v)
                })
            } else {
                None
            };
            let mut signals: Vec<(&B::Fence, u64)> = ours.iter().map(|(f, v)| (&**f, *v)).collect();
            if let Some((f, v)) = &epoch_sig {
                signals.push((&**f, *v));
            }
            if let Err(e) = host.execute(&q, &refs, &[], &signals) {
                stats.refused += 1;
                tracing::warn!(error = %e, "a piece of title work was refused");
                if let Some((f, v)) = &epoch_sig {
                    host.backend.signal(f, *v);
                }
            }
            *gpu_epoch += 1;
            back.submitted(host, ours);
        };
        // A present whose drawable does not come is dropped once it has waited this long: it must
        // not hold the queue (see the arm below).
        let mut signal = |n: usize, _: u64| {
            match actions.get(&n) {
                Some(QueuedAction::Signal(q, f, value)) => {
                    if let Err(e) = host.execute(q, &[], &[], &[(&**f, *value)]) {
                        tracing::warn!(error = %e, "a held fence signal was not queued");
                    }
                }
                Some(QueuedAction::Present(queue, surface, planes, since)) => {
                    let planes: Vec<&B::Texture> = planes.iter().map(|t| &**t).collect();
                    match host.backend.try_present(queue, surface, &planes) {
                        // `PresentX` is a display-side operation: it is not part of the queue's
                        // GPU timeline, so a display that hands out no drawable must not hold the
                        // work submitted after it. It did: a present at a queue's head waited
                        // 7-9 s for a drawable (the window's compositor had stopped consuming
                        // frames), the frame's fence signal and its cross-queue barrier writes
                        // queued behind it, and every other queue stayed on its barrier word while
                        // the title's frame thread waited on that fence -- frames of 6 s with the
                        // CPU at 0.28 cores and the GPU at 22% (run of 27 November 22:08). Past
                        // the display's window the frame is dropped, as one the display cannot
                        // show, and later presents are dropped at once until one is shown again.
                        Ok(false) if present_gives_up(*starved, *since, std::time::Instant::now()) => {
                            if !*starved {
                                *starved = true;
                                static DROPPED: AtomicU64 = AtomicU64::new(0);
                                let n = DROPPED.fetch_add(1, Ordering::Relaxed) + 1;
                                if n.is_power_of_two() {
                                    tracing::warn!(
                                        n,
                                        waited = since.elapsed().as_secs_f64(),
                                        "a present waited for a drawable past the display's window and was dropped (the frame is not shown, the queue goes on)"
                                    );
                                }
                            }
                        }
                        Ok(false) => return false,
                        Ok(true) => {
                            if std::mem::replace(starved, false) {
                                tracing::info!("a present was shown again: the display hands out drawables");
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "a held present was not queued");
                            // No drawable now or no worker: the display is not showing frames, and
                            // the next one must not wait for it either.
                            *starved = true;
                        }
                    }
                }
                None => {}
            }
            actions.remove(&n);
            true
        };
        // The safety valve the Windows pump has and this one did not (`queue::VALVE`): a head whose
        // memory waits nothing satisfies -- a write we do not make, a pass dropped for a refused
        // shader among the writers -- holds its queue for good, and the title thread waiting on that
        // queue enters a watchdog scope it never leaves (300 s; run `20261128-133911-YLju97` died
        // there with its last ten minutes silent). Past `VALVE` the head runs anyway and names what
        // it waited for. Well past the longest wait that resolves (a world frame's barrier, 3 to
        // 20 s), so a frame in flight is never let go early.
        for (queue, waits) in valve_forced(held, Self::VALVE) {
            stats.forced += 1;
            let limit = Self::VALVE;
            tracing::warn!(
                queue = format_args!("{queue:#x}"),
                ?waits,
                "held work runs without its waits after {limit:?} (safety valve: nothing wrote what it waited for)"
            );
        }
        let done = held.pump(&mut World {
            read: &read,
            fence: &fence,
            run: &mut run,
            signal: &mut signal,
        });
        stats.waited += done.waited.len() as u64;
        for token in done.retired_waits {
            waiting.remove(&token);
        }
    }

    /// How long a queue's head may wait for a memory value before [`Self::pump`] runs it anyway:
    /// longer than any wait that resolves (a world frame's cross-queue barrier, 3 to 20 s in the
    /// runs of 28 November), far short of the title's 300 s watchdog scope.
    pub const VALVE: Duration = Duration::from_secs(60);

    /// The backend queue of each title queue made so far.
    pub fn backend_queues(&self) -> impl Iterator<Item = (usize, &B::Queue)> {
        self.queues.iter().map(|(k, q)| (*k, &**q))
    }

    /// Queues whose head has been held longer than `limit`: a line each, with the waits that do
    /// not hold (address, comparison, reference, mask, current value) and, for a head that is no
    /// segment, what it is waiting for: a present's drawable, a fence signal or a fence wait.
    pub fn stuck(&self, limit: std::time::Duration) -> Vec<String> {
        let mut out: Vec<String> = self
            .requested
            .iter()
            .filter_map(|(k, &(want, since))| {
                let f = self.fences.get(k)?;
                let done = self.host.backend.completed(f);
                (done < want && since.elapsed() >= limit)
                    .then(|| format!("fence {k:#x}: signal of {want} queued {:.1?} ago, GPU at {done}", since.elapsed()))
            })
            .collect();
        let memory = self.host.memory();
        let read = |address: u64, bytes: u8| read_title(&memory, address, bytes);
        out.extend(self
            .held
            .blocked(&read)
            .into_iter()
            .filter(|(.., since, _, _)| *since >= limit)
            .map(|(q, items, since, unmet, head)| {
                let waits: Vec<String> = unmet
                    .iter()
                    .map(|w| {
                        format!(
                            "{:#x} cmp {} ref {:#x} mask {:#x} now {:?}",
                            w.address,
                            w.comparison,
                            w.reference,
                            w.mask,
                            read(w.address, w.bytes)
                        )
                    })
                    .collect();
                let what = match head {
                    Head::Segment => String::new(),
                    Head::Signal(token) => match self.actions.get(&token) {
                        Some(QueuedAction::Present(..)) => ", a present (no drawable for it yet)".into(),
                        Some(QueuedAction::Signal(..)) => ", a fence signal".into(),
                        None => String::new(),
                    },
                    Head::FenceWait(_) => ", a wait on a fence value".into(),
                };
                format!("queue {q:#x}: {items} held, head for {since:.1?}{what}, waits {waits:?}")
            }));
        out
    }

    /// Gives the host the app's drawable layer (`CAMetalLayer` on Metal, by address): frames the
    /// title presents are shown there from now on.
    pub fn set_surface(&mut self, layer: *mut std::ffi::c_void) -> Result<(), String> {
        self.surface = Some(Rc::new(self.host.backend.create_surface(layer)?));
        Ok(())
    }

    /// `PresentX`: show title textures `keys` after all earlier work on `queue`, including held
    /// work: the first under the others, each later one blended over by its alpha.
    /// `Ok(true)` means accepted (possibly deferred); deferred errors are logged by pump, and a
    /// present the display hands no drawable for is dropped after [`PRESENT_WAIT`] rather than
    /// holding the queue (see [`Self::pump`]).
    /// `Ok(false)` when there is no surface or the title has none of these textures.
    pub fn present(&mut self, queue: usize, keys: &[usize]) -> Result<bool, String> {
        self.pump();
        if self.surface.is_none() {
            return Ok(false);
        }
        let tex: Vec<_> = keys.iter().filter_map(|&k| self.texture_of(k)).collect();
        if tex.is_empty() {
            return Ok(false);
        }
        let q = self.queue(queue)?;
        if let Some(surface) = &self.surface {
            let n = self.next_action;
            self.next_action += 1;
            self.actions.insert(
                n,
                QueuedAction::Present(q, surface.clone(), tex, std::time::Instant::now()),
            );
            self.held.push(queue, Item::Signal(n, 0));
        }
        self.pump();
        Ok(true)
    }

    /// The surface presented to (to read the frame counter, say).
    pub fn surface(&self) -> Option<&B::Surface> {
        self.surface.as_deref()
    }

    /// Whether any queue holds work (a timer should test it again soon).
    pub fn holds(&self) -> bool {
        !self.held.is_empty()
    }

    /// Makes sure title queue `key` has a backend queue before work is held for it.
    pub fn ensure_queue(&mut self, key: usize) -> Result<(), String> {
        self.queue(key).map(|_| ())
    }
}

/// Whether a submission queues this segment. With `VICTORY_SKIP_EMPTY_LISTS`, a segment with no
/// commands and no waits is left out: it would burn a command buffer and its fence and do
/// nothing. A segment that waits is kept even with no commands: its waits still hold everything
/// after it on the queue (and that is the segment's whole effect).
fn keeps_segment<B: GraphicsBackend>(skip_empty: bool, list: &NativeList<B>, waits: &[Wait]) -> bool {
    !skip_empty || !list.is_empty() || !waits.is_empty()
}

/// The value a wait tests: from the pool when the address lies in one, else read from the
/// process's own memory (which may fault-free fail: the read is checked by the kernel).
fn read_title(memory: &PoolTable, address: u64, bytes: u8) -> Option<u64> {
    let n = usize::from(bytes.min(8));
    let at = match memory.resolve(address, n as u64) {
        Some((_, p)) => p,
        None => address as *const u8,
    };
    let mut v = [0u8; 8];
    crate::native::safe_read(at, &mut v[..n]).then(|| u64::from_le_bytes(v))
}

#[cfg(all(test, target_os = "macos"))]
mod scheduling_tests {
    use super::*;
    use victory_backend::backend::ComputeBackend;
    use victory_metal::metal::Metal;

    /// A present waits for a drawable only its window, and a starved display none: while it held
    /// the queue for 7-9 s (run of 27 November 22:08) the frame's fence signal and its cross-queue
    /// barrier writes queued behind it, the GPU idled and the title's frame thread blocked on the
    /// fence -- frames of 6 s at 0.28 CPU cores and 22% GPU.
    #[test]
    fn a_present_waits_for_a_drawable_only_its_window() {
        let since = std::time::Instant::now();
        assert!(!present_gives_up(false, since, since), "queued just now");
        assert!(!present_gives_up(false, since, since + PRESENT_WAIT / 2), "inside the window");
        assert!(present_gives_up(false, since, since + PRESENT_WAIT), "past it: the frame is dropped");
        assert!(
            present_gives_up(true, since, since + PRESENT_STARVED_WAIT),
            "a display that starved one gets only a short window for the next"
        );
        assert!(!present_gives_up(true, since, since), "and a drawable already on its way is taken");
    }

    #[test]
    fn skipping_empty_lists_keeps_segments_that_do_something_or_wait() {
        let empty = NativeList::<Metal>::default();
        let mut work = NativeList::<Metal>::default();
        work.write_value(0, 7, 4);
        let wait = Wait {
            address: 0x1000,
            bytes: 4,
            comparison: 3,
            reference: 1,
            mask: u64::from(u32::MAX),
        };
        assert!(keeps_segment(false, &empty, &[]), "off: every segment is submitted");
        assert!(!keeps_segment(true, &empty, &[]), "on: nothing to do, nothing to submit");
        assert!(keeps_segment(true, &empty, &[wait]), "its waits must still hold the queue");
        assert!(keeps_segment(true, &work, &[]), "commands are submitted");
    }

    #[test]
    fn an_empty_submission_does_not_make_a_command_buffer() {
        let Ok(metal) = Metal::new() else { return };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        title.skip_empty_lists = true;
        title.ensure_queue(1).unwrap();
        title.submit_synced(1, vec![(NativeList::default(), Vec::new())]);
        assert!(!title.held.holds_work(1));
        assert_eq!((title.stats.segments, title.stats.skipped, title.stats.pieces), (1, 1, 0));
    }

    #[test]
    fn an_empty_segment_that_waits_is_held() {
        let Ok(metal) = Metal::new() else { return };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        title.skip_empty_lists = true;
        title.ensure_queue(1).unwrap();
        let wait = Wait {
            address: 0x1000,
            bytes: 4,
            comparison: 3,
            reference: 1,
            mask: u64::from(u32::MAX),
        };
        title.submit_synced(1, vec![(NativeList::default(), vec![wait])]);
        assert!(title.held.holds_work(1), "the waits keep their place");
        assert_eq!(title.stats.skipped, 0);
    }

    #[test]
    fn signalling_a_fence_releases_all_its_queued_wait_tokens() {
        let Ok(metal) = Metal::new() else {
            eprintln!("skipping fence test: no Metal device");
            return;
        };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        for _ in 0..1024 {
            title.wait_queue(1, 10, 0, 1);
        }
        title.wait_queue(2, 20, 0, 1);
        title.pump();
        title.pump();
        assert_eq!(title.waiting.len(), 1025);

        title.signal_cpu(10, 0, 1);
        assert!(!title.held.holds_work(1));
        assert!(title.held.holds_work(2));
        assert_eq!(title.waiting.len(), 1);
        title.signal_cpu(20, 0, 1);
        assert!(title.waiting.is_empty());
    }

    #[test]
    fn a_wait_retains_its_fence_across_release_and_address_reuse() {
        let Ok(metal) = Metal::new() else { return };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        title.ensure_queue(1).unwrap();
        let old = title.fence(10, 0).unwrap();
        let weak = Rc::downgrade(&old);
        title.wait_queue(1, 10, 0, 5);
        title.forget_fence(10);
        drop(old);
        assert!(weak.upgrade().is_some(), "queued wait owns the backend fence");
        title.signal_cpu(10, 0, 5); // A new fence at the same title address.
        assert!(title.held.holds_work(1), "the replacement must not satisfy the old wait");
        title.host.backend.signal(&weak.upgrade().unwrap(), 5);
        title.pump();
        assert!(!title.held.holds_work(1));
        assert!(weak.upgrade().is_none(), "completed waits release their retained fences");
    }

    #[test]
    fn a_committed_producer_does_not_put_an_unsatisfied_wait_on_metal() {
        let Ok(metal) = Metal::new() else { return };
        let mut title = Title::new(metal, Arc::new(PoolTable::default()));
        let producer = title.queue(1).unwrap();
        title.ensure_queue(2).unwrap();
        let gate = title.fence(10, 0).unwrap();
        // Deliberately stall a backend producer. A title wait must stay CPU-side even after
        // its signal commits; the same rule applies if this producer subsequently fails.
        title.host.execute(&producer, &[], &[(&*gate, 1)], &[]).unwrap();
        title.signal_queue(1, 20, 0, 7);
        title.wait_queue(2, 20, 0, 7);
        assert!(title.held.holds_work(2));
        title.host.backend.signal(&gate, 1);
        let start = std::time::Instant::now();
        while title.held.holds_work(2) {
            assert!(start.elapsed() < std::time::Duration::from_secs(3));
            title.pump();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
/// [`Title::pump`]'s safety valve: the segments held past `limit` at the heads of blocked queues,
/// each with the waits it could not satisfy (they lose them and run).
fn valve_forced<S>(held: &mut Held<S>, limit: Duration) -> Vec<(usize, Vec<ValueWait>)> {
    held.force_older_than(limit)
}

#[cfg(test)]
mod valve_tests {
    use super::*;
    use crate::hold::Item;

    /// The valve takes a head whose waits nothing wrote once it is past the deadline, and leaves
    /// one inside it: the deadline sits between the longest wait that resolves (a world frame's
    /// barrier) and the title's 300 s watchdog scope.
    #[test]
    fn the_valve_takes_only_a_head_past_its_deadline() {
        let mut held: Held<&'static str> = Held::default();
        held.push(2, Item::Segment("frame 5287", vec![ValueWait { address: 0x100, bytes: 8, comparison: 7, reference: 5287, mask: u64::MAX }]));
        assert!(valve_forced(&mut held, Title::<victory_metal::metal::Metal>::VALVE).is_empty(), "a wait inside the deadline holds");
        let forced = valve_forced(&mut held, Duration::ZERO);
        assert_eq!(forced.len(), 1);
        assert_eq!(forced[0].0, 2);
        assert_eq!(forced[0].1.len(), 1, "the head loses the wait that never held");
        assert!(valve_forced(&mut held, Duration::ZERO).is_empty(), "a head already let go is not taken twice");
    }
}

