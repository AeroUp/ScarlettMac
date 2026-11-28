//! `VICTORY_TRACE_GPU_VERTICES=1` (#8): what feeds the vertices of draws whose vertex shader
//! fetches them itself (GPU-driven models: an empty input layout, vertices read from views), which
//! `VICTORY_DUMP_VERTICES` (input-assembler streams only) does not see.
//!
//! Four parts, all logged as numbers and shader hashes only (no bytes are logged or written):
//!
//! - Dropped writers. A dispatch dropped for want of a host compute pipeline (a refused, untranslated
//!   or uncompiled shader) never writes what it would have. What it could have written is noted
//!   with its shader hash and why it was dropped, by kind (`through`):
//!   - `table uav` (precise): its tables' buffer unordered-access views ([`NativeHost::table_uav_ranges`]);
//!   - `root uav` (estimated): the root views its program's metadata names as UAVs ([`WriteProfile`]),
//!     from the address to the end of the buffer resource holding it, else of the title mapping
//!     holding it (a root descriptor has no size);
//!   - `root view` (estimated): every root view, sized the same way, when the program's metadata
//!     is not known;
//!   - `heap uav (any)` (coarse): every buffer UAV of the bound resource heap, when the program
//!     indexes the heap itself and stores through descriptors; kept apart from the others;
//!   - `title memory (computed)` (no range): the program stores at addresses it computes (global,
//!     flat or scalar stores not through a root UAV's own SGPRs); named, not checked.
//!
//!   The set is per frame, bounded, and cleared at each present. Each draw's vertex streams, index
//!   buffer, and the vertex shader's own buffer reads (root views, table views) are checked against
//!   it: the first draw of each pipeline that reads such a range through each kind is warned about
//!   once, and a line per frame counts the dropped writers and the draws that read them.
//! - Early statistics. For the first draws (`VICTORY_TRACE_GPU_VERTICES_DRAWS=<draws>[,<seconds>]`,
//!   default 512 from the first such draw) whose vertex shader reads no attribute through the input
//!   assembler, each buffer view that shader reads is summarised once per pipeline and view when
//!   the draw is submitted (before the GPU has run it or anything before it in its submission):
//!   its range, how much of it the title maps (a table view over a gap or unmapped memory is the
//!   null view, zeros), the share of zero dwords and its dwords as float32 (range, not finite, over
//!   1e6), over its first 1 MiB. A view the host makes the null view is warned about for every such
//!   draw's pipeline, whatever the draw count.
//! - Late statistics (`VICTORY_TRACE_GPU_VERTICES_LATE=<views per present>[,<presents>]`, default
//!   32,60). The same views read again on the CPU (title memory is shared with the GPU) once Metal
//!   has reported on every command buffer committed before the present after the draw, and again
//!   `<presents>` presents later: "never written" against "written later". At most `<views per
//!   present>` are read at a present; 0 turns this off.
//! - Verdicts (`VICTORY_TRACE_GPU_VERTICES_VERDICT=<presents>`, default 300). Every `<presents>`
//!   presents (pipelines that changed), and once when the draw window is full and every late sample
//!   is taken (all of them): one line per vertex pipeline with its zero shares early, late and
//!   later, its draws that read a dropped writer's range by kind, and those writers named.
//!
//! Recording and submission are on different threads, so the writers are noted when a list is
//! recorded and the draws checked when it is submitted. shortcut: a writer noted after a present
//! for work submitted before it is cleared too early, and a writer dropped in an earlier frame (a
//! one-time build at load) is not seen; widen the window if a run needs it.
//!
//! Off unless set: one cached check per draw and dropped dispatch.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::host::NativeHost;
use super::list::{Binding, NativeHeap, Root};
use super::memory::PoolTable;
use victory_backend::resources::GraphicsBackend;

/// Ranges noted per frame at most; more are counted, not kept.
const MAX_WRITERS: usize = 4096;
/// Pipelines, and (pipeline, view) pairs, remembered at most for the once-only lines.
const MAX_SEEN: usize = 8192;
/// Bytes of a view the statistics read at most.
const SAMPLE: u64 = 1 << 20;
/// Null-view warnings logged at most.
const MAX_NULL_LINES: u32 = 1024;
/// Heap entries read at most for a heap-indexing writer, per heap and frame.
const MAX_HEAP_SCAN: usize = 1 << 17;
/// Heap-indexing and computed-address writers noted per frame at most.
const MAX_UNSIZED: usize = 64;
/// Late samples waiting at most.
const MAX_PENDING: usize = 4096;
/// Writers named per pipeline at most.
const MAX_NAMED: usize = 8;

pub(crate) const TABLE_UAV: &str = "table uav";
pub(crate) const ROOT_UAV: &str = "root uav";
pub(crate) const ROOT_VIEW: &str = "root view";
pub(crate) const HEAP_UAV: &str = "heap uav (any)";
pub(crate) const COMPUTED: &str = "title memory (computed)";
/// The kinds with a range, in the verdict's order.
const KINDS: [&str; 4] = [TABLE_UAV, ROOT_UAV, ROOT_VIEW, HEAP_UAV];

/// How far a hit through `through` can be trusted.
pub(crate) fn precision(through: &str) -> &'static str {
    match through {
        TABLE_UAV => "precise",
        ROOT_UAV | ROOT_VIEW => "estimated",
        _ => "coarse",
    }
}

fn kind_index(through: &str) -> usize {
    KINDS
        .iter()
        .position(|k| *k == through)
        .unwrap_or(KINDS.len() - 1)
}

/// Whether the trace is on (`VICTORY_TRACE_GPU_VERTICES`, any value but `0`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ON.get_or_init(|| crate::switch::on("VICTORY_TRACE_GPU_VERTICES"));
    #[cfg(test)]
    return env || FORCED.with(std::cell::Cell::get);
    #[cfg(not(test))]
    env
}

#[cfg(test)]
thread_local! {
    /// Tests switch the trace on for their own thread only.
    static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Switches the trace on for this thread, with nothing noted yet (a test's own state, also when
/// tests share a thread).
#[cfg(test)]
pub(crate) fn force_on_this_thread() {
    FORCED.with(|f| f.set(true));
    with_state(|s| *s = State::default());
}

/// The trace's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    /// Draws that get statistics, from how many seconds after the first one.
    pub draws: u32,
    pub after: u64,
    /// Late samples read at most per present (0: none), and the presents before the second one.
    pub late: usize,
    pub later: u64,
    /// Presents between verdicts (0: only when the window is done).
    pub verdict: u64,
}

/// `<a>[,<b>]` of variable `var`, each defaulted.
fn pair(var: &str, a: u64, b: u64) -> (u64, u64) {
    let v = std::env::var(var).unwrap_or_default();
    let mut parts = v.split(',');
    let mut next = |d: u64| {
        parts
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(d)
    };
    (next(a), next(b))
}

/// The settings from the environment (`VICTORY_TRACE_GPU_VERTICES_DRAWS`, `_LATE`, `_VERDICT`).
fn env_config() -> Config {
    static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
    *CONFIG.get_or_init(|| {
        let (draws, after) = pair("VICTORY_TRACE_GPU_VERTICES_DRAWS", 512, 0);
        let (late, later) = pair("VICTORY_TRACE_GPU_VERTICES_LATE", 32, 60);
        let (verdict, _) = pair("VICTORY_TRACE_GPU_VERTICES_VERDICT", 300, 0);
        Config {
            draws: u32::try_from(draws).unwrap_or(u32::MAX),
            after,
            late: late as usize,
            later,
            verdict,
        }
    })
}

/// What a compute program may write, from its container's metadata and code (no translation):
/// which kinds of `through` a dropped dispatch of it notes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WriteProfile {
    /// Root parameters the metadata names as UAVs (root descriptors).
    pub root_uavs: Vec<u8>,
    /// It takes descriptors from the resource heap itself (root type 5, as `native_programs`
    /// infers it), and the metadata names a UAV there.
    pub heap: bool,
    pub heap_uav_declared: bool,
    /// It stores or does atomics through descriptors (buffer, typed buffer, image).
    pub descriptor_stores: bool,
    /// It stores or does atomics at addresses it computes (global, flat or scalar memory not
    /// through a root UAV's own SGPRs).
    pub computed: bool,
    /// The code could be decoded (else every flag above that the code sets is assumed).
    pub decoded: bool,
}

impl WriteProfile {
    /// Whether it may write through any UAV of the bound heap.
    pub(crate) fn heap_uav(&self) -> bool {
        self.heap && (self.heap_uav_declared || self.descriptor_stores || !self.decoded)
    }

    /// Whether it may write title memory at computed addresses.
    pub(crate) fn computed_writes(&self) -> bool {
        self.computed || !self.decoded
    }
}

/// [`WriteProfile`] of a shader container's compute program; `None` if it has none.
pub(crate) fn profile_of(container: &[u8]) -> Option<WriteProfile> {
    use victory_shader::{Container, FourCC, HwStage};
    let c = Container::parse(container).ok()?;
    let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
    let programs = victory_shader::native_programs(part.data).ok()?;
    programs
        .iter()
        .find(|p| p.stage == HwStage::Cs)
        .map(profile_of_program)
}

/// A global access's scalar base field when it has none (`off`: the address is a VGPR pair), as
/// `victory_shader::gfx10` reads it.
const SADDR_OFF: u8 = 125;

/// [`WriteProfile`] of one program. Guess: a global store whose scalar base is a root
/// descriptor's own SGPR pair writes that root UAV; any other (a base copied elsewhere, a VGPR
/// address) is counted as computed, so this errs towards naming a writer.
pub(crate) fn profile_of_program(p: &victory_shader::NativeProgram) -> WriteProfile {
    use victory_shader::{ResourceType, RootType};
    let mut root_uavs: Vec<u8> = p
        .used
        .iter()
        .filter(|u| u.kind == ResourceType::Uav && u.root == RootType::Descriptor)
        .map(|u| u.slot)
        .collect();
    root_uavs.sort_unstable();
    root_uavs.dedup();
    let root_sgprs: Vec<u8> = p
        .layout
        .iter()
        .filter(|q| q.root == RootType::Descriptor)
        .map(|q| p.stage.user_sgpr(q.user_data))
        .collect();
    let mut out = WriteProfile {
        root_uavs,
        heap: p.layout.iter().any(|q| q.root == RootType::ResourceHeap),
        heap_uav_declared: p
            .used
            .iter()
            .any(|u| u.kind == ResourceType::Uav && u.root == RootType::ResourceHeap),
        ..WriteProfile::default()
    };
    let Ok(code) = victory_shader::gfx10::decode(&p.code) else {
        return out;
    };
    out.decoded = true;
    for inst in &code {
        let Some(name) = inst.name() else { continue };
        let starts = |prefixes: &[&str]| prefixes.iter().any(|x| name.starts_with(x));
        if starts(&[
            "buffer_store",
            "buffer_atomic",
            "tbuffer_store",
            "image_store",
            "image_atomic",
            "s_buffer_store",
            "s_buffer_atomic",
        ]) {
            out.descriptor_stores = true;
        } else if starts(&["global_store", "global_atomic"]) {
            let saddr = inst
                .words
                .get(1)
                .map_or(SADDR_OFF, |w| (w >> 16 & 0x7F) as u8);
            if saddr == SADDR_OFF || !root_sgprs.contains(&saddr) {
                out.computed = true;
            }
        } else if starts(&["flat_store", "flat_atomic", "s_store", "s_atomic"]) {
            out.computed = true;
        }
    }
    out
}

/// Which binding of a draw reads a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Via {
    /// An input-assembler vertex stream (its slot).
    Stream(usize),
    Index,
    /// A root view the vertex shader reads (its root parameter).
    Root(u8),
    /// A buffer view in a table the vertex shader reads (root parameter, entry from the table's
    /// start).
    Table(u8, u32),
}

impl std::fmt::Display for Via {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Via::Stream(s) => write!(f, "vertex stream {s}"),
            Via::Index => write!(f, "index buffer"),
            Via::Root(s) => write!(f, "vertex shader root view {s}"),
            Via::Table(s, e) => write!(f, "vertex shader table {s} entry {e}"),
        }
    }
}

/// A buffer view the vertex shader reads, as the host decodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VsView {
    pub via: Via,
    pub addr: u64,
    /// Its size in bytes (0 for a root view, which has none: [`extent`] sizes it).
    pub size: u64,
    /// The entry is a buffer view the host makes no buffer of (the null view).
    pub null: bool,
}

/// A range a dropped dispatch may have written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Writer {
    pub lo: u64,
    pub hi: u64,
    /// The dropped dispatch's shaders (`compute <hash>`).
    pub shader: Arc<str>,
    /// One of [`KINDS`].
    pub through: &'static str,
    /// Why it was dropped (`refused`, `not translated yet`, `no host pipeline yet`).
    pub why: &'static str,
}

/// `[lo, hi)` ranges, sorted and merged, shared by the writers of one heap.
type Ranges = Arc<Vec<(u64, u64)>>;

/// A heap-indexing writer: every buffer UAV range of the bound heap.
struct HeapWriter {
    shader: Arc<str>,
    why: &'static str,
    ranges: Ranges,
}

/// A range a draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub via: Via,
    pub lo: u64,
    pub hi: u64,
}

/// The first draw of a pipeline that reads, through one kind, what a dropped dispatch would have
/// written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Hit {
    pub pipeline: Arc<str>,
    pub read: Read,
    pub writer: Writer,
    /// Bytes the read and the written range share.
    pub bytes: u64,
}

/// A dropped writer a draw reads from: its shaders, why it was dropped, and the kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Touch {
    pub shader: Arc<str>,
    pub why: &'static str,
    pub through: &'static str,
}

impl std::fmt::Display for Touch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}; {})", self.shader, self.why, self.through)
    }
}

/// What checking one draw found: the warnings due, and every writer it reads from.
#[derive(Debug, Default)]
pub(crate) struct Drawn {
    pub hits: Vec<Hit>,
    pub touched: Vec<Touch>,
}

/// One frame's counts, for the line at its present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameCounts {
    /// Dispatches dropped, and the distinct sized ranges they noted (and how many past the bound).
    pub dispatches: u64,
    pub ranges: usize,
    pub over: u64,
    /// Heap-indexing writers and their merged ranges; computed-address writers.
    pub heap_writers: usize,
    pub heap_ranges: usize,
    pub computed_writers: usize,
    /// Draws checked; those reading a precise or estimated range, those reading only a heap
    /// writer's, and the bindings that read a precise or estimated one.
    pub draws: u64,
    pub draws_hit: u64,
    pub draws_hit_heap_only: u64,
    pub reads_hit: u64,
}

/// The dropped writers of one frame and the draws that read them.
#[derive(Default)]
pub(crate) struct Overlaps {
    writers: Vec<Writer>,
    seen: HashSet<(u64, u64, Arc<str>, &'static str)>,
    heap: Vec<HeapWriter>,
    computed: Vec<(Arc<str>, &'static str)>,
    counts: FrameCounts,
    /// Pipelines already warned about, by kind (bit [`kind_index`]), for the run.
    warned: HashMap<Arc<str>, u8>,
}

/// The range of sorted, merged `ranges` sharing the most bytes with `[lo, hi)` (of the first 64
/// that touch it), and how many.
fn heap_overlap(ranges: &[(u64, u64)], lo: u64, hi: u64) -> Option<((u64, u64), u64)> {
    let i = ranges.partition_point(|r| r.1 <= lo);
    ranges[i..]
        .iter()
        .take_while(|r| r.0 < hi)
        .take(64)
        .map(|&r| (r, hi.min(r.1).saturating_sub(lo.max(r.0))))
        .filter(|&(_, n)| n > 0)
        .max_by_key(|&(_, n)| n)
}

/// `ranges` (`[lo, hi)`) sorted, with those that overlap or touch joined.
pub(crate) fn merge(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|r| r.1 > r.0);
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.0 <= last.1 => last.1 = last.1.max(r.1),
            _ => out.push(r),
        }
    }
    out
}

fn add_touch(out: &mut Vec<Touch>, shader: &Arc<str>, why: &'static str, through: &'static str) {
    if !out
        .iter()
        .any(|t| t.through == through && t.shader == *shader)
    {
        out.push(Touch {
            shader: shader.clone(),
            why,
            through,
        });
    }
}

impl Overlaps {
    /// A dispatch of `shader` was dropped (`why`); it may have written `ranges` (`[lo, hi)`, kind).
    pub(crate) fn dropped(
        &mut self,
        shader: &Arc<str>,
        why: &'static str,
        ranges: impl IntoIterator<Item = (u64, u64, &'static str)>,
    ) {
        self.counts.dispatches += 1;
        for (lo, hi, through) in ranges {
            if hi <= lo || !self.seen.insert((lo, hi, shader.clone(), through)) {
                continue;
            }
            if self.writers.len() >= MAX_WRITERS {
                self.counts.over += 1;
                continue;
            }
            self.writers.push(Writer {
                lo,
                hi,
                shader: shader.clone(),
                through,
                why,
            });
        }
    }

    /// The same dispatch indexes the heap itself: it may have written any of `ranges` (merged).
    pub(crate) fn dropped_heap(&mut self, shader: &Arc<str>, why: &'static str, ranges: Ranges) {
        if ranges.is_empty()
            || self
                .heap
                .iter()
                .any(|h| h.shader == *shader && Arc::ptr_eq(&h.ranges, &ranges))
        {
            return;
        }
        if self.heap.len() >= MAX_UNSIZED {
            self.counts.over += 1;
            return;
        }
        self.heap.push(HeapWriter {
            shader: shader.clone(),
            why,
            ranges,
        });
    }

    /// The same dispatch stores at addresses it computes: named, with no range.
    pub(crate) fn dropped_computed(&mut self, shader: &Arc<str>, why: &'static str) {
        if self.computed.len() < MAX_UNSIZED && !self.computed.iter().any(|(s, _)| s == shader) {
            self.computed.push((shader.clone(), why));
        }
    }

    pub(crate) fn has_writers(&self) -> bool {
        !self.writers.is_empty() || !self.heap.is_empty()
    }

    /// The noted sized range sharing the most bytes with `[lo, hi)`, and how many.
    #[cfg(test)]
    pub(crate) fn overlap(&self, lo: u64, hi: u64) -> Option<(&Writer, u64)> {
        self.writers
            .iter()
            .filter_map(|w| {
                let n = hi.min(w.hi).saturating_sub(lo.max(w.lo));
                (n > 0).then_some((w, n))
            })
            .max_by_key(|&(_, n)| n)
    }

    /// Checks a draw of `pipeline` that reads `reads`: counted; for each kind, the first read
    /// that hits it (with its largest overlap) is returned the first time this pipeline has one.
    pub(crate) fn draw(&mut self, pipeline: &str, reads: &[Read]) -> Drawn {
        self.counts.draws += 1;
        let mut drawn = Drawn::default();
        if !self.has_writers() {
            return drawn;
        }
        let mut first: [Option<(Read, Writer, u64)>; KINDS.len()] = Default::default();
        let (mut sized, mut heap) = (0, 0);
        for r in reads {
            let mut hit = false;
            for w in &self.writers {
                let n = r.hi.min(w.hi).saturating_sub(r.lo.max(w.lo));
                if n == 0 {
                    continue;
                }
                hit = true;
                add_touch(&mut drawn.touched, &w.shader, w.why, w.through);
                let k = kind_index(w.through);
                match &first[k] {
                    Some((fr, _, fnn)) if fr.via != r.via || *fnn >= n => {}
                    _ => first[k] = Some((*r, w.clone(), n)),
                }
            }
            if hit {
                sized += 1;
            }
            for h in &self.heap {
                let Some(((lo, hi), n)) = heap_overlap(&h.ranges, r.lo, r.hi) else {
                    continue;
                };
                heap += 1;
                add_touch(&mut drawn.touched, &h.shader, h.why, HEAP_UAV);
                let k = kind_index(HEAP_UAV);
                if first[k].is_none() {
                    let w = Writer {
                        lo,
                        hi,
                        shader: h.shader.clone(),
                        through: HEAP_UAV,
                        why: h.why,
                    };
                    first[k] = Some((*r, w, n));
                }
            }
        }
        if sized > 0 {
            self.counts.draws_hit += 1;
            self.counts.reads_hit += sized;
        } else if heap > 0 {
            self.counts.draws_hit_heap_only += 1;
        }
        let bits = self.warned.get(pipeline).copied();
        if bits.is_none() && self.warned.len() >= MAX_SEEN {
            return drawn;
        }
        let mut b = bits.unwrap_or(0);
        let mut name: Option<Arc<str>> = None;
        for (k, f) in first.into_iter().enumerate() {
            let Some((read, writer, bytes)) = f else {
                continue;
            };
            if b & 1 << k != 0 {
                continue;
            }
            b |= 1 << k;
            let pipeline = name.get_or_insert_with(|| pipeline.into()).clone();
            drawn.hits.push(Hit {
                pipeline,
                read,
                writer,
                bytes,
            });
        }
        if let Some(n) = name {
            self.warned.insert(n, b);
        }
        drawn
    }

    /// The frame ended: its counts, and its writers forgotten.
    pub(crate) fn present(&mut self) -> FrameCounts {
        let mut counts = std::mem::take(&mut self.counts);
        counts.ranges = self.writers.len();
        counts.heap_writers = self.heap.len();
        counts.heap_ranges = self.heap.iter().map(|h| h.ranges.len()).sum();
        counts.computed_writers = self.computed.len();
        self.writers.clear();
        self.seen.clear();
        self.heap.clear();
        self.computed.clear();
        counts
    }
}

/// `[a, end)` of the buffer in `spans` (by start, `[start, end)`) holding `a`, the one starting
/// nearest below it; `None` when none of the 64 starting nearest below it holds it.
pub(crate) fn extent(spans: &[(u64, u64)], a: u64) -> Option<(u64, u64)> {
    let i = spans.partition_point(|&(s, _)| s <= a);
    spans[..i]
        .iter()
        .rev()
        .take(64)
        .find(|&&(_, e)| a < e)
        .map(|&(_, e)| (a, e))
}

/// What the statistics read of one view.
#[derive(Default)]
pub(crate) struct Summary {
    /// Bytes read (mapped and readable), dwords among them and how many were 0.
    pub read: u64,
    pub dwords: u64,
    pub zero: u64,
    /// The dwords as float32.
    pub floats: super::geometry_trace::Stats,
}

impl Summary {
    /// Adds `bytes`, read from `offset` bytes into the view: only whole dwords at the view's
    /// dword boundaries count.
    pub(crate) fn add(&mut self, bytes: &[u8], offset: u64) {
        self.read += bytes.len() as u64;
        let skip = ((4 - offset % 4) % 4) as usize;
        for d in bytes.get(skip..).unwrap_or_default().as_chunks::<4>().0 {
            self.dwords += 1;
            if *d == [0; 4] {
                self.zero += 1;
            }
            self.floats
                .add(victory_backend::resources::VertexFormat::Float, d);
        }
    }

    fn zero_share(&self) -> f64 {
        share(self.dwords, self.zero)
    }
}

fn share(dwords: u64, zero: u64) -> f64 {
    if dwords == 0 {
        0.0
    } else {
        zero as f64 / dwords as f64
    }
}

/// Reads the first [`SAMPLE`] bytes of `[addr, addr + len)` where the title maps them.
fn summarise(memory: &PoolTable, addr: u64, len: u64) -> Summary {
    let mut s = Summary::default();
    let n = len.min(SAMPLE) as usize;
    if n == 0 || addr == 0 {
        return s;
    }
    let mut buf = vec![0u8; n];
    let pooled = memory.read_mapped_with(addr as usize, &mut buf, |at, out| {
        let ok = super::safe_read(at as *const u8, out);
        if ok {
            s.add(out, at as u64 - addr);
        }
        ok
    });
    // Outside every pool (the host's own memory): read as it is.
    if pooled.is_none() && super::safe_read(addr as *const u8, &mut buf) {
        s.add(&buf, 0);
    }
    s
}

/// Dwords and zero dwords over some views.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    pub views: u32,
    pub dwords: u64,
    pub zero: u64,
}

impl Tally {
    fn add(&mut self, s: &Summary) {
        self.views += 1;
        self.dwords += s.dwords;
        self.zero += s.zero;
    }

    /// The zero share as the verdict shows it (`-`: nothing read).
    fn text(&self) -> String {
        if self.dwords == 0 {
            "-".into()
        } else {
            format!("{:.4}", share(self.dwords, self.zero))
        }
    }
}

/// What the verdict of one pipeline is made from.
#[derive(Default)]
struct Pipe {
    early: Tally,
    late: Tally,
    later: Tally,
    /// Draws that read a dropped writer's range, by kind ([`KINDS`]).
    kinds: [u64; KINDS.len()],
    writers: Vec<Touch>,
    /// Computed-address writers dropped in frames it drew in (`frame`: the last merged).
    unbounded: Vec<(Arc<str>, &'static str)>,
    frame: u64,
    changed: bool,
}

/// One verdict line's fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Verdict {
    pub pipeline: String,
    /// Views summarised early, late, later.
    pub views: String,
    pub zero_early: String,
    pub zero_late: String,
    pub zero_later: String,
    pub overlaps: String,
    pub writers: String,
    pub unbounded: String,
}

impl Pipe {
    fn verdict(&self, name: &str) -> Verdict {
        let overlaps = KINDS
            .iter()
            .zip(self.kinds)
            .map(|(k, n)| format!("{k}:{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        let list = |v: Vec<String>| {
            if v.is_empty() {
                "-".into()
            } else {
                v.join(", ")
            }
        };
        Verdict {
            pipeline: name.into(),
            views: format!(
                "{}/{}/{}",
                self.early.views, self.late.views, self.later.views
            ),
            zero_early: self.early.text(),
            zero_late: self.late.text(),
            zero_later: self.later.text(),
            overlaps,
            writers: list(self.writers.iter().map(Touch::to_string).collect()),
            unbounded: list(
                self.unbounded
                    .iter()
                    .map(|(s, why)| format!("{s} ({why}; {COMPUTED})"))
                    .collect(),
            ),
        }
    }
}

/// The verdict entry of `name`, made if there is room.
fn pipe<'a>(pipes: &'a mut HashMap<Arc<str>, Pipe>, name: &str) -> Option<&'a mut Pipe> {
    if !pipes.contains_key(name) {
        if pipes.len() >= MAX_SEEN {
            return None;
        }
        pipes.insert(name.into(), Pipe::default());
    }
    pipes.get_mut(name)
}

/// A view to read again.
struct Pending {
    pipeline: Arc<str>,
    via: Via,
    addr: u64,
    size: u64,
    early: f64,
    /// Late: the submission watermark of the present after its draw (set at that present).
    watermark: Option<u64>,
    /// Later: the present it is due at.
    due: Option<u64>,
}

#[derive(Default)]
struct State {
    overlaps: Overlaps,
    /// The buffer resources' ranges by start, made at the first need in a frame.
    spans: Option<Vec<(u64, u64)>>,
    /// Buffer UAV ranges of heaps (by start) read this frame.
    heaps: Vec<(u64, Ranges)>,
    /// Draws whose vertex shader reads no attribute, so far, and when the first was.
    gpu_draws: u32,
    start: Option<std::time::Instant>,
    /// (pipeline name pointer, view, address, size) already summarised, and already checked
    /// for the null view.
    summarised: HashSet<(usize, Via, u64, u64)>,
    checked: HashSet<(usize, Via, u64, u64)>,
    /// Pipelines logged as reading no buffer view, and null-view warnings so far.
    no_views: HashSet<usize>,
    null_lines: u32,
    /// Settings set by a test, else the environment's.
    config: Option<Config>,
    /// Title memory, for the late samples.
    memory: Option<Arc<PoolTable>>,
    presents: u64,
    pending: VecDeque<Pending>,
    pending_over: u64,
    pipes: HashMap<Arc<str>, Pipe>,
    verdict_done: bool,
    /// What was logged, for tests.
    #[cfg(test)]
    logged: Logged,
}

/// What was logged, for tests: hits, null views `(view, address, size, how)`, early summaries
/// `(view, bytes read, dwords, zero dwords)`, late ones `(view, later, dwords, zero dwords)` and
/// verdicts.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct Logged {
    pub hits: Vec<Hit>,
    pub null: Vec<(Via, u64, u64, &'static str)>,
    pub summaries: Vec<(Via, u64, u64, u64)>,
    pub late: Vec<(Via, bool, u64, u64)>,
    pub verdicts: Vec<Verdict>,
}

impl State {
    fn config(&self) -> Config {
        self.config.unwrap_or_else(env_config)
    }

    /// From `a` to the end of the buffer resource holding it, else of the title mapping holding
    /// it, else one byte.
    fn extent<B: GraphicsBackend>(&mut self, host: &NativeHost<B>, a: u64) -> (u64, u64) {
        let spans = self.spans.get_or_insert_with(|| host.buffer_spans());
        extent(spans, a)
            .or_else(|| host.memory().mapping_at(a).map(|m| (a, m.at + m.len)))
            .unwrap_or((a, a.saturating_add(1)))
    }
}

/// The trace's state: one for the process (lists are recorded and submitted on different
/// threads), one per thread under test (tests run in parallel).
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    #[cfg(not(test))]
    {
        static STATE: std::sync::Mutex<Option<State>> = std::sync::Mutex::new(None);
        f(STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_default())
    }
    #[cfg(test)]
    {
        thread_local! {
            static STATE: std::cell::RefCell<State> = std::cell::RefCell::default();
        }
        STATE.with_borrow_mut(f)
    }
}

/// A dispatch of `shader` dropped for want of a host compute pipeline (`why`), with root
/// parameters `root` and resource heap `heap`: notes what it may have written. `profile`: what
/// its program may write ([`profile_of`]); `None` when not known, and every root view is noted.
pub(crate) fn dropped_dispatch<B: GraphicsBackend>(
    host: &NativeHost<B>,
    root: &[Option<Root<B>>],
    heap: Option<&NativeHeap<B>>,
    shader: &str,
    why: &'static str,
    profile: Option<&WriteProfile>,
) {
    let tables = host.table_uav_ranges(root);
    let shader: Arc<str> = shader.into();
    with_state(|s| {
        let mut ranges: Vec<(u64, u64, &'static str)> = tables
            .iter()
            .map(|&(a, n)| (a, a.saturating_add(n), TABLE_UAV))
            .collect();
        for (slot, r) in root.iter().enumerate() {
            let Some(Root::View(a)) = r else { continue };
            let through = match profile {
                Some(p) if p.root_uavs.iter().any(|&u| usize::from(u) == slot) => ROOT_UAV,
                Some(_) => continue,
                None => ROOT_VIEW,
            };
            if *a != 0 {
                let (lo, hi) = s.extent(host, *a);
                ranges.push((lo, hi, through));
            }
        }
        s.overlaps.dropped(&shader, why, ranges);
        let Some(p) = profile else { return };
        if p.computed_writes() {
            s.overlaps.dropped_computed(&shader, why);
        }
        if let Some(h) = heap
            && p.heap_uav()
        {
            let ranges = match s.heaps.iter().find(|(at, _)| *at == h.start) {
                Some((_, r)) => r.clone(),
                None => {
                    let r = Arc::new(merge(host.heap_uav_ranges(h, MAX_HEAP_SCAN)));
                    s.heaps.push((h.start, r.clone()));
                    r
                }
            };
            s.overlaps.dropped_heap(&shader, why, ranges);
        }
    });
}

/// The frame ended: its count line, its writers forgotten, the late samples that are due read,
/// and the verdicts due logged. `watermark`: the backend's submission count, taken under the
/// title's lock (every draw submitted so far is in a command buffer below it); `landed(w)`:
/// whether every command buffer below `w` has completed.
/// shortcut: a draw submitted on another thread between the watermark and this call takes it
/// too, and may be read again a little early.
pub(crate) fn present(watermark: u64, landed: impl Fn(u64) -> bool) {
    with_state(|s| present_in(s, watermark, landed));
}

fn present_in(s: &mut State, watermark: u64, landed: impl Fn(u64) -> bool) -> FrameCounts {
    s.spans = None;
    s.heaps.clear();
    let counts = s.overlaps.present();
    if counts.dispatches > 0 || counts.draws_hit > 0 || counts.draws_hit_heap_only > 0 {
        tracing::info!(
            frame = crate::frame::current(),
            dropped_dispatches = counts.dispatches,
            ranges = counts.ranges,
            ranges_over_bound = counts.over,
            heap_writers = counts.heap_writers,
            heap_ranges = counts.heap_ranges,
            computed_writers = counts.computed_writers,
            draws = counts.draws,
            draws_reading_them = counts.draws_hit,
            draws_reading_only_heap_uavs = counts.draws_hit_heap_only,
            reads = counts.reads_hit,
            "gpu vertices: what dropped dispatches would have written, and the draws that read it"
        );
    }
    s.presents += 1;
    let cfg = s.config();
    for p in s
        .pending
        .iter_mut()
        .filter(|p| p.due.is_none() && p.watermark.is_none())
    {
        p.watermark = Some(watermark);
    }
    let mut landed_at: Option<(u64, bool)> = None;
    let (mut taken, mut i) = (0, 0);
    while i < s.pending.len() && taken < cfg.late {
        let ready = match (s.pending[i].due, s.pending[i].watermark) {
            (Some(due), _) => s.presents >= due,
            (None, Some(w)) => match landed_at {
                Some((at, ok)) if at == w => ok,
                _ => {
                    let ok = landed(w);
                    landed_at = Some((w, ok));
                    ok
                }
            },
            (None, None) => false,
        };
        if !ready {
            i += 1;
            continue;
        }
        let Some(p) = s.pending.remove(i) else { break };
        taken += 1;
        let Some(memory) = &s.memory else { continue };
        let sum = summarise(memory, p.addr, p.size);
        let later = p.due.is_some();
        #[cfg(test)]
        s.logged.late.push((p.via, later, sum.dwords, sum.zero));
        let when = if later {
            format!("{} presents later", cfg.later)
        } else {
            "after the GPU ran its draw".into()
        };
        tracing::info!(
            pipeline = &*p.pipeline,
            binding = %p.via,
            view = format!("{:#x}+{:#x}", p.addr, p.size),
            when,
            read = sum.read,
            zero_dwords_early = format!("{:.4}", p.early),
            zero_dwords = format!("{:.4}", sum.zero_share()),
            floats = %sum.floats,
            "gpu vertices: a vertex shader's buffer view, read again"
        );
        if let Some(e) = pipe(&mut s.pipes, &p.pipeline) {
            if later {
                e.later.add(&sum)
            } else {
                e.late.add(&sum)
            }
            e.changed = true;
        }
        if !later && cfg.later > 0 {
            s.pending.push_back(Pending {
                due: Some(s.presents + cfg.later),
                ..p
            });
        }
    }
    let done = s.start.is_some() && s.gpu_draws >= cfg.draws && s.pending.is_empty();
    if done && !s.verdict_done {
        s.verdict_done = true;
        verdicts(s, true);
    } else if cfg.verdict > 0 && s.presents.is_multiple_of(cfg.verdict) {
        verdicts(s, false);
    }
    counts
}

/// One line per pipeline (`all`, else those changed since the last verdict).
fn verdicts(s: &mut State, all: bool) {
    let mut names: Vec<&Arc<str>> = s
        .pipes
        .iter()
        .filter(|(_, p)| all || p.changed)
        .map(|(n, _)| n)
        .collect();
    names.sort();
    let names: Vec<Arc<str>> = names.into_iter().cloned().collect();
    for name in names {
        let Some(p) = s.pipes.get_mut(&name) else {
            continue;
        };
        p.changed = false;
        let v = p.verdict(&name);
        tracing::info!(
            pipeline = v.pipeline,
            views = v.views,
            zero_early = v.zero_early,
            zero_late = v.zero_late,
            zero_later = v.zero_later,
            overlaps = v.overlaps,
            writers = v.writers,
            unbounded = v.unbounded,
            window_done = all,
            pending = s.pending.len(),
            pending_over_bound = s.pending_over,
            "gpu vertices: verdict for a vertex pipeline"
        );
        #[cfg(test)]
        s.logged.verdicts.push(v);
    }
}

/// A draw being submitted: its vertex streams (`(slot, address, size)`, `u32::MAX` for no size),
/// its index buffer when it is indexed, and its vertex shader's binding and root parameters.
/// `mesh`: a mesh dispatch, which reads neither.
pub(crate) fn draw<B: GraphicsBackend>(
    host: &NativeHost<B>,
    root: &[Option<Root<B>>],
    binding: Option<&Binding>,
    vertex: &[(usize, u64, u32)],
    index: Option<(u64, u32)>,
    mesh: bool,
) {
    let views = binding.map(|b| host.vs_views(root, b)).unwrap_or_default();
    let name = binding.map_or("(no translated binding)", |b| &*b.name);
    let key = binding.map_or(0, |b| b.name.as_ptr() as usize);
    let size = |n: u32| match n {
        0 | u32::MAX => 1,
        n => u64::from(n),
    };
    let gpu_fetched = binding.is_some_and(|b| b.vertex_attributes == 0) && !mesh;
    with_state(|s| {
        if s.memory.is_none() {
            s.memory = Some(host.memory());
        }
        // Root views have no size: from the address to the end of the buffer that holds it.
        let views: Vec<VsView> = views
            .into_iter()
            .map(|mut v| {
                if matches!(v.via, Via::Root(_)) && !v.null {
                    let (lo, hi) = s.extent(host, v.addr);
                    v.size = hi - lo;
                }
                v
            })
            .collect();
        let drawn = if s.overlaps.has_writers() {
            let mut reads: Vec<Read> = Vec::new();
            if !mesh {
                reads.extend(vertex.iter().map(|&(slot, a, n)| Read {
                    via: Via::Stream(slot),
                    lo: a,
                    hi: a.saturating_add(size(n)),
                }));
                reads.extend(index.map(|(a, n)| Read {
                    via: Via::Index,
                    lo: a,
                    hi: a.saturating_add(size(n)),
                }));
            }
            reads.extend(views.iter().filter(|v| !v.null).map(|v| Read {
                via: v.via,
                lo: v.addr,
                hi: v.addr.saturating_add(v.size.max(1)),
            }));
            s.overlaps.draw(name, &reads)
        } else {
            s.overlaps.draw(name, &[])
        };
        for h in &drawn.hits {
            #[cfg(test)]
            s.logged.hits.push(h.clone());
            tracing::warn!(
                pipeline = &*h.pipeline,
                binding = %h.read.via,
                read = format!("{:#x}+{:#x}", h.read.lo, h.read.hi - h.read.lo),
                dropped = &*h.writer.shader,
                why = h.writer.why,
                through = h.writer.through,
                precision = precision(h.writer.through),
                written = format!("{:#x}+{:#x}", h.writer.lo, h.writer.hi - h.writer.lo),
                overlap = h.bytes,
                "gpu vertices: a draw reads what a dropped dispatch would have written this frame (first for this pipeline and kind)"
            );
        }
        if !drawn.touched.is_empty()
            && let Some(p) = pipe(&mut s.pipes, name)
        {
            let mut kinds = [false; KINDS.len()];
            for t in &drawn.touched {
                kinds[kind_index(t.through)] = true;
                if p.writers.len() < MAX_NAMED && !p.writers.contains(t) {
                    p.writers.push(t.clone());
                }
            }
            for (n, k) in p.kinds.iter_mut().zip(kinds) {
                *n += u64::from(k);
            }
            p.changed = true;
        }
        if !gpu_fetched {
            return;
        }
        if !s.overlaps.computed.is_empty()
            && let Some(p) = pipe(&mut s.pipes, name)
            && p.frame != s.presents + 1
        {
            p.frame = s.presents + 1;
            for (c, why) in &s.overlaps.computed {
                if p.unbounded.len() < MAX_NAMED && !p.unbounded.iter().any(|(u, _)| u == c) {
                    p.unbounded.push((c.clone(), why));
                    p.changed = true;
                }
            }
        }
        let cfg = s.config();
        let start = *s.start.get_or_insert_with(std::time::Instant::now);
        let counted = start.elapsed().as_secs() >= cfg.after && s.gpu_draws < cfg.draws;
        if counted {
            s.gpu_draws += 1;
        }
        let heap = binding.is_some_and(|b| b.heap);
        for v in &views {
            let k = (key, v.via, v.addr, v.size);
            let check = s.checked.len() < MAX_SEEN && s.checked.insert(k);
            let summarise_it = counted && s.summarised.len() < MAX_SEEN && s.summarised.insert(k);
            if !check && !summarise_it {
                continue;
            }
            let (mapped, how) = if v.null {
                (0, "no buffer: the null view")
            } else {
                host.coverage(v.addr, v.size, matches!(v.via, Via::Root(_)))
            };
            let null = v.null || how.ends_with("the null view");
            if check && null && s.null_lines < MAX_NULL_LINES {
                s.null_lines += 1;
                #[cfg(test)]
                s.logged.null.push((v.via, v.addr, v.size, how));
                tracing::warn!(
                    pipeline = name,
                    binding = %v.via,
                    view = format!("{:#x}+{:#x}", v.addr, v.size),
                    mapped,
                    how,
                    "gpu vertices: a vertex shader reads a buffer view the host makes the null view (zeros)"
                );
            }
            if summarise_it {
                let sum = if v.null {
                    Summary::default()
                } else {
                    summarise(&host.memory(), v.addr, v.size)
                };
                #[cfg(test)]
                s.logged
                    .summaries
                    .push((v.via, sum.read, sum.dwords, sum.zero));
                tracing::info!(
                    draw = s.gpu_draws,
                    pipeline = name,
                    heap_indexed = heap,
                    binding = %v.via,
                    view = format!("{:#x}+{:#x}", v.addr, v.size),
                    mapped,
                    how,
                    read = sum.read,
                    zero_dwords = format!("{:.4}", sum.zero_share()),
                    floats = %sum.floats,
                    "gpu vertices: a vertex shader's buffer view"
                );
                let early = sum.zero_share();
                if let Some(p) = pipe(&mut s.pipes, name) {
                    p.early.add(&sum);
                    p.changed = true;
                }
                if !v.null && cfg.late > 0 {
                    if s.pending.len() < MAX_PENDING {
                        s.pending.push_back(Pending {
                            pipeline: name.into(),
                            via: v.via,
                            addr: v.addr,
                            size: v.size,
                            early,
                            watermark: None,
                            due: None,
                        });
                    } else {
                        s.pending_over += 1;
                    }
                }
            }
        }
        if counted && views.is_empty() && s.no_views.len() < MAX_SEEN && s.no_views.insert(key) {
            tracing::info!(
                draw = s.gpu_draws,
                pipeline = name,
                heap_indexed = heap,
                "gpu vertices: a draw fetching its own vertices reads no buffer view the host sees"
            );
        }
    })
}

#[cfg(test)]
pub(crate) fn logged_for_test() -> Logged {
    with_state(|s| s.logged.clone())
}

/// A present in a test, with the late samples' watermark and completion given.
#[cfg(test)]
pub(crate) fn present_with_for_test(watermark: u64, landed: impl Fn(u64) -> bool) -> FrameCounts {
    with_state(|s| present_in(s, watermark, landed))
}

/// A present in a test whose late samples are all due.
#[cfg(test)]
pub(crate) fn present_for_test() -> FrameCounts {
    present_with_for_test(0, |_| true)
}

/// Settings for this thread's trace.
#[cfg(test)]
pub(crate) fn configure_for_test(config: Config) {
    with_state(|s| s.config = Some(config));
}
#[cfg(test)]
mod tests {
    use super::*;

    fn shader(s: &str) -> Arc<str> {
        s.into()
    }

    fn read(via: Via, lo: u64, hi: u64) -> Read {
        Read { via, lo, hi }
    }

    const WHY: &str = "refused";

    #[test]
    fn ranges_overlap_only_where_they_share_bytes() {
        let mut o = Overlaps::default();
        o.dropped(
            &shader("compute aa"),
            WHY,
            [(0x1000, 0x2000, TABLE_UAV), (0x8000, 0x8001, ROOT_VIEW)],
        );
        assert_eq!(o.overlap(0x0, 0x1000), None, "ends where the write starts");
        assert_eq!(
            o.overlap(0x2000, 0x3000),
            None,
            "starts where the write ends"
        );
        assert_eq!(
            o.overlap(0x1ff0, 0x3000).map(|(w, n)| (w.lo, n)),
            Some((0x1000, 0x10))
        );
        assert_eq!(
            o.overlap(0x0, 0x10000).map(|(w, n)| (w.lo, n)),
            Some((0x1000, 0x1000)),
            "the largest overlap"
        );
        assert_eq!(
            o.overlap(0x7000, 0x9000).map(|(w, n)| (w.through, n)),
            Some((ROOT_VIEW, 1))
        );
        // Empty and repeated ranges are not kept.
        o.dropped(
            &shader("compute aa"),
            WHY,
            [(0x5000, 0x5000, TABLE_UAV), (0x1000, 0x2000, TABLE_UAV)],
        );
        assert_eq!(o.writers.len(), 2);
        assert_eq!(o.counts.dispatches, 2);
    }

    #[test]
    fn the_writers_are_bounded() {
        let mut o = Overlaps::default();
        let s = shader("compute bb");
        o.dropped(
            &s,
            WHY,
            (0..MAX_WRITERS as u64 + 5).map(|k| (k * 16, k * 16 + 8, TABLE_UAV)),
        );
        assert_eq!(o.writers.len(), MAX_WRITERS);
        assert_eq!(o.present().over, 5);
    }

    #[test]
    fn a_present_clears_the_frames_writers() {
        let mut o = Overlaps::default();
        o.dropped(&shader("compute cc"), WHY, [(0x1000, 0x2000, TABLE_UAV)]);
        o.dropped_computed(&shader("compute c2"), WHY);
        assert_eq!(
            o.draw(
                "vertex v1 pixel p1",
                &[read(Via::Stream(0), 0x1800, 0x1900)]
            )
            .hits
            .len(),
            1
        );
        let counts = o.present();
        assert_eq!(
            counts,
            FrameCounts {
                dispatches: 1,
                ranges: 1,
                computed_writers: 1,
                draws: 1,
                draws_hit: 1,
                reads_hit: 1,
                ..FrameCounts::default()
            }
        );
        assert!(!o.has_writers() && o.computed.is_empty());
        // The next frame: nothing dropped, nothing read, and the same range may be noted again.
        assert!(
            o.draw(
                "vertex v2 pixel p2",
                &[read(Via::Stream(0), 0x1800, 0x1900)]
            )
            .hits
            .is_empty()
        );
        assert_eq!(
            o.present(),
            FrameCounts {
                draws: 1,
                ..FrameCounts::default()
            }
        );
        o.dropped(&shader("compute cc"), WHY, [(0x1000, 0x2000, TABLE_UAV)]);
        assert!(o.has_writers());
    }

    #[test]
    fn a_pipeline_is_warned_about_once_per_kind() {
        let mut o = Overlaps::default();
        o.dropped(&shader("compute dd"), WHY, [(0x1000, 0x2000, TABLE_UAV)]);
        let reads = [
            read(Via::Index, 0x0, 0x10),
            read(Via::Table(2, 3), 0x1f00, 0x2100),
        ];
        let d = o.draw("vertex v pixel p", &reads);
        let [hit] = &d.hits[..] else { panic!("{d:?}") };
        assert_eq!(hit.read.via, Via::Table(2, 3));
        assert_eq!(hit.bytes, 0x100);
        assert_eq!((&*hit.writer.shader, hit.writer.why), ("compute dd", WHY));
        assert_eq!(
            d.touched,
            [Touch {
                shader: shader("compute dd"),
                why: WHY,
                through: TABLE_UAV
            }]
        );
        assert!(
            o.draw("vertex v pixel p", &reads).hits.is_empty(),
            "warned once"
        );
        // Not even after the frame ends.
        o.present();
        o.dropped(&shader("compute dd"), WHY, [(0x1000, 0x2000, TABLE_UAV)]);
        assert!(o.draw("vertex v pixel p", &reads).hits.is_empty());
        // Another kind is warned about on its own, as is another pipeline.
        o.dropped(&shader("compute ee"), WHY, [(0x0, 0x8, ROOT_UAV)]);
        let d = o.draw("vertex v pixel p", &reads);
        assert_eq!(
            d.hits
                .iter()
                .map(|h| (h.writer.through, h.read.via))
                .collect::<Vec<_>>(),
            [(ROOT_UAV, Via::Index)]
        );
        assert_eq!(d.touched.len(), 2);
        assert_eq!(o.draw("vertex w pixel p", &reads).hits.len(), 2);
        assert_eq!(o.counts.draws_hit, 3);
    }

    #[test]
    fn heap_writers_are_coarse_and_counted_apart() {
        assert_eq!(
            merge(vec![
                (0x30, 0x40),
                (0x0, 0x10),
                (0x10, 0x18),
                (0x8, 0xc),
                (0x50, 0x50)
            ]),
            [(0x0, 0x18), (0x30, 0x40)]
        );
        let ranges = Arc::new(merge(vec![(0x1000, 0x2000), (0x3000, 0x3100)]));
        assert_eq!(heap_overlap(&ranges, 0x2000, 0x3000), None);
        assert_eq!(
            heap_overlap(&ranges, 0x1f00, 0x3080),
            Some(((0x1000, 0x2000), 0x100))
        );
        let mut o = Overlaps::default();
        let s = shader("compute hh");
        o.dropped_heap(&s, WHY, ranges.clone());
        o.dropped_heap(&s, WHY, ranges.clone());
        assert_eq!(o.heap.len(), 1, "once per shader and heap");
        let d = o.draw(
            "vertex v pixel p",
            &[read(Via::Table(0, 0), 0x3010, 0x3020)],
        );
        assert_eq!(d.hits.len(), 1);
        assert_eq!(
            (
                d.hits[0].writer.through,
                precision(d.hits[0].writer.through),
                d.hits[0].writer.lo,
                d.hits[0].bytes
            ),
            (HEAP_UAV, "coarse", 0x3000, 0x10)
        );
        // A precise writer of the same bytes is still warned about.
        o.dropped(&shader("compute tt"), WHY, [(0x3000, 0x3008, TABLE_UAV)]);
        let d = o.draw(
            "vertex v pixel p",
            &[read(Via::Table(0, 0), 0x3000, 0x3020)],
        );
        assert_eq!(
            d.hits
                .iter()
                .map(|h| precision(h.writer.through))
                .collect::<Vec<_>>(),
            ["precise"]
        );
        let c = o.present();
        assert_eq!(
            (
                c.draws_hit,
                c.draws_hit_heap_only,
                c.heap_writers,
                c.heap_ranges
            ),
            (1, 1, 1, 2)
        );
    }

    #[test]
    fn a_root_view_reaches_the_end_of_the_buffer_holding_it() {
        let spans = [(0x1000, 0x9000), (0x2000, 0x3000), (0x4000, 0x5000)];
        assert_eq!(
            extent(&spans, 0x2800),
            Some((0x2800, 0x3000)),
            "the one starting nearest below"
        );
        assert_eq!(
            extent(&spans, 0x3800),
            Some((0x3800, 0x9000)),
            "an outer one past the inner ones"
        );
        assert_eq!(extent(&spans, 0x9000), None);
        assert_eq!(extent(&spans, 0x800), None);
    }

    #[test]
    fn a_summary_counts_whole_dwords_from_the_views_start() {
        let mut s = Summary::default();
        let words: Vec<u8> = [0.0f32, 1.5, -2.0, f32::NAN, 3e7]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        s.add(&words, 0);
        // Two bytes in: the first whole dword starts at the next boundary.
        s.add(&[9, 9, 0, 0, 0, 0, 7], 2);
        assert_eq!((s.read, s.dwords, s.zero), (27, 6, 2));
        assert!((s.zero_share() - 2.0 / 6.0).abs() < 1e-9);
        let text = s.floats.to_string();
        assert!(text.contains("c0 -2.000000..30000000.000000"), "{text}");
        assert!(text.contains("6 read, 1 not finite, 1 over 1e6"), "{text}");
    }

    /// A compute program of `words` with root UAV `u` in slot 1 (user data 4), and the resource
    /// heap at user data 6 when `heap`.
    fn program(words: &[u32], heap: bool) -> victory_shader::NativeProgram {
        use victory_shader::native::{ComputeIo, StageIo};
        use victory_shader::{HwStage, ResourceType, RootParam, RootType, UsedResource};
        let param = |root, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data,
            slot,
        };
        let mut layout = vec![param(RootType::Descriptor, 4, 1)];
        if heap {
            layout.push(param(RootType::ResourceHeap, 6, 252));
        }
        let used = vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 4,
            slot: 1,
        }];
        victory_shader::NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 16,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 8 << 1,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn a_programs_write_profile_names_its_root_uavs_heap_and_computed_stores() {
        const END: u32 = 0xBF81_0000; // s_endpgm
        // global_store_dword v1, v2, s[4:5]: through the root UAV's own SGPRs.
        let root = profile_of_program(&program(&[0xDC70_8000, 0x0004_0201, END], false));
        assert_eq!(
            root,
            WriteProfile {
                root_uavs: vec![1],
                decoded: true,
                ..WriteProfile::default()
            }
        );
        assert!(!root.heap_uav() && !root.computed_writes());
        // global_store_dword v[6:7], v2, off: an address the program computes.
        let computed = profile_of_program(&program(&[0xDC70_8000, 0x007D_0206, END], false));
        assert!(computed.computed_writes() && !computed.heap_uav());
        // A heap-indexing program with a buffer store (buffer_store_dword v1, off, s[8:11]).
        let heap = profile_of_program(&program(&[0xE070_0000, 0x8002_0100, END], true));
        assert!(
            heap.heap && heap.descriptor_stores && heap.heap_uav(),
            "{heap:?}"
        );
        // Code that does not decode: assumed to write anywhere it could.
        let unknown = profile_of_program(&program(&[0xFFFF_FFFF], true));
        assert!(!unknown.decoded && unknown.heap_uav() && unknown.computed_writes());
    }

    #[test]
    fn a_verdict_shows_shares_kinds_and_named_writers() {
        let mut p = Pipe::default();
        let sum = |dwords, zero| Summary {
            dwords,
            zero,
            ..Summary::default()
        };
        p.early.add(&sum(100, 98));
        p.early.add(&sum(100, 100));
        p.late.add(&sum(100, 99));
        p.kinds = [0, 12, 0, 3];
        p.writers.push(Touch {
            shader: shader("compute ab95b63f73b77253"),
            why: "refused",
            through: ROOT_UAV,
        });
        p.unbounded
            .push((shader("compute ae97872296679711"), "refused"));
        assert_eq!(
            p.verdict("vertex 11 pixel 22"),
            Verdict {
                pipeline: "vertex 11 pixel 22".into(),
                views: "2/1/0".into(),
                zero_early: "0.9900".into(),
                zero_late: "0.9900".into(),
                zero_later: "-".into(),
                overlaps: "table uav:0 root uav:12 root view:0 heap uav (any):3".into(),
                writers: "compute ab95b63f73b77253 (refused; root uav)".into(),
                unbounded: "compute ae97872296679711 (refused; title memory (computed))".into(),
            }
        );
    }

    #[test]
    fn late_samples_wait_for_the_gpu_and_are_bounded_per_present() {
        let mut s = State {
            config: Some(Config {
                draws: 512,
                after: 0,
                late: 2,
                later: 3,
                verdict: 0,
            }),
            memory: Some(Arc::new(PoolTable::default())),
            ..State::default()
        };
        for k in 0..3 {
            s.pending.push_back(Pending {
                pipeline: shader("vertex v pixel p"),
                via: Via::Table(0, k),
                addr: 0x1000,
                size: 16,
                early: 1.0,
                watermark: None,
                due: None,
            });
        }
        // The first present after the draws: its command buffers are not done.
        present_in(&mut s, 7, |w| w < 7);
        assert!(s.logged.late.is_empty());
        assert!(s.pending.iter().all(|p| p.watermark == Some(7)));
        // Done: two at this present, the third at the next; each due again 3 presents later.
        present_in(&mut s, 9, |w| w <= 9);
        assert_eq!(
            s.logged.late.iter().map(|l| (l.0, l.1)).collect::<Vec<_>>(),
            [(Via::Table(0, 0), false), (Via::Table(0, 1), false)]
        );
        present_in(&mut s, 9, |_| true);
        assert_eq!(s.logged.late.len(), 3);
        assert_eq!(
            s.pending.iter().map(|p| p.due).collect::<Vec<_>>(),
            [Some(5), Some(5), Some(6)]
        );
        present_in(&mut s, 9, |_| true);
        present_in(&mut s, 9, |_| true);
        assert_eq!(
            s.logged.late.iter().filter(|l| l.1).count(),
            2,
            "the later samples due at present 5"
        );
    }
}
