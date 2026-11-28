//! `VICTORY_TRACE_GPU_VERTICES=1` (#8): what feeds the vertices of draws whose vertex shader
//! fetches them itself (GPU-driven models: an empty input layout, vertices read from views), which
//! `VICTORY_DUMP_VERTICES` (input-assembler streams only) does not see.
//!
//! Two parts, both logged as numbers only (no bytes are logged or written):
//!
//! - Dropped writers. A dispatch dropped for want of a host compute pipeline (a refused or missing
//!   shader) never writes what it would have. The title ranges it could have written are noted
//!   with its shader hash: its tables' buffer unordered-access views (object or placed, as
//!   [`NativeHost::table_uav_ranges`] reads them) and its root views. A refused program does not
//!   say which root views it writes, so every one is noted (`through = "root view"`), from its
//!   address to the end of the buffer resource that holds it, or one byte when none does. The set
//!   is per frame, bounded, and cleared at each present. Each draw's vertex streams, index buffer,
//!   and the vertex shader's own buffer reads (root views, table views) are checked against it: the
//!   first draw of each pipeline that reads such a range is warned about once, and a line per
//!   frame counts the dropped writers and the draws that read them.
//! - Statistics. For the first draws (`VICTORY_TRACE_GPU_VERTICES_DRAWS=<draws>[,<seconds>]`,
//!   default 512 from the first such draw) whose vertex shader reads no attribute through the input
//!   assembler, each buffer view that shader reads is summarised once per pipeline and view: its
//!   range, how much of it the title maps (a table view over a gap or unmapped memory is the null
//!   view, zeros), the share of zero dwords and its dwords as float32 (range, not finite, over
//!   1e6), over its first 1 MiB. A view the host makes the null view is warned about for every such
//!   draw's pipeline, whatever the draw count.
//!
//! Recording and submission are on different threads, so the writers are noted when a list is
//! recorded and the draws checked when it is submitted. shortcut: a writer noted after a present
//! for work submitted before it is cleared too early, and a writer dropped in an earlier frame (a
//! one-time build at load) is not seen; widen the window if a run needs it.
//!
//! Off unless set: one cached check per draw and dropped dispatch.

use std::collections::HashSet;
use std::sync::Arc;

use super::host::NativeHost;
use super::list::{Binding, Root};
use victory_backend::resources::GraphicsBackend;

/// Ranges noted per frame at most; more are counted, not kept.
const MAX_WRITERS: usize = 4096;
/// Pipelines, and (pipeline, view) pairs, remembered at most for the once-only lines.
const MAX_SEEN: usize = 8192;
/// Bytes of a view the statistics read at most.
const SAMPLE: u64 = 1 << 20;
/// Null-view warnings logged at most.
const MAX_NULL_LINES: u32 = 1024;

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

/// `VICTORY_TRACE_GPU_VERTICES_DRAWS=<draws>[,<seconds>]`: how many draws get statistics, from
/// how many seconds after the first one.
fn draws_config() -> (u32, u64) {
    static CONFIG: std::sync::OnceLock<(u32, u64)> = std::sync::OnceLock::new();
    *CONFIG.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_GPU_VERTICES_DRAWS").unwrap_or_default();
        let mut parts = v.split(',');
        let draws = parts
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(512);
        let after = parts
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        (draws, after)
    })
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
    /// `table uav` or `root view`.
    pub through: &'static str,
}

/// A range a draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub via: Via,
    pub lo: u64,
    pub hi: u64,
}

/// The first draw of a pipeline that reads what a dropped dispatch would have written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Hit {
    pub pipeline: Arc<str>,
    pub read: Read,
    pub writer: Writer,
    /// Bytes the read and the written range share.
    pub bytes: u64,
}

/// One frame's counts, for the line at its present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameCounts {
    /// Dispatches dropped, and the distinct ranges they noted (and how many past the bound).
    pub dispatches: u64,
    pub ranges: usize,
    pub over: u64,
    /// Draws checked, those reading a noted range, and their bindings that do.
    pub draws: u64,
    pub draws_hit: u64,
    pub reads_hit: u64,
}

/// The dropped writers of one frame and the draws that read them.
#[derive(Default)]
pub(crate) struct Overlaps {
    writers: Vec<Writer>,
    seen: HashSet<(u64, u64, Arc<str>, &'static str)>,
    counts: FrameCounts,
    /// Pipelines already warned about (for the run).
    warned: HashSet<Arc<str>>,
}

impl Overlaps {
    /// A dispatch of `shader` was dropped; it may have written `ranges` (`[lo, hi)`).
    pub(crate) fn dropped(
        &mut self,
        shader: &Arc<str>,
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
            });
        }
    }

    pub(crate) fn has_writers(&self) -> bool {
        !self.writers.is_empty()
    }

    /// The noted range sharing the most bytes with `[lo, hi)`, and how many.
    pub(crate) fn overlap(&self, lo: u64, hi: u64) -> Option<(&Writer, u64)> {
        self.writers
            .iter()
            .filter_map(|w| {
                let n = hi.min(w.hi).saturating_sub(lo.max(w.lo));
                (n > 0).then_some((w, n))
            })
            .max_by_key(|&(_, n)| n)
    }

    /// Checks a draw of `pipeline` that reads `reads`: counted, and the first hit returned the
    /// first time this pipeline has one.
    pub(crate) fn draw(&mut self, pipeline: &str, reads: &[Read]) -> Option<Hit> {
        self.counts.draws += 1;
        if self.writers.is_empty() {
            return None;
        }
        let mut first = None;
        let mut hits = 0;
        for r in reads {
            if let Some((w, n)) = self.overlap(r.lo, r.hi) {
                hits += 1;
                if first.is_none() {
                    first = Some((*r, w.clone(), n));
                }
            }
        }
        let (read, writer, bytes) = first?;
        self.counts.draws_hit += 1;
        self.counts.reads_hit += hits;
        if self.warned.contains(pipeline) || self.warned.len() >= MAX_SEEN {
            return None;
        }
        let pipeline: Arc<str> = pipeline.into();
        self.warned.insert(pipeline.clone());
        Some(Hit {
            pipeline,
            read,
            writer,
            bytes,
        })
    }

    /// The frame ended: its counts, and its writers forgotten.
    pub(crate) fn present(&mut self) -> FrameCounts {
        let mut counts = std::mem::take(&mut self.counts);
        counts.ranges = self.writers.len();
        self.writers.clear();
        self.seen.clear();
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
        if self.dwords == 0 {
            0.0
        } else {
            self.zero as f64 / self.dwords as f64
        }
    }
}

/// Reads the first [`SAMPLE`] bytes of `[addr, addr + len)` where the title maps them.
fn summarise(memory: &super::memory::PoolTable, addr: u64, len: u64) -> Summary {
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

#[derive(Default)]
struct State {
    overlaps: Overlaps,
    /// The buffer resources' ranges by start, made at the first need in a frame.
    spans: Option<Vec<(u64, u64)>>,
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
    /// What was logged, for tests: hits, null views `(view, address, size, how)`, and summaries
    /// `(view, bytes read, dwords, zero dwords)`.
    #[cfg(test)]
    logged: Logged,
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct Logged {
    pub hits: Vec<Hit>,
    pub null: Vec<(Via, u64, u64, &'static str)>,
    pub summaries: Vec<(Via, u64, u64, u64)>,
}

impl State {
    fn extent<B: GraphicsBackend>(&mut self, host: &NativeHost<B>, a: u64) -> (u64, u64) {
        let spans = self.spans.get_or_insert_with(|| host.buffer_spans());
        extent(spans, a).unwrap_or((a, a.saturating_add(1)))
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

/// A dispatch of `shader` dropped for want of a host compute pipeline, with root parameters
/// `root`: notes what it may have written.
pub(crate) fn dropped_dispatch<B: GraphicsBackend>(
    host: &NativeHost<B>,
    root: &[Option<Root<B>>],
    shader: &str,
) {
    let tables = host.table_uav_ranges(root);
    let shader: Arc<str> = shader.into();
    with_state(|s| {
        let mut ranges: Vec<(u64, u64, &'static str)> = tables
            .iter()
            .map(|&(a, n)| (a, a.saturating_add(n), "table uav"))
            .collect();
        for r in root.iter().flatten() {
            if let Root::View(a) = r
                && *a != 0
            {
                let (lo, hi) = s.extent(host, *a);
                ranges.push((lo, hi, "root view"));
            }
        }
        s.overlaps.dropped(&shader, ranges);
    });
}

/// The frame ended: its count line, and its writers forgotten.
pub(crate) fn present() {
    let counts = with_state(|s| {
        s.spans = None;
        s.overlaps.present()
    });
    if counts.dispatches > 0 || counts.draws_hit > 0 {
        tracing::info!(
            frame = crate::frame::current(),
            dropped_dispatches = counts.dispatches,
            ranges = counts.ranges,
            ranges_over_bound = counts.over,
            draws = counts.draws,
            draws_reading_them = counts.draws_hit,
            reads = counts.reads_hit,
            "gpu vertices: what dropped dispatches would have written, and the draws that read it"
        );
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
) -> Option<Hit> {
    let views = binding.map(|b| host.vs_views(root, b)).unwrap_or_default();
    let name = binding.map_or("(no translated binding)", |b| &*b.name);
    let key = binding.map_or(0, |b| b.name.as_ptr() as usize);
    let size = |n: u32| match n {
        0 | u32::MAX => 1,
        n => u64::from(n),
    };
    let gpu_fetched = binding.is_some_and(|b| b.vertex_attributes == 0) && !mesh;
    with_state(|s| {
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
        let hit = if s.overlaps.has_writers() {
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
        if let Some(h) = &hit {
            #[cfg(test)]
            s.logged.hits.push(h.clone());
            tracing::warn!(
                pipeline = &*h.pipeline,
                binding = %h.read.via,
                read = format!("{:#x}+{:#x}", h.read.lo, h.read.hi - h.read.lo),
                dropped = &*h.writer.shader,
                through = h.writer.through,
                written = format!("{:#x}+{:#x}", h.writer.lo, h.writer.hi - h.writer.lo),
                overlap = h.bytes,
                "gpu vertices: a draw reads what a dropped dispatch would have written this frame (first for this pipeline)"
            );
        }
        if !gpu_fetched {
            return hit;
        }
        let (limit, after) = draws_config();
        let start = *s.start.get_or_insert_with(std::time::Instant::now);
        let counted = start.elapsed().as_secs() >= after && s.gpu_draws < limit;
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
        hit
    })
}

#[cfg(test)]
pub(crate) fn logged_for_test() -> Logged {
    with_state(|s| s.logged.clone())
}

#[cfg(test)]
pub(crate) fn present_for_test() -> FrameCounts {
    with_state(|s| {
        s.spans = None;
        s.overlaps.present()
    })
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

    #[test]
    fn ranges_overlap_only_where_they_share_bytes() {
        let mut o = Overlaps::default();
        o.dropped(
            &shader("compute aa"),
            [(0x1000, 0x2000, "table uav"), (0x8000, 0x8001, "root view")],
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
            Some(("root view", 1))
        );
        // Empty and repeated ranges are not kept.
        o.dropped(
            &shader("compute aa"),
            [(0x5000, 0x5000, "table uav"), (0x1000, 0x2000, "table uav")],
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
            (0..MAX_WRITERS as u64 + 5).map(|k| (k * 16, k * 16 + 8, "table uav")),
        );
        assert_eq!(o.writers.len(), MAX_WRITERS);
        assert_eq!(o.present().over, 5);
    }

    #[test]
    fn a_present_clears_the_frames_writers() {
        let mut o = Overlaps::default();
        o.dropped(&shader("compute cc"), [(0x1000, 0x2000, "table uav")]);
        assert!(
            o.draw(
                "vertex v1 pixel p1",
                &[read(Via::Stream(0), 0x1800, 0x1900)]
            )
            .is_some()
        );
        let counts = o.present();
        assert_eq!(
            counts,
            FrameCounts {
                dispatches: 1,
                ranges: 1,
                over: 0,
                draws: 1,
                draws_hit: 1,
                reads_hit: 1
            }
        );
        assert!(!o.has_writers());
        // The next frame: nothing dropped, nothing read, and the same range may be noted again.
        assert_eq!(
            o.draw(
                "vertex v2 pixel p2",
                &[read(Via::Stream(0), 0x1800, 0x1900)]
            ),
            None
        );
        assert_eq!(
            o.present(),
            FrameCounts {
                draws: 1,
                ..FrameCounts::default()
            }
        );
        o.dropped(&shader("compute cc"), [(0x1000, 0x2000, "table uav")]);
        assert!(o.has_writers());
    }

    #[test]
    fn a_pipeline_is_warned_about_once() {
        let mut o = Overlaps::default();
        o.dropped(&shader("compute dd"), [(0x1000, 0x2000, "table uav")]);
        let reads = [
            read(Via::Index, 0x0, 0x10),
            read(Via::Table(2, 3), 0x1f00, 0x2100),
        ];
        let hit = o.draw("vertex v pixel p", &reads).expect("first hit");
        assert_eq!(hit.read.via, Via::Table(2, 3));
        assert_eq!(hit.bytes, 0x100);
        assert_eq!(&*hit.writer.shader, "compute dd");
        assert_eq!(o.draw("vertex v pixel p", &reads), None, "warned once");
        // Not even after the frame ends.
        o.present();
        o.dropped(&shader("compute dd"), [(0x1000, 0x2000, "table uav")]);
        assert_eq!(o.draw("vertex v pixel p", &reads), None);
        // Another pipeline is warned about on its own.
        assert!(o.draw("vertex w pixel p", &reads).is_some());
        assert_eq!(o.counts.draws_hit, 2);
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
}
