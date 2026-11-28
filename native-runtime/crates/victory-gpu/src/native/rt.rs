//! Per-frame ray-tracing telemetry on the native host.
//!
//! The title builds its acceleration structures with the console's BVH builder
//! ([`super::builder`]) and walks them from compute programs that contain
//! `image_bvh*_intersect_ray` (`victory_shader`'s BVH lowering). Both live on the
//! title's own lists, so the host can count them as it records them:
//!
//! * **Acceleration-structure builds**: dispatches of a builder pipeline, with
//!   the thread counts of a `DispatchX` (one thread a node for its node
//!   writers) or the group counts of a plain dispatch, and any the builder's
//!   hold dropped (`VICTORY_BVH_BUILDER=1`).
//! * **Ray dispatches**: dispatches of a program that walks a structure, direct
//!   and indirect, and for an indirect one the record's three group counts as
//!   title memory holds them when the command is recorded (the GPU may still
//!   write them earlier in the same submission; a zero there says the pass was
//!   recorded with no groups at least at that point).
//! * **Dropped ray work**: a walked program's dispatch the host could not run
//!   (no host pipeline) and a builder dispatch its hold dropped.
//!
//! `VICTORY_TRACE_RT=1` logs one line a presented frame. Without it a line is
//! logged every 256 frames that carried any of this work, so a normal run shows
//! whether ray tracing ran at all. Counts are what the host recorded; the GPU
//! reads an indirect record later than the count recorded here.
//!
//! Observed, not console behaviour: the counters and their lines.

use std::collections::BTreeSet;
use std::sync::Mutex;

/// One frame's counters.
#[derive(Default, Clone)]
struct Frame {
    /// The frame the counters belong to (`u64::MAX`: none yet).
    frame: u64,
    /// The builder's dispatches that ran, and the nodes their counts name
    /// (thread counts of a `DispatchX`, or groups of a plain dispatch).
    builder_dispatches: u64,
    builder_nodes: u64,
    builder_dims: [u64; 3],
    /// The builder's dispatches a hold dropped.
    builder_dropped: u64,
    /// Walked programs' dispatches: direct and indirect, the indirect records
    /// recorded, how many of those were zero, the groups they asked for, and
    /// dispatches dropped for want of a host pipeline.
    ray_direct: u64,
    ray_indirect: u64,
    ray_records: u64,
    ray_zero: u64,
    ray_groups: u64,
    ray_dropped: u64,
    /// The walked programs involved ("compute <hash>"), each with its indirect records,
    /// zero records and groups asked for.
    ray_shaders: BTreeSet<String>,
    by_shader: std::collections::BTreeMap<String, [u64; 4]>,
}

/// The counters since the last present.
static FRAME: Mutex<Frame> = Mutex::new(Frame {
    frame: u64::MAX,
    builder_dispatches: 0,
    builder_nodes: 0,
    builder_dims: [0; 3],
    builder_dropped: 0,
    ray_direct: 0,
    ray_indirect: 0,
    ray_records: 0,
    ray_zero: 0,
    ray_groups: 0,
    ray_dropped: 0,
    ray_shaders: BTreeSet::new(),
    by_shader: std::collections::BTreeMap::new(),
});

fn with(f: impl FnOnce(&mut Frame)) {
    let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
    let now = crate::frame::current();
    if frame.frame != now {
        *frame = Frame {
            frame: now,
            ..Default::default()
        };
    }
    f(&mut frame);
}

/// A builder dispatch that ran, with the counts it asked for: its `DispatchX`
/// thread counts (one a node) or a plain dispatch's groups.
pub(crate) fn note_builder(counts: [u32; 3], threads: bool) {
    with(|f| {
        f.builder_dispatches += 1;
        for (d, c) in f.builder_dims.iter_mut().zip(counts) {
            *d += u64::from(c);
        }
        let nodes =
            u64::from(counts[0]) * u64::from(counts[1].max(1)) * u64::from(counts[2].max(1));
        f.builder_nodes += if threads { nodes } else { 0 };
    });
}

/// A builder dispatch a hold dropped.
pub(crate) fn note_builder_dropped() {
    with(|f| f.builder_dropped += 1);
}

/// A walked program's direct dispatch: its group counts.
pub(crate) fn note_ray_direct(shader: &str, groups: [u32; 3]) {
    with(|f| {
        f.ray_direct += 1;
        let n = groups.iter().map(|&g| u64::from(g)).product::<u64>();
        f.ray_groups += n;
        f.ray_shaders.insert(shader.to_owned());
        let e = f.by_shader.entry(shader.to_owned()).or_default();
        e[3] += n;
    });
}

/// A walked program's indirect dispatch: the record's three words as title
/// memory holds them when the command is recorded.
pub(crate) fn note_ray_indirect(shader: &str, record: [u32; 3]) {
    with(|f| {
        f.ray_indirect += 1;
        f.ray_records += 1;
        let zero = record.contains(&0);
        if zero {
            f.ray_zero += 1;
        }
        let n = if zero { 0 } else { record.iter().map(|&g| u64::from(g)).product::<u64>() };
        f.ray_groups += n;
        f.ray_shaders.insert(shader.to_owned());
        let e = f.by_shader.entry(shader.to_owned()).or_default();
        e[0] += 1;
        e[1] += 1;
        e[2] += u64::from(zero);
        e[3] += n;
    });
}

/// A walked program's dispatch the host dropped (no host pipeline).
pub(crate) fn note_ray_dropped(shader: &str) {
    with(|f| {
        f.ray_dropped += 1;
        f.ray_shaders.insert(shader.to_owned());
    });
}

/// Whether `VICTORY_TRACE_RT` is set: the per-frame lines and the indirect readback
/// (`victory_metal::gfx::rt_tracing`).
pub(crate) fn tracing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRACE_RT").is_some())
}

fn per_frame() -> bool {
    tracing()
}

/// At each present: the frame's builder and ray work. Every frame under
/// `VICTORY_TRACE_RT`, else every 256th frame that carried any (and the first
/// one), so a run does not fill its log with counters.
pub(crate) fn present() {
    let taken = {
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.frame == u64::MAX {
            return;
        }
        std::mem::take(&mut *frame)
    };
    let any = taken.builder_dispatches
        + taken.builder_dropped
        + taken.ray_direct
        + taken.ray_indirect
        + taken.ray_dropped
        > 0;
    let frame = taken.frame;
    let every = frame.is_multiple_of(256);
    if !per_frame() && !(any && every) {
        return;
    }
    let shaders: Vec<String> = taken
        .by_shader
        .iter()
        .map(|(name, n)| format!("{name} indirect={} zero={} groups={}", n[0], n[2], n[3]))
        .collect();
    tracing::info!(
        frame,
        builder_dispatches = taken.builder_dispatches,
        builder_nodes = taken.builder_nodes,
        builder_dims = ?taken.builder_dims,
        builder_dropped = taken.builder_dropped,
        ray_dispatches = taken.ray_direct + taken.ray_indirect,
        ray_direct = taken.ray_direct,
        ray_indirect = taken.ray_indirect,
        ray_records = taken.ray_records,
        ray_zero = taken.ray_zero,
        ray_groups = taken.ray_groups,
        ray_dropped = taken.ray_dropped,
        shaders = ?shaders,
        "rt: a frame's acceleration-structure and ray work"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    static LOCK: Mutex<()> = Mutex::new(());

    /// Builder and ray counts add up per frame; a zero indirect record counts as
    /// zero groups; a new frame starts over.
    #[test]
    fn a_frame_counts_builder_nodes_and_ray_groups() {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        *FRAME.lock().unwrap() = Frame {
            frame: u64::MAX,
            ..Default::default()
        };
        note_builder([45, 1, 1], true);
        note_builder([8, 2, 1], false);
        note_ray_direct("compute a", [12, 1, 1]);
        note_ray_indirect("compute b", [0, 0, 0]);
        note_ray_indirect("compute b", [5, 1, 1]);
        note_ray_dropped("compute c");
        let f = FRAME.lock().unwrap().clone();
        assert_eq!((f.builder_dispatches, f.builder_nodes), (2, 45));
        assert_eq!((f.ray_direct, f.ray_indirect), (1, 2));
        assert_eq!((f.ray_records, f.ray_zero), (2, 1));
        assert_eq!(f.ray_groups, 12 + 5);
        assert_eq!(f.ray_dropped, 1);
        assert_eq!(f.ray_shaders.len(), 3);
    }
}
