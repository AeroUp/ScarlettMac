//! `VICTORY_TRACE_FRAME=1`: what a frame's command buffers cost, per command buffer and per pass.
//!
//! A world frame's 5 s are not one thing: the CPU encodes the title's passes into a handful of
//! command buffers, the GPU runs them, and the present waits for a drawable. The HUD's CPU and GPU
//! percentages are sums over a tick, so they swing between 100 % (a 1.4 s command buffer completing
//! inside the tick) and 0 % (the render thread encoding); they never say which pass or which call
//! took the frame. This module gives each command buffer a GPU time (`GPUStartTime` to `GPUEndTime`,
//! which Metal fills in on completion), the wall time its encoding took and its encoders' labels.
//! The host drains the window at each present (`victory_gpu::native::glue::present`) and logs one
//! line, with the five command buffers that held the GPU longest.
//!
//! `VICTORY_TRACE_FRAME=passes` additionally puts every pass in a command buffer of its own, so a
//! buffer's GPU time is a pass's time and the ten passes that held the GPU longest come out of the
//! same window. That split serializes the passes (each piece waits for the one before), so those
//! numbers attribute time; the plain frame line keeps the title's real batching.

use std::cell::RefCell;
use std::sync::Mutex;

/// How many completed command buffers the window keeps if no present drains it (a title that
/// stops presenting): the oldest are dropped, so a stalled run does not grow without bound.
const RECORD_CAP: usize = 2048;

/// `VICTORY_TRACE_FRAME`: unset or `0` off, `1` per command buffer, `passes` (or any other value)
/// per pass too. Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_TRACE_FRAME").is_ok_and(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && v != "false"
        })
    })
}

/// Whether per-pass timestamps are wanted (`VICTORY_TRACE_FRAME=passes`).
pub fn passes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        enabled()
            && std::env::var("VICTORY_TRACE_FRAME")
                .is_ok_and(|v| v.trim() == "passes")
    })
}

thread_local! {
    /// The passes of the command buffer being encoded, by key (see [`pass`]).
    static PASSES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Records a pass of the command buffer being encoded, when `VICTORY_TRACE_FRAME=passes` splits a
/// submission into one command buffer a pass: each buffer's GPU time is then one pass's, and its
/// label is the pass. Metal's timestamp counter samples would attribute passes without splitting,
/// but this device (18 GB M3 Pro, AGXG15XFamily, macOS 27) refuses `sampleCountersInBuffer` at every
/// encoder boundary it advertises with `supportsCounterSampling`, and the refusal is a validation
/// assertion that aborts the process. The split is the portable way.
pub(crate) fn pass(key: String) {
    if passes() {
        PASSES.with(|p| p.borrow_mut().push(key));
    }
}

/// The passes recorded for the command buffer being committed.
pub(crate) fn take() -> Vec<String> {
    PASSES.with(|p| std::mem::take(&mut *p.borrow_mut()))
}

/// Drops what a submission that did not reach a commit recorded.
pub(crate) fn clear() {
    PASSES.with(|p| p.borrow_mut().clear());
}

/// How many passes one command buffer holds under `VICTORY_TRACE_FRAME=passes`: one, so each
/// command buffer's GPU time (`GPUStartTime` to `GPUEndTime`, which Metal reports on every device)
/// is one pass's. The split makes each piece wait for the one before it, so the times are for
/// attribution, not a baseline; the frame line without `=passes` keeps the real batching.
pub(crate) fn passes_per_buffer() -> usize {
    1
}

/// What a command buffer cost: its encoding (CPU, from the call to `commit`), its GPU time (from
/// Metal's own start and end times) and, if sampled, its passes' GPU times.
pub struct Buffer {
    pub serial: u64,
    pub queue: usize,
    /// GPU milliseconds, from `GPUStartTime` to `GPUEndTime`.
    pub gpu_ms: f64,
    /// CPU milliseconds spent encoding it (creating the command buffer to `commit`).
    pub encode_ms: f64,
    /// Seconds since the process started: committed, done. `done_at - committed_at` is the wall
    /// time the command buffer took from the commit to the GPU's report (queue wait included).
    pub committed_at: f64,
    pub done_at: f64,
    pub passes: Vec<(String, f64)>,
    /// The pass this command buffer held, when `=passes` gave it one.
    pub pass: Option<String>,
    /// What the command buffer's encoders were (`VICTORY_TRACE_GPU_FAILS` labels), capped: the
    /// frame line names the longest buffers.
    pub label: String,
}

/// A present's window: every command buffer that completed since the last present, aggregated.
pub struct Window {
    pub buffers: u32,
    /// GPU milliseconds summed over the window's buffers (overlapping ones count twice).
    pub gpu_ms: f64,
    /// CPU milliseconds summed over the window's buffers' encoding.
    pub encode_ms: f64,
    /// Command buffers committed and not finished when the window was taken.
    pub in_flight: u32,
    /// Per queue (its pointer): command buffers and GPU milliseconds.
    pub queues: Vec<(usize, u32, f64)>,
    /// Per pass key: total GPU milliseconds and how many passes it was. Sorted, longest first.
    pub passes: Vec<(String, f64, u32)>,
    /// The window's longest command buffers: GPU milliseconds, serial, encoders. Longest first.
    pub buffers_top: Vec<(f64, u64, String)>,
}

static DONE: Mutex<Vec<Buffer>> = Mutex::new(Vec::new());
static IN_FLIGHT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A command buffer was committed: it is in flight from here to its completion.
pub(crate) fn committed() {
    IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A command buffer completed: keep its record for the next present's window. A buffer `=passes`
/// gave one pass of its own counts as that pass, at its whole GPU time.
pub(crate) fn completed(mut b: Buffer) {
    IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(k) = &b.pass {
        b.passes = vec![(k.clone(), b.gpu_ms)];
    }
    if let Ok(mut done) = DONE.lock() {
        if done.len() >= RECORD_CAP {
            done.remove(0);
        }
        done.push(b);
    }
}

/// Takes the window since the last present: the buffers that completed, aggregated.
pub fn window() -> Window {
    let done = std::mem::take(&mut *DONE.lock().unwrap_or_else(|e| e.into_inner()));
    let mut w = Window {
        buffers: 0,
        gpu_ms: 0.0,
        encode_ms: 0.0,
        in_flight: IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed),
        queues: Vec::new(),
        passes: Vec::new(),
        buffers_top: Vec::new(),
    };
    for b in &done {
        w.buffers += 1;
        w.gpu_ms += b.gpu_ms;
        w.encode_ms += b.encode_ms;
        match w.queues.iter_mut().find(|(q, ..)| *q == b.queue) {
            Some((_, n, ms)) => {
                *n += 1;
                *ms += b.gpu_ms;
            }
            None => w.queues.push((b.queue, 1, b.gpu_ms)),
        }
        for (key, ms) in &b.passes {
            match w.passes.iter_mut().find(|(k, ..)| k == key) {
                Some((_, total, n)) => {
                    *total += ms;
                    *n += 1;
                }
                None => w.passes.push((key.clone(), *ms, 1)),
            }
        }
        w.buffers_top.push((b.gpu_ms, b.serial, b.label.clone()));
    }
    w.queues.sort_by(|a, b| b.2.total_cmp(&a.2));
    w.passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    w.buffers_top.sort_by(|a, b| b.0.total_cmp(&a.0));
    w.buffers_top.truncate(5);
    w
}

/// The top `n` passes of a window as one readable line: `total_ms x count key`, longest first.
pub fn top_passes(w: &Window, n: usize) -> String {
    w.passes
        .iter()
        .take(n)
        .map(|(k, ms, c)| format!("{ms:.0}ms x{c} {k}"))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// `s` cut to `n` bytes at a character boundary, for a pass or command buffer label.
pub(crate) fn cap(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// A pass key for a render pass: its first target's size and format and the shader of the pipeline
/// with the most draws in it (the pass's dominant program), so the same pass of later frames
/// aggregates with this one.
pub(crate) fn render_key(width: usize, height: usize, format: i64, dominant: Option<&str>) -> String {
    match dominant {
        Some(d) => format!("render {width}x{height}f{format} {d}"),
        None => format!("render {width}x{height}f{format}"),
    }
}

/// A pass key for a compute pass: its first dispatch's pipeline and grid, so the same dispatch
/// aggregates with its later frames.
pub(crate) fn compute_key(first: Option<&str>, groups: [usize; 3], n: usize) -> String {
    let [x, y, z] = groups;
    match first {
        Some(d) => format!("compute {x}x{y}x{z} x{n} {d}"),
        None => format!("compute {x}x{y}x{z} x{n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(serial: u64, queue: usize, gpu_ms: f64, passes: &[(&str, f64)]) -> Buffer {
        Buffer {
            serial,
            queue,
            gpu_ms,
            encode_ms: 1.0,
            committed_at: 0.0,
            done_at: 1.0,
            passes: passes.iter().map(|(k, ms)| ((*k).to_string(), *ms)).collect(),
            pass: None,
            label: String::new(),
        }
    }

    #[test]
    fn the_window_sums_buffers_by_queue_and_ranks_passes() {
        // No other test records command buffers, so the window holds only these two.
        let _ = window();
        committed();
        completed(buffer(1, 2, 4.0, &[("a", 3.0), ("b", 1.0)]));
        committed();
        completed(buffer(2, 2, 4.0, &[("a", 2.0)]));
        let w = window();
        assert_eq!(w.buffers, 2);
        assert_eq!(w.in_flight, 0);
        assert!((w.gpu_ms - 8.0).abs() < 1e-9);
        assert!((w.encode_ms - 2.0).abs() < 1e-9);
        assert_eq!(w.queues, vec![(2, 2, 8.0)]);
        // The pass that appeared twice is summed and first; its count is the passes, not buffers.
        assert_eq!(w.passes[0].0, "a");
        assert_eq!(w.passes[0].2, 2);
        assert!((w.passes[0].1 - 5.0).abs() < 1e-9);
        assert_eq!(w.passes[1].1, 1.0);
        assert_eq!(top_passes(&w, 1), "5ms x2 a");
        // A second window has nothing: the first took it all.
        assert_eq!(window().buffers, 0);
    }

    #[test]
    fn a_pass_key_names_the_target_and_the_dominant_pipeline() {
        assert_eq!(
            render_key(1920, 1080, 10, Some("ps 5f8b2e3b")),
            "render 1920x1080f10 ps 5f8b2e3b"
        );
        assert_eq!(render_key(64, 64, 3, None), "render 64x64f3");
        assert_eq!(compute_key(Some("cs 4359f972"), [60, 34, 1], 3), "compute 60x34x1 x3 cs 4359f972");
        assert_eq!(compute_key(None, [8, 8, 1], 0), "compute 8x8x1 x0");
    }
}
