//! Frame pacing for the window's HUD: when each frame was presented, kept in atomics so the
//! AppKit main thread can read it without taking any lock (a frame stalled on the render thread
//! must not freeze the overlay that shows the stall).
//!
//! `present.rs` calls [`presented`] once per frame, whether the window is visible or not, so the
//! count is frames the title asked the host to show. The ring holds the gap since the previous
//! present, which is what a frame-time sparkline draws; the first frame after a stall shows as one
//! long sample, and [`Pacing::last_micros`] lets a reader say how long ago the last one was.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Frame gaps kept: eight seconds at 60 fps.
pub const SAMPLES: usize = 480;

/// The process's first present: `Instant` is not `const`, so the clock starts on first use.
static START: OnceLock<Instant> = OnceLock::new();
/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);
/// The clock reading (`micros`, see [`now_micros`]) of the last present.
static LAST: AtomicU64 = AtomicU64::new(0);
/// Whether `LAST` holds a present: a present in the clock's first microsecond reads 0, which is
/// not a missing one.
static RECORDED: AtomicBool = AtomicBool::new(false);
/// How many gaps have been written: the next slot is `CURSOR % SAMPLES`.
static CURSOR: AtomicU64 = AtomicU64::new(0);
/// The gap before each present, in microseconds, ring order (`CURSOR`).
static GAPS: [AtomicU64; SAMPLES] = [const { AtomicU64::new(0) }; SAMPLES];

/// Microseconds on the process's own monotonic clock.
fn now_micros() -> u64 {
    START
        .get_or_init(Instant::now)
        .elapsed()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// A frame was presented: count it and record the gap since the previous one.
pub fn presented() {
    let now = now_micros();
    let previous = LAST.swap(now, Ordering::Relaxed);
    FRAMES.fetch_add(1, Ordering::Release);
    if RECORDED.swap(true, Ordering::Relaxed) {
        let slot = CURSOR.fetch_add(1, Ordering::Relaxed) % SAMPLES as u64;
        GAPS[slot as usize].store(now.saturating_sub(previous), Ordering::Relaxed);
    }
}

/// A lock-free copy of the pacing counters, newest last.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Pacing {
    /// Frames presented so far.
    pub frames: u64,
    /// Microseconds since the last present, 0 if none yet.
    pub last_micros: u64,
    /// Microseconds since the clock started (see [`presented`]).
    pub now_micros: u64,
    /// The recent frame gaps in microseconds, oldest first.
    pub recent: Vec<u64>,
}

/// Reads the counters. Cheap and lock-free; the values are individually consistent, the set is a
/// moment behind a frame being recorded (one frame of skew at most, which a HUD cannot show).
pub fn snapshot() -> Pacing {
    let frames = FRAMES.load(Ordering::Acquire);
    let cursor = CURSOR.load(Ordering::Relaxed);
    let written = cursor.min(SAMPLES as u64) as usize;
    let first = if (cursor as usize) < SAMPLES {
        0
    } else {
        (cursor as usize) % SAMPLES
    };
    let mut recent = Vec::with_capacity(written);
    for i in 0..written {
        recent.push(GAPS[(first + i) % SAMPLES].load(Ordering::Relaxed));
    }
    Pacing {
        frames,
        last_micros: LAST.load(Ordering::Relaxed),
        now_micros: now_micros(),
        recent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_present_is_counted_and_its_gap_is_kept() {
        let before = snapshot();
        presented();
        presented();
        let after = snapshot();
        assert!(after.frames >= before.frames + 2);
        assert!(after.last_micros >= before.last_micros);
        assert!(after.recent.len() <= SAMPLES);
        assert!(!after.recent.is_empty(), "a second present records a gap");
        // A gap is a plain difference of two clock readings; never negative.
        assert!(after.recent.iter().all(|&us| us < 60_000_000));
    }

    #[test]
    fn the_ring_keeps_only_the_newest_samples() {
        for _ in 0..(SAMPLES + 5) {
            presented();
        }
        let p = snapshot();
        assert_eq!(p.recent.len(), SAMPLES, "a full ring, no more");
    }
}
