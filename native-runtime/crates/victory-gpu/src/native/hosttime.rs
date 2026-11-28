//! `VICTORY_TRACE_FRAME=1`: the host's side of a frame, by where the time went.
//!
//! A world frame's 5 s were split between the title's own threads and the host's calls: the frame
//! line logs, for each present, the time spent in `ExecuteCommandLists` (waiting for the lock that
//! keeps submissions in order, reading and uploading textures off the host's lock, and encoding the
//! lists under it), the time the host's lock was waited for and held, and what pipeline compiles
//! and texture reads cost. [`victory_metal::frametime`] gives the GPU side of the same present
//! (per command buffer, and per pass with `VICTORY_TRACE_FRAME=passes`); this module is the CPU
//! side, and `native::glue::present` joins the two into the frame line.
//!
//! Everything here is atomics: the title calls the host from several threads and the present must
//! not wait on any of them. The counters are what has accumulated since the last present; nothing
//! is logged when `VICTORY_TRACE_FRAME` is off, and the adders are one branch on a cached answer.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Whether the frame line is on (`VICTORY_TRACE_FRAME`, any value but `0`/empty).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_TRACE_FRAME").is_ok_and(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && v != "false"
        })
    })
}

/// The process's start, for the trace's `t=` figures (the same epoch as the Metal trace's).
fn since_start() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[derive(Default)]
struct Counters {
    // --- host calls ---
    submits: AtomicU64,
    /// Wall time in `ExecuteCommandLists`, from entry to return.
    submit_us: AtomicU64,
    /// Of that, waiting for the lock that keeps submissions in order.
    order_wait_us: AtomicU64,
    /// Reading, hashing and detiling textures without the host's lock.
    prepare_us: AtomicU64,
    /// Uploading them under it: the time in the call, and how much of it waited for the lock.
    apply_us: AtomicU64,
    apply_wait_us: AtomicU64,
    /// Encoding and committing the lists under it, and how much of it waited.
    encode_us: AtomicU64,
    encode_wait_us: AtomicU64,
    presents: AtomicU64,
    present_us: AtomicU64,
    // --- the host's lock, every caller ---
    lock_calls: AtomicU64,
    lock_wait_us: AtomicU64,
    lock_hold_us: AtomicU64,
    // --- pipelines ---
    /// Pipelines put off for a later frame (their draws dropped this one).
    put_offs: AtomicU64,
    /// Pipeline creations asked of the backend, and what they took (compiling or waiting).
    pipes_asked: AtomicU64,
    pipes_us: AtomicU64,
    /// Of those, the ones Metal had to compile here (nothing in the cache): the frame's cost.
    pipes_made: AtomicU64,
    // --- textures ---
    /// Submissions whose texture work was left for the next frame (a frame's work budget).
    sync_deferred: AtomicU64,
    sync_batches: AtomicU64,
    sync_textures: AtomicU64,
    sync_first: AtomicU64,
    sync_unchanged: AtomicU64,
    sync_bytes: AtomicU64,
    sync_prepare_us: AtomicU64,
    sync_upload_us: AtomicU64,
    sync_read_us: AtomicU64,
    sync_hash_us: AtomicU64,
    sync_detile_us: AtomicU64,
    sync_probe_us: AtomicU64,
    /// The scanner's thread (`textures::scan_round`): bytes of texture memory it read and hashed,
    /// and the time each took.
    scan_bytes: AtomicU64,
    scan_read_us: AtomicU64,
    scan_hash_us: AtomicU64,
}

static C: Counters = Counters {
    submits: AtomicU64::new(0),
    submit_us: AtomicU64::new(0),
    order_wait_us: AtomicU64::new(0),
    prepare_us: AtomicU64::new(0),
    apply_us: AtomicU64::new(0),
    apply_wait_us: AtomicU64::new(0),
    encode_us: AtomicU64::new(0),
    encode_wait_us: AtomicU64::new(0),
    presents: AtomicU64::new(0),
    present_us: AtomicU64::new(0),
    lock_calls: AtomicU64::new(0),
    lock_wait_us: AtomicU64::new(0),
    lock_hold_us: AtomicU64::new(0),
    put_offs: AtomicU64::new(0),
    pipes_asked: AtomicU64::new(0),
    pipes_us: AtomicU64::new(0),
    pipes_made: AtomicU64::new(0),
    sync_deferred: AtomicU64::new(0),
    sync_batches: AtomicU64::new(0),
    sync_textures: AtomicU64::new(0),
    sync_first: AtomicU64::new(0),
    sync_unchanged: AtomicU64::new(0),
    sync_bytes: AtomicU64::new(0),
    sync_prepare_us: AtomicU64::new(0),
    sync_upload_us: AtomicU64::new(0),
    sync_read_us: AtomicU64::new(0),
    sync_hash_us: AtomicU64::new(0),
    sync_detile_us: AtomicU64::new(0),
    sync_probe_us: AtomicU64::new(0),
    scan_bytes: AtomicU64::new(0),
    scan_read_us: AtomicU64::new(0),
    scan_hash_us: AtomicU64::new(0),
};

fn add(c: &AtomicU64, n: u64) {
    c.fetch_add(n, Ordering::Relaxed);
}

fn us(d: Duration) -> u64 {
    d.as_micros() as u64
}

thread_local! {
    /// How long this thread's last `TITLE` lock call waited: a caller that times a whole
    /// `with_title` phase splits it into waiting for the lock and working under it.
    static LAST_WAIT_US: Cell<u64> = const { Cell::new(0) };
    /// A number for this host thread, for a stall report that names which thread is in which
    /// phase (the loader's own dump names the title's threads by their Windows ids).
    static TID: u64 = NEXT_TID.fetch_add(1, Ordering::Relaxed);
}

static NEXT_TID: AtomicU64 = AtomicU64::new(1);

/// What a host call is doing, for `VICTORY_STALL_SECONDS`: a stall report that only named the
/// frame would not say whether the render thread was reading textures, encoding or waiting for
/// the GPU. The most recent phase a host call entered is kept with its start; two atomic stores
/// per phase, so this is on whatever the trace switch says.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Waiting for the lock that keeps submissions in order.
    Order,
    /// Reading, hashing and detiling textures, without the host's lock.
    Prepare,
    /// Uploading textures, probing and applying the scan, under the host's lock.
    Apply,
    /// Recording and committing the command lists, under the host's lock.
    Encode,
    /// A present, and whatever it waits for.
    Present,
    /// Making and compiling a pipeline.
    Pipeline,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Order => "waiting for the submission order",
            Phase::Prepare => "reading and detiling textures",
            Phase::Apply => "uploading textures (host lock)",
            Phase::Encode => "encoding command lists (host lock)",
            Phase::Present => "presenting",
            Phase::Pipeline => "making a pipeline",
        }
    }
}

/// The phase a host call is in: its one-based code (0: none), when it began (µs since the
/// process's clock started) and the host thread's number.
static PHASE: AtomicUsize = AtomicUsize::new(0);
static PHASE_AT_US: AtomicU64 = AtomicU64::new(0);
static PHASE_TID: AtomicU64 = AtomicU64::new(0);

fn phase_code(phase: Phase) -> usize {
    phase as usize + 1
}

/// A host call entered `phase`.
pub fn enter(phase: Phase) {
    PHASE_AT_US.store((since_start() * 1e6) as u64, Ordering::Relaxed);
    PHASE_TID.store(TID.with(|t| *t), Ordering::Relaxed);
    PHASE.store(phase_code(phase), Ordering::Release);
}

/// No host call is in a phase any more (the call returned).
pub fn leave() {
    PHASE.store(0, Ordering::Release);
}

/// The phase the last host call entered and how long ago, if one is in a phase: what a stall
/// report says when no present has come for a while.
pub fn phase_now() -> Option<(&'static str, f64, u64)> {
    let code = PHASE.load(Ordering::Acquire);
    if code == 0 {
        return None;
    }
    let phase = match code - 1 {
        0 => Phase::Order,
        1 => Phase::Prepare,
        2 => Phase::Apply,
        3 => Phase::Encode,
        4 => Phase::Present,
        5 => Phase::Pipeline,
        _ => return None,
    };
    let at = PHASE_AT_US.load(Ordering::Relaxed) as f64 / 1e6;
    Some((phase.name(), (since_start() - at).max(0.0), PHASE_TID.load(Ordering::Relaxed)))
}

/// The host's lock was waited for and then held: from [`lock_wait`] and [`lock_held`].
pub(crate) fn lock_waited(d: Duration) {
    LAST_WAIT_US.with(|w| w.set(us(d)));
    if enabled() {
        add(&C.lock_calls, 1);
        add(&C.lock_wait_us, us(d));
    }
}

/// How long this thread's last lock call waited for the lock.
pub(crate) fn last_wait() -> Duration {
    Duration::from_micros(LAST_WAIT_US.with(|w| w.get()))
}

pub(crate) fn lock_held(d: Duration) {
    if enabled() {
        add(&C.lock_hold_us, us(d));
    }
}

/// One `ExecuteCommandLists`: what it cost, by phase.
pub(crate) struct Submit {
    pub order_wait: Duration,
    pub prepare: Duration,
    pub apply: Duration,
    /// Of `apply`, waiting for the host's lock (the rest is the uploads).
    pub apply_wait: Duration,
    pub encode: Duration,
    /// Of `encode`, waiting for the host's lock.
    pub encode_wait: Duration,
    pub wall: Duration,
}

pub(crate) fn submitted(s: Submit) {
    if enabled() {
        add(&C.submits, 1);
        add(&C.submit_us, us(s.wall));
        add(&C.order_wait_us, us(s.order_wait));
        add(&C.prepare_us, us(s.prepare));
        add(&C.apply_us, us(s.apply));
        add(&C.apply_wait_us, us(s.apply_wait));
        add(&C.encode_us, us(s.encode));
        add(&C.encode_wait_us, us(s.encode_wait));
    }
}

/// A present call's own cost (the trace line is logged by the caller).
pub(crate) fn presented(d: Duration) {
    if enabled() {
        add(&C.presents, 1);
        add(&C.present_us, us(d));
    }
}

/// A pipeline put off for a later frame.
pub(crate) fn put_off() {
    if enabled() {
        add(&C.put_offs, 1);
    }
}

/// A pipeline was asked of the backend: `made` when Metal compiled it (else it was cached).
pub(crate) fn pipeline(d: Duration, made: bool) {
    if enabled() {
        add(&C.pipes_asked, 1);
        add(&C.pipes_us, us(d));
        if made {
            add(&C.pipes_made, 1);
        }
    }
}

/// Why textures were listed for sync since the last present: reason, count, bytes of title memory.
static WHY: Mutex<Vec<(&'static str, u64, u64)>> = Mutex::new(Vec::new());

/// One reason's textures of one submission (added to the window's totals).
pub(crate) fn texture_why(reason: &'static str, count: u64, bytes: u64) {
    if !enabled() || count == 0 {
        return;
    }
    let mut why = WHY.lock().unwrap_or_else(|e| e.into_inner());
    match why.iter_mut().find(|(r, ..)| *r == reason) {
        Some((_, c, b)) => {
            *c += count;
            *b += bytes;
        }
        None => why.push((reason, count, bytes)),
    }
}

/// Takes the window's reasons, longest bytes first.
pub fn take_why() -> Vec<(&'static str, u64, u64)> {
    let mut why = WHY.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = std::mem::take(&mut *why);
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out
}

/// A submission left its textures for the next frame: the frame's texture-work budget was spent.
pub(crate) fn sync_deferred() {
    if enabled() {
        add(&C.sync_deferred, 1);
    }
}

/// What a submission's texture work cost, as `textures::SyncCost` measures it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn textures(
    batches: u64,
    textures: u64,
    first: u64,
    unchanged: u64,
    bytes: u64,
    probe: Duration,
    upload: Duration,
    prepare: Duration,
    read: Duration,
    hash: Duration,
    detile: Duration,
) {
    if enabled() {
        add(&C.sync_batches, batches);
        add(&C.sync_textures, textures);
        add(&C.sync_first, first);
        add(&C.sync_unchanged, unchanged);
        add(&C.sync_bytes, bytes);
        add(&C.sync_probe_us, us(probe));
        add(&C.sync_upload_us, us(upload));
        add(&C.sync_prepare_us, us(prepare));
        add(&C.sync_read_us, us(read));
        add(&C.sync_hash_us, us(hash));
        add(&C.sync_detile_us, us(detile));
    }
}

/// What a scanner round read and hashed (its thread's time, not a submission's).
pub(crate) fn scanned(bytes: u64, read: Duration, hash: Duration) {
    if enabled() {
        add(&C.scan_bytes, bytes);
        add(&C.scan_read_us, us(read));
        add(&C.scan_hash_us, us(hash));
    }
}

/// What has accumulated since the last present.
#[derive(Default, Debug, Clone, Copy)]
pub struct Taken {
    pub submits: u64,
    pub submit_us: u64,
    pub order_wait_us: u64,
    pub prepare_us: u64,
    pub apply_us: u64,
    pub apply_wait_us: u64,
    pub encode_us: u64,
    pub encode_wait_us: u64,
    pub presents: u64,
    pub present_us: u64,
    pub lock_calls: u64,
    pub lock_wait_us: u64,
    pub lock_hold_us: u64,
    pub put_offs: u64,
    pub sync_deferred: u64,
    pub pipes_asked: u64,
    pub pipes_us: u64,
    pub pipes_made: u64,
    pub sync_batches: u64,
    pub sync_textures: u64,
    pub sync_first: u64,
    pub sync_unchanged: u64,
    pub sync_bytes: u64,
    pub sync_probe_us: u64,
    pub sync_upload_us: u64,
    pub sync_prepare_us: u64,
    pub sync_read_us: u64,
    pub sync_hash_us: u64,
    pub sync_detile_us: u64,
    pub scan_bytes: u64,
    pub scan_read_us: u64,
    pub scan_hash_us: u64,
}

fn take(c: &AtomicU64) -> u64 {
    c.swap(0, Ordering::Relaxed)
}

/// Takes the window: every counter zeroed for the next one.
pub fn take_window() -> Taken {
    Taken {
        submits: take(&C.submits),
        submit_us: take(&C.submit_us),
        order_wait_us: take(&C.order_wait_us),
        prepare_us: take(&C.prepare_us),
        apply_us: take(&C.apply_us),
        apply_wait_us: take(&C.apply_wait_us),
        encode_us: take(&C.encode_us),
        encode_wait_us: take(&C.encode_wait_us),
        presents: take(&C.presents),
        present_us: take(&C.present_us),
        lock_calls: take(&C.lock_calls),
        lock_wait_us: take(&C.lock_wait_us),
        lock_hold_us: take(&C.lock_hold_us),
        put_offs: take(&C.put_offs),
        sync_deferred: take(&C.sync_deferred),
        pipes_asked: take(&C.pipes_asked),
        pipes_us: take(&C.pipes_us),
        pipes_made: take(&C.pipes_made),
        sync_batches: take(&C.sync_batches),
        sync_textures: take(&C.sync_textures),
        sync_first: take(&C.sync_first),
        sync_unchanged: take(&C.sync_unchanged),
        sync_bytes: take(&C.sync_bytes),
        sync_probe_us: take(&C.sync_probe_us),
        sync_upload_us: take(&C.sync_upload_us),
        sync_prepare_us: take(&C.sync_prepare_us),
        sync_read_us: take(&C.sync_read_us),
        sync_hash_us: take(&C.sync_hash_us),
        sync_detile_us: take(&C.sync_detile_us),
        scan_bytes: take(&C.scan_bytes),
        scan_read_us: take(&C.scan_read_us),
        scan_hash_us: take(&C.scan_hash_us),
    }
}

/// Presents so far, and milliseconds since the one before: the frame line's `frame` and `gap_ms`.
///
/// The gap is the present's own wall time: the title's frame, from one `PresentX` to the next.
pub fn next_present() -> (u64, f64, f64) {
    static N: AtomicU64 = AtomicU64::new(0);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now_us = (since_start() * 1e6) as u64;
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    let last = LAST.swap(now_us, Ordering::Relaxed);
    let gap_ms = if last == 0 { 0.0 } else { now_us.saturating_sub(last) as f64 / 1000.0 };
    (n, now_us as f64 / 1e6, gap_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host call's phase is readable from another thread with the name, its age and the host
    /// thread's number: what the stall report prints when a frame has not come.
    #[test]
    fn the_phase_in_progress_is_readable() {
        // The phases are one global: hold the tests that write it one at a time.
        static ONE: Mutex<()> = Mutex::new(());
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        leave();
        assert_eq!(phase_now(), None);
        let before = phase_now();
        enter(Phase::Encode);
        let (name, age, tid) = phase_now().expect("a phase is in progress");
        assert_eq!(name, "encoding command lists (host lock)");
        assert!(age >= 0.0);
        assert!(tid >= 1 && before.is_none());
        std::thread::sleep(Duration::from_millis(20));
        let (_, later, _) = phase_now().unwrap();
        assert!(later > age, "the age grows while the call is in the phase");
        enter(Phase::Prepare);
        assert_eq!(phase_now().unwrap().0, "reading and detiling textures");
        leave();
        assert_eq!(phase_now(), None);
    }
}
