//! The one native host the title-facing objects call on macOS: `queue.rs`, `fence.rs`, `list.rs`,
//! `descriptor.rs` and `resource.rs` forward to the functions here (the same call sites that go to
//! `crate::host` on Windows).
//!
//! Threads. The title calls from many threads; the backend's objects and the `Rc`s inside
//! [`Title`] are not thread safe, so everything that touches them runs under one lock (`TITLE`),
//! always taken before a list's own lock, and objects holding such `Rc`s (lists) are dropped under
//! it. The GPU never waits on the lock: submissions are asynchronous, and a timer thread moves held
//! work and fence values along.

use super::memory::{PoolObserver, PoolTable};
use super::state::Graphics;
use super::title::{Guessed, Pipe, TargetFormats, Title, TitleList, Wait};
use crate::object::This;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use victory_backend::resources::{DrawCall, HeapKind};
use victory_metal::metal::Metal;

pub type MetalTitle = Title<Metal>;
pub type MetalList = TitleList<Metal>;

struct Shared(MetalTitle);
// SAFETY: every access to the title's `Rc`s and Objective-C objects is under `TITLE`.
unsafe impl Send for Shared {}

static TITLE: Mutex<Option<Shared>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);
static POOLS: OnceLock<Arc<PoolTable>> = OnceLock::new();
/// Host texture copy bytes of the live title, and the budget they are kept under (0: no budget),
/// published under the title lock by the host timer and read lock-free by the window's HUD.
static TEXTURE_LIVE: AtomicU64 = AtomicU64::new(0);
static TEXTURE_LIMIT: AtomicU64 = AtomicU64::new(0);

/// The table of the title's memory pools. Feed it through [`pool_observer`].
pub fn pools() -> &'static Arc<PoolTable> {
    POOLS.get_or_init(Default::default)
}

/// What `victory_hle`'s physical-memory observer forwards its four calls to
/// (`pool_created`, `pool_released`, `mapped`, `unmapped`): a plain handle on the pool table the
/// native host reads, safe to call from any title thread, before or after the GPU exists.
pub fn pool_observer() -> Arc<dyn PoolObserver> {
    pools().clone()
}

/// Called with the event handle (as the title gave it) when a fence a title waits on with
/// `SetEventOnCompletion` is reached. The runtime sets it; without one, events are not signalled.
static EVENT_HOOK: OnceLock<fn(usize)> = OnceLock::new();

pub fn set_event_hook(hook: fn(usize)) {
    let _ = EVENT_HOOK.set(hook);
}

/// Who in the title's code asked for the host call being served (image offsets), for a
/// diagnostic: `victory-gpu` has no loader, so the app hands it one
/// (`native_run::connect_gpu`). Empty when none is set.
static CALLER_HOOK: OnceLock<fn() -> Vec<usize>> = OnceLock::new();

pub fn set_caller_hook(hook: fn() -> Vec<usize>) {
    let _ = CALLER_HOOK.set(hook);
}

/// The calling title thread's chain, or empty with no hook: only for a diagnostic, never for
/// deciding anything, since it unwinds the title's stack.
pub(crate) fn title_caller() -> Vec<usize> {
    CALLER_HOOK.get().map_or_else(Vec::new, |f| f())
}

pub(crate) fn signal_event(event: usize) {
    if let Some(hook) = EVENT_HOOK.get() {
        hook(event);
    }
}

/// The host's lock, held. Its wait and its hold are the frame line's `lock_wait` and `lock_hold`
/// (`VICTORY_TRACE_FRAME`); both are one atomic add when the trace is off.
struct Hold {
    guard: std::sync::MutexGuard<'static, Option<Shared>>,
    since: std::time::Instant,
    /// Where it was taken (`VICTORY_TRACE_LOCK`).
    site: super::locktime::Site,
}

impl std::ops::Deref for Hold {
    type Target = Option<Shared>;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl std::ops::DerefMut for Hold {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        let held = self.since.elapsed();
        super::hosttime::lock_held(held);
        super::locktime::held(self.site, held);
    }
}

fn lock() -> Hold {
    let started = std::time::Instant::now();
    let guard = TITLE.lock().unwrap_or_else(|e| e.into_inner());
    super::hosttime::lock_waited(started.elapsed());
    Hold { guard, since: std::time::Instant::now(), site: super::locktime::site() }
}

/// Set once the host is on its way out: it takes no more GPU work, from the title's threads or
/// from the timer below. See [`stop_work`].
static STOPPED: AtomicBool = AtomicBool::new(false);

/// Stops the host taking new GPU work, and waits for the call already in it (if any): once it
/// returns, no thread the title or the host drives can commit another command buffer, so what
/// `victory_metal::gfx::stop_gpu` wakes and drains is all the GPU has. Without this a title thread
/// or the timer below submits again while the stop is draining and emptying the sets, and the
/// work it commits outlives the stop.
pub fn stop_work() {
    STOPPED.store(true, Ordering::SeqCst);
    // A call that read the flag as unset before that holds the lock; taking it now waits for that
    // call to finish. Calls from here on return before taking it (and those already waiting on it
    // find the flag set when they get it).
    drop(lock());
}

/// Whether the host has stopped taking work ([`stop_work`]).
fn stopped() -> bool {
    STOPPED.load(Ordering::Relaxed)
}

/// Runs `f` on the native host, made on first use; `None` if there is no Metal device (the call
/// is then ignored, as when there is no host GPU on Windows), or if the host is stopping
/// ([`stop_work`]).
pub(crate) fn with_title<R>(f: impl FnOnce(&mut MetalTitle) -> R) -> Option<R> {
    if stopped() {
        return None;
    }
    // A submission the GPU ended while the builder ran stops it (`VICTORY_BVH_BUILDER=1`).
    crate::native::builder::poll_lost();
    let mut guard = lock();
    // Set while this call waited for the lock: it must not submit after the stop began.
    if stopped() {
        return None;
    }
    if guard.is_none() {
        if FAILED.load(Ordering::Relaxed) {
            return None;
        }
        let console = crate::console::current();
        let (gpu_optimal, standard) = console.title_memory();
        match Metal::new_for_title((gpu_optimal + standard) as u64) {
            Ok(metal) => {
                let budget =
                    super::textures::texture_budget(std::env::var("VICTORY_METAL_TEXTURE_BUDGET_MB").ok().as_deref());
                let idle_frames =
                    super::textures::idle_frames(std::env::var("VICTORY_TEXTURE_IDLE_FRAMES").ok().as_deref());
                let suggested = super::textures::suggested_texture_budget(
                    metal.recommended_working_set(),
                    (gpu_optimal + standard) as u64,
                );
                tracing::info!(
                    console = console.name(),
                    title_memory_mib = (gpu_optimal + standard) / (1 << 20),
                    recommended_working_set_mib = metal.recommended_working_set() / (1 << 20),
                    texture_budget_mib = budget >> 20,
                    suggested_texture_budget_mib = suggested >> 20,
                    texture_idle_frames = idle_frames,
                    "native host: Metal"
                );
                let mut title = Title::new(metal, pools().clone());
                title.set_texture_budget(budget);
                title.set_texture_idle_frames(idle_frames);
                crate::ambient::announce();
                *guard = Some(Shared(title));
                start_timer();
            }
            Err(e) => {
                FAILED.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "no native host GPU");
                return None;
            }
        }
    }
    guard.as_mut().map(|t| f(&mut t.0))
}

/// Moves held work along and raises the title's copies of fence values as the GPU signals them
/// (titles read a fence's value from memory), without a call of the title's to drive it.
fn start_timer() {
    let _ = std::thread::Builder::new()
        .name("victory native host".into())
        .spawn(|| {
            let _site = super::locktime::at(super::locktime::Site::Timer);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(1));
                let mut guard = lock();
                // The host is on its way out: pumping here would commit work after the stop
                // drained the GPU (see `stop_work`).
                if stopped() {
                    return;
                }
                if let Some(t) = guard.as_mut() {
                    refresh(&mut t.0);
                }
            }
        });
}

fn refresh(t: &mut MetalTitle) {
    // The builder stops if the GPU ended a submission while it ran (`VICTORY_BVH_BUILDER=1`).
    crate::native::builder::poll_lost();
    t.pump();
    // The HUD's texture figures are refreshed here, under the lock this thread already holds: the
    // AppKit thread then reads them without ever waiting on the title.
    TEXTURE_LIVE.store(t.budget.live, Ordering::Relaxed);
    TEXTURE_LIMIT.store(t.budget.limit, Ordering::Relaxed);
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|l| l.elapsed().as_secs() >= 5) {
            let stuck = {
                let _timed = super::locktime::time(super::locktime::Part::Stuck);
                t.stuck(std::time::Duration::from_secs(2))
            };
            if !stuck.is_empty() {
                *last = Some(std::time::Instant::now());
                tracing::warn!("work held for a wait that does not hold: {stuck:?}");
                // What a held queue or a fence waits for may be owed by a command buffer the GPU
                // lost (out of memory): those are paid from the CPU so the title goes on.
                let lost = victory_metal::gfx::recover_lost();
                if lost > 0 {
                    tracing::warn!("recovered {lost} command buffers the GPU never reported on");
                }
                // Nothing is submitted while every queue is held, so the per-submission report
                // of command buffers the GPU has not finished would stay silent.
                if env_set!("VICTORY_TRACE_GPU_FAILS") {
                    let flying = victory_metal::gfx::report_in_flight();
                    let order: Vec<String> = t
                        .backend_queues()
                        .map(|(k, q)| (k, q.order_state()))
                        .filter(|(_, (done, made))| done != made)
                        .map(|(k, (done, made))| format!("queue {k:#x} at {done} of {made}"))
                        .collect();
                    tracing::warn!(
                        "command buffers in flight while work is held: {flying}; order events behind: {order:?}; commands dropped at unmapped memory so far: {}",
                        t.host.unmapped()
                    );
                }
            }
        }
    }
    for (fence, value) in t.completed() {
        crate::fence::raise(fence as This, value);
    }
}

/// Runs `f` on the native host and a title list's recording.
pub(crate) fn with_list<R>(
    this: This,
    f: impl FnOnce(&mut MetalTitle, &mut MetalList) -> R,
) -> Option<R> {
    if this.is_null() {
        return None;
    }
    let _site = super::locktime::at(super::locktime::Site::Record);
    with_title(|t| {
        let list = crate::list::native(this);
        let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
        f(t, &mut list)
    })
}

// --- queues and fences ---

/// A `with_title` in [`submit`]'s phases: its result, its wall time, and how much of that waited
/// for the host's lock (the frame line's `apply_wait_ms`/`encode_wait_ms`).
fn timed_body<T>(f: impl FnOnce(&mut MetalTitle) -> T) -> (Option<T>, std::time::Duration, std::time::Duration) {
    let started = std::time::Instant::now();
    let r = with_title(f);
    (r, started.elapsed(), super::hosttime::last_wait())
}

/// `ExecuteCommandLists`.
pub(crate) fn submit(queue: This, lists: &[This]) {
    if env_set!("VICTORY_TRACE_WRITES") {
        tracing::warn!("submit to queue {queue:p}: lists {lists:?}");
    }
    // One submission at a time, in the order they came: the lock below is let go while the
    // textures are read, and the next submission must not overtake this one meanwhile.
    static ORDER: Mutex<()> = Mutex::new(());
    let entry = std::time::Instant::now();
    let ordering = std::time::Instant::now();
    super::hosttime::enter(super::hosttime::Phase::Order);
    let _order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
    let order_wait = ordering.elapsed();
    super::hosttime::enter(super::hosttime::Phase::Prepare);
    // Tests share one title and count the uploads of their own `sync_dirty` calls, which another
    // test's submission with textures read but not yet uploaded would upset: there the lock is
    // kept from the probes to the uploads (`Title::submit`).
    let unlocked = !cfg!(test);
    let mut named = Default::default();
    let mut prepare = std::time::Duration::ZERO;
    let mut apply = std::time::Duration::ZERO;
    let mut apply_wait = std::time::Duration::ZERO;
    // The lists hold `Rc`s the recording threads share, so they are taken under the title lock --
    // once, moved out rather than copied (`TitleList::take_segments`): the segment copies were the
    // frame's `malloc` time on the render thread, under this lock.
    let mut segments = Vec::new();
    {
        let _site = super::locktime::at(super::locktime::Site::Take);
        let (taken, took, waited) = timed_body(|t| {
            let mut out = Vec::new();
            for &l in lists.iter().filter(|l| !l.is_null()) {
                let list = crate::list::native(l);
                let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
                out.extend(list.take_segments(t));
            }
            out
        });
        apply += took;
        apply_wait += waited;
        segments = taken.unwrap_or_default();
    }
    let mut jobs = Vec::new();
    if unlocked {
        let _site = super::locktime::at(super::locktime::Site::Textures);
        // What the frame has left for textures ([`textures::FRAME_TEXTURE_WORK`]): once spent,
        // this submission's textures are left for the next frame rather than holding this one
        // past the title's watchdog.
        let left = with_title(|t| t.texture_work_left(super::textures::FRAME_TEXTURE_WORK));
        if left.is_some_and(|left| left.is_zero()) {
            super::hosttime::sync_deferred();
            let (_, took, waited) = timed_body(|t| {
                if t.names_wanted() {
                    named = t.keep_named(segments.iter().map(|(l, _)| l));
                }
            });
            apply += took;
            apply_wait += waited;
        } else {
            super::hosttime::enter(super::hosttime::Phase::Apply);
            let (_, took, waited) = timed_body(|t| {
                if t.names_wanted() {
                    named = t.keep_named(segments.iter().map(|(l, _)| l));
                }
                jobs = t.sync_jobs();
            });
            apply += took;
            apply_wait += waited;
            with_title(|t| t.spent_texture_work(took));
            let mut batches = super::textures::batches(jobs).into_iter();
            let mut deferred = Vec::new();
            while let Some(batch) = batches.next() {
                if with_title(|t| t.texture_work_left(super::textures::FRAME_TEXTURE_WORK)).is_some_and(|left| left.is_zero()) {
                    deferred.extend(batch);
                    deferred.extend(batches.by_ref().flatten());
                    break;
                }
                // Without the lock: the title's other threads go on recording.
                super::hosttime::enter(super::hosttime::Phase::Prepare);
                let read = std::time::Instant::now();
                let prepared = super::textures::prepare(&batch);
                prepare += read.elapsed();
                with_title(|t| t.spent_texture_work(read.elapsed()));
                super::hosttime::enter(super::hosttime::Phase::Apply);
                let started = std::time::Instant::now();
                {
                    let _site = super::locktime::at(super::locktime::Site::Upload);
                    with_title(|t| t.sync_apply(prepared));
                }
                apply += started.elapsed();
                apply_wait += super::hosttime::last_wait();
                with_title(|t| t.spent_texture_work(started.elapsed()));
            }
            if !deferred.is_empty() {
                super::hosttime::sync_deferred();
                with_title(|t| t.defer_reads(deferred));
            }
        }
    }
    super::hosttime::enter(super::hosttime::Phase::Encode);
    let encoding = std::time::Instant::now();
    let site = super::locktime::at(super::locktime::Site::Encode);
    with_title(|t| {
        // What the GPU drew, cleared or wrote through views is no longer filled from title
        // memory, which the title changes under its render targets (noise uploaded over them).
        let drew = std::mem::take(&mut *DREW.lock().unwrap_or_else(|e| e.into_inner()));
        for key in drew.into_iter().chain(t.host.take_gpu_written()) {
            t.gpu_written(key);
        }
        if t.ensure_queue(queue as usize).is_err() {
            t.let_named_go(named);
            return;
        }
        if unlocked {
            t.submit_synced(queue as usize, segments);
        } else {
            t.submit(queue as usize, segments);
        }
        t.let_named_go(named);
        // `VICTORY_TRACE_DESCRIPTORS=sites`: the heaps' pages are closed again for the next
        // interval, so the title's first store into each is seen.
        if super::descriptor_trace::sites() {
            victory_platform::watch::watch(&t.host.heap_ranges());
        }
    });
    drop(site);
    super::hosttime::submitted(super::hosttime::Submit {
        order_wait,
        prepare,
        apply,
        apply_wait,
        encode: encoding.elapsed(),
        encode_wait: super::hosttime::last_wait(),
        wall: entry.elapsed(),
    });
    super::hosttime::leave();
}

pub(crate) fn queue_signal(queue: This, fence: This, value: u64) {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.signal_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_wait(queue: This, fence: This, value: u64) {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.wait_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_released(queue: This) {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    with_title(|t| t.forget_queue(queue as usize));
}

/// `ID3D12Fence::Signal`.
pub(crate) fn fence_signal(fence: This, value: u64) {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    let initial = crate::fence::initial(fence);
    with_title(|t| t.signal_cpu(fence as usize, initial, value));
}

/// The value fence `fence` has completed, queues pumped first.
pub(crate) fn fence_poll(fence: This) -> u64 {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        t.pump();
        t.fence_value(fence as usize, initial)
    })
    .unwrap_or(initial)
}

/// `GetCompletedValue` of a fence that has no GPU behind it yet: nothing to ask.
pub(crate) fn fence_released(fence: This) {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    with_title(|t| t.forget_fence(fence as usize));
}

/// Blocks until fence `fence` reaches `value` (the title's `SetEventOnCompletion` with no event):
/// held work keeps moving while it waits. False after `timeout`.
pub(crate) fn fence_wait(fence: This, value: u64, timeout: std::time::Duration) -> bool {
    let _site = super::locktime::at(super::locktime::Site::Queue);
    let initial = crate::fence::initial(fence);
    let start = std::time::Instant::now();
    loop {
        match with_title(|t| t.try_wait(fence as usize, initial, value)) {
            Some(true) => return true,
            None => return false,
            Some(false) => {}
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

// --- present ---

/// Gives the host the app's drawable layer (a `CAMetalLayer`, by address; it must outlive the
/// host): frames the title presents are drawn into it from then on. Without one, `PresentX` shows
/// nothing.
pub fn set_layer(layer: *mut std::ffi::c_void) -> Result<(), String> {
    with_title(|t| t.set_surface(layer)).unwrap_or_else(|| Err("no native host GPU".into()))
}

/// A layer drawing to the host's device, `width` by `height`, for an app that has none (or a test):
/// hand [`victory_metal::present::Layer::as_ptr`] to [`set_layer`] and keep the layer alive.
pub fn create_layer(width: usize, height: usize) -> Result<victory_metal::present::Layer, String> {
    with_title(|t| t.host.backend.create_layer(width, height)).ok_or_else(|| "no native host GPU".into())
}

/// `VICTORY_TRACE_TARGETS=1`: per frame, what drew into or cleared which title texture, and what
/// was presented (logged at each present).
pub(crate) fn tracing_targets() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_set!("VICTORY_TRACE_TARGETS"))
}

/// Draws and clears by title texture since the last present, and dropped texture operations.
static TRACE: Mutex<Option<std::collections::HashMap<usize, [u32; 3]>>> = Mutex::new(None);
/// Draws dropped since the last present, by reason (with `VICTORY_TRACE_TARGETS`).
static DROP_REASONS: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
/// Draws dropped since the last present for want of a host pipeline, by shaders and targets.
static DROP_PIPES: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());
static DROPPED_OPS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Textures a recorded draw or clear writes, made the GPU's ([`Title::gpu_written`]) at the next
/// submission.
static DREW: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());

/// Counts texture `key` for the frame trace: `which` 0 drawn or copied into, 1 cleared, 2 a draw
/// into it dropped. Drawn and cleared, it is also the GPU's from the next submission on ([`DREW`]).
fn trace_bump(key: usize, which: usize) {
    if which < 2 {
        DREW.lock().unwrap_or_else(|e| e.into_inner()).insert(key);
    }
    if tracing_targets() {
        TRACE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().entry(key).or_default()[which] += 1;
    }
}

/// `VICTORY_DUMP_TITLE=<dir>,<seconds>,<seconds>...`: at the first present after each time, the
/// title memory behind every 8-bit colour target drawn to since the last present is read whole,
/// the fraction of non-zero bytes logged, and the memory detiled into a scratch texture and
/// written as `<dir>/m<time>_<n>.ppm`: what the title (CPU, DMA) put there, whatever the GPU did.
#[inline(never)]
fn dump_title_memory(stats: &std::collections::HashMap<usize, [u32; 3]>) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static DONE: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TITLE") else { return };
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let now = start.elapsed().as_secs();
    let Some(time) = parts
        .filter_map(|p| p.parse::<u64>().ok())
        .find(|&t| now >= t && !DONE.lock().unwrap_or_else(|e| e.into_inner()).contains(&t))
    else {
        return;
    };
    DONE.lock().unwrap_or_else(|e| e.into_inner()).push(time);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let tiled = super::textures::read_memory(info.memory, info.size);
            let nz = tiled.iter().filter(|&&b| b != 0).count();
            tracing::info!(time, n, key = format!("{key:#x}"), at = format!("{:#x}", info.memory as usize), size = info.size, nonzero = nz, fraction = nz as f64 / tiled.len().max(1) as f64, "title memory of a colour target");
            let Some(desc) = info.desc() else { continue };
            let Ok(tex) = t.host.texture(&desc) else { continue };
            if super::textures::upload(&t.host.backend, &tex, &info, &tiled).is_err() {
                continue;
            }
            let (w, h) = (info.width as usize, info.height as usize);
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("m{time}_{n}.ppm")), out);
        }
    });
}

/// `VICTORY_DUMP_TARGETS=<dir>,<seconds>`: once, at the first present after that many seconds,
/// every 8-bit colour target drawn to since the last present is written as `<dir>/t<n>.ppm`; HDR
/// and 10-bit targets (DXGI 2, 10, 24, 26) are tone-mapped into `..._f<format>.ppm`, their range logged.
fn dump_targets(stats: &std::collections::HashMap<usize, [u32; 3]>) {
    use std::sync::atomic::AtomicBool;
    static DONE: AtomicBool = AtomicBool::new(false);
    static LAST: Mutex<Option<u64>> = Mutex::new(None);
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TARGETS") else { return };
    // `<dir>,<seconds>[,<every>]`: from that second on, once or every that many seconds.
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let secs = parts.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(25);
    let every = parts.next().and_then(|s| s.parse::<u64>().ok());
    let now = start.elapsed().as_secs();
    let due = match (every, *LAST.lock().unwrap_or_else(|e| e.into_inner())) {
        (Some(e), Some(last)) => now >= last + e,
        _ => true,
    };
    if now < secs || !due || (every.is_none() && DONE.swap(true, Ordering::Relaxed)) {
        return;
    }
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(now);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            let (w, h) = (info.width as usize, info.height as usize);
            if matches!(info.format, 2 | 10 | 24 | 26) {
                // HDR and 10-bit targets: tone-mapped to 8 bits, with their range logged.
                let Some(tex) = t.texture_of(*key) else { continue };
                let bpp = match info.format { 2 => 16, 10 => 8, _ => 4 };
                let mut raw = vec![0u8; w * h * bpp];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                let rgb: Vec<[f32; 3]> = raw.chunks_exact(bpp).map(|p| decode_hdr(info.format, p)).collect();
                let finite = rgb.iter().flatten().filter(|v| v.is_finite());
                let (mut max, mut sum, mut count, mut lit) = (0f32, 0f64, 0usize, 0usize);
                for v in finite {
                    max = max.max(*v);
                    sum += f64::from(*v);
                    count += 1;
                    lit += usize::from(*v > 1e-4);
                }
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend(rgb.iter().flat_map(|c| c.map(|v| if v.is_finite() && v > 0.0 { ((v / (1.0 + v)).sqrt() * 255.0) as u8 } else { 0 })));
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_f{}.ppm", info.memory as usize, info.format)), out);
                tracing::info!(n, key = format!("{key:#x}"), w, h, format = info.format, max, mean = sum / count.max(1) as f64, lit = lit as f64 / (rgb.len() * 3).max(1) as f64, "target dumped");
                continue;
            }
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let Some(tex) = t.texture_of(*key) else { continue };
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}.ppm", info.memory as usize)), out);
            // The alpha channel as grey.
            let mut alpha = format!("P6\n{w} {h}\n255\n").into_bytes();
            alpha.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[3], p[3], p[3]]));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_alpha.ppm", info.memory as usize)), alpha);
            tracing::info!(n, key = format!("{key:#x}"), w, h, "target dumped");
        }
    });
}

/// `VICTORY_DUMP_DEPTH=<dir>,<seconds>[,<every>]`: once, at the first present after that many
/// seconds, every live depth target is read back, its per-slice non-zero fraction logged and the
/// slices written as one `.pgm` (8-bit, the top byte of each texel), with its colour twin's when it
/// has one: proof that a shadow map holds the geometry its pass drew, and what the shaders sample.
#[inline(never)]
fn dump_depth_targets() {
    use std::sync::atomic::AtomicBool;
    static DONE: AtomicBool = AtomicBool::new(false);
    static LAST: Mutex<Option<u64>> = Mutex::new(None);
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let Ok(v) = std::env::var("VICTORY_DUMP_DEPTH") else { return };
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let secs = parts.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(25);
    let every = parts.next().and_then(|s| s.parse::<u64>().ok());
    let start = START.get_or_init(std::time::Instant::now);
    let now = start.elapsed().as_secs();
    let due = match (every, *LAST.lock().unwrap_or_else(|e| e.into_inner())) {
        (Some(e), Some(last)) => now >= last + e,
        _ => true,
    };
    if now < secs || !due || (every.is_none() && DONE.swap(true, Ordering::Relaxed)) {
        return;
    }
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(now);
    let dir = std::path::PathBuf::from(dir);
    with_title(|t| {
        // The reads are the CPU's: let the queues reach the values the title last signalled, so
        // the shadow pass recorded before this present has run.
        for (fence, value) in t.queue_fences.values() {
            if !victory_backend::backend::ComputeBackend::wait(&t.host.backend, fence, *value, 5000) {
                tracing::warn!(value, "a queue did not finish before the depth dump");
            }
        }
        let mut targets = t.depth_targets();
        targets.sort_by_key(|(_, i, _)| std::cmp::Reverse((i.width as u64) * i.height as u64 * i.depth_or_layers as u64));
        let mut written = 0usize;
        for (key, info, tex) in targets {
            let Some(desc) = info.desc() else { continue };
            let (texel, depth16) = match desc.format {
                victory_backend::resources::Format::D16Unorm => (2usize, true),
                victory_backend::resources::Format::D32Float => (4usize, false),
                other => {
                    tracing::info!(key = format!("{key:#x}"), format = info.format, ?other, "depth target not dumped (format)");
                    continue;
                }
            };
            let (w, h) = (desc.width, desc.height);
            let layers = desc.depth_or_layers.max(1);
            let mut rows: Vec<u8> = Vec::new();
            let mut slices_log = Vec::new();
            let mut hist = std::collections::HashMap::<u16, usize>::new();
            let mut any = false;
            for slice in 0..layers {
                let mut raw = vec![0u8; w * h * texel];
                victory_backend::resources::ResourceBackend::read_texture(
                    &t.host.backend,
                    &tex,
                    0,
                    slice,
                    &mut raw,
                    w * texel,
                    w * h * texel,
                );
                let nonzero = raw.chunks_exact(texel).filter(|p| p.iter().any(|&b| b != 0)).count();
                any |= nonzero > 0;
                slices_log.push(format!("{nonzero}/{}", w * h));
                if depth16 {
                    for p in raw.chunks_exact(2) {
                        *hist.entry(u16::from_le_bytes([p[0], p[1]])).or_default() += 1;
                    }
                }
                if written < 12 {
                    for p in raw.chunks_exact(texel) {
                        rows.push(if depth16 {
                            (u16::from_le_bytes([p[0], p[1]]) >> 8) as u8
                        } else {
                            let v = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                            (v.clamp(0.0, 1.0) * 255.0) as u8
                        });
                    }
                }
            }
            // The most common texel is the clear (the far plane); the rest is what a pass drew,
            // which separates a solid far plane from a map with geometry.
            let (background, background_frac, distinct) = if depth16 {
                let (value, count) = hist
                    .iter()
                    .max_by_key(|&(_, &n)| n)
                    .map(|(&v, &n)| (v, n))
                    .unwrap_or((0, 1));
                (
                    format!("0x{value:04x}"),
                    count as f64 / hist.values().sum::<usize>().max(1) as f64,
                    hist.len(),
                )
            } else {
                ("-".to_owned(), 0.0, 0)
            };
            if any && written < 12 && !rows.is_empty() {
                let mut out = format!("P5\n{w} {}\n255\n", h * layers).into_bytes();
                out.extend_from_slice(&rows);
                let name = format!("depth_{key:x}_{w}x{h}x{layers}_f{}.pgm", info.format);
                let _ = std::fs::write(dir.join(&name), out);
                written += 1;
            }
            let twin = t.host.twin_report(&tex).map(|(twin, twin_texel, newer)| {
                let mut nonzero = 0usize;
                let mut total = 0usize;
                for slice in 0..layers {
                    let mut raw = vec![0u8; w * h * twin_texel];
                    victory_backend::resources::ResourceBackend::read_texture(
                        &t.host.backend,
                        &twin,
                        0,
                        slice,
                        &mut raw,
                        w * twin_texel,
                        w * h * twin_texel,
                    );
                    total += w * h;
                    nonzero += raw.chunks_exact(twin_texel).filter(|p| p.iter().any(|&b| b != 0)).count();
                }
                format!("twin texel {twin_texel} newer {newer}, {nonzero}/{total} nonzero")
            });
            // A placed view over this target that still has a copy of its own would sample zeros.
            let stale = t.placed_copies_over(&info);
            tracing::info!(
                key = format!("{key:#x}"),
                at = format!("{:#x}", info.memory as usize),
                format = info.format,
                w,
                h,
                layers,
                mips = info.layout.mips.len(),
                flags = info.flags,
                slices = ?slices_log,
                background,
                background_frac,
                distinct,
                twin = twin.as_deref().unwrap_or(""),
                placed_copies = ?stale,
                "depth target read back"
            );
        }
    });
}

/// One texel of an HDR or 10-bit colour target (DXGI 2, 10, 24, 26) as linear RGB.
fn decode_hdr(format: u32, p: &[u8]) -> [f32; 3] {
    let word = |i: usize| u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
    // An unsigned small float: `mbits` mantissa bits over a 5-bit exponent.
    let small = |v: u32, mbits: u32| {
        let (e, m) = ((v >> mbits) & 0x1f, v & ((1 << mbits) - 1));
        if e == 0 { m as f32 * 2f32.powi(-14 - mbits as i32) } else { (1.0 + m as f32 / (1 << mbits) as f32) * 2f32.powi(e as i32 - 15) }
    };
    match format {
        2 => [0, 4, 8].map(|i| f32::from_bits(word(i))),
        10 => [0, 2, 4].map(|i| {
            let h = u32::from(u16::from_le_bytes([p[i], p[i + 1]]));
            let v = small(h & 0x7fff, 10);
            if h >> 15 == 1 { -v } else if (h >> 10) & 0x1f == 0x1f { f32::NAN } else { v }
        }),
        24 => [0, 10, 20].map(|s| ((word(0) >> s) & 0x3ff) as f32 / 1023.0),
        _ => {
            let v = word(0);
            [small(v & 0x7ff, 6), small((v >> 11) & 0x7ff, 6), small(v >> 22, 5)]
        }
    }
}

/// A texture operation the native host does not do (copies, resolves), counted by the trace.
pub(crate) fn dropped_op(what: &str, dst: This, src: This) {
    crate::list::DROPPED_KINDS[3].fetch_add(1, Ordering::Relaxed);
    warn_once("a texture copy or resolve");
    // Once per operation and title formats (0: a buffer or not a texture), so a refused kind of
    // copy shows what it was.
    static SEEN: Mutex<Vec<(String, u32, u32)>> = Mutex::new(Vec::new());
    let format = |r: This| crate::resource::texture_info(r).map_or(0, |i| i.format);
    let key = (what.to_owned(), format(src), format(dst));
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&key) {
        tracing::warn!(what, src_format = key.1, dst_format = key.2, "a texture copy refused (first of this kind)");
        seen.push(key);
    }
    drop(seen);
    if tracing_targets() {
        DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()).push(format!("{what} {src:?}->{dst:?}"));
    }
}

/// `VICTORY_TRACE_RECORD_WRITERS`: one compute program's dispatches since the last present, run
/// or dropped: the buffers its tables hold for unordered access (title memory start and size),
/// and the textures they hold as shader resources and for unordered access.
#[derive(Default)]
struct RecordDispatches {
    buffers: std::collections::BTreeSet<(u64, u64)>,
    reads: std::collections::BTreeSet<usize>,
    writes: std::collections::BTreeSet<usize>,
}

/// `VICTORY_TRACE_RECORD_WRITERS`: the compute dispatches since the last present by program and
/// outcome, and each indirect mesh execution's first record address with its records and how
/// many had a group count of 0 in title memory as the execution was recorded.
#[derive(Default)]
struct RecordTrace {
    dispatches: std::collections::BTreeMap<String, RecordDispatches>,
    executions: std::collections::BTreeMap<u64, [u64; 2]>,
}

static RECORD_TRACE: Mutex<Option<RecordTrace>> = Mutex::new(None);

fn tracing_record_writers() -> bool {
    env_set!("VICTORY_TRACE_RECORD_WRITERS")
}

/// `VICTORY_TRACE_RECORD_WRITERS=<hash>[,<hash>...]`: the records watched are those of the
/// indirect dispatches of these compute shaders, not the mesh records (`None`: the mesh records).
fn record_writers_of() -> Option<&'static str> {
    static OF: OnceLock<Option<String>> = OnceLock::new();
    OF.get_or_init(|| std::env::var("VICTORY_TRACE_RECORD_WRITERS").ok().filter(|v| v.len() >= 16)).as_deref()
}

/// The records and zero records of the executions whose first record is inside one of `buffers`.
fn records_in(buffers: &std::collections::BTreeSet<(u64, u64)>, executions: &std::collections::BTreeMap<u64, [u64; 2]>) -> [u64; 2] {
    executions
        .iter()
        .filter(|(at, _)| buffers.iter().any(|&(start, size)| (start..start.saturating_add(size)).contains(at)))
        .fold([0; 2], |a, (_, n)| [a[0] + n[0], a[1] + n[1]])
}

#[cfg(test)]
#[test]
fn records_are_counted_for_the_buffers_that_hold_their_first_record() {
    let executions = [(0x1000, [4, 4]), (0x1ff4, [2, 1]), (0x2000, [8, 0])].into_iter().collect();
    assert_eq!(records_in(&[(0x1000, 0x1000)].into_iter().collect(), &executions), [6, 5]);
    assert_eq!(records_in(&[(0x1000, 0x1000), (0x1800, 0x1000)].into_iter().collect(), &executions), [14, 5]);
    assert_eq!(records_in(&[(0x3000, 0x10), (u64::MAX, 2)].into_iter().collect(), &executions), [0, 0]);
}

fn trace_record_dispatch(t: &MetalTitle, l: &MetalList, outcome: &str) {
    if !tracing_record_writers() {
        return;
    }
    let (buffers, reads, writes) = t.host.table_views(l.list.compute_args());
    let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
    let d = trace.get_or_insert_default().dispatches.entry(format!("{:?} {outcome}", stage_hashes(l.pipeline))).or_default();
    d.buffers.extend(buffers);
    d.reads.extend(reads);
    d.writes.extend(writes);
}

/// `VICTORY_TRACE_RECORD_WRITERS`, at each present: the frame's indirect mesh records and how
/// many were 0 as recorded (the GPU may still write them: a culling pass that ran fills them
/// later, one that was dropped never does); each compute program whose tables hold a record
/// buffer for unordered access, with the records in those buffers and the textures it reads;
/// and for each such texture, the compute programs that hold it for unordered access.
#[inline(never)]
fn trace_record_writers() {
    let Some(trace) = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let total = trace.executions.values().fold([0u64; 2], |a, n| [a[0] + n[0], a[1] + n[1]]);
    if total[0] == 0 {
        return;
    }
    let texture = |r: usize| {
        let i = crate::resource::texture_info(r as This);
        let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
        format!("{r:#x} fmt {f} {w}x{h}")
    };
    tracing::info!(records = total[0], zero = total[1], "mesh records");
    let mut read = std::collections::BTreeSet::new();
    for (name, d) in &trace.dispatches {
        let n = records_in(&d.buffers, &trace.executions);
        if n[0] == 0 {
            continue;
        }
        read.extend(&d.reads);
        let reads: Vec<String> = d.reads.iter().map(|&r| texture(r)).collect();
        tracing::info!(records = n[0], zero = n[1], holder = name, ?reads, "mesh record buffer holder");
    }
    for r in read {
        let writers: Vec<&String> = trace.dispatches.iter().filter(|(_, d)| d.writes.contains(&r)).map(|(n, _)| n).collect();
        tracing::info!(texture = texture(r), ?writers, "texture read by a mesh record buffer holder");
    }
}

#[inline(never)]
fn trace_present(resource: This) {
        let stats = TRACE.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        let ops = std::mem::take(&mut *DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()));
        let reasons = std::mem::take(&mut *DROP_REASONS.lock().unwrap_or_else(|e| e.into_inner()));
        let mut pipes = std::mem::take(&mut *DROP_PIPES.lock().unwrap_or_else(|e| e.into_inner()));
        pipes.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (what, n) in &pipes {
            tracing::info!(n, what, "draws dropped (no host pipeline, or an indirect execution)");
        }
        let described: Vec<String> = with_title(|t| {
            stats
                .iter()
                .map(|(k, [d, c, x])| {
                    let i = crate::resource::texture_info(*k as This);
                    let (f, w, h, a) = i.map_or((0, 0, 0, 0), |i| (i.format, i.width, i.height, i.memory as usize));
                    let mark = if *k == resource as usize { " PRESENTED" } else { "" };
                    let _ = &t;
                    format!("{k:#x} fmt {f} {w}x{h} at {a:#x} draws {d} clears {c} dropped {x}{mark}")
                })
                .collect()
        })
        .unwrap_or_default();
        dump_targets(&stats);
        dump_title_memory(&stats);
        let pi = crate::resource::texture_info(resource);
        tracing::info!(
            present = ?resource,
            at = pi.as_ref().map_or(0, |i| i.memory as usize),
            fmt = pi.as_ref().map_or(0, |i| i.format),
            targets = ?described,
            dropped_ops = ?ops.iter().take(8).collect::<Vec<_>>(),
            dropped = ops.len(),
            copies = COPIES[1].swap(0, Ordering::Relaxed),
            ?reasons,
            "frame targets"
        );
}

/// `VICTORY_TRACE_PRESENT=<every>`: the title resources `PresentX` has shown.
static PRESENTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn present_every() -> Option<u32> {
    static EVERY: OnceLock<Option<u32>> = OnceLock::new();
    *EVERY.get_or_init(|| std::env::var("VICTORY_TRACE_PRESENT").ok().and_then(|v| v.parse().ok()))
}

/// `VICTORY_TRACE_PRESENT=<every>`: a draw into a texture the title has presented (its present
/// pass), the first and then every `<every>`th, at most 12: its pipeline's shaders, root
/// parameters and binding, which textures its tables hold and how much of each is non-zero, and
/// the first bytes of its constant buffers, all as the draw is recorded.
fn trace_present_draw(t: &MetalTitle, l: &MetalList) {
    use std::sync::atomic::AtomicU32;
    static N: AtomicU32 = AtomicU32::new(0);
    let Some(every) = present_every().map(|e| e.max(1)) else { return };
    {
        let shown = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
        if !l.target_keys.iter().any(|k| shown.contains(k)) {
            return;
        }
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n % every != 0 || n / every >= 12 {
        return;
    }
    let shaders = stage_hashes(l.pipeline);
    let targets: Vec<String> = l.target_keys.iter().map(|k| format!("{k:#x}")).collect();
    tracing::info!(n, pipeline = format!("{:#x}", l.pipeline), ?shaders, ?targets, "present draw");
    let (root, binding) = l.list.graphics_args();
    if let Some(b) = binding {
        // Filter, address u v w, mip LOD bias, anisotropy, comparison, border, min and max LOD.
        let ps = b.ps.as_ref().map(|p| &p.static_samplers);
        tracing::info!(vs = ?b.static_samplers, ?ps, "present draw static samplers");
    }
    t.host.trace_args(root, binding);
}

/// `VICTORY_TRACE_FRAME`: one line for what the frame since the last present cost, the host's
/// calls and the GPU's command buffers joined. The host's figures are the calls the title made
/// (submissions, their phases, the host's lock) and the texture and pipeline work they did; the
/// GPU's are `victory_metal::frametime`'s command buffers that completed in the window, and with
/// `VICTORY_TRACE_FRAME=passes` the ten passes that held it longest.
#[inline(never)]
fn frame_line() {
    if !super::hosttime::enabled() {
        return;
    }
    let h = super::hosttime::take_window();
    let w = victory_metal::frametime::window();
    let (frame, t, gap_ms) = super::hosttime::next_present();
    let ms = |us: u64| us as f64 / 1000.0;
    let mb = |b: u64| if b >= 1 << 30 { format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64) } else { format!("{} MiB", b >> 20) };
    let queues = w
        .queues
        .iter()
        .map(|(q, n, ms)| format!("q{q:x} {n}x {ms:.0}ms"))
        .collect::<Vec<_>>()
        .join(", ");
    // Where the memory is when a frame stalls: the process footprint and what the system keeps
    // resident (the gap is what it compressed or swapped out).
    let footprint_mib = victory_platform::mem::footprint() >> 20;
    #[cfg(target_os = "macos")]
    let resident_mib = victory_platform::mem::resident() >> 20;
    #[cfg(not(target_os = "macos"))]
    let resident_mib = 0;
    tracing::info!(
        frame,
        t,
        gap_ms,
        presents = h.presents,
        submits = h.submits,
        submit_ms = ms(h.submit_us),
        order_wait_ms = ms(h.order_wait_us),
        prepare_ms = ms(h.prepare_us),
        apply_ms = ms(h.apply_us),
        apply_wait_ms = ms(h.apply_wait_us),
        encode_ms = ms(h.encode_us),
        encode_wait_ms = ms(h.encode_wait_us),
        present_ms = ms(h.present_us),
        lock_calls = h.lock_calls,
        lock_wait_ms = ms(h.lock_wait_us),
        lock_hold_ms = ms(h.lock_hold_us),
        gpu_cb = w.buffers,
        gpu_ms = w.gpu_ms,
        cb_encode_ms = w.encode_ms,
        in_flight = w.in_flight,
        put_off = h.put_offs,
        pipes = h.pipes_asked,
        pipes_made = h.pipes_made,
        pipes_ms = ms(h.pipes_us),
        tex_batches = h.sync_batches,
        tex_deferred = h.sync_deferred,
        tex = h.sync_textures,
        tex_first = h.sync_first,
        tex_same = h.sync_unchanged,
        tex_mb = h.sync_bytes >> 20,
        tex_probe_ms = ms(h.sync_probe_us),
        tex_prepare_ms = ms(h.sync_prepare_us),
        tex_upload_ms = ms(h.sync_upload_us),
        tex_read_ms = ms(h.sync_read_us),
        tex_hash_ms = ms(h.sync_hash_us),
        tex_detile_ms = ms(h.sync_detile_us),
        footprint_mib,
        resident_mib,
        queues = %queues,
        "frame time"
    );
    // `VICTORY_TRACE_LOCK=1`: `lock_hold_ms` by where the lock was taken and by what ran under it.
    let lock = super::locktime::take_line();
    if !lock.is_empty() {
        tracing::info!("lock time frame={frame} {lock}");
    }
    if !w.passes.is_empty() {
        tracing::info!(frame, "frame passes by GPU time: {}", victory_metal::frametime::top_passes(&w, 10));
    }
    let why = super::hosttime::take_why();
    if !why.is_empty() {
        let why = why
            .iter()
            .map(|(r, n, b)| format!("{r} {n} ({})", mb(*b)))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(frame, "frame texture sync reasons: {why}");
    }
    if !w.buffers_top.is_empty() {
        let buffers = w
            .buffers_top
            .iter()
            .map(|(ms, serial, label)| format!("#{serial} {ms:.0}ms {label}"))
            .collect::<Vec<_>>()
            .join(" || ");
        tracing::info!(frame, "frame command buffers: {buffers}");
    }
}

/// `PresentX` on title queue `queue`: title textures `planes` over the layer, the first under the
/// others (`Title::present`). The traces name the first.
pub(crate) fn present(queue: This, planes: &[This]) {
    let _site = super::locktime::at(super::locktime::Site::Present);
    let Some(&resource) = planes.first() else { return };
    let present_started = std::time::Instant::now();
    if present_every().is_some() {
        let mut shown = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
        for &p in planes {
            if !shown.contains(&(p as usize)) {
                shown.push(p as usize);
            }
        }
    }
    // Diagnostic: VICTORY_PRESENT_DELAY_MS slows every present, to tell a trigger that follows
    // frames from one that follows time.
    static DELAY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *DELAY.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_DELAY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    });
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    if super::title_watch::enabled() {
        if let Some(memory) = with_title(|t| t.host.memory()) {
            super::title_watch::poll(&memory);
        }
    }
    if tracing_record_writers() {
        trace_record_writers();
    }
    #[cfg(target_os = "macos")]
    if super::gpu_vertex_trace::on() {
        super::gpu_vertex_trace::present();
    }
    super::rt::present();
    dump_depth_targets();
    if tracing_targets() {
        trace_present(resource);
    }
    pass_present(resource);
    let shown = with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        let keys: Vec<usize> = planes.iter().map(|&p| p as usize).collect();
        t.present(queue as usize, &keys)
    });
    match shown {
        Some(Err(e)) => tracing::warn!(error = %e, "a frame was not presented"),
        Some(Ok(false)) => {
            NO_TEXTURE.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
    super::hosttime::presented(present_started.elapsed());
    frame_line();
}

/// The last presented frame's size, read without waiting for the GPU (see
/// [`victory_metal::present::Surface::last_size`]): [`last_frame`] reads its pixels back, which
/// blocks until the present's command buffer completes.
pub fn last_frame_size() -> Option<(usize, usize)> {
    with_title(|t| t.surface().and_then(|s| s.last_size())).flatten()
}

/// Frames presented to the layer so far.
pub fn frames_presented() -> u64 {
    with_title(|t| t.surface().map_or(0, |s| s.frames())).unwrap_or(0)
}

/// Frame pacing for the window's HUD: frames presented, and the recent gap between presents.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct FramePacing {
    /// Frames presented so far.
    pub frames: u64,
    /// Microseconds since the last present (0: none yet).
    pub since_last_micros: u64,
    /// The recent frame gaps in microseconds, oldest first.
    pub recent: Vec<u64>,
}

/// The lock-free frame pacing counters: what the HUD's FPS, frame time and sparkline show. Takes no
/// lock, so it answers while the render thread is stalled in a submission.
pub fn frame_pacing() -> FramePacing {
    let p = victory_metal::pace::snapshot();
    FramePacing {
        frames: p.frames,
        since_last_micros: if p.frames == 0 {
            0
        } else {
            p.now_micros.saturating_sub(p.last_micros)
        },
        recent: p.recent,
    }
}

/// Draws in the passes submitted to the GPU so far (lock-free, for the HUD).
pub fn draws() -> u64 {
    super::host::draws()
}

/// GPU nanoseconds of completed command buffers so far (lock-free, for the HUD).
pub fn gpu_busy_nanos() -> u64 {
    victory_metal::gfx::gpu_busy_nanos()
}

/// Bytes of host texture copies kept by the live title, and the budget they are kept under (0: no
/// budget). Lock-free: the host's timer publishes them ([`refresh`]).
pub fn texture_budget_bytes() -> (u64, u64) {
    (
        TEXTURE_LIVE.load(Ordering::Relaxed),
        TEXTURE_LIMIT.load(Ordering::Relaxed),
    )
}

/// The pixels (BGRA) of the last frame presented, once the GPU has drawn it; for tests.
pub fn last_frame() -> Option<(usize, usize, Vec<u8>)> {
    with_title(|t| t.surface().and_then(|s| s.read_last())).flatten()
}

// --- resources and descriptors ---

/// A buffer resource object `res` of `len` bytes at title address `addr`.
/// `own`: its memory is the host's (committed, or in a heap), not title memory a pool holds.
pub(crate) fn buffer_created(res: This, addr: u64, len: u64, own: bool) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    with_title(|t| match own {
        true => t.host.register_own_buffer(res as usize, addr, len),
        false => t.host.register_buffer(res as usize, addr, len),
    });
}

/// A pipeline object's last reference went: the host pipelines made for it are forgotten, as the
/// title makes thousands of variants (device slot 0x1b8) a minute and a new one often gets a freed
/// one's address (which drew text with the old variant's blend state: glyphs as boxes).
pub(crate) fn pipeline_released(pipeline: This) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    let key = pipeline as usize;
    with_title(|t| {
        t.pipelines.retain(|(k, _), _| *k != key);
        t.console_pipes.retain(|(k, _, _), _| *k != key);
        t.pipeline_streams.remove(&key);
        t.thread_pipes.remove(&key);
    });
}

pub(crate) fn resource_released(res: This) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    with_title(|t| {
        t.host.release_resource(res as usize);
        t.texture_released(res as usize);
    });
}

/// A title texture object: its host texture is made, and filled from title memory when work is
/// next submitted.
pub(crate) fn texture_created(res: This) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    if let Some(info) = crate::resource::texture_info(res) {
        with_title(|t| t.texture_created(res as usize, info));
    }
}

/// `OMSetRenderTargets`: the host textures of the views' resources, in the views' formats.
pub(crate) fn targets(this: This, color: &[u64], depth: u64) {
    with_list(this, |t, l| {
        let views: Vec<_> = color.iter().map(|&h| crate::descriptor::rtv_record(h)).collect();
        let depth_record = crate::descriptor::dsv_record(depth);
        let depth_key = depth_record.map(|r| r.resource as usize);
        // A depth view may name one array slice or mip: the attachment is a one-layer view of
        // the target, not the whole array (Metal's depth attachment renders into one layer).
        let depth = depth_record.and_then(|r| t.dsv_target_of(r));
        if tracing_targets() {
            trace_depth_target(&depth_record, depth_key);
        }
        l.target_keys = views.iter().flatten().map(|r| r.resource as usize).chain(depth_key).collect();
        // A view the host cannot make keeps its slot: binding the rest would move every later
        // target down one, and `SV_TargetN` would write a surface this pass does not own.
        let made: Option<Vec<_>> = views
            .iter()
            .map(|slot| slot.and_then(|r| t.target_of(r)))
            .collect();
        let Some(made) = made else {
            l.targets_ready = false;
            l.targets = TargetFormats::default();
            l.target_keys.clear();
            l.list.set_render_targets(&[], None);
            warn_once("a render target the host cannot make; the passes that use it are refused");
            return;
        };
        if depth_record.is_some() && depth.is_none() {
            // A depth view of a subresource the host cannot make (multisampled, or past the
            // texture): binding the whole texture would draw into another mip or slice.
            l.targets_ready = false;
            l.targets = TargetFormats::default();
            l.target_keys.clear();
            l.list.set_render_targets(&[], None);
            warn_once("a depth view the host cannot make; the passes that use it are refused");
            return;
        }
        l.targets_ready = true;
        l.targets = TargetFormats {
            color: made.iter().map(|(_, format)| *format).collect(),
            depth: depth.as_ref().map(|(_, format)| *format),
        };
        let color: Vec<_> = made.into_iter().map(|(texture, _)| texture).collect();
        // Debugging: no depth attachment at all.
        let drop_depth = env_set!("VICTORY_DEBUG_NO_DEPTH_ATTACH");
        if drop_depth {
            l.targets.depth = None;
        }
        let depth = depth.as_ref().map(|(texture, _)| texture).filter(|_| !drop_depth);
        l.list.set_render_targets(&color, depth);
    });
}

/// `VICTORY_TRACE_TARGETS=1`: the first distinct depth attachments bound, with the subresource
/// their view asks for, so a view of one array slice is told from a whole-array binding.
#[cfg(target_os = "macos")]
fn trace_depth_target(
    record: &Option<crate::descriptor::RtvRecord>,
    key: Option<usize>,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(usize, u32, u32, u32)>>> = Mutex::new(None);
    let Some(record) = record else { return };
    let (words, described) = record.words();
    let at = record.resource as usize;
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_default();
    if seen.len() >= 64 || !seen.insert((at, words[1], words[2], words[3])) {
        return;
    }
    let info = crate::resource::texture_info(record.resource);
    let (w, h, layers, mips, flags) = info.as_ref().map_or((0, 0, 0, 0, 0), |i| {
        (i.width, i.height, i.depth_or_layers, i.layout.mips.len().max(1), i.flags)
    });
    let made = key.is_some();
    tracing::info!(
        at = format!("{at:#x}"),
        described,
        ?words,
        format = words[0],
        view_dimension = words[1],
        mip = words[2],
        slice = words[3],
        w,
        h,
        layers,
        mips,
        flags,
        made,
        "depth attachment"
    );
}

/// `ClearRenderTargetView` against the attachment named by the descriptor.
pub(crate) fn clear_rtv(this: This, handle: u64, color: [f32; 4], rects: &[crate::list::Rect]) {
    with_list(this, |t, l| {
        if let Some(record) = crate::descriptor::rtv_record(handle)
            && let Some((texture, _)) = t.target_of(record)
        {
            trace_bump(record.resource as usize, 1);
            if pass_on() {
                pass_note_op(l, &format!("clear-rtv {color:?} rects={}", rects.len()), None, Some(record.resource as usize));
            }
            if rects.is_empty() {
                l.list.clear_render_target(&texture, color);
            } else {
                l.list.clear_render_target_rects(&texture, color, rects);
            }
        }
    });
}

#[inline(never)]
fn trace_clear_dsv(flags: u32, depth: f32) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 6 || (n % 40 == 0 && n < 3000) {
        tracing::info!(flags, depth, n, "ClearDepthStencilView");
    }
}

/// `ClearDepthStencilView`: depth (flag 1) and stencil (flag 2) are independent. A view of one
/// array slice (or a mip) clears through a one-layer view of it, and a view of the whole array
/// clears every layer, since a Metal depth attachment renders into one layer.
pub(crate) fn clear_dsv(this: This, handle: u64, flags: u32, depth: f32, stencil: u8, count: u32) {
    if tracing_targets() {
        trace_clear_dsv(flags, depth);
    }
    // Neither rectangle bounds nor attachment mip/layer selection are represented by
    // NativeList's clear command. Do not widen an unsupported clear to the base texture.
    if count != 0 || flags & 3 == 0 {
        return;
    }
    let Some(record) = crate::descriptor::dsv_record(handle) else {
        return;
    };
    with_list(this, |t, l| {
        let Some(info) = crate::resource::texture_info(record.resource) else {
            trace_clear_skipped("no title texture");
            return;
        };
        // A D3D12 depth-stencil view is 1D or 2D; a 3D slice is not a depth attachment.
        if info.dimension == victory_formats::texture::Dimension::Tex3d {
            trace_clear_skipped("a 3D depth target");
            return;
        }
        let Some((mip, first, layers)) = record.native_dsv_range(
            info.depth_or_layers,
            info.layout.mips.len().max(1),
            info.layout.desc.samples,
        ) else {
            trace_clear_skipped("a view the host cannot clear");
            return;
        };
        for slice in first..first + layers {
            let Some((tex, _)) = t.target_view_at(record, mip, slice) else {
                trace_clear_skipped("no host view");
                continue;
            };
            trace_bump(record.resource as usize, 1);
            if pass_on() {
                pass_note_op(l, &format!("clear-dsv flags={flags} depth={depth} at=mip{mip} slice{slice}"), None, Some(record.resource as usize));
            }
            l.list.clear_depth_stencil(
                &tex,
                (flags & 1 != 0).then_some(depth),
                (flags & 2 != 0).then_some(stencil),
            );
        }
    });
}

/// `VICTORY_TRACE_TARGETS=1`: the first depth clears the host drops, and why.
fn trace_clear_skipped(why: &str) {
    if !tracing_targets() {
        return;
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 12 {
        tracing::warn!(why, n, "a depth clear was not recorded");
    }
}

/// Shader-visible descriptor heaps by address range: start, end, sampler heap, count. The host
/// heap is made when a list first names the heap, from what the title wrote by then.
static HEAPS: Mutex<Vec<(u64, u64, bool, usize)>> = Mutex::new(Vec::new());

pub(crate) fn heap_created(start: u64, samplers: bool, count: usize) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    let step = if samplers { 16 } else { 32 };
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((start, start + step * count as u64, samplers, count));
}

pub(crate) fn heap_released(start: u64) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|h| h.0 != start);
    with_title(|t| t.host.drop_heap(start));
}

/// The host heap the descriptor at `handle` is in (made now if it was not) and its index.
fn heap_of(t: &mut MetalTitle, handle: u64) -> Option<(Rc<super::list::NativeHeap<Metal>>, usize)> {
    if let Some(found) = t.host.heap_at(handle) {
        return Some(found);
    }
    let (start, _, samplers, count) = HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .copied()
        .find(|h| (h.0..h.1).contains(&handle))?;
    let kind = if samplers {
        HeapKind::Sampler
    } else {
        HeapKind::Resource
    };
    let step = if samplers { 16 } else { 32 };
    match t.host.heap(kind, start, count) {
        Ok(heap) => {
            // What the title wrote before the heap was made.
            for i in 0..count as u64 {
                let _ = t.host.mirror(start + i * step);
            }
            t.host.heap_at(handle).or(Some((heap, 0)))
        }
        Err(e) => {
            tracing::warn!(error = %e, "no host descriptor heap");
            None
        }
    }
}

/// The title wrote a view at `handle` (the call `descriptor::mirror` is on Windows).
pub(crate) fn mirror(handle: u64) {
    let _site = super::locktime::at(super::locktime::Site::Resource);
    with_title(|t| {
        match crate::descriptor::view(handle) {
            // A texture the title placed in its own memory has no resource object: the host
            // makes one the first time a view names it.
            Some(crate::descriptor::View::Placed(_, res, d)) if res[0] > 1 => {
                let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                let key = super::textures::placed_key(address, &res);
                if !t.host.knows(key)
                    && let Some(info) = crate::resource::placed_info(address, &res)
                    && let Some((pool, _)) = t.host.memory().resolve(address, 1)
                {
                    t.texture_placed(address, key, pool.id, info.size);
                    if !t.alias_placed(address, key, &info) {
                        t.texture_created(key, info);
                        t.realize(key);
                    }
                }
            }
            // A view of a texture object that no list has used yet. A sampled one waits for the
            // budget room it needs (the submission that names it makes it), an unordered one is
            // written, so it is made here.
            Some(crate::descriptor::View::Shader(r, _)) => {
                t.realize_sampled(r);
            }
            Some(crate::descriptor::View::Unordered(r, ..)) => {
                t.realize(r);
            }
            _ => {}
        }
        if let Err(e) = t.host.mirror(handle) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 3000 {
                tracing::info!(error = %e, handle = format!("{handle:#x}"), "a descriptor was not mirrored");
            }
        }
    });
}

// --- list recording ---

pub(crate) fn set_pipeline(this: This, pipeline: This) {
    with_list(this, |_, l| l.pipeline = pipeline as usize);
}

/// The signature of the strides a described pipeline's streams step by for this draw: for each
/// stream of its input layout, the slot and the stride (the bound view's, or where the layout's
/// elements end when the draw bound none). Metal's vertex descriptor holds those strides, so the
/// host pipeline is made per signature; two draws whose views step differently in the same slot of
/// the same pipeline must not share one. 0 for a pipeline with no stage-in inputs.
fn stream_signature(t: &mut MetalTitle, key: usize, strides: &[u32]) -> u64 {
    if !t.pipeline_streams.contains_key(&key) {
        // Once a pipeline: its layout's own stream ends (a guessed pipeline's come from its draw,
        // and it caches per strides itself).
        let streams = crate::pipeline::native_stages(key as This)
            .and_then(|(_, graphics)| graphics)
            .filter(|g| !g.is_guessed() && !g.inputs().is_empty())
            .map(|g| {
                g.stream_strides()
                    .into_iter()
                    .map(|(slot, stride, _)| (slot as u32, stride as u32))
                    .collect()
            })
            .unwrap_or_default();
        t.pipeline_streams.insert(key, streams);
    }
    strides_signature(t.pipeline_streams.get(&key).map(Vec::as_slice).unwrap_or(&[]), strides)
}

/// [`stream_signature`]'s hash for a pipeline whose streams are `record` (slot, layout end) and
/// whose draw bound `strides`: 0 when the pipeline reads no stage-in inputs.
fn strides_signature(record: &[(u32, u32)], strides: &[u32]) -> u64 {
    const MAX_STRIDE: u32 = 4096;
    if record.is_empty() {
        return 0;
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &(slot, layout_end) in record {
        let bound = strides.get(slot as usize).copied().unwrap_or(0);
        let stride = if bound == 0 || bound > MAX_STRIDE { layout_end } else { bound };
        h = (h ^ (u64::from(slot) << 32 | u64::from(stride))).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// The host pipeline of title pipeline `key` for a draw into `targets` with `strides` bound, made
/// at first use. A console pipeline's state names no target formats, so it is made again for each
/// set of targets; and a described pipeline's vertex descriptor names the strides its streams
/// step by, so it is made again for each set of strides the draw bound. Both are part of the key.
fn pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats, strides: &[u32]) -> Pipe<Metal> {
    let geometry = stream_signature(t, key, strides);
    if let Some(p) = t.pipelines.get(&(key, geometry)) {
        return p.clone();
    }
    let console_key = (key, targets.clone(), geometry);
    if let Some(p) = t.console_pipes.get(&console_key) {
        return p.clone();
    }
    // A cold first world frame binds ~1,300 pipelines; translating them all here holds the frame
    // past the title's watchdog. Once the frame has spent its budget, a pipeline whose shaders are
    // not translated yet is put off: its work is dropped this time (not cached, so a later draw
    // makes it), and its shaders go to the front of the prewarm pool.
    let frame = crate::frame::current();
    let left = frame_left();
    // The BVH builder's pipelines are the exception: the title creates and dispatches them once
    // each, after the frame's budget is spent, so putting their translations off drops the build
    // that makes every ray-traced shadow's acceleration structure. They are translated here
    // instead (tens of programs of some 0.2-0.6 s each, on this thread).
    if left.is_zero()
        && !crate::pipeline::is_builder(key as This)
        && let Some((stages, _)) = crate::pipeline::native_stages(key as This)
        && stages.iter().any(|(_, s)| !s.native_ready())
    {
        for (_, s) in stages.iter().filter(|(_, s)| !s.native_ready()) {
            crate::pipeline::prewarm_next(s);
        }
        super::hosttime::put_off();
        let n = PUT_OFF.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(frame, put_off = n, "pipelines put off for their translations so far");
        }
        return Pipe::Refused;
    }
    // A pipeline Metal is still compiling when the budget ends is put off the same way.
    super::hosttime::enter(super::hosttime::Phase::Pipeline);
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let (made, console) = make_pipe(t, key, targets, strides);
    super::hosttime::leave();
    if t.shaders.put_off != put_off {
        return made;
    }
    if console {
        t.console_pipes.insert(console_key, made.clone());
    } else {
        t.pipelines.insert((key, geometry), made.clone());
    }
    made
}

/// What the current frame has left of its budget for new pipelines (zero: spent).
fn frame_left() -> std::time::Duration {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).left(crate::frame::current(), std::time::Instant::now(), crate::prewarm::FIRST_BIND_BUDGET)
}

/// [`frame_left`] for tests, which read it to see whether a compile spent its frame.
#[cfg(test)]
pub(crate) fn frame_left_now() -> std::time::Duration {
    frame_left()
}

/// Spends what the current frame has left of its budget for new pipelines: from here on [`pipe`]
/// puts off the frame's pipelines whose shaders are not translated yet and [`compile_wait`] waits
/// nothing, until the next frame starts with the budget whole again.
pub(crate) fn spend_frame_budget() {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).spend(crate::frame::current(), std::time::Instant::now());
}

/// A pipeline compile that ended in `error`: if Metal's compiler died on it rather than refusing
/// a shader the host cannot run ([`victory_metal::is_compiler_crash`]), the frame spends the rest
/// of its pipeline budget. One frame on 27 November spent 7.3 s on nine crashes of the same
/// pipeline's compiles (issue #181), about 0.85 s each; with the budget spent, that frame's other
/// new pipelines are put off and retried in later frames, where the crash memory refuses the bad
/// ones at once. A single crash can still cost the frame the ~0.85 s it takes, or on macOS 15 the
/// ~30 s the compiler retries for, out of its budget.
pub(crate) fn compile_failed(name: &str, error: &str) {
    if !victory_metal::is_compiler_crash(error) {
        return;
    }
    spend_frame_budget();
    // The frame this was last logged for: one warning a frame, however many pipelines crash.
    static WARNED: Mutex<u64> = Mutex::new(u64::MAX);
    let frame = crate::frame::current();
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if *warned != frame {
        *warned = frame;
        tracing::warn!(frame, pipeline = name, "a compile crashed Metal's compiler; no new pipelines this frame");
    }
}

/// A compile that ended in a crash of Metal's compiler spends the rest of the frame's budget
/// (issue #181: nine of them held one frame for 7.3 s) and is what `compile_wait` sees, so the
/// frame is handed no new pipeline; one refused for a reason of the host's spends nothing.
#[cfg(test)]
#[test]
fn a_compiler_crash_spends_the_frame_budget() {
    fn whole_frame() {
        *BUDGET.lock().unwrap_or_else(|e| e.into_inner()) = crate::prewarm::FrameBudget::new();
    }
    struct Whole;
    impl Drop for Whole {
        fn drop(&mut self) {
            whole_frame();
        }
    }
    // The budget is the process's and every other test reads it under `TITLE`: hold it, and start
    // the frame here, so no test of theirs runs while this one has spent the budget.
    let _title = lock();
    whole_frame();
    let _whole = Whole;
    let before = frame_left_now();
    assert!(!before.is_zero(), "a frame starts with its budget");
    // A refusal of `make_render`'s, not a crash.
    compile_failed("a guessed pipeline", "a target format the host does not know");
    let left = frame_left_now();
    assert!(!left.is_zero(), "a refusal is not a crash");
    assert!(left <= before);
    assert!(!frame_spent_by_crash());

    compile_failed(
        "a guessed pipeline",
        "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. \
         This error occurred after multiple retries.",
    );
    assert!(frame_left_now().is_zero(), "a crash spends the frame");
    assert!(frame_spent_by_crash(), "and `compile_wait` tells it from a frame out of time");
}

/// A title call never waits out a frame's budget for a pipeline compile: the wait is the archive's
/// window, and none after the frame's first pipeline that misses it. The budget used to be handed
/// to the compile, on a title's thread and under the host's lock: while it lasted the pump could
/// not hand held work to the GPU, so every queue stayed on its cross-queue barrier word (the world
/// run of 27 November 21:32: `work held for a wait that does not hold`, head for 20+ s, and
/// `metal: compute pipeline compiled: 28993 ms`), and the title's 30 s watchdog ended the run.
#[test]
fn a_frame_never_waits_out_its_budget_for_a_compile() {
    fn whole_frame() {
        *BUDGET.lock().unwrap_or_else(|e| e.into_inner()) = crate::prewarm::FrameBudget::new();
    }
    struct Whole;
    impl Drop for Whole {
        fn drop(&mut self) {
            whole_frame();
        }
    }
    // The budget is the process's: hold the title lock, as the test above does, so no other test's
    // title call resets the frame under this one.
    let _title = lock();
    whole_frame();
    let _whole = Whole;
    assert!(!frame_left_now().is_zero(), "the frame has its budget for translations");
    let window = compile_wait_now(0);
    assert!(
        window <= std::time::Duration::from_millis(20),
        "a compile waits the archive's window, not the frame's 10 s budget: {window:?}"
    );
    assert!(!window.is_zero(), "a pipeline the archive holds is still waited for");
    assert_eq!(
        compile_wait_now(1),
        std::time::Duration::ZERO,
        "after the frame's first missed compile, its other pipelines are put off at once"
    );
}

/// How long a title call may hold the frame while Metal makes a pipeline: the few ms a pipeline
/// the disk cache and the archive hold needs, and no more.
///
/// It used to be what the frame had left of [`crate::prewarm::FIRST_BIND_BUDGET`]: up to 10 s per
/// pipeline. A pipeline is made from the title's own call (`SetPipelineState` is not where a host
/// pipeline is made: the draw or dispatch that first needs it is), so that wait ran on a title
/// thread while [`with_title`] held the host's lock. Under the lock nothing else in the host runs:
/// the timer cannot pump held work, so every queue's head stays on its cross-queue barrier word and
/// the GPU idles (`work held for a wait that does not hold`: `... cmp 7 ref 0x14a7 ... now
/// Some(5286)`) for as long as Metal takes, and the title's own frame pays it as well. The world
/// run of 27 November 21:32 has both: frames of 20-30 s with the GPU idle, `metal: compute pipeline
/// compiled: 28993 ms`, and the 30 s watchdog ending the run.
///
/// Compiles run on `pipeline-compile` threads (`victory_metal::metal::within`) whatever this is: a
/// pipeline not made in this window is put off (that draw or dispatch is dropped this time, nothing
/// is cached) and a later draw picks it up once the compile thread is done. After the frame's first
/// pipeline that misses the window, the rest of the frame asks for none, so a cold frame spends at
/// most one window in the host and goes on; the pipelines it asked for arrive over the next frames,
/// as they did past the budget before ([`crate::prewarm::FrameBudget`]).
fn compile_wait(t: &MetalTitle) -> std::time::Duration {
    compile_wait_for(t.shaders.put_off)
}

/// [`compile_wait`] for a frame whose pipelines put off so far number `put_off`.
fn compile_wait_for(put_off: u64) -> std::time::Duration {
    const ARCHIVED: std::time::Duration = std::time::Duration::from_millis(20);
    if frame_spent_by_crash() {
        return std::time::Duration::ZERO;
    }
    /// The frame last seen here, and the pipelines put off when it began.
    static PAST: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let frame = crate::frame::current();
    let mut past = PAST.lock().unwrap_or_else(|e| e.into_inner());
    if past.0 != frame {
        *past = (frame, put_off);
    }
    if past.1 == put_off { ARCHIVED } else { std::time::Duration::ZERO }
}

/// [`compile_wait`] for tests, without a title: the window a frame with `put_off` pipelines put off
/// so far waits.
#[cfg(test)]
pub(crate) fn compile_wait_now(put_off: u64) -> std::time::Duration {
    compile_wait_for(put_off)
}

/// Whether a compile that crashed Metal's compiler spent the current frame ([`frame_left`] is
/// zero for a frame that is merely past its time too; this tells the two apart).
fn frame_spent_by_crash() -> bool {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).spent_by_crash()
}

/// When this frame made its first new pipeline, and pipelines put off since ([`pipe`]).
static BUDGET: Mutex<crate::prewarm::FrameBudget> = Mutex::new(crate::prewarm::FrameBudget::new());
static PUT_OFF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Runs a translation `what` names on a thread of its own, once at a time per name, for a
/// pipeline put off past the frame's budget ([`guess`]).
fn put_off(what: String, translate: impl FnOnce() + Send + 'static) {
    static RUNNING: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
    if !RUNNING.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(what.clone()) {
        return;
    }
    let n = PUT_OFF.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(put_off = n, "pipelines put off for their translations so far");
    }
    let _ = std::thread::Builder::new().name("shader-put-off".into()).spawn(move || {
        crate::prewarm::lower_priority();
        translate();
        if let Some(r) = RUNNING.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            r.remove(&what);
        }
    });
}

// Pass trace (`VICTORY_TRACE_PASSES=<dir>,<skip>,<presents>`, docs/native-host.md): every draw,
// dispatch, clear and copy of a few world frames in queue order (`PASS`), the commands the host
// could not prepare (`PASSFAIL`), and at the window's last present each texture they named
// (`PASSTEX`, with a .ppm in <dir>). Off unless the variable is set: callers test `pass_on()`
// before building a line, so with it unset no line is formatted and nothing is kept.
use super::pass_on;
static PASS_KEYS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
// Why the last draw on this thread was dropped, for its `PASS` line.
thread_local! { static LAST_WHY: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") }; }

/// `VICTORY_TRACE_PASSES`' value: the folder for images, how many world presents to let pass
/// first, how many presents the window lasts after the one that opens it, and (optional) a
/// frame number the window opens at instead of after `skip` world presents. A script that wants
/// the world frame cannot count presents: the loading screens present thousands of times, and
/// which of their presents counts as a "world" one (500 draws or more since the last) depends
/// on how the machine happened to schedule the load. The title's frame number is the same from
/// run to run, so `...,<frame>` opens the window at the first present at or after that frame
/// (`0` or absent: the presents count only).
fn pass_config(v: &str) -> (String, u64, i64, Option<u64>) {
    let mut parts = v.split(',');
    let dir = parts.next().filter(|d| !d.is_empty()).unwrap_or(".").to_owned();
    let skip = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(100);
    let presents = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(4);
    let frame = parts.next().and_then(|s| s.trim().parse().ok()).filter(|&f| f > 0);
    (dir, skip, presents, frame)
}

#[cfg(test)]
#[test]
fn pass_trace_reads_its_switch_and_stays_off_without_it() {
    assert_eq!(pass_config("/tmp/p,8,5"), ("/tmp/p".into(), 8, 5, None));
    assert_eq!(pass_config("/tmp/p"), ("/tmp/p".into(), 100, 4, None));
    assert_eq!(pass_config(",x,"), (".".into(), 100, 4, None));
    assert_eq!(pass_config("/tmp/p,8,5,1900"), ("/tmp/p".into(), 8, 5, Some(1900)));
    assert_eq!(pass_config("/tmp/p,8,5,0"), ("/tmp/p".into(), 8, 5, None), "0: no frame");
    // No test sets `VICTORY_TRACE_PASSES`, so the window never opens and nothing is noted.
    assert!(!pass_on());
}

/// Whether the pass trace's window opens at a present: at or after `frame` when one is given,
/// else once `world` presents that recorded 500 draws or more have gone by.
fn pass_opens(world: u64, skip: u64, frame: Option<u64>, now_frame: u64) -> bool {
    match frame {
        Some(frame) => now_frame >= frame,
        None => world >= skip,
    }
}

#[cfg(test)]
#[test]
fn pass_trace_opens_on_a_frame_number_or_a_count_of_world_presents() {
    // A frame number is exact: the window is closed before it and open from it on.
    assert!(!pass_opens(9999, 0, Some(1900), 1899));
    assert!(pass_opens(0, 100, Some(1900), 1900));
    assert!(pass_opens(0, 100, Some(1900), 2200));
    // No frame: the count of world presents, as before.
    assert!(!pass_opens(2, 3, None, 5000));
    assert!(pass_opens(3, 3, None, 5000));
}

fn pass_tex(k: usize) -> String {
    match crate::resource::texture_info(k as This) {
        Some(i) => format!("{k:#x}:f{}:{}x{}@{:#x}", i.format, i.width, i.height, i.memory as usize),
        None => format!("{k:#x}:?"),
    }
}

/// One piece of work, as it is recorded: its shaders, targets and the textures its tables name.
fn pass_note(t: &MetalTitle, l: &mut MetalList, kind: &str, outcome: &str, compute: bool, n: u32) {
    let sh = stage_hashes(l.pipeline);
    let (reads, writes, none, keys) = if compute {
        // A dispatch's tables are read to the entries its shader names, as a draw's are: a
        // bindless compute can name a hundred entries, and the ones a shadow pass reads (its
        // cascades, its depth) are not always in the first few.
        let (root, binding) = (l.list.compute_args(), l.list.compute_binding());
        let mut limits: Vec<(u8, usize)> = Vec::new();
        if let Some(b) = binding.filter(|_| outcome.starts_with("ran")) {
            limits.extend(b.table_entries.iter().copied());
            // A slot the shader names no entry of: the pass trace's own floor (24).
            for slot in 0..root.len() {
                if !limits.iter().any(|(s, _)| *s as usize == slot) {
                    limits.push((slot as u8, 24));
                }
            }
        }
        t.host.pass_views(root, &limits)
    } else {
        let (root, binding) = l.list.graphics_args();
        let mut limits: Vec<(u8, usize)> = Vec::new();
        if let Some(b) = binding.filter(|_| outcome.starts_with("ran")) {
            limits.extend(b.table_entries.iter().copied());
            if let Some(ps) = &b.ps {
                limits.extend(ps.table_entries.iter().copied());
            }
            if limits.is_empty() { limits.push((255, 0)); }
        }
        t.host.pass_views(root, &limits)
    };
    let tg: Vec<String> = if compute { Vec::new() } else { l.target_keys.iter().map(|&k| pass_tex(k)).collect() };
    {
        let mut ks = PASS_KEYS.lock().unwrap_or_else(|e| e.into_inner());
        for k in keys.iter().chain(if compute { [].iter() } else { l.target_keys.iter() }) {
            if !ks.contains(k) { ks.push(*k); }
        }
    }
    let at = l.list.cmds.len();
    l.list.notes.push((at, format!("kind={kind} n={n} out={outcome:?} sh={sh:?} tg={tg:?} rd={reads:?} wr={writes:?} none={none} topo={}", l.topology)));
}

fn pass_note_op(l: &mut MetalList, what: &str, src: Option<usize>, dst: Option<usize>) {
    {
        let mut ks = PASS_KEYS.lock().unwrap_or_else(|e| e.into_inner());
        for k in src.iter().chain(dst.iter()) {
            if !ks.contains(k) { ks.push(*k); }
        }
    }
    let at = l.list.cmds.len();
    l.list.notes.push((at, format!("kind={what} src={:?} dst={:?}", src.map(pass_tex), dst.map(pass_tex))));
}

/// At each present: switch the window on and off, and while on write every texture the window's
/// work named: its range, how much is non-zero and not finite, and (colour formats) an image.
fn pass_present(resource: This) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static LEFT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
    static NUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static CONFIG: OnceLock<Option<(String, u64, i64, Option<u64>)>> = OnceLock::new();
    let Some((dir, skip, presents, frame)) =
        CONFIG.get_or_init(|| std::env::var("VICTORY_TRACE_PASSES").ok().map(|v| pass_config(&v)))
    else {
        return;
    };
    let (skip, presents, frame) = (*skip, *presents, *frame);
    let num = NUM.fetch_add(1, Ordering::Relaxed);
    let now = START.get_or_init(std::time::Instant::now).elapsed().as_secs();
    let left = LEFT.load(Ordering::Relaxed);
    // `skip` counts world presents (500 draws or more recorded since the last present).
    static LASTREC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let rec = RECORDED.load(Ordering::Relaxed);
    if rec - LASTREC.swap(rec, Ordering::Relaxed) >= 500 {
        WORLD.fetch_add(1, Ordering::Relaxed);
    }
    if left == i64::MIN {
        if pass_opens(WORLD.load(Ordering::Relaxed), skip, frame, crate::frame::current()) {
            LEFT.store(presents, Ordering::Relaxed);
            super::PASS_ON.store(true, Ordering::Relaxed);
            eprintln!("PASSWINDOW start present={num} t={now} frame={}", crate::frame::current());
        }
        return;
    }
    if left < 0 {
        return;
    }
    eprintln!("PASSPRESENT present={num} left={left} resource={}", pass_tex(resource as usize));
    let dump = left == 0 || std::env::var("VICTORY_TRACE_PASSES_DUMP_EACH").is_ok();
    if left == 0 {
        super::PASS_ON.store(false, Ordering::Relaxed);
    }
    LEFT.store(left - 1, Ordering::Relaxed);
    if !dump {
        return;
    }
    let mut keys = PASS_KEYS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if !keys.contains(&(resource as usize)) { keys.push(resource as usize); }
    with_title(|t| {
        for key in keys {
            let Some(info) = crate::resource::texture_info(key as This) else { continue };
            let (w, h) = (info.width as usize, info.height as usize);
            let bpp = match info.format { 2 => 16, 10 | 11 | 16 => 8, 24 | 26 | 28 | 29 | 34 | 41 | 42 | 67 | 87 | 91 => 4, 49 | 54 | 56 | 57 => 2, 61 | 62 | 65 => 1, _ => 0 };
            // A 3D texture (a colour grading table) is read whole; arrays are skipped.
            let d = if info.dimension == victory_formats::texture::Dimension::Tex3d { info.depth_or_layers.max(1) as usize } else { 1 };
            if bpp == 0 || w * h == 0 || w * h > 3840 * 2160 || d == 1 && info.depth_or_layers > 1 && info.format != 28 {
                eprintln!("PASSTEX present={num} key={key:#x} fmt={} {w}x{h} layers={} skipped", info.format, info.depth_or_layers);
                continue;
            }
            let Some(tex) = t.texture_of(key) else {
                eprintln!("PASSTEX present={num} key={key:#x} fmt={} {w}x{h} no host texture", info.format);
                continue;
            };
            let mut raw = vec![0u8; w * h * d * bpp];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
            let nonzero = raw.chunks_exact(bpp).filter(|p| p.iter().any(|&b| b != 0)).count() as f64 / (w * h * d) as f64;
            let mut slices = String::new();
            if d > 1 {
                let lit = raw.chunks_exact(w * h * bpp).filter(|s| s.iter().any(|&b| b != 0)).count();
                let distinct = raw.chunks_exact(bpp).collect::<std::collections::HashSet<_>>().len();
                slices = format!(" depth={d} slices_nonzero={lit} distinct_texels={distinct}");
            }
            let name = format!("p{num}_{key:x}_f{}_{w}x{h}.ppm", info.format);
            let mut out = format!("P6\n{w} {}\n255\n", h * d).into_bytes();
            let mut extra = String::new();
            match info.format {
                2 | 10 | 24 | 26 => {
                    let rgb: Vec<[f32; 3]> = raw.chunks_exact(bpp).map(|p| decode_hdr(info.format, p)).collect();
                    let (mut max, mut min, mut sum, mut bad, mut neg) = (f32::MIN, f32::MAX, 0f64, 0usize, 0usize);
                    for v in rgb.iter().flatten() {
                        if !v.is_finite() { bad += 1; continue; }
                        max = max.max(*v); min = min.min(*v); sum += f64::from(*v); neg += usize::from(*v < 0.0);
                    }
                    // format 10: count infinities and NaNs by their bits (decode_hdr folds inf).
                    let mut nan16 = 0usize;
                    if info.format == 10 {
                        nan16 = raw.chunks_exact(2).filter(|p| u16::from_le_bytes([p[0], p[1]]) & 0x7c00 == 0x7c00).count();
                    }
                    extra = format!(" max={max} min={min} mean={} notfinite={bad} infnan16={nan16} negative={neg}", sum / (rgb.len() * 3).max(1) as f64);
                    // A few texels of 32-bit floats (exposure and histogram results): every channel, alpha too.
                    if info.format == 2 && w * h <= 16 {
                        let v: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                        extra += &format!(" values={v:?}");
                    }
                    out.extend(rgb.iter().flat_map(|c| c.map(|v| if v.is_finite() && v > 0.0 { ((v / (1.0 + v)).sqrt() * 255.0) as u8 } else { 0 })));
                }
                28 | 29 | 87 | 91 => {
                    let bgra = matches!(info.format, 87 | 91);
                    out.extend(raw.chunks_exact(4).flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
                }
                _ => {
                    // Any other: the low byte of up to three channels, as grey or colour.
                    out.extend(raw.chunks_exact(bpp).flat_map(|p| [p[0], p[bpp / 2], p[bpp - 1]]));
                }
            }
            let _ = std::fs::write(std::path::Path::new(dir).join(&name), out);
            eprintln!("PASSTEX present={num} key={key:#x} fmt={} {w}x{h} at={:#x} nonzero={nonzero:.4}{slices}{extra} file={name}", info.format, info.memory as usize);
        }
    });
}

/// Work dropped because its pipeline has no host pipeline.
static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The same split into draws (what `recorded` counts) and dispatches: a run's dropped-draw
/// share is the first over `recorded`, not `DROPPED` over `recorded`.
static DROPPED_DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DROPPED_DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The same, by the shaders of the pipeline it wanted: what a refusal costs in draws (and in
/// dispatches, which are not draws), the order to fix reasons in (`run-report.py` reads these
/// lines), run-wide and cumulative.
static DROPPED_SHADERS: Mutex<Vec<(String, u64, u64)>> = Mutex::new(Vec::new());

/// Counts a draw (`dispatch` false) or a dispatch dropped for want of a host pipeline under its
/// shaders: `(shaders, draws, dispatches)`.
fn drop_shaders(pipeline: usize, dispatch: bool) {
    let key = match stage_hashes(pipeline).join(" ") {
        key if key.is_empty() => format!("pipeline {pipeline:#x} (no stages)"),
        key => key,
    };
    let mut d = DROPPED_SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    let at = match d.iter().position(|(k, _, _)| *k == key) {
        Some(i) => i,
        None => {
            d.push((key, 0, 0));
            d.len() - 1
        }
    };
    if dispatch {
        d[at].2 += 1;
        DROPPED_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        d[at].1 += 1;
        DROPPED_DRAWS.fetch_add(1, Ordering::Relaxed);
    }
}

/// What [`log_dropped_shaders`] last wrote, so a line is a shader that grew (its value is the
/// run's total so far: a reader takes the last line of a shader).
static DROPPED_SHOWN: Mutex<Vec<(String, u64, u64)>> = Mutex::new(Vec::new());

/// Logs the drops by shader, most costly first, with the reason the shader was refused: the
/// ranking of what fixing a reason is worth in draws.
fn log_dropped_shaders(frame: u64) {
    let mut d = DROPPED_SHADERS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if d.is_empty() {
        return;
    }
    d.sort_by_key(|(_, draws, dispatches)| std::cmp::Reverse(draws + dispatches));
    let stats = shader_stats();
    let mut shown = DROPPED_SHOWN.lock().unwrap_or_else(|e| e.into_inner());
    let mut changed = 0;
    for (what, draws, dispatches) in d.iter().take(16) {
        match shown.iter_mut().find(|(k, _, _)| k == what) {
            Some((_, d0, d1)) if *d0 == *draws && *d1 == *dispatches => continue,
            Some(e) => {
                e.1 = *draws;
                e.2 = *dispatches;
            }
            None => shown.push((what.clone(), *draws, *dispatches)),
        }
        changed += 1;
        let reason = stats
            .as_ref()
            .and_then(|s| {
                s.refused
                    .iter()
                    .find(|(k, _)| k.starts_with(what.as_str()) && k[what.len()..].starts_with(": "))
            })
            .map(|(k, _)| k[what.len() + 2..].to_owned())
            .unwrap_or_else(|| "not refused: not translated yet, or put off".to_owned());
        tracing::info!(
            frame,
            draws = *draws,
            dispatches = *dispatches,
            shaders = &what[..],
            reason = reason.as_str(),
            "work dropped by shader"
        );
    }
    if changed > 0 {
        let draws: u64 = d.iter().map(|(_, n, _)| n).sum();
        let dispatches: u64 = d.iter().map(|(_, _, n)| n).sum();
        tracing::info!(frame, draws, dispatches, shaders = d.len(), "work dropped by shader, all");
    }
}

/// A draw into `l`'s targets that the host drops, counted per target for the trace.
fn trace_drop(l: &MetalList) {
    l.target_keys.iter().for_each(|&k| trace_bump(k, 2));
}

/// A dropped draw or execution, by what it was (its shaders and targets), for the frame trace.
pub(crate) fn trace_drop_pipe(key: String) {
    let mut d = DROP_PIPES.lock().unwrap_or_else(|e| e.into_inner());
    match d.iter_mut().find(|(k, _)| *k == key) {
        Some((_, n)) => *n += 1,
        None => d.push((key, 1)),
    }
}

/// Work dropped since the last present, by reason, for the `VICTORY_TRACE_TARGETS` frame line.
pub(crate) fn trace_reason(why: &'static str) {
    if tracing_targets() {
        let mut r = DROP_REASONS.lock().unwrap_or_else(|e| e.into_inner());
        match r.iter_mut().find(|(w, _)| *w == why) {
            Some((_, n)) => *n += 1,
            None => r.push((why, 1)),
        }
    }
}

/// Counts a draw or dispatch the host drops, and warns the first time for each `why`.
fn dropped(why: &'static str) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    if pass_on() {
        LAST_WHY.with(|w| w.set(why));
    }
    warn_once(why);
    trace_reason(why);
}

/// A warning logged once per distinct `what` for the run.
pub(crate) fn warn_once(what: &'static str) {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&what) {
        seen.push(what);
        tracing::warn!("native host drops work: {what} (first time; counted in \"native draws so far\")");
    }
}

/// Draws the title recorded, and those that went on to the host list.
static RECORDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENCODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Presents whose resource had no host texture.
static NO_TEXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(draws recorded, draws handed to the host list, draws dropped for want of a host pipeline,
/// presents that found no host texture)`, for the frame log.
pub(crate) fn draw_counts() -> (u64, u64, u64, u64) {
    (
        RECORDED.load(Ordering::Relaxed),
        ENCODED.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        NO_TEXTURE.load(Ordering::Relaxed),
    )
}

/// The frame log's line of native draw counters, and the commands dropped by kind (kept out of
/// `PresentX`, whose frame it would otherwise enlarge).
#[inline(never)]
pub(crate) fn log_draw_counts(frame: u64) {
    let (recorded, encoded, no_pipeline, no_texture) = draw_counts();
    let (mesh, mesh_indirect, _, copies, _) = crate::list::dropped_kinds();
    let copies_recorded = COPIES[0].load(Ordering::Relaxed);
    let indirect = crate::list::indirect_dropped();
    let (r, heap_cleared, [dropped_unset_root, dropped_unmapped, dropped_failed, typed_views, resource_heaps], metal_mb) = with_title(|t| {
        let h = &t.host;
        (h.backend.residency_counts(), h.cleared(), [h.unset_roots(), h.unmapped(), h.failed(), h.typed_views(), h.resource_heaps() as u64], victory_backend::backend::ComputeBackend::allocated_bytes(&h.backend) >> 20)
    })
    .unwrap_or_default();
    // Where the memory is when command buffers stall (#79): Metal's own count and the process footprint.
    #[cfg(target_os = "macos")]
    let footprint_mb = victory_platform::mem::footprint() >> 20;
    #[cfg(not(target_os = "macos"))]
    let footprint_mb = 0usize;
    log_dropped_shaders(frame);
    let no_pipeline_draws = DROPPED_DRAWS.load(Ordering::Relaxed);
    let no_pipeline_dispatches = DROPPED_DISPATCHES.load(Ordering::Relaxed);
    tracing::info!(frame, recorded, encoded, no_pipeline, no_pipeline_draws, no_pipeline_dispatches, no_texture, mesh, mesh_indirect, indirect, texture_copies = copies, copies_recorded, use_resource = r.use_resource, heap_resources = r.heap_resources, heap_entries = r.heap_entries, heap_allocations = r.heap_allocations, heap_set_mb = r.heap_set_bytes >> 20, metal_mb, footprint_mb, heap_cleared, dropped_unset_root, dropped_unmapped, dropped_failed, typed_views, resource_heaps, "native draws so far");
    if let Some(n) = super::host::heap_set_trace()
        && let Some([all, idle, never]) = with_title(|t| t.host.heap_set_report(n)).flatten()
    {
        let mb = |b: u64| b >> 20;
        tracing::info!(frame, idle_frames = n, textures = all.0, textures_mb = mb(all.1), buffers = all.2, buffers_mb = mb(all.3), idle_textures = idle.0, idle_textures_mb = mb(idle.1), idle_buffers = idle.2, idle_buffers_mb = mb(idle.3), never_textures = never.0, never_textures_mb = mb(never.1), never_buffers = never.2, never_buffers_mb = mb(never.3), "heap set: allocations (once over all heaps) no bound table named for that many frames (idle), or ever (never)");
    }
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *TRACE.get_or_init(|| std::env::var_os("VICTORY_TRACE_SHADER_CACHE").is_some_and(|v| !v.is_empty() && v != "0")) {
        let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
        let (libraries_cached, libraries_converted) = victory_metal::disk::counts();
        tracing::info!(frame, shaders_cached, shaders_translated, libraries_cached, libraries_converted, "native shader cache so far");
    }
    let (typed_views, typed_views_live) =
        with_title(|t| (t.host.typed_views(), t.host.live_typed_views())).unwrap_or_default();
    if typed_views > 0 {
        tracing::info!(frame, typed_views, typed_views_live, "native typed buffer views");
    }
    super::descriptor_trace::log(frame);
}

/// A title pipeline object's host pipeline: its shaders translated for the host, then converted.
/// The second result says the pipeline depends on the draw's targets.
fn make_pipe(
    t: &mut MetalTitle,
    key: usize,
    targets: &TargetFormats,
    strides: &[u32],
) -> (Pipe<Metal>, bool) {
    let Some((stages, graphics)) = crate::pipeline::native_stages(key as This) else {
        return (Pipe::Refused, false);
    };
    let graphics = graphics.map(|g| {
        // The layout's own stream ends, before a draw's strides change them: what a host
        // pipeline's vertex descriptor must be made for (`stream_signature`).
        if !g.is_guessed() && !g.inputs().is_empty() {
            let streams: Vec<(u32, u32)> = g
                .stream_strides()
                .into_iter()
                .map(|(slot, stride, _)| (slot as u32, stride as u32))
                .collect();
            t.pipeline_streams.insert(key, streams);
        }
        let g = g.with_strides(strides);
        if g.names_targets() {
            (g, false)
        } else {
            (super::console_state::graphics_state_for(&g, targets.color.clone(), targets.depth), true)
        }
    });
    let console = graphics.as_ref().is_some_and(|g| g.1);
    if console && env_set!("VICTORY_TRACE_BIND") {
        tracing::info!(pipeline = key, ?targets, state = ?graphics.as_ref().map(|g| &g.0), "console pipeline state at its draw");
    }
    let graphics = graphics.map(|g| g.0);
    (make_host_pipe(t, stages, graphics), console)
}

/// The host pipeline of compute pipeline `key` for dispatches of thread counts, made at first
/// use: its shader translated for them (`Options::thread_counts`).
fn thread_pipe(t: &mut MetalTitle, key: usize) -> Pipe<Metal> {
    if let Some(p) = t.thread_pipes.get(&key) {
        return p.clone();
    }
    // As in `pipe`: a builder dispatch counted in threads must find its form, or the structure
    // it builds is missing for the rest of the run. For a shader the prewarm pool has, any other
    // thread-count translation not made yet is left to a worker: it is unbounded, and the frame
    // must not hold the title's watchdog for it.
    if !crate::pipeline::is_builder(key as This)
        && let Some(s) = crate::pipeline::compute_shader(key as This)
        && s.pool_has_it()
        && !s.threads_ready()
    {
        put_off(format!("{} threads", s.hash), move || drop(s.native_threads_shader()));
        return Pipe::Refused;
    }
    let (hash, shader) = crate::pipeline::native_threads(key as This);
    let name = format!("compute {hash} by thread counts");
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let made = match shader {
        Ok(cs) => t.compute_pipe(&cs, &name),
        Err(e) => {
            let why = e.clone();
            t.refused_translation(&name, e);
            // The builder's thread form is refused: a dispatch counted in threads cannot run as
            // groups (its lanes past the counts would write over memory that is not theirs), and
            // the builder's structures cannot be trusted from here on (`native::builder`).
            if crate::pipeline::is_builder(key as This) {
                crate::native::builder::trip(format!(
                    "the BVH builder's {name} has no form for dispatches counted in threads: {why}"
                ));
            }
            Pipe::Refused
        }
    };
    if t.shaders.put_off == put_off {
        t.thread_pipes.insert(key, made.clone());
    }
    made
}

fn make_host_pipe(
    t: &mut MetalTitle,
    stages: Vec<(&'static str, Arc<crate::pipeline::Shader>)>,
    graphics: Option<super::state::Graphics>,
) -> Pipe<Metal> {
    if stages.is_empty() && graphics.as_ref().is_some_and(|g| g.is_guessed()) {
        return Pipe::Skipped;
    }
    let name = stages
        .iter()
        .map(|(stage, s)| format!("{stage} {}", s.hash))
        .collect::<Vec<_>>()
        .join(" ");
    // A pixel shader drawing into integer targets is translated for them (Metal refuses a float
    // output for an integer attachment).
    let integer = graphics.as_ref().and_then(|g| g.color_formats().ok()).map_or(0, |c| super::shaders::integer_targets(&c));
    let mut hosts = Vec::new();
    for (stage, shader) in &stages {
        // The prewarm pool translates what the title's thread has not got yet: a translation can
        // take tens of seconds, and the frame must not hold the title's watchdog (30 s) waiting
        // for one. The pipeline is put off (counted, so `pipe` does not keep the refusal) and a
        // later draw picks it up. A shader the pool does not have (the pool is off) is translated
        // here, as before.
        if shader.plain_off_frame() {
            crate::pipeline::prewarm_next(shader);
            // Counted as put off, so `pipe` does not keep the refusal.
            t.shaders.put_off += 1;
            return Pipe::Refused;
        }
        let made = shader.native().map(Arc::clone).map_err(str::to_owned);
        let made = match made {
            Ok(_) if *stage == "pixel" && integer != 0 => {
                if shader.pool_has_it() && !shader.targets_cached(integer) {
                    let s = shader.clone();
                    put_off(format!("{} targets {integer}", s.hash), move || drop(s.translate_targets(integer)));
                    // Counted as put off, so `pipe` does not keep the refusal.
                    t.shaders.put_off += 1;
                    return Pipe::Refused;
                }
                shader.translate_targets(integer).map_err(|e| format!("integer targets {integer:#b}: {e}"))
            }
            made => made,
        };
        match made {
            Ok(h) => hosts.push((*stage, h)),
            Err(e) => {
                t.refused_translation(&name, e.to_string());
                return Pipe::Refused;
            }
        }
    }
    let find = |stage: &str| hosts.iter().find(|h| h.0 == stage).map(|h| &*h.1);
    match graphics {
        None => match find("compute") {
            Some(cs) => t.compute_pipe(cs, &name),
            None => Pipe::Refused,
        },
        Some(g) if g.is_guessed() => match hosts.iter().find(|h| h.0 == "vertex") {
            Some((_, vs)) => Pipe::Guess(Rc::new(Guessed {
                vs_source: stages.iter().find(|s| s.0 == "vertex").expect("a vertex stage").1.clone(),
                vs: vs.clone(),
                ps: hosts.iter().find(|h| h.0 == "pixel").map(|h| h.1.clone()),
                name,
                state: g.clone(),
                made: Default::default(),
            })),
            None => Pipe::Refused,
        },
        Some(g) => match find("vertex") {
            Some(vs) => t.render_pipe(vs, find("pixel"), &g, &name),
            None => {
                t.refused_translation(&name, "a graphics pipeline without a vertex shader".into());
                Pipe::Refused
            }
        },
    }
}

/// The shader path's counters: pipelines made and cached, refusals by reason.
pub fn shader_stats() -> Option<super::shaders::ShaderStats> {
    with_title(|t| t.shaders.clone())
}

/// Makes title pipeline `key` the compute pipeline of a pre-translated DXIL container with the
/// given group size (what a translated shader's pipeline is made of); for diagnostics and tests.
pub fn install_compute_pipeline(key: usize, dxil: &[u8], threads: [usize; 3]) -> Result<(), String> {
    with_title(|t| {
        let pso = t.host.compute_pipeline(dxil, threads)?;
        t.pipelines.insert((key, 0), Pipe::Compute(pso));
        Ok(())
    })
    .unwrap_or_else(|| Err("no native host GPU".into()))
}

#[cfg(test)]
pub(crate) fn install_test_render_pipeline(
    key: usize,
    pso: Rc<<Metal as victory_backend::resources::GraphicsBackend>::RenderPipeline>,
) {
    with_title(|t| {
        t.pipelines.insert((key, 0), Pipe::Render(pso, None));
    });
}

pub(crate) fn dispatch(this: This, x: u32, y: u32, z: u32) {
    with_list(this, |t, l| {
        if let Some(why) = builder_hold(l.pipeline) {
            builder_dropped(t, l, &format!("[{x}, {y}, {z}]"), &why);
            return;
        }
        match pipe(t, l.pipeline, &TargetFormats::default(), &[]) {
            Pipe::Compute(pso) => {
                watch_dispatch(t, l);
                trace_record_dispatch(t, l, "ran");
                if crate::pipeline::traces(l.pipeline as This) {
                    super::rt::note_ray_direct(&stage_hashes(l.pipeline).join(" "), [x, y, z]);
                }
                if crate::pipeline::is_builder(l.pipeline as This) {
                    super::rt::note_builder([x, y, z], false);
                }
                if pass_on() {
                    pass_note(t, l, &format!("dispatch[{x},{y},{z}]"), "ran", true, 1);
                }
                l.list.set_compute_pipeline(&pso);
                if let Err(e) = l.list.dispatch(x, y, z) {
                    tracing::warn!(error = %e, "dispatch");
                }
                if x != 0 && y != 0 && z != 0 {
                    record_ambient(t, l, pso.binding.as_ref());
                }
            }
            _ => {
                dropped_dispatch(t, l, &format!("groups [{x}, {y}, {z}]"));
                // A refused resolve still leaves the light to write: its light image is in the
                // tables the title bound for it, and the pass goes where the dispatch would have.
                if x != 0 && y != 0 && z != 0 {
                    record_ambient(t, l, None);
                }
            }
        }
    });
}

/// `VICTORY_WATCH_TITLE`: the unordered-access buffer views this compute dispatch's tables hold,
/// with its shader hashes, for any that covers the watched range (the candidate writers of a page
/// whose bytes changed at a present).
fn watch_dispatch(t: &MetalTitle, l: &MetalList) {
    if !super::title_watch::enabled() {
        return;
    }
    let source = format!("compute {}", stage_hashes(l.pipeline).join(" "));
    for (addr, size) in t.host.table_uav_ranges(l.list.compute_args()) {
        super::title_watch::view(&source, addr, size);
    }
}

/// Why a dispatch of title pipeline `pipeline` is held this frame, if it is: it is the console's
/// BVH builder's, whose forms are not all settled or which has tripped (`native::builder`).
/// `None` with the switch off, for any other pipeline, and once the builder runs.
fn builder_hold(pipeline: usize) -> Option<String> {
    if !crate::pipeline::is_builder(pipeline as This) {
        return None;
    }
    crate::native::builder::held()
}

/// A dispatch of the BVH builder dropped because the builder is held (`builder_hold`).
fn builder_dropped(t: &MetalTitle, l: &mut MetalList, how: &str, why: &str) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    if crate::pipeline::is_builder(l.pipeline as This) {
        super::rt::note_builder_dropped();
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            why,
            "native host holds the BVH builder back: its dispatches are dropped (VICTORY_BVH_BUILDER=1; first time)"
        );
    });
    if pass_on() {
        pass_note(t, l, &format!("dispatch {how}"), "held back", true, 1);
    }
    trace_record_dispatch(t, l, "dropped");
}

/// A dispatch dropped for want of a host compute pipeline; with `VICTORY_TRACE_TARGETS`, counted
/// by its shader, `how` (group counts, or indirect) and the textures its tables hold as
/// unordered-access views (what it may have written) for the frame line.
fn dropped_dispatch(t: &MetalTitle, l: &mut MetalList, how: &str) {
    if crate::pipeline::traces(l.pipeline as This) {
        super::rt::note_ray_dropped(&stage_hashes(l.pipeline).join(" "));
    }
    dropped("a dispatch without a host compute pipeline");
    drop_shaders(l.pipeline, true);
    #[cfg(target_os = "macos")]
    trace_dropped_writes(t, l);
    if pass_on() {
        pass_note(t, l, &format!("dispatch {how}"), "a dispatch without a host compute pipeline", true, 1);
    }
    trace_record_dispatch(t, l, "dropped");
    if tracing_targets() {
        let uavs: Vec<String> = t
            .host
            .table_uav_textures(l.list.compute_args())
            .into_iter()
            .map(|r| {
                let i = crate::resource::texture_info(r as This);
                let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
                format!("{r:#x} fmt {f} {w}x{h}")
            })
            .collect();
        trace_drop_pipe(format!("{:?} {how} uavs {uavs:?}", stage_hashes(l.pipeline)));
    }
}

/// `VICTORY_TRACE_GPU_VERTICES`: what a dispatch dropped for want of a host compute pipeline
/// may have written this frame, for the draws that read it (`gpu_vertex_trace`).
#[cfg(target_os = "macos")]
fn trace_dropped_writes(t: &MetalTitle, l: &MetalList) {
    if super::gpu_vertex_trace::on() {
        super::gpu_vertex_trace::dropped_dispatch(&t.host, l.list.compute_args(), &stage_hashes(l.pipeline).join(" "));
    }
}

/// The stages of title pipeline `pipeline` with their shader hashes (for traces).
fn stage_hashes(pipeline: usize) -> Vec<String> {
    crate::pipeline::native_stages(pipeline as This)
        .map(|(stages, _)| stages.iter().map(|(s, sh)| format!("{s} {}", sh.hash)).collect())
        .unwrap_or_default()
}

/// Records the ambient stand-in's pass after the dispatch just recorded when that dispatch writes
/// the title's indirect diffuse light (`crate::ambient`, `VICTORY_AMBIENT=1`). The light is found
/// by the unordered view of an R9G9B9E5 image the dispatch binds, not by the program's name: the
/// console's containers carry none. The depth image beside it is not used by this slice.
fn record_ambient(
    t: &mut MetalTitle,
    l: &mut MetalList,
    binding: Option<&super::list::Binding>,
) {
    if !crate::ambient::on() {
        return;
    }
    let lights = crate::ambient::bound_lights(l.list.compute_args(), binding, |resource| {
        let info = crate::resource::texture_info(resource as This)?;
        Some((info.format, info.width, info.height))
    });
    let mut seen: Vec<usize> = Vec::new();
    for light in lights {
        if seen.contains(&light.resource) {
            continue;
        }
        seen.push(light.resource);
        static MATCHED: std::sync::Once = std::sync::Once::new();
        MATCHED.call_once(|| {
            tracing::info!(
                resource = format!("{:#x}", light.resource),
                view_format = light.view_format,
                resource_format = light.resource_format,
                slot = light.slot,
                offset = light.offset,
                width = light.size[0],
                height = light.size[1],
                shader = ?stage_hashes(l.pipeline),
                "ambient light: the first dispatch that writes it"
            );
        });
        let Some(texture) = t.texture_of(light.resource) else {
            continue;
        };
        // What the pass writes is the GPU's from now on (no upload from title memory).
        t.gpu_written(light.resource);
        let pass = crate::ambient::Pass {
            level: crate::ambient::level(),
            tint: crate::ambient::tint(),
            size: light.size,
        };
        l.list.ambient(&texture, crate::ambient::constants(&pass));
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                width = light.size[0],
                height = light.size[1],
                format = light.view_format,
                slot = light.slot,
                offset = light.offset,
                shader = ?stage_hashes(l.pipeline),
                "ambient light: the first image it writes"
            );
        });
    }
}

/// `DispatchX` with thread counts (`crate::list::dispatch_x`). Without the translation for
/// them it is dropped: run as group counts, its lanes past the counts would write over memory
/// that is not theirs.
pub(crate) fn dispatch_threads(this: This, threads: [u32; 3]) {
    with_list(this, |t, l| {
        if let Some(why) = builder_hold(l.pipeline) {
            builder_dropped(t, l, &format!("threads{threads:?}"), &why);
            return;
        }
        match thread_pipe(t, l.pipeline) {
            Pipe::Compute(pso) => {
                if crate::pipeline::is_builder(l.pipeline as This) {
                    super::rt::note_builder(threads, true);
                }
                if crate::pipeline::traces(l.pipeline as This) {
                    super::rt::note_ray_direct(&stage_hashes(l.pipeline).join(" "), threads);
                }
                if pass_on() {
                    pass_note(t, l, &format!("dispatch-threads{threads:?}"), "ran", true, 1);
                }
                l.list.set_compute_pipeline(&pso);
                if let Err(e) = l.list.dispatch_threads(threads) {
                    tracing::warn!(error = %e, "dispatch of thread counts");
                }
                record_ambient(t, l, pso.binding.as_ref());
            }
            _ => {
                dropped("a dispatch of thread counts without a host compute pipeline");
                drop_shaders(l.pipeline, true);
                #[cfg(target_os = "macos")]
                trace_dropped_writes(t, l);
                if pass_on() {
                    pass_note(t, l, &format!("dispatch-threads{threads:?}"), "a dispatch of thread counts without a host compute pipeline", true, 1);
                }
                if threads.iter().all(|&n| n != 0) {
                    record_ambient(t, l, None);
                }
            }
        }
    });
}

/// `IASetPrimitiveTopology`.
pub(crate) fn topology(this: This, topology: u32) {
    with_list(this, |_, l| l.topology = topology);
}

/// `RSSetViewports`: the first one.
pub(crate) fn viewports(this: This, first: Option<[f32; 6]>) {
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(?first, "viewport set");
        }
    }
    with_list(this, |_, l| {
        l.list.set_viewport(first.map(|v| victory_backend::resources::Viewport {
            x: v[0],
            y: v[1],
            width: v[2],
            height: v[3],
            min_depth: v[4],
            max_depth: v[5],
        }))
    });
}

/// `RSSetScissorRects`: the first one (left, top, right, bottom).
pub(crate) fn scissors(this: This, first: Option<[i32; 4]>) {
    with_list(this, |_, l| l.list.set_scissor(first));
}

pub(crate) fn blend_factor(this: This, factor: Option<[f32; 4]>) {
    with_list(this, |_, l| l.list.set_blend_factor(factor.unwrap_or([1.0; 4])));
}

pub(crate) fn stencil_ref(this: This, reference: u32) {
    with_list(this, |_, l| l.list.set_stencil_ref(reference));
}

/// `IASetIndexBuffer`: the buffer's address, its view size and whether its indices are 32 bits.
/// An empty view (a null address or no bytes) clears the binding.
pub(crate) fn index_buffer(this: This, view: Option<(u64, u32, bool)>) {
    with_list(this, |_, l| match view {
        Some((address, size, wide)) if address != 0 && size != 0 => {
            l.list.set_index_buffer_view(address, size, wide)
        }
        _ => l.list.clear_index_buffer(),
    });
}

/// Title ABI for `D3D12_VERTEX_BUFFER_VIEW`.
#[repr(C)]
pub(crate) struct VertexBufferView {
    address: u64,
    size: u32,
    stride: u32,
}

pub(crate) fn vertex_buffers(this: This, start: u32, count: u32, views: &[VertexBufferView]) {
    with_list(this, |_, l| {
        for i in 0..count as usize {
            let slot = start as usize + i;
            let (address, size, stride) = match views.get(i) {
                Some(v) if v.address != 0 && v.size != 0 => (v.address, v.size, v.stride),
                _ => (0, 0, 0),
            };
            l.list.set_vertex_buffer_view(slot, address, size);
            if l.strides.len() <= slot {
                l.strides.resize(slot + 1, 0);
            }
            l.strides[slot] = stride;
        }
    });
}

/// The host pipeline of a guessed pipeline for the targets and vertex buffers the list has bound.
fn guess(t: &mut MetalTitle, l: &MetalList, g: &Guessed<Metal>, rect: bool) -> Pipe<Metal> {
    let key = format!("{:?}{rect}", l.strides);
    if let Some(p) = g.made.borrow().get(&key) {
        return p.clone();
    }
    // As in `pipe`: for a shader the prewarm pool has, a translation not made yet goes to a worker
    // and the draw is dropped this time (nothing is cached; the worker's result is, on disk or in
    // the shader). The frame must not translate one inline: it is unbounded, and the title's
    // watchdog waits only 30 s. A shader the pool does not have (the pool is off) is translated
    // here, as before.
    if rect && !g.vs_source.rect_ready() && g.vs_source.pool_has_it() {
        let s = g.vs_source.clone();
        put_off(format!("{} rect", s.hash), move || drop(s.native_rect()));
        return Pipe::Refused;
    }
    let rect_vs = if rect {
        match g.vs_source.native_rect() {
            Ok(h) => Some(h.clone()),
            Err(e) => {
                t.refused_translation(&g.name, format!("rectangle list: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    // Parameters the pixel shader reads and the vertex shader does not write: written (undefined)
    // too, as the converter refuses a pipeline whose fragment inputs the vertex stage lacks.
    let base_vs = rect_vs.as_deref().unwrap_or(&g.vs);
    let extra = g.ps.as_ref().map_or(0, |ps| {
        super::state::texcoords(&ps.dxil, victory_shader::FourCC::ISG1)
            & !super::state::texcoords(&base_vs.dxil, victory_shader::FourCC::OSG1)
    });
    let widened = if extra != 0 {
        let mut copy = [0u8; 32];
        for &(orig, new) in g.ps.as_ref().map_or(&[][..], |ps| &ps.splits[..]) {
            if orig < 32 && new < 32 {
                copy[new as usize] = orig as u8 + 1;
            }
        }
        if g.vs_source.pool_has_it() && !g.vs_source.extra_cached(rect, extra, copy) {
            let s = g.vs_source.clone();
            put_off(format!("{} {rect} {extra} {copy:?}", s.hash), move || drop(s.translate_extra(rect, extra, copy)));
            return Pipe::Refused;
        }
        match g.vs_source.translate_extra(rect, extra, copy) {
            Ok(h) => Some(h),
            Err(e) => {
                t.refused_translation(&g.name, format!("extra outputs: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    let vs = widened.as_deref().unwrap_or(base_vs);
    let reads = super::state::vertex_inputs(&vs.dxil).unwrap_or_default();
    let state = g.state.for_draw(l.targets.color.clone(), l.targets.depth, l.strides.clone(), &reads, rect);
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let made = t.render_pipe(vs, g.ps.as_deref(), &state, &g.name);
    // Put off, it is asked for again at every draw until Metal has made it: logged once made.
    if t.shaders.put_off == put_off {
        tracing::info!(name = %g.name, ?reads, targets = ?l.targets, strides = ?l.strides, "guessed pipeline");
        if tracing_targets() {
            tracing::info!(name = %g.name, ?state, depth_test = ?state.depth_test(), stencil = ?state.stencil(), cull = ?state.cull(), color = ?state.color_state(), "guessed pipeline state");
        }
        g.made.borrow_mut().insert(key, made.clone());
    }
    made
}

/// Sets the draw pipeline, topology and bindings. Returns rectangle mode for a drawable pipeline.
/// `mesh`: the call is a mesh dispatch, which only a mesh pipeline runs (and a draw only a vertex
/// pipeline).
fn graphics(t: &mut MetalTitle, l: &mut MetalList, mesh: bool) -> Option<bool> {
    if !pass_on() {
        return graphics_inner(t, l, mesh);
    }
    LAST_WHY.with(|w| w.set(""));
    let r = graphics_inner(t, l, mesh);
    let why = LAST_WHY.with(|w| w.get());
    let out = match r {
        Some(true) => "ran-rect",
        Some(false) => "ran",
        None if why.is_empty() => "dropped: a mesh pipeline without a vertex stage",
        None => why,
    };
    pass_note(t, l, if mesh { "mesh" } else { "draw" }, out, false, 1);
    r
}

fn graphics_inner(t: &mut MetalTitle, l: &mut MetalList, mesh: bool) -> Option<bool> {
    if !l.targets_ready {
        dropped("a draw of a pass whose render targets the host cannot make");
        trace_drop(l);
        return None;
    }
    // A rectangle list (topology 17) of a guessed pipeline is drawn as instances of four
    // vertices, a triangle strip, each instance a rectangle (`Options::rect_list`). A described
    // pipeline has no fourth-corner shader translation, so it falls back to its supplied vertices
    // drawn as triangles rather than issuing 4-vertex instanced strips.
    let rect = l.topology == 17;
    // `VICTORY_TRACE_GAP_VIEWS=1`: once a pipeline, whether a draw binds a heap holding buffer
    // views the host shows as a wrap of mapped title pages (or, before the prefix fix, as the null
    // view), with the pipeline's shaders -- which says whether the world's GPU-driven programs
    // read the streamed buffers whose ranges have gaps.
    if env_set!("VICTORY_TRACE_GAP_VIEWS") {
        static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let key = l.pipeline as usize;
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 400 && !seen.contains(&key) {
            if let Some(h) = l.list.bound_heaps().0 {
                let views = h.title_views.borrow();
                if !views.is_empty() {
                    seen.push(key);
                    let wraps: Vec<(usize, u64, u64)> = views
                        .iter()
                        .map(|(&i, &(a, l, _))| (i, a, l))
                        .take(8)
                        .collect();
                    tracing::info!(
                        pipeline = key,
                        shaders = ?stage_hashes(l.pipeline),
                        heap = format!("{:#x}", h.start),
                        entries = h.count,
                        wrapped_views = views.len(),
                        ?wraps,
                        "a draw binds a heap with buffer views over title pages"
                    );
                }
            }
        }
    }
    let targets = l.targets.clone();
    let (made, rect) = match pipe(t, l.pipeline, &targets, &l.strides) {
        Pipe::Guess(g) => (guess(t, l, &g, rect), rect),
        p if rect => {
            if env_set!("VICTORY_TRACE_RECTS") {
                tracing::info!(pipeline = ?l.pipeline, kind = ?std::mem::discriminant(&p), "rect draw of a pipeline that is not guessed");
            }
            warn_once("a rectangle-list draw of a described pipeline, drawn without its fourth corner");
            (p, false)
        }
        Pipe::Skipped => {
            warn_once("a draw of a guessed pipeline without a vertex stage (a mesh pipeline)");
            trace_reason("a mesh pipeline");
            trace_drop(l);
            return None;
        }
        p => (p, false),
    };
    let Pipe::Render(pso, binding) = made else {
        dropped("a draw without a host render pipeline (refused or not translated)");
        drop_shaders(l.pipeline, false);
        trace_drop(l);
        if tracing_targets() {
            trace_drop_pipe(format!("{:?} into {:?}", stage_hashes(l.pipeline), l.targets));
        }
        return None;
    };
    if pso.mesh_threads().is_some() != mesh {
        dropped(if mesh { "a mesh dispatch of a vertex pipeline" } else { "a draw of a mesh pipeline" });
        trace_drop(l);
        return None;
    }
    let topology = if rect {
        Some(victory_backend::resources::Topology::TriangleStrip)
    } else if l.topology == 17 || mesh {
        // Described rectangle lists draw each 3-corner rectangle as a triangle; a mesh shader
        // declares its own topology and ignores the list's.
        Some(victory_backend::resources::Topology::Triangles)
    } else {
        super::state::topology(l.topology)
    };
    let Some(topology) = topology else {
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if l.topology < 64 && SEEN.fetch_or(1 << l.topology, Ordering::Relaxed) >> l.topology & 1 == 0 {
            tracing::warn!(topology = l.topology, "a draw dropped for its primitive topology");
        }
        dropped("a draw of a primitive topology the host does not draw");
        trace_drop(l);
        return None;
    };
    l.list.set_graphics_pipeline(&pso, topology);
    l.list.set_graphics_binding(binding);
    Some(rect)
}

pub(crate) fn draw(this: This, vertices: u32, instances: u32, first: u32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        let late = START.get_or_init(std::time::Instant::now).elapsed().as_secs() >= 26;
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || (late && n % 7 == 0 && n < 4000) {
            with_list(this, |_, l| {
                let mut head = [0u8; 72];
                let at = l.list.vertex_buffer_address(0);
                let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                tracing::info!(vertices, instances, first, topology = l.topology, vb0 = ?at, ok, head = ?head, strides = ?l.strides, formats = ?l.targets, "draw");
            });
        }
    }
    // Debugging: only indexed draws (the content), not the boot passes.
    if env_set!("VICTORY_DEBUG_ONLY_INDEXED") {
        return;
    }
    with_list(this, |t, l| {
        if let Some(rect) = graphics(t, l, false) {
            if rect && env_set!("VICTORY_TRACE_RECTS") {
                static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
                let mut seen = SEEN.lock().unwrap();
                if seen.iter().filter(|&&p| p == l.pipeline as usize).count() < 3 {
                    seen.push(l.pipeline as usize);
                    let stride = l.strides.first().copied().unwrap_or(0) as usize;
                    let mut head = vec![0u8; (stride * 3).clamp(12, 192)];
                    let at = l.list.vertex_buffer_address(0).map(|a| a + (first as u64) * stride as u64);
                    let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                    let floats: Vec<f32> = head.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    tracing::info!(pipeline = ?l.pipeline, vertices, first, strides = ?l.strides, ok, ?floats, "rect draw");
                }
            }
            let call = if rect {
                l.list.draw_instanced(4, vertices / 3, 0, first / 3)
            } else {
                dump_vertices(
                    l,
                    DrawCall::Arrays {
                        vertex_count: vertices,
                        instance_count: instances,
                        first_vertex: first,
                        first_instance,
                    },
                );
                l.list.draw_instanced(vertices, instances, first, first_instance)
            };
            match call {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    trace_present_draw(t, l);
                    l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                }
                Err(e) => tracing::warn!(error = %e, "draw"),
            }
        }
    });
}

/// A direct mesh dispatch (`DispatchMesh`, or `DispatchMeshX` without an argument address): the
/// bound mesh pipeline runs `[x, y, z]` groups, as an indirect one runs its record's (#353). With
/// no mesh pipeline it is dropped for the reason `graphics` gives, and counted in the frame line's
/// `mesh=` as before.
pub(crate) fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    with_list(this, |t, l| {
        if graphics(t, l, true).is_none() {
            crate::list::DROPPED_KINDS[0].fetch_add(1, Ordering::Relaxed);
            return;
        }
        if pass_on() {
            pass_note_op(l, &format!("mesh[{x},{y},{z}]"), None, None);
        }
        match l.list.dispatch_mesh(x, y, z) {
            Ok(()) => {
                ENCODED.fetch_add(1, Ordering::Relaxed);
                l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                trace_present_draw(t, l);
            }
            Err(e) => tracing::warn!(error = %e, "mesh dispatch"),
        }
    });
}

/// `VICTORY_DUMP_VERTICES=<dir>,<draws>[,<seconds>]`: the next draws' vertex and index data, from
/// `seconds` after the first draw, read the way the input assembler reads it (`geometry_trace`),
/// one text file a draw.
fn dump_vertices(l: &MetalList, call: DrawCall) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let Some((dir, draws, after)) = super::geometry_trace::config() else { return };
    let start = START.get_or_init(std::time::Instant::now);
    if start.elapsed().as_secs() < after {
        return;
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n >= draws {
        return;
    }
    super::geometry_trace::dump(
        &dir,
        n,
        &super::geometry_trace::Bound {
            pipeline: l.pipeline,
            topology: (l.topology, super::state::topology(l.topology).map(topology_name)),
            strides: &l.strides,
            call,
            targets: &l.targets,
            vertex_views: l.list.vertex_views(),
            index_view: l.list.index_view(),
        },
    );
}

/// `topology`'s name for a trace (`Triangles`, ...).
fn topology_name(t: victory_backend::resources::Topology) -> &'static str {
    use victory_backend::resources::Topology::*;
    match t {
        Points => "points",
        Lines => "lines",
        LineStrip => "line strip",
        Triangles => "triangles",
        TriangleStrip => "triangle strip",
    }
}

/// `VICTORY_TRACE_INDEX`'s value: how many direct indexed draws to trace, and how many seconds
/// after the first one to start (`None`: off). The seconds let a trace of the world skip the
/// boot's and the menu's draws, which would spend the count before the content is drawn.
fn index_trace_config(v: &str) -> Option<(u32, u64)> {
    let mut parts = v.split(',');
    let limit: u32 = parts.next()?.trim().parse().ok()?;
    let after: u64 = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    (limit > 0).then_some((limit, after))
}

/// `VICTORY_TRACE_INDEX=<draws>[,<seconds>]`: for the first `<draws>` direct indexed draws from
/// `<seconds>` after the first one, reads back the indices the draw will fetch, at the width the
/// host bound and at the other one, and the vertices each bound stream can hold (`view size /
/// stride`). A 32-bit index buffer the host reads as 16 bits (or the reverse) draws a mesh as a
/// lattice of long thin triangles: every index becomes two 16-bit ones, and each pair joins
/// vertices far apart in the mesh. One line a draw.
fn trace_index(l: &MetalList, indices: u32, first: u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CONFIG: std::sync::OnceLock<Option<(u32, u64)>> = std::sync::OnceLock::new();
    let Some((limit, after)) = *CONFIG.get_or_init(|| {
        index_trace_config(&std::env::var("VICTORY_TRACE_INDEX").ok()?)
    }) else {
        return;
    };
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    if START.get_or_init(std::time::Instant::now).elapsed().as_secs() < after {
        return;
    }
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n >= limit || indices == 0 {
        return;
    }
    let Some((addr, size, wide)) = l.list.index_view() else { return };
    let width = if wide { 4u64 } else { 2 };
    let at = addr + u64::from(first) * width;
    let sample = (indices as usize).min(1 << 16);
    let bytes = super::textures::read_memory(at as *const u8, sample * width as usize);
    let mut max16 = 0u32;
    for c in bytes.chunks_exact(2) {
        max16 = max16.max(u32::from(u16::from_le_bytes([c[0], c[1]])));
    }
    let mut max32 = 0u32;
    for c in bytes.chunks_exact(4) {
        max32 = max32.max(u32::from_le_bytes(c.try_into().expect("4 bytes")));
    }
    let views: Vec<String> = l
        .list
        .vertex_views()
        .into_iter()
        .map(|(slot, a, size)| {
            let stride = l.strides.get(slot).copied().unwrap_or(0);
            let capacity = if stride > 0 { u64::from(size) / u64::from(stride) } else { 0 };
            format!("{slot}@{a:#x}+{size} stride {stride} vertices {capacity}")
        })
        .collect();
    tracing::info!(
        draw = n,
        indices,
        first,
        wide,
        index_size = size,
        index_at = format!("{addr:#x}"),
        max_u16 = max16,
        max_u32 = max32,
        ?views,
        "a draw's index data read back"
    );
}

pub(crate) fn draw_indexed(this: This, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    with_list(this, |t, l| {
        if let Some(rect) = graphics(t, l, false) {
            if rect {
                // shortcut: indexed rectangle lists are not drawn.
                dropped("an indexed rectangle-list draw");
                if pass_on() {
                    pass_note_op(l, "DROPPED-indexed-rect-list", None, None);
                }
                trace_drop(l);
                return;
            }
            if let Some(view) = l.list.index_view() {
                dump_vertices(
                    l,
                    DrawCall::Indexed {
                        index_count: indices,
                        instance_count: instances,
                        first_index: first,
                        base_vertex: base,
                        first_instance,
                        wide: view.2,
                    },
                );
            }
            match l.list.draw_indexed_instanced(indices, instances, first, base, first_instance) {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    trace_index(l, indices, first);
                    trace_present_draw(t, l);
                    l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                }
                Err(e) => tracing::warn!(error = %e, "indexed draw"),
            }
        }
    });
}

/// `ExecuteIndirect` of a single-command signature: `max` draws or dispatches, record `i` at title
/// address `records + i * stride`, each read on the GPU; with a count buffer at title address
/// `count`, record `i` runs only if `i` is below the count, also read on the GPU (D3D12 runs
/// min(count, max) records). False when the host cannot run it; the caller logs
/// that. Indexed rectangle lists drop here.
pub(crate) fn execute_indirect(
    this: This,
    command: crate::signature::Command,
    stride: u32,
    max: u32,
    records: u64,
    count: Option<u64>,
) -> bool {
    use crate::signature::Command;
    let at = |i: u32| records + u64::from(i) * u64::from(stride);
    match command {
        Command::DispatchMesh => {
            RECORDED.fetch_add(u64::from(max), Ordering::Relaxed);
            with_list(this, |t, l| {
                if graphics(t, l, true).is_none() {
                    return;
                }
                if pass_on() {
                    pass_note_op(l, &format!("mesh-records n={max} counted={}", count.is_some()), None, None);
                }
                if tracing_record_writers() && record_writers_of().is_none() {
                    let zero = (0..max)
                        .filter(|&i| super::textures::read_memory(at(i) as *const u8, 4).iter().all(|&b| b == 0))
                        .count() as u64;
                    let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
                    let n = trace.get_or_insert_default().executions.entry(at(0)).or_default();
                    n[0] += u64::from(max);
                    n[1] += zero;
                }
                for i in 0..max {
                    match l.list.dispatch_mesh_indirect(at(i), count.map(|c| (c, i))) {
                        Ok(()) => {
                            ENCODED.fetch_add(1, Ordering::Relaxed);
                            l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                        }
                        Err(e) => tracing::warn!(error = %e, "indirect mesh dispatch"),
                    }
                }
                trace_present_draw(t, l);
            });
            true
        }
        Command::Dispatch => {
            with_list(this, |t, l| {
                if let Some(why) = builder_hold(l.pipeline) {
                    builder_dropped(t, l, "indirect", &why);
                    return;
                }
                match pipe(t, l.pipeline, &TargetFormats::default(), &[]) {
                    Pipe::Compute(pso) => {
                        trace_record_dispatch(t, l, "ran");
                        if pass_on() {
                            pass_note(t, l, "dispatch-indirect", "ran", true, max);
                        }
                        l.list.set_compute_pipeline(&pso);
                        let traced = crate::pipeline::traces(l.pipeline as This).then(|| stage_hashes(l.pipeline).join(" "));
                        for i in 0..max {
                            if let Some(of) = record_writers_of()
                                && stage_hashes(l.pipeline).iter().any(|h| h.strip_prefix("compute ").is_some_and(|h| of.contains(h)))
                            {
                                let zero = super::textures::read_memory(at(i) as *const u8, 4).iter().all(|&b| b == 0);
                                let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
                                let n = trace.get_or_insert_default().executions.entry(at(i)).or_default();
                                n[0] += 1;
                                n[1] += u64::from(zero);
                            }
                            if crate::list::trace_dispatch_flags() {
                                static SEEN: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::new());
                                let mut seen = SEEN.lock().unwrap();
                                let key = (l.pipeline as usize, count.is_some());
                                if seen.iter().filter(|k| **k == key).count() < 3 {
                                    seen.push(key);
                                    // What title memory holds now, not what the GPU will read.
                                    let now: Vec<u32> = super::textures::read_memory(at(i) as *const u8, 12)
                                        .chunks_exact(4)
                                        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                                        .collect();
                                    tracing::warn!(shader = ?stage_hashes(l.pipeline), ?command, counted = count.is_some(), stride, record = format_args!("{:#x}", at(i)), ?now, "an indirect dispatch (the first three of each pipeline)");
                                }
                            }
                            if let Some(name) = &traced {
                                let record: Vec<u32> = super::textures::read_memory(at(i) as *const u8, 12)
                                    .chunks_exact(4)
                                    .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                                    .collect();
                                super::rt::note_ray_indirect(name, [record[0], record[1], record[2]]);
                            }
                            let r = match count {
                                Some(c) => l.list.dispatch_indirect_counted(at(i), c, i),
                                None => l.list.dispatch_indirect(at(i)),
                            };
                            if let Err(e) = r {
                                tracing::warn!(error = %e, "indirect dispatch");
                            }
                        }
                        record_ambient(t, l, pso.binding.as_ref());
                    }
                    _ => {
                        dropped_dispatch(t, l, "indirect");
                        record_ambient(t, l, None);
                    }
                }
            });
            true
        }
        Command::Draw | Command::DrawIndexed => {
            RECORDED.fetch_add(u64::from(max), Ordering::Relaxed);
            with_list(this, |t, l| match graphics(t, l, false) {
                // shortcut: an indexed rectangle list is not drawn, as directly.
                Some(true) if command == Command::DrawIndexed => {
                    dropped("an indirect rectangle-list draw");
                    if pass_on() {
                        pass_note_op(l, &format!("DROPPED-indirect-rect-list n={max}"), None, None);
                    }
                    trace_drop(l);
                    true
                }
                Some(rect) => {
                    for i in 0..max {
                        let indexed = command == Command::DrawIndexed;
                        let r = match count {
                            // As a direct one: four-vertex instances, the record rewritten on the GPU.
                            _ if rect => l.list.draw_rects_indirect(at(i), count.map(|c| (c, i))),
                            Some(c) => l.list.draw_indirect_counted(at(i), indexed, c, i),
                            None => l.list.draw_indirect(at(i), indexed),
                        };
                        match r {
                            Ok(()) => {
                                ENCODED.fetch_add(1, Ordering::Relaxed);
                                l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                            }
                            Err(e) => tracing::warn!(error = %e, "indirect draw"),
                        }
                    }
                    true
                }
                None => true,
            })
            .unwrap_or(true)
        }
    }
}

pub(crate) fn root_constants(this: This, compute: bool, slot: u32, offset: u32, words: &[u32]) {
    with_list(this, |_, l| {
        l.list
            .set_root_constants_at(compute, slot as usize, offset as usize, words)
    });
}

pub(crate) fn root_view(this: This, compute: bool, slot: u32, address: u64) {
    with_list(this, |_, l| {
        if compute {
            l.list.set_compute_root_view(slot as usize, address)
        } else {
            l.list.set_graphics_root_view(slot as usize, address)
        }
    });
}

pub(crate) fn root_table(this: This, compute: bool, slot: u32, handle: u64) {
    with_list(this, |t, l| {
        let Some((heap, index)) = heap_of(t, handle) else {
            return;
        };
        if compute {
            l.list.set_compute_root_table(slot as usize, &heap, index)
        } else {
            l.list.set_graphics_root_table(slot as usize, &heap, index)
        }
    });
}

/// `SetDescriptorHeaps`: the title's heap objects.
pub(crate) fn set_heaps(this: This, heaps: &[This]) {
    with_list(this, |t, l| {
        let (mut views, mut samplers) = (None, None);
        for &h in heaps {
            let Some((start, sampler, _)) = crate::descriptor::span(h) else {
                continue;
            };
            let Some((heap, _)) = heap_of(t, start) else {
                continue;
            };
            if sampler {
                samplers = Some(heap);
            } else {
                views = Some(heap);
            }
        }
        l.list.set_heaps(views.as_ref(), samplers.as_ref());
    });
}

/// `CopyBufferRegion` between title buffers.
pub(crate) fn copy_buffer(this: This, dst: (This, u64), src: (This, u64), len: u64) {
    let (Some((to, to_size)), Some((from, from_size))) = (
        crate::resource::buffer_bytes(dst.0),
        crate::resource::buffer_bytes(src.0),
    ) else {
        return;
    };
    if dst.1.saturating_add(len) > to_size || src.1.saturating_add(len) > from_size {
        tracing::warn!(len, "copy outside its buffer");
        return;
    }
    // SAFETY: inside the buffers, checked above.
    let (to, from) = unsafe { (to.add(dst.1 as usize), from.add(src.1 as usize)) };
    copy_memory(this, to as u64, from as u64, len);
}

/// `CopyMemoryToMemoryX` and buffer copies: title addresses.
pub(crate) fn copy_memory(this: This, dst: u64, src: u64, len: u64) {
    with_list(this, |_, l| {
        l.list.copy_buffer_region(dst, src, len as usize);
    });
}

fn subresource_mip_slice(info: &super::textures::TextureInfo, subresource: u32) -> Option<(usize, usize)> {
    let mips = info.layout.mips.len().max(1);
    let slices = if info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        info.layout.slices.max(1) as usize
    };
    let sub = subresource as usize;
    (sub < mips * slices).then(|| (sub % mips, sub / mips))
}

/// The host texture's level for title level `mip`: `None` for a top level the host texture was
/// made without (`VICTORY_TEXTURE_DROP_TOP_MIP`). Host level `mip - dropped` has the title level's
/// size. Shortcut: a copy to or from a dropped level is skipped, not scaled; the switch is off by
/// default.
fn host_mip(dropped: usize, mip: usize) -> Option<usize> {
    mip.checked_sub(dropped)
}

fn mip_dims(info: &super::textures::TextureInfo, mip: usize) -> [usize; 3] {
    [
        (info.width as usize >> mip).max(1),
        (info.height.max(1) as usize >> mip).max(1),
        info.layout.subresource_depth(mip as u32).max(1) as usize,
    ]
}

fn copy_box(
    info: &super::textures::TextureInfo,
    mip: usize,
    source_box: Option<[u32; 6]>,
) -> Option<([usize; 3], [usize; 3])> {
    let dims = mip_dims(info, mip);
    let Some([l, t, f, r, b, back]) = source_box else {
        return Some(([0, 0, 0], dims));
    };
    if r < l || b < t || back < f {
        return None;
    }
    let origin = [l as usize, t as usize, f as usize];
    let size = [(r - l) as usize, (b - t) as usize, (back - f) as usize];
    (0..3).all(|i| origin[i].saturating_add(size[i]) <= dims[i]).then_some((origin, size))
}

fn footprint_span(
    res: This,
    fp: &crate::resource::PlacedFootprint,
    at: [u32; 3],
    size: [u32; 3],
) -> Option<(u64, usize, usize, usize)> {
    // 115 (B4G4R4A4_UNORM) needs CPU nibble rotation; depth formats need plane-separated copies.
    if fp.format == 115 {
        return None;
    }
    let fmt = super::descriptors::format(fp.format)?;
    // A depth and stencil footprint names both planes in one element; the host copies the depth
    // plane alone (as a 4-byte format: `R32_TYPELESS` and the like).
    if fmt == victory_backend::resources::Format::D32FloatS8 {
        return None;
    }
    let (_, block) = crate::resource::element(fp.format)?;
    let (base, buf_size) = crate::resource::buffer_bytes(res)?;
    let (start, end) = fp.span(at, size)?;
    let byte_end = fp.offset.checked_add(end)?;
    if byte_end > buf_size {
        return None;
    }
    let addr = (base as u64).checked_add(fp.offset + start)?;
    let len = usize::try_from(end - start).ok()?;
    let row_pitch = fp.row_pitch as usize;
    let slice_pitch = row_pitch.checked_mul((fp.height as usize).div_ceil(block as usize))?;
    Some((addr, len, row_pitch, slice_pitch))
}

/// Texture copies recorded so far, and since the last present (the `VICTORY_TRACE_TARGETS` frame
/// line).
static COPIES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A texture copy was recorded: counted, logged once per kind with `VICTORY_TRACE_TARGETS`, and its
/// destination texture (if any) is the GPU's from now on ([`Title::gpu_written`]).
fn copied(t: &mut MetalTitle, what: &str, src: This, dst: Option<This>) {
    COPIES[0].fetch_add(1, Ordering::Relaxed);
    COPIES[1].fetch_add(1, Ordering::Relaxed);
    if let Some(d) = dst {
        t.gpu_written(d as usize);
        trace_bump(d as usize, 0);
    }
    if tracing_targets() {
        static KINDS: Mutex<Vec<(String, Option<victory_backend::resources::Format>, Option<victory_backend::resources::Format>)>> = Mutex::new(Vec::new());
        let kind = (what.to_owned(), t.format_of(src as usize), dst.and_then(|d| t.format_of(d as usize)));
        let mut kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
        if !kinds.contains(&kind) {
            tracing::info!(what, src = ?kind.1, dst = ?kind.2, "a texture copy (first of its kind)");
            kinds.push(kind);
        }
    }
}

/// Whether texels of `src` copy into `dst` on the host: one blit (the same format, or plain
/// formats of one size through a view), or through a buffer, formats of one block size where one
/// is block-compressed or depth (plain texels into a GPU encoder's compressed blocks, a depth
/// plane into a 4-byte float), as D3D12 copies them.
fn copy_formats(src: victory_backend::resources::Format, dst: victory_backend::resources::Format) -> bool {
    use victory_backend::resources::Format;
    let bytes = |f: Format| if f == Format::D32FloatS8 { 4 } else { f.block().1 };
    let special = |f: Format| f.block().0 > 1 || f.is_depth();
    victory_metal::gfx::direct_copy(src, dst) || ((special(src) || special(dst)) && bytes(src) == bytes(dst))
}

/// `size` source texels in the destination's texels: a plain texel is a whole compressed block
/// (cut at the destination's edge, from `at`), and a block one texel.
fn dest_size(src: victory_backend::resources::Format, dst: victory_backend::resources::Format, size: [usize; 3], at: [usize; 3], dims: [usize; 3]) -> [usize; 3] {
    let (sw, dw) = (src.block().0, dst.block().0);
    std::array::from_fn(|i| match i {
        2 => size[2],
        _ if dw > sw => (size[i] * dw).min(dims[i].saturating_sub(at[i])),
        _ if sw > dw => size[i].div_ceil(sw),
        _ => size[i],
    })
}

/// Whether a footprint's rows hold the texels of a texture (a depth and stencil texture's: its
/// 4-byte depth plane).
fn rows_match(fp: &crate::resource::PlacedFootprint, info: &super::textures::TextureInfo) -> bool {
    match (super::descriptors::format(fp.format), info.desc()) {
        (Some(f), Some(d)) => copy_formats(f, d.format),
        _ => false,
    }
}

/// `ResolveSubresource`: host textures have one sample, so a resolve copies what both
/// subresources hold (the console clips a larger source to the destination, as the D3D12 host
/// does); the format argument names a view the copy makes when it needs one.
pub(crate) fn resolve(this: This, dst: This, dst_sub: u32, src: This, src_sub: u32) -> bool {
    let (Some(si), Some(di)) = (crate::resource::texture_info(src), crate::resource::texture_info(dst)) else {
        return false;
    };
    let (Some((sm, _)), Some((dm, _))) = (subresource_mip_slice(&si, src_sub), subresource_mip_slice(&di, dst_sub)) else {
        return false;
    };
    let (a, b) = (mip_dims(&si, sm), mip_dims(&di, dm));
    let region = [0, 0, 0, a[0].min(b[0]) as u32, a[1].min(b[1]) as u32, a[2].min(b[2]) as u32];
    copy_texture_region(this, crate::list::CopyLocation::sub(dst, dst_sub), [0; 3], crate::list::CopyLocation::sub(src, src_sub), Some(region))
}

/// `CopyResource` between two title textures. Returns `false` if the host cannot copy them.
pub(crate) fn copy_texture_resource(this: This, dst: This, src: This) -> bool {
    let (Some(src_info), Some(dst_info)) = (
        crate::resource::texture_info(src),
        crate::resource::texture_info(dst),
    ) else {
        return false;
    };
    let mips = src_info.layout.mips.len().max(1).min(dst_info.layout.mips.len().max(1));
    let src_slices = if src_info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        src_info.layout.slices.max(1) as usize
    };
    let dst_slices = if dst_info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        dst_info.layout.slices.max(1) as usize
    };
    let slices = src_slices.min(dst_slices);
    with_list(this, |t, l| {
        let (Some(src_tex), Some(dst_tex)) = (t.texture_of(src as usize), t.texture_of(dst as usize)) else {
            return false;
        };
        let (Some(sf), Some(df)) = (t.format_of(src as usize), t.format_of(dst as usize)) else {
            return false;
        };
        if !copy_formats(sf, df) {
            return false;
        }
        let (src_drop, dst_drop) = (t.host.dropped_count(src as usize), t.host.dropped_count(dst as usize));
        for slice in 0..slices {
            for mip in 0..mips {
                let (Some(src_mip), Some(dst_mip)) = (host_mip(src_drop, mip), host_mip(dst_drop, mip)) else {
                    continue;
                };
                let s = mip_dims(&src_info, mip);
                let d = mip_dims(&dst_info, mip);
                let size = [s[0].min(d[0]), s[1].min(d[1]), s[2].min(d[2])];
                l.list.copy_texture_region(
                    &src_tex,
                    src_mip,
                    slice,
                    [0, 0, 0],
                    size,
                    &dst_tex,
                    dst_mip,
                    slice,
                    [0, 0, 0],
                );
            }
        }
        copied(t, "CopyResource", src, Some(dst));
        if pass_on() {
            pass_note_op(l, "copy-resource", Some(src as usize), Some(dst as usize));
        }
        true
    })
    .unwrap_or(false)
}

/// `CopyTextureRegion` on macOS: texture-to-texture and placed footprint <-> texture.
pub(crate) fn copy_texture_region(
    this: This,
    dst: crate::list::CopyLocation,
    origin: [u32; 3],
    src: crate::list::CopyLocation,
    source_box: Option<[u32; 6]>,
) -> bool {
    const SUBRESOURCE: u32 = 0;
    const FOOTPRINT: u32 = 1;
    let dst_origin = [origin[0] as usize, origin[1] as usize, origin[2] as usize];
    match (dst.kind, src.kind) {
        (SUBRESOURCE, SUBRESOURCE) => {
            let (Some(src_info), Some(dst_info)) = (
                crate::resource::texture_info(src.resource),
                crate::resource::texture_info(dst.resource),
            ) else {
                return false;
            };
            let Some((src_mip, src_slice)) = subresource_mip_slice(&src_info, src.footprint.offset as u32) else {
                return false;
            };
            let Some((dst_mip, dst_slice)) = subresource_mip_slice(&dst_info, dst.footprint.offset as u32) else {
                return false;
            };
            let Some((src_origin, size)) = copy_box(&src_info, src_mip, source_box) else {
                return false;
            };
            let (Some(sf), Some(df)) = (src_info.desc().map(|d| d.format), dst_info.desc().map(|d| d.format)) else {
                return false;
            };
            if !copy_formats(sf, df) {
                return false;
            }
            let dst_dims = mip_dims(&dst_info, dst_mip);
            let to = dest_size(sf, df, size, dst_origin, dst_dims);
            if (0..3).any(|i| dst_origin[i].saturating_add(to[i]) > dst_dims[i]) {
                return false;
            }
            if size.contains(&0) {
                return true;
            }
            with_list(this, |t, l| {
                let (Some(src_tex), Some(dst_tex)) =
                    (t.texture_of(src.resource as usize), t.texture_of(dst.resource as usize))
                else {
                    return false;
                };
                let (Some(src_mip), Some(dst_mip)) = (
                    host_mip(t.host.dropped_count(src.resource as usize), src_mip),
                    host_mip(t.host.dropped_count(dst.resource as usize), dst_mip),
                ) else {
                    return true;
                };
                l.list.copy_texture_region(
                    &src_tex,
                    src_mip,
                    src_slice,
                    src_origin,
                    size,
                    &dst_tex,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                );
                copied(t, "CopyTextureRegion", src.resource, Some(dst.resource));
                if pass_on() {
                    pass_note_op(l, "copy-region", Some(src.resource as usize), Some(dst.resource as usize));
                }
                true
            })
            .unwrap_or(false)
        }
        (SUBRESOURCE, FOOTPRINT) => {
            let Some(dst_info) = crate::resource::texture_info(dst.resource).filter(|i| rows_match(&src.footprint, i)) else {
                return false;
            };
            let Some((dst_mip, dst_slice)) = subresource_mip_slice(&dst_info, dst.footprint.offset as u32) else {
                return false;
            };
            let (src_at, size_u32) = match source_box {
                Some([l, t, f, r, b, back]) if r >= l && b >= t && back >= f => {
                    ([l, t, f], [r - l, b - t, back - f])
                }
                Some(_) => return false,
                None => (
                    [0, 0, 0],
                    [src.footprint.width, src.footprint.height, src.footprint.depth],
                ),
            };
            let size = size_u32.map(|v| v as usize);
            if size.contains(&0) {
                return true;
            }
            let dst_dims = mip_dims(&dst_info, dst_mip);
            if (0..3).any(|i| dst_origin[i].saturating_add(size[i]) > dst_dims[i]) {
                return false;
            }
            let Some((src_addr, len, row_pitch, slice_pitch)) =
                footprint_span(src.resource, &src.footprint, src_at, size_u32)
            else {
                return false;
            };
            with_list(this, |t, l| {
                let Some(dst_tex) = t.texture_of(dst.resource as usize) else {
                    return false;
                };
                let Some(dst_mip) = host_mip(t.host.dropped_count(dst.resource as usize), dst_mip) else {
                    return true;
                };
                l.list.copy_buffer_to_texture(
                    src_addr,
                    len,
                    row_pitch,
                    slice_pitch,
                    size,
                    &dst_tex,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                );
                copied(t, "CopyTextureRegion", src.resource, Some(dst.resource));
                if pass_on() {
                    pass_note_op(l, "copy-region", Some(src.resource as usize), Some(dst.resource as usize));
                }
                true
            })
            .unwrap_or(false)
        }
        (FOOTPRINT, SUBRESOURCE) => {
            let Some(src_info) = crate::resource::texture_info(src.resource).filter(|i| rows_match(&dst.footprint, i)) else {
                return false;
            };
            let Some((src_mip, src_slice)) = subresource_mip_slice(&src_info, src.footprint.offset as u32) else {
                return false;
            };
            let Some((src_origin, size)) = copy_box(&src_info, src_mip, source_box) else {
                return false;
            };
            if size.contains(&0) {
                return true;
            }
            let size_u32 = size.map(|v| v as u32);
            let Some((dst_addr, len, row_pitch, slice_pitch)) =
                footprint_span(dst.resource, &dst.footprint, origin, size_u32)
            else {
                return false;
            };
            with_list(this, |t, l| {
                let Some(src_tex) = t.texture_of(src.resource as usize) else {
                    return false;
                };
                let Some(src_mip) = host_mip(t.host.dropped_count(src.resource as usize), src_mip) else {
                    return true;
                };
                l.list.copy_texture_to_buffer(
                    &src_tex,
                    src_mip,
                    src_slice,
                    src_origin,
                    size,
                    dst_addr,
                    len,
                    row_pitch,
                    slice_pitch,
                );
                copied(t, "CopyTextureRegion", src.resource, None);
                if pass_on() {
                    pass_note_op(l, "copy-to-buffer", Some(src.resource as usize), None);
                }
                true
            })
            .unwrap_or(false)
        }
        _ => false,
    }
}

/// A bottom-of-pipe value write of `value`, `bytes` long.
pub(crate) fn write_value(this: This, address: u64, value: u64, bytes: usize) -> bool {
    if env_set!("VICTORY_TRACE_WRITES") {
        let inside = pools().resolve(address, bytes as u64).is_some();
        tracing::warn!("list write {value:#x} ({bytes} bytes) at {address:#x}, in a pool: {inside}, list {this:p}");
    }
    with_list(this, |_, l| l.list.write_value(address, value, bytes)).is_some()
}

/// A wait for a memory value, recorded at the current position of the list.
pub(crate) fn wait_value(this: This, w: Wait) {
    if env_set!("VICTORY_TRACE_WRITES") {
        tracing::warn!("list wait at {:#x} cmp {} ref {:#x}, list {this:p}", w.address, w.comparison, w.reference);
    }
    with_list(this, |_, l| l.wait(w));
}

/// `Reset`: forgets what was recorded (the host objects it named go with it, under the lock).
pub(crate) fn reset_list(this: This) {
    with_list(this, |_, l| l.reset());
}

/// Drops a list object, whose `Rc`s must be dropped under the lock.
pub(crate) fn locked<R>(f: impl FnOnce() -> R) -> R {
    let mut f = Some(f);
    match with_title(|_| (f.take().expect("once"))()) {
        Some(r) => r,
        None => (f.take().expect("not run"))(),
    }
}

/// What the host adds to the process's memory, for the accounting line (`apps/victory` logs it
/// every 10 s beside the title's own committed memory): the texture copies and their budget, what
/// Metal has allocated (textures, and buffers -- the title memory it wraps counts too), the DXIL
/// the shader translations hold, inside it the copies waiting for a submission to name them, and
/// the process's footprint. `None` before the host exists.
pub struct MemoryParts {
    pub copies: u64,
    pub copies_limit: u64,
    pub metal: u64,
    pub dxil: u64,
    /// Shader containers the title's shaders keep in host memory, and the depth twins' bytes: both
    /// are part of what the memory line calls `other` or `metal_own`.
    pub containers: u64,
    pub twins: u64,
    pub footprint: u64,
    pub deferred: usize,
    /// Why the decoded copies are needed, and how much is sampled from title memory
    /// (`super::textures::DecodedReport`).
    pub decoded: super::textures::DecodedReport,
    /// Bytes whose title pages were given back to the OS (`Title::released_bytes`).
    pub released: u64,
}

pub fn memory_parts() -> Option<MemoryParts> {
    with_title(|t| MemoryParts {
        copies: t.budget.live,
        copies_limit: t.budget.limit,
        metal: victory_backend::backend::ComputeBackend::allocated_bytes(&t.host.backend),
        dxil: crate::pipeline::held_dxil_bytes(),
        containers: crate::pipeline::container_bytes(),
        twins: t.host.twins_bytes(),
        footprint: victory_platform::mem::footprint() as u64,
        deferred: t.deferred.len(),
        decoded: t.decoded_report(),
        released: t.released_bytes(),
    })
}

/// The render-target shapes the title created, largest first by bytes in title memory (see
/// [`super::textures::render_target_shapes`]).
pub fn render_target_shapes() -> Vec<(u64, u64, u32, u32, u32, u32, bool)> {
    super::textures::render_target_shapes()
}

/// Counters of the host for tests and the frame line.
pub fn stats() -> Option<super::title::Stats> {
    with_title(|t| t.stats.clone())
}

#[cfg(test)]
mod index_trace_tests {
    use super::{index_trace_config, strides_signature};

    /// The index trace takes a count, and optionally a delay so a world trace skips the boot's
    /// draws; a zero count or a value that is not a count leaves it off.
    #[test]
    fn the_index_trace_takes_a_count_and_a_delay() {
        assert_eq!(index_trace_config("500"), Some((500, 0)));
        assert_eq!(index_trace_config("12,300"), Some((12, 300)));
        assert_eq!(index_trace_config(" 8 , 470 "), Some((8, 470)));
        // A delay that is not a number starts at once, as the vertex dump's does.
        assert_eq!(index_trace_config("4,later"), Some((4, 0)));
        assert_eq!(index_trace_config("0"), None);
        assert_eq!(index_trace_config("0,300"), None);
        assert_eq!(index_trace_config(""), None);
        assert_eq!(index_trace_config("many"), None);
    }

    /// A host pipeline's vertex descriptor holds its streams' strides, so two draws whose bound
    /// strides step a stream differently get different host pipelines; a slot the layout does not
    /// read, or a stride the assembler would not take (0, or past its limit), changes nothing.
    #[test]
    fn a_pipeline_gets_a_host_pipeline_per_stride_its_draws_bind() {
        // The layout: slot 0's elements end at 32 bytes, slot 1's at 12.
        let record = [(0u32, 32u32), (1, 12)];
        let sig = |strides: &[u32]| strides_signature(&record, strides);
        assert_eq!(sig(&[32, 12]), sig(&[32, 12]));
        // The same PSO drawn with a shorter or a longer vertex view is another pipeline: Metal's
        // descriptor would otherwise keep the stride of whichever draw made it.
        assert_ne!(sig(&[32, 12]), sig(&[24, 12]));
        assert_ne!(sig(&[32, 12]), sig(&[40, 12]));
        // A bound buffer the layout does not read is no variant.
        assert_eq!(sig(&[32, 12, 64]), sig(&[32, 12]));
        // A slot the draw bound no stride for (0), or a junk one past the assembler's limit, steps
        // by the layout's elements, so it is the same pipeline as the layout's own end.
        assert_eq!(sig(&[0, 12]), sig(&[32, 12]));
        assert_eq!(sig(&[0xFFFF_FFFF, 12]), sig(&[32, 12]));
        // A pipeline with no stage-in inputs has one host pipeline whatever is bound.
        assert_eq!(strides_signature(&[], &[32, 12]), 0);
        assert_eq!(strides_signature(&[], &[]), 0);
    }
}

#[cfg(test)]
mod stop_tests {
    use super::{stop_work, stopped};

    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }

    /// In a child process of the next test (`stop_work` is for the process's end and stays set):
    /// once it has run, the host takes no more work.
    #[test]
    fn stopped_child() {
        if std::env::var_os("VICTORY_STOP_CHILD").is_none() {
            return;
        }
        assert!(!stopped(), "the host starts taking work");
        stop_work();
        assert!(stopped(), "and takes none once the stop began");
        // SAFETY: the test child ends here; there is nothing to resume.
        unsafe { _exit(0) }
    }

    #[test]
    fn a_stopped_host_takes_no_more_work() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["stopped_child", "--nocapture"])
            .env("VICTORY_STOP_CHILD", "1")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
    }
}
