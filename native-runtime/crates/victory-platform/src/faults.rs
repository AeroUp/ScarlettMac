//! Title pages the host has taken away, and the title's accesses to them (macOS).
//!
//! `victory_gpu` releases a decoded copy's title pages (`MADV_FREE_REUSABLE`), keeps the decoded
//! Metal texture as the only copy, and makes the pages `PROT_NONE`, so the OS may reclaim them
//! instead of swapping. The title still owns the range: a streamed mip's store, a copy out of it,
//! any CPU read faults.
//!
//! The fault path is a signal handler, which runs on the *faulting thread* and may not take
//! locks, allocate or call Metal. So the handler only recognises the page (a lock-free set) and
//! hands the work to a helper thread of ours, waiting for it: the helper re-materialises the
//! console bytes from the decoded texture (`victory_gpu`'s callback, which also removes the
//! protection), the handler returns, and the faulting instruction runs again. A fault that is not
//! on a page of ours is left to the loader's own handler (the write watch, then the title's SEH
//! exceptions), exactly as before.
//!
//! The set holds page numbers (16 KiB pages, the kernel's on Apple silicon), fixed in size:
//! `PAGES` entries, open-addressed with tombstones (`0` empty, `usize::MAX` deleted). It holds
//! far more pages than released copies can cover (a few GiB), and a lookup that misses only means
//! the fault goes the old way.

use std::sync::atomic::{AtomicIsize, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// The kernel's page (Apple silicon).
const PAGE: usize = 1 << 14;
/// Pages tracked: 2^18 × 16 KiB = 4 GiB of released copies. Each entry is a word.
const PAGES: usize = 1 << 18;
const EMPTY: usize = 0;
const TOMB: usize = usize::MAX;

static SET: [AtomicUsize; PAGES] = [const { AtomicUsize::new(EMPTY) }; PAGES];
static INSTALLED: AtomicU64 = AtomicU64::new(0);
static HANDLER: AtomicUsize = AtomicUsize::new(0);
/// The fault the handler is waiting on, and the helper's answer (0: none, 1: restored, 2: no).
static REQUEST: AtomicUsize = AtomicUsize::new(0);
static DONE: AtomicIsize = AtomicIsize::new(0);
static FAULTS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static OPENED: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" {
    fn mprotect(addr: *mut std::ffi::c_void, len: usize, prot: i32) -> i32;
}
const PROT_READ_WRITE: i32 = 3;

/// The callback: re-materialise the page holding `at`; `true` when it did (the access may run).
type Handler = fn(usize) -> bool;

/// Faults on our pages handled, ones the helper could not (the fault then goes the old way), and
/// page writes served by opening the page alone (no restore: the writer replaces the bytes).
pub fn counts() -> (u64, u64, u64) {
    (
        FAULTS.load(Ordering::Relaxed),
        MISSES.load(Ordering::Relaxed),
        OPENED.load(Ordering::Relaxed),
    )
}

/// Whether a callback is installed (the fault path is in use).
pub fn active() -> bool {
    INSTALLED.load(Ordering::Acquire) == 2
}

/// Installs the callback and starts its helper thread, once per process.
pub fn install(handler: Handler) -> bool {
    match INSTALLED.load(Ordering::Acquire) {
        2 => return true,
        1 => return false,
        _ => {}
    }
    HANDLER.store(handler as usize, Ordering::Release);
    let spawned = std::thread::Builder::new().name("page-restore".into()).spawn(helper);
    if spawned.is_err() {
        INSTALLED.store(1, Ordering::Release);
        return false;
    }
    INSTALLED.store(2, Ordering::Release);
    true
}

/// The helper: waits for a request, restores the page, answers. Not a signal handler, so it may
/// take the title lock, allocate and call Metal. Polls: the handler that posts a request can only
/// use atomics (a condition variable is not signal-safe).
fn helper() {
    loop {
        if REQUEST.load(Ordering::Acquire) == 0 {
            std::thread::sleep(Duration::from_micros(200));
            continue;
        }
        let at = REQUEST.swap(0, Ordering::AcqRel);
        let handler = HANDLER.load(Ordering::Acquire);
        let handler: Option<Handler> = (handler != 0).then(|| {
            // SAFETY: installed once with this exact type.
            unsafe { std::mem::transmute::<usize, Handler>(handler) }
        });
        let restored = handler.is_some_and(|f| f(at));
        if restored {
            FAULTS.fetch_add(1, Ordering::Relaxed);
        } else {
            MISSES.fetch_add(1, Ordering::Relaxed);
        }
        DONE.store(if restored { 1 } else { 2 }, Ordering::Release);
    }
}

/// Called from the signal handler for a fault at `at`: `true` when it was ours and the page is
/// back (the access runs again); `false` to leave it to the loader's handler.
///
/// A write opens the page here and now: the writer is about to replace the bytes, so nothing
/// must be re-materialised, and `mprotect` is all it takes — no Metal and no title lock, so a
/// host worker that holds the title (or waits on one that does) cannot deadlock. A read needs the
/// old bytes back, which needs the helper (and its 5 s bound, so a helper that cannot restore
/// does not hold the title for good: the fault then goes the old way, the title's SEH exception).
///
/// Signal-safe: atomics, `mprotect`, a clock read and a spin.
pub fn on_fault(at: usize, write: bool) -> bool {
    let page = at & !(PAGE - 1);
    if !active() || !contains(page) {
        return false;
    }
    if write {
        if open_for_write(page, PAGE) {
            OPENED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        return false;
    }
    DONE.store(0, Ordering::Release);
    REQUEST.store(at, Ordering::Release);
    let started = std::time::Instant::now();
    loop {
        match DONE.load(Ordering::Acquire) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        if started.elapsed() > Duration::from_secs(5) {
            return false;
        }
        std::hint::spin_loop();
    }
}

/// Takes the pages `[at, at + len)` (whole pages inside the range) into the set.
pub fn release(at: usize, len: usize) {
    for page in (at..at.saturating_add(len)).step_by(PAGE) {
        if page & (PAGE - 1) == 0 {
            insert(page);
        }
    }
}

/// Opens the released pages of `[at, at + len)` for writing: the writer is about to replace their
/// bytes, so there is nothing to re-materialise from the texture. Lock-free and signal-safe
/// (`mprotect` only), so host code calls it before writing title memory (DirectStorage staging,
/// file reads, XG tiling) and the signal handler serves a write fault with it. `true` when a page
/// in the range was ours.
pub fn open_for_write(at: usize, len: usize) -> bool {
    let end = at.saturating_add(len);
    let mut page = at & !(PAGE - 1);
    let mut opened = false;
    while page < end {
        if contains(page) {
            remove(page);
            // SAFETY: a page this module released earlier; the caller writes title memory here.
            unsafe { mprotect(page as *mut _, PAGE, PROT_READ_WRITE) };
            opened = true;
        }
        page = page.saturating_add(PAGE);
        if page == 0 {
            break;
        }
    }
    opened
}

/// Whether every page of `[at, at + len)` is still released. A host write opens only the pages it
/// touches, leaving the copy half out: the GPU thread drops such a copy whole (its remaining pages
/// are opened and the bytes re-read).
pub fn all_out(at: usize, len: usize) -> bool {
    let end = at.saturating_add(len);
    let mut page = at & !(PAGE - 1);
    if page >= end {
        return false;
    }
    while page < end {
        if !contains(page) {
            return false;
        }
        page = page.saturating_add(PAGE);
        if page == 0 {
            break;
        }
    }
    true
}

/// Gives the pages `[at, at + len)` back out of the set (a restore, or the texture released).
pub fn restore(at: usize, len: usize) {
    for page in (at..at.saturating_add(len)).step_by(PAGE) {
        if page & (PAGE - 1) == 0 {
            remove(page);
        }
    }
}

/// Whether `page` (a 16 KiB-aligned address) is in the set.
fn contains(page: usize) -> bool {
    let start = slot(page);
    for i in 0..PAGES {
        let at = &SET[(start + i) % PAGES];
        match at.load(Ordering::Acquire) {
            EMPTY => return false,
            TOMB => continue,
            p if p == page => return true,
            _ => {}
        }
    }
    false
}

fn insert(page: usize) {
    let start = slot(page);
    for i in 0..PAGES {
        let at = &SET[(start + i) % PAGES];
        let found = at.load(Ordering::Acquire);
        if found == page {
            return;
        }
        if found == EMPTY || found == TOMB {
            if at.compare_exchange(found, page, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return;
            }
            // Another thread took the slot; the next one may be free.
        }
    }
}

fn remove(page: usize) {
    let start = slot(page);
    for i in 0..PAGES {
        let at = &SET[(start + i) % PAGES];
        match at.load(Ordering::Acquire) {
            EMPTY => return,
            p if p == page => {
                let _ = at.compare_exchange(page, TOMB, Ordering::AcqRel, Ordering::Acquire);
                return;
            }
            _ => {}
        }
    }
}

/// The first slot `page` may sit in: a multiplicative hash of the page number.
fn slot(page: usize) -> usize {
    (page >> 14).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize % PAGES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The page set: what is released is found, what is restored is not, and a tombstone does
    /// not stop a later probe.
    #[test]
    fn released_pages_are_found_until_restored() {
        let a = 0x1000_0000usize;
        let b = a + 3 * PAGE;
        release(a, 4 * PAGE);
        assert!(contains(a));
        assert!(contains(b));
        assert!(!contains(a + 5 * PAGE));
        restore(b, PAGE);
        assert!(contains(a));
        assert!(!contains(b), "restored pages are no longer ours");
        restore(a, 4 * PAGE);
        assert!(!contains(a));
        // Another page that hashes to the same start is still found after a tombstone.
        release(a, PAGE);
        release(a + 5 * PAGE, PAGE);
        assert!(contains(a) && contains(a + 5 * PAGE));
        restore(a, PAGE);
        restore(a + 5 * PAGE, PAGE);
    }

    /// The hook leaves a fault on a page of ours to the helper, and a fault elsewhere to the
    /// loader: a page that was never released is not ours.
    #[test]
    fn a_fault_outside_the_set_is_not_ours() {
        assert!(!on_fault(0x2000_0000, false));
        assert!(!on_fault(0x2000_0000, true));
    }

    /// A write opens the page and needs no helper: the store lands, and the copy is no longer
    /// wholly out (the GPU thread drops it whole rather than reading stale bytes).
    #[test]
    fn a_write_fault_opens_the_page_without_the_helper() {
        assert!(install(|_| false), "installed (no helper reached)");
        let layout = std::alloc::Layout::from_size_align(PAGE, PAGE).unwrap();
        // SAFETY: one page, page-aligned.
        let page = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!page.is_null());
        // SAFETY: our own page.
        unsafe { page.cast::<u64>().write(0x1111_2222_3333_4444) };
        release(page as usize, PAGE);
        // SAFETY: our own page.
        unsafe { mprotect(page.cast(), PAGE, 0) };
        assert!(all_out(page as usize, PAGE), "the page is ours and out");
        let before = counts().2;
        assert!(on_fault(page as usize + 8, true), "a write is served by opening the page");
        // SAFETY: the page was opened by `on_fault`.
        unsafe { page.cast::<u64>().write(0x5555_6666_7777_8888) };
        assert_eq!(unsafe { page.cast::<u64>().read() }, 0x5555_6666_7777_8888);
        assert!(!all_out(page as usize, PAGE), "and the copy is no longer wholly out");
        assert_eq!(counts().2, before + 1);
        assert_eq!(counts().1, 0, "no helper was asked");
        // SAFETY: the allocation above.
        unsafe { std::alloc::dealloc(page, layout) };
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SigAction {
        handler: usize,
        mask: u32,
        flags: i32,
    }

    unsafe extern "C" {
        fn sigaction(signal: i32, action: *const SigAction, old: *mut SigAction) -> i32;
        fn mprotect(addr: *mut std::ffi::c_void, len: usize, prot: i32) -> i32;
    }

    static PROBE_PAGE: AtomicUsize = AtomicUsize::new(0);

    /// The signal a page of ours raises: `on_fault` puts it back, and the load runs again. No
    /// allocation, lock or panic in here: a failure leaves the process with a status.
    unsafe extern "C" fn on_fault_signal(_signal: i32) {
        let page = PROBE_PAGE.load(Ordering::Relaxed);
        if page == 0 || !on_fault(page, false) {
            unsafe extern "C" {
                #[link_name = "_exit"]
                fn exit_process(code: i32) -> !;
            }
            // SAFETY: a plain exit(2) call.
            unsafe { exit_process(70) }
        }
    }

    /// A released page that is read and stored: the helper restores it, then the access runs.
    #[test]
    fn a_fault_on_a_released_page_is_restored() {
        const PAGE: usize = super::PAGE;
        // A handler that asks `on_fault` and returns: the access runs again. Only installed for
        // this process, once.
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| unsafe {
            let action = SigAction { handler: on_fault_signal as *const () as usize, mask: 0, flags: 0 };
            // A fault on a page with no access is a bus error on this platform, and a segment
            // one on others: both.
            sigaction(10, &action, std::ptr::null_mut());
            sigaction(11, &action, std::ptr::null_mut());
        });
        let restored = |_at: usize| {
            let base = PROBE_PAGE.load(Ordering::Relaxed);
            // SAFETY: the probe's own page.
            unsafe { mprotect(base as *mut _, PAGE, 3) == 0 }
        };
        assert!(install(restored), "the helper thread starts");
        let layout = std::alloc::Layout::from_size_align(PAGE, PAGE).unwrap();
        // SAFETY: one page, page-aligned.
        let page = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!page.is_null());
        PROBE_PAGE.store(page as usize, Ordering::Release);
        // SAFETY: our own page.
        unsafe { page.cast::<u64>().write(0x1234_5678_9abc_def0) };
        release(page as usize, PAGE);
        // SAFETY: our own page.
        unsafe { mprotect(page.cast(), PAGE, 0) };
        // SAFETY: the helper puts the page back before the load runs.
        assert_eq!(unsafe { page.cast::<u64>().read() }, 0x1234_5678_9abc_def0);
        // SAFETY: as above.
        unsafe { page.cast::<u64>().write(0x0fed_cba9_8765_4321) };
        assert_eq!(unsafe { page.cast::<u64>().read() }, 0x0fed_cba9_8765_4321);
        assert_eq!(counts().0, 1, "the first fault, and only it, was read back");
        restore(page as usize, PAGE);
        // SAFETY: the allocation above.
        unsafe { std::alloc::dealloc(page, layout) };
        PROBE_PAGE.store(0, Ordering::Release);
    }
}
