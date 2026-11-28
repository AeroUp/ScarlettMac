//! `api-ms-win-crt-heap`: the host allocator (16-byte aligned, and
//! `_aligned_*` memory is freeable by `free`, so either free works).

use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use super::{EINVAL, ENOMEM, exports, libc, set_errno};

static NEW_MODE: AtomicI32 = AtomicI32::new(0);

/// Bytes live in the title's C runtime heap -- the host allocator behind its `malloc`/`new`.
///
/// On the console this heap comes out of the title's memory budget like everything else, so it is
/// counted as the title's (not the host's): `xmem::title_memory` reports it, and the memory line
/// shows it. Without this the title's pools (7,607 MiB, its whole console budget) plus 2+ GB of
/// heap looked like 7.6 GB used, while the process held 13 GB.
static HEAP_LIVE: AtomicUsize = AtomicUsize::new(0);

/// [`HEAP_LIVE`] in bytes.
pub fn heap_bytes() -> u64 {
    HEAP_LIVE.load(Ordering::Relaxed) as u64
}

/// Counts `p` (a live block) into [`HEAP_LIVE`], and returns it.
fn counted(p: *mut c_void) -> *mut c_void {
    // SAFETY: a block this module allocated.
    let bytes = if p.is_null() { 0 } else { unsafe { libc::malloc_size(p) } };
    HEAP_LIVE.fetch_add(bytes, Ordering::Relaxed);
    p
}

/// The block `p` held, for [`free`] and `realloc` (0 for null).
fn block_size(p: *mut c_void) -> usize {
    // SAFETY: a block this module allocated.
    if p.is_null() { 0 } else { unsafe { libc::malloc_size(p) } }
}

fn checked(p: *mut c_void) -> *mut c_void {
    if p.is_null() {
        set_errno(ENOMEM);
    }
    counted(p)
}

pub extern "system" fn malloc(size: usize) -> *mut c_void {
    // SAFETY: plain allocation.
    checked(unsafe { libc::malloc(size.max(1)) })
}

pub extern "system" fn calloc(count: usize, size: usize) -> *mut c_void {
    let (count, size) = if count == 0 || size == 0 {
        (1, 1)
    } else {
        (count, size)
    };
    // SAFETY: plain allocation (overflow gives null).
    checked(unsafe { libc::calloc(count, size) })
}

pub extern "system" fn realloc(p: *mut c_void, size: usize) -> *mut c_void {
    if size == 0 && !p.is_null() {
        // The CRT frees on a zero size and returns null.
        // SAFETY: the title's own allocation.
        unsafe { libc::free(p) };
        return std::ptr::null_mut();
    }
    let before = block_size(p);
    // SAFETY: null or the title's own allocation.
    let q = unsafe { libc::realloc(p, size.max(1)) };
    if q.is_null() {
        // The old block is still the title's.
        set_errno(ENOMEM);
        return q;
    }
    HEAP_LIVE.fetch_sub(before, Ordering::Relaxed);
    checked(q)
}

pub extern "system" fn free(p: *mut c_void) {
    if !p.is_null() {
        HEAP_LIVE.fetch_sub(block_size(p), Ordering::Relaxed);
        // SAFETY: the title's own allocation.
        unsafe { libc::free(p) };
    }
}

pub extern "system" fn _recalloc(p: *mut c_void, count: usize, size: usize) -> *mut c_void {
    let Some(total) = count.checked_mul(size) else {
        set_errno(ENOMEM);
        return std::ptr::null_mut();
    };
    if total == 0 {
        return if p.is_null() {
            calloc(1, 1)
        } else {
            free(p);
            std::ptr::null_mut()
        };
    }
    let old = if p.is_null() {
        0
    } else {
        // SAFETY: the title's own allocation.
        unsafe { libc::malloc_size(p) }
    };
    let q = realloc(p, total);
    if !q.is_null() && total > old {
        // SAFETY: the new tail of the block just resized.
        unsafe { q.cast::<u8>().add(old).write_bytes(0, total - old) };
    }
    q
}

pub extern "system" fn _msize(p: *mut c_void) -> usize {
    if p.is_null() {
        set_errno(EINVAL);
        return usize::MAX;
    }
    // SAFETY: the title's own allocation.
    unsafe { libc::malloc_size(p) }
}

pub extern "system" fn _expand(_p: *mut c_void, _size: usize) -> *mut c_void {
    // Never grows in place.
    std::ptr::null_mut()
}

pub extern "system" fn _aligned_malloc(size: usize, align: usize) -> *mut c_void {
    if !align.is_power_of_two() {
        set_errno(EINVAL);
        return std::ptr::null_mut();
    }
    let mut out = std::ptr::null_mut();
    // SAFETY: a valid out pointer; `align` is a power of two.
    let r = unsafe { libc::posix_memalign(&mut out, align.max(16), size.max(1)) };
    if r != 0 {
        set_errno(ENOMEM);
        return std::ptr::null_mut();
    }
    counted(out)
}

pub extern "system" fn _aligned_free(p: *mut c_void) {
    free(p);
}

pub extern "system" fn _aligned_realloc(p: *mut c_void, size: usize, align: usize) -> *mut c_void {
    if p.is_null() {
        return _aligned_malloc(size, align);
    }
    if size == 0 {
        free(p);
        return std::ptr::null_mut();
    }
    let q = _aligned_malloc(size, align);
    if !q.is_null() {
        // SAFETY: both are live blocks of at least the copied length.
        unsafe {
            let n = libc::malloc_size(p).min(size);
            std::ptr::copy_nonoverlapping(p.cast::<u8>(), q.cast::<u8>(), n);
        }
        free(p);
    }
    q
}

pub extern "system" fn _aligned_msize(p: *mut c_void, _align: usize, _offset: usize) -> usize {
    _msize(p)
}

pub extern "system" fn _set_new_mode(mode: i32) -> i32 {
    if !(0..=1).contains(&mode) {
        set_errno(EINVAL);
        return -1;
    }
    NEW_MODE.swap(mode, Ordering::Relaxed)
}

pub extern "system" fn _query_new_mode() -> i32 {
    NEW_MODE.load(Ordering::Relaxed)
}

/// No new-handler: `operator new` does not retry.
pub extern "system" fn _callnewh(_size: usize) -> i32 {
    0
}

pub extern "system" fn _set_new_handler(_handler: usize) -> usize {
    0
}

pub extern "system" fn _get_heap_handle() -> usize {
    1
}

/// `_HEAPOK`.
pub extern "system" fn _heapchk() -> i32 {
    -2
}

pub extern "system" fn _heapmin() -> i32 {
    0
}

exports!(pub(super) fn lookup;
    malloc, calloc, realloc, free, _recalloc, _msize, _expand, _aligned_malloc, _aligned_free,
    _aligned_realloc, _aligned_msize, _set_new_mode, _query_new_mode, _callnewh,
    _set_new_handler, _get_heap_handle, _heapchk, _heapmin;
    "_malloc_base" => malloc, "_calloc_base" => calloc, "_realloc_base" => realloc,
    "_free_base" => free, "_msize_base" => _msize, "_recalloc_base" => _recalloc);

#[cfg(test)]
mod tests {
    use super::*;

    /// The title's heap is the title's memory: what it allocates is counted, and what it frees is
    /// given back (other tests allocate at the same time, so only this test's own growth is
    /// asserted).
    #[test]
    fn the_title_heap_is_counted_and_given_back() {
        let before = heap_bytes();
        let p = malloc(4096);
        assert!(!p.is_null());
        assert!(heap_bytes() >= before + 4096, "malloc counts the block");
        let q = realloc(p, 8192);
        assert!(!q.is_null());
        assert!(heap_bytes() >= before + 8192, "realloc counts what replaced it");
        free(q);
        let a = malloc(65536);
        assert!(!a.is_null());
        assert!(heap_bytes() >= before + 65536, "and a later block is counted too");
        free(a);
    }

    #[test]
    fn blocks_are_16_byte_aligned_and_resizable() {
        let p = malloc(100);
        assert_eq!(p as usize % 16, 0);
        assert!(_msize(p) >= 100);
        let p = realloc(p, 1000);
        assert!(_msize(p) >= 1000);
        assert!(realloc(p, 0).is_null());
        let z = calloc(4, 8);
        // SAFETY: 32 bytes were allocated.
        assert!(
            unsafe { std::slice::from_raw_parts(z.cast::<u8>(), 32) }
                .iter()
                .all(|&b| b == 0)
        );
        let z = _recalloc(z, 8, 8);
        // SAFETY: 64 bytes now.
        assert!(
            unsafe { std::slice::from_raw_parts(z.cast::<u8>(), 64) }
                .iter()
                .all(|&b| b == 0)
        );
        free(z);
        let a = _aligned_malloc(10, 256);
        assert_eq!(a as usize % 256, 0);
        let a = _aligned_realloc(a, 5000, 256);
        assert_eq!(a as usize % 256, 0);
        _aligned_free(a);
        assert!(_aligned_malloc(8, 3).is_null());
        assert_eq!(_set_new_mode(1), 0);
        assert_eq!(_set_new_mode(0), 1);
    }

    #[test]
    fn calloc_and_recalloc_handle_zero_counts_and_sizes() {
        let a = calloc(0, usize::MAX);
        assert!(!a.is_null());
        free(a);
        let b = calloc(usize::MAX, 0);
        assert!(!b.is_null());
        free(b);

        let p = calloc(4, 8);
        assert!(!p.is_null());
        assert!(_recalloc(p, 0, 8).is_null());
        let p = calloc(4, 8);
        assert!(!p.is_null());
        assert!(_recalloc(p, 4, 0).is_null());

        let z = _recalloc(std::ptr::null_mut(), 0, 8);
        assert!(!z.is_null());
        // SAFETY: one zeroed byte was allocated.
        assert_eq!(unsafe { z.cast::<u8>().read() }, 0);
        free(z);
    }
}
