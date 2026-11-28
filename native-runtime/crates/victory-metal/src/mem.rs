//! Page-aligned host reservations, standing in for what the native host will hand us.

use std::ffi::c_void;

/// Apple Silicon pages are 16 KiB; `bytesNoCopy` wants this alignment and a multiple of it in length.
pub fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// An anonymous mapping, zero-filled, optionally at a requested fixed address.
pub struct Region {
    pub ptr: *mut u8,
    pub len: usize,
}

impl Region {
    /// `fixed` maps at exactly that address (how the title sees its own memory).
    pub fn new(len: usize, fixed: Option<usize>) -> Option<Region> {
        let mut flags = libc::MAP_ANON | libc::MAP_PRIVATE;
        if fixed.is_some() {
            flags |= libc::MAP_FIXED;
        }
        let want = fixed.unwrap_or(0) as *mut c_void;
        // SAFETY: anonymous mapping; a fixed address is the caller's chosen free range.
        let p = unsafe { libc::mmap(want, len, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
        (p != libc::MAP_FAILED).then(|| Region { ptr: p.cast(), len })
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from mmap in `new`.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_allocate(task: u32, addr: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(task: u32, addr: u64, size: u64) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mach_vm_remap(
        target: u32,
        addr: *mut u64,
        size: u64,
        mask: u64,
        flags: i32,
        src_task: u32,
        src_addr: u64,
        copy: i32,
        cur: *mut i32,
        max: *mut i32,
        inherit: u32,
    ) -> i32;
}

const VM_FLAGS_ANYWHERE: i32 = 1;
const VM_INHERIT_SHARE: u32 = 0;

/// Two virtual aliases of the same physical pages (the shape of "one mach VM object, mapped twice").
pub struct Alias {
    pub primary: *mut u8,
    pub alias: *mut u8,
    pub len: usize,
}

impl Alias {
    pub fn new(len: usize) -> Result<Alias, i32> {
        let mut a = 0u64;
        // SAFETY: plain Mach VM calls on our own task with local out-parameters.
        unsafe {
            let t = mach_task_self();
            let r = mach_vm_allocate(t, &mut a, len as u64, VM_FLAGS_ANYWHERE);
            if r != 0 {
                return Err(r);
            }
            let (mut b, mut cur, mut max) = (0u64, 0i32, 0i32);
            let r = mach_vm_remap(
                t,
                &mut b,
                len as u64,
                0,
                VM_FLAGS_ANYWHERE,
                t,
                a,
                0,
                &mut cur,
                &mut max,
                VM_INHERIT_SHARE,
            );
            if r != 0 {
                mach_vm_deallocate(t, a, len as u64);
                return Err(r);
            }
            Ok(Alias {
                primary: a as *mut u8,
                alias: b as *mut u8,
                len,
            })
        }
    }
}

impl Drop for Alias {
    fn drop(&mut self) {
        // SAFETY: both ranges came from the Mach calls in `new`.
        unsafe {
            let t = mach_task_self();
            mach_vm_deallocate(t, self.primary as u64, self.len as u64);
            mach_vm_deallocate(t, self.alias as u64, self.len as u64);
        }
    }
}

const VM_FLAGS_OVERWRITE: i32 = 0x4000;

/// One zero-filled page every hole of a [`PageMap`] can share (made once, never written).
fn shared_zero_page(page: usize) -> Option<*mut u8> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ZERO: AtomicUsize = AtomicUsize::new(0);
    let seen = ZERO.load(Ordering::Relaxed);
    if seen != 0 {
        return Some(seen as *mut u8);
    }
    let region = Region::new(page, None)?;
    let ptr = region.ptr;
    // The page lives as long as the process: every hole maps it.
    std::mem::forget(region);
    let _ = ZERO.compare_exchange(0, ptr as usize, Ordering::Relaxed, Ordering::Relaxed);
    Some(ptr)
}

/// A private range of `len` bytes whose every `page`-sized page shows either a page of the title's
/// memory or a hole: a page covers what the title maps there, shared (a remap, not a copy, so
/// later writes by either side are seen by both), and a hole reads 0. A buffer view over a range
/// the title maps only in part wraps this, where D3D12 tier 2 reads 0 only at unmapped tiles.
///
/// A hole is one shared zero page when `zero_shared` (`true` for a view nothing writes); a
/// writable view leaves its holes as its own zero pages, so a stored tile cannot reach another
/// view's hole.
pub struct PageMap {
    pub ptr: *mut u8,
    pub len: usize,
}

impl PageMap {
    /// `source(k)` is page `k`'s title address to share, `None` for a hole.
    pub fn new(
        len: usize,
        page: usize,
        zero_shared: bool,
        source: impl FnMut(usize) -> Option<usize>,
    ) -> Result<PageMap, i32> {
        if page == 0 || len == 0 || !len.is_multiple_of(page) {
            return Err(-1);
        }
        let region = Region::new(len, None).ok_or(-1)?;
        let mut map = PageMap { ptr: region.ptr, len };
        // The range is this struct's now (dropped by `Drop`).
        std::mem::forget(region);
        match map.refresh(page, zero_shared, source) {
            Ok(()) => Ok(map),
            Err(e) => Err(e),
        }
    }

    /// [`Self::new`]'s page mapping over the range already held: each page is remapped from its
    /// source now (the range and its address stay, so a buffer made over it stays valid).
    pub fn refresh(
        &mut self,
        page: usize,
        zero_shared: bool,
        mut source: impl FnMut(usize) -> Option<usize>,
    ) -> Result<(), i32> {
        let (len, ptr) = (self.len, self.ptr);
        let zero = if zero_shared { shared_zero_page(page) } else { None };
        let t = unsafe { mach_task_self() };
        for k in 0..len / page {
            let src = match source(k) {
                Some(va) => va as u64,
                None => match zero {
                    Some(z) => z as u64,
                    // A private zero page, already there: nothing to remap.
                    None => continue,
                },
            };
            let (mut at, mut cur, mut max) = ((ptr as usize + k * page) as u64, 0i32, 0i32);
            // SAFETY: Mach VM call on our own task; `at` is a page of our own reservation, `src`
            // a page of the title's memory (or the shared zero page).
            let r = unsafe {
                mach_vm_remap(
                    t,
                    &mut at,
                    page as u64,
                    0,
                    VM_FLAGS_OVERWRITE,
                    t,
                    src,
                    0,
                    &mut cur,
                    &mut max,
                    VM_INHERIT_SHARE,
                )
            };
            if r != 0 {
                return Err(r);
            }
        }
        Ok(())
    }
}

impl Drop for PageMap {
    fn drop(&mut self) {
        // SAFETY: the range came from `Region::new`/the Mach calls in `new`.
        unsafe { mach_vm_deallocate(mach_task_self(), self.ptr as u64, self.len as u64) };
    }
}
