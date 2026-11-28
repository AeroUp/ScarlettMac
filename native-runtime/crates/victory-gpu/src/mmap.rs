//! A cache entry's bytes, mapped read-only instead of read into the heap.
//!
//! The decoded-texture cache (`texture_cache`) keeps the Metal-ready bytes of a texture on disk.
//! A texture upload only reads them, and they are the same bytes on every run, so mapping them is
//! what the page cache is for: the pages are clean and file-backed, the kernel can drop or re-read
//! them without a write to swap, and the upload reads them in place — no second copy of the level
//! in anonymous memory, which on a title that streams (hundreds of MiB a second) was as much dirty
//! memory as the copies themselves.
//!
//! Entries are written to a temporary name and renamed, so a mapping never sees a change under it
//! (that is also why a mapped entry needs no lock). Where the platform has no `mmap` the bytes are
//! read as before: the caller's type is the same.

use std::path::Path;
use std::sync::Arc;

/// A file's bytes, mapped read-only or (where the platform has no `mmap`) read into the heap.
pub(crate) struct Map {
    backing: Backing,
}

enum Backing {
    /// A private read-only mapping: `at` is `len` bytes of the file.
    #[cfg(unix)]
    Mapped { at: *const u8, len: usize },
    /// The file's bytes, for a platform without `mmap`.
    Read(Vec<u8>),
}

// SAFETY: a mapping is immutable for its life (cache entries are replaced by a rename, never
// written in place) and holds no thread-local state, so sharing one between the threads that read
// a submission's textures is safe.
unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Map {
    /// Maps `path` read-only (`None`-like error if it cannot be opened or is empty).
    pub(crate) fn open(path: &Path) -> std::io::Result<Arc<Map>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            if len == 0 {
                return Ok(Arc::new(Map { backing: Backing::Read(Vec::new()) }));
            }
            // SAFETY: a plain `mmap` of an open file's first `len` bytes, read-only and private.
            let at = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len as usize,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if at == MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Arc::new(Map { backing: Backing::Mapped { at: at.cast(), len: len as usize } }))
        }
        #[cfg(not(unix))]
        {
            let _ = file;
            Ok(Arc::new(Map { backing: Backing::Read(std::fs::read(path)?) }))
        }
    }

    /// The file's bytes.
    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.backing {
            #[cfg(unix)]
            // SAFETY: the mapping is `len` readable bytes made in `open`, alive as long as `self`.
            Backing::Mapped { at, len } => unsafe { std::slice::from_raw_parts(*at, *len) },
            Backing::Read(bytes) => bytes,
        }
    }

    /// Where `slice` (which must be inside [`Self::bytes`]) starts in the file.
    pub(crate) fn offset_of(&self, slice: &[u8]) -> usize {
        slice.as_ptr() as usize - self.bytes().as_ptr() as usize
    }
}

#[cfg(unix)]
impl Drop for Map {
    fn drop(&mut self) {
        if let Backing::Mapped { at, len } = self.backing {
            // SAFETY: the mapping `open` made, unmapped once.
            unsafe { munmap(at.cast_mut().cast(), len) };
        }
    }
}

#[cfg(unix)]
const PROT_READ: i32 = 1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 2;

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
}

#[cfg(unix)]
const MAP_FAILED: *mut std::ffi::c_void = usize::MAX as *mut std::ffi::c_void;

#[cfg(test)]
mod tests {
    use super::*;

    /// A mapped file reads back byte for byte, and a slice of it knows where it starts.
    #[test]
    fn a_mapped_file_reads_back_and_knows_its_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry");
        let bytes: Vec<u8> = (0..70_000u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let map = Map::open(&path).unwrap();
        assert_eq!(map.bytes(), bytes.as_slice());
        assert_eq!(map.offset_of(&map.bytes()[100..200]), 100);
        // An empty file maps nothing and reads as nothing (the caller treats it as a miss).
        let empty = dir.path().join("empty");
        std::fs::write(&empty, b"").unwrap();
        assert!(Map::open(&empty).unwrap().bytes().is_empty());
    }
}
