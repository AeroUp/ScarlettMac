//! Live bytes of Metal allocations by kind, for the host's frame log.
//!
//! `metal_mb` on "native draws so far" is what the device reports it holds (`currentAllocatedSize`).
//! It says how much, not what: a run that fills the Mac's memory cannot tell a pile of host texture
//! copies from a pile of host buffers, and so cannot say what to cut. These counters split it, and
//! whatever they do not account for is left as the difference (`other_mb` in the log line).
//!
//! One device per process, so the figures are process-wide, like `metal::LIBRARIES`.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// What an allocation is, as the frame log splits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Host copies of title textures, and the render and depth targets the host makes.
    Texture,
    /// No-copy buffers over title memory: the title's own pages, wrapped, not the host's extra.
    Pool,
    /// Host buffers the host allocates (uploads, staging, scratch, indirect records).
    Buffer,
    /// Descriptor heaps' own buffers.
    Heap,
}

const KINDS: usize = 4;

static BYTES: [AtomicU64; KINDS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ALLOCATIONS: [AtomicU64; KINDS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

impl Kind {
    fn slot(self) -> usize {
        match self {
            Kind::Texture => 0,
            Kind::Pool => 1,
            Kind::Buffer => 2,
            Kind::Heap => 3,
        }
    }
}

/// Bytes and allocation counts by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocCounts {
    pub texture_bytes: u64,
    pub pool_bytes: u64,
    pub buffer_bytes: u64,
    pub heap_bytes: u64,
    pub texture_allocations: u64,
    pub pool_allocations: u64,
    pub buffer_allocations: u64,
    pub heap_allocations: u64,
}

impl AllocCounts {
    /// Every kind added up: what the counters account for of `metal_mb`.
    pub fn bytes(&self) -> u64 {
        self.texture_bytes + self.pool_bytes + self.buffer_bytes + self.heap_bytes
    }
}

/// Bytes and allocations of every kind now live.
pub fn counts() -> AllocCounts {
    let b = |k: Kind| BYTES[k.slot()].load(Relaxed);
    let n = |k: Kind| ALLOCATIONS[k.slot()].load(Relaxed);
    AllocCounts {
        texture_bytes: b(Kind::Texture),
        pool_bytes: b(Kind::Pool),
        buffer_bytes: b(Kind::Buffer),
        heap_bytes: b(Kind::Heap),
        texture_allocations: n(Kind::Texture),
        pool_allocations: n(Kind::Pool),
        buffer_allocations: n(Kind::Buffer),
        heap_allocations: n(Kind::Heap),
    }
}

/// One allocation, counted while it lives: taken back when dropped. Held by the backend's
/// `Texture` and `Buffer`, so the count follows the resource to its last reference.
#[derive(Debug)]
pub struct Counted {
    kind: Kind,
    bytes: u64,
}

impl Counted {
    /// Count `bytes` of `kind` from now until the returned value drops.
    pub fn new(kind: Kind, bytes: usize) -> Counted {
        let bytes = bytes as u64;
        if bytes > 0 {
            BYTES[kind.slot()].fetch_add(bytes, Relaxed);
            ALLOCATIONS[kind.slot()].fetch_add(1, Relaxed);
        }
        Counted { kind, bytes }
    }

    /// A placeholder that counts nothing (a texture view: its bytes belong to the allocation it
    /// views, which is counted once, by whoever made that).
    pub fn none() -> Counted {
        Counted {
            kind: Kind::Texture,
            bytes: 0,
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        if self.bytes > 0 {
            BYTES[self.kind.slot()].fetch_sub(self.bytes, Relaxed);
            ALLOCATIONS[self.kind.slot()].fetch_sub(1, Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters follow lives, not calls: a drop takes its bytes back, and only its own.
    #[test]
    fn a_counted_allocation_is_live_until_it_drops() {
        let before = counts();
        let a = Counted::new(Kind::Texture, 4096);
        let after_make = counts();
        assert_eq!(after_make.texture_bytes - before.texture_bytes, 4096);
        assert_eq!(after_make.texture_allocations - before.texture_allocations, 1);
        assert_eq!(after_make.pool_bytes, before.pool_bytes, "another kind moved");
        let b = Counted::new(Kind::Texture, 4096);
        assert_eq!(counts().texture_bytes - before.texture_bytes, 8192);
        drop(a);
        assert_eq!(
            counts().texture_bytes - before.texture_bytes,
            4096,
            "took both"
        );
        drop(b);
        assert_eq!(counts(), before);
    }

    /// A kind keeps its own: a pool buffer's bytes never read as a host buffer's.
    #[test]
    fn kinds_are_counted_apart() {
        let before = counts();
        let pool = Counted::new(Kind::Pool, 1 << 20);
        let host = Counted::new(Kind::Buffer, 2 << 20);
        let heap = Counted::new(Kind::Heap, 4 << 20);
        let now = counts();
        assert_eq!(now.pool_bytes - before.pool_bytes, 1 << 20);
        assert_eq!(now.buffer_bytes - before.buffer_bytes, 2 << 20);
        assert_eq!(now.heap_bytes - before.heap_bytes, 4 << 20);
        assert_eq!(now.texture_bytes, before.texture_bytes);
        assert_eq!(now.bytes() - before.bytes(), 7 << 20);
        drop((pool, host, heap));
        assert_eq!(counts(), before);
    }

    /// A view counts nothing: its bytes are the viewed allocation's.
    #[test]
    fn an_uncounted_allocation_moves_nothing() {
        let before = counts();
        let none = Counted::none();
        assert_eq!(counts(), before);
        drop(none);
        assert_eq!(counts(), before);
    }
}
