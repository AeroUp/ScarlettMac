//! The pool-offset contract: title memory at addresses a shader computes, on the native host.
//!
//! A translated shader reads a pointer it computed (`global_load`, `s_load` from a loaded address,
//! stores, atomics, BVH nodes) by looking its 40-bit title address up in a page table, the root SRV
//! `t0` in `victory_shader::emit::MEMORY_SPACE` (translator notes, "Title memory"): per 64 KiB title
//! page a resource-heap index of a raw buffer view and the low dword of the title address that view
//! starts at, so that the byte offset in the view is `address low dword - start`. DXIL has no way to
//! dereference a 64-bit address, and the converter's raw buffers are views `{GPU address, size}`
//! read from the descriptor heap, so this is also what Metal can express; the translator stays as
//! it is, and the host fills the table differently:
//!
//! * D3D12: views of host *copies* of title memory, refreshed per submission and copied back.
//! * Here: views of the pools themselves. A mapping `(title address, length, pool, offset)` of the
//!   pool table gives each of its 64 KiB pages `(pair of views of the pool buffer, start)`. There
//!   is one SRV/UAV pair of descriptors (two identical entries) per `(pool, 2 GiB chunk)`, a
//!   [`CHUNK`] plus a page long, because a view's size is 32 bits. The pairs live at fixed
//!   resource-heap indices just past the title's own entries of the heap the shader runs with
//!   (a heap is [`RESERVED`] entries longer than the title made it, and has a table of its own,
//!   whose view indices say where). Pair 0 is the null view (size 0, bounds-checked by the
//!   converter): what a page no pool shows reads as, with its writes dropped.
//! * `Options::zero_copy_memory`: the shader marks no pages and binds no bitmap, so the table is
//!   all the host binds besides the heap.
//!
//! The table is rebuilt (into a new buffer: the GPU may still read the old one) when the pool table
//! changed since it was made. shortcut: whole, not incrementally; at 64 KiB a page and 8 bytes an
//! entry, 16 GiB of mappings is 2 MiB of words. A 64 KiB title page that is not wholly inside one
//! mapping of one pool (mappings of 16 KiB granularity, or a pool boundary) is a null page.

use std::rc::Rc;
use victory_backend::backend::ComputeBackend;

/// Bytes of a title page of the table (the console's allocation granularity).
pub const PAGE: u64 = 1 << 16;
/// How much of a pool one view pair covers before the next one starts: `size` is 32 bits.
pub const CHUNK: u64 = 1 << 31;
/// Pairs of views the host writes into every resource heap (pair 0 is the null view).
pub const PAIRS: usize = 256;
/// Entries the host adds to every resource heap, for the view pairs.
pub const RESERVED: usize = 2 * PAIRS;

/// Whether a resource heap of `count` entries has its pairs below the translator's clamp of heap
/// indices (and its own null view).
pub fn fits(count: usize) -> bool {
    count + RESERVED < victory_shader::emit::NULL_BUFFER_VIEW as usize
}

/// Level 1 of the table: 4096 dwords (address bits 28-39), a level-2 block number.
const LEVEL_1: usize = 4096;
/// A level-2 block: 4096 entries of two dwords (address bits 16-27).
const BLOCK: usize = 4096 * 2;

/// What a view pair shows: `[chunk * CHUNK, ..)` of pool `pool` (whose mapping is at `base`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PairKey {
    pub pool: usize,
    pub base: usize,
    /// The pool's length.
    pub len: u64,
    pub chunk: u64,
}

/// A mapping of the pool table, as the table builder reads it.
#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub at: u64,
    pub len: u64,
    pub pool: usize,
    pub base: usize,
    pub pool_len: u64,
    pub offset: u64,
}

/// The view pairs made so far: pair `n` (n >= 1) is `keys[n - 1]`, `None` once its pool was
/// released (the slot is free for the next pool). `epoch` changes whenever a slot that existed
/// changes (released or reused), so a heap whose pairs were written in another epoch writes them
/// all again; otherwise it only needs the ones after those it has.
#[derive(Default, Debug)]
pub struct Pairs {
    pub keys: Vec<Option<PairKey>>,
    pub epoch: u64,
}

impl Pairs {
    fn pair(&mut self, key: PairKey) -> Option<u32> {
        let n = match self.keys.iter().rposition(|k| *k == Some(key)) {
            Some(n) => n,
            None => match self.keys.iter().position(Option::is_none) {
                Some(n) => {
                    self.keys[n] = Some(key);
                    self.epoch += 1;
                    n
                }
                None if self.keys.len() + 1 < PAIRS => {
                    self.keys.push(Some(key));
                    self.keys.len() - 1
                }
                None => {
                    static FULL: std::sync::Once = std::sync::Once::new();
                    FULL.call_once(|| {
                        tracing::warn!(pairs = PAIRS - 1, "every title memory view pair is taken: pages of further pools read as zero and drop their writes (first time)");
                    });
                    return None;
                }
            },
        };
        Some(n as u32 + 1)
    }

    /// Pool `pool` was released: its pairs are free for other pools.
    pub fn release(&mut self, pool: usize) {
        for k in &mut self.keys {
            if k.is_some_and(|k| k.pool == pool) {
                *k = None;
                self.epoch += 1;
            }
        }
    }

    /// The bytes of the pool a pair shows: from `chunk * CHUNK`, a chunk and a page long, as far
    /// as the pool goes.
    pub fn span(key: &PairKey) -> (u64, u32) {
        let from = key.chunk * CHUNK;
        (from, (key.len - from).min(CHUNK + PAGE) as u32)
    }
}

/// Heap index of the SRV of pair `n` in a heap whose pairs start at `first` (its title entries).
pub fn pair_index(first: u32, n: u32) -> u32 {
    first + 2 * n
}

/// The page table, in dwords: level 1, then the null block 0, then the blocks the mappings reach.
pub fn build(maps: &[Mapping], pairs: &mut Pairs, first: u32) -> Vec<u32> {
    // The pool table hands its mappings over in title-address order; a test may not.
    if maps.is_sorted_by_key(|m| m.at) {
        return build_runs(maps, pairs, first);
    }
    let mut sorted: Vec<Mapping> = maps.to_vec();
    sorted.sort_by_key(|m| m.at);
    build_runs(&sorted, pairs, first)
}

fn build_runs(maps: &[Mapping], pairs: &mut Pairs, first: u32) -> Vec<u32> {
    let null = pair_index(first, 0);
    let mut words = vec![0u32; LEVEL_1];
    for _ in 0..BLOCK / 2 {
        words.extend([null, 0]);
    }
    let mut i = 0;
    while i < maps.len() {
        let m = &maps[i];
        if m.len == 0 {
            i += 1;
            continue;
        }
        // The run of mappings that covers `m`'s first byte, as `PoolTable::resolve` walks it (each
        // following mapping starts where the last ended, in the same pool at the next offset). The
        // title reports an allocation one mapping at a time, so a page's 64 KiB are often in
        // several mappings: one view shows them all.
        let mut run_end = m.at + m.len;
        let mut next = i + 1;
        while let Some(n) = maps.get(next)
            && n.len > 0
            && n.at == run_end
            && n.pool == m.pool
            && n.offset == m.offset + (run_end - m.at)
        {
            run_end += n.len;
            next += 1;
        }
        // Each mapping fills the pages whose first byte it holds (a page that starts before it is
        // filled from the mapping before it, or not at all), as far as the run covers them and the
        // pool has bytes.
        let end = run_end & !(PAGE - 1);
        for m in &maps[i..next] {
            for page in (m.at.next_multiple_of(PAGE)..end).step_by(PAGE as usize) {
                // Addresses of 40 bits only; the translator sends the rest to block 0.
                if page >> 40 != 0 {
                    break;
                }
                let pool_at = m.offset + (page - m.at);
                if pool_at + PAGE > m.pool_len {
                    continue;
                }
                let key = PairKey {
                    pool: m.pool,
                    base: m.base,
                    len: m.pool_len,
                    chunk: pool_at / CHUNK,
                };
                let Some(pair) = pairs.pair(key) else { continue };
                let l1 = (page >> 28) as usize;
                if words[l1] == 0 {
                    words[l1] = (words.len() - LEVEL_1) as u32 / BLOCK as u32;
                    words.extend((0..BLOCK / 2).flat_map(|_| [null, 0]));
                }
                let at = LEVEL_1 + words[l1] as usize * BLOCK + 2 * ((page >> 16) as usize & 0xFFF);
                let in_view = pool_at - key.chunk * CHUNK;
                words[at] = pair_index(first, pair);
                words[at + 1] = (page as u32).wrapping_sub(in_view as u32);
            }
        }
        i = next;
    }
    words
}

/// What a built table makes of the title's mappings, for `VICTORY_TRACE_PAGES`: every 64 KiB page
/// any mapping touches, how many read (through a pair view), and how many are null although the
/// title maps all of them (`split`, over more than one mapping or pool and so no one view) or only
/// part of them (`gap`, a page the title really left a hole in: reads there are zeros to the title
/// too). `joined`: pages that read only because the mappings that cover them were joined into one
/// run.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct PageTally {
    pub pages: u64,
    pub readable: u64,
    pub joined: u64,
    pub split: u64,
    pub gap: u64,
}

pub fn tally(maps: &[Mapping], words: &[u32]) -> PageTally {
    let mut sorted: Vec<&Mapping> = maps.iter().filter(|m| m.len > 0).collect();
    sorted.sort_by_key(|m| m.at);
    let mut pages: Vec<u64> = Vec::new();
    for m in &sorted {
        let from = m.at & !(PAGE - 1);
        let end = m.at + m.len;
        pages.extend((from..end).step_by(PAGE as usize).filter(|p| p >> 40 == 0));
    }
    pages.sort_unstable();
    pages.dedup();
    let mut t = PageTally::default();
    // The first mapping that may hold a page's bytes: mappings are sorted and pages ascend, so it
    // only moves forward.
    let mut first = 0;
    for &page in &pages {
        t.pages += 1;
        while first < sorted.len() && sorted[first].at + sorted[first].len <= page {
            first += 1;
        }
        if lookup(words, page).is_some() {
            t.readable += 1;
            // Whether one mapping alone covers the page: what a table of whole mappings could fill.
            let whole = sorted[first..]
                .iter()
                .take_while(|m| m.at < page + PAGE)
                .any(|m| m.at <= page && page + PAGE <= m.at + m.len && m.offset + (page - m.at) + PAGE <= m.pool_len);
            t.joined += u64::from(!whole);
        } else if covered(&sorted, first, page) {
            t.split += 1;
        } else {
            t.gap += 1;
        }
    }
    t
}

/// Whether the title maps every byte of `page` (in any pools), from the mapping `sorted[first..]` on.
fn covered(sorted: &[&Mapping], first: usize, page: u64) -> bool {
    let end = page + PAGE;
    let mut at = page;
    for m in &sorted[first..] {
        if m.at > at {
            return false;
        }
        if m.at + m.len > at {
            at = m.at + m.len;
        }
        if at >= end {
            return true;
        }
    }
    false
}

/// A root SRV or UAV address as the translator's root views want it (`emit`, "Root SRVs and root
/// UAVs"): the heap index of the view pair's SRV (the UAV is the next entry) and the byte offset of
/// the address in the view. The first containing mapping wins. An unmapped address gets the null
/// view at offset 0.
pub fn root_view(maps: &[Mapping], pairs: &mut Pairs, first: u32, address: u64) -> (u32, u32) {
    let mapping = maps.iter().find(|m| address >= m.at && address - m.at < m.len);
    mapped_root_view(mapping, pairs, first, address)
}

/// Root view in a pool-table snapshot: starts are strictly increasing and no range extends
/// past the next start, so only the last mapping starting at or before the address can contain it.
#[cfg(test)]
fn sorted_root_view(maps: &[Mapping], pairs: &mut Pairs, first: u32, address: u64) -> (u32, u32) {
    let end = maps.partition_point(|m| m.at <= address);
    let mapping = maps[..end].last().filter(|m| address - m.at < m.len);
    mapped_root_view(mapping, pairs, first, address)
}

pub(super) fn mapped_root_view(mapping: Option<&Mapping>, pairs: &mut Pairs, first: u32, address: u64) -> (u32, u32) {
    let null = (pair_index(first, 0), 0);
    let Some(m) = mapping else { return null };
    let pool_at = m.offset + (address - m.at);
    let key = PairKey {
        pool: m.pool,
        base: m.base,
        len: m.pool_len,
        chunk: pool_at / CHUNK,
    };
    match pairs.pair(key) {
        Some(pair) => (pair_index(first, pair), (pool_at - key.chunk * CHUNK) as u32),
        None => null,
    }
}

/// `VICTORY_TRACE_ROOT_SPAN`: how far from `address` the title's range stays linear in the pool the
/// mapping holding it shows (the next mappings start where the last ended, in the same pool, at the
/// pool offset that follows), up to `limit` bytes; the mapping holding `address`, and the one the
/// title has at the end of the run (`None`: nothing mapped there). `None` if nothing holds `address`.
/// `sorted`: indices of `maps` by address.
pub fn linear_run(
    maps: &[Mapping],
    sorted: &[usize],
    address: u64,
    limit: u64,
) -> Option<(u64, Mapping, Option<Mapping>)> {
    let m = *maps.iter().find(|m| address >= m.at && address - m.at < m.len)?;
    let (mut end, mut pool_end) = (m.at + m.len, m.offset + m.len);
    loop {
        let from = sorted.partition_point(|&i| maps[i].at < end);
        let mut at_end = sorted[from..].iter().map(|&i| maps[i]).take_while(|n| n.at == end);
        let first = at_end.clone().next();
        match at_end.find(|n| n.pool == m.pool && n.offset == pool_end) {
            Some(n) if end - address < limit => (end, pool_end) = (end + n.len, pool_end + n.len),
            _ => return Some((end - address, m, first)),
        }
    }
}

/// What the translated shader does with `words` for the address `va` (the lookup of
/// `emit::lower::memory_entry`): the heap index of the view and the offset in it, or `None` for the
/// null view.
pub fn lookup(words: &[u32], va: u64) -> Option<(u32, u32)> {
    let block = if va >> 40 == 0 {
        words[(va >> 28) as usize] as usize
    } else {
        0
    };
    let at = LEVEL_1 + block * BLOCK + 2 * ((va >> 16) as usize & 0xFFF);
    let (view, start) = (words[at], words[at + 1]);
    (view != words[LEVEL_1]).then(|| (view, (va as u32).wrapping_sub(start)))
}

/// A zeroed GPU buffer of the host's own: the table, the null view. The backend owns its memory,
/// so work submitted with it keeps it after the host replaced it (a table is replaced whenever a
/// mapping changed, and the GPU may run work submitted with the old one much later: a queue held
/// behind a wait).
pub struct Pages<B: ComputeBackend> {
    pub buf: Rc<B::Buffer>,
    ptr: *mut u8,
    len: usize,
}

impl<B: ComputeBackend> Pages<B> {
    /// `bytes` rounded up to 16 KiB (a page on Apple silicon; a multiple of every page size here).
    pub fn new(backend: &B, bytes: usize) -> Result<Pages<B>, String> {
        let len = bytes.max(1).next_multiple_of(1 << 14);
        let (buf, ptr) = backend.host_buffer(len)?;
        Ok(Pages {
            buf: Rc::new(buf),
            ptr,
            len,
        })
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Bytes it holds (rounded up to 16 KiB).
    pub fn bytes(&self) -> usize {
        self.len
    }

    /// Writes `words` from dword `at` on. The GPU may be reading the buffer: a reader sees each
    /// dword old or new (one aligned store a dword), not the words all old or all new.
    pub fn write_at(&self, at: usize, words: &[u32]) {
        assert!((at + words.len()) * 4 <= self.len);
        for (i, &w) in words.iter().enumerate() {
            // SAFETY: inside the buffer `buf` keeps, which is dword aligned.
            unsafe { self.ptr.cast::<u32>().add(at + i).write_volatile(w) };
        }
    }

    /// Writes `words` from the start.
    pub fn write(&self, words: &[u32]) {
        assert!(words.len() * 4 <= self.len);
        // SAFETY: inside the buffer `buf` keeps; the GPU does not read a table before it is bound.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), self.ptr.cast(), words.len()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_root_view_is_a_pair_and_an_offset_in_it() {
        let maps = [
            map(0x10_0000_0000, 3 * CHUNK, 7, 3 * CHUNK, 0),
            map(0x20_0000_0000, PAGE, 8, PAGE, 0),
        ];
        let mut pairs = Pairs::default();
        let first = 1000;
        // The second chunk of pool 7, 0x123 bytes in.
        let a = 0x10_0000_0000 + CHUNK + 0x123;
        assert_eq!(root_view(&maps, &mut pairs, first, a), (pair_index(first, 1), 0x123));
        // The same pool and chunk is the same pair; another pool is the next.
        assert_eq!(root_view(&maps, &mut pairs, first, a + 8), (pair_index(first, 1), 0x12b));
        assert_eq!(root_view(&maps, &mut pairs, first, 0x20_0000_0010).0, pair_index(first, 2));
        // Nothing mapped: the null view.
        assert_eq!(root_view(&maps, &mut pairs, first, 0x30_0000_0000), (pair_index(first, 0), 0));
    }

    #[test]
    fn sorted_root_views_match_linear_lookup_at_mapping_and_chunk_boundaries() {
        let start = (1 << 32) + 100;
        let maps = [
            map(start, 0, 1, 16, 0),
            map(start + 2, 8, 2, CHUNK + 4, CHUNK - 4),
            map(start + 10, 3, 3, 32, 17),
            map(start + 14, 0, 4, 16, 0),
            map(start + 17, 2, 2, CHUNK + 4, 1),
        ];
        // Try every prefix so each mapping is also tested as the final entry, and the table can be empty.
        for len in 0..=maps.len() {
            let maps = &maps[..len];
            let (mut sorted, mut linear) = (Pairs::default(), Pairs::default());
            for address in (start - 1..=start + 20).chain([0, u64::MAX]) {
                assert_eq!(
                    sorted_root_view(maps, &mut sorted, FIRST, address),
                    root_view(maps, &mut linear, FIRST, address),
                    "len {len}, address {address:#x}"
                );
                assert_eq!(sorted.keys, linear.keys);
            }
        }
    }

    #[test]
    fn root_view_uses_the_first_match_in_an_unsorted_slice() {
        let maps = [map(12, 4, 1, 32, 23), map(10, 10, 2, 16, 0)];
        let mut pairs = Pairs::default();
        assert_eq!(root_view(&maps, &mut pairs, FIRST, 13), (pair_index(FIRST, 1), 24));
        assert_eq!(pairs.keys[0].unwrap().pool, 1);
    }

    #[test]
    fn a_linear_run_follows_mappings_that_continue_the_pool() {
        // 64 KiB of pool 10 at its offset 0x10000, the next 64 KiB continuing it, then a 64 KiB
        // that is pool 10 elsewhere (an alias), then pool 11.
        let maps = [
            map(0x8_0000_0000, PAGE, 10, 1 << 30, 0x1_0000),
            map(0x8_0002_0000, PAGE, 10, 1 << 30, 0x9_0000),
            map(0x8_0001_0000, PAGE, 10, 1 << 30, 0x2_0000),
            map(0x8_0003_0000, PAGE, 11, PAGE, 0),
        ];
        let mut sorted: Vec<usize> = (0..maps.len()).collect();
        sorted.sort_by_key(|&i| maps[i].at);
        let (run, m, next) = linear_run(&maps, &sorted, 0x8_0000_0100, 1 << 20).unwrap();
        assert_eq!((run, m.offset), (2 * PAGE - 0x100, 0x1_0000));
        assert_eq!(next.map(|n| n.offset), Some(0x9_0000));
        // Stops once the limit is reached; nothing mapped is None.
        assert_eq!(linear_run(&maps, &sorted, 0x8_0000_0100, 0x100).unwrap().0, PAGE - 0x100);
        let (run, _, next) = linear_run(&maps, &sorted, 0x8_0003_0000, 1 << 20).unwrap();
        assert_eq!((run, next.is_none()), (PAGE, true));
        assert!(linear_run(&maps, &sorted, 0x9_0000_0000, 1 << 20).is_none());
    }

    /// Where a heap of a thousand title entries has its pairs.
    const FIRST: u32 = 1000;

    fn map(at: u64, len: u64, pool: usize, pool_len: u64, offset: u64) -> Mapping {
        Mapping {
            at,
            len,
            pool,
            base: 0x1_0000_0000 * pool,
            pool_len,
            offset,
        }
    }

    /// What the lookup says an address of the title is: the pair's view and where in the pool.
    fn resolve(words: &[u32], pairs: &Pairs, va: u64) -> Option<(usize, u64)> {
        let (view, offset) = lookup(words, va)?;
        let key = pairs.keys[((view - FIRST) / 2 - 1) as usize].unwrap();
        Some((key.pool, key.chunk * CHUNK + u64::from(offset)))
    }

    #[test]
    fn a_page_resolves_to_its_pool_offset() {
        let mut pairs = Pairs::default();
        // A pool of 4 MiB shown at two addresses: one above 4 GiB, one below, at an offset.
        let maps = [
            map(0x12_3450_0000, 0x40_0000, 1, 0x40_0000, 0),
            map(0x7000_0000, 0x10_0000, 1, 0x40_0000, 0x2_0000),
            map(0x20_0000_0000, 0x2_0000, 2, 0x2_0000, 0),
        ];
        let words = build(&maps, &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x12_3450_0000), Some((1, 0)));
        assert_eq!(resolve(&words, &pairs, 0x12_3450_1234), Some((1, 0x1234)));
        assert_eq!(resolve(&words, &pairs, 0x12_347F_FFFC), Some((1, 0x2F_FFFC)));
        assert_eq!(resolve(&words, &pairs, 0x7000_0010), Some((1, 0x2_0010)));
        assert_eq!(resolve(&words, &pairs, 0x7000_0000 + 0xF_FFFC), Some((1, 0x2_0000 + 0xF_FFFC)));
        assert_eq!(resolve(&words, &pairs, 0x20_0001_0000), Some((2, 0x1_0000)));
        // Nothing past the mappings, nor in another 256 MiB region, nor above 40 bits.
        for va in [0x12_3490_0000, 0x7010_0000, 0x6FFF_0000, 0x55_0000_0040, 1 << 41, 0] {
            assert_eq!(resolve(&words, &pairs, va), None, "{va:#x}");
        }
        // Two views of the pool: one pair.
        assert_eq!(pairs.keys.len(), 2);
    }

    #[test]
    fn a_pool_bigger_than_a_chunk_has_a_pair_per_chunk_and_offsets_stay_in_the_view() {
        let mut pairs = Pairs::default();
        let len = 3 * CHUNK + 0x3_0000;
        let at = 0x40_0000_0000u64;
        let words = build(&[map(at, len, 5, len, 0)], &mut pairs, FIRST);
        for off in [0, CHUNK - 4, CHUNK, CHUNK + 4, 2 * CHUNK + 0x1234, 3 * CHUNK + 0x2_FFFC] {
            let r = resolve(&words, &pairs, at + off).unwrap();
            assert_eq!(r, (5, off), "{off:#x}");
            // The offset is inside the view of that pair (a chunk and a page long, to the pool's end).
            let (view, o) = lookup(&words, at + off).unwrap();
            let key = pairs.keys[((view - FIRST) / 2 - 1) as usize].unwrap();
            let (_, size) = Pairs::span(&key);
            assert!(u64::from(o) + 4 <= u64::from(size), "{off:#x}");
        }
        assert_eq!(pairs.keys.len(), 4);
    }

    #[test]
    fn mappings_off_the_64k_grid_leave_null_pages() {
        let mut pairs = Pairs::default();
        // From 16 KiB past a 64 KiB boundary, to a 16 KiB-aligned pool offset: the first partial
        // page is null, the four whole ones after it are not, and the last partial one is null.
        let words = build(&[map(0x1000_4000, 0x4_C000, 1, 0x10_0000, 0x4000)], &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x1000_4000), None);
        for va in [0x1001_0000u64, 0x1004_FFFC] {
            assert_eq!(resolve(&words, &pairs, va), Some((1, 0x4000 + va - 0x1000_4000)));
        }
        assert_eq!(resolve(&words, &pairs, 0x1005_0000), None);
        assert_eq!(resolve(&words, &pairs, 0x1010_0000), None);
        // A mapping whose last page runs past the pool is null there.
        let mut pairs = Pairs::default();
        let words = build(&[map(0x2000_0000, 0x2_0000, 1, 0x1_8000, 0)], &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x2000_0000), Some((1, 0)));
        assert_eq!(resolve(&words, &pairs, 0x2001_0000), None);
    }

    #[test]
    fn pages_mapped_one_piece_at_a_time_are_one_run() {
        // The title reports an allocation one 16 KiB mapping at a time: four of them cover one
        // 64 KiB page, continuing one another in title memory and in pool 3's bytes.
        let page = 0x3000_0000u64;
        let maps: Vec<Mapping> = (0..4)
            .map(|k| map(page + k * 0x4000, 0x4000, 3, 0x10_0000, 0x2_0000 + k * 0x4000))
            .collect();
        let mut pairs = Pairs::default();
        let words = build(&maps, &mut pairs, FIRST);
        // Every byte of the page reads the pool where the title has it, as the pool table resolves
        // the same range as one buffer.
        for off in [0, 0x3FFC, 0x4000, 0xFFFF] {
            assert_eq!(resolve(&words, &pairs, page + off), Some((3, 0x2_0000 + off)), "{off:#x}");
        }
        // Nothing past the last piece.
        assert_eq!(resolve(&words, &pairs, page + PAGE), None);
        // A piece that continues the title's addresses but not the pool's: the page is not one run.
        let mut split = maps.clone();
        split[2].offset = 0x9_0000;
        let mut pairs = Pairs::default();
        let words = build(&split, &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, page + 0x1000), None);
        // A page whose first piece is not mapped stays null: the title has nothing at its start.
        let mut pairs = Pairs::default();
        let words = build(&maps[1..], &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, page + 0x5000), None);
        // Mappings given in another order build the same table (the pool table hands them over in
        // title order; a test may not).
        let mut reversed = maps.clone();
        reversed.reverse();
        let mut pairs = Pairs::default();
        let words = build(&reversed, &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, page + 0xFFFF), Some((3, 0x2_FFFF)));
    }

    #[test]
    fn a_tally_counts_the_pages_the_table_cannot_show() {
        let page = 0x5000_0000u64;
        // One page in four same-pool pieces: readable, and only by joining them.
        let pieces: Vec<Mapping> = (0..4)
            .map(|k| map(page + k * 0x4000, 0x4000, 1, 0x10_0000, 0x1_0000 + k * 0x4000))
            .collect();
        let mut pairs = Pairs::default();
        let words = build(&pieces, &mut pairs, FIRST);
        assert_eq!(tally(&pieces, &words), PageTally { pages: 1, readable: 1, joined: 1, split: 0, gap: 0 });
        // The same page in two pools: the title maps every byte, but no one view shows them.
        let mut two = pieces.clone();
        two[2].pool = 2;
        let mut pairs = Pairs::default();
        let words = build(&two, &mut pairs, FIRST);
        assert_eq!(tally(&two, &words), PageTally { pages: 1, readable: 0, joined: 0, split: 1, gap: 0 });
        // A page with a hole the title really left: bytes missing at its very start.
        let mut pairs = Pairs::default();
        let words = build(&pieces[1..], &mut pairs, FIRST);
        assert_eq!(tally(&pieces[1..], &words), PageTally { pages: 1, readable: 0, joined: 0, split: 0, gap: 1 });
        // A page one mapping covers whole needs no joining.
        let whole = [map(page, PAGE, 3, PAGE, 0)];
        let mut pairs = Pairs::default();
        let words = build(&whole, &mut pairs, FIRST);
        assert_eq!(tally(&whole, &words), PageTally { pages: 1, readable: 1, joined: 0, split: 0, gap: 0 });
    }

    #[test]
    fn views_are_stable_and_bounded() {        let mut pairs = Pairs::default();
        let a = map(0x1000_0000, 0x1_0000, 1, 0x1_0000, 0);
        build(&[a], &mut pairs, FIRST);
        let first = pairs.keys.clone();
        // The same pool at another address, and one more pool: earlier pairs keep their numbers.
        build(&[a, map(0x2000_0000, 0x1_0000, 2, 0x1_0000, 0)], &mut pairs, FIRST);
        assert_eq!(pairs.keys[..1], first[..]);
        assert_eq!(pairs.keys.len(), 2);
        // Past the heap's reserved entries pages are null, not another pair's.
        let many: Vec<Mapping> = (0..PAIRS as u64 * 2)
            .map(|i| map(0x10_0000_0000 + i * PAGE, PAGE, 100 + i as usize, PAGE, 0))
            .collect();
        let words = build(&many, &mut pairs, FIRST);
        assert_eq!(pairs.keys.len(), PAIRS - 1);
        assert_eq!(resolve(&words, &pairs, 0x10_0000_0000 + 2 * PAGE).unwrap().0, 102);
        assert_eq!(resolve(&words, &pairs, 0x10_0000_0000 + (PAIRS as u64 * 2 - 1) * PAGE), None);
        // The pairs and the translator's own null views fit below the clamp.
        assert!(fits(1_000_000) && !fits(victory_shader::emit::NULL_BUFFER_VIEW as usize));
    }

    #[test]
    fn a_released_pools_pairs_go_to_the_next_pools() {
        let mut pairs = Pairs::default();
        // Every pair taken.
        let many: Vec<Mapping> = (0..PAIRS as u64)
            .map(|i| map(0x10_0000_0000 + i * PAGE, PAGE, 100 + i as usize, PAGE, 0))
            .collect();
        build(&many, &mut pairs, FIRST);
        assert_eq!(pairs.keys.len(), PAIRS - 1);
        let fresh = map(0x20_0000_0000, PAGE, 999, PAGE, 0);
        assert_eq!(resolve(&build(&[fresh], &mut pairs, FIRST), &pairs, 0x20_0000_0000), None);
        // Pool 105 goes: its pair (6) is the new pool's, and heaps must write their pairs again.
        let epoch = pairs.epoch;
        pairs.release(105);
        assert_ne!(pairs.epoch, epoch);
        let words = build(&[fresh], &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x20_0000_0010), Some((999, 0x10)));
        assert_eq!(lookup(&words, 0x20_0000_0000).unwrap().0, pair_index(FIRST, 6));
        assert_eq!(pairs.keys.len(), PAIRS - 1);
    }
}
