//! Title textures on the native host: the console's tiled layouts (`victory_formats::texture`)
//! detiled from title memory into backend textures.
//!
//! A title texture lives in title memory in the console's swizzled layout; the backend's texture is
//! Apple's own. Each texture the host samples is detiled into a linear staging buffer and uploaded,
//! mip by mip and slice by slice, when the memory's content changed since the last time (a hash of
//! the memory, taken when work is submitted). Render targets and depth targets are the GPU's: they
//! start empty and are not read from memory. With `VICTORY_TEXTURE_BACK=1`, textures compute writes
//! are retiled back into title memory ([`Back`]).
//!
//! shortcut: the detile runs on the CPU (`Layout::detile`, the code the D3D12 host uses too) and
//! a thread hashes the memory of every sampled texture round after round (see `Watch`); a compute detile that reads the
//! pool buffer directly and a per-page change test (what `watch.rs` gives the D3D12 host) are the
//! next step. A change detiles the whole texture.
//!
//! A submission's textures are read, hashed and detiled on a few threads without the host's lock
//! ([`prepare`]); only the probes that pick them ([`Title::sync_jobs`]) and the uploads
//! ([`Title::sync_apply`]) hold it. Under the lock that work stopped every thread of a title that
//! streams (hundreds of MiB of textures a second), for most of each second.

use super::descriptors::format;
use super::host::NativeHost;
use super::memory::PoolTable;
use super::list::NativeList;
use super::title::Title;
use crate::content_hash::ContentHash;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use victory_backend::resources::*;
use victory_formats::texture::{Dimension, Layout, Swizzle};

/// The pages a decoded copy's title mapping gave back, on macOS (`victory_platform::faults`).
/// No other host protects title pages, so every call is a no-op there and `active` is false.
#[cfg(target_os = "macos")]
mod faults {
    pub use victory_platform::faults::*;
}
#[cfg(not(target_os = "macos"))]
mod faults {
    pub fn active() -> bool {
        false
    }
    pub fn release(_: usize, _: usize) {}
    pub fn restore(_: usize, _: usize) {}
    pub fn all_out(_: usize, _: usize) -> bool {
        false
    }
}

/// Who in the title asked for what is being made (`VICTORY_TRACE_TARGETS`): the macOS host's
/// caller hook, and nothing where there is none.
fn caller() -> Vec<usize> {
    #[cfg(target_os = "macos")]
    {
        super::glue::title_caller()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}

/// `D3D12_RESOURCE_FLAGS`.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED_ACCESS: u32 = 4;

/// A title texture as the title made it: its description and where its memory is.
#[derive(Clone)]
pub struct TextureInfo {
    pub dimension: Dimension,
    /// DXGI format number, the console's (before `pc_format`).
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// Array length, or depth for 3D.
    pub depth_or_layers: u32,
    pub flags: u32,
    /// The layout of plane 0 (depth, for depth and stencil).
    pub layout: Layout,
    /// The description's layout word before it was resolved: 0 is the console's own choice,
    /// taken from the description's flags (`layout` has the swizzle it resolved to).
    pub layout_word: u32,
    pub memory: *const u8,
    pub size: usize,
}

impl TextureInfo {
    pub fn render_target(&self) -> bool {
        self.flags & (RENDER_TARGET | DEPTH_STENCIL) != 0
    }

    /// The backend's description of it; `None` for formats or shapes the host does not know.
    pub fn desc(&self) -> Option<TextureDesc> {
        let depth = self.flags & DEPTH_STENCIL != 0;
        let mut format = format(self.format)?;
        // A typeless 32-bit float is a depth target when the texture is one.
        if depth && format == Format::R32Float {
            format = Format::D32Float;
        }
        if depth && format == Format::R16Unorm {
            format = Format::D16Unorm;
        }
        let (dim, d) = match self.dimension {
            Dimension::Tex3d => (TexDim::D3, self.depth_or_layers as usize),
            _ if self.depth_or_layers > 1 => (TexDim::D2Array, self.depth_or_layers as usize),
            _ => (TexDim::D2, 1),
        };
        Some(TextureDesc {
            dim,
            format,
            width: self.width as usize,
            height: self.height.max(1) as usize,
            depth_or_layers: d,
            mips: self.layout.mips.len().max(1),
            render_target: self.render_target(),
            unordered_access: self.flags & UNORDERED_ACCESS != 0,
        })
    }

    /// Top mip levels left off the host texture (0 or 1; see [`drop_top_mip`]).
    pub fn dropped_levels(&self) -> usize {
        dropped_levels(
            drop_top_mip(),
            self.render_target() || self.flags & UNORDERED_ACCESS != 0,
            self.dimension == Dimension::Tex3d,
            self.layout.mips.len(),
            self.width.max(self.height),
        )
    }

    /// [`Self::desc`] with `dropped` leading levels left off (at least one host mip kept): what the
    /// host texture is made as.
    pub fn host_desc_with(&self, dropped: usize) -> Option<TextureDesc> {
        let mut desc = self.desc()?;
        let n = dropped.min(desc.mips.saturating_sub(1));
        desc.width = (desc.width >> n).max(1);
        desc.height = (desc.height >> n).max(1);
        desc.mips -= n;
        Some(desc)
    }

    /// [`Self::host_desc_with`] for the levels [`drop_top_mip`] leaves off.
    pub fn host_desc(&self) -> Option<TextureDesc> {
        self.host_desc_with(self.dropped_levels())
    }
}

/// Largest side (texels) from which a sampled texture loses its top level under [`drop_top_mip`].
const DROP_TOP_MIP_MIN_SIDE: u32 = 1024;

/// `VICTORY_TEXTURE_DROP_TOP_MIP=1`: large sampled textures are made on the host without their
/// top mip (a quarter of the memory of that level, about 75% of the texture's): for hosts whose
/// unified memory cannot hold the title's full set (an iPad's ~9 GiB before it swaps).
/// Sampling with implicit LOD then uses the next level, a little blurrier up close. Shortcut, not
/// console behaviour: explicit-LOD samples come out one level blurrier and integer texel loads
/// address the smaller texture; render targets, UAV and 3D textures are never touched.
pub fn drop_top_mip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TEXTURE_DROP_TOP_MIP").is_ok_and(|v| v == "1"))
}

/// Levels to leave off: one for a large, sampled-only 2D texture that has a level below its top.
fn dropped_levels(enabled: bool, written_by_gpu: bool, volume: bool, mips: usize, largest_side: u32) -> usize {
    let eligible = enabled && !written_by_gpu && !volume && mips >= 2 && largest_side >= DROP_TOP_MIP_MIN_SIDE;
    usize::from(eligible)
}

/// Number of leading title mip levels (`dropped_levels()` or more) that should be left off `info`
/// right now because leading fine mips above the tail have zero mapped pages in `pools`. Outside
/// pools (`!pools.any_mapped(memory, size)`), falls back to `info.dropped_levels()`.
pub(super) fn unmapped_leading_mips(info: &TextureInfo, pools: &PoolTable) -> usize {
    let floor = info.dropped_levels();
    let Some(desc) = info.desc() else {
        return floor;
    };
    if info.render_target()
        || desc.format.is_depth()
        || info.flags & UNORDERED_ACCESS != 0
        || info.dimension == Dimension::Tex3d
        || !info.layout.addressable()
        || desc.mips < 2
        || !pools.any_mapped(info.memory as u64, info.size as u64)
    {
        return floor;
    }
    let slices = info.layout.slices.max(1) as usize;
    let slice_size = info.layout.slice_size as usize;
    let limit = info.layout.first_tail_mip.min(desc.mips as u32 - 1);
    let mut unmapped = 0usize;
    for mip in 0..limit {
        let mip_start = info.layout.mips[mip as usize].offset as usize;
        let mip_end = if mip > 0 {
            info.layout.mips[mip as usize - 1].offset as usize
        } else {
            slice_size
        };
        if mip_start >= mip_end {
            break;
        }
        let len = (mip_end - mip_start) as u64;
        let mapped = (0..slices).any(|s| {
            pools.any_mapped(info.memory as u64 + (s * slice_size + mip_start) as u64, len)
        });
        if mapped {
            break;
        }
        unmapped += 1;
    }
    unmapped.max(floor).min(desc.mips - 1)
}

/// A view's title mip range `(first, count)` on a host texture made without its top `dropped`
/// levels (`host_mips` left): the same levels where they exist, else the nearest host level.
pub fn host_mip_range(first: usize, count: usize, dropped: usize, host_mips: usize) -> (usize, usize) {
    if dropped == 0 || host_mips == 0 {
        return (first, count);
    }
    let end = (first + count.max(1)).saturating_sub(dropped).clamp(1, host_mips);
    let start = first.saturating_sub(dropped).min(end - 1);
    (start, end - start)
}

/// Uploads and change probes must see the same bytes, including across holes in title memory.
/// False if they could not be read at once (a hole: zeros there). When `known_partial` says an
/// earlier whole read of this texture met an unmapped or protected range, skips the initial
/// whole-range kernel read that would copy the readable prefix and then fail at the hole, and
/// returns `true` again once every page is readable.
fn read_texture_bytes_known(at: *const u8, out: &mut [u8], known_partial: bool) -> bool {
    #[cfg(target_os = "macos")]
    return read_texture_chunk(
        at,
        out,
        known_partial,
        super::safe_read,
        victory_platform::mem::next_region,
    );
    #[cfg(not(target_os = "macos"))]
    {
        let _ = known_partial;
        let read = super::safe_read(at, out);
        if !read {
            out.fill(0);
        }
        read
    }
}

#[cfg(target_os = "macos")]
fn read_texture_chunk(
    at: *const u8,
    out: &mut [u8],
    known_partial: bool,
    mut read: impl FnMut(*const u8, &mut [u8]) -> bool,
    mut next_region: impl FnMut(usize) -> Option<victory_platform::mem::Region>,
) -> bool {
    use victory_platform::mem::{Protect, page_size};

    if out.is_empty() {
        return true;
    }
    let base = at.addr();
    let page = page_size();
    let single_page = out.len() <= page - base % page;
    if (!known_partial || single_page) && read(at, out) {
        return true;
    }
    if single_page {
        out.fill(0);
        return false;
    }
    let Some(end) = base.checked_add(out.len()) else {
        out.fill(0);
        return false;
    };
    let mut flush_readable = |start: usize, stop: usize, out: &mut [u8], whole: &mut bool| {
        if start >= stop {
            return;
        }
        if !read(start as *const u8, &mut out[start - base..stop - base]) {
            *whole = false;
            // The mapping may have changed after the query. Salvage this span page by page.
            let mut pos = start;
            while pos < stop {
                let n = (page - pos % page).min(stop - pos);
                let part = &mut out[pos - base..pos - base + n];
                if !read(pos as *const u8, part) {
                    part.fill(0);
                }
                pos += n;
            }
        }
    };
    let mut cur = base;
    let mut whole = true;
    let mut run_start: Option<usize> = None;
    while cur < end {
        let Some(region) = next_region(cur) else {
            break;
        };
        let start = cur.max(region.base);
        let stop = end.min(region.base.saturating_add(region.len));
        if start >= stop {
            break;
        }
        if cur < start {
            if let Some(rs) = run_start.take() {
                flush_readable(rs, cur, out, &mut whole);
            }
            out[cur - base..start - base].fill(0);
            whole = false;
        }
        if region.protect.0 & Protect::READ.0 == 0 {
            if let Some(rs) = run_start.take() {
                flush_readable(rs, start, out, &mut whole);
            }
            out[start - base..stop - base].fill(0);
            whole = false;
        } else if run_start.is_none() {
            run_start = Some(start);
        }
        cur = stop;
    }
    if let Some(rs) = run_start.take() {
        flush_readable(rs, cur, out, &mut whole);
    }
    if cur < end {
        out[cur - base..].fill(0);
        whole = false;
    }
    whole
}

/// Reads `[at, at + out.len())` of title memory: coalescing mapped spans through [`PoolTable`]
/// first when a pool backs the range (one fault-checked `safe_read` per contiguous mapped span and
/// zero syscalls over unmapped fine-mip holes), else through [`read_texture_bytes_known`].
fn read_bytes(memory: Option<&PoolTable>, at: usize, out: &mut [u8], known_partial: bool) -> bool {
    if let Some(whole) = memory.and_then(|m| {
        m.read_mapped_with(at, out, |span_at, dst| {
            if super::safe_read(span_at as *const u8, dst) {
                true
            } else {
                let _ = read_texture_bytes_known(span_at as *const u8, dst, true);
                false
            }
        })
    }) {
        return whole;
    }
    read_texture_bytes_known(at as *const u8, out, known_partial)
}

/// Reads `len` bytes of title memory (zeros where it cannot be read: a page the title never mapped).
pub fn read_memory(at: *const u8, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let _ = read_into(None, at as usize, &mut out, false);
    out
}

/// Fills `out` from title memory at `at`, with zeros where it cannot be read; whether some of it
/// could not be read at once (`VICTORY_TRACE_TEXTURE_SYNC=1` counts those textures).
fn read_into(memory: Option<&PoolTable>, at: usize, out: &mut [u8], known_partial: bool) -> bool {
    !read_bytes(memory, at, out, known_partial)
}

/// Host texels as title memory holds them: the reverse of [`to_host`].
fn from_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_right(4).to_le_bytes();
        }
    }
}

/// Title texels as the host format holds them: DXGI `B4G4R4A4_UNORM` (115) has blue in the low
/// nibble and alpha in the top one, Metal's `ABGR4Unorm` alpha low and red top. The host's own
/// writes and reads agree with each other, so only bytes from title memory are converted.
pub fn to_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_left(4).to_le_bytes();
        }
    }
}

/// One mip of one slice, detiled: what `ResourceBackend::upload_texture` takes, and what the
/// decoded-texture cache keeps (`texture_cache`).
#[derive(Clone)]
pub(crate) struct Level {
    pub(crate) mip: usize,
    pub(crate) slice: usize,
    pub(crate) texels: Texels,
    pub(crate) row: usize,
    pub(crate) slice_pitch: usize,
}

/// A level's texels: owned (what a fresh detile makes, and what a deflated cache entry is
/// inflated into), or a window of a cache entry's mapping — its pages are clean and file-backed,
/// so an upload reads them without a copy of the level in anonymous memory (`mmap`).
#[derive(Clone)]
pub(crate) enum Texels {
    Owned(Vec<u8>),
    Mapped(std::sync::Arc<crate::mmap::Map>, std::ops::Range<usize>),
}

impl Texels {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Texels::Owned(bytes) => bytes,
            Texels::Mapped(map, range) => &map.bytes()[range.clone()],
        }
    }

    /// Whether the texels are a cache entry's mapping rather than the heap (tests: a cache hit
    /// uploads from the file, a decode from memory).
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn is_mapped(&self) -> bool {
        matches!(self, Texels::Mapped(..))
    }
}

impl From<Vec<u8>> for Texels {
    fn from(bytes: Vec<u8>) -> Self {
        Texels::Owned(bytes)
    }
}

fn is_all_zero(bytes: &[u8]) -> bool {
    let (chunks, tail) = bytes.as_chunks::<64>();
    chunks.iter().all(|&c| c == [0u8; 64]) && tail.iter().all(|&b| b == 0)
}

/// Pages (16 KiB, the release path's own granularity) of title memory a *host* writer — a
/// DirectStorage read, a file read, XG tiling — wrote, each with the serial of the write that last
/// touched it. It is how the host tells the bytes of a texture that arrived through the title's
/// streamer from ones the title's own code stored: the report counts the decoded copies by it
/// (`DecodedReport::streamed`), and a streamed copy's console bytes are given back to the OS as
/// soon as they are uploaded (`Title::release_streamed`), since a streamed texture is read by the
/// GPU and not by the CPU again.
static HOST_WRITES: std::sync::Mutex<std::collections::BTreeMap<usize, u64>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Serial of the next host write: monotonic, so "written since" is one comparison.
static WRITE_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes of title memory host writers wrote since the start (cumulative: the streamed total).
static STREAMED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Pages [`HOST_WRITES`] keeps at most (2^20 × 16 KiB = 16 GiB of written memory): past it the
/// entries older than the last 2^20 writes go, and a texture read from memory written longer ago
/// is not counted as streamed (it is not released: the conservative side).
const HOST_WRITES_MAX: usize = 1 << 20;

/// The serial of host writes so far: a texture read records it, and the next read asks
/// [`host_wrote_since`] about the writes after it.
pub(crate) fn write_serial() -> u64 {
    WRITE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Bytes of title memory host writers (DirectStorage staging, file reads, XG tiling) wrote since
/// the start.
pub fn streamed_bytes() -> u64 {
    STREAMED_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Records that a host writer wrote `[at, at + len)`: [`crate::title_memory_write`] calls it
/// before the write. Cheap (a map insert per page) and takes no lock the GPU thread waits on.
pub(crate) fn host_wrote(at: usize, len: usize) {
    const PAGE: usize = 1 << 14;
    if len == 0 {
        return;
    }
    let serial = WRITE_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    STREAMED_BYTES.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
    let mut writes = HOST_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let mut page = at & !(PAGE - 1);
    let end = at.saturating_add(len);
    while page < end {
        writes.insert(page, serial);
        page = page.saturating_add(PAGE);
        if page == 0 {
            break;
        }
    }
    if writes.len() > HOST_WRITES_MAX {
        let floor = serial.saturating_sub(HOST_WRITES_MAX as u64);
        writes.retain(|_, s| *s >= floor);
    }
}

/// Whether a host writer wrote any page of `[at, at + len)` after `since` (a serial from
/// [`write_serial`]).
pub(crate) fn host_wrote_since(at: usize, len: usize, since: u64) -> bool {
    const PAGE: usize = 1 << 14;
    let end = at.saturating_add(len);
    let mut page = at & !(PAGE - 1);
    if page >= end || len == 0 {
        return false;
    }
    let writes = HOST_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    while page < end {
        if writes.get(&page).is_some_and(|&s| s > since) {
            return true;
        }
        page = page.saturating_add(PAGE);
        if page == 0 {
            break;
        }
    }
    false
}

/// Detiles the mips of `tiled` needed to bring a host texture up to date. Mips whose bit in
/// `uploaded_mips` (`mip < 32`) is still clear sit at zero in the fresh host texture, so non-tail
/// mips whose tiled bytes are all zero (unmapped fine mips of a streamed texture) skip staging
/// allocation, CPU detile, and `upload_texture`.
/// `skip`: top levels the host texture was made without ([`TextureInfo::dropped_levels`]); title
/// level `m` becomes host level `m - skip` and the skipped ones are not detiled.
fn detile_changed(
    layout: &Layout,
    dimension: Dimension,
    format: u32,
    tiled: &[u8],
    uploaded_mips: u32,
    skip: usize,
) -> (Vec<Level>, u32, Result<(), String>) {
    if !layout.addressable() {
        return (
            Vec::new(),
            0,
            Err(format!(
                "detile mip 0 slice 0: {:?}",
                victory_formats::texture::Error::Unsupported
            )),
        );
    }
    let slices = if dimension == Dimension::Tex3d { 1 } else { layout.slices };
    let n = layout.desc.element_bytes as usize;
    let slice_size = layout.slice_size as usize;
    let mut levels = Vec::new();
    let mut detiled_mips = 0u32;
    for mip in skip as u32..layout.mips.len() as u32 {
        let bit = if mip < 32 { 1u32 << mip } else { 0 };
        if bit != 0 && uploaded_mips & bit == 0 && mip < layout.first_tail_mip {
            let mip_start = layout.mips[mip as usize].offset as usize;
            let mip_end = if mip > 0 {
                layout.mips[mip as usize - 1].offset as usize
            } else {
                slice_size
            };
            let zero_across_slices = mip_start <= mip_end
                && (0..layout.slices as usize).all(|s| {
                    let base = s * slice_size;
                    tiled
                        .get(base + mip_start..base + mip_end)
                        .is_some_and(is_all_zero)
                });
            if zero_across_slices {
                continue;
            }
        }
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        for slice in 0..slices {
            let mut texels = vec![0u8; slice_pitch * depth];
            if let Err(e) = layout.detile(tiled, mip, slice, &mut texels, row, slice_pitch) {
                return (levels, detiled_mips, Err(format!("detile mip {mip} slice {slice}: {e:?}")));
            }
            to_host(format, &mut texels);
            levels.push(Level { mip: mip as usize - skip, slice: slice as usize, texels: texels.into(), row, slice_pitch });
        }
        detiled_mips |= bit;
    }
    (levels, detiled_mips, Ok(()))
}

/// Detiles every mip of every slice of `tiled` (a texture's whole memory): the levels detiled,
/// and the error that stopped it short if one did.
/// `skip` as for [`detile_changed`].
fn detile(layout: &Layout, dimension: Dimension, format: u32, tiled: &[u8], skip: usize) -> (Vec<Level>, Result<(), String>) {
    let (levels, _, result) = detile_changed(layout, dimension, format, tiled, u32::MAX, skip);
    (levels, result)
}

fn upload_levels<B: ResourceBackend>(backend: &B, tex: &B::Texture, levels: &[Level]) {
    for l in levels {
        backend.upload_texture(tex, l.mip, l.slice, l.texels.as_slice(), l.row, l.slice_pitch);
    }
}

/// Detiles every mip of every slice of `tiled` (the texture's whole memory) into `tex`.
pub fn upload<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
) -> Result<(), String> {
    upload_from(backend, tex, info, tiled, 0)
}

/// [`upload`] into a host texture made without the title's top `skip` levels.
pub fn upload_from<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
    skip: usize,
) -> Result<(), String> {
    let (levels, result) = detile(&info.layout, info.dimension, info.format, tiled, skip);
    upload_levels(backend, tex, &levels);
    result
}

/// The reverse of [`upload`]: every mip of every slice of `tex`, retiled into `tiled` (the
/// texture's whole memory; bytes between subresources stay as they are).
fn download<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &mut [u8],
) -> Result<(), String> {
    let layout = &info.layout;
    let slices = if info.dimension == Dimension::Tex3d {
        1
    } else {
        layout.slices
    };
    let n = layout.desc.element_bytes as usize;
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        let mut linear = vec![0u8; slice_pitch * depth];
        for slice in 0..slices {
            backend.read_texture(tex, mip as usize, slice as usize, &mut linear, row, slice_pitch);
            from_host(info.format, &mut linear);
            layout
                .tile(tiled, mip, slice, &linear, row, slice_pitch)
                .map_err(|e| format!("tile mip {mip} slice {slice}: {e:?}"))?;
        }
    }
    Ok(())
}

/// The largest texture copied back (the ocean maps are small; a guess until a world run's
/// "copied back" lines show what is written).
const BACK_LIMIT: usize = 4 << 20;

/// Textures compute work wrote, to be copied back into title memory once it completes
/// (`VICTORY_TEXTURE_BACK=1`). Host textures are the GPU's own, so without this the title's CPU
/// reads what was there before (VIctory-macOS issue #46: boats sink).
///
/// shortcut: copied back before a fence value is shown to the title; a title that instead polls a
/// label the GPU writes into memory may read a frame late.
pub(super) struct Back<B: PresentBackend> {
    /// A fence of ours per title queue, and the last value queued on it.
    fences: HashMap<usize, (Rc<B::Fence>, u64)>,
    /// Submissions not yet copied back: reached when the fence has the value.
    pending: Vec<(Rc<B::Fence>, u64, Vec<usize>)>,
    /// Textures already logged.
    logged: std::collections::HashSet<usize>,
}

impl<B: PresentBackend> Default for Back<B> {
    fn default() -> Self {
        Back {
            fences: HashMap::new(),
            pending: Vec::new(),
            logged: Default::default(),
        }
    }
}

impl<B: PresentBackend> Back<B> {
    /// The signal the next work of title queue `key` carries (`None` with copy-back off): in the
    /// work's own command buffer, so it does not hang on the order command buffers complete in.
    pub(super) fn next(&mut self, host: &NativeHost<B>, key: usize) -> Option<(Rc<B::Fence>, u64)> {
        if !host.collecting_written() {
            return None;
        }
        let (fence, value) = match self.fences.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => match host.backend.create_fence(0) {
                Ok(f) => e.insert((Rc::new(f), 0)),
                Err(error) => {
                    tracing::warn!(%error, "no fence to copy textures back by");
                    return None;
                }
            },
        };
        *value += 1;
        Some((fence.clone(), *value))
    }

    /// That work went to the GPU: if it wrote textures, they are copied back once `signal` is
    /// reached.
    pub(super) fn submitted(&mut self, host: &NativeHost<B>, signal: Option<(Rc<B::Fence>, u64)>) {
        let written = host.take_written();
        if let Some((fence, value)) = signal
            && !written.is_empty()
        {
            self.pending.push((fence, value, written));
        }
    }
}

/// Textures whose memory the scanner thread watches, and those it found changed.
///
/// Hashing every sampled texture's memory on every submission (150 ms at 170 MiB, under the
/// title lock the title's other threads wait on) made a title that streams textures slow to a
/// crawl as they accumulated. A thread of its own hashes them in rounds, off the lock, and a
/// submission only looks at the textures it marked.
/// shortcut: a change is seen one round later (a round is as long as the hashing takes, about
/// 100 ms for 150 MiB), so a texture written just before it is sampled may show its old content
/// for a few frames; a per-page write test would make that exact.
#[derive(Default)]
struct Watch {
    /// Title texture key to (address, bytes, whether it is a render target: its first sight is
    /// its baseline, not a change; which texture of that key: [`Tracked::serial`]).
    entries: HashMap<usize, (usize, usize, bool, u64)>,
    /// The pool table of the title host that realized each watched texture, so pool-backed scans
    /// read in userspace rather than trapping into Mach VM per page.
    pools: HashMap<usize, Arc<PoolTable>>,
    dirty: std::collections::HashSet<usize>,
    /// The hash of each texture's memory as last read whole, by the scanner or by a submission.
    /// What a submission read, the host texture holds: the scanner does not report it again (it
    /// used to report every new texture and every write a probe had caught already, and each
    /// report is the whole texture read once more).
    seen: HashMap<usize, ContentHash>,
    /// Textures whose last whole read met an unmapped or protected range (fine mips not mapped yet).
    partial: std::collections::HashSet<usize>,
    /// Bumped whenever a submission begins or applies a whole read of a watched texture, so an
    /// in-flight scanner round that started earlier discards its older (possibly torn) hash.
    epoch: HashMap<usize, u64>,
}

impl Watch {
    /// Texture `key` is gone. Its hash goes with it: the key (an object's address, a placement)
    /// names another texture later, and a render target compared with the old one's hash had
    /// its memory uploaded over what was drawn into it.
    /// The copy's title pages are out: reading them would upload zeros over it, so nothing
    /// watches it until they are back ([`Title::restore_pages`]). Its key is not forgotten: a
    /// scanner round that is mid-read discards its own reading (`known` finds no entry).
    fn skip(&mut self, key: usize) {
        self.released(key);
    }

    fn released(&mut self, key: usize) {
        self.entries.remove(&key);
        self.pools.remove(&key);
        self.dirty.remove(&key);
        self.seen.remove(&key);
        self.partial.remove(&key);
        self.epoch.remove(&key);
    }
}

type Watched = std::sync::Mutex<Option<Watch>>;

static WATCH: Watched = std::sync::Mutex::new(None);

fn watch<R>(f: impl FnOnce(&mut Watch) -> R) -> R {
    watch_in(&WATCH, f)
}

fn watch_in<R>(watched: &Watched, f: impl FnOnce(&mut Watch) -> R) -> R {
    f(watched.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// One round of the scanner: hashes the memory of each texture `watched` lists, without holding
/// it, and marks those that changed since they were last read whole.
fn scan_round(watched: &Watched, buf: &mut [u8]) {
    let list: Vec<(usize, (usize, usize, bool, u64))> = watch_in(watched, |w| {
        let Watch { entries, pools, seen, partial, epoch, .. } = w;
        pools.retain(|k, _| entries.contains_key(k));
        seen.retain(|k, _| entries.contains_key(k));
        partial.retain(|k| entries.contains_key(k));
        epoch.retain(|k, _| entries.contains_key(k));
        entries.iter().map(|(&k, &v)| (k, v)).collect()
    });
    for &(key, entry) in &list {
        let (at, size, target, _) = entry;
        // What it is compared with; `None` (skipped) if it was released since the list.
        let known = |w: &Watch| {
            (w.entries.get(&key) == Some(&entry))
                .then(|| (w.seen.get(&key).copied(), w.epoch.get(&key).copied().unwrap_or(0)))
        };
        let Some((state @ (before, _), was_partial, pool)) = watch_in(watched, |w| {
            known(w).map(|s| (s, w.partial.contains(&key), w.pools.get(&key).cloned()))
        }) else {
            continue;
        };
        let mut h = crate::content_hash::Hasher::new();
        let mut unreadable = false;
        // `VICTORY_TRACE_FRAME`: the round's read and hash time (two clock reads a MiB).
        let (mut read, mut hashed) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for off in (0..size).step_by(buf.len()) {
            let n = buf.len().min(size - off);
            let started = std::time::Instant::now();
            unreadable |= !read_bytes(pool.as_deref(), at + off, &mut buf[..n], was_partial);
            let between = std::time::Instant::now();
            h.update(&buf[..n]);
            read += between - started;
            hashed += between.elapsed();
        }
        let h = h.finalize();
        #[cfg(target_os = "macos")]
        super::hosttime::scanned(size as u64, read, hashed);
        #[cfg(not(target_os = "macos"))]
        let _ = (read, hashed);
        watch_in(watched, |w| {
            // Released, or begun/applied by a submission while it was hashed here: this reading is
            // the older one, and the next round compares with the submission's.
            if known(w) != Some(state) {
                return;
            }
            if unreadable {
                w.partial.insert(key);
            } else {
                w.partial.remove(&key);
            }
            w.seen.insert(key, h);
            if before != Some(h) && !(target && before.is_none()) {
                w.dirty.insert(key);
            }
        });
    }
}

fn start_scanner() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory texture scan".into()).spawn(|| {
            let mut buf = vec![0u8; 1 << 20];
            loop {
                scan_round(&WATCH, &mut buf);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    });
}

/// A texture the host keeps for a title texture.
pub struct Tracked<B: ResourceBackend> {
    pub texture: Rc<B::Texture>,
    info: TextureInfo,
    /// Which texture this is: its key (an object's address, a placement) names another one later.
    serial: u64,
    /// The hash of the memory last uploaded; `None` for targets and before the first upload.
    last: Option<ContentHash>,
    /// Uploads of its memory so far.
    pub uploads: u64,
    /// Bitmask of mip levels (`mip < 32`) uploaded with non-zero texels (or `u32::MAX` for targets
    /// and UAV textures that GPU passes may write). Clear bits sit at zero in the fresh host
    /// texture, so all-zero fine mips of streamed textures skip detile and `upload_texture`.
    uploaded_mips: u32,
    /// Submission serial (`Title::sync_serial`) through which its memory is probed at submission
    /// rather than left to the scanner (0: cold), and the hash of its samples when it was last
    /// read whole (a probe that reads others reports a write).
    hot_until: u64,
    probe: Option<ContentHash>,
    /// Whether its last write showed in the sampled probe (a movie's planes, a streamed texture:
    /// written whole): then it is read whole only every `WHOLE_EVERY` probes (backing off across
    /// consecutive unchanged whole checks), and probes skipped since. Set at each upload
    /// (`sync_apply`), and returned to periodic checks once a follow-up whole read finds no write.
    sampled_sees_writes: bool,
    unchanged_whole: u8,
    skipped: u32,
    /// Whether its last whole read met an unmapped or protected range (fine mips not mapped yet).
    partial: bool,
    /// Its last upload was all zeros: it is probed (`Title::blank_keys`) until written, however
    /// long that takes (a movie's planes are made blank about 40 s before the movie decodes into
    /// them; once left to the scanner, their first frames showed green for a second or two).
    blank: bool,
    /// Render-target views already made, kept so the same attachment keeps its identity.
    #[cfg(target_os = "macos")]
    target_views: Vec<(ViewDesc, Rc<B::Texture>)>,
    /// The GPU wrote it (a copy, draw, clear or dispatch): its host texture holds what title
    /// memory does not, and is never filled from that memory again.
    gpu_written: bool,
    /// The submission ([`Budget::serial`]) it was last named or shown at: the texture budget gives
    /// up the least recently used first.
    used: u64,
    /// The submission whose tables last named it: a copy named in the last [`named_keep`]
    /// submissions is never given up. The budget was taking copies a later draw of the same or the
    /// next frame needed, which were decoded and uploaded again - the world's 400-1600 textures,
    /// 0.9-1.9 GiB a frame.
    last_named: u64,
    /// The GPU epoch ([`Title::gpu_epoch`]) of the last submission that could read it: one that
    /// bound a table whose entries show it, or a view heap a heap-indexing shader of it indexes
    /// ([`Title::keep_named`]). A copy some heap entry shows is given up only once that work has
    /// finished ([`Title::retired_epoch`]), since the GPU reads a heap entry when its draw runs.
    reach: u64,
    /// The presented frame ([`crate::queue::presented`]) it was last used at: a view named it, a
    /// submission in flight could read it through an entry, or its memory changed and was
    /// uploaded — not merely shown by an entry ([`Self::keep_to_budget`]). A copy nothing uses for
    /// [`Budget::idle_frames`] frames is given up (`VICTORY_TEXTURE_IDLE_FRAMES`).
    last_frame: u64,
    /// Leading title mip levels omitted from [`Self::texture`].
    dropped: usize,
    /// The identity of the memory under it ([`memory_identity`]): copies over the same pool pages
    /// mapped at two title addresses are one entry in [`Title::at_memory`].
    at: usize,
    /// Sampled straight from title memory ([`Title::zero_copy`]): no decoded copy, so it holds no
    /// bytes of the budget, is never uploaded or watched, and is never given up.
    zero_copy: bool,
    /// Whether the content its host texture holds arrived through a host writer — a DirectStorage
    /// read, a file read, XG tiling — rather than the title's own stores ([`host_wrote_since`]).
    /// The report splits the decoded copies by it; [`Title::release_streamed`] gives such a copy's
    /// console bytes back to the OS as soon as it is uploaded.
    streamed: bool,
    /// [`write_serial`] when its memory was last read whole: writes after it are what
    /// [`Self::streamed`] is decided by.
    write_serial: u64,
    /// A host write landed in its pages while they were released: the streamer is filling this
    /// texture again, so its pages are not given back eagerly any more (each would fault back into
    /// a whole re-read of it).
    restreamed: bool,
    /// When a view, a submission or an upload last used it: the clock
    /// [`Title::release_quiescent`] gives its pages back on. Frames do not do: a paged world
    /// presents once every several seconds, so 600 frames can be hours.
    last_used_at: std::time::Instant,
}

impl<B: ResourceBackend> Tracked<B> {
    /// Whether the host texture is no longer filled from title memory: the GPU wrote it
    /// ([`Title::gpu_written`]). One never uploaded and not a render target takes its first upload.
    fn kept_from_memory(&self) -> bool {
        self.gpu_written && (self.uploads > 0 || self.info.render_target())
    }

    /// Where its texels are in title memory (tests: `pool` releases what an earlier test left).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn title_memory(&self) -> usize {
        self.info.memory as usize
    }
}

/// The other format of an sRGB pair.
#[cfg(target_os = "macos")]
fn srgb_twin(f: Format) -> Option<Format> {
    Some(match f {
        Format::Rgba8Unorm => Format::Rgba8UnormSrgb,
        Format::Rgba8UnormSrgb => Format::Rgba8Unorm,
        Format::Bgra8Unorm => Format::Bgra8UnormSrgb,
        Format::Bgra8UnormSrgb => Format::Bgra8Unorm,
        _ => return None,
    })
}

/// Submissions a texture whose memory changed stays hot without a further change.
pub(super) const HOT: u32 = 600;
/// Time one submission spends probing hot textures, and again blank ones. Boot movies and UI
/// glyph caches have a few hot textures and a few hundred blank ones (the movie's planes among
/// them, blank until the title decodes into them), all of which a submission must reach: a count
/// limit left a plane's first frames unseen (solid green). When world streaming realizes thousands
/// of blank textures the time bounds the probing and the cursor resumes where the last submission
/// stopped, while the scanner thread catches the rest off the host lock.
///
/// shortcut: past a few thousand blank textures a write is noticed a few submissions late (unless
/// the scanner catches it first); at least `MIN_SYNC_HOT` of each are probed whatever the time.
const PROBE_BUDGET: std::time::Duration = std::time::Duration::from_millis(2);
/// Bytes of changed texture memory one **frame** (between presents) admits from the scanner into
/// its submissions' reads. A world load lists 0.9 to 1.9 GiB a frame (`runs/20261128-092107-w3oQfk`:
/// reasons `scanner` 386 MiB to 1.1 GiB and `first` 25 to 545 MiB) over some 30 submissions, whose
/// whole-memory reads and hashes were 1.7 and 1.2 s of the frame's `prepare_ms`. A texture never
/// uploaded is not counted (its content is what a first draw samples); the rest is read up to this
/// many bytes a frame and the remainder stays in the watch's dirty set for the next, so a streamed
/// frame catches up over a few frames instead of paying all of it in one. The bound is per frame,
/// not per submission: 30 submissions each below a per-submission bound still made 1.1 GiB a frame
/// (the first run of a per-submission `SCAN_BUDGET`). Guess: under the world's own 1.9 GiB.
const SCAN_BUDGET: u64 = 384 << 20;
/// How much of one frame's wall time the host's texture work may take: listing what to sync,
/// reading, hashing and detiling the textures, and uploading them. Bytes are not enough of a
/// bound ([`SCAN_BUDGET`]): the same changed memory is read in a second while it is resident and
/// in twenty when the Mac is swapping it, and that took whole frames past the title's 30 s
/// watchdog on 28 November (frames of 19.6, 20.4 and 37.9 s in a row, the last ending the run at
/// the fail-fast). Past this, `glue::submit` defers the rest of the frame's texture reads: they
/// stay changed, or not uploaded yet, for the next submission, so a texture is at most a frame
/// late instead of ending the run.
pub(super) const FRAME_TEXTURE_WORK: std::time::Duration = std::time::Duration::from_millis(3000);
/// Hot textures a submission probes at least, however long the probes take.
const MIN_SYNC_HOT: usize = 32;
/// Pages of a large hot texture's memory a probe reads, spread over it.
const PROBE_PAGES: usize = 8;
/// Bytes sampled per page when probing a large hot texture.
const PROBE_SAMPLE: usize = 256;
/// Hot textures up to this size whose last write its samples missed are read whole at every probe
/// (a glyph cache gets a few glyphs at a time). Those written so that the samples showed it (a
/// streamed texture, written whole after its blank upload) are read whole now and then
/// (`WHOLE_EVERY`): in the world they are thousands, and read whole at every probe they were 2,000
/// textures and 500 MiB a second, most of what a submission read.
///
/// shortcut: an atlas whose first glyphs reach a sampled page gets its next ones up to
/// `WHOLE_EVERY` probes late, once (then its samples have missed a write); per-sample hashes
/// would tell a partial write from a whole one.
const PROBE_WHOLE: usize = 4 << 20;

/// A hot texture whose writes show in its samples (or whose follow-up whole check after a missed
/// write found nothing further) is read whole once in this many probes, doubling after each
/// further unchanged whole check up to `WHOLE_EVERY << WHOLE_BACKOFF_MAX`, to catch a write that
/// misses the samples (a glyph added to an atlas a movie-like rewrite went through). Reading the
/// movie's chroma planes whole at every submission cost 15 ms a submission, halving the movie's
/// frame rate.
pub(super) const WHOLE_EVERY: u32 = 8;
/// Maximum doubling steps (`WHOLE_EVERY << 5` = 256 probes) for repeated unchanged whole checks
/// while the background scanner continues to watch every texture (a guess).
const WHOLE_BACKOFF_MAX: u32 = 5;
/// Bytes of hot textures one submission reads whole to check them, once `MIN_SYNC_HOT` were
/// probed: the reading is off the host's lock but not free (a guess at what a submission can
/// spare: about 5 ms of one thread under Rosetta).
const WHOLE_BUDGET: usize = 8 << 20;
/// Of those, bytes of the ones read whole only now and then (`WHOLE_EVERY`: their samples showed
/// their last write). In the world they are nearly all of the hot textures and the title submits
/// some 30 times a frame; one that is due and not read stays due (a guess, as above: 1 ms).
const WHOLE_SPARE: usize = 1 << 20;

/// Byte offset of a `PROBE_SAMPLE`-aligned window inside mip `mip` of slice 0 (at its center
/// element so letterboxed images and mip-tail micro-tiles land on real texels rather than zero
/// padding).
fn mip_sample_offset(layout: &Layout, size: usize, mip: u32) -> Option<usize> {
    let m = layout.mips.get(mip as usize)?;
    let byte = layout
        .offset(mip, m.width / 2, m.height / 2, 0)
        .unwrap_or_else(|_| m.offset + m.tail.map_or(0, |t| u64::from(t.offset)));
    let off = (byte as usize) & !(PROBE_SAMPLE - 1);
    (off < size).then_some(off)
}

/// Hash of the bytes a probe of `size` bytes at `at` reads ([`probe_read`]).
fn probe_hash(
    memory: Option<&PoolTable>,
    at: usize,
    size: usize,
    layout: Option<&Layout>,
    whole: bool,
    known_partial: bool,
    scratch: &mut Vec<u8>,
) -> ContentHash {
    let mut hasher = crate::content_hash::Hasher::new();
    probe_read(memory, at, size, layout, whole, known_partial, scratch, |b| {
        hasher.update(b);
    });
    hasher.finalize()
}

/// Whether `PROBE_SAMPLE` bytes in the middle (or in the coarsest mips of a mipmapped texture
/// whose fine mips are still unmapped) of the `size` bytes at `at` are not all zero: a blank
/// texture's probe (a few hundred blank textures are probed every submission).
///
/// shortcut: a first write that misses both the middle and the coarse mip samples (a glyph in a
/// blank single-mip atlas) waits for the scanner; textures written whole or streamed coarse-first
/// are seen at once.
fn probe_nonzero(at: usize, size: usize, layout: &Layout, scratch: &mut Vec<u8>) -> bool {
    if size == 0 {
        return false;
    }
    let mut offsets = [0usize; 4];
    let mut count = 0usize;
    let mut push = |off: usize| {
        if off < size && !offsets[..count].contains(&off) && count < offsets.len() {
            offsets[count] = off;
            count += 1;
        }
    };
    push((size / 2) & !4095);
    if layout.mips.len() > 1 {
        let last = layout.mips.len() as u32 - 1;
        if let Some(off) = mip_sample_offset(layout, size, layout.first_tail_mip.min(last)) {
            push(off);
        }
        if let Some(off) = mip_sample_offset(layout, size, last) {
            push(off);
        }
        if layout.first_tail_mip > 0
            && let Some(off) = mip_sample_offset(layout, size, layout.first_tail_mip - 1)
        {
            push(off);
        }
    }
    for &off in &offsets[..count] {
        let n = PROBE_SAMPLE.min(size - off);
        if scratch.len() < n {
            scratch.resize(n, 0);
        }
        if super::safe_read((at + off) as *const u8, &mut scratch[..n])
            && scratch[..n].iter().any(|&b| b != 0)
        {
            return true;
        }
    }
    false
}

/// Hands `seen` the bytes a probe of `size` bytes at `at` reads: all of them if `whole`, else
/// `PROBE_PAGES` samples of `PROBE_SAMPLE` bytes spread across the texture plus one sample per mip
/// level when `layout` has a mip chain, reusing `scratch`.
///
/// shortcut: when sampled, a write that misses every sampled window goes unseen until a periodic
/// whole probe or the scanner's next round; a texture the title rewrites whole (a movie's planes)
/// or streams mip by mip (reverse-mip GFX10 layout, where mip 0 sits at the top of the buffer) is
/// caught at once without forcing immediate unchanged whole re-reads.
#[allow(clippy::too_many_arguments)]
fn probe_read(
    memory: Option<&PoolTable>,
    at: usize,
    size: usize,
    layout: Option<&Layout>,
    whole: bool,
    known_partial: bool,
    scratch: &mut Vec<u8>,
    mut seen: impl FnMut(&[u8]),
) {
    if whole || size <= PROBE_PAGES * PROBE_SAMPLE {
        const CHUNK: usize = 1 << 20;
        let cap = size.min(CHUNK);
        if scratch.len() < cap {
            scratch.resize(cap, 0);
        }
        for off in (0..size).step_by(CHUNK) {
            let n = CHUNK.min(size - off);
            read_bytes(memory, at + off, &mut scratch[..n], known_partial);
            seen(&scratch[..n]);
        }
        return;
    }
    const PAGE: usize = 4096;
    let pages = size.div_ceil(PAGE);
    let mut offsets = [0usize; PROBE_PAGES + 24];
    let mut count = 0usize;
    let mut push = |off: usize| {
        if off < size && !offsets[..count].contains(&off) && count < offsets.len() {
            offsets[count] = off;
            count += 1;
        }
    };
    for k in 0..PROBE_PAGES {
        push(k * pages / PROBE_PAGES * PAGE);
    }
    if let Some(layout) = layout.filter(|l| l.mips.len() > 1) {
        for mip in 0..layout.mips.len() as u32 {
            if let Some(off) = mip_sample_offset(layout, size, mip) {
                push(off);
            }
        }
    }
    if scratch.len() < PROBE_SAMPLE {
        scratch.resize(PROBE_SAMPLE, 0);
    }
    for &off in &offsets[..count] {
        let n = PROBE_SAMPLE.min(size - off);
        if !super::safe_read((at + off) as *const u8, &mut scratch[..n]) {
            scratch[..n].fill(0);
        }
        seen(&scratch[..n]);
    }
}

/// For the trace: hot textures a probe read whole, and their bytes.
static WHOLE_PROBES: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
/// For the trace: placements that took the place of another at their address, and those of them
/// that had been at that address before.
static REPLACED: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Probes `keys` round-robin from `cursor`, each at most once. An expired deadline, or
/// `WHOLE_BUDGET` of whole reads, stops the round after at least `MIN_SYNC_HOT` probes (no deadline:
/// neither does). Those whose memory changed, and the hot ones to be read whole to tell, go to
/// `dirty`. Hot textures (`hot_serial = Some(sync_serial)`) that stay unchanged for `HOT`
/// submissions cool down and leave without spending a probe once expired; blank ones (`None`) stay
/// until they are written.
#[allow(clippy::too_many_arguments)]
fn probe_round<B: ResourceBackend>(
    memory: &PoolTable,
    textures: &mut Textures<B>,
    keys: &mut Vec<usize>,
    cursor: &mut usize,
    until: Option<std::time::Instant>,
    hot_serial: Option<u64>,
    scratch: &mut Vec<u8>,
    dirty: &mut std::collections::HashSet<usize>,
) {
    if keys.is_empty() {
        *cursor = 0;
        return;
    }
    // Partition invariant:
    // - Before wrap (`wrapped == false`, `start <= idx <= keys.len()`):
    //   `keys[0..start]` is unvisited, `keys[start..idx]` was visited and kept, and
    //   `keys[idx..keys.len()]` is unvisited (`swap_remove(idx)` pulls an unvisited tail element).
    // - After wrap (`wrapped == true`, `0 <= idx <= start <= keys.len()`):
    //   `keys[0..idx]` was visited after wrap, `keys[idx..start]` is unvisited, and
    //   `keys[start..keys.len()]` was visited before wrap. Decrementing `start`, swapping `idx`
    //   with `start` (the last unvisited slot), and `swap_remove(start)` (pulling an already-
    //   visited tail element into `start`) keeps both visited slices disjoint from `keys[idx..start]`.
    let remove = |keys: &mut Vec<usize>, idx: usize, start: &mut usize, wrapped: bool| {
        if wrapped {
            *start -= 1;
            keys.swap(idx, *start);
            keys.swap_remove(*start);
        } else {
            keys.swap_remove(idx);
        }
    };
    let mut start = *cursor % keys.len();
    let mut idx = start;
    let mut wrapped = false;
    let mut probed = 0;
    // Bytes of the textures to be read whole, and of those among them read so only now and then.
    let mut whole = 0;
    let mut spare = 0;
    loop {
        if !wrapped && idx >= keys.len() {
            idx = 0;
            wrapped = true;
        }
        if wrapped && idx >= start {
            break;
        }
        if probed >= MIN_SYNC_HOT && until.is_some_and(|until| std::time::Instant::now() >= until || whole >= WHOLE_BUDGET) {
            break;
        }
        let key = keys[idx];
        let Some(t) = textures.get_mut(&key) else {
            remove(keys, idx, &mut start, wrapped);
            continue;
        };
        if let Some(sync_serial) = hot_serial {
            if t.hot_until == 0 || t.hot_until < sync_serial {
                t.hot_until = 0;
                remove(keys, idx, &mut start, wrapped);
                continue;
            }
        } else if !t.blank {
            remove(keys, idx, &mut start, wrapped);
            continue;
        }
        probed += 1;
        let at = t.info.memory as usize;
        let leave = if let Some(sync_serial) = hot_serial {
            let sampled = probe_hash(
                Some(memory),
                at,
                t.info.size,
                Some(&t.info.layout),
                false,
                t.partial,
                scratch,
            );
            // Its samples at the last upload (`sync_apply` moves them on).
            let before = t.probe;
            let changed = before != Some(sampled);
            if !changed && t.uploads >= 2 && t.info.size <= PROBE_WHOLE {
                let every = if t.sampled_sees_writes {
                    WHOLE_EVERY << u32::from(t.unchanged_whole).min(WHOLE_BACKOFF_MAX)
                } else {
                    1
                };
                t.skipped = t.skipped.saturating_add(1);
                if t.skipped >= every && !(t.sampled_sees_writes && spare >= WHOLE_SPARE) {
                    t.skipped = 0;
                    if t.sampled_sees_writes {
                        spare += t.info.size;
                    }
                    // Read whole with the textures the submission reads anyway, without the host's
                    // lock ([`prepare`]), and uploaded if that is not what the host texture holds
                    // (a write the samples missed: `sync_apply`). Hashed here, the hot textures
                    // of a title that streams were hundreds of MiB a second under the lock.
                    dirty.insert(key);
                    whole += t.info.size;
                    WHOLE_PROBES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WHOLE_PROBES[1].fetch_add(t.info.size as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if changed {
                t.skipped = 0;
                dirty.insert(key);
            } else if t.hot_until <= sync_serial {
                t.hot_until = 0;
            }
            t.hot_until == 0
        } else {
            // Written at last.
            let written = probe_nonzero(at, t.info.size, &t.info.layout, scratch);
            if written {
                dirty.insert(key);
                t.blank = false;
            }
            written
        };
        if leave {
            remove(keys, idx, &mut start, wrapped);
        } else {
            idx += 1;
        }
    }
    *cursor = idx;
}

/// The key of a texture a view names by address and description (no resource object): a number no
/// object address has (the top bit is set).
pub fn placed_key(address: u64, description: &[u32; 14]) -> usize {
    let mut h = blake3::Hasher::new();
    h.update(&address.to_le_bytes());
    for (i, w) in description.iter().enumerate() {
        // Words 1 and 13 are padding (stack bytes in the title's records).
        h.update(&(if i == 1 || i == 13 { 0 } else { *w }).to_le_bytes());
    }
    let b = h.finalize();
    (u64::from_le_bytes(b.as_bytes()[..8].try_into().expect("8 bytes")) | 1 << 63) as usize
}

/// Whether a placed view takes the host texture of a title texture the GPU writes at its memory
/// (`VICTORY_PLACED_TWIN`, see [`Title::alias_written_twin`]). Off unless set.
pub fn placed_twin() -> bool {
    #[cfg(all(test, target_os = "macos"))]
    if let Some(on) = TEST_PLACED_TWIN.with(std::cell::Cell::get) {
        return on;
    }
    crate::switch::on("VICTORY_PLACED_TWIN")
}

#[cfg(all(test, target_os = "macos"))]
thread_local! {
    /// A test's own value for [`placed_twin`], so that tests running in parallel see the default.
    static TEST_PLACED_TWIN: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(all(test, target_os = "macos"))]
/// Sets [`placed_twin`] for this thread and returns the value before, to restore it with.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn set_placed_twin(on: Option<bool>) -> Option<bool> {
    TEST_PLACED_TWIN.with(|t| t.replace(on))
}

/// Whether `twin`, a title texture at the memory of placed view `view`, has the shape `view` can
/// read: its dimension, width, height, depth or layers and sample count match, its element is the
/// same size (bytes and block) and uncompressed (the host casts every uncompressed format of one
/// texel size to every other, and among block formats only the sRGB and signed twins), and its mip
/// chain is at least as long.
fn same_shape_and_element(view: &TextureInfo, twin: &TextureInfo) -> bool {
    let element = crate::resource::element(twin.format).filter(|&(_, block)| block == 1);
    twin.memory == view.memory
        && element.is_some()
        && element == crate::resource::element(view.format)
        && view.dimension == twin.dimension
        && (view.width, view.height, view.depth_or_layers)
            == (twin.width, twin.height, twin.depth_or_layers)
        && view.layout.desc.samples == twin.layout.desc.samples
        && view.layout.mips.len() <= twin.layout.mips.len()
}

/// Whether `twin`, a title texture at the memory of placed view `view`, is one the GPU writes in a
/// form `view` can read ([`Title::alias_written_twin`]): [`same_shape_and_element`] and a written
/// use. A depth texture is left out: Metal keeps a depth texture's format in a colour view, so the
/// view would not read what it asks for.
pub(crate) fn written_twin(view: &TextureInfo, twin: &TextureInfo) -> bool {
    const WRITTEN: u32 = RENDER_TARGET | DEPTH_STENCIL | UNORDERED_ACCESS;
    twin.flags & WRITTEN != 0
        && twin.flags & DEPTH_STENCIL == 0
        && same_shape_and_element(view, twin)
}

impl<B: PresentBackend> Title<B> {
    /// Turns copying GPU-written textures back into title memory on or off.
    pub fn set_texture_back(&mut self, on: bool) {
        self.host.collect_written(on);
    }

    /// Copies back what completed submissions wrote.
    pub(super) fn copy_back(&mut self) {
        let mut done = Vec::new();
        let backend = &self.host.backend;
        self.back.pending.retain(|(fence, value, keys)| {
            let reached = backend.completed(fence) >= *value;
            if reached {
                done.extend(keys.iter().copied());
            }
            !reached
        });
        done.sort_unstable();
        done.dedup();
        for key in done {
            self.write_back(key);
        }
    }

    /// Retiles the host texture of `key` into its title memory.
    fn write_back(&mut self, key: usize) {
        let pools = self.host.memory();
        // Not a texture (a buffer view), or released since: nothing to copy.
        let Some(t) = self.textures.get_mut(&key) else { return };
        let info = &t.info;
        // Render and depth targets are left out: nothing says the title reads one. A zero-copy
        // texture wrote the title's own pages: there is nothing to copy back.
        if t.zero_copy || info.flags & UNORDERED_ACCESS == 0 || info.render_target() {
            return;
        }
        let described = (info.format, info.width, info.height, info.depth_or_layers, info.size);
        if info.size > BACK_LIMIT {
            if self.back.logged.insert(key) {
                tracing::info!(texture = key, ?described, "a GPU-written texture is too large to copy back");
            }
            return;
        }
        let mut tiled = vec![0u8; info.size];
        let _ = read_into(Some(&pools), info.memory as usize, &mut tiled, t.partial);
        // The title wrote the memory since the host texture was filled from it: its bytes stay,
        // and the next submission uploads them.
        if t.last != Some(ContentHash::of(&tiled)) {
            return;
        }
        if let Err(error) = download(&self.host.backend, &t.texture, info, &mut tiled) {
            if self.back.logged.insert(key) {
                tracing::warn!(texture = key, ?described, %error, "a GPU-written texture was not copied back");
            }
            return;
        }
        let hash = ContentHash::of(&tiled);
        if t.last == Some(hash) {
            return;
        }
        // Through the pool's own mapping; `None` when the title unmapped the memory meanwhile.
        let Ok((_buffer, at)) = self.host.resolve(info.memory as u64, info.size as u64) else { return };
        #[cfg(target_os = "macos")]
        super::title_watch::view("texture write-back", info.memory as u64, info.size as u64);
        // SAFETY: `resolve` gives `size` mapped, writable bytes of the pool; the buffer is kept.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), at.cast_mut(), tiled.len()) };
        // What the host texture holds, so the scanner's sighting of this write uploads nothing.
        t.last = Some(hash);
        if self.back.logged.insert(key) {
            tracing::info!(texture = key, ?described, layout = ?info.layout.desc, "a GPU-written texture is copied back into title memory");
        }
    }

    /// A title texture exists. Its host texture is made when a view, a render target or a clear
    /// first names it (most of a title's textures are never used), and filled from title memory
    /// when work next goes to the GPU.
    pub fn texture_created(&mut self, key: usize, info: TextureInfo) {
        self.note_remade(&info);
        // The sizes the title renders at: one line for each distinct render-target shape, so a run
        // shows at once whether it took the Series S's internal resolution or a higher one (the
        // frame dump's 1920x1080 is the layer, not necessarily what it draws into), and the bytes
        // each shape holds, for the memory line's top ten.
        if info.render_target() || info.flags & UNORDERED_ACCESS != 0 {
            static SEEN: std::sync::Mutex<Vec<(u32, u32, u32, u32)>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let shape = (info.width, info.height, info.format, info.flags & (RENDER_TARGET | DEPTH_STENCIL | UNORDERED_ACCESS));
            if seen.len() < 48 && !seen.contains(&shape) {
                seen.push(shape);
                // Who in the title asked for it: a 3840x2160 target on a title told a 1080p
                // display has to be traced to the call that chose it, and this shape line is
                // where a run shows the choice.
                tracing::info!(
                    w = info.width,
                    h = info.height,
                    format = info.format,
                    layers = info.depth_or_layers,
                    render_target = info.render_target(),
                    unordered_access = info.flags & UNORDERED_ACCESS != 0,
                    caller = ?caller(),
                    "a render target the title created"
                );
            }
            drop(seen);
            let mut shapes = TARGET_SHAPES.lock().unwrap_or_else(|e| e.into_inner());
            let e = shapes
                .get_or_insert_with(HashMap::new)
                .entry(shape)
                .or_insert((0u64, 0u64, info.depth_or_layers));
            e.0 += 1;
            e.1 += info.size as u64;
        }
        self.pending.insert(key, info);
    }

    /// [`Self::realize`] for a texture a *descriptor* names (a sampled view) rather than a list
    /// uses: while the copy budget is spent, the copy waits for the submission that names it
    /// ([`Self::keep_named`]) instead of being made when the title writes the descriptor. The
    /// title's heaps name every texture its streamer has resident, far more than a frame samples,
    /// and a copy made for a descriptor nothing draws with is the budget's largest cost (the run
    /// of 27 November 22:23: 5.4 GB of copies over a 3 GB budget, a 14.9 GB footprint on an 18 GB
    /// machine, frames paged to 7.8 s). A draw samples a texture only through a table its
    /// submission binds, and `keep_named` makes and mirrors the entries for those.
    pub fn realize_sampled(&mut self, key: usize) {
        if self.textures.contains_key(&key) || self.copies_of.contains_key(&key) {
            return;
        }
        let over = self.budget.limit > 0 && self.budget.live >= self.budget.limit;
        if over && self.pending.contains_key(&key) {
            // A texture sampled or rendered where the title wrote it holds no copy: no reason to
            // wait for a submission to name it.
            let free = self.pending.get(&key).is_some_and(|info| {
                let dropped = unmapped_leading_mips(info, &self.host.memory());
                info.host_desc_with(dropped)
                    .is_some_and(|desc| self.zero_copy && self.decoded_reason(info, &desc).is_none())
            });
            if !free {
                self.deferred.insert(key);
                return;
            }
        }
        self.realize(key);
    }

    /// Why `info` needs a decoded copy: Metal cannot sample (or render into) the title's bytes as
    /// they lie. `None` when it can, which [`Self::zero_copy_texture`] then makes where the title
    /// wrote them; [`Decoded`] is what the report counts the copies by.
    fn decoded_reason(&self, info: &TextureInfo, desc: &TextureDesc) -> Option<Decoded> {
        // Depth and stencil: Metal's combined depth texel layout is not the console's plane 0,
        // and the host keeps depth twins for views of them.
        if info.flags & DEPTH_STENCIL != 0 || desc.format.is_depth() {
            return Some(Decoded::Depth);
        }
        if info.layout.desc.samples > 1 {
            return Some(Decoded::Multisampled);
        }
        // One buffer-backed texture is one 2D image of one mip: a linear mip chain (smallest
        // first) or a slice stack is not one.
        let one_image = info.dimension == Dimension::Tex2d
            && info.depth_or_layers == 1
            && info.layout.mips.len() == 1
            && info.layout.mips[0].depth <= 1
            && info.dropped_levels() == 0
            && desc.mips == 1
            && desc.width == info.width as usize
            && desc.height == info.height as usize;
        // Block-compressed bytes cannot be a texture over a buffer at all (Metal asserts on the
        // alignment query), so BC1-BC7 are decoded whatever their layout or tiling. Checked
        // before both, so the report tells a streamed asset (BC, tiled, a mip chain) from an
        // uncompressed mip chain or tiling, which a host could some day alias.
        if desc.format.block().0 == 4 {
            return Some(if one_image { Decoded::BcPack } else { Decoded::BcMipped });
        }
        if !one_image {
            return Some(Decoded::Layout);
        }
        // The host must read the console's bytes as its own: DXGI B4G4R4A4's nibbles are rotated
        // on upload, and a format Metal lacks is never a texture at all.
        let Some(host_format) = format(info.format) else {
            return Some(Decoded::Format);
        };
        if info.format == 115 || desc.format != host_format {
            return Some(Decoded::Format);
        }
        if info.layout.desc.swizzle != Swizzle::LINEAR {
            return Some(Decoded::Tiled);
        }
        if self.zero_copy_row(info).is_none()
            || self.host.memory().resolve(info.memory as u64, info.size as u64).is_none()
        {
            return Some(Decoded::Alignment);
        }
        None
    }

    /// The title's row stride, in bytes (blocks for block compression), when it meets Metal's
    /// `linear_alignment` for the format: addrlib's linear rows are 256-byte aligned.
    fn zero_copy_row(&self, info: &TextureInfo) -> Option<usize> {
        let desc = info.desc()?;
        let align = self.host.backend.linear_alignment(desc.format).max(1);
        let row_pitch = info.layout.mips.first()?.pitch as usize * info.layout.desc.element_bytes as usize;
        (row_pitch != 0 && row_pitch.is_multiple_of(align)).then_some(row_pitch)
    }

    /// The buffer over the pool's pages under `info`, and that memory's pool address.
    fn zero_copy_buffer(&self, info: &TextureInfo) -> Option<(Rc<B::Buffer>, *const u8)> {
        self.host.memory_buffer(info.memory as u64, info.size as u64)
    }

    /// `info` made over the title's own pages: sampled, rendered into or written through a
    /// dispatch straight where the title put it, with no decoded copy ([`Self::decoded_reason`]).
    fn zero_copy_texture(&self, info: &TextureInfo, desc: &TextureDesc) -> Option<Rc<B::Texture>> {
        if !self.zero_copy {
            return None;
        }
        if self.decoded_reason(info, desc).is_some() {
            return None;
        }
        let row_pitch = self.zero_copy_row(info)?;
        let (buffer, at) = self.zero_copy_buffer(info)?;
        match self.host.backend.create_texture_over_memory(&buffer, at, desc, row_pitch) {
            Ok(tex) => Some(Rc::new(tex)),
            Err(e) => {
                // Logged once per run: a layout Metal refuses keeps a decoded copy, and the
                // report counts it by reason.
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| tracing::info!(error = %e, "a texture over title memory was refused; decoded"));
                None
            }
        }
    }

    /// Turns sampling straight from title memory on or off for this title (tests: the default is
    /// off there, see [`Title::zero_copy`]).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn set_zero_copy(&mut self, on: bool) {
        self.zero_copy = on;
    }

    /// Turns giving a decoded copy's title pages back on or off for this title (tests: the
    /// default is off there, see [`Title::release_pages`]).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn set_release_pages(&mut self, on: bool) {
        self.release_pages = on;
    }

    /// Turns giving a streamed copy's pages back at its upload on or off for this title (tests: the
    /// default is off there, see [`Title::release_streamed`]).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn set_release_streamed(&mut self, on: bool) {
        self.release_streamed = on;
    }

    /// Makes `key` look unused since `at` (tests: the release threshold is wall time).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn set_texture_last_used_at(&mut self, key: usize, at: std::time::Instant) {
        if let Some(t) = self.textures.get_mut(&self.owner_of(key)) {
            t.last_used_at = at;
        }
    }

    /// Whether title texture `key` is sampled straight from title memory (no decoded copy).
    pub fn is_zero_copy(&self, key: usize) -> bool {
        self.textures.get(&self.owner_of(key)).is_some_and(|t| t.zero_copy)
    }

    /// Makes the host texture of title texture `key` if it has none yet and registers it for
    /// views. `false` when there is none (not a title texture, or one the host cannot represent).
    pub fn realize(&mut self, key: usize) -> bool {
        self.deferred.remove(&key);
        if self.textures.contains_key(&key) || self.copies_of.contains_key(&key) {
            return true;
        }
        let Some(info) = self.pending.remove(&key) else {
            return false;
        };
        self.note_remade(&info);
        let dropped = unmapped_leading_mips(&info, &self.host.memory());
        let Some(desc) = info.host_desc_with(dropped) else {
            // Loud once per format: views of it get no texture and their draws sample nothing.
            static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&info.format) {
                seen.push(info.format);
                tracing::warn!(format = info.format, w = info.width, h = info.height, flags = info.flags, "a title texture format the host does not represent (first of this format)");
            }
            return false;
        };
        if self.host.is_evicted(key) {
            BUDGET_TRACE[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let at = memory_identity(&self.host.memory(), info.memory, info.size);
        // Sampled straight from title memory where the layout allows it: no decode, no upload, no
        // bytes of the budget, and a title memory write is seen by the next draw as it is.
        if let Some(tex) = self.zero_copy_texture(&info, &desc) {
            self.at_memory.entry(at).or_default().push(key);
            self.host.set_dropped_levels(key, 0, desc.mips);
            self.host.register_texture(key, &tex);
            static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ZERO_COPY[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ZERO_COPY[1].fetch_add(info.size as u64, std::sync::atomic::Ordering::Relaxed);
            self.textures.insert(
                key,
                Tracked {
                    texture: tex,
                    info,
                    serial,
                    last: None,
                    uploads: 0,
                    uploaded_mips: u32::MAX,
                    hot_until: 0,
                    probe: None,
                    sampled_sees_writes: false,
                    unchanged_whole: 0,
                    skipped: 0,
                    partial: false,
                    blank: false,
                    #[cfg(target_os = "macos")]
                    target_views: Vec::new(),
                    gpu_written: false,
                    zero_copy: true,
                    streamed: false,
                    write_serial: write_serial(),
                    restreamed: false,
                    last_used_at: std::time::Instant::now(),
                    used: self.budget.serial,
                    last_named: 0,
                    reach: 0,
                    last_frame: crate::queue::presented(),
                    dropped: 0,
                    at,
                },
            );
            return true;
        }
        if let Some(owner) = self.identical_copy(at, &info) {
            let tex = self.textures[&owner].texture.clone();
            let owner_dropped = self.textures[&owner].dropped;
            let owner_mips = info.desc().map_or(1, |d| d.mips.saturating_sub(owner_dropped).max(1));
            self.host.set_dropped_levels(key, owner_dropped, owner_mips);
            self.host.register_texture(key, &tex);
            self.copies_of.insert(key, (owner, info));
            SHARED_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        let tex = match self.host.texture(&desc) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, ?desc, "no host texture");
                return false;
            }
        };
        self.at_memory.entry(at).or_default().push(key);
        self.host.set_dropped_levels(key, dropped, desc.mips);
        self.host.register_texture(key, &tex);
        self.budget.live += info.size as u64;
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Depth-stencil textures are GPU-written depth targets whose CPU tile layout (e.g.
        // 4-byte or 2-byte plane 0) does not match Metal's combined depth-stencil texel layout,
        // so watching or uploading CPU memory cannot produce valid replaceRegion transfers.
        if !desc.format.is_depth() {
            start_scanner();
            if info.render_target() {
                // Render targets are not read at first submission (`unloaded`), so the scanner
                // takes their baseline directly; what the title (CPU, DMA) writes into their
                // memory afterwards is uploaded when it changes.
                let pools = self.host.memory();
                watch(|w| {
                    w.entries.insert(key, (info.memory as usize, info.size, true, serial));
                    w.pools.insert(key, pools);
                });
                // Debugging (VICTORY_MAGENTA=1): render targets start magenta, so a frame that
                // stays so was never drawn into, and a black one was drawn black.
                if env_set!("VICTORY_MAGENTA") {
                    let m = &info.layout.mips[0];
                    let n = info.layout.desc.element_bytes as usize;
                    let row = m.width as usize * n;
                    let pixel = [0xffu8, 0, 0xff, 0xff];
                    let linear: Vec<u8> = (0..row * m.height as usize).map(|i| pixel[i % 4]).collect();
                    self.host.backend.upload_texture(&tex, 0, 0, &linear, row, row * m.height as usize);
                }
            } else {
                // Sampled textures enter `WATCH` once their first submission read populates `seen`
                // (`sync_apply`), avoiding a duplicate whole read by the background scanner.
                self.unloaded.insert(key);
            }
        }
        let uploaded_mips = if info.render_target() || info.flags & UNORDERED_ACCESS != 0 {
            u32::MAX
        } else {
            0
        };
        let (placed_mem, placed_size) = (info.memory as u64, info.size as u64);
        let floor_dropped = info.dropped_levels();
        self.textures.insert(
            key,
            Tracked {
                texture: tex,
                info,
                serial,
                last: None,
                uploads: 0,
                uploaded_mips,
                hot_until: 0,
                probe: None,
                sampled_sees_writes: false,
                unchanged_whole: 0,
                skipped: 0,
                partial: false,
                blank: false,
                #[cfg(target_os = "macos")]
                #[cfg(target_os = "macos")]
                target_views: Vec::new(),
                gpu_written: false,
                zero_copy: false,
                streamed: false,
                write_serial: write_serial(),
                restreamed: false,
                last_used_at: std::time::Instant::now(),
                used: self.budget.serial,
                last_named: 0,
                reach: 0,
                last_frame: crate::queue::presented(),
                dropped,
                at,
            },
        );
        if dropped > floor_dropped {
            self.trimmed.insert(key);
        }
        if key >> 63 != 0 && !self.placed_chunks.contains_key(&key) {
            let ranges = self.host.memory().pool_ranges(placed_mem, placed_size);
            self.track_placed_pools(key, &ranges);
        }
        if self.covered_evicted.remove(&key) && self.has_live_placed_overlap(key) {
            let epoch = self.gpu_epoch;
            self.record_epoch(epoch);
            self.covered.entry(key).or_insert(epoch);
        }
        true
    }

    /// A live copy that title texture `info` would duplicate: one over the same physical memory
    /// (`at`, [`memory_identity`]) with the same layout, dimensions and host format. Two textures
    /// over the same bytes read the same texels (B4G4R4A4 is converted on upload), so a view of
    /// either names one host texture.
    ///
    /// A copy filled from title memory is shared with another one nothing has written yet. A
    /// render target, a depth texture or a copy the GPU wrote holds what title memory does not, so
    /// it is shared only with another written one over the same *pool* pages (`at` with its top
    /// bit set, [`memory_identity`]): two resources the title placed over one physical range alias
    /// the same bytes on the console, and one host texture holds what either of them writes. A
    /// title address is not enough there: a released object's address goes to the next object, and
    /// that one starts with its own contents.
    fn identical_copy(&self, at: usize, info: &TextureInfo) -> Option<usize> {
        let desc = info.desc()?;
        let aliased = at >> 63 != 0;
        let writes = |i: &TextureInfo| i.render_target() || i.flags & UNORDERED_ACCESS != 0;
        let written = writes(info);
        self.at_memory.get(&at)?.iter().copied().find(|k| {
            let t = &self.textures[k];
            let o = &t.info;
            if o.size != info.size
                || o.layout != info.layout
                || o.dimension != info.dimension
                || (o.width, o.height, o.depth_or_layers) != (info.width, info.height, info.depth_or_layers)
                || o.flags & UNORDERED_ACCESS != info.flags & UNORDERED_ACCESS
                || (o.format == 115) != (info.format == 115)
                || !o.desc().is_some_and(|d| d.format == desc.format)
            {
                return false;
            }
            if written || writes(o) || t.gpu_written {
                aliased && written && writes(o)
            } else {
                true
            }
        })
    }

    /// The key whose copy title texture `key` uses: its own, or the identical copy it shares.
    fn owner_of(&self, key: usize) -> usize {
        self.copies_of.get(&key).map_or(key, |&(owner, _)| owner)
    }

    /// The format of the host texture of title texture `key`.
    pub fn format_of(&self, key: usize) -> Option<Format> {
        let key = self.owner_of(key);
        self.textures.get(&key).and_then(|t| t.info.desc()).map(|d| d.format)
    }

    /// The host texture of title texture `key`.
    pub fn texture_of(&mut self, key: usize) -> Option<Rc<B::Texture>> {
        self.realize(key);
        let key = self.owner_of(key);
        let serial = self.budget.serial;
        let frame = crate::queue::presented();
        self.textures.get_mut(&key).map(|t| {
            (t.used, t.last_frame, t.last_used_at) = (serial, frame, std::time::Instant::now());
            t.texture.clone()
        })
    }

    /// The live depth-stencil textures, made, with their descriptions: what `VICTORY_DUMP_DEPTH`
    /// reads back to show a shadow map holds geometry.
    #[cfg(target_os = "macos")]
    pub fn depth_targets(&mut self) -> Vec<(usize, TextureInfo, Rc<B::Texture>)> {
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| t.info.flags & DEPTH_STENCIL != 0 && !t.zero_copy)
            .map(|(&k, _)| k)
            .collect();
        keys.into_iter()
            .filter_map(|key| {
                let serial = self.budget.serial;
                let frame = crate::queue::presented();
                let t = self.textures.get_mut(&key)?;
                (t.used, t.last_frame, t.last_used_at) = (serial, frame, std::time::Instant::now());
                Some((key, t.info.clone(), t.texture.clone()))
            })
            .collect()
    }

    /// Diagnostic (`VICTORY_DUMP_DEPTH`): the placed keys that start over depth target `info`'s
    /// memory and still have a host texture of their own, `(address, key, width, height, format)`.
    /// After [`Self::alias_depth`] there should be none at the target's start: one means the
    /// target appeared after the view was mirrored, and the view still samples a blank copy.
    #[cfg(target_os = "macos")]
    pub fn placed_copies_over(&self, info: &TextureInfo) -> Vec<(u64, usize, u32, u32, u32)> {
        let at = info.memory as u64;
        let end = at.saturating_add(info.size as u64);
        self.placed
            .range(at..end)
            .filter_map(|(&addr, &(key, _))| {
                self.textures
                    .get(&key)
                    .map(|t| (addr, key, t.info.width, t.info.height, t.info.format))
            })
            .take(16)
            .collect()
    }

    /// Resolves a render-target view to its attachment and its effective format. The base
    /// attachment can use the texture itself. A narrowed view also supplies the mip's extent.
    #[cfg(target_os = "macos")]
    pub(crate) fn target_of(
        &mut self,
        record: crate::descriptor::RtvRecord,
    ) -> Option<(Rc<B::Texture>, Format)> {
        self.target_of_kind(record, false)
    }

    /// [`Self::target_of`] for a depth-stencil view: its dimension words are numbered as
    /// D3D12's depth views number them, and a view of one array slice becomes a one-layer view
    /// of the depth texture (Metal's depth attachment renders into one layer, slice 0 unless the
    /// view selects it).
    #[cfg(target_os = "macos")]
    pub(crate) fn dsv_target_of(
        &mut self,
        record: crate::descriptor::RtvRecord,
    ) -> Option<(Rc<B::Texture>, Format)> {
        self.target_of_kind(record, true)
    }

    #[cfg(target_os = "macos")]
    fn target_of_kind(
        &mut self,
        record: crate::descriptor::RtvRecord,
        depth_view: bool,
    ) -> Option<(Rc<B::Texture>, Format)> {
        let key = record.resource as usize;
        self.realize(key);
        let key = self.owner_of(key);
        let t = self.textures.get_mut(&key)?;
        (t.used, t.last_frame) = (self.budget.serial, crate::queue::presented());
        t.last_used_at = std::time::Instant::now();
        let desc = t.info.desc()?;
        let subresource = if depth_view {
            record.native_dsv_subresource(
                t.info.dimension,
                t.info.depth_or_layers,
                desc.mips,
                t.info.layout.desc.samples,
            )
        } else {
            record.native_rtv_subresource(
                t.info.dimension,
                t.info.depth_or_layers,
                desc.mips,
                t.info.layout.desc.samples,
            )
        };
        let (mip, slice) = subresource?;
        self.target_view_at(record, mip, slice)
    }

    /// The attachment texture for `(mip, slice)` of `record`'s resource, in the view's format
    /// where the console has one: the texture itself for the base subresource, else a one-mip,
    /// one-layer view of it.
    #[cfg(target_os = "macos")]
    pub(crate) fn target_view_at(
        &mut self,
        record: crate::descriptor::RtvRecord,
        mip: usize,
        slice: usize,
    ) -> Option<(Rc<B::Texture>, Format)> {
        let key = record.resource as usize;
        self.realize(key);
        let key = self.owner_of(key);
        let t = self.textures.get_mut(&key)?;
        (t.used, t.last_frame) = (self.budget.serial, crate::queue::presented());
        t.last_used_at = std::time::Instant::now();
        let desc = t.info.desc()?;
        let base = (mip, slice) == (0, 0);
        let own = desc.format;
        let twin = format(record.format()).filter(|&f| srgb_twin(own) == Some(f));
        // A failed format reinterpretation can fall back to the resource format, but must
        // keep the same mip and slice.
        for format in twin.into_iter().chain([own]) {
            if base && format == own {
                return Some((t.texture.clone(), own));
            }
            let view = ViewDesc {
                format,
                mips: if base { (0, usize::MAX) } else { (mip, 1) },
                slices: if base { (0, usize::MAX) } else { (slice, 1) },
            };
            if let Some((_, texture)) = t.target_views.iter().find(|(cached, _)| *cached == view) {
                return Some((texture.clone(), format));
            }
            match self.host.backend.create_texture_view(&t.texture, &view) {
                Ok(texture) => {
                    let texture = Rc::new(texture);
                    self.host.note_target_view(&texture, &t.texture);
                    t.target_views.push((view, texture.clone()));
                    return Some((texture, format));
                }
                Err(e) => tracing::warn!(error = %e, ?view, "no host view for a render target view"),
            }
        }
        None
    }

    /// The GPU writes title texture `key` (made already): a copy, draw or clear recorded into it,
    /// or a prepared dispatch's unordered-access view of it. From now on the host texture is the
    /// GPU's and the CPU-side sync no longer fills it from title memory (which titles change under
    /// their render targets: uploaded, that was block noise over what the GPU drew).
    /// A sampled texture not uploaded yet still takes its first upload, at the submission and so
    /// before the write runs (a copy of a region keeps the rest).
    /// shortcut: what the title writes into its memory afterwards is not seen; when a title is
    /// found to refill a GPU-written texture from the CPU, upload a change again.
    pub fn gpu_written(&mut self, key: usize) {
        // A placed view sharing an object's texture ([`Self::alias_placed`]) writes the object's.
        let key = self
            .shared
            .iter()
            .find(|(_, placements)| placements.iter().any(|&(_, placed)| placed == key))
            .map_or(key, |(&object, _)| object);
        let key = self.owner_of(key);
        // A released copy the GPU is about to write: its pages are `PROT_NONE` and its console bytes
        // were given back. They are put back from the texture (the last complete content) before the
        // write runs, and the copy stays home from then on: a GPU-written texture is never released.
        if self.released.contains_key(&key) && !self.restore_pages(key) {
            if let Some(t) = self.textures.get(&key) {
                let info = t.info.clone();
                let (tstart, tend) = page_range(info.memory as usize, info.size);
                faults::restore(tstart, tend - tstart);
                let _ = protect(tstart, tend - tstart, PROT_READ_WRITE);
            }
            self.released.remove(&key);
            self.unloaded.insert(key);
        }
        if let Some(t) = self.textures.get_mut(&key) {
            t.gpu_written = true;
            t.last_frame = crate::queue::presented();
            t.last_used_at = std::time::Instant::now();
            (t.hot_until, t.blank) = (0, false);
            if t.kept_from_memory() {
                watch(|w| w.released(key));
            }
        }
    }

    /// How many times title texture `key`'s memory was detiled into its host texture.
    pub fn uploads_of(&self, key: usize) -> u64 {
        self.textures.get(&self.owner_of(key)).map_or(0, |t| t.uploads)
    }

    pub fn texture_released(&mut self, key: usize) {
        // A released copy's pages must not outlive it: the title unmaps or reuses that memory, and
        // a `PROT_NONE` page it reuses would fault with no copy left to serve it (DirectStorage
        // reusing a streamed texture's memory is exactly that). There is nothing to restore: the
        // bytes are the title's to replace.
        if self.released.remove(&key).is_some()
            && let Some(t) = self.textures.get(&key)
        {
            let info = t.info.clone();
            let (tstart, tend) = page_range(info.memory as usize, info.size);
            faults::restore(tstart, tend - tstart);
            let _ = protect(tstart, tend - tstart, PROT_READ_WRITE);
        }
        self.copies_of.remove(&key);
        self.covered.remove(&key);
        self.trimmed.remove(&key);
        self.deferred.remove(&key);
        self.untrack_placed_pools(key);
        if let Some(t) = self.textures.remove(&key) {
            if !t.zero_copy {
                self.budget.live -= t.info.size as u64;
            }
            let at = t.at;
            if let Some(keys) = self.at_memory.get_mut(&at) {
                keys.retain(|&k| k != key);
                if keys.is_empty() {
                    self.at_memory.remove(&at);
                }
            }
            // The copy stays with those that shared it: the first takes it over (their entries
            // show it, so it is neither made nor uploaded again), the others share its.
            let mut sharing: Vec<usize> = self.copies_of.iter().filter(|(_, (o, _))| *o == key).map(|(&k, _)| k).collect();
            sharing.sort_unstable();
            if let Some((&first, rest)) = sharing.split_first() {
                let (_, info) = self.copies_of.remove(&first).expect("listed");
                for k in rest {
                    self.copies_of.get_mut(k).expect("listed").0 = first;
                }
                if self.unloaded.contains(&key) {
                    self.unloaded.insert(first);
                }
                if !t.zero_copy {
                    self.budget.live += info.size as u64;
                }
                self.at_memory.entry(at).or_default().push(first);
                let assigned = info.memory as usize;
                watch(|w| w.entries.insert(first, (assigned, info.size, false, t.serial)));
                let mut t = t;
                t.info = info;
                if t.dropped > t.info.dropped_levels() {
                    self.trimmed.insert(first);
                }
                self.textures.insert(first, t);
            }
        }
        // Placed views that shared this texture ([`Self::alias_placed`]) go with it.
        if let Some(placements) = self.shared.remove(&key) {
            for (address, placed) in placements {
                if self.placed.get(&address).is_some_and(|&(k, _)| k == placed) {
                    self.placed.remove(&address);
                    self.host.release_resource(placed);
                }
            }
        }
        // A placed view that shared a texture and is released itself: no longer its object's.
        for placements in self.shared.values_mut() {
            placements.retain(|&(_, placed)| placed != key);
        }
        self.shared.retain(|_, placements| !placements.is_empty());
        self.pending.remove(&key);
        self.unloaded.remove(&key);
        self.hot_keys.retain(|&k| k != key);
        self.blank_keys.retain(|&k| k != key);
        watch(|w| w.released(key));
    }

    /// Indexes `ranges` (`(pool_id, offset, len)`) under `key` in [`Self::placed_pools`].
    fn track_placed_pools(&mut self, key: usize, ranges: &[(usize, u64, u64)]) {
        self.untrack_placed_pools(key);
        if ranges.is_empty() {
            return;
        }
        for &(pid, off, clen) in ranges {
            self.placed_pool_reach = self.placed_pool_reach.max(clen);
            self.placed_pools
                .entry(pid)
                .or_default()
                .entry(off)
                .or_default()
                .push((key, clen));
        }
        self.placed_chunks.insert(key, ranges.to_vec());
    }

    /// Removes `key`'s physical pool entries from [`Self::placed_pools`].
    fn untrack_placed_pools(&mut self, key: usize) {
        let Some(chunks) = self.placed_chunks.remove(&key) else {
            return;
        };
        for (pid, off, _) in chunks {
            let Some(map) = self.placed_pools.get_mut(&pid) else {
                continue;
            };
            if let Some(entries) = map.get_mut(&off) {
                entries.retain(|&(k, _)| k != key);
                if entries.is_empty() {
                    map.remove(&off);
                }
            }
            if map.is_empty() {
                self.placed_pools.remove(&pid);
            }
        }
    }

    /// A view placed texture `key` at title `address` (in pool `pool`). The title has no object to
    /// release for it, so the texture another description placed at that address before goes
    /// now (guess: the title is done with it once it places another there), and every texture
    /// placed in a pool goes when the pool is released ([`Self::sync_dirty`]).
    pub fn texture_placed(&mut self, address: u64, key: usize, pool: usize, size: usize) {
        // A view of a copy given up before comes back at its own address: not a new placement.
        if self.placed.get(&address).is_none_or(|&(k, _)| k != key) {
            let ranges = self.host.memory().pool_ranges(address, size as u64);
            self.give_up_overlapped(address, key, size as u64, &ranges);
            self.track_placed_pools(key, &ranges);
        }
        self.placed_reach = self.placed_reach.max(size as u64);
        if let Some((old, _)) = self.placed.insert(address, (key, pool))
            && old != key
        {
            if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
                // Whether two descriptions keep taking one address from each other.
                static WERE: std::sync::Mutex<Option<std::collections::HashSet<(u64, usize)>>> = std::sync::Mutex::new(None);
                let mut were = WERE.lock().unwrap_or_else(|e| e.into_inner());
                let were = were.get_or_insert_default();
                were.insert((address, old));
                REPLACED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                REPLACED[1].fetch_add(u64::from(were.contains(&(address, key))), std::sync::atomic::Ordering::Relaxed);
            }
            self.host.forget_resource(old);
            self.covered_evicted.remove(&old);
            self.texture_released(old);
        }
    }

    /// Placed copies over `[address, address + size)` (by title VA or by physical pool chunks
    /// `ranges` across VA aliases) that start elsewhere, which placed texture `key` now covers,
    /// are marked covered: given up once nothing uses them ([`Self::give_up_covered`]). Guess: the
    /// title is done with a placed view once it places another over its memory (it does so when it
    /// streams something else in). Never marked: copies that hold what title memory does not
    /// (render targets, depth, GPU-written, ever viewed for unordered access).
    fn give_up_overlapped(
        &mut self,
        address: u64,
        key: usize,
        size: u64,
        ranges: &[(usize, u64, u64)],
    ) {
        let end = address.saturating_add(size);
        let same_addr_old = self.placed.get(&address).map(|&(k, _)| k);
        let mut over: Vec<usize> = self
            .placed
            .range(address.saturating_sub(self.placed_reach)..end)
            .filter(|&(&a, &(k, _))| a != address && k != key)
            .map(|(_, &(k, _))| k)
            .filter(|k| {
                self.textures.get(k).is_some_and(|t| {
                    let o = t.info.memory as u64;
                    o < end
                        && o + t.info.size as u64 > address
                        && self.holds_only_memory(*k, t)
                })
            })
            .collect();
        for &(pid, off, clen) in ranges {
            let Some(map) = self.placed_pools.get(&pid) else {
                continue;
            };
            let pend = off.saturating_add(clen);
            for (&p_off, entries) in map.range(off.saturating_sub(self.placed_pool_reach)..pend) {
                if p_off >= pend {
                    break;
                }
                for &(ok, ok_len) in entries {
                    if ok != key
                        && Some(ok) != same_addr_old
                        && p_off.saturating_add(ok_len) > off
                        && self
                            .textures
                            .get(&ok)
                            .is_some_and(|t| self.holds_only_memory(ok, t))
                    {
                        over.push(ok);
                    }
                }
            }
        }
        over.sort_unstable();
        over.dedup();
        if !over.is_empty() {
            let epoch = self.gpu_epoch;
            self.record_epoch(epoch);
            for k in over {
                self.covered.entry(k).or_insert(epoch);
            }
        }
    }

    fn record_epoch(&mut self, epoch: u64) {
        if epoch > self.retired_epoch
            && self.pending_epochs.back().is_none_or(|(e, _)| *e < epoch)
            && !self.queue_fences.is_empty()
        {
            self.pending_epochs
                .push_back((epoch, self.queue_fences.values().cloned().collect()));
        }
    }

    /// Whether any other live texture covered by [`Self::holds_only_memory`] in `placed` or
    /// `placed_pools` currently overlaps `key`'s range (used when re-realizing a covered copy).
    fn has_live_placed_overlap(&self, key: usize) -> bool {
        let Some(t) = self.textures.get(&key) else {
            return false;
        };
        if !self.holds_only_memory(key, t) {
            return false;
        }
        let address = t.info.memory as u64;
        let end = address.saturating_add(t.info.size as u64);
        if self
            .placed
            .range(address.saturating_sub(self.placed_reach)..end)
            .filter(|&(_, &(k, _))| k != key)
            .any(|(_, &(k, _))| {
                self.textures.get(&k).is_some_and(|o| {
                    let oa = o.info.memory as u64;
                    oa < end
                        && oa.saturating_add(o.info.size as u64) > address
                        && self.holds_only_memory(k, o)
                })
            })
        {
            return true;
        }
        let Some(chunks) = self.placed_chunks.get(&key) else {
            return false;
        };
        for &(pid, off, clen) in chunks {
            let Some(map) = self.placed_pools.get(&pid) else {
                continue;
            };
            let pend = off.saturating_add(clen);
            for (&p_off, entries) in map.range(off.saturating_sub(self.placed_pool_reach)..pend) {
                if p_off >= pend {
                    break;
                }
                if entries.iter().any(|&(ok, ok_len)| {
                    ok != key
                        && p_off.saturating_add(ok_len) > off
                        && self
                            .textures
                            .get(&ok)
                            .is_some_and(|o| self.holds_only_memory(ok, o))
                }) {
                    return true;
                }
            }
        }
        false
    }

    /// Advances [`Self::retired_epoch`] across GPU epochs whose fences have all completed.
    fn retire_gpu_epochs(&mut self) {
        while let Some((_, fences)) = self.pending_epochs.front() {
            if !fences
                .iter()
                .all(|(f, v)| self.host.backend.completed(f) >= *v)
            {
                break;
            }
            let (epoch, _) = self.pending_epochs.pop_front().expect("front exists");
            self.retired_epoch = self.retired_epoch.max(epoch);
        }
    }

    /// At each sync: covered copies ([`Self::give_up_overlapped`]) that nothing uses any more are
    /// given up as the budget gives one up ([`Self::evictable`]: no heap entry shows one, so no
    /// work in flight reads it through an entry, and no list holds one), and none a submission in
    /// flight names. Under a texture budget (`budget.limit > 0`), a covered copy whose heap
    /// entries were not overwritten yet is also given up once all GPU work queued before it was
    /// covered has finished, clearing its stale heap entries (`evict_texture`).
    /// One named again later is made again from title memory.
    fn give_up_covered(&mut self) {
        self.retire_gpu_epochs();
        let has_budget = self.budget.limit > 0;
        let mut entries: Vec<(u64, usize)> = self.covered.iter().map(|(&k, &e)| (e, k)).collect();
        entries.sort_unstable();
        for (covered_epoch, k) in entries {
            let Some(t) = self.textures.get(&k) else {
                self.covered.remove(&k);
                continue;
            };
            if self.budget.named.contains_key(&k)
                || !self.holds_only_memory(k, t)
                || Rc::strong_count(&t.texture) > 2
            {
                continue;
            }
            let evict_shown = has_budget && covered_epoch <= self.retired_epoch;
            if self.host.texture_in_use(k) && !evict_shown {
                continue;
            }
            let info = t.info.clone();
            OVERLAP_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            OVERLAP_TRACE[1].fetch_add(info.size as u64, std::sync::atomic::Ordering::Relaxed);
            self.host.evict_texture(k);
            self.covered_evicted.insert(k);
            self.texture_released(k);
            self.pending.insert(k, info);
        }
    }

    /// A placed texture `key` at `address` names the host texture of a title texture the GPU
    /// writes at the same memory, when one matches, so its views cast that texture instead of
    /// reading a copy only title memory fills.
    ///
    /// The R9G9B9E5 view of an R32_UINT object and a depth view of a depth target are always
    /// shared ([`Self::alias_rgb9e5`], [`Self::alias_depth`]); `VICTORY_PLACED_TWIN=1` adds every
    /// other written texture the view could read ([`Self::alias_written_twin`]).
    pub fn alias_placed(&mut self, address: u64, key: usize, info: &TextureInfo) -> bool {
        self.alias_rgb9e5(address, key, info)
            || self.alias_depth(address, key, info)
            || (placed_twin() && self.alias_written_twin(address, key, info))
    }

    /// A placed view of a depth texture `key` over a live depth target of the same shape and
    /// element at the same memory: `key` names the target's host texture, which the shadow
    /// cascades and light faces were rendered into, instead of a copy only title memory fills.
    ///
    /// A placed copy of a depth texture is never filled from title memory (depth and
    /// depth-stencil targets are GPU-written and neither watched nor uploaded), and Metal has no
    /// way to write a depth texture from compute anyway (`depth_twin` handles the unordered
    /// views), so the copy would stay zero: every comparison sample would fail and no sun shadow
    /// would ever show. The target and the view describe the same memory, so one host texture is
    /// what both need. `false` (nothing done) for any other placed texture.
    fn alias_depth(&mut self, address: u64, key: usize, info: &TextureInfo) -> bool {
        // Object keys are addresses; placed keys have the top bit set ([`placed_key`]).
        let object = |k: usize| k >> 63 == 0;
        let Some(depth) = self
            .textures
            .iter()
            .map(|(&k, t)| (k, &t.info))
            .chain(self.pending.iter().map(|(&k, i)| (k, i)))
            .find(|&(k, i)| object(k) && i.flags & DEPTH_STENCIL != 0 && same_shape_and_element(info, i))
            .map(|(k, _)| k)
        else {
            return false;
        };
        if !self.share_texture(depth, address, key) {
            return false;
        }
        static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 64 && !seen.contains(&address) {
            seen.push(address);
            tracing::info!(
                placed = format!("{key:#x}"),
                object = format!("{depth:#x}"),
                format = info.format,
                w = info.width,
                h = info.height,
                layers = info.depth_or_layers,
                "a placed view of a depth texture reads the depth target at its memory"
            );
        }
        true
    }

    /// A placed R9G9B9E5 texture `key` over a live R32_UINT resource object of the same shape and
    /// element at the same memory (the title reads as floats what compute wrote as integers):
    /// `key` names the object's host texture, which its views cast, instead of a copy only title
    /// memory fills. The two may resolve to different console layouts (an R32_UINT render target
    /// to `SW_64KB_R_X`, an automatic placed view to `SW_4KB_S`): the host texture holds what the
    /// GPU wrote, not the console's tiled bytes, so the cast is the same. `false` (nothing done)
    /// for any other placed texture.
    fn alias_rgb9e5(&mut self, address: u64, key: usize, info: &TextureInfo) -> bool {
        let over = |o: &TextureInfo| o.format == 42 && same_shape_and_element(info, o);
        // Object keys are addresses; placed keys have the top bit set ([`placed_key`]).
        let object = |k: usize| k >> 63 == 0;
        if info.format != 67 {
            return false;
        }
        let Some(obj) = self
            .textures
            .iter()
            .map(|(&k, t)| (k, &t.info))
            .chain(self.pending.iter().map(|(&k, i)| (k, i)))
            .find(|&(k, i)| object(k) && over(i))
            .map(|(k, _)| k)
        else {
            return false;
        };
        if !self.share_texture(obj, address, key) {
            return false;
        }
        static ALIASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = ALIASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 1000 {
            tracing::info!(placed = format!("{key:#x}"), object = format!("{obj:#x}"), aliased = n, "a placed R9G9B9E5 view shares an R32_UINT texture");
        }
        true
    }

    /// `VICTORY_PLACED_TWIN=1`: the host texture of a title texture the GPU writes at the memory
    /// of placed view `info` (`key` at `address`), when the view's description is one the console
    /// laid out itself (`layout_word` 0) and the two shapes and element sizes match
    /// ([`written_twin`]). `false` (nothing done) for any other placed texture.
    ///
    /// Why: an automatic layout is resolved from the description's own flags (a render target
    /// resolves to `SW_64KB_R_X`, else `SW_4KB_S`), so a placed view of the same memory without
    /// those flags gets different resolved bytes than the texture written through it, and so a
    /// copy of its own that only title memory fills: nothing the GPU wrote reaches it. The
    /// screen reflections' history is such a texture (R32_UINT, render target and unordered
    /// access) read through R9G9B9E5 placed views. One line per address the first times, both
    /// descriptions.
    pub fn alias_written_twin(&mut self, address: u64, key: usize, info: &TextureInfo) -> bool {
        // An explicit layout describes memory the title laid out itself, not one the console
        // resolved from the flags: no twin.
        if info.layout_word != 0 {
            return false;
        }
        // Object keys are addresses; placed keys have the top bit set ([`placed_key`]).
        let object = |k: usize| k >> 63 == 0;
        let Some((twin, written)) = self
            .textures
            .iter()
            .map(|(&k, t)| (k, &t.info))
            .chain(self.pending.iter().map(|(&k, i)| (k, i)))
            .find(|&(k, i)| object(k) && written_twin(info, i))
            .map(|(k, i)| (k, i.clone()))
        else {
            return false;
        };
        if !self.share_texture(twin, address, key) {
            return false;
        }
        static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 64 && !seen.contains(&address) {
            seen.push(address);
            tracing::info!(
                address = format_args!("{address:#x}"),
                view = ?(info.format, info.layout_word, info.flags, info.layout.mips.len()),
                written = ?(written.format, written.layout_word, written.flags, written.layout.mips.len()),
                "VICTORY_PLACED_TWIN: a placed view views the title texture written at its memory"
            );
        }
        true
    }

    /// Registers title texture `object`'s host texture under placed `key`, which views it too
    /// ([`Self::alias_placed`]).
    fn share_texture(&mut self, object: usize, address: u64, key: usize) -> bool {
        let Some(texture) = self.texture_of(object) else { return false };
        self.host.register_texture(key, &texture);
        // The copy that holds it: an object sharing an identical one ([`Self::identical_copy`])
        // lives under its owner's key.
        let owner = self.owner_of(object);
        let placements = self.shared.entry(owner).or_default();
        if !placements.contains(&(address, key)) {
            placements.push((address, key));
        }
        true
    }

    /// Releases the textures placed in pools the title released since the last call.
    fn release_placed(&mut self) {
        let gone = self.host.released_pools();
        if gone.is_empty() {
            return;
        }
        let dead: Vec<u64> = self
            .placed
            .iter()
            .filter(|(_, (_, pool))| gone.contains(pool))
            .map(|(&a, _)| a)
            .collect();
        for a in dead {
            let (key, _) = self.placed.remove(&a).expect("listed above");
            self.host.release_resource(key);
            self.covered_evicted.remove(&key);
            self.texture_released(key);
        }
    }

    /// Brings the textures the host samples up to date with title memory: those whose memory
    /// changed since their last upload are detiled again.
    pub fn sync_textures(&mut self) {
        // A host writer may have opened a released copy's pages (see `drop_host_written`).
        self.drop_host_written(crate::queue::presented());
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| !t.info.render_target())
            .map(|(&k, _)| k)
            .collect();
        let jobs = self.jobs(keys);
        self.sync(jobs);
    }

    /// What a submission does: the textures not uploaded yet, and those the scanner or hot probe
    /// found changed since. (`glue::submit` takes the same steps, the middle one without the
    /// host's lock.)
    pub fn sync_dirty(&mut self) {
        let jobs = self.sync_jobs();
        self.sync(jobs);
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn sync_dirty_without_probe_deadline(&mut self) {
        let jobs = self.sync_jobs_within(None);
        self.sync(jobs);
    }

    /// Reads `jobs` and uploads what changed.
    pub(super) fn sync(&mut self, jobs: Vec<Job>) {
        for batch in batches(jobs) {
            let prepared = prepare(&batch);
            self.sync_apply(prepared);
        }
    }

    /// The textures a submission must read: those not uploaded yet, and those the scanner or
    /// the probes of hot and blank textures found changed.
    pub(super) fn sync_jobs(&mut self) -> Vec<Job> {
        self.sync_jobs_within(Some(PROBE_BUDGET))
    }

    /// [`Self::sync_jobs`] with `budget` of time for each round of probes (none: every hot and
    /// blank texture is probed).
    pub(super) fn sync_jobs_within(&mut self, budget: Option<std::time::Duration>) -> Vec<Job> {
        self.sync_jobs_bounded(budget, SCAN_BUDGET)
    }

    /// [`Self::sync_jobs_within`], with at most `scan_budget` bytes of scanner-reported texture
    /// memory read this submission ([`SCAN_BUDGET`]; a test sets its own). What is left waits in
    /// the watch's dirty set for the next submission, so a world load's 0.9 to 1.9 GiB a frame of
    /// changed textures does not have to be read, hashed and detiled inside one frame.
    pub(super) fn sync_jobs_bounded(
        &mut self,
        budget: Option<std::time::Duration>,
        scan_budget: u64,
    ) -> Vec<Job> {
        self.sync_serial = self.sync_serial.wrapping_add(1);
        // A host writer may have opened a released copy's pages since the last sync (DirectStorage
        // staging, a file read, XG tiling): those bytes are new, so the copy goes before the
        // scanner decides what changed.
        self.drop_host_written(crate::queue::presented());
        self.release_placed();
        for key in self.host.take_wanted() {
            self.realize(key);
        }
        self.give_up_covered();
        self.release_quiescent(crate::queue::presented());
        self.keep_to_budget();
        let pools = self.host.memory();
        self.upgrade_trimmed_textures(&pools);
        let mut dirty = watch(|w| std::mem::take(&mut w.dirty));
        // What the scanner alone reported, before the probes add to `dirty` (`VICTORY_TRACE_FRAME`).
        #[cfg(target_os = "macos")]
        let scanner_dirty: HashSet<usize> = dirty.clone();
        // Hot textures (a movie's planes, a glyph cache) are probed synchronously, round-robin within
        // a time budget so world streaming (thousands of blank textures) never stalls submissions;
        // the ones that changed lately first, then the blank ones.
        let started = std::time::Instant::now();
        let until = budget.map(|budget| started + budget);
        probe_round(&pools, &mut self.textures, &mut self.hot_keys, &mut self.hot_cursor, until, Some(self.sync_serial), &mut self.probe_buf, &mut dirty);
        // The blank ones get a budget of their own: a busy movie must not starve its next planes.
        let hot = std::time::Instant::now();
        let until = budget.map(|budget| hot + budget);
        probe_round(&pools, &mut self.textures, &mut self.blank_keys, &mut self.blank_cursor, until, None, &mut self.probe_buf, &mut dirty);
        let probes = SyncCost { probe: [hot - started, hot.elapsed()], hot: self.hot_keys.len() as u64, ..Default::default() };
        probes.publish();
        if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
            probes.log();
        }
        let unloaded = std::mem::take(&mut self.unloaded);
        let released = &self.released;
        let mut keys: Vec<usize> = keys_to_sync(&unloaded, &dirty)
            .into_iter()
            .filter(|k| !released.contains_key(k))
            .collect();
        // A texture never uploaded goes now (its content is what a draw samples first); a change
        // the scanner or a probe found waits for the next submission once this frame's budget is
        // spent.
        let frame = crate::queue::presented();
        if self.budget.scan_frame != frame {
            self.budget.scan_frame = frame;
            self.budget.scan_bytes = 0;
        }
        let left = scan_budget.saturating_sub(self.budget.scan_bytes);
        let mut admitted = 0u64;
        let mut deferred = Vec::new();
        keys.retain(|&k| {
            if unloaded.contains(&k) {
                return true;
            }
            let Some(t) = self.textures.get(&k) else { return true };
            let size = t.info.size as u64;
            if scan_budget > 0 && admitted.saturating_add(size) > left {
                deferred.push(k);
                return false;
            }
            admitted = admitted.saturating_add(size);
            true
        });
        self.budget.scan_bytes = self.budget.scan_bytes.saturating_add(admitted);
        if !deferred.is_empty() {
            watch(|w| {
                for k in deferred {
                    w.dirty.insert(k);
                }
            });
        }
        // `VICTORY_TRACE_FRAME`: why this submission syncs what it does. A new sampled texture
        // (`first`), a host copy given up and needed again (`evicted`), a change the scanner's
        // watch saw (`scanner`), one a hot or blank texture's probe found (`probe`), or a target
        // the GPU wrote (`gpu written`, which `jobs` skips: nothing of it is read).
        #[cfg(target_os = "macos")]
        if super::hosttime::enabled() {
            let mut by: Vec<(&'static str, u64, u64)> = Vec::new();
            let mut add = |reason: &'static str, size: u64| match by.iter_mut().find(|(r, ..)| *r == reason) {
                Some((_, c, b)) => {
                    *c += 1;
                    *b += size;
                }
                None => by.push((reason, 1, size)),
            };
            for &k in &keys {
                let Some(t) = self.textures.get(&k) else { continue };
                let size = t.info.size as u64;
                let reason = if t.kept_from_memory() {
                    "gpu written"
                } else if unloaded.contains(&k) {
                    if t.last.is_none() { "first" } else { "evicted" }
                } else if dirty.contains(&k) && !scanner_dirty.contains(&k) {
                    "probe"
                } else {
                    "scanner"
                };
                add(reason, size);
            }
            for (reason, count, bytes) in by {
                super::hosttime::texture_why(reason, count, bytes);
            }
        }
        self.jobs(keys)
    }

    /// What this frame has left of [`FRAME_TEXTURE_WORK`] for textures: zero once a frame's
    /// submissions have spent it, whatever a world load lists. The work it does not do stays
    /// listed (changed, or not uploaded yet) for the next frame.
    pub(super) fn texture_work_left(&mut self, budget: std::time::Duration) -> std::time::Duration {
        self.texture_work_left_at(crate::queue::presented(), budget)
    }

    /// [`Self::texture_work_left`] in frame `frame` (tests: the frame counter does not move
    /// without a present).
    pub(super) fn texture_work_left_at(&mut self, frame: u64, budget: std::time::Duration) -> std::time::Duration {
        if self.budget.work_frame != frame {
            self.budget.work_frame = frame;
            self.budget.work_spent = std::time::Duration::ZERO;
        }
        budget.saturating_sub(self.budget.work_spent)
    }

    /// That this frame's textures took `took`: [`Self::texture_work_left`] shrinks by it.
    pub(super) fn spent_texture_work(&mut self, took: std::time::Duration) {
        self.budget.work_spent = self.budget.work_spent.saturating_add(took);
    }

    /// Textures listed for this submission's sync that this frame may not read: they stay
    /// changed, and a never-uploaded one stays a first upload, for the next submission
    /// ([`FRAME_TEXTURE_WORK`]).
    pub(super) fn defer_reads(&mut self, jobs: Vec<Job>) {
        if jobs.is_empty() {
            return;
        }
        watch(|w| {
            for j in &jobs {
                w.dirty.insert(j.key);
            }
        });
        for j in jobs {
            if j.last.is_none() {
                self.unloaded.insert(j.key);
            }
        }
    }

    pub(super) fn jobs(&self, keys: Vec<usize>) -> Vec<Job> {
        let pools = self.host.memory();
        let job = |key| {
            let t = self.textures.get(&key)?;
            // Depth-stencil memory is not laid out as the host's texels are (`realize`).
            if t.info.desc().is_some_and(|d| d.format.is_depth()) || t.kept_from_memory() {
                return None;
            }
            Some(Job {
                key,
                serial: t.serial,
                pools: pools.clone(),
                memory: t.info.memory as usize,
                size: t.info.size,
                format: t.info.format,
                dimension: t.info.dimension,
                layout: t.info.layout.clone(),
                last: t.last,
                uploaded_mips: t.uploaded_mips,
                partial: t.partial,
                skip: t.dropped,
                write_serial: t.write_serial,
            })
        };
        let out: Vec<Job> = keys.into_iter().filter_map(job).collect();
        if !out.is_empty() {
            watch(|w| {
                for j in &out {
                    if w.entries.contains_key(&j.key) {
                        *w.epoch.entry(j.key).or_default() += 1;
                    }
                }
            });
        }
        out
    }

    /// Expands any live trimmed textures whose previously unmapped leading fine mips now have pool
    /// mappings, re-mirroring heap entries and queueing them for fresh detile upload.
    fn upgrade_trimmed_textures(&mut self, pools: &PoolTable) {
        let pool_gen = pools.generation();
        if self.trimmed.is_empty() || pool_gen == self.trimmed_pool_gen {
            return;
        }
        self.trimmed_pool_gen = pool_gen;
        let keys: Vec<usize> = self.trimmed.iter().copied().collect();
        for key in keys {
            // Its pages are out: the copy is the only whole one, and a texture made again here
            // starts empty (the pages are put back before it is replaced).
            if self.released.contains_key(&key) {
                continue;
            }
            let Some((new_dropped, floor, new_desc)) = self.textures.get(&key).and_then(|t| {
                let d = unmapped_leading_mips(&t.info, pools);
                (d < t.dropped)
                    .then(|| t.info.host_desc_with(d).map(|desc| (d, t.info.dropped_levels(), desc)))
                    .flatten()
            }) else {
                if !self.textures.contains_key(&key) {
                    self.trimmed.remove(&key);
                }
                continue;
            };
            let Ok(new_tex) = self.host.texture(&new_desc) else {
                continue;
            };
            if let Some(t) = self.textures.get_mut(&key) {
                t.texture = new_tex.clone();
                t.dropped = new_dropped;
                t.uploaded_mips = 0;
                t.last = None;
                t.partial = new_dropped > floor;
                #[cfg(target_os = "macos")]
                t.target_views.clear();
            }
            self.host.upgrade_texture(key, &new_tex, new_dropped, new_desc.mips);
            let sharers: Vec<usize> = self
                .copies_of
                .iter()
                .filter(|(_, (o, _))| *o == key)
                .map(|(&s, _)| s)
                .collect();
            for sharer in sharers {
                self.host.upgrade_texture(sharer, &new_tex, new_dropped, new_desc.mips);
            }
            self.unloaded.insert(key);
            if new_dropped <= floor {
                self.trimmed.remove(&key);
            }
        }
    }

    /// Uploads what [`prepare`] detiled, into the textures that are still the ones it read.
    pub(super) fn sync_apply(&mut self, prepared: Prepared) {
        let Title {
            host,
            textures,
            sync_serial,
            hot_keys,
            blank_keys,
            stats,
            ..
        } = self;
        let started = std::time::Instant::now();
        let mut cost = SyncCost { batches: 1, prepare: prepared.took, ..Default::default() };
        let mut read = Vec::new();
        let mut streamed = Vec::new();
        for r in prepared.ready {
            cost.textures += 1;
            cost.bytes += r.bytes as u64;
            cost.read += r.cost[0];
            cost.hash += r.cost[1];
            cost.detile += r.cost[2];
            cost.unreadable += u64::from(r.unreadable);
            // Released while it was read (its key may name another texture by now).
            // Or a copy into it was recorded while it was read: the GPU's from now on.
            let Some(t) = textures.get_mut(&r.key).filter(|t| t.serial == r.serial && !t.kept_from_memory()) else {
                continue;
            };
            let key = r.key;
            t.partial = r.unreadable;
            t.skipped = 0;
            let entry = (t.info.memory as usize, t.info.size, t.info.render_target(), t.serial);
            cost.first += u64::from(t.last.is_none());
            // What a probe of this memory reads: the next one reports a write since, not this
            // content again (a texture's first probe used to, and that was a whole read wasted).
            let sampled = t.probe.replace(r.probe);
            // `last` again: a copy back into title memory may have moved it meanwhile.
            let Some((levels, detiled_mips, detiled)) = r.levels.filter(|_| t.last != Some(r.hash)) else {
                // Nothing changed since the last upload: if its previous write missed the samples,
                // this follow-up whole read has checked for an immediate next write, and further
                // checks return to `WHOLE_EVERY` within `WHOLE_SPARE` with backoff.
                if t.sampled_sees_writes {
                    t.unchanged_whole = t.unchanged_whole.saturating_add(1);
                } else {
                    t.sampled_sees_writes = true;
                }
                if !t.kept_from_memory() {
                    read.push((key, entry, r.hash, r.unreadable));
                }
                cost.unchanged += 1;
                continue;
            };
            t.unchanged_whole = 0;
            if let Some(sampled) = sampled {
                // Whoever found this write: one its samples do not show has it read whole at
                // every probe while it is hot, one they show only now and then.
                t.sampled_sees_writes = sampled != r.probe;
            }
            cost.blank += u64::from(r.blank);
            if !t.info.render_target() {
                if r.blank && !t.blank {
                    blank_keys.push(key);
                } else if !r.blank && t.uploads > 0 {
                    // Written again after its first upload: hot for the next `HOT` submissions.
                    if t.hot_until == 0 {
                        hot_keys.push(key);
                    }
                    t.hot_until = sync_serial.wrapping_add(u64::from(HOT));
                }
                t.blank = r.blank;
            }
            // Pass trace: uploads from title memory into textures the GPU writes (targets, UAVs).
            if super::pass_on() && (t.info.render_target() || t.info.flags & UNORDERED_ACCESS != 0) {
                eprintln!("PASSUPLOAD key={key:#x} fmt={} {}x{} at={:#x} rt={} uav={} uploads={} blank={} unreadable={}", t.info.format, t.info.width, t.info.height, t.info.memory as usize, t.info.render_target(), t.info.flags & UNORDERED_ACCESS != 0, t.uploads, r.blank, r.unreadable);
            }
            upload_levels(&host.backend, &t.texture, &levels);
            t.last = Some(r.hash);
            t.streamed = r.streamed;
            t.write_serial = r.read_serial;
            // Its memory changed: the title is using it (a streamed texture, a movie's plane).
            t.last_frame = crate::queue::presented();
            t.last_used_at = std::time::Instant::now();
            match detiled {
                Ok(()) => {
                    t.uploads += 1;
                    stats.detiles += 1;
                    if !t.info.render_target() && t.info.flags & UNORDERED_ACCESS == 0 {
                        t.uploaded_mips = if r.blank { 0 } else { t.uploaded_mips | detiled_mips };
                    }
                }
                Err(e) => {
                    stats.undetilable += 1;
                    tracing::warn!(texture = key, error = %e, "a title texture was not detiled");
                }
            }
            if !t.kept_from_memory() {
                read.push((key, entry, r.hash, r.unreadable));
            }
            // A streamed texture whose pages are all there: the streamer put the console's bytes
            // in title memory for the GPU, and the decoded copy holds them now. The title's pages
            // go back to the OS, so the texture costs one copy and not two — the win this path
            // exists for. A partial one (fine mips not mapped yet) waits: the next mip it streams
            // would fault a page back at once.
            if r.streamed && !r.blank && !r.unreadable && !t.restreamed {
                streamed.push(key);
            }
        }
        let pools = host.memory();
        watch(|w| {
            for (key, entry, hash, unreadable) in read {
                let (_, _, target, serial) = entry;
                let watch_entry = if target {
                    w.entries.get(&key).is_some_and(|e| e.3 == serial)
                } else {
                    w.entries.entry(key).or_insert(entry).3 == serial
                };
                if watch_entry {
                    w.pools.entry(key).or_insert_with(|| pools.clone());
                    *w.epoch.entry(key).or_default() += 1;
                    if unreadable {
                        w.partial.insert(key);
                    } else {
                        w.partial.remove(&key);
                    }
                    // What the scanner reported while this was read is this same content.
                    if w.seen.insert(key, hash) == Some(hash) {
                        w.dirty.remove(&key);
                    }
                }
            }
        });
        // Outside the loop above (it borrows the textures): a streamed copy's console bytes go back
        // to the OS now that the decoded copy holds them.
        if self.release_streamed {
            for key in streamed {
                self.release_pages(key);
            }
        }
        cost.upload = started.elapsed();
        cost.publish();
        if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
            cost.log();
        }
    }
}

/// A texture whose memory a submission reads: what the reading needs, taken under the host's lock
/// so that [`prepare`] runs without it.
pub(super) struct Job {
    pub(super) key: usize,
    serial: u64,
    pools: Arc<PoolTable>,
    memory: usize,
    size: usize,
    format: u32,
    dimension: Dimension,
    layout: Layout,
    /// The hash of what the host texture holds: memory that still hashes to it is not detiled.
    last: Option<ContentHash>,
    uploaded_mips: u32,
    /// Whether an earlier whole read met an unmapped or protected range.
    partial: bool,
    /// Top levels the host texture was made without ([`TextureInfo::dropped_levels`]).
    skip: usize,
    /// [`write_serial`] when its memory was last read whole ([`Tracked::write_serial`]).
    write_serial: u64,
}

/// What a [`Job`]'s memory held when [`prepare`] read it.
struct Ready {
    key: usize,
    serial: u64,
    bytes: usize,
    hash: ContentHash,
    /// What a sampled probe of that memory hashes to ([`probe_hash`]).
    probe: ContentHash,
    /// Some of it could not be read (and counts as zeros).
    unreadable: bool,
    /// All zeros.
    blank: bool,
    /// The content read arrived through a host writer (a DirectStorage read, a file read, XG
    /// tiling): [`host_wrote_since`] found a write after the last whole read.
    streamed: bool,
    /// [`write_serial`] before the read: what the next read's [`Self::streamed`] compares with.
    read_serial: u64,
    /// Detiled levels, bitmask of mips detiled, and how that went; `None` when the host texture
    /// holds this content already.
    levels: Option<(Vec<Level>, u32, Result<(), String>)>,
    /// Time reading, hashing and detiling.
    cost: [std::time::Duration; 3],
}

/// What [`prepare`] made of a batch of jobs, and how long it took.
pub(super) struct Prepared {
    ready: Vec<Ready>,
    took: std::time::Duration,
}

/// Threads a submission's textures are read on (the submitting thread is one of them).
const SYNC_THREADS: usize = 6;
/// Texture memory read at a time: its detiled copy is held until it is uploaded.
const SYNC_BATCH: usize = 256 << 20;

/// `jobs` in runs of at most `SYNC_BATCH` bytes of texture memory (a larger texture alone).
pub(super) fn batches(jobs: Vec<Job>) -> Vec<Vec<Job>> {
    let mut out: Vec<Vec<Job>> = Vec::new();
    let mut bytes = 0;
    for job in jobs {
        if out.is_empty() || bytes + job.size > SYNC_BATCH {
            out.push(Vec::new());
            bytes = 0;
        }
        bytes += job.size;
        out.last_mut().expect("pushed above").push(job);
    }
    out
}

impl Job {
    /// Reads its memory into `tiled` (kept from job to job: a fresh buffer each time maps and
    /// faults in new pages) and detiles it unless it is what the host texture holds.
    fn read(&self, tiled: &mut Vec<u8>) -> Ready {
        let started = std::time::Instant::now();
        // Before the read: a host write that lands after it is the next read's to see, and one that
        // landed before it (the streamer's) is this read's.
        let read_serial = write_serial();
        // Sampled before the whole read: a write between the two is then a change to the next
        // probe, not part of what it compares with.
        let probe = probe_hash(
            Some(&self.pools),
            self.memory,
            self.size,
            Some(&self.layout),
            false,
            self.partial,
            tiled,
        );
        if tiled.len() < self.size {
            tiled.resize(self.size, 0);
        }
        let tiled = &mut tiled[..self.size];
        let unreadable = read_into(Some(&self.pools), self.memory, tiled, self.partial);
        let read = started.elapsed();
        let hash = ContentHash::of(tiled);
        let hashed = started.elapsed();
        let changed = self.last != Some(hash);
        let addressable = self.layout.addressable();
        let blank = changed && addressable && is_all_zero(tiled);
        let levels = changed.then(|| {
            if blank && self.uploaded_mips == 0 {
                // Guess: fresh host textures start zeroed (`MTLStorageMode::Shared`; Metal does not
                // document it, see docs/native-host.md), so a never-uploaded
                // all-zero texture needs neither staging allocation, CPU detile, nor `replaceRegion`.
                (Vec::new(), 0, Ok(()))
            } else if let Some(levels) = crate::texture_cache::levels(
                &self.layout,
                self.dimension,
                self.format,
                &hash,
                self.uploaded_mips,
                self.skip,
            ) {
                // The same bytes in the same layout were detiled before (an earlier run, or this
                // texture streamed in again): no read of them was needed beyond the hash.
                let all = if self.uploaded_mips == 0 { u32::MAX } else { self.uploaded_mips };
                (levels, all, Ok(()))
            } else {
                let made = detile_changed(&self.layout, self.dimension, self.format, tiled, self.uploaded_mips, self.skip);
                if made.2.is_ok() {
                    crate::texture_cache::store(&self.layout, self.dimension, self.format, &hash, &made.0);
                }
                made
            }
        });
        Ready {
            key: self.key,
            serial: self.serial,
            bytes: self.size,
            hash,
            probe,
            unreadable,
            blank,
            streamed: host_wrote_since(self.memory, self.size, self.write_serial),
            read_serial,
            levels,
            cost: [read, hashed - read, started.elapsed() - hashed],
        }
    }
}

/// Reads, hashes and detiles the textures of `jobs`, in their order. This is the bulk of a texture
/// sync and needs nothing of the host, so it runs without the host's lock, on a few threads.
pub(super) fn prepare(jobs: &[Job]) -> Prepared {
    let started = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::Mutex::new(Vec::with_capacity(jobs.len()));
    let work = || {
        let mut tiled = Vec::new();
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(job) = jobs.get(i) else { break };
            let ready = job.read(&mut tiled);
            done.lock().unwrap_or_else(|e| e.into_inner()).push((i, ready));
        }
    };
    std::thread::scope(|s| {
        for _ in 1..SYNC_THREADS.min(jobs.len()) {
            // Without the thread, the others (this one at least) read its share.
            let _ = std::thread::Builder::new().name("victory texture read".into()).spawn_scoped(s, work);
        }
        work();
    });
    let mut done = done.into_inner().unwrap_or_else(|e| e.into_inner());
    done.sort_by_key(|&(i, _)| i);
    Prepared { ready: done.into_iter().map(|(_, r)| r).collect(), took: started.elapsed() }
}

/// What syncing textures cost: logged with `VICTORY_TRACE_TEXTURE_SYNC=1`, summed over each second.
#[derive(Default)]
struct SyncCost {
    batches: u64,
    textures: u64,
    /// Never uploaded before (a new texture, or a render target's memory written for the first time).
    first: u64,
    /// Read and hashed, and what the host texture holds already.
    unchanged: u64,
    /// All zeros (detiled and uploaded all the same).
    blank: u64,
    /// With memory that could not be read.
    unreadable: u64,
    bytes: u64,
    /// Hot textures at a submission (the last of the second is logged).
    hot: u64,
    /// Under the host's lock: probing hot textures and blank ones, and uploading.
    probe: [std::time::Duration; 2],
    upload: std::time::Duration,
    /// What the reading took the submission, without the host's lock.
    prepare: std::time::Duration,
    /// On the reading threads, summed: reading title memory, hashing it, detiling it.
    read: std::time::Duration,
    hash: std::time::Duration,
    detile: std::time::Duration,
}

impl SyncCost {
    /// `VICTORY_TRACE_FRAME`: what this submission's texture work cost, for the frame line (the
    /// same figures the one-second line below prints, without needing its switch).
    fn publish(&self) {
        #[cfg(target_os = "macos")]
        super::hosttime::textures(
            self.batches,
            self.textures,
            self.first,
            self.unchanged,
            self.bytes,
            self.probe[0] + self.probe[1],
            self.upload,
            self.prepare,
            self.read,
            self.hash,
            self.detile,
        );
    }

    fn log(self) {
        static SUM: std::sync::Mutex<Option<(std::time::Instant, [u64; 14])>> = std::sync::Mutex::new(None);
        let mut sum = SUM.lock().unwrap_or_else(|e| e.into_inner());
        let (since, s) = sum.get_or_insert_with(|| (std::time::Instant::now(), [0; 14]));
        let us = |d: std::time::Duration| d.as_micros() as u64;
        let add = [
            self.batches, self.textures, self.first, self.unchanged, self.blank, self.unreadable, self.bytes,
            us(self.probe[0]), us(self.probe[1]), us(self.upload), us(self.prepare), us(self.read), us(self.hash), us(self.detile),
        ];
        s.iter_mut().zip(add).for_each(|(s, a)| *s += a);
        static HOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if self.hot > 0 {
            HOT.store(self.hot, std::sync::atomic::Ordering::Relaxed);
        }
        if since.elapsed().as_secs() >= 1 {
            let take = |a: &std::sync::atomic::AtomicU64| a.swap(0, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "texture sync in the last {:.1} s: {} batches, {} textures ({} for the first time, {} unchanged, {} blank, {} partly unreadable), {} MiB; under the lock: probes of {} hot textures {} ms ({} read whole, {} MiB), of blank ones {} ms, uploads {} ms; off it: {} ms (on its threads: read {} ms, hash {} ms, detile {} ms); placements replaced {} ({} by one that was there before)",
                since.elapsed().as_secs_f32(), s[0], s[1], s[2], s[3], s[4], s[5], s[6] >> 20,
                take(&HOT), s[7] / 1000, take(&WHOLE_PROBES[0]), take(&WHOLE_PROBES[1]) >> 20, s[8] / 1000, s[9] / 1000,
                s[10] / 1000, s[11] / 1000, s[12] / 1000, s[13] / 1000,
                take(&REPLACED[0]), take(&REPLACED[1])
            );
            *sum = None;
        }
    }
}

/// The texture budget: how many bytes of sampled host texture copies the host keeps before it
/// gives up the least recently used of those it may ([`Title::keep_to_budget`]).
#[derive(Default)]
pub struct Budget {
    /// Bytes allowed (0: no budget).
    pub limit: u64,
    /// Bytes of every host texture copy made from a title texture (its size in title memory).
    pub live: u64,
    /// Frames a host texture copy may go unused before it is given up (`VICTORY_TEXTURE_IDLE_FRAMES`,
    /// 0: never).
    pub idle_frames: u64,
    /// Submissions so far: the clock [`Tracked::used`] is read on.
    serial: u64,
    /// The present ([`crate::queue::presented`]) the scanner's budget was last counted for, and
    /// the bytes of changed texture memory its submissions have admitted since.
    scan_frame: u64,
    scan_bytes: u64,
    /// The next submission a check over budget looks for textures to give up (a check that found
    /// too few waits [`RECHECK`] submissions: what entries show changes as the title rewrites them).
    next_check: u64,
    /// The next submission the idle check looks for textures to give up (it walks every texture,
    /// so it runs every [`IDLE_CHECK`] submissions rather than at each one).
    next_idle: u64,
    /// When the trace last logged.
    logged: Option<std::time::Instant>,
    /// Resources the tables of submissions being made (and of their held work) name, with how many
    /// of those submissions name each: never given up until the submission is handed to its queue
    /// ([`Title::keep_named`], [`Title::let_named_go`]).
    named: HashMap<usize, u32>,
    /// The frame ([`crate::queue::presented`]) the texture work below was last counted in, and how
    /// much of one frame's [`FRAME_TEXTURE_WORK`] its submissions have spent.
    work_frame: u64,
    work_spent: std::time::Duration,
}

impl Budget {
    /// Whether the trace logs at this submission's check (once a second).
    fn trace_due(&self) -> bool {
        env_set!("VICTORY_TRACE_TEXTURE_SYNC") && self.logged.is_none_or(|t| t.elapsed().as_secs() >= 1)
    }
}

/// Submissions between checks while over budget with too little to give up.
pub(super) const RECHECK: u64 = 64;

/// Submissions between idle checks: the check walks every made texture, so it runs every few
/// frames rather than at every submission (a copy may go a few frames after its threshold).
pub(super) const IDLE_CHECK: u64 = RECHECK;

/// Frames a copy must have gone unused for the budget to give it up first. A copy the last frames
/// used is one the world is reading: giving it up made it again at once and the picture flickered
/// (as with `SOON_FRAMES` in the Windows host). Past the limit the check takes copies older than
/// this first, and only when none is left the least recently used of the newer ones: staying over
/// the limit is what the budget exists to avoid.
pub(super) const SOON_FRAMES: u64 = 10;

/// The budget when Metal's working set leaves the title's memory too little (Series X on 16 GB).
/// Guess: about what a world frame samples; a smaller one would give up and remake textures each
/// frame.
pub const MIN_TEXTURE_BUDGET: u64 = 1536 << 20;

/// Bytes of decoded host texture copies to allow when nothing is set: a small pool, since a
/// texture whose layout Metal can address is sampled straight from title memory
/// ([`Title::zero_copy_texture`]) and holds none. The target is a total footprint near the
/// console's (8 GB on a Series S), so the decoded copies are hundreds of MiB, not the title's
/// memory.
pub const DEFAULT_TEXTURE_BUDGET: u64 = 512 << 20;

/// Bytes of decoded host texture copies to allow: `VICTORY_METAL_TEXTURE_BUDGET_MB` MiB, else
/// [`DEFAULT_TEXTURE_BUDGET`]. `0` (or anything not a number) turns the budget off.
pub fn texture_budget(var: Option<&str>) -> u64 {
    match var.map(str::trim) {
        None | Some("") => DEFAULT_TEXTURE_BUDGET,
        Some(v) => v.parse::<u64>().map_or(0, |mb| mb << 20),
    }
}

/// Frames a host texture copy may go unused before it is given up:
/// `VICTORY_TEXTURE_IDLE_FRAMES`. Off (0) unless that is set to a number: a copy given up is
/// made again from title memory, so a copy the title stops using but reads again later costs an
/// upload, and only runs can say how often that happens.
pub fn idle_frames(var: Option<&str>) -> u64 {
    var.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0)
}

/// The budget to try on this Mac, logged at startup: what Metal's recommended working set leaves
/// after the title's memory, less a quarter of the working set for everything else the process
/// holds (the host's own Metal allocations: descriptor heaps, pipelines, staging buffers, and the
/// app itself), at least [`MIN_TEXTURE_BUDGET`].
///
/// Guess: on a Series S run of 27 November (18 GB M3 Pro) Metal's working set was 13,639 MiB, the
/// title declared 7,776, and with 7,445 MiB of copies the footprint reached 14,324 MiB: the copies
/// cannot take all that the title's memory leaves, since the pools themselves and about 2.2 GB
/// more are the process's too.
pub fn suggested_texture_budget(working_set: u64, title_memory: u64) -> u64 {
    // Whole MiB: the figure is a setting a person types.
    let bytes = working_set
        .saturating_sub(title_memory)
        .saturating_sub(working_set / 4)
        .max(MIN_TEXTURE_BUDGET);
    bytes & !((1 << 20) - 1)
}

/// Which of `candidates` (key, frame last used) have gone unused for `idle_frames` frames at
/// `frame`, oldest use first (the lower key first among equals, so the choice does not depend on
/// map order). Empty with `idle_frames` 0 (idle give-up off).
pub(super) fn choose_idle(mut candidates: Vec<(usize, u64)>, frame: u64, idle_frames: u64) -> Vec<usize> {
    if idle_frames == 0 {
        return Vec::new();
    }
    candidates.retain(|&(_, last)| frame.saturating_sub(last) >= idle_frames);
    candidates.sort_unstable_by_key(|&(key, last)| (last, key));
    candidates.into_iter().map(|(key, _)| key).collect()
}

/// Which of `candidates` (key, bytes, last use) to give up to free `over` bytes: the least recently
/// used first (the lower key first among equals, so the choice does not depend on map order).
pub(super) fn choose_evictions(mut candidates: Vec<(usize, u64, u64)>, over: u64) -> Vec<usize> {
    candidates.sort_unstable_by_key(|&(key, _, used)| (used, key));
    let mut freed = 0;
    candidates
        .into_iter()
        .take_while(|&(_, bytes, _)| {
            let more = freed < over;
            freed += bytes;
            more
        })
        .map(|(key, ..)| key)
        .collect()
}

/// The physical memory under `memory` (`size` bytes), as an identity copies share by: the pool's
/// memory entry and the offset in it, so the same pool pages mapped at two title addresses are one
/// identity; the title address itself for memory no pool holds, or for a range only part of which
/// is mapped (its unmapped pages read zeros, which a fully mapped range over the same pages does
/// not). Pages that come from several mappings of the pools — a texture the title reported one page
/// at a time, or one over two of its pools — hash as the runs themselves, in order: the same pages
/// at two title addresses are still one identity. The top bit is set for pool memory, which no
/// title address has (the host's pools are its own mappings).
pub(super) fn memory_identity(pools: &PoolTable, memory: *const u8, size: usize) -> usize {
    let runs = pools.pool_ranges(memory as u64, size as u64);
    // A range the pools do not hold whole is not the same bytes: what a gap reads is zeros.
    if runs.is_empty() || runs.iter().map(|&(_, _, len)| len).sum::<u64>() != size as u64 {
        return memory as usize;
    }
    let mut h = blake3::Hasher::new();
    for (id, offset, len) in runs {
        let Some(key) = pools.pool_key(id) else { return memory as usize };
        for word in [key, offset, len] {
            h.update(&word.to_le_bytes());
        }
    }
    let b = h.finalize();
    (u64::from_le_bytes(b.as_bytes()[..8].try_into().expect("8 bytes")) | 1 << 63) as usize
}

/// Bytes of title memory under `ranges` (start, length), each byte once: copies over one memory
/// (aliasing objects, placed descriptions at other starts) are counted once here.
pub(super) fn distinct_bytes(mut ranges: Vec<(usize, usize)>) -> u64 {
    ranges.sort_unstable();
    let (mut total, mut end) = (0u64, 0usize);
    for (start, len) in ranges {
        let stop = start.saturating_add(len);
        if stop > end {
            total += (stop - start.max(end)) as u64;
            end = stop;
        }
    }
    total
}

/// `VICTORY_TEXTURE_ZERO_COPY=0` turns off sampling straight from title memory (every sampled
/// texture is decoded into a host copy instead). On by default where the layout allows it.
pub(crate) fn zero_copy_on() -> bool {
    !std::env::var("VICTORY_TEXTURE_ZERO_COPY").is_ok_and(|v| v == "0")
}

/// Textures sampled straight from title memory and the title bytes they cover: they have no
/// decoded copy, so they are not in the budget.
static ZERO_COPY: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// `madvise(MADV_FREE_REUSABLE)`: the pages can be reclaimed without being written to swap.
const MADV_FREE_REUSABLE: i32 = 7;

unsafe extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

/// Whether a copy last used at `last_used` (and at presented frame `last_frame`) has gone idle
/// long enough to give its pages back: the seconds threshold **or** the frames threshold,
/// whichever comes first. A paged world presents once every several seconds, so a frame count
/// alone can take hours (600 frames were 9 minutes with nothing released on 27 November); a fast
/// title reaches 600 frames before 20 s, so its frames decide.
pub(super) fn idle_enough(
    now: std::time::Instant,
    last_used: std::time::Instant,
    last_frame: u64,
    frame: u64,
    quiet_secs: u64,
    quiet_frames: u64,
) -> bool {
    now.saturating_duration_since(last_used).as_secs() >= quiet_secs
        || last_frame.saturating_add(quiet_frames) < frame
}

/// The whole 16 KiB pages the range `[at, at + len)` covers entirely.
fn page_range(at: usize, len: usize) -> (usize, usize) {
    let page = 1usize << 14;
    let start = at.saturating_add(page - 1) & !(page - 1);
    let end = at.saturating_add(len) & !(page - 1);
    (start, end.max(start))
}

unsafe extern "C" {
    fn mprotect(addr: *mut std::ffi::c_void, len: usize, prot: i32) -> i32;
}
const PROT_NONE: i32 = 0;
const PROT_READ_WRITE: i32 = 3;

/// Makes `[at, at + len)` unreadable and unwritable (a title access faults; the helper thread
/// re-materialises the page), or readable and writable again.
fn protect(at: usize, len: usize, prot: i32) -> Result<(), ()> {
    // SAFETY: the caller gives a mapped range of title memory.
    (unsafe { mprotect(at as *mut _, len, prot) } == 0).then_some(()).ok_or(())
}

fn madvise_reusable(at: usize, len: usize) -> Result<(), ()> {
    // SAFETY: the caller gives a mapped range of title memory.
    let rc = unsafe { madvise(at as *mut _, len, MADV_FREE_REUSABLE) };
    if rc != 0 {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| tracing::warn!(at = format!("{at:#x}"), len, errno = std::io::Error::last_os_error().raw_os_error(), "MADV_FREE_REUSABLE refused"));
        return Err(());
    }
    Ok(())
}

/// `VICTORY_TEXTURE_RELEASE_PAGES=0` turns it off: give up a decoded copy's title pages to the OS
/// once nothing has used the copy for a while, so the decoded Metal texture is the only resident
/// copy of the bytes (`Title::release_pages`). The pages are `PROT_NONE` while released, so a
/// title access faults into the `page-restore` helper (`victory_platform::faults`), which puts
/// them back from the texture before the access runs. On by default.
pub fn release_pages_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_TEXTURE_RELEASE_PAGES").is_ok_and(|v| v == "0"))
}

/// `VICTORY_TEXTURE_RELEASE_STREAMED=0` turns it off: give a streamed copy's title pages back to
/// the OS as soon as it is uploaded, rather than once it has gone unused for `release_seconds`.
/// A texture whose bytes a DirectStorage read put into title memory is there for the GPU: the
/// decoded Metal texture is the only copy anything samples, and the title's own CPU access to
/// those pages (which a streamer does not do) faults them back from it (`Title::release_pages`).
/// That is the duplicate copy: the console's bytes in the title's pages beside the decoded one.
/// On by default, and pointless without the page-fault path (`faults::active`).
pub fn release_streamed_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_TEXTURE_RELEASE_STREAMED").is_ok_and(|v| v == "0"))
}

/// The process's resident bytes and `phys_footprint`, as the memory line samples them (0 off
/// macOS): `VICTORY_TRACE_RELEASE_RSS` logs both across a release batch.
fn resident_and_footprint() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    return (victory_platform::mem::resident() as u64, victory_platform::mem::footprint() as u64);
    #[cfg(not(target_os = "macos"))]
    (0, 0)
}

/// Pages given back to the OS by [`Title::release_pages`], and the titles whose pages are out.
static RELEASED_TRACE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// Of those, the streamed copies and bytes given back as soon as they were uploaded
/// ([`Title::release_streamed`]): the duplicate copy removed. [2] and [3] are the ones a host write
/// took back (a texture the streamer fills again), and their bytes.
static RELEASED_STREAM: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Streamed copies whose title pages were given back, and their bytes: what
/// `VICTORY_TEXTURE_RELEASE_STREAMED` removed from title memory.
pub fn streamed_released_bytes() -> u64 {
    RELEASED_STREAM[1].load(std::sync::atomic::Ordering::Relaxed)
}

/// Streamed copies whose released pages a host write opened again (the streamer filling a texture
/// it had finished), and their bytes: a cost of releasing eagerly, logged beside the win.
pub fn streamed_released_undone() -> (u64, u64) {
    (
        RELEASED_STREAM[2].load(std::sync::atomic::Ordering::Relaxed),
        RELEASED_STREAM[3].load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Bytes of decoded copies whose title pages were given back to the OS, and how many.
pub fn released_bytes() -> u64 {
    RELEASED_TRACE[1].load(std::sync::atomic::Ordering::Relaxed)
}

/// Seconds a decoded copy must go unused before its title pages are released
/// (`VICTORY_TEXTURE_RELEASE_SECONDS`, default 20). The clock is wall time: a paged world
/// presents once every several seconds, so a frame count is no measure of "nothing uses it" (600
/// frames was 9 minutes with no release on 27 November).
pub fn release_seconds() -> u64 {
    static SECONDS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SECONDS.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_RELEASE_SECONDS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(20)
    })
}

/// Frames a decoded copy must also go unused for (`VICTORY_TEXTURE_RELEASE_FRAMES`, default 600):
/// whichever of the two thresholds is reached first releases it, so a fast title's frames and a
/// paged one's seconds both work.
pub fn release_frames() -> u64 {
    static FRAMES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FRAMES.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_RELEASE_FRAMES")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(600)
    })
}

/// What the decoded copies are needed for, in bytes, and the zero-copy side: the memory report's
/// answer to "why is the copy pool this big" (`glue::memory_parts`).
#[derive(Default, Clone, Copy, Debug)]
pub struct DecodedReport {
    /// Bytes of every decoded copy: the reasons below (`tiled` to `other`) add up to it exactly,
    /// each copy counted once under the first reason [`Title::decoded_reason`] finds.
    pub decoded: u64,
    /// Uncompressed, one image, and the console's swizzle is not linear.
    pub tiled: u64,
    /// Block-compressed (BCPack's output, unpacked by the title), one image of one mip.
    pub bcpack: u64,
    /// Block-compressed with a mip chain, slices or a volume: the typical streamed asset.
    pub bc_mipped: u64,
    /// A format Metal cannot read as the console's bytes.
    pub format: u64,
    /// Depth and stencil.
    pub depth: u64,
    /// Uncompressed, and a mip chain, a slice stack or a volume.
    pub layout: u64,
    /// Multisampled.
    pub multisampled: u64,
    /// A linear layout whose rows or offset miss Metal's alignment, or no pool holds it.
    pub alignment: u64,
    /// A copy Metal could have taken from title memory (zero-copy off, or creation refused).
    pub other: u64,
    /// Of the decoded copies, those the GPU writes (render targets, depth, unordered access), and
    /// the rest, which hold only the title's bytes (what page release and the budget act on):
    /// the two add up to `decoded`.
    pub written: u64,
    pub sampled_only: u64,
    /// Of the decoded copies, those whose content arrived through a host writer (a DirectStorage
    /// read, a file read, XG tiling): the streamer's textures, whose console bytes are all still in
    /// title memory as well — the duplicate copy this host exists to remove.
    pub streamed: u64,
    /// Of those, the ones only the console's swizzle kept from being a texture over title memory
    /// (detiled, not decoded from another format).
    pub streamed_detiled: u64,
    /// Of those, the ones unpacked from BCPack/BCn (a decoder, not a detile).
    pub streamed_bcpack: u64,
    /// Bytes of title memory under the decoded copies ([`Title::memory_under_copies`]): with the
    /// copies, what a streamed texture costs twice.
    pub under_copies: u64,
    /// Of those, bytes of copies over memory another live copy already holds
    /// ([`Title::duplicate_copies`]): the same pool pages at one title address or several.
    pub duplicate_copies: u64,
    /// Bytes and count sampled or rendered where the title wrote them.
    pub zero_copy: u64,
    pub zero_copy_copies: u64,
    /// Of the decoded copies, those a block permutation could turn into the pages themselves
    /// ([`Title::in_place_bytes`]): uncompressed, tiled, one mip, sampled only, and their linear
    /// image fits the bytes the title allocated. Decoding those in place would leave one resident
    /// copy instead of two.
    pub in_place: u64,
    pub in_place_copies: u64,
}

impl DecodedReport {
    /// The reasons, in the memory line's order: `tiled`, `bcpack`, `bc_mipped`, `format`,
    /// `depth`, `layout`, `multisampled`, `alignment`, `other`. They add up to `decoded`.
    pub fn reasons(&self) -> [u64; 9] {
        [
            self.tiled,
            self.bcpack,
            self.bc_mipped,
            self.format,
            self.depth,
            self.layout,
            self.multisampled,
            self.alignment,
            self.other,
        ]
    }

    /// [`Self::reasons`] in whole MiB that add up to `decoded >> 20` exactly: each is rounded
    /// down, and the MiB the rounding lost go to those with the largest remainders.
    pub fn reasons_mib(&self) -> [u64; 9] {
        let bytes = self.reasons();
        let mut mib = bytes.map(|b| b >> 20);
        let total = bytes.iter().sum::<u64>() >> 20;
        let mut short = total.saturating_sub(mib.iter().sum());
        let mut order: Vec<usize> = (0..bytes.len()).collect();
        order.sort_by_key(|&i| (std::cmp::Reverse(bytes[i] & ((1 << 20) - 1)), i));
        for i in order {
            if short == 0 {
                break;
            }
            mib[i] += 1;
            short -= 1;
        }
        mib
    }
}

impl<B: PresentBackend> Title<B> {
    /// The bytes of `t`'s decoded copy when a block permutation of the title's own pages could
    /// hold Metal's linear image in the same bytes (`None`: it cannot): an uncompressed, tiled,
    /// single-mip 2D texture that is only sampled. The console's tiling moves whole elements and
    /// the format is Metal's own, so the copy would be the pages themselves, no second resident
    /// copy. Block-compressed textures do not qualify: Metal cannot make a texture over a buffer
    /// in a BC format. What keeps this from being done is the title's own later use of those
    /// pages (a mip streamed into them, a copy out of them), which needs write protection; this
    /// measures what would be won.
    fn in_place_bytes(&self, t: &Tracked<B>) -> Option<u64> {
        if t.zero_copy {
            return None;
        }
        let info = &t.info;
        let desc = info.host_desc_with(t.dropped)?;
        // `block()` is (texels per block side, bytes per block).
        let (side, element_bytes) = desc.format.block();
        // A texture over a buffer cannot be block-compressed (Metal asserts), so only an
        // uncompressed tiled format could be permuted into Metal's linear image in place.
        if side != 1
            || desc.mips != 1
            || info.dimension != Dimension::Tex2d
            || info.depth_or_layers != 1
            || info.render_target()
            || info.flags & UNORDERED_ACCESS != 0
            || desc.format.is_depth()
            || info.format == 115
            || info.layout.desc.swizzle == Swizzle::LINEAR
            || format(info.format)? != desc.format
        {
            return None;
        }
        // Metal's rows, and the row pitch the pages can hold: at least the width's bytes, and a
        // whole element's multiple.
        let rows = u64::from(info.height.max(1));
        let need = u64::from(info.width) * element_bytes as u64;
        let pitch = info.layout.size / rows;
        (pitch >= need && pitch.is_multiple_of(element_bytes as u64)).then_some(info.size as u64)
    }

    /// The info of copy `key` (tests: what a re-make of the same bytes is given).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn info_for_test(&self, key: usize) -> Option<TextureInfo> {
        self.textures.get(&key).map(|t| t.info.clone())
    }

    /// Whether `key`'s title pages are out (tests: [`Self::release_pages`]).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn is_released(&self, key: usize) -> bool {
        self.released.contains_key(&self.owner_of(key))
    }

    /// Bytes of decoded copies whose title pages are out (see [`Self::release_pages`]).
    pub fn released_bytes(&self) -> u64 {
        self.released.values().copied().sum()
    }

    /// Gives the title pages of decoded copy `key` back to the OS: `madvise(MADV_FREE_REUSABLE)`
    /// over the pool's own pages, then `PROT_NONE` over them, so the decoded Metal texture is the
    /// only resident copy and the pages can be reclaimed instead of swapped. The pages are
    /// registered with `victory_platform::faults` before they are protected: a title read or store
    /// then faults, the loader's signal handler hands it to the `page-restore` helper, and
    /// [`Self::restore_at`] puts the console bytes back from the texture (`Layout::tile` through
    /// `download`, as `write_back` does) and makes the pages writable before the access runs. Only
    /// a copy nothing samples (no table names it, no submission could reach it, not this frame)
    /// qualifies.
    ///
    /// `false` for a texture that is not a decoded copy, is GPU-written (its texels are not the
    /// title's), is a render or depth target, or whose memory is not one pool run.
    pub(super) fn release_pages(&mut self, key: usize) -> bool {
        // No helper, no protection: a title access would fault with nobody to serve it.
        if !faults::active() {
            return false;
        }
        let Some(t) = self.textures.get(&key) else { return false };
        // Whole pages the texture covers alone: protecting a page only partly its own would fault
        // (or hide) a neighbouring texture's title stores too.
        if t.zero_copy
            || t.gpu_written
            || t.info.render_target()
            || t.info.flags & UNORDERED_ACCESS != 0
            || !t.info.layout.addressable()
        {
            return false;
        }
        let info = t.info.clone();
        // Only the title's own mapping is released and protected. Its range must start on a page
        // and cover whole ones, so no neighbour allocation (a fence value, a smaller texture)
        // shares a page with it. The pool's alias of the same bytes is left writable: it may be
        // less aligned (4 KiB in an x86-64 process under Rosetta) and protecting it would round
        // out over its neighbours.
        let (tstart, tend) = page_range(info.memory as usize, info.size);
        let len = tend.saturating_sub(tstart);
        if len == 0
            || tstart != info.memory as usize
            || tend != (info.memory as usize).saturating_add(info.size)
            || madvise_reusable(tstart, len).is_err()
        {
            return false;
        }
        // Unreadable and unwritable, so a title access faults and the helper
        // (`victory_platform::faults`) re-materialises the page from the texture before the access
        // runs. Registered first, so the handler knows the page is ours.
        faults::release(tstart, len);
        if protect(tstart, len, PROT_NONE).is_err() {
            // Not every mapping allows it: put the pages back and keep decoding this one.
            faults::restore(tstart, len);
            let _ = protect(tstart, len, PROT_READ_WRITE);
            return false;
        }
        self.released.insert(key, info.size as u64);
        RELEASED_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        RELEASED_TRACE[1].fetch_add(info.size as u64, std::sync::atomic::Ordering::Relaxed);
        if t.streamed {
            RELEASED_STREAM[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            RELEASED_STREAM[1].fetch_add(info.size as u64, std::sync::atomic::Ordering::Relaxed);
        }
        // The scanner and the probes read title memory: a read of a page that is out is zeros (the
        // kernel's read fails, nothing faults for it), which would upload over the copy. The copy
        // is not watched until its pages are back (`restore_pages`), and a title or host write
        // that opens a page is what brings it back for a fresh read (`drop_host_written`).
        watch(|w| w.skip(key));
        true
    }

    /// Puts `key`'s console bytes back into its (released) title pages, re-tiled from the decoded
    /// texture, and makes them writable again: what a fault handler does before letting the
    /// title's access run, and what is needed before the copy is given up or read again.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(super) fn restore_pages(&mut self, key: usize) -> bool {
        let Some(&bytes) = self.released.get(&key) else { return false };
        let Some(t) = self.textures.get(&key) else {
            self.released.remove(&key);
            return false;
        };
        let info = t.info.clone();
        let serial = t.serial;
        let Ok((_buffer, at)) = self.host.resolve(info.memory as u64, info.size as u64) else { return false };
        // Ours no more first (a fault on a page already restored must not wait), then writable.
        // Only the title's mapping was released; the pool's alias stayed writable.
        let (tstart, tend) = page_range(info.memory as usize, info.size);
        faults::restore(tstart, tend - tstart);
        let _ = protect(tstart, tend - tstart, PROT_READ_WRITE);
        let mut tiled = vec![0u8; info.size];
        if download(&self.host.backend, &t.texture, &info, &mut tiled).is_err() {
            return false;
        }
        // SAFETY: `resolve` gave `size` mapped, writable bytes of the pool; the buffer is kept.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), at.cast_mut(), tiled.len()) };
        if let Some(t) = self.textures.get_mut(&key) {
            t.last = Some(ContentHash::of(&tiled));
            // The title read or wrote those pages: they are its own now, and are not given back
            // again (a copy it reads back every frame must not fault every frame).
            t.streamed = false;
        }
        // Watched again, from the content it holds: the scanner's next round compares with it
        // rather than reporting the restore as a write.
        let pools = self.host.memory();
        watch(|w| {
            w.entries.insert(key, (info.memory as usize, info.size, info.render_target(), serial));
            w.pools.insert(key, pools);
            w.seen.insert(key, hash);
        });
        self.released.remove(&key);
        tracing::debug!(texture = key, bytes, "a decoded texture's pages are back in title memory");
        true
    }

    /// Drops the decoded copies whose pages a host writer opened ([`faults::open_for_write`]): a
    /// DirectStorage request staging into title memory, a file read, XG tiling. The written pages
    /// are open and out of the set; the rest of the copy is opened here, and the copy leaves
    /// `released` as stale — the bytes in title memory are the new ones, so it is read again
    /// (`unloaded`) instead of being restored from a texture that holds the old ones.
    fn drop_host_written(&mut self, frame: u64) {
        if self.released.is_empty() {
            return;
        }
        let now = std::time::Instant::now();
        let keys: Vec<usize> = self
            .released
            .keys()
            .copied()
            .filter(|&k| {
                self.textures
                    .get(&k)
                    .is_some_and(|t| !faults::all_out(t.info.memory as usize, t.info.size))
            })
            .collect();
        for key in keys {
            let Some(t) = self.textures.get(&key) else {
                self.released.remove(&key);
                continue;
            };
            let info = t.info.clone();
            let (tstart, tend) = page_range(info.memory as usize, info.size);
            faults::restore(tstart, tend - tstart);
            let _ = protect(tstart, tend - tstart, PROT_READ_WRITE);
            self.released.remove(&key);
            if let Some(t) = self.textures.get_mut(&key) {
                t.last_used_at = now;
                t.last_frame = frame;
                // The streamer is filling it again: releasing its pages again would only fault
                // them back into another whole re-read.
                t.restreamed = true;
            }
            if let Some(t) = self.textures.get(&key) {
                RELEASED_STREAM[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                RELEASED_STREAM[3].fetch_add(t.info.size as u64, std::sync::atomic::Ordering::Relaxed);
            }
            self.unloaded.insert(key);
        }
    }

    /// Releases the pages of decoded copies nothing has used for [`release_frames`] frames (with
    /// `VICTORY_TEXTURE_RELEASE_PAGES=1`). Every released copy is skipped by the scanner: reading
    /// pages the OS may have emptied would upload zeros over it.
    fn release_quiescent(&mut self, frame: u64) {
        self.release_quiescent_at(frame, std::time::Instant::now());
    }

    /// [`Self::release_quiescent`] at `now`: a copy goes once it has gone unused for the seconds
    /// threshold **or** the frames threshold, whichever comes first (a paged world's frames are
    /// minutes apart; a fast title's seconds pass between two of its frames).
    ///
    /// Nothing is released while the copies are over the budget: the budget's next check would
    /// take those same idle copies, and to take a released one it first re-tiles it back into
    /// the title's pages from the GPU ([`Self::keep_to_budget_at`]), work thrown away at once.
    pub(super) fn release_quiescent_at(&mut self, frame: u64, now: std::time::Instant) {
        if !self.release_pages || self.budget.limit == 0 || self.budget.live > self.budget.limit {
            return;
        }
        let (quiet_secs, quiet_frames) = (release_seconds(), release_frames());
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(k, t)| {
                !self.released.contains_key(k)
                    && !self.budget.named.contains_key(k)
                    && idle_enough(now, t.last_used_at, t.last_frame, frame, quiet_secs, quiet_frames)
            })
            .map(|(&k, _)| k)
            .collect();
        // `VICTORY_TRACE_RELEASE_RSS`: whether giving pages back frees any RAM. The pool's Metal
        // buffer maps the same pages (`newBufferWithBytesNoCopy`), which may keep them resident.
        let trace = env_set!("VICTORY_TRACE_RELEASE_RSS") && !keys.is_empty();
        let before = trace.then(resident_and_footprint);
        let (mut copies, mut bytes) = (0u64, 0u64);
        for key in keys {
            if self.release_pages(key) {
                copies += 1;
                bytes += self.released.get(&key).copied().unwrap_or(0);
            }
        }
        if let Some((resident, footprint)) = before
            && copies > 0
        {
            let (resident_after, footprint_after) = resident_and_footprint();
            let mib = |after: u64, before: u64| ((after as f64 - before as f64) / 1048576.0 * 10.0).round() / 10.0;
            tracing::info!(
                released_copies = copies,
                released_mib = (bytes as f64 / 1048576.0 * 10.0).round() / 10.0,
                released_total_mib = self.released_bytes() >> 20,
                resident_delta_mib = mib(resident_after, resident),
                footprint_delta_mib = mib(footprint_after, footprint),
                resident_mib = resident_after >> 20,
                footprint_mib = footprint_after >> 20,
                "release batch: decoded copies' title pages given back, and the process's resident size and footprint across the batch"
            );
        }
    }

    /// [`Self::restore_pages`] for the released copy whose title pages hold `addr`: what the fault
    /// helper calls (it holds the title lock, so this is ordinary code, not a signal handler).
    pub fn restore_at(&mut self, addr: usize) -> bool {
        let key = self.released.keys().copied().find(|k| {
            self.textures.get(k).is_some_and(|t| {
                let at = t.info.memory as usize;
                (at..at.saturating_add(t.info.size)).contains(&addr)
            })
        });
        key.is_some_and(|k| self.restore_pages(k))
    }

    /// [`DecodedReport`] of the copies there are now.
    pub fn decoded_report(&self) -> DecodedReport {
        let mut r = DecodedReport::default();
        for t in self.textures.values() {
            let size = t.info.size as u64;
            if t.zero_copy {
                r.zero_copy += size;
                r.zero_copy_copies += 1;
                continue;
            }
            r.decoded += size;
            if self.in_place_bytes(t).is_some() {
                r.in_place += size;
                r.in_place_copies += 1;
            }
            if t.info.render_target() || t.gpu_written || t.info.flags & UNORDERED_ACCESS != 0 {
                r.written += size;
            } else {
                r.sampled_only += size;
            }
            // One reason per copy, so the reasons add up to `decoded`.
            let desc = t.info.host_desc_with(t.dropped);
            let reason = desc.as_ref().and_then(|d| self.decoded_reason(&t.info, d));
            if t.streamed {
                r.streamed += size;
                let block_compressed = desc.as_ref().is_some_and(|d| d.format.block().0 == 4);
                if block_compressed || reason == Some(Decoded::BcPack) {
                    r.streamed_bcpack += size;
                } else {
                    r.streamed_detiled += size;
                }
            }
            match reason {
                Some(Decoded::Tiled) => r.tiled += size,
                Some(Decoded::BcPack) => r.bcpack += size,
                Some(Decoded::BcMipped) => r.bc_mipped += size,
                Some(Decoded::Format) => r.format += size,
                Some(Decoded::Depth) => r.depth += size,
                Some(Decoded::Layout) => r.layout += size,
                Some(Decoded::Multisampled) => r.multisampled += size,
                Some(Decoded::Alignment) => r.alignment += size,
                None => r.other += size,
            }
        }
        r.under_copies = self.memory_under_copies().0;
        r.duplicate_copies = self.duplicate_copies().0;
        r
    }
}

/// Why a decoded copy is needed: Metal cannot sample (or render into) the title's bytes as they
/// lie. [`Title::decoded_report`] sums the pool by these for the memory line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decoded {
    /// The console's swizzle (4KB_S, 64KB_*, ...) is not linear.
    Tiled,
    /// Block-compressed (BC1-BC7: BCPack's output, unpacked by the title), one image of one mip.
    /// Metal has the formats but not over a buffer: `minimumLinearTextureAlignmentForPixelFormat`
    /// asserts "linear textures do not support compressed pixel formats", so these always need a
    /// decoded copy.
    BcPack,
    /// Block-compressed with a mip chain, slices or a volume (a streamed asset): as `BcPack`.
    BcMipped,
    /// The format is not Metal's own bytes for it (a conversion on upload, such as DXGI
    /// `B4G4R4A4`'s rotated nibbles).
    Format,
    /// Depth and stencil: Metal's depth texel layout and the host's depth twins.
    Depth,
    /// Uncompressed, with more than one mip, or a slice/array or 3D volume, which one
    /// buffer-backed texture cannot hold: the mip chain is packed into the title's linear layout.
    Layout,
    /// Multisampled.
    Multisampled,
    /// A linear layout whose rows or offset miss Metal's `linear_alignment`, or no pool holds it.
    Alignment,
}

/// For the trace, as of the last submission: the bytes of copies and how many a table names (which
/// must stay) and those that only a heap entry a shader indexes shows (possible reads, not names:
/// the budget may give them up).
static HEAP_ONLY: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// For the trace: copies given up though a heap entry still showed them (a possible read, not a
/// name: the budget prefers the memory), and their bytes.
static SHOWN_TRACE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// For the trace: textures given up under the budget, their bytes, and evicted ones made again.
static BUDGET_TRACE: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// For the trace: copies made again within [`named_keep`] submissions of the budget giving them
/// up (the decode and upload it caused again), and their bytes.
pub(super) static CHURN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Submissions a copy given up is watched for a re-make in, whatever the eviction rule
/// (`VICTORY_TEXTURE_NAMED_KEEP=0` included): the churn figure is comparable across runs.
const CHURN_WINDOW: u64 = 8;

/// Submissions a copy a submission's tables named is kept for,
/// whatever its place in the LRU order: the budget was giving up copies a later draw of the same
/// or the next frame needed, and they were decoded and uploaded again under the host lock (the
/// world's 400-1600 textures, 0.9-1.9 GiB a frame). `VICTORY_TEXTURE_NAMED_KEEP=0` restores the
/// old rule, so the churn can be measured against the same world.
pub(super) fn named_keep() -> u64 {
    static KEEP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_NAMED_KEEP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}
/// For the trace: textures given up for going unused (`VICTORY_TEXTURE_IDLE_FRAMES`), and their
/// bytes.
static IDLE_TRACE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// For the trace: placed copies given up as another placement covered them, and their bytes.
static OVERLAP_TRACE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// For the trace: title textures that took an identical live copy's texture instead of their own.
static SHARED_TRACE: [std::sync::atomic::AtomicU64; 1] = [const { std::sync::atomic::AtomicU64::new(0) }; 1];

impl<B: PresentBackend> Title<B> {
    /// Sets the texture budget in bytes (0: none).
    pub fn set_texture_budget(&mut self, bytes: u64) {
        self.budget.limit = bytes;
    }

    /// Sets how many frames a host texture copy may go unused before it is given up (0: never).
    pub fn set_texture_idle_frames(&mut self, frames: u64) {
        self.budget.idle_frames = frames;
    }

    /// Stamps title texture `key` as last used at presented frame `frame` (tests: the frame
    /// counter does not move without a present).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn set_texture_last_frame(&mut self, key: usize, frame: u64) {
        if let Some(t) = self.textures.get_mut(&self.owner_of(key)) {
            t.last_frame = frame;
        }
    }

    /// Stamps title texture `key` as reachable by the work of GPU epoch `reach` (tests).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn set_texture_reach(&mut self, key: usize, reach: u64) {
        if let Some(t) = self.textures.get_mut(&self.owner_of(key)) {
            t.reach = reach;
        }
    }

    /// Bytes of host texture copies the budget counts (tests).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn budget_live_for_test(&self) -> u64 {
        self.budget.live
    }

    /// Counts `key` as named by a submission in flight, or not (tests: [`Self::keep_named`] names
    /// what a submission's tables reach, which needs a recorded draw).
    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn name_texture_for_test(&mut self, key: usize, named: bool) {
        if named {
            self.budget.named.insert(key, 1);
            let naming = self.budget.serial + 1;
            if let Some(t) = self.textures.get_mut(&key) {
                t.last_named = naming;
            }
        } else {
            self.budget.named.remove(&key);
        }
    }

    /// Whether `t` (title texture `key`) may be given up: a sampled texture filled only from title
    /// memory (not a render target, depth or GPU-written texture, nor one ever viewed for unordered
    /// access), that no recorded or submitted work holds (the host's two references are all there
    /// are, or one for a copy several textures share), and that no work still in flight could read
    /// through a heap entry it shows. A heap-indexing shader takes its index from root constants
    /// or memory, so an entry showing it may be read by any draw binding its heap: the copy goes
    /// only once every submission that bound such a heap before it was last reached has finished
    /// ([`Tracked::reach`], [`Self::retired_epoch`]), and a later one that reaches it makes it
    /// again ([`Self::keep_named`]). The budget's first tier: a copy the GPU wrote goes only after
    /// every one of these ([`Self::written_evictable`]).
    fn evictable(&self, key: usize, t: &Tracked<B>) -> bool {
        !t.zero_copy
            && !self.released.contains_key(&key)
            && self.holds_only_memory(key, t)
            && Rc::strong_count(&t.texture) <= 2
            && (!self.host.shown_by_heap(key) || t.reach <= self.retired_epoch)
    }

    /// Whether no submission's tables named `t` for [`named_keep`] submissions: a copy a draw of
    /// this or the next frame reads is not given up (it would be decoded and uploaded again at
    /// once, which is what the world's frames were spending).
    fn named_long_enough_ago(&self, t: &Tracked<B>) -> bool {
        t.last_named == 0 || self.budget.serial.saturating_sub(t.last_named) > named_keep()
    }

    /// Whether copy `t` (title texture `key`, which the GPU wrote) may be given up although what it
    /// holds is the GPU's, not title memory's: nothing recorded or submitted holds it, no work in
    /// flight could read it through a heap entry, and no use at presented frame `frame` touched it
    /// (a render target the current frame draws into, for one). Made again it starts empty, so only
    /// what the title redraws, recomputes or rewrites comes back; it is the budget's last resort,
    /// after every copy filled from title memory (`keep_to_budget_at`).
    fn written_evictable(&self, key: usize, t: &Tracked<B>) -> bool {
        // A zero-copy texture wrote title memory itself: there is no copy to give up.
        if t.zero_copy {
            return false;
        }
        // "Not this presented frame" was the old rule, and in a world that presents every 7.8 s it
        // never held: `crate::queue::presented()` barely moves, so a render target used once had
        // `last_frame == frame` for minutes and nothing the GPU wrote was ever given up (the run of
        // 27 November 22:23: 4.5 GB of copies against a 3 GB budget, 2.1 GB of them written). The
        // submission being synced is the unit that matters: a copy it names is in `budget.named`
        // and stays, one it does not and no recorded or in-flight work holds (`named`, the strong
        // count, `reach`) may go, and the next frame that draws into it makes it again empty.
        !self.holds_only_memory(key, t)
            && Rc::strong_count(&t.texture) <= 2
            && (!self.host.shown_by_heap(key) || t.reach <= self.retired_epoch)
            && !self.budget.named.contains_key(&key)
    }

    /// Whether copy `t` (title texture `key`) holds only what title memory does: a sampled texture
    /// (not a render target or depth), not GPU-written, never viewed for unordered access.
    fn holds_only_memory(&self, key: usize, t: &Tracked<B>) -> bool {
        !t.info.render_target()
            && !t.gpu_written
            && t.info.desc().is_some_and(|d| !d.format.is_depth())
            && !self.host.ever_unordered(key)
    }

    /// Remembers that `info`'s bytes were given up at this submission, so a copy made again over
    /// them soon is counted ([`CHURN`]).
    fn note_evicted(&mut self, info: &TextureInfo) {
        let at = memory_identity(&self.host.memory(), info.memory, info.size);
        self.evicted.insert(at, (self.budget.serial, info.size as u64));
        if self.evicted.len() > 8192 {
            let now = self.budget.serial;
            self.evicted.retain(|_, &mut (serial, _)| now.saturating_sub(serial) <= 1024);
        }
    }

    /// Counts a copy made again over bytes the budget just gave up: the decode and upload the
    /// named window exists to stop.
    fn note_remade(&mut self, info: &TextureInfo) {
        let at = memory_identity(&self.host.memory(), info.memory, info.size);
        if let Some((serial, bytes)) = self.evicted.remove(&at) {
            if self.budget.serial.saturating_sub(serial) <= CHURN_WINDOW {
                for (i, v) in [(0, 1), (1, bytes), (2, 1), (3, bytes)] {
                    CHURN[i].fetch_add(v, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// For the trace: bytes of host copies over memory another live copy holds (the same pool
    /// pages at one title address or several, where layout or format kept them from sharing), and
    /// how many such memories have more than one copy.
    pub(super) fn duplicate_copies(&self) -> (u64, usize) {
        let mut per: HashMap<usize, (u64, u64, u32)> = HashMap::new();
        for t in self.textures.values() {
            let e = per.entry(t.at).or_insert((0, 0, 0));
            let size = t.info.size as u64;
            e.0 += size;
            e.1 = e.1.max(size);
            e.2 += 1;
        }
        let (mut duplicate, mut memories) = (0, 0);
        for &(sum, largest, count) in per.values() {
            if count > 1 {
                duplicate += sum - largest;
                memories += 1;
            }
        }
        (duplicate, memories)
    }

    /// For the trace: bytes of title memory under the copies, each byte once, and bytes of copies
    /// outside the title's pools. Copies in a pool are placed by their pool offset, so pages the
    /// title maps at two addresses count once.
    pub(super) fn memory_under_copies(&self) -> (u64, u64) {
        let pools = self.host.memory();
        let (mut ranges, mut outside) = (Vec::new(), 0);
        for t in self.textures.values() {
            match pools.resolve(t.info.memory as u64, 1) {
                Some((_, at)) => ranges.push((at as usize, t.info.size)),
                None => {
                    ranges.push((t.info.memory as usize, t.info.size));
                    outside += t.info.size as u64;
                }
            }
        }
        (distinct_bytes(ranges), outside)
    }

    /// Before the sync of a submission of `lists` (with the work held already): a texture the
    /// records of the tables they bind name, or any texture of a view heap one of their
    /// heap-indexing shaders indexes, is made again if it was given up, and uploaded at that sync;
    /// none is given up then. The host mirrors those records when it encodes the lists, and an
    /// entry naming a texture not made yet shows nothing: the draw would sample transparent black.
    /// The names stay out of the budget's reach until the caller hands them to
    /// [`Self::let_named_go`], once the submission is with its queue, so another submission's sync
    /// in between cannot give them up. Nothing is read under the budget with nothing given up
    /// (`glue::submit` runs one submission at a time, so none goes over budget in between), unless
    /// the trace logs at this submission, so its figures match a run with the budget on.
    #[must_use = "the names stay kept until handed to `let_named_go`"]
    /// Whether [`Self::keep_named`] has anything to do: over budget, a texture given up, or the
    /// trace due. Checked before the lists' snapshots are taken for it.
    pub(super) fn names_wanted(&self) -> bool {
        let over = self.budget.limit > 0 && self.budget.live > self.budget.limit;
        // The next submission runs the idle sweep (`keep_to_budget` counts it): its tables are
        // read first, so a copy they name is made again and none they name is given up by it.
        let idle_due =
            self.budget.idle_frames > 0 && self.budget.serial.wrapping_add(1) >= self.budget.next_idle;
        // A copy a descriptor named under the budget was left pending: the next submission's
        // tables decide which of those are sampled, and make them.
        over || !self.deferred.is_empty() || idle_due || self.host.any_evicted() || self.budget.trace_due()
    }

    pub(super) fn keep_named<'a>(&mut self, lists: impl Iterator<Item = &'a NativeList<B>>) -> HashSet<usize>
    where
        B: 'a,
    {
        if !self.names_wanted() {
            return HashSet::new();
        }
        // shortcut: a record changed after this read (before held work is encoded) still gets the
        // cleared entry and `take_wanted`; reading again at encoding needs the title's state there.
        let held: Vec<Rc<NativeList<B>>> = self.held.segments().cloned().collect();
        let mut all: Vec<&NativeList<B>> = lists.collect();
        all.extend(held.iter().map(|l| &**l));
        // What the submission's tables name is what its draws and dispatches read: it must be
        // there, and stays out of the budget's reach until the submission is with its queue. What
        // only a heap entry shows is a *possible* read (a heap-indexing shader takes its index
        // from memory): naming those kept every copy a bindless title ever put in a heap out of
        // the budget's reach for good, since every submission names the whole heap again, and the
        // limit was never enforced (#79, #294). They are not named: the budget may give one up and
        // clear its entries, which a shader that indexes one of them samples as transparent black,
        // and a record a table names later makes it again as for any given-up copy.
        let named = self.host.resources_named(all.iter().copied());
        // For the trace: what only a heap a shader indexes shows, which the budget may take.
        let shown = self.host.shown_by_indexed_heaps(all.iter().copied());
        let only_heap = shown.iter().filter(|k| !named.contains(k));
        let bytes: u64 = only_heap
            .clone()
            .filter_map(|k| self.textures.get(&self.owner_of(*k)))
            .map(|t| t.info.size as u64)
            .sum();
        let count = only_heap.count() as u64;
        let named_bytes: u64 = named
            .iter()
            .filter_map(|k| self.textures.get(&self.owner_of(*k)))
            .map(|t| t.info.size as u64)
            .sum();
        HEAP_ONLY[0].store(bytes, std::sync::atomic::Ordering::Relaxed);
        HEAP_ONLY[1].store(count, std::sync::atomic::Ordering::Relaxed);
        HEAP_ONLY[2].store(named_bytes, std::sync::atomic::Ordering::Relaxed);
        HEAP_ONLY[3].store(named.len() as u64, std::sync::atomic::Ordering::Relaxed);
        // The submission about to be encoded is the next GPU epoch: a copy it names stays until
        // the work queued so far has finished, which is when the GPU has read the entries this
        // submission and the ones before it could name.
        let epoch = self.gpu_epoch;
        self.record_epoch(epoch);
        let naming = self.budget.serial + 1;
        for &key in &named {
            // A copy left pending by a descriptor under the budget is made here, as one given up
            // by the budget is made again: its heap entries were written showing nothing.
            let deferred = self.deferred.remove(&key);
            if deferred || self.host.is_evicted(key) {
                self.realize(key);
                // A bindless heap's entries are outside every table span this submission's draws
                // bind, so nothing else writes back the ones the budget cleared.
                self.host.remirror_shown(key);
            }
            let owner = self.owner_of(key);
            if let Some(t) = self.textures.get_mut(&owner) {
                t.reach = epoch;
                t.last_named = naming;
            }
        }
        for &key in &named {
            *self.budget.named.entry(key).or_default() += 1;
        }
        named
    }

    /// The submission that [`Self::keep_named`] returned `named` for is with its queue: what only
    /// it named may be given up again.
    pub(super) fn let_named_go(&mut self, named: HashSet<usize>) {
        for key in named {
            if let Some(n) = self.budget.named.get_mut(&key) {
                *n -= 1;
                if *n == 0 {
                    self.budget.named.remove(&key);
                }
            }
        }
    }

    /// Past the budget, gives up the least recently used textures that may be given up
    /// ([`Self::evictable`]) until under it; with `VICTORY_TEXTURE_IDLE_FRAMES` set, also every
    /// one nothing has used for that many frames, under the budget or not. One a view names again
    /// is made and uploaded again. Textures the submission's tables name or reach through a heap a
    /// heap-indexing shader indexes ([`Self::keep_named`]) are stamped used now; one a heap entry
    /// merely shows is not, so it keeps aging until no work in flight can read it and the check
    /// takes it. When no copy filled from title memory is left the check takes the GPU's own, least
    /// recently used first ([`Self::written_evictable`]): a world's copies are nearly all render
    /// targets and unordered-access targets, and a limit none of them counts against is not a
    /// limit. What the GPU wrote is gone then; the title redraws or recomputes it, and the host
    /// makes the copy again empty at its next use (the hosts's `pending`, [`Self::realize`]).
    pub(super) fn keep_to_budget(&mut self) {
        self.keep_to_budget_at(crate::queue::presented());
    }

    /// [`Self::keep_to_budget`] at presented frame `frame` (tests: the frame counter does not move
    /// without a present).
    pub(super) fn keep_to_budget_at(&mut self, frame: u64) {
        self.budget.serial += 1;
        let serial = self.budget.serial;
        let trace = self.budget.trace_due();
        let check = self.budget.limit > 0
            && self.budget.live > self.budget.limit
            && serial >= self.budget.next_check;
        let idle_frames = self.budget.idle_frames;
        let idle_due = idle_frames > 0 && serial >= self.budget.next_idle;
        if !check && !trace && !idle_due {
            return;
        }
        // Copied from a heap entry is only safe once the work that could have read it has finished.
        self.retire_gpu_epochs();
        if check {
            // A released copy's bytes live only in its Metal texture (its title pages are
            // `PROT_NONE`), so the tiers below cannot take it as it stands: that is how copies
            // passed the budget by everything released (28 November 08:10: 3546 MiB against a
            // 3.00 GB cap, the HUD's TEX line). Put the console bytes back into the title's pages
            // first - that memory is the title's, not the budget's - oldest first, until what is
            // over the limit is covered, and the pass below may then take them like any other copy.
            let mut back: Vec<(std::time::Instant, usize, u64)> = self
                .released
                .iter()
                .filter_map(|(&k, &bytes)| {
                    let t = self.textures.get(&k)?;
                    // Only what the budget could take once its pages are back: a copy a live
                    // submission or a heap entry in flight could still read stays released.
                    (self.holds_only_memory(k, t)
                        && Rc::strong_count(&t.texture) <= 2
                        && (!self.host.shown_by_heap(k) || t.reach <= self.retired_epoch))
                    .then(|| (t.last_used_at, k, bytes))
                })
                .collect();
            back.sort_unstable_by_key(|&(used, ..)| used);
            let mut restored = 0u64;
            for (_, key, bytes) in back {
                if restored >= self.budget.live - self.budget.limit {
                    break;
                }
                if self.restore_pages(key) {
                    restored += bytes;
                }
            }
        }
        // Two tiers may go. A copy filled only from title memory costs one upload to make again,
        // so it goes first; one the GPU wrote (a render target, a depth texture, one it wrote or
        // wrote through an unordered view) starts empty when made again, so the budget takes one
        // only when no copy of the first tier is left, and never one this presented frame touched.
        let (mut old, mut recent, mut idle) = (Vec::new(), Vec::new(), Vec::new());
        // Named by a submission in the last [`named_keep`] submissions: taken only when nothing
        // colder can cover the budget (they come back decoded and uploaded, which is the churn).
        let (mut old_named, mut recent_named) = (Vec::new(), Vec::new());
        let (mut old_written, mut recent_written) = (Vec::new(), Vec::new());
        let (mut shown, mut stale, mut written) = (0, 0, 0);
        let keys: Vec<usize> = self.textures.keys().copied().collect();
        for key in keys {
            let t = &self.textures[&key];
            let named = self.budget.named.contains_key(&key);
            let memory_only = self.holds_only_memory(key, t);
            let may_go = if memory_only {
                self.evictable(key, t)
            } else {
                written += t.info.size as u64;
                self.written_evictable(key, t)
            };
            let seen = named || self.host.texture_in_use(key);
            // Used now: a submission names it, or one in flight could read it through an entry.
            // Merely being shown does not count, or a bindless heap would keep every copy it ever
            // showed looking freshly used and neither the idle sweep nor the LRU order would age.
            let used_now = named || t.reach > self.retired_epoch;
            if seen {
                shown += t.info.size as u64;
                if may_go {
                    // A heap entry shows it, but no submission that could read that entry is
                    // still in flight: the budget may take it (it comes back if one reaches it).
                    stale += t.info.size as u64;
                }
            }
            // Over the budget, a copy the last frames used is given up only when no older one is
            // left: giving it up makes it again at once (the picture flickers).
            if !named && may_go {
                let candidate = (key, t.info.size as u64, t.used);
                let (old, recent) = if memory_only {
                    if self.named_long_enough_ago(t) {
                        (&mut old, &mut recent)
                    } else {
                        (&mut old_named, &mut recent_named)
                    }
                } else {
                    (&mut old_written, &mut recent_written)
                };
                if frame.saturating_sub(t.last_frame) > SOON_FRAMES {
                    old.push(candidate);
                } else {
                    recent.push(candidate);
                }
                idle.push((key, t.last_frame));
            }
            if used_now {
                let t = self.textures.get_mut(&key).expect("listed");
                (t.used, t.last_frame) = (serial, frame);
            }
        }
        let evictable: u64 = old.iter().chain(&recent).map(|c| c.1).sum();
        let evictable_written: u64 = old_written.iter().chain(&recent_written).map(|c| c.1).sum();
        if idle_due {
            self.budget.next_idle = serial + IDLE_CHECK;
            for key in choose_idle(idle, frame, idle_frames) {
                let Some(t) = self.textures.get(&key) else { continue };
                let bytes = t.info.size as u64;
                let info = t.info.clone();
                self.note_evicted(&info);
                self.host.evict_texture(key);
                self.texture_released(key);
                // Made again when a view names it next ([`Self::realize`]).
                self.pending.insert(key, info);
                IDLE_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                IDLE_TRACE[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            }
            // What the idle sweep gave up is gone from the over-budget candidates too.
            old.retain(|&(key, ..)| self.textures.contains_key(&key));
            recent.retain(|&(key, ..)| self.textures.contains_key(&key));
            old_written.retain(|&(key, ..)| self.textures.contains_key(&key));
            recent_written.retain(|&(key, ..)| self.textures.contains_key(&key));
        }
        if check && self.budget.live > self.budget.limit {
            // Copies filled from title memory are taken down to a low-water mark: enforced byte by
            // byte at the cap, the next submission was over again and took what it had just made,
            // which is the churn. A sixteenth of the cap is the hysteresis. The GPU's own copies
            // (render targets and depth) are only ever taken down to the cap, so headroom never
            // costs what the title redraws.
            let low = self.budget.limit - self.budget.limit / 16;
            let mut chosen: Vec<usize> = Vec::new();
            let mut freed = 0u64;
            for (tier_index, tier) in
                [old, recent, old_named, recent_named, old_written, recent_written].into_iter().enumerate()
            {
                let target = if tier_index < 4 { low } else { self.budget.limit };
                let need = self.budget.live.saturating_sub(target).saturating_sub(freed);
                if need == 0 {
                    break;
                }
                let pick = choose_evictions(tier, need);
                freed += pick
                    .iter()
                    .filter_map(|k| self.textures.get(k))
                    .map(|t| t.info.size as u64)
                    .sum::<u64>();
                chosen.extend(pick);
            }
            for key in chosen {
                let Some(t) = self.textures.get(&key) else { continue };
                let bytes = t.info.size as u64;
                let info = t.info.clone();
                if self.host.shown_by_heap(key) {
                    SHOWN_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    SHOWN_TRACE[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                }
                self.note_evicted(&info);
                self.host.evict_texture(key);
                self.texture_released(key);
                // Made again when a view names it next ([`Self::realize`]).
                self.pending.insert(key, info);
                BUDGET_TRACE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BUDGET_TRACE[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            }
            if self.budget.live > self.budget.limit {
                self.budget.next_check = serial + RECHECK;
            }
        }
        if trace {
            self.budget.logged = Some(std::time::Instant::now());
            let take = |a: &std::sync::atomic::AtomicU64| a.swap(0, std::sync::atomic::Ordering::Relaxed);
            let (distinct, outside) = self.memory_under_copies();
            let (duplicate, memories) = self.duplicate_copies();
            // What the process's footprint is made of, so a run that pages can say which part to
            // cut: the title's own memory (the pools the host maps, which its copies duplicate),
            // Metal's own count, and the footprint the HUD shows beside them.
            let (_, maps) = self.host.memory().mappings();
            let mapped: u64 = maps.iter().map(|&(_, len, ..)| len as u64).sum();
            let metal = victory_backend::backend::ComputeBackend::allocated_bytes(&self.host.backend);
            #[cfg(target_os = "macos")]
            let footprint = victory_platform::mem::footprint() as u64;
            #[cfg(not(target_os = "macos"))]
            let footprint = 0u64;
            tracing::info!(
                "host texture copies: {} MiB, budget {} MiB, idle {} frames; {} MiB shown by heap entries, named by the submission's tables or viewed for unordered access, {} MiB could be given up, {} MiB of that only by entries no work in flight can read yet; {} MiB the GPU wrote (render targets, depth, unordered access), {} MiB of those could be given up; given up in the last second {} ({} MiB), {} ({} MiB) idle, made again {}, {} made again within {} submissions of being given up ({} MiB) in the last second and {} ({} MiB) so far; {} MiB of distinct title memory under the copies, {} MiB of copies outside the title's pools, {} MiB of copies duplicating another copy of the same memory ({} memories); in the last second {} textures shared an identical copy ({} share one now), {} placed copies another placement covered were given up ({} MiB), {} covered still in use; title memory mapped {} MiB in {} mappings, Metal {} MiB, footprint {} MiB, {} copies waiting for a submission to name them",
                self.budget.live >> 20, self.budget.limit >> 20, idle_frames, shown >> 20, evictable >> 20, stale >> 20,
                written >> 20, evictable_written >> 20,
                take(&BUDGET_TRACE[0]), take(&BUDGET_TRACE[1]) >> 20, take(&IDLE_TRACE[0]), take(&IDLE_TRACE[1]) >> 20,
                take(&BUDGET_TRACE[2]),
                take(&CHURN[0]), CHURN_WINDOW, take(&CHURN[1]) >> 20,
                take(&CHURN[2]), take(&CHURN[3]) >> 20,
                distinct >> 20, outside >> 20, duplicate >> 20, memories,
                take(&SHARED_TRACE[0]), self.copies_of.len(), take(&OVERLAP_TRACE[0]), take(&OVERLAP_TRACE[1]) >> 20, self.covered.len(),
                mapped >> 20, maps.len(), metal >> 20, footprint >> 20, self.deferred.len(),
            );
            let (zero, zero_bytes) = self
                .textures
                .values()
                .filter(|t| t.zero_copy)
                .fold((0u64, 0u64), |(n, b), t| (n + 1, b + t.info.size as u64));
            tracing::info!(
                "texture zero copy: {zero} textures ({} MiB of title memory) sampled where the title wrote them, {} made so far; {} decoded copies ({} MiB)",
                zero_bytes >> 20,
                ZERO_COPY[0].load(std::sync::atomic::Ordering::Relaxed),
                self.textures.len() as u64 - zero,
                self.budget.live >> 20
            );
            let load = |n: usize| HEAP_ONLY[n].load(std::sync::atomic::Ordering::Relaxed);
            if load(0) > 0 || load(2) > 0 {
                tracing::info!(
                    "texture budget: {} MiB in {} copies a submission's tables name (kept for its draw), {} MiB in {} copies only heap entries a shader indexes show (a possible read: the budget may give them up)",
                    load(2) >> 20,
                    load(3),
                    load(0) >> 20,
                    load(1)
                );
            }
            let (given_up_shown, given_up_shown_mib) = (take(&SHOWN_TRACE[0]), take(&SHOWN_TRACE[1]) >> 20);
            if given_up_shown > 0 {
                tracing::info!(
                    "texture budget: {given_up_shown} copies ({given_up_shown_mib} MiB) given up though a heap entry still showed them; a shader that indexes one of their cleared entries samples transparent black"
                );
            }
        }
    }
}

fn keys_to_sync(unloaded: &HashSet<usize>, dirty: &HashSet<usize>) -> Vec<usize> {
    unloaded.iter().chain(dirty.difference(unloaded)).copied().collect()
}

/// The textures of a title host, by title resource key.
pub type Textures<B> = HashMap<usize, Tracked<B>>;

/// Render-target shapes the title created `(w, h, format, flags)`: how many, their bytes in title
/// memory, and their layer count -- the memory line's top ten is what says which of them to cut.
type TargetShapes = HashMap<(u32, u32, u32, u32), (u64, u64, u32)>;
static TARGET_SHAPES: std::sync::Mutex<Option<TargetShapes>> =
    std::sync::Mutex::new(None);

/// The render-target shapes by bytes, largest first, as `(bytes, count, w, h, format, layers,
/// render_target)` -- for the accounting line.
pub fn render_target_shapes() -> Vec<(u64, u64, u32, u32, u32, u32, bool)> {
    let shapes = TARGET_SHAPES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(shapes) = shapes.as_ref() else { return Vec::new() };
    let mut out: Vec<_> = shapes
        .iter()
        .map(|(&(w, h, format, flags), &(count, bytes, layers))| {
            (bytes, count, w, h, format, layers, flags & RENDER_TARGET != 0)
        })
        .collect();
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.3.cmp(&a.3)));
    out
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// The release threshold is whichever comes first: a paged world's seconds, a fast title's
    /// frames. A copy used a second ago is kept even after 10,000 frames (a paged title's frame
    /// count does not move).
    #[test]
    fn the_release_threshold_is_seconds_or_frames() {
        let now = std::time::Instant::now();
        let used = |secs: u64| now - std::time::Duration::from_secs(secs);
        // Seconds: 21 s idle with 20 s the threshold, while the frame count never moved.
        assert!(idle_enough(now, used(21), 0, 0, 20, 600));
        // Just used: neither threshold is met, though many frames passed.
        assert!(!idle_enough(now, used(1), 0, 10_000, 20, 600));
        // Frames: 600 presented frames with a fresh clock (a fast title) release too.
        assert!(idle_enough(now, used(1), 0, 601, 20, 600));
        // A just-used copy at frame 601 with a fresh clock: frames decide.
        assert!(!idle_enough(now, used(1), 601, 601, 20, 600));
    }

    /// The memory line's reasons in whole MiB add up to the decoded pool's MiB: rounding each
    /// down alone would lose up to a MiB per reason.
    #[test]
    fn the_reasons_in_mib_add_up_to_the_decoded_pool() {
        let half = 1u64 << 19;
        let r = DecodedReport {
            tiled: (3 << 20) + half + 1,
            bcpack: (2 << 20) + half,
            bc_mipped: 700 << 20,
            layout: half - 1,
            decoded: (3 << 20) + half + 1 + (2 << 20) + half + (700 << 20) + half - 1,
            ..Default::default()
        };
        assert_eq!(r.reasons().iter().sum::<u64>(), r.decoded);
        let mib = r.reasons_mib();
        assert_eq!(mib.iter().sum::<u64>(), r.decoded >> 20, "{mib:?}");
        // 706.5 MiB, 705 rounded down one by one: the lost MiB goes to the largest remainder.
        assert_eq!(&mib[..3], &[4, 2, 700]);
        assert_eq!(mib[5], 0);
    }

    /// A title texture at `memory`, for the [`written_twin`] test.
    fn described(
        format: u32,
        (width, height, depth): (u32, u32, u32),
        mips: u32,
        samples: u32,
        flags: u32,
        layout_word: u32,
        memory: usize,
    ) -> TextureInfo {
        use victory_formats::texture::{Desc, Swizzle};
        let (element_bytes, block) = crate::resource::element(format).unwrap();
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width,
            height,
            depth,
            mips,
            element_bytes: element_bytes as u32,
            block: block as u32,
            samples,
            // Multisampled layouts may not be linear; the swizzle does not matter here.
            swizzle: if samples > 1 { Swizzle::SW_4KB_S } else { Swizzle::LINEAR },
        })
        .unwrap_or_else(|e| panic!("{format} {width}x{height}x{depth} mips {mips} samples {samples}: {e:?}"));
        TextureInfo {
            dimension: Dimension::Tex2d,
            format,
            width,
            height,
            depth_or_layers: depth,
            flags,
            size: layout.size as usize,
            layout,
            layout_word,
            memory: memory as *const u8,
        }
    }

    /// `VICTORY_PLACED_TWIN`'s match ([`written_twin`]): a written texture of the same shape is
    /// what a placed view can read, in any format of the same element; another memory, shape,
    /// element, sample count or shorter chain is not, and neither is a texture the GPU does not
    /// write (a depth one from its colour view: the host cannot cast a depth texture).
    #[test]
    fn a_written_twin_is_the_same_shape_and_element() {
        // DXGI 10: R16G16B16A16_FLOAT, 8 bytes an element, a render target.
        let written = described(10, (32, 32, 1), 4, 1, RENDER_TARGET, 0x104, 0x1000);
        // DXGI 17: R32G32_UINT, also 8 bytes.
        let view = described(17, (32, 32, 1), 4, 1, 0, 0, 0x1000);
        assert!(written_twin(&view, &written));
        // Any written use is enough (depth aside), and the view may have fewer mips.
        assert!(written_twin(&view, &TextureInfo { flags: UNORDERED_ACCESS, ..written.clone() }));
        assert!(written_twin(&view, &TextureInfo { flags: RENDER_TARGET | UNORDERED_ACCESS, ..written.clone() }));
        assert!(written_twin(&described(17, (32, 32, 1), 1, 1, 0, 0, 0x1000), &written));
        // Not what the GPU writes.
        assert!(!written_twin(&view, &TextureInfo { flags: 0, ..written.clone() }));
        // Another memory, shape, depth or sample count.
        assert!(!written_twin(&view, &described(10, (32, 32, 1), 4, 1, RENDER_TARGET, 0x104, 0x2000)));
        assert!(!written_twin(&view, &described(10, (16, 32, 1), 4, 1, RENDER_TARGET, 0x104, 0x1000)));
        assert!(!written_twin(&view, &described(10, (32, 16, 1), 4, 1, RENDER_TARGET, 0x104, 0x1000)));
        assert!(!written_twin(&view, &described(10, (32, 32, 2), 4, 1, RENDER_TARGET, 0x104, 0x1000)));
        assert!(!written_twin(&view, &described(10, (32, 32, 1), 1, 2, RENDER_TARGET, 0x104, 0x1000)));
        // Another element (RGBA8, 4 bytes), a longer chain than the written texture has, or block
        // formats of one element size that are not each other's view (BC1 and BC4).
        assert!(!written_twin(&described(28, (32, 32, 1), 4, 1, 0, 0, 0x1000), &written));
        assert!(!written_twin(&described(17, (32, 32, 1), 5, 1, 0, 0, 0x1000), &written));
        assert!(!written_twin(
            &described(80, (32, 32, 1), 4, 1, 0, 0, 0x1000),
            &described(71, (32, 32, 1), 4, 1, RENDER_TARGET, 0x104, 0x1000)
        ));
        // A depth texture's memory is not read through a colour view.
        let depth = described(40, (32, 32, 1), 4, 1, DEPTH_STENCIL, 0x104, 0x1000);
        assert!(!written_twin(&described(41, (32, 32, 1), 4, 1, 0, 0, 0x1000), &depth));
    }

    #[test]
    fn sync_keys_include_overlaps_once_and_keep_unloaded_first() {
        let unloaded = (0..20_000).collect();
        let dirty = (10_000..30_000).collect();
        let mut keys = keys_to_sync(&unloaded, &dirty);

        let (first, rest) = keys.split_at_mut(unloaded.len());
        first.sort_unstable();
        rest.sort_unstable();
        assert!(first.iter().copied().eq(0..20_000));
        assert!(rest.iter().copied().eq(20_000..30_000));
    }

    fn watched(entries: impl Into<HashMap<usize, (usize, usize, bool, u64)>>) -> Watched {
        Watched::new(Some(Watch { entries: entries.into(), ..Default::default() }))
    }

    fn dirty_this_round(watched: &Watched) -> std::collections::HashSet<usize> {
        scan_round(watched, &mut [0; 2]);
        watch_in(watched, |w| std::mem::take(&mut w.dirty))
    }

    fn seen(watched: &Watched) -> HashMap<usize, ContentHash> {
        watch_in(watched, |w| w.seen.clone())
    }

    #[test]
    fn scanner_forgets_removed_keys_and_reports_them_when_they_reappear() {
        let bytes = [1u8, 2, 3];
        let entry = (bytes.as_ptr() as usize, bytes.len(), false, 0);
        let w = watched([(1, entry), (2, entry)]);

        assert_eq!(dirty_this_round(&w), [1, 2].into());
        assert!(dirty_this_round(&w).is_empty());

        watch_in(&w, |w| w.entries.remove(&1));
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w), HashMap::from([(2, ContentHash::of(&bytes))]));

        watch_in(&w, |w| w.entries.insert(1, entry));
        assert_eq!(dirty_this_round(&w), [1].into());
    }

    #[test]
    fn scanner_uses_the_first_render_target_hash_as_its_baseline() {
        let mut bytes = [3u8];
        let w = watched([(1, (bytes.as_ptr() as usize, bytes.len(), true, 0))]);
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w)[&1], ContentHash::of(&bytes));

        bytes[0] = 4;
        assert_eq!(dirty_this_round(&w), [1].into());
        assert_eq!(seen(&w)[&1], ContentHash::of(&bytes));
    }

    /// What a submission read whole is what the scanner compares with: it reports the write after
    /// it, not that content once more.
    #[test]
    fn scanner_compares_with_what_a_submission_read() {
        let mut bytes = [3u8];
        let w = watched([(1, (bytes.as_ptr() as usize, bytes.len(), false, 0))]);
        assert_eq!(dirty_this_round(&w), [1].into());
        bytes[0] = 4;
        // A submission read it meanwhile (`sync_apply`).
        watch_in(&w, |w| w.seen.insert(1, ContentHash::of(&bytes)));
        assert!(dirty_this_round(&w).is_empty());
        bytes[0] = 5;
        assert_eq!(dirty_this_round(&w), [1].into());
    }

    /// A render target released and made again under the same key (an object's address is used
    /// again) starts over: its first sight is its baseline. Compared with the old texture's hash
    /// it counted as written, and its memory was uploaded over what the GPU had drawn into it.
    #[test]
    fn scanner_starts_over_for_a_render_target_made_again_under_a_released_key() {
        let mut bytes = [3u8];
        let entry = |serial| (bytes.as_ptr() as usize, bytes.len(), true, serial);
        let (old, new) = (entry(0), entry(1));
        let w = watched([(1, old)]);
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w)[&1], ContentHash::of(&bytes));

        // Released and made again between two rounds, its memory holding something else by then.
        watch_in(&w, |w| {
            w.released(1);
            w.entries.insert(1, new);
        });
        bytes[0] = 6;
        assert!(dirty_this_round(&w).is_empty(), "the new texture's first sight is not a change");
        assert_eq!(seen(&w)[&1], ContentHash::of(&bytes));
        bytes[0] = 7;
        assert_eq!(dirty_this_round(&w), [1].into(), "a write into it after that is");
    }

    /// Every mip, array slice and depth slice goes up and comes back where it was: 1D, 2D and 3D,
    /// element sizes 1 to 8, the shapes a world run names (#46).
    #[test]
    fn download_undoes_upload_for_every_mip_and_slice() {
        use victory_formats::texture::{Desc, Swizzle};
        let Ok(metal) = victory_metal::metal::Metal::new() else { return };
        use Dimension::{Tex1d, Tex2d, Tex3d};
        // (dimension, DXGI format, element bytes, width, height, depth or layers, mips)
        let shapes = [
            (Tex1d, 28, 4, 256, 1, 1, 1),
            (Tex2d, 61, 1, 128, 64, 1, 5),
            (Tex2d, 10, 8, 256, 256, 1, 9),
            (Tex2d, 41, 4, 64, 64, 4, 6),
            (Tex3d, 41, 4, 256, 256, 16, 1),
            (Tex3d, 17, 8, 64, 64, 64, 1),
            (Tex3d, 28, 4, 32, 32, 32, 6),
        ];
        let mut ran = 0;
        for (dimension, format, element_bytes, width, height, depth, mips) in shapes {
            for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S, Swizzle::SW_64KB_D_X] {
                let what = format!("{dimension:?} {format} {width}x{height}x{depth} mips {mips} {swizzle:?}");
                let desc = Desc { dimension, width, height, depth, mips, element_bytes, block: 1, samples: 1, swizzle };
                // Not every mode exists for every shape.
                let Ok(layout) = Layout::new(&desc) else { continue };
                let mut want = vec![0u8; layout.size as usize];
                let slices = if dimension == Tex3d { 1 } else { layout.slices };
                let tiled = (0..layout.mips.len() as u32).all(|mip| {
                    let m = &layout.mips[mip as usize];
                    let row = (m.width * element_bytes) as usize;
                    let pitch = row * m.height as usize;
                    (0..slices).all(|slice| {
                        let n = pitch * layout.subresource_depth(mip).max(1) as usize;
                        let linear: Vec<u8> = (0..n).map(|i| (i % 251) as u8 ^ (mip * 37 + slice * 11) as u8).collect();
                        layout.tile(&mut want, mip, slice, &linear, row, pitch).is_ok()
                    })
                });
                if !tiled {
                    continue;
                }
                let info = TextureInfo {
                    dimension,
                    format,
                    width,
                    height,
                    depth_or_layers: depth,
                    flags: UNORDERED_ACCESS,
                    size: want.len(),
                    layout_word: 0x100 | layout.desc.swizzle.0,
                    layout,
                    memory: std::ptr::null(),
                };
                let tex = metal.create_texture(&info.desc().unwrap()).unwrap();
                upload(&metal, &tex, &info, &want).unwrap();
                let mut got = vec![0u8; want.len()];
                download(&metal, &tex, &info, &mut got).unwrap();
                let at = got.iter().zip(&want).position(|(a, b)| a != b);
                assert!(at.is_none(), "{what}: differs first at byte {at:?}");
                eprintln!("round trip: {what}");
                ran += 1;
            }
        }
        // Linear at least, for each shape.
        assert!(ran >= shapes.len(), "{ran}");
    }

    #[test]
    fn from_host_undoes_to_host() {
        let title: Vec<u8> = (0..=255).collect();
        for format in [28, 115] {
            let mut texels = title.clone();
            to_host(format, &mut texels);
            from_host(format, &mut texels);
            assert_eq!(texels, title, "{format}");
        }
    }

    #[test]
    fn an_inaccessible_megabyte_takes_one_failed_read_and_one_region_query() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, protect};

        let len = 1 << 20;
        let memory = Region::new(len, None).unwrap();
        // SAFETY: the test owns this entire mapping until the reader finishes.
        unsafe { protect(memory.ptr, len, Protect::NO_ACCESS) }.unwrap();
        let mut out = vec![0xff; len];
        let (mut reads, mut queries) = (0, 0);
        read_texture_chunk(
            memory.ptr,
            &mut out,
            false,
            |at, out| {
                reads += 1;
                crate::native::safe_read(at, out)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(out.iter().all(|&b| b == 0));
        assert_eq!((reads, queries), (1, 1));
    }

    #[test]
    fn a_region_that_changes_after_the_query_still_preserves_readable_pages() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, page_size, protect};

        let page = page_size();
        let memory = Region::new(4 * page, None).unwrap();
        // SAFETY: the test owns all four pages and writes them before changing protection.
        unsafe {
            memory.ptr.write_bytes(0x72, 4 * page);
            protect(memory.ptr, page, Protect::NO_ACCESS).unwrap();
        }
        let mut out = vec![0xff; 4 * page - 2];
        read_texture_chunk(
            memory.ptr.wrapping_add(1),
            &mut out,
            false,
            crate::native::safe_read,
            |at| {
                let region = next_region(at);
                if at == memory.ptr.addr() + page {
                    // SAFETY: the third page is ours. Revoke access just after querying the span.
                    unsafe { protect(memory.ptr.add(2 * page), page, Protect::NO_ACCESS) }.unwrap();
                }
                region
            },
        );
        for (offset, &byte) in out.iter().enumerate() {
            let readable = matches!((offset + 1) / page, 1 | 3);
            assert_eq!(byte, if readable { 0x72 } else { 0 });
        }
    }

    #[test]
    fn a_texture_read_preserves_texels_on_both_sides_of_an_unreadable_page() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, page_size, protect};

        let page = page_size();
        let memory = Region::new(3 * page, None).unwrap();
        // SAFETY: all three pages belong to this test. Protect only the middle one after filling it.
        unsafe {
            memory.ptr.write_bytes(0x31, 2 * page);
            memory.ptr.add(2 * page).write_bytes(0x72, page);
            protect(memory.ptr.add(page), page, Protect::NO_ACCESS).unwrap();
        }
        let start = memory.ptr.wrapping_add(page - 7);
        let mut expected = vec![0; 7 + page + 11];
        expected[..7].fill(0x31);
        expected[7 + page..].fill(0x72);

        let bytes = read_memory(start, expected.len());
        assert_eq!(&bytes[..7], &expected[..7]);
        assert!(bytes[7..7 + page].iter().all(|&b| b == 0));
        assert_eq!(&bytes[7 + page..], &expected[7 + page..]);
        let mut scratch = vec![0xff; expected.len()];
        assert_eq!(
            probe_hash(None, start as usize, expected.len(), None, true, false, &mut scratch),
            ContentHash::of(&expected)
        );
    }

    #[test]
    fn probe_hash_detects_writes_in_sampled_and_whole_modes() {
        let mut mem = vec![0u8; 16 * 4096];
        let mut scratch = Vec::new();
        let h0_sampled = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, false, false, &mut scratch);
        let h0_whole = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, true, false, &mut scratch);
        // Flip a byte in the first sampled window.
        mem[42] = 0xAB;
        let h1_sampled = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, false, false, &mut scratch);
        let h1_whole = probe_hash(None, mem.as_ptr() as usize, mem.len(), None, true, false, &mut scratch);
        assert_ne!(h0_sampled, h1_sampled);
        assert_ne!(h0_whole, h1_whole);
        // Flip a byte outside sampled windows: whole mode catches it.
        mem[42] = 0;
        mem[1000] = 0xCD;
        assert_ne!(
            h0_whole,
            probe_hash(None, mem.as_ptr() as usize, mem.len(), None, true, false, &mut scratch)
        );
    }

    #[test]
    fn mip_aware_probes_catch_tail_and_coarse_mips_in_reverse_mip_layout() {
        use victory_formats::texture::{Desc, PipeConfig, Swizzle};

        // 1024x1024 BC7 (16 B per 4x4 block), 11 mips, 64 KiB XOR-swizzled with non-zero pipe-bank XOR:
        // mips 4..10 sit in the 64 KiB mip tail (past byte 0x400, leaving 0..256 permanently zero),
        // mips 3..1 sit in [64 KiB .. 0.25 * size), and mip 0 occupies the top 75% of the buffer.
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 1024,
            height: 1024,
            depth: 1,
            mips: 11,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        };
        let layout = Layout::new(&desc)
            .unwrap()
            .with_pipe_bank_xor(0x1b, PipeConfig::DEFAULT);
        let mut tiled = vec![0u8; layout.size as usize];
        let mut scratch = Vec::new();
        assert!(!probe_nonzero(tiled.as_ptr() as usize, tiled.len(), &layout, &mut scratch));

        // Tile only the smallest 1x1 mip (`mip = 10`) into the 64 KiB tail while mip 0 stays zero:
        // `probe_nonzero` must catch it on the next submission despite `(size / 2) & !4095` and
        // `0..256` both being zero.
        layout.tile(&mut tiled, 10, 0, &[0x5au8; 16], 16, 16).unwrap();
        assert_eq!(&tiled[..256], &[0u8; 256], "tail padding at 0..256 stays zero");
        assert!(
            probe_nonzero(tiled.as_ptr() as usize, tiled.len(), &layout, &mut scratch),
            "blank probe must detect writes into the coarsest tail mip while mip 0 is zero"
        );

        // Now step from smallest mip to mip 0 and confirm every single mip write changes the
        // sampled `probe_hash` (keeping `sampled_sees_writes == true`).
        tiled.fill(0);
        let mut prev = probe_hash(
            None,
            tiled.as_ptr() as usize,
            tiled.len(),
            Some(&layout),
            false,
            true,
            &mut scratch,
        );
        for mip in (0..desc.mips).rev() {
            let m = &layout.mips[mip as usize];
            let row = m.width as usize * 16;
            let linear = vec![0x30u8.wrapping_add(mip as u8 * 13); row * m.height as usize];
            layout.tile(&mut tiled, mip, 0, &linear, row, linear.len()).unwrap();
            let next = probe_hash(
                None,
                tiled.as_ptr() as usize,
                tiled.len(),
                Some(&layout),
                false,
                true,
                &mut scratch,
            );
            assert_ne!(next, prev, "sampled probe missed streaming write of mip {mip}");
            prev = next;
        }
    }

    #[test]
    fn read_texture_chunk_coalesces_adjacent_readable_regions() {
        use victory_platform::mem::{Protect, Region};

        let chunk = 64 << 10;
        let backing = vec![0x4du8; 8 * chunk];
        let base = backing.as_ptr().addr();
        let mut out = vec![0xffu8; 8 * chunk];
        let mut reads = 0usize;
        // Simulate 4 separate 64 KiB `mach_vm_map` regions (all readable) followed by one 256 KiB
        // `NO_ACCESS` reservation tail.
        let whole = read_texture_chunk(
            backing.as_ptr(),
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                let off = at.addr() - base;
                buf.copy_from_slice(&backing[off..off + buf.len()]);
                true
            },
            |cur| {
                let idx = (cur - base) / chunk;
                if idx < 4 {
                    Some(Region {
                        base: base + idx * chunk,
                        len: chunk,
                        protect: Protect::READ_WRITE,
                    })
                } else if idx < 8 {
                    Some(Region {
                        base: base + 4 * chunk,
                        len: 4 * chunk,
                        protect: Protect::NO_ACCESS,
                    })
                } else {
                    None
                }
            },
        );
        assert!(!whole);
        assert_eq!(reads, 1, "4 adjacent readable 64 KiB regions must coalesce into 1 read");
        assert!(out[..4 * chunk].iter().all(|&b| b == 0x4d));
        assert!(out[4 * chunk..].iter().all(|&b| b == 0));
    }

    #[test]
    fn a_known_partial_chunk_skips_the_whole_read_until_every_page_is_readable() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, page_size, protect};

        let page = page_size();
        let memory = Region::new(4 * page, None).unwrap();
        // SAFETY: the test owns all four pages; pages 0..2 hold coarse-mip data and 2..4 are unmapped/no-access.
        unsafe {
            memory.ptr.write_bytes(0x5a, 2 * page);
            protect(memory.ptr.add(2 * page), 2 * page, Protect::NO_ACCESS).unwrap();
        }

        let mut out = vec![0xff; 4 * page];
        let (mut reads, mut queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            false,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(!whole);
        assert_eq!((reads, queries), (2, 2));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0));

        // Once marked partial, the doomed whole-span read is skipped and only readable spans are read.
        out.fill(0xff);
        (reads, queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(!whole);
        assert_eq!((reads, queries), (1, 2));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0));

        // SAFETY: restore access to pages 2..4 (fine mips mapped) and fill them.
        unsafe {
            protect(memory.ptr.add(2 * page), 2 * page, Protect::READ_WRITE).unwrap();
            memory.ptr.add(2 * page).write_bytes(0xa5, 2 * page);
        }
        out.fill(0xff);
        (reads, queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(whole, "once every page is readable again, the partial flag clears");
        assert_eq!((reads, queries), (1, 1));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0xa5));
    }

    #[test]
    fn probe_round_wraparound_removals_visit_each_live_entry_once() {
        use crate::native::vertex_tests::Capture;
        use victory_formats::texture::{Desc, Swizzle};

        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 4,
            height: 4,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .unwrap();
        let mem = [0x42u8; 256];
        let pools = PoolTable::default();
        let mut scratch = Vec::new();
        let probe = probe_hash(
            Some(&pools),
            mem.as_ptr() as usize,
            mem.len(),
            Some(&layout),
            false,
            false,
            &mut scratch,
        );
        let sync_serial = 10u64;

        let make_tracked = |hot_until: u64| Tracked::<Capture> {
            texture: Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 4, 4)),
            used: 0,
            info: TextureInfo {
                dimension: Dimension::Tex2d,
                format: 28,
                width: 4,
                height: 4,
                depth_or_layers: 1,
                flags: 0,
                layout: layout.clone(),
                layout_word: 0x100 | layout.desc.swizzle.0,
                memory: mem.as_ptr(),
                size: mem.len(),
            },
            zero_copy: false,
            streamed: false,
            write_serial: 0,
            restreamed: false,
            last_used_at: std::time::Instant::now(),
            serial: 1,
            last: Some(ContentHash::of(&mem)),
            uploads: 2,
            uploaded_mips: u32::MAX,
            hot_until,
            probe: Some(probe),
            sampled_sees_writes: true,
            unchanged_whole: 1,
            skipped: 0,
            partial: false,
            blank: false,
            target_views: Vec::new(),
            gpu_written: false,
            last_frame: 0,
            last_named: 0,
            reach: 0,
            dropped: 0,
            at: 0,
        };

        // 1. Explicit trace removing entries after wraparound (`cursor = 4` on 6 entries `[0..6]`,
        //    so `[4, 5]` are visited before wrap and `[0, 2]` are removed after wrap while `[1, 3]`
        //    survive).
        let mut textures: Textures<Capture> = [
            (0, make_tracked(5)),  // expired before probe (removed after wrap)
            (1, make_tracked(20)), // survives after wrap
            (2, make_tracked(10)), // probed once, then cools (`hot_until == sync_serial`) after wrap
            (3, make_tracked(20)), // survives after wrap
            (4, make_tracked(20)), // visited and kept before wrap
            (5, make_tracked(20)), // visited and kept before wrap
        ]
        .into_iter()
        .collect();
        let mut keys = vec![0, 1, 2, 3, 4, 5];
        let mut cursor = 4usize;
        let mut dirty = HashSet::new();
        probe_round(
            &pools,
            &mut textures,
            &mut keys,
            &mut cursor,
            None,
            Some(sync_serial),
            &mut scratch,
            &mut dirty,
        );
        assert!(dirty.is_empty());
        assert_eq!(textures[&0].skipped, 0, "expired entry evicted without probing");
        for k in [1, 2, 3, 4, 5] {
            assert_eq!(textures[&k].skipped, 1, "live entry {k} must be visited once, not skipped or visited twice");
        }
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 3, 4, 5]);

        // 2. Exhaustive check across all 6 initial cursor positions and all 3^6 = 729 combinations
        //    of (already expired = 5, cools after probe = 10, stays hot = 20): 4,374 schedules.
        const N: usize = 6;
        for total_states in 0..3usize.pow(N as u32) {
            let mut state = [0u64; N];
            let mut rem = total_states;
            for s in &mut state {
                *s = match rem % 3 {
                    0 => 5,
                    1 => 10,
                    _ => 20,
                };
                rem /= 3;
            }
            for start_cursor in 0..N {
                let mut textures: Textures<Capture> = (0..N)
                    .map(|k| (k, make_tracked(state[k])))
                    .collect();
                let mut keys: Vec<usize> = (0..N).collect();
                let mut cursor = start_cursor;
                let mut dirty = HashSet::new();
                probe_round(
                    &pools,
                    &mut textures,
                    &mut keys,
                    &mut cursor,
                    None,
                    Some(sync_serial),
                    &mut scratch,
                    &mut dirty,
                );
                let mut want_survivors = Vec::new();
                for k in 0..N {
                    let expected_visits = u32::from(state[k] >= sync_serial);
                    assert_eq!(
                        textures[&k].skipped, expected_visits,
                        "state={state:?} start={start_cursor} key={k}"
                    );
                    if state[k] > sync_serial {
                        want_survivors.push(k);
                    } else {
                        assert_eq!(textures[&k].hot_until, 0);
                    }
                }
                keys.sort_unstable();
                assert_eq!(keys, want_survivors, "state={state:?} start={start_cursor}");
            }
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_hot_texture_serial_cooling_under_probe_budget() {
        use crate::native::vertex_tests::Capture;
        use std::time::{Duration, Instant};
        use victory_formats::texture::{Desc, Swizzle};

        const N: usize = 5_000;
        const SUBMISSIONS: u64 = 800;
        const SIZE: usize = 64 << 10;
        let backing = vec![0x5au8; N * SIZE];
        let pools = PoolTable::default();
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 128,
            height: 128,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .unwrap();
        let mut scratch = Vec::new();
        let mut textures: Textures<Capture> = (0..N)
            .map(|i| {
                let ptr = unsafe { backing.as_ptr().add(i * SIZE) };
                let probe = probe_hash(
                    Some(&pools),
                    ptr as usize,
                    SIZE,
                    Some(&layout),
                    false,
                    false,
                    &mut scratch,
                );
                (
                    i,
                    Tracked {
                        texture: Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 128, 128)),
                        used: 0,
                        info: TextureInfo {
                            dimension: Dimension::Tex2d,
                            format: 28,
                            width: 128,
                            height: 128,
                            depth_or_layers: 1,
                            flags: 0,
                            layout: layout.clone(),
                            layout_word: 0x100 | layout.desc.swizzle.0,
                            memory: ptr,
                            size: SIZE,
                        },
                        serial: 1,
                        last: None,
                        uploads: 1,
                        uploaded_mips: u32::MAX,
                        hot_until: u64::from(HOT),
                        probe: Some(probe),
                        sampled_sees_writes: true,
                        unchanged_whole: 1,
                        skipped: 0,
                        partial: false,
                        blank: false,
                        target_views: Vec::new(),
                        gpu_written: false,
                        zero_copy: false,
                        streamed: false,
                        write_serial: 0,
                        restreamed: false,
                        last_used_at: std::time::Instant::now(),
                        last_frame: 0,
                        last_named: 0,
                        reach: 0,
                        dropped: 0,
                        at: 0,
                    },
                )
            })
            .collect();
        let mut keys: Vec<usize> = (0..N).collect();
        let mut cursor = 0usize;
        let mut dirty = HashSet::new();
        let mut early_ms = 0.0;
        let mut post_cool_ms = 0.0;
        for s in 1..=SUBMISSIONS {
            let t0 = Instant::now();
            probe_round(
                &pools,
                &mut textures,
                &mut keys,
                &mut cursor,
                Some(t0 + PROBE_BUDGET),
                Some(s),
                &mut scratch,
                &mut dirty,
            );
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if s <= u64::from(HOT) {
                early_ms += ms;
            } else {
                post_cool_ms += ms;
            }
            let _ = Duration::ZERO;
        }
        eprintln!(
            "BENCH #252 ({N} hot 64 KiB textures, {SUBMISSIONS} submissions, 2 ms budget):\n\
             - submissions 1..={HOT}: avg {:.3} ms/sub (budget-bound at 2.0 ms)\n\
             - submission {}: hot_keys remaining = {} (vs {N} under per-visit decrement until ~{} submissions)\n\
             - submissions {}..={SUBMISSIONS}: avg {:.4} ms/sub",
            early_ms / f64::from(HOT),
            SUBMISSIONS,
            keys.len(),
            N * HOT as usize / 180,
            HOT + 1,
            post_cool_ms / (SUBMISSIONS - u64::from(HOT)) as f64,
        );
        assert_eq!(keys.len(), 0);
    }
}

#[cfg(test)]
mod drop_top_mip_tests {
    use super::{dropped_levels, host_mip_range};

    #[test]
    fn only_large_sampled_2d_textures_with_a_lower_level_lose_their_top() {
        assert_eq!(dropped_levels(true, false, false, 11, 2048), 1);
        assert_eq!(dropped_levels(true, false, false, 2, 1024), 1);
        assert_eq!(dropped_levels(false, false, false, 11, 2048), 0, "off unless asked for");
        assert_eq!(dropped_levels(true, true, false, 11, 2048), 0, "render targets and UAVs keep it");
        assert_eq!(dropped_levels(true, false, true, 11, 2048), 0, "3D textures keep it");
        assert_eq!(dropped_levels(true, false, false, 1, 4096), 0, "no lower level to fall back on");
        assert_eq!(dropped_levels(true, false, false, 10, 1023), 0, "small textures keep it");
    }

    #[test]
    fn views_map_title_levels_onto_the_host_texture() {
        // Nothing dropped: unchanged.
        assert_eq!(host_mip_range(2, 3, 0, 0), (2, 3));
        // Whole chain (11 title levels, 10 on the host).
        assert_eq!(host_mip_range(0, 11, 1, 10), (0, 10));
        // A view of the dropped top level alone sees the host's top.
        assert_eq!(host_mip_range(0, 1, 1, 10), (0, 1));
        // Lower levels keep their content, one index down.
        assert_eq!(host_mip_range(3, 2, 1, 10), (2, 2));
        // A count past the end is clamped to the host's levels.
        assert_eq!(host_mip_range(5, 99, 1, 10), (4, 6));
    }
}
