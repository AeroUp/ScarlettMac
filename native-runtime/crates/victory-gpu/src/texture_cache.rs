//! Decoded textures kept on disk: the result of detiling a console-layout texture, keyed by the
//! bytes it was detiled from and the layout it was detiled for, so a later run (or the same
//! texture streamed in again) skips the detile and its `to_host` conversion entirely.
//!
//! **Off unless `VICTORY_TEXTURE_CACHE=1`.** Under Rosetta a lookup that hits costs more than the
//! detile it saves (a 2048x2048 RGBA8 texture with its mips, 21 MiB: detiled in 2.4 to 5.8 ms, read
//! back from an entry in about 67 ms, stored in about 90 ms, mostly inflating, deflating and the
//! entry's blake3; BC1 and BC7: 0.3 and 0.6 ms against 2 and 3 ms for a hit), and it needs every
//! texture's bytes hashed whole for its key.
//!
//! This is the texture side of the shader cache's design (`shader_cache`): content-addressed, one
//! file per entry, written to a temporary name and renamed so a reader sees all of it or none, and
//! each entry starts with the blake3 sum of the rest: an entry that is damaged or cut short is a
//! miss and is detiled again. The key holds the format, the layout (`Desc` and pipe configuration)
//! and the blake3 of the tiled bytes read from title memory — not the levels a decode happened to
//! make — so identical bytes in the same layout are one entry wherever they came from (a file, a
//! pool, another run, another texture) and a texture streamed in mip by mip adds to that entry
//! instead of writing another copy. Entries carry the levels they hold, and a lookup takes the
//! ones the caller still needs.
//!
//! **Size.** A world streams far more textures than fit on a disk, so the folder is capped
//! (`VICTORY_TEXTURE_CACHE_MB`, default 10 GiB) and pruned least-recently-used: at startup, and
//! again while it grows past the cap, the oldest files (by modification time, which a read
//! refreshes) are deleted until it is back under. The cap is a cost, not a promise: a texture
//! whose entry was pruned is detiled again.
//!
//! **Compression.** A level that is block-compressed (BC1–BC7, BCPack's output) is already
//! compressed and is stored as it is; any other level is deflated when that makes it smaller.
//! Each level says which, so a reader takes both.
//!
//! **Mapping.** A reader maps the entry (`mmap`) instead of reading it: a level stored as it is
//! is uploaded straight out of the file's clean, file-backed pages (no copy in the heap, and
//! nothing for the OS to swap), and only a deflated one is inflated into memory.
//!
//! The folder is the user's cache folder (`use_dir`, from `native_run`), never the repo.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use victory_formats::texture::{Dimension, Layout};

use crate::mmap::Map;
use crate::native::textures::{Level, Texels};

/// The entry format, and what a decode depends on: raise it when `Layout::detile` or `to_host`
/// changes the bytes a decode makes, or when the entry's layout changes (a bump invalidates every
/// entry).
const DECODE_VERSION: u32 = 2;
/// Names the file format: the start of every entry.
const MAGIC: &[u8; 4] = b"VTX2";
/// Bytes of the cache folder when `VICTORY_TEXTURE_CACHE_MB` is not set: 10 GiB.
const DEFAULT_LIMIT_MIB: u64 = 10 * 1024;
/// The cap a prune aims at, as a share of the limit: pruning to the limit exactly would prune on
/// almost every store.
const PRUNE_TO: u64 = 9;

static CACHE: OnceLock<Option<Arc<Cache>>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static PRUNED: AtomicU64 = AtomicU64::new(0);

/// A folder of decoded textures.
pub struct Cache {
    root: PathBuf,
    /// Bytes of the entries on disk (recomputed exactly by each prune).
    used: AtomicU64,
    /// The cap in bytes (0: no cap).
    limit: u64,
    /// One prune at a time.
    pruning: AtomicBool,
}

/// Whether the cache is wanted: `VICTORY_TEXTURE_CACHE` (`on`) set to anything but empty or `0`,
/// and `VICTORY_NO_TEXTURE_CACHE` (`off`) not (it still turns the cache off).
pub fn wanted(on: Option<&str>, off: Option<&str>) -> bool {
    let set = |v: Option<&str>| v.is_some_and(|v| !v.trim().is_empty() && v.trim() != "0");
    set(on) && !set(off)
}

/// Keeps decoded textures in `dir` from now on (creating it) when [`wanted`]. `None` when it is
/// not, or when the folder cannot be made. Returns the folder.
pub fn use_dir(dir: PathBuf) -> Option<PathBuf> {
    let var = |name| std::env::var(name).ok();
    if !wanted(var("VICTORY_TEXTURE_CACHE").as_deref(), var("VICTORY_NO_TEXTURE_CACHE").as_deref()) {
        tracing::info!("decoded texture cache off (VICTORY_TEXTURE_CACHE=1 turns it on)");
        let _ = CACHE.set(None);
        return None;
    }
    open_dir(dir)
}

/// [`use_dir`] whatever the environment says.
pub fn open_dir(dir: PathBuf) -> Option<PathBuf> {
    let limit = limit_mib(std::env::var("VICTORY_TEXTURE_CACHE_MB").ok().as_deref()) << 20;
    match Cache::open(dir, limit) {
        Ok(cache) => {
            tracing::info!(dir = %cache.root.display(), limit_mib = cache.limit >> 20, used_mib = cache.used.load(Ordering::Relaxed) >> 20, "decoded texture cache");
            let cache = Arc::new(cache);
            let root = cache.root.clone();
            let _ = CACHE.set(Some(cache.clone()));
            // A full folder (a previous run's uncapped one, say) is pruned off the start-up path:
            // deleting tens of thousands of files there would hold the title's first frame.
            if cache.limit > 0 {
                let spawned = std::thread::Builder::new()
                    .name("decoded-texture-cache-prune".into())
                    .spawn(move || cache.prune());
                if spawned.is_err() {
                    // No thread: prune inline rather than start over the cap.
                    if let Some(cache) = CACHE.get().and_then(Option::as_ref) {
                        cache.prune();
                    }
                }
            }
            Some(root)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no decoded texture cache");
            let _ = CACHE.set(None);
            None
        }
    }
}

/// Decoded textures found on disk and made since the start, and bytes pruned.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Bytes deleted by pruning since the start.
pub fn pruned() -> u64 {
    PRUNED.load(Ordering::Relaxed)
}

/// The cap in MiB: `VICTORY_TEXTURE_CACHE_MB`, else [`DEFAULT_LIMIT_MIB`]; `0` (or a value that is
/// not a number) turns the cap off.
pub fn limit_mib(var: Option<&str>) -> u64 {
    match var.map(str::trim) {
        None | Some("") => DEFAULT_LIMIT_MIB,
        Some(v) => v.parse().unwrap_or(0),
    }
}

impl Cache {
    /// Opens the folder at `root`, measuring what it holds. `limit`: bytes allowed (0: none).
    pub fn open(root: impl Into<PathBuf>, limit: u64) -> std::io::Result<Cache> {
        let root = root.into();
        if !outside_repo(&root) {
            return Err(std::io::Error::other(format!("{} is inside the repository", root.display())));
        }
        std::fs::create_dir_all(&root)?;
        let used = walk(&root).iter().map(|(_, size, _)| *size).sum();
        Ok(Cache { root, used: AtomicU64::new(used), limit, pruning: AtomicBool::new(false) })
    }

    /// The decoded levels of a texture with `tiled`'s bytes in `layout`: the ones the caller still
    /// needs (`uploaded_mips` says which title mips the host texture already has, `skip` the top
    /// levels it was made without), or `None` when the entry is missing, damaged, or does not hold
    /// every needed level. `hash` is blake3 of the whole tiled buffer.
    pub(crate) fn levels(
        &self,
        layout: &Layout,
        dimension: Dimension,
        format: u32,
        hash: &blake3::Hash,
        uploaded_mips: u32,
        skip: usize,
    ) -> Option<Vec<Level>> {
        let path = self.path_for(&key(layout, dimension, format, hash));
        let found = Map::open(&path).ok().and_then(|map| unseal_decode(&map)).and_then(|stored| {
            let needed: Vec<usize> = needed_levels(layout, uploaded_mips, skip);
            let have: Vec<&Level> = stored
                .values()
                .filter(|l| needed.contains(&l.mip))
                .collect();
            (have.len() == needed.len()).then(|| have.into_iter().cloned().collect())
        });
        if found.is_some() {
            // Least recently *used*: a read refreshes the file's time (`prune` deletes by it).
            if let Ok(file) = std::fs::File::open(&path) {
                let _ = file.set_modified(SystemTime::now());
            }
        }
        (if found.is_some() { &HITS } else { &MISSES }).fetch_add(1, Ordering::Relaxed);
        found
    }

    /// Keeps `made` under the key of their source (see [`Self::levels`]), adding to the levels an
    /// entry already holds: a texture streamed in mip by mip writes one entry, not one per mip.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store(
        &self,
        layout: &Layout,
        dimension: Dimension,
        format: u32,
        hash: &blake3::Hash,
        levels: &[Level],
    ) {
        if levels.is_empty() {
            return;
        }
        let path = self.path_for(&key(layout, dimension, format, hash));
        let dir = path.parent().expect("keyed paths have a parent");
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::debug!(error = %e, "decoded texture not kept");
            return;
        }
        // What the entry holds already (a partial decode of the same bytes), plus what is new.
        let mut stored: BTreeMap<(usize, usize), Level> =
            Map::open(&path).ok().and_then(|map| unseal_decode(&map)).unwrap_or_default();
        let before = stored.len();
        for level in levels {
            stored.insert((level.mip, level.slice), level.clone());
        }
        let block_compressed = is_block_compressed(format);
        let body = encode(&stored, block_compressed);
        let bytes = [MAGIC.as_slice(), blake3::hash(&body).as_bytes(), body.as_slice()].concat();
        let tmp = dir.join(format!(".tmp-{}-{}", std::process::id(), &key(layout, dimension, format, hash)[..16]));
        let written = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, &path));
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        if stored.len() != before {
            self.used.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        }
        self.prune_if_over();
    }

    /// Prunes now when the folder is past its cap, on a thread of its own (the caller is a
    /// texture-reading thread that the frame is waiting for).
    fn prune_if_over(&self) {
        if self.limit == 0 || self.used.load(Ordering::Relaxed) <= self.limit {
            return;
        }
        if self.pruning.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(cache) = CACHE.get().and_then(Option::as_ref).cloned() else {
            self.pruning.store(false, Ordering::Release);
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("decoded-texture-cache-prune".into())
            .spawn(move || cache.prune_claimed());
        if spawned.is_err() {
            self.pruning.store(false, Ordering::Release);
        }
    }

    /// Deletes the least recently used entries until the folder is back under the cap (and a
    /// little below it, so the next stores do not prune again at once). One prune at a time.
    pub fn prune(&self) {
        if self.limit == 0 || self.pruning.swap(true, Ordering::AcqRel) {
            return;
        }
        self.prune_claimed();
    }

    /// [`Self::prune`] for a caller that already holds the prune flag (the store that went over,
    /// the start-up thread).
    fn prune_claimed(&self) {
        let limit = self.limit;
        if limit == 0 {
            self.pruning.store(false, Ordering::Release);
            return;
        }
        let target = limit / 10 * PRUNE_TO;
        let mut files = walk(&self.root);
        let total: u64 = files.iter().map(|(_, size, _)| *size).sum();
        if total <= target {
            self.used.store(total, Ordering::Relaxed);
            self.pruning.store(false, Ordering::Release);
            return;
        }
        // Oldest first: the modification time is a read's (or a store's) time.
        files.sort_by_key(|(_, _, at)| *at);
        let mut left = total;
        let mut deleted = 0u64;
        let mut freed = 0u64;
        for (path, size, _) in files {
            if left <= target {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                left -= size;
                freed += size;
                deleted += 1;
            }
        }
        self.used.store(left, Ordering::Relaxed);
        PRUNED.fetch_add(freed, Ordering::Relaxed);
        tracing::info!(
            before_mib = total >> 20,
            after_mib = left >> 20,
            limit_mib = limit >> 20,
            deleted,
            freed_mib = freed >> 20,
            "decoded texture cache pruned"
        );
        self.pruning.store(false, Ordering::Release);
    }

    /// Where entry `key` lives: two hex digits of folder, the rest the file's name.
    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(&key[..2]).join(&key[2..])
    }
}

/// `(path, bytes, modified)` of every entry file under `root` (temporary files are not entries).
fn walk(root: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    let mut out = Vec::new();
    let Ok(dirs) = std::fs::read_dir(root) else { return out };
    for dir in dirs.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let Ok(files) = std::fs::read_dir(&dir) else { continue };
        for file in files.flatten() {
            let path = file.path();
            let name = file.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            let Ok(meta) = file.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            let at = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            out.push((path, meta.len(), at));
        }
    }
    out
}

/// The host levels a decode must make for a request: title mip `t` (from `skip` up) whose bit
/// `uploaded_mips` does not hold, as host mip `t - skip`.
fn needed_levels(layout: &Layout, uploaded_mips: u32, skip: usize) -> Vec<usize> {
    (skip..layout.mips.len())
        .filter(|t| *t >= 32 || uploaded_mips & (1 << t) == 0)
        .map(|t| t - skip)
        .collect()
}

/// Whether a console format is block-compressed (BC1–BC7: already compressed in memory).
fn is_block_compressed(format: u32) -> bool {
    crate::resource::element(format).is_some_and(|(_, block)| block == 4)
}

/// The entry's key: what a decode depends on. `hash` is the tiled bytes' blake3 (the whole
/// buffer, as the sync takes it), so any change to the source is another entry; the levels made
/// are not part of it (an entry takes more of them as they are made).
fn key(layout: &Layout, dimension: Dimension, format: u32, hash: &blake3::Hash) -> String {
    let mut h = blake3::Hasher::new();
    h.update(&DECODE_VERSION.to_le_bytes());
    h.update(&format.to_le_bytes());
    h.update(format!("{dimension:?}").as_bytes());
    // The layout as `Layout` builds it: its description and the pipe configuration decide every
    // block offset, `first_tail_mip` and `slice_size` follow from it.
    h.update(format!("{:?}|{:?}|{}|{}|{}", layout.desc, layout.config, layout.slices, layout.size, layout.first_tail_mip).as_bytes());
    h.update(hash.as_bytes());
    h.finalize().to_hex().to_string()
}

/// The entry body: the level count, then each level with its mip, slice, sizes and bytes.
///
/// A level is deflated (zlib, level 1: fast) unless it is block-compressed already or deflating
/// does not make it smaller; the flag byte says which. `block_compressed`: the whole texture's
/// format is (BC1–BC7), so no level is worth compressing.
fn encode(levels: &BTreeMap<(usize, usize), Level>, block_compressed: bool) -> Vec<u8> {
    let mut out = (levels.len() as u32).to_le_bytes().to_vec();
    for level in levels.values() {
        let texels = level.texels.as_slice();
        let compressed = (!block_compressed)
            .then(|| miniz_oxide::deflate::compress_to_vec_zlib(texels, 1))
            .filter(|z| z.len() + 1 < texels.len());
        let (flag, bytes) = match &compressed {
            Some(z) => (1u8, z.as_slice()),
            None => (0u8, texels),
        };
        for v in [level.mip as u32, level.slice as u32, level.row as u32, level.slice_pitch as u32, bytes.len() as u32] {
            out.extend(v.to_le_bytes());
        }
        out.push(flag);
        out.extend(bytes);
    }
    out
}

/// An entry's levels, out of its mapping: a level stored as it is stays a window of the file (its
/// upload reads the page cache), a deflated one is inflated.
#[allow(clippy::type_complexity)]
fn unseal_decode(map: &Arc<Map>) -> Option<BTreeMap<(usize, usize), Level>> {
    let bytes = map.bytes();
    let rest = bytes.strip_prefix(MAGIC.as_slice())?;
    let (sum, body) = rest.split_at_checked(32)?;
    (blake3::hash(body).as_bytes() == sum).then(|| decode(map, body)).flatten()
}

fn decode(map: &Arc<Map>, mut body: &[u8]) -> Option<BTreeMap<(usize, usize), Level>> {
    fn word(body: &mut &[u8]) -> Option<u32> {
        let (head, rest) = body.split_at_checked(4)?;
        *body = rest;
        Some(u32::from_le_bytes(head.try_into().ok()?))
    }
    let n = word(&mut body)? as usize;
    let mut levels = BTreeMap::new();
    for _ in 0..n {
        let mip = word(&mut body)? as usize;
        let slice = word(&mut body)? as usize;
        let row = word(&mut body)? as usize;
        let slice_pitch = word(&mut body)? as usize;
        let len = word(&mut body)? as usize;
        let (flag, rest) = body.split_first()?;
        body = rest;
        let (bytes, rest) = body.split_at_checked(len)?;
        body = rest;
        let texels = match flag {
            0 => {
                let at = map.offset_of(bytes);
                Texels::Mapped(map.clone(), at..at + len)
            }
            1 => Texels::Owned(miniz_oxide::inflate::decompress_to_vec_zlib(bytes).ok()?),
            _ => return None,
        };
        levels.insert((mip, slice), Level { mip, slice, texels, row, slice_pitch });
    }
    body.is_empty().then_some(levels)
}

/// The cache of this process, if one was opened ([`use_dir`]).
fn open() -> Option<&'static Arc<Cache>> {
    CACHE.get().and_then(Option::as_ref)
}

/// [`Cache::levels`] for this process's cache, `None` without one.
pub(crate) fn levels(
    layout: &Layout,
    dimension: Dimension,
    format: u32,
    hash: &blake3::Hash,
    uploaded_mips: u32,
    skip: usize,
) -> Option<Vec<Level>> {
    open()?.levels(layout, dimension, format, hash, uploaded_mips, skip)
}

/// [`Cache::store`] for this process's cache, nothing without one.
pub(crate) fn store(
    layout: &Layout,
    dimension: Dimension,
    format: u32,
    hash: &blake3::Hash,
    levels: &[Level],
) {
    if let Some(cache) = open() {
        cache.store(layout, dimension, format, hash, levels);
    }
}

/// Whether an entry for exactly what the sync would key exists (tests: that the detile was kept).
#[cfg(test)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn contains(layout: &Layout, dimension: Dimension, format: u32, hash: &blake3::Hash) -> bool {
    open().is_some_and(|c| c.path_for(&key(layout, dimension, format, hash)).is_file())
}

/// Whether `path` is outside the repository (the cache is never kept in it).
pub(crate) fn outside_repo(path: &Path) -> bool {
    let Ok(path) = path.canonicalize() else { return true };
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    !path.starts_with(repo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Swizzle};

    fn layout() -> Layout {
        Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 2,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_4KB_S,
        })
        .unwrap()
    }

    fn levels() -> Vec<Level> {
        vec![
            Level { mip: 0, slice: 0, texels: vec![7; 256].into(), row: 16, slice_pitch: 256 },
            Level { mip: 1, slice: 0, texels: vec![9; 64].into(), row: 8, slice_pitch: 64 },
        ]
    }

    /// A decode kept on disk comes back byte for byte, for the same source bytes and layout; a
    /// damaged entry is a miss (the texture is detiled again), and other bytes are other entries.
    #[test]
    fn a_decoded_texture_comes_back_from_the_disk() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path(), 0).unwrap();
        let (l, made) = (layout(), levels());
        let hash = blake3::hash(b"tiled bytes");
        assert!(cache.levels(&l, Dimension::Tex2d, 71, &hash, 0, 0).is_none());
        cache.store(&l, Dimension::Tex2d, 71, &hash, &made);
        let back = cache.levels(&l, Dimension::Tex2d, 71, &hash, 0, 0).expect("stored");
        assert_eq!(back.len(), 2);
        for (a, b) in back.iter().zip(&made) {
            assert_eq!((a.mip, a.slice, a.row, a.slice_pitch), (b.mip, b.slice, b.row, b.slice_pitch));
            assert_eq!(a.texels.as_slice(), b.texels.as_slice());
        }
        // A level already uploaded is not needed, and not asked of the entry.
        assert_eq!(cache.levels(&l, Dimension::Tex2d, 71, &hash, 1, 0).unwrap().len(), 1);
        // Another format, another layout, other bytes: other entries.
        assert!(cache.levels(&l, Dimension::Tex2d, 28, &hash, 0, 0).is_none());
        assert!(cache.levels(&l, Dimension::Tex2d, 71, &blake3::hash(b"other"), 0, 0).is_none());
        // Damaged (a byte changed): a miss.
        let path = walk(dir.path()).into_iter().map(|(p, ..)| p).next().expect("an entry");
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(cache.levels(&l, Dimension::Tex2d, 71, &hash, 0, 0).is_none());
    }

    /// The same bytes in the same layout are one entry, and a texture streamed one mip at a time
    /// adds to it: the second store writes the same file, holding both levels.
    #[test]
    fn the_same_bytes_are_one_entry_that_takes_more_levels() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path(), 0).unwrap();
        let (l, made) = (layout(), levels());
        let hash = blake3::hash(b"tiled bytes");
        cache.store(&l, Dimension::Tex2d, 71, &hash, &made[..1]);
        cache.store(&l, Dimension::Tex2d, 71, &hash, &made[1..]);
        assert_eq!(walk(dir.path()).len(), 1, "one file for one content and layout");
        assert_eq!(cache.levels(&l, Dimension::Tex2d, 71, &hash, 0, 0).unwrap().len(), 2);
    }

    /// A cache hit reads the entry's file rather than the heap where it can: a block-compressed
    /// level is stored as it is, so it is served as a window of the mapping (the upload reads the
    /// file's clean pages, no second copy of it in memory), while a level deflated to fit is
    /// inflated into the heap. Either way the bytes that come back are the ones stored.
    #[test]
    fn a_cache_hit_uploads_a_block_compressed_level_from_the_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path(), 0).unwrap();
        let l = layout();
        let hash = blake3::hash(b"tiled bytes");
        // Both levels of the layout, so a hit finds every level it needs.
        let big = |mip: usize, byte: u8| Level {
            mip,
            slice: 0,
            texels: vec![byte; 4096].into(),
            row: 64,
            slice_pitch: 4064,
        };
        let levels = |byte: u8| vec![big(0, byte + 1), big(1, byte)];
        // 71 is BC1: block-compressed, kept as stored.
        cache.store(&l, Dimension::Tex2d, 71, &hash, &levels(0x5a));
        let back = cache.levels(&l, Dimension::Tex2d, 71, &hash, 0, 0).expect("stored");
        assert!(back.iter().all(|level| level.texels.is_mapped()), "a BC level comes from the mapping");
        assert_eq!(back[0].texels.as_slice(), vec![0x5b; 4096].as_slice());
        // 28 is RGBA8: deflated, so a hit inflates it into the heap instead.
        cache.store(&l, Dimension::Tex2d, 28, &hash, &levels(0x33));
        let back = cache.levels(&l, Dimension::Tex2d, 28, &hash, 0, 0).expect("stored");
        assert!(back.iter().all(|level| !level.texels.is_mapped()), "a deflated level is inflated");
        assert_eq!(back[1].texels.as_slice(), vec![0x33; 4096].as_slice());
    }

    /// The folder is capped: a prune keeps the most recently used entries and deletes the rest,
    /// down to the cap's headroom.
    #[test]
    fn the_folder_is_pruned_least_recently_used_first() {
        let dir = tempfile::tempdir().unwrap();
        let (l, made) = (layout(), levels());
        // Three entries of about the same size, the first used longest ago.
        let mut sizes = Vec::new();
        for n in 0..3u8 {
            let hash = blake3::hash(&[n]);
            let cache = Cache::open(dir.path(), 0).unwrap();
            cache.store(&l, Dimension::Tex2d, 71, &hash, &made);
            let files = walk(dir.path());
            let (path, size, _) = files.iter().max_by_key(|(_, size, _)| *size).unwrap();
            sizes.push((path.clone(), *size));
            let at = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000 + n as u64);
            std::fs::File::options().write(true).open(path).unwrap().set_modified(at).unwrap();
        }
        let total: u64 = sizes.iter().map(|(_, s)| *s).sum();
        // A cap of two entries: the oldest goes, the two others stay.
        let cache = Cache::open(dir.path(), total / 3 * 2 + total / 100).unwrap();
        cache.prune();
        assert_eq!(walk(dir.path()).len(), 2, "one entry deleted");
        assert!(!sizes[0].0.exists(), "the oldest is the one given up");
        assert!(sizes[1].0.exists() && sizes[2].0.exists());
    }

    /// Off unless asked for: a hit costs more than the detile it saves (the module's notes).
    #[test]
    fn the_cache_is_off_unless_asked_for() {
        assert!(!wanted(None, None));
        assert!(!wanted(Some(""), None));
        assert!(!wanted(Some("0"), None));
        assert!(wanted(Some("1"), None));
        assert!(wanted(Some("1"), Some("0")));
        assert!(!wanted(Some("1"), Some("1")));
        assert!(!wanted(None, Some("1")));
    }

    /// The cap's own setting: 10 GiB by default, `VICTORY_TEXTURE_CACHE_MB` in MiB, 0 off.
    #[test]
    fn the_cap_is_ten_gib_unless_set() {
        assert_eq!(limit_mib(None), 10 * 1024);
        assert_eq!(limit_mib(Some("2048")), 2048);
        assert_eq!(limit_mib(Some(" 512 ")), 512);
        assert_eq!(limit_mib(Some("0")), 0);
        assert_eq!(limit_mib(Some("lots")), 0);
    }
}
