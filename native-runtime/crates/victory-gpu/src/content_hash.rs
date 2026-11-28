//! The hash the native host tells title texture memory apart by: whether a texture's bytes changed
//! since they were last read (a submission's sync, the scanner's rounds, the hot and blank
//! probes), and the source key of the decoded-texture cache (`texture_cache`).
//!
//! It was blake3, which runs at about 2.2 GB/s a thread under Rosetta (the runtime is x86-64
//! code): 0.44 s of the reading threads' time per second of world, plus the scanner's thread and
//! the probes under the host's lock. Nothing here needs a cryptographic hash: the bytes are the
//! user's own title's, and a collision would show one stale texture until its next write. XXH3's
//! 128-bit hash runs at about 30 GB/s a thread under Rosetta (SSE2), so it is the default.
//! `VICTORY_TEXTURE_HASH=blake3` brings back blake3 for a before-and-after run: its hashes are
//! blake3's own 32 bytes, so the cache's keys are the ones an older build wrote.
//!
//! An XXH3 hash fills 16 of the 32 bytes and a fixed tag the others, so the two never name the
//! same entry of the cache.

use std::sync::OnceLock;

/// Which function [`ContentHash`] uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// XXH3, 128 bits (the default).
    Xxh3,
    /// blake3, as before (`VICTORY_TEXTURE_HASH=blake3`).
    Blake3,
}

/// The second half of an XXH3 [`ContentHash`].
const XXH3_TAG: &[u8; 16] = b"xxh3-128 content";

/// The kind `VICTORY_TEXTURE_HASH` asks for: `blake3` (any case), else XXH3.
pub fn kind_from(var: Option<&str>) -> Kind {
    match var.map(str::trim) {
        Some(v) if v.eq_ignore_ascii_case("blake3") => Kind::Blake3,
        _ => Kind::Xxh3,
    }
}

/// This process's kind, read once (and logged then).
pub fn kind() -> Kind {
    static KIND: OnceLock<Kind> = OnceLock::new();
    *KIND.get_or_init(|| {
        let kind = kind_from(std::env::var("VICTORY_TEXTURE_HASH").ok().as_deref());
        tracing::info!(?kind, "texture memory hash (VICTORY_TEXTURE_HASH=blake3 for the old one)");
        kind
    })
}

/// What some title memory hashed to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// The hash of `bytes`, with this process's [`kind`].
    pub fn of(bytes: &[u8]) -> ContentHash {
        ContentHash::of_kind(kind(), bytes)
    }

    /// The hash of `bytes` with `kind`.
    pub fn of_kind(kind: Kind, bytes: &[u8]) -> ContentHash {
        match kind {
            Kind::Xxh3 => ContentHash::xxh3(xxhash_rust::xxh3::xxh3_128(bytes)),
            Kind::Blake3 => ContentHash(*blake3::hash(bytes).as_bytes()),
        }
    }

    fn xxh3(h: u128) -> ContentHash {
        let mut out = [0u8; 32];
        out[..16].copy_from_slice(&h.to_le_bytes());
        out[16..].copy_from_slice(XXH3_TAG);
        ContentHash(out)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A [`ContentHash`] of bytes handed over in pieces: the same hash as the pieces joined.
pub enum Hasher {
    Xxh3(Box<xxhash_rust::xxh3::Xxh3>),
    Blake3(Box<blake3::Hasher>),
}

impl Hasher {
    /// With this process's [`kind`].
    #[allow(clippy::new_without_default)]
    pub fn new() -> Hasher {
        Hasher::of_kind(kind())
    }

    pub fn of_kind(kind: Kind) -> Hasher {
        match kind {
            Kind::Xxh3 => Hasher::Xxh3(Box::default()),
            Kind::Blake3 => Hasher::Blake3(Box::default()),
        }
    }

    pub fn update(&mut self, bytes: &[u8]) -> &mut Hasher {
        match self {
            Hasher::Xxh3(h) => h.update(bytes),
            Hasher::Blake3(h) => {
                h.update(bytes);
            }
        }
        self
    }

    pub fn finalize(&self) -> ContentHash {
        match self {
            Hasher::Xxh3(h) => ContentHash::xxh3(h.digest128()),
            Hasher::Blake3(h) => ContentHash(*h.finalize().as_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XXH3 unless `VICTORY_TEXTURE_HASH=blake3`; blake3's hashes are its own bytes, so the cache
    /// keys of an older build still match with it.
    #[test]
    fn texture_memory_hashes_with_xxh3_unless_blake3_is_asked_for() {
        assert_eq!(kind_from(None), Kind::Xxh3);
        assert_eq!(kind_from(Some("")), Kind::Xxh3);
        assert_eq!(kind_from(Some("xxh3")), Kind::Xxh3);
        assert_eq!(kind_from(Some(" BLAKE3 ")), Kind::Blake3);
        let bytes = b"tiled bytes";
        assert_eq!(ContentHash::of_kind(Kind::Blake3, bytes).as_bytes(), blake3::hash(bytes).as_bytes());
        let xxh3 = ContentHash::of_kind(Kind::Xxh3, bytes);
        assert_eq!(xxh3.as_bytes()[..16], xxhash_rust::xxh3::xxh3_128(bytes).to_le_bytes());
        assert_ne!(xxh3, ContentHash::of_kind(Kind::Blake3, bytes));
        assert_ne!(xxh3, ContentHash::of_kind(Kind::Xxh3, b"tiled bytez"));
    }

    /// The scanner hashes a texture 1 MiB at a time and a submission's sync all at once: both must
    /// come to the same hash, or every round would report every texture changed.
    #[test]
    fn a_hash_in_pieces_is_the_hash_of_the_whole() {
        let bytes: Vec<u8> = (0..3_000_001u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for kind in [Kind::Xxh3, Kind::Blake3] {
            for piece in [1000usize, 4096, 1 << 20] {
                let mut h = Hasher::of_kind(kind);
                for chunk in bytes.chunks(piece) {
                    h.update(chunk);
                }
                assert_eq!(h.finalize(), ContentHash::of_kind(kind, &bytes), "{kind:?} in pieces of {piece}");
            }
            assert_eq!(Hasher::of_kind(kind).finalize(), ContentHash::of_kind(kind, &[]));
        }
    }
}
