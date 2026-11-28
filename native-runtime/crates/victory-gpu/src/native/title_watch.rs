//! `VICTORY_WATCH_TITLE=<address>[,<length>]`: what a GPU write does to a range of title memory.
//!
//! Title memory is zero copy here: a title buffer view points into a pool, so a translated shader
//! that writes past its view, or through a page-table entry for an address it computed wrongly,
//! changes the bytes the title's CPU code later reads. This switch follows one range (by default
//! one 64 KiB title page) while a run goes:
//!
//! * every unordered-access buffer view that covers it is logged when the view is mirrored into a
//!   descriptor heap and when a compute dispatch is recorded, with the dispatch's shader hashes:
//!   the candidate writers;
//! * the bytes themselves are read at each present, and a change is logged with the offset and the
//!   old and new dwords: proof that a write landed there between the last present and this one.
//!
//! Off unless `VICTORY_WATCH_TITLE` is set, and bounded (one line per distinct view and the first
//! 64 changes), so it does not flood a run. It reads title memory only, never writes it.

use std::sync::{Mutex, OnceLock};

/// `VICTORY_WATCH_TITLE`, `address[,length]` in hex: the range to follow, one title page by
/// default. `None` when it is unset or unusable (logged once).
pub fn range() -> Option<(u64, u64)> {
    static RANGE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        let value = std::env::var("VICTORY_WATCH_TITLE").ok()?;
        let mut parts = value.split(',');
        let Some(at) = parts.next().and_then(hex) else {
            tracing::warn!(value, "VICTORY_WATCH_TITLE: expected <address>[,<length>] in hex; not watching");
            return None;
        };
        let len = match parts.next() {
            Some(v) => match hex(v) {
                Some(v) if v != 0 => v,
                _ => {
                    tracing::warn!(value, "VICTORY_WATCH_TITLE: the length is not a non-zero hex number; not watching");
                    return None;
                }
            },
            None => 1 << 16,
        };
        Some((at, len))
    })
}

fn hex(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    (!s.is_empty()).then(|| u64::from_str_radix(s, 16).ok()).flatten()
}

/// Whether the switch is on.
pub fn enabled() -> bool {
    range().is_some()
}

/// Whether `[addr, addr + len)` touches the watched range.
pub fn covers(addr: u64, len: u64) -> bool {
    let Some((watched, watched_len)) = range() else { return false };
    let end = addr.saturating_add(len);
    addr < watched.saturating_add(watched_len) && watched < end
}

/// Logs an unordered-access buffer view `[addr, addr + len)` that covers the watched range: at
/// most once per distinct `(source, addr, len)`.
pub fn view(source: &str, addr: u64, len: u64) {
    if !covers(addr, len) {
        return;
    }
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut h);
    let key = (h.finish(), addr, len);
    static SEEN: Mutex<Option<HashSet<(u64, u64, u64)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(HashSet::new);
    if seen.len() >= 256 || !seen.insert(key) {
        return;
    }
    let (watched, watched_len) = range().unwrap_or_default();
    tracing::warn!(
        watched = format!("{watched:#x}"),
        watched_len,
        view = format!("{addr:#x}"),
        len,
        source,
        "an unordered-access buffer view covers the watched title memory (a candidate GPU writer)"
    );
}

/// Reads the watched range at each present and logs a change since the last one: the offset, the
/// bytes before and after, and how much differs. At most 64 lines.
pub fn poll(memory: &super::memory::PoolTable) {
    let Some((at, len)) = range() else { return };
    let Ok(len) = usize::try_from(len) else { return };
    let mut now = vec![0u8; len];
    let read = |addr: usize, out: &mut [u8]| super::safe_read(addr as *const u8, out);
    match memory.read_mapped_with(at as usize, &mut now, |addr, out| read(addr, out)) {
        // Nothing of the range is in a pool (the title's image, or memory not mapped yet): read it
        // where it lies if it is mapped at all.
        None if !read(at as usize, &mut now) => return,
        _ => {}
    }
    static LAST: Mutex<Option<(u64, Vec<u8>)>> = Mutex::new(None);
    static CHANGES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let Some((last_at, before)) = last.as_ref() else {
        tracing::info!(
            watched = format!("{at:#x}"),
            len,
            bytes = ?&now[..len.min(32)],
            "VICTORY_WATCH_TITLE: following title memory (the first read is the baseline)"
        );
        *last = Some((at, now));
        return;
    };
    if *last_at != at || before.len() != now.len() || before[..] == now[..] {
        if *last_at != at {
            *last = Some((at, now));
        }
        return;
    }
    let changed = before.iter().zip(&now).filter(|(a, b)| a != b).count();
    let first = before.iter().zip(&now).position(|(a, b)| a != b).unwrap_or(0);
    let n = CHANGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 64 {
        let lo = first.saturating_sub(8);
        let hi = (first + 24).min(len);
        tracing::warn!(
            watched = format!("{at:#x}"),
            offset = first,
            changed,
            before = ?&before[lo..hi],
            after = ?&now[lo..hi],
            "the watched title memory changed since the last present"
        );
    }
    *last = Some((at, now));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_hex_parse() {
        assert_eq!(hex("0x10"), Some(16));
        assert_eq!(hex(" 10 "), Some(16));
        assert_eq!(hex(""), None);
        assert_eq!(hex("zz"), None);
    }

    #[test]
    fn coverage_touches_both_ends() {
        // `view` and `covers` read the process's switch; test the range arithmetic on its own.
        let (at, len) = (0x2000u64, 0x1000u64);
        let overlaps = |addr: u64, n: u64| addr < at + len && at < addr.saturating_add(n);
        assert!(overlaps(0x2000, 1));
        assert!(overlaps(0x2fff, 1));
        assert!(overlaps(0x1fff, 0x1002));
        assert!(!overlaps(0x1000, 0x1000));
        assert!(!overlaps(0x3000, 0x1000));
        assert!(!overlaps(0, 0));
    }
}
