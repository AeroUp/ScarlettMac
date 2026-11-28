//! `VICTORY_TRACE_LOCK=1` (with `VICTORY_TRACE_FRAME=1`): what the host's lock is held for.
//!
//! The frame line's `lock_hold_ms` is one sum over every caller of `glue::with_title`. This splits
//! it two ways, into a `lock time` line after each frame line:
//!
//! - by *site*, where the lock was taken: a list's recording call, the submission's phases (taking
//!   the lists, the texture bookkeeping, the uploads, the encoding), the host's 1 ms timer, queue
//!   and fence calls, descriptor and resource calls, the present, anything else. These do not
//!   overlap and add up to `lock_hold_ms`.
//! - by *part*, what the work under it was, wherever it ran: `Host::execute` and its pieces (the
//!   per-command preparation, the table remirrors, the typed buffer views, the root arguments, the
//!   page table builds, residency, the depth twins, the Metal encoding), `Title::pump`, and the
//!   texture side's naming, listing and budget passes. Parts nest (the remirror is inside the
//!   preparation, which is inside `execute`, which is inside a pump), so they do not add up.
//!
//! Off, every timer is one read of a cached switch; nothing is counted or logged.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Whether the split is on (`VICTORY_TRACE_LOCK`, any value but `0`/empty). The line is logged
/// with the frame line, so it needs `VICTORY_TRACE_FRAME` too.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_TRACE_LOCK").is_ok_and(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && v != "false"
        })
    })
}

/// Where the host's lock was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    Other,
    /// A command list's recording call (`glue::with_list`).
    Record,
    /// `ExecuteCommandLists`: taking the lists' segments.
    Take,
    /// `ExecuteCommandLists`: the texture bookkeeping (naming, listing, the frame's work budget).
    Textures,
    /// `ExecuteCommandLists`: uploading what was read off the lock.
    Upload,
    /// `ExecuteCommandLists`: queueing the segments, pumping and encoding.
    Encode,
    /// The host's 1 ms timer (`glue::refresh`).
    Timer,
    /// Queue signals and waits, fence signals, polls and waits.
    Queue,
    /// Descriptor heaps, descriptor writes and resource creation or release.
    Resource,
    /// `PresentX`.
    Present,
}

const SITES: [(Site, &str); 10] = [
    (Site::Record, "record"),
    (Site::Take, "take"),
    (Site::Textures, "textures"),
    (Site::Upload, "upload"),
    (Site::Encode, "encode"),
    (Site::Timer, "timer"),
    (Site::Queue, "queue"),
    (Site::Resource, "resource"),
    (Site::Present, "present"),
    (Site::Other, "other"),
];

/// Work done under the lock, wherever it was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    /// `Title::pump`: what may run goes to the GPU (contains `Execute`).
    Pump,
    /// `NativeHost::execute`: one piece of title work prepared and encoded.
    Execute,
    /// `NativeHost::follow_title_pages`: heap views whose title pages moved.
    Follow,
    /// `NativeHost::prep`: one command's preparation (contains the next four).
    Prep,
    /// `NativeHost::remirror_tables`: the bound tables' records compared and mirrored again.
    Remirror,
    /// `NativeHost::mirror`: one descriptor record decoded and written (from any caller).
    Mirror,
    /// `NativeHost::materialize_typed_buffers`: typed buffer views made for the bound tables.
    Typed,
    /// The root arguments of a command (`translated_args`, `memory_args`, `root_args`, ...).
    Args,
    /// `NativeHost::memory_table` building a page table again (the pool table changed).
    PageTable,
    /// `make_resident`: the pool buffers the piece reaches.
    Resident,
    /// The depth twins' copies and staging.
    Twins,
    /// `submit_passes`: encoding and committing the Metal command buffers.
    Metal,
    /// `Title::keep_named`: the submission's tables read for the texture budget.
    Names,
    /// Of `Names`, the figures only the budget trace prints (`HEAP_ONLY`).
    HeapShown,
    /// `Title::sync_jobs`: the texture bookkeeping before the reads (contains `Budget`).
    Jobs,
    /// `Title::keep_to_budget`.
    Budget,
    /// The timer's check for held work that does not move (`Title::stuck`).
    Stuck,
}

const PARTS: [(Part, &str); 17] = [
    (Part::Pump, "pump"),
    (Part::Execute, "execute"),
    (Part::Follow, "follow"),
    (Part::Prep, "prep"),
    (Part::Remirror, "remirror"),
    (Part::Mirror, "mirror"),
    (Part::Typed, "typed"),
    (Part::Args, "args"),
    (Part::PageTable, "page_table"),
    (Part::Resident, "resident"),
    (Part::Twins, "twins"),
    (Part::Metal, "metal"),
    (Part::Names, "names"),
    (Part::HeapShown, "heap_shown"),
    (Part::Jobs, "jobs"),
    (Part::Budget, "budget"),
    (Part::Stuck, "stuck"),
];

/// Counts that are not a timed part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Count {
    /// Records a table remirror compared with what it last mirrored.
    Checked,
    /// Of those, records that had changed.
    Changed,
    /// Typed buffer views made (a Metal texture each).
    TypedViews,
}

const COUNTS: [(Count, &str); 3] =
    [(Count::Checked, "records_checked"), (Count::Changed, "records_changed"), (Count::TypedViews, "typed_views")];

static SITE_US: [AtomicU64; SITES.len()] = [const { AtomicU64::new(0) }; SITES.len()];
static SITE_CALLS: [AtomicU64; SITES.len()] = [const { AtomicU64::new(0) }; SITES.len()];
static PART_US: [AtomicU64; PARTS.len()] = [const { AtomicU64::new(0) }; PARTS.len()];
static PART_CALLS: [AtomicU64; PARTS.len()] = [const { AtomicU64::new(0) }; PARTS.len()];
static COUNTED: [AtomicU64; COUNTS.len()] = [const { AtomicU64::new(0) }; COUNTS.len()];

fn site_index(site: Site) -> usize {
    SITES.iter().position(|(s, _)| *s == site).expect("every site is listed")
}

fn part_index(part: Part) -> usize {
    PARTS.iter().position(|(p, _)| *p == part).expect("every part is listed")
}

thread_local! {
    /// Where this thread's next lock is taken ([`at`]).
    static SITE: Cell<Site> = const { Cell::new(Site::Other) };
}

/// Marks this thread's locks as taken at `site` until the guard drops (the previous site is put
/// back, so a guard inside another nests).
pub(crate) fn at(site: Site) -> AtGuard {
    if !enabled() {
        return AtGuard(None);
    }
    AtGuard(Some(SITE.with(|s| s.replace(site))))
}

#[must_use = "the site holds while the guard lives"]
pub(crate) struct AtGuard(Option<Site>);

impl Drop for AtGuard {
    fn drop(&mut self) {
        if let Some(was) = self.0 {
            SITE.with(|s| s.set(was));
        }
    }
}

/// The site of this thread's next lock.
pub(crate) fn site() -> Site {
    if !enabled() {
        return Site::Other;
    }
    SITE.with(|s| s.get())
}

/// The lock taken at `site` was held for `held`.
pub(crate) fn held(site: Site, held: Duration) {
    if enabled() {
        let i = site_index(site);
        SITE_US[i].fetch_add(held.as_micros() as u64, Ordering::Relaxed);
        SITE_CALLS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Times `part` until the guard drops.
pub(crate) fn time(part: Part) -> Timed {
    Timed(enabled().then(|| (part, Instant::now())))
}

#[must_use = "the part is timed while the guard lives"]
pub(crate) struct Timed(Option<(Part, Instant)>);

impl Drop for Timed {
    fn drop(&mut self) {
        if let Some((part, since)) = self.0 {
            let i = part_index(part);
            PART_US[i].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            PART_CALLS[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Adds `n` to `count`.
pub(crate) fn count(count: Count, n: u64) {
    if enabled() && n > 0 {
        let i = COUNTS.iter().position(|(c, _)| *c == count).expect("every count is listed");
        COUNTED[i].fetch_add(n, Ordering::Relaxed);
    }
}

/// What has accumulated since the last call, as `name=value` fields, every counter zeroed: per
/// site `hold_<site>_ms` and `<site>_calls`, per part `<part>_ms` and `<part>_n`, then the counts.
/// Empty when the split is off.
pub(crate) fn take_line() -> String {
    if !enabled() {
        return String::new();
    }
    let ms = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / 1000.0;
    let n = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
    let mut out = Vec::new();
    for (i, (_, name)) in SITES.iter().enumerate() {
        out.push(format!("hold_{name}_ms={:.1} {name}_calls={}", ms(&SITE_US[i]), n(&SITE_CALLS[i])));
    }
    for (i, (_, name)) in PARTS.iter().enumerate() {
        out.push(format!("{name}_ms={:.1} {name}_n={}", ms(&PART_US[i]), n(&PART_CALLS[i])));
    }
    for (i, (_, name)) in COUNTS.iter().enumerate() {
        out.push(format!("{name}={}", n(&COUNTED[i])));
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every site, part and count has its own slot and a name a field can carry.
    #[test]
    fn every_site_part_and_count_is_listed_once() {
        for (i, (s, name)) in SITES.iter().enumerate() {
            assert_eq!(site_index(*s), i);
            assert!(!name.is_empty() && !name.contains(' '));
        }
        for (i, (p, name)) in PARTS.iter().enumerate() {
            assert_eq!(part_index(*p), i);
            assert!(!name.is_empty() && !name.contains(' '));
        }
        let names: std::collections::HashSet<&str> =
            SITES.iter().map(|s| s.1).chain(PARTS.iter().map(|p| p.1)).chain(COUNTS.iter().map(|c| c.1)).collect();
        assert_eq!(names.len(), SITES.len() + PARTS.len() + COUNTS.len());
    }
}
