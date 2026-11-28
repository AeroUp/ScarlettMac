//! The title's own watchdog, and the crash it makes when a scope has been
//! entered for too long.
//!
//! The title's watchdog thread (`image+0x2215a60` in the build we test)
//! sleeps a second, then looks at the scopes title threads have entered
//! (`image+0xb5b6b0` enters one, `image+0xb5b6f0` leaves it). Each active
//! scope holds the time it was entered. A scope older than 300 s crashes
//! the process; with no scope active at all, the fallback is 30 s without
//! the global heartbeat at `image+0x971fb8`.
//!
//! The crash is deliberate: the watchdog calls the registered crash
//! callback, which builds its report and then stores the fail-fast marker
//! `0x4d8bd421` at address 0 (`image+0x23cf683`). That store is why a fault
//! there reports `rax=0`: the instruction is `mov dword ptr [0],
//! 0x4d8bd421`, not a store through a pointer the host returned. HLE
//! allocation failures are logged where they happen (`XMemAllocDefault`,
//! `XMemVirtualAlloc refused`, `XMemAllocatePhysicalPages failed`, a page
//! mapping failed); none of them is this fault.
//!
//! `VICTORY_TRACE_TITLE_WATCHDOG=1` reads the scope table when the marker
//! fault arrives and says which entries are active and how much older than
//! the newest one they were entered. The table's address is for the build
//! we test (`image+0x6844a20`, 64 entries of 0x48 bytes: a count at +4 and
//! up to eight entry times at +8), and the entries' times are the HLE's
//! QPC, nanoseconds since the guest clock started. On a build with another
//! layout the count check refuses the table rather than printing nonsense.

/// `mov dword ptr [0x0], <marker>`: the title's fail-fast store (opcode, SIB with no base, a
/// zero displacement; the marker follows).
const FAIL_FAST_INSTRUCTION: [u8; 7] = [0xc7, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00];

/// Markers the store has carried: `0x4d8bd421` in the build of 27 November, and **0** in the build
/// of 28 November. The later watchdog (image+0x2215c35) stores zero, so the run of 28 November
/// 07:35 (`eef00a59`) was read as a null some host call returned, and its scope table -- the one
/// thing that says which title scope was stuck -- was never printed.
const FAIL_FAST_MARKERS: [u32; 2] = [0x4d8b_d421, 0];

/// `image+0x6844a20`: the per-thread scope table, for this build.
const TABLE_RVA: usize = 0x0684_4a20;
/// `image+0x971fb8`: the title watchdog's global heartbeat, for this build: the fallback clock
/// is 30 s without a change to it.
const HEARTBEAT_RVA: usize = 0x0971_fb8;
/// Entries in the table.
const GROUPS: usize = 64;
/// The age at which the watchdog fails a scope a title thread entered and did not leave, in
/// seconds: its other clock, alongside 30 s without the global heartbeat.
pub const SCOPE_DEADLINE_S: f64 = 300.0;
/// Bytes per entry: a dword, a count, and eight entry times.
const STRIDE: usize = 0x48;
/// Entry times an entry holds.
const TIMES: usize = 8;

/// One active watchdog scope: its table entry and the times it was entered.
#[derive(Debug, PartialEq, Eq)]
pub struct Scope {
    pub group: usize,
    pub entered: Vec<u64>,
}

/// Whether `rip` points at the title's fail-fast store.
pub fn is_fail_fast(rip: usize) -> bool {
    // SAFETY: `rip` is the faulting instruction, inside the title's
    // executable image; eleven bytes are mapped there.
    let at = unsafe { (rip as *const u8).cast::<[u8; 11]>().read_unaligned() };
    if at[..7] != FAIL_FAST_INSTRUCTION {
        return false;
    }
    let marker = u32::from_le_bytes(at[7..11].try_into().unwrap_or_default());
    FAIL_FAST_MARKERS.contains(&marker)
}

/// The scopes in a scope table, as its bytes read.
fn scopes_in(bytes: &[u8]) -> Option<Vec<Scope>> {
    if bytes.len() < GROUPS * STRIDE {
        return None;
    }
    let mut scopes = Vec::new();
    for group in 0..GROUPS {
        let entry = &bytes[group * STRIDE..(group + 1) * STRIDE];
        let count = u32::from_le_bytes(entry[4..8].try_into().ok()?);
        if count == 0 {
            continue;
        }
        if count as usize > TIMES {
            return None;
        }
        let mut entered = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let at = 8 + i * 8;
            let time = u64::from_le_bytes(entry[at..at + 8].try_into().ok()?);
            if time != 0 {
                entered.push(time);
            }
        }
        if !entered.is_empty() {
            scopes.push(Scope { group, entered });
        }
    }
    Some(scopes)
}

/// [`scopes_in`] for the stall monitor: an entry whose count is past the eight times it holds is
/// read as its first eight, and only a count no nesting explains (past [`GROUPS`]) refuses the
/// table.
///
/// A scope a thread nests more than eight deep -- the world load's call stack does -- made the
/// crash-time reader refuse the whole table, which is right for a fault (another build's layout
/// must not pass for an active scope) but blind for a monitor: the run of 28 November
/// (`20261128-133911-YLju97`) reported no scope at all for its last ten minutes, and then died on
/// the watchdog's 300 s scope branch (a breakpoint at `image+0x5149aec` from its own thread,
/// `23d0f9a`, `23d1211`), so nothing named the thread that never left its scope.
fn scopes_for_monitor(bytes: &[u8]) -> Option<Vec<Scope>> {
    if bytes.len() < GROUPS * STRIDE {
        return None;
    }
    let mut scopes = Vec::new();
    for group in 0..GROUPS {
        let entry = &bytes[group * STRIDE..(group + 1) * STRIDE];
        let count = u32::from_le_bytes(entry[4..8].try_into().ok()?);
        if count == 0 {
            continue;
        }
        // Nesting is counted, not a layout: keep what fits.
        if count as usize > GROUPS {
            return None;
        }
        let mut entered = Vec::with_capacity(TIMES.min(count as usize));
        for i in 0..TIMES.min(count as usize) {
            let at = 8 + i * 8;
            let time = u64::from_le_bytes(entry[at..at + 8].try_into().ok()?);
            if time != 0 {
                entered.push(time);
            }
        }
        if !entered.is_empty() {
            scopes.push(Scope { group, entered });
        }
    }
    Some(scopes)
}

/// Logs the scopes active at a watchdog fault. `base` is the title image's
/// base address. Called only when the fail-fast store is recognised.
pub fn log(base: usize) {
    // SAFETY: the table is in the title image's data; it is mapped for as
    // long as the image is. The count check below refuses another layout.
    let bytes = unsafe {
        std::slice::from_raw_parts((base + TABLE_RVA) as *const u8, GROUPS * STRIDE)
    };
    let Some(scopes) = scopes_in(bytes) else {
        tracing::warn!(
            at = format_args!("{:#x}", base + TABLE_RVA),
            "title watchdog: the scope table does not have this build's layout"
        );
        return;
    };
    let Some(newest) = scopes
        .iter()
        .flat_map(|s| s.entered.iter())
        .max()
        .copied()
    else {
        tracing::warn!(
            "title watchdog: no scope is active (the 30 s idle heartbeat path fired)"
        );
        return;
    };
    let detail = std::env::var_os("VICTORY_TRACE_TITLE_WATCHDOG").is_some();
    for scope in &scopes {
        let oldest = scope.entered.iter().copied().min().unwrap_or(newest);
        let age = newest.saturating_sub(oldest);
        if detail {
            tracing::warn!(
                group = scope.group,
                depth = scope.entered.len(),
                oldest = format_args!("{oldest:#x}"),
                age_s = format_args!("{:.1}", age as f64 / 1e9),
                "title watchdog: an active scope"
            );
        }
    }
    tracing::warn!(
        active = scopes.len(),
        newest = format_args!("{newest:#x}"),
        "title watchdog: scopes at the fail-fast marker"
    );
}

/// One scope active now: its table entry, how deep it was entered, and how long ago its oldest
/// entry was entered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Active {
    pub group: usize,
    pub depth: usize,
    pub age_s: f64,
}

/// The scopes active at `now` (the HLE's QPC, `victory_gpu::clock::counter`: the same clock the
/// table's entry times are on), with each one's age: what a stall monitor reads to see a scope a
/// title thread entered and did not leave (the watchdog fails one after 300 s). `None` when the
/// table does not have this build's layout, as in [`log`].
pub fn active(base: usize, now: u64) -> Option<Vec<Active>> {
    // SAFETY: the table is in the title image's data, mapped for the image's life; the count
    // check in `scopes_in` refuses another layout.
    let bytes = unsafe {
        std::slice::from_raw_parts((base + TABLE_RVA) as *const u8, GROUPS * STRIDE)
    };
    active_in(bytes, now)
}

/// [`active`] over a table's bytes.
fn active_in(bytes: &[u8], now: u64) -> Option<Vec<Active>> {
    let scopes = scopes_for_monitor(bytes)?;
    Some(
        scopes
            .iter()
            .filter_map(|s| {
                let oldest = s.entered.iter().copied().min()?;
                Some(Active {
                    group: s.group,
                    depth: s.entered.len(),
                    age_s: now.saturating_sub(oldest) as f64 / 1e9,
                })
            })
            .collect(),
    )
}

/// The title watchdog's global heartbeat (`image+0x971fb8`), read without changing it. Its
/// fallback clock fails the process 30 s after it last changed, so a stall monitor compares it
/// with what it read a second ago to know that clock is running.
pub fn heartbeat(base: usize) -> Option<u64> {
    Some(heartbeat_word(base + HEARTBEAT_RVA))
}

fn heartbeat_word(at: usize) -> u64 {
    // SAFETY: `at` is inside the image's data (or, in tests, a local): a plain read.
    unsafe { (at as *const u64).read_volatile() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marker_store_is_recognised_in_both_builds() {
        let mut bytes = [0u8; 16];
        // 27 November: `mov dword ptr [0], 0x4d8bd421`.
        bytes[..11].copy_from_slice(&[
            0xc7, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x21, 0xd4, 0x8b, 0x4d,
        ]);
        assert!(is_fail_fast(bytes.as_ptr() as usize));
        // 28 November: the same store of zero. A run that ended there is the watchdog too, and
        // has to name the scope that was stuck.
        bytes[..11].copy_from_slice(&[0xc7, 0x04, 0x25, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(is_fail_fast(bytes.as_ptr() as usize));
        // Another immediate is another store, and another instruction is not this one.
        bytes[7] = 1;
        assert!(!is_fail_fast(bytes.as_ptr() as usize));
        bytes[7] = 0;
        bytes[0] = 0xc6;
        assert!(!is_fail_fast(bytes.as_ptr() as usize));
    }

    #[test]
    fn scopes_are_read_by_their_count() {
        let mut table = vec![0u8; GROUPS * STRIDE];
        let at = |group: usize, offset: usize| group * STRIDE + offset;
        table[at(3, 4)..at(3, 8)].copy_from_slice(&2u32.to_le_bytes());
        table[at(3, 8)..at(3, 16)].copy_from_slice(&100u64.to_le_bytes());
        table[at(3, 16)..at(3, 24)].copy_from_slice(&200u64.to_le_bytes());
        table[at(7, 4)..at(7, 8)].copy_from_slice(&1u32.to_le_bytes());
        table[at(7, 8)..at(7, 16)].copy_from_slice(&300u64.to_le_bytes());
        // A count past the eight times an entry holds is another layout.
        table[at(9, 4)..at(9, 8)].copy_from_slice(&9u32.to_le_bytes());

        assert_eq!(scopes_in(&table), None);
        table[at(9, 4)..at(9, 8)].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            scopes_in(&table),
            Some(vec![
                Scope {
                    group: 3,
                    entered: vec![100, 200]
                },
                Scope {
                    group: 7,
                    entered: vec![300]
                }
            ])
        );
        assert_eq!(scopes_in(&table[..STRIDE]), None);
    }

    #[test]
    fn a_zero_time_is_not_an_active_scope() {
        let mut table = vec![0u8; GROUPS * STRIDE];
        let at = 5 * STRIDE;
        table[at + 4..at + 8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(scopes_in(&table), Some(Vec::new()));
    }

    /// A stall monitor reads each active scope's age against its own QPC: the entry a title
    /// thread entered 250 s ago is the one the watchdog fails at 300, whatever the newest entry
    /// in the table is.
    #[test]
    fn active_scopes_carry_their_age() {
        let mut table = vec![0u8; GROUPS * STRIDE];
        let at = |group: usize, offset: usize| group * STRIDE + offset;
        table[at(2, 4)..at(2, 8)].copy_from_slice(&2u32.to_le_bytes());
        table[at(2, 8)..at(2, 16)].copy_from_slice(&50_000_000_000u64.to_le_bytes());
        table[at(2, 16)..at(2, 24)].copy_from_slice(&200_000_000_000u64.to_le_bytes());
        table[at(9, 4)..at(9, 8)].copy_from_slice(&1u32.to_le_bytes());
        table[at(9, 8)..at(9, 16)].copy_from_slice(&250_000_000_000u64.to_le_bytes());
        let now = 300_000_000_000;
        assert_eq!(
            active_in(&table, now),
            Some(vec![
                Active { group: 2, depth: 2, age_s: 250.0 },
                Active { group: 9, depth: 1, age_s: 50.0 },
            ])
        );
        // A clock behind an entry (impossible but readable) does not make a negative age.
        let behind = active_in(&table, 1).unwrap();
        assert!(behind.iter().all(|a| a.age_s == 0.0));
        assert_eq!(active_in(&table[..STRIDE], now), None);
    }

    /// A scope nested deeper than the eight times an entry holds is read as its first eight by
    /// the monitor, so a deep call stack does not blind it; the crash-time reader still refuses
    /// the table then.
    #[test]
    fn a_deep_scope_is_read_by_the_monitor_and_refused_by_the_crash_reader() {
        let mut table = vec![0u8; GROUPS * STRIDE];
        let at = 4 * STRIDE;
        table[at + 4..at + 8].copy_from_slice(&12u32.to_le_bytes());
        for i in 0..12 {
            let t = 1_000_000 + i as u64;
            table[at + 8 + i * 8..at + 16 + i * 8].copy_from_slice(&t.to_le_bytes());
        }
        assert_eq!(scopes_in(&table), None, "the crash reader refuses a count past eight");
        let scopes = scopes_for_monitor(&table).expect("the monitor reads it");
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].entered.len(), 8);
        // A count no nesting explains is another layout, even to the monitor.
        table[at + 4..at + 8].copy_from_slice(&4096u32.to_le_bytes());
        assert_eq!(scopes_for_monitor(&table), None);
    }

    /// The heartbeat is a word of the image's data, read in place.
    #[test]
    fn the_heartbeat_is_read_where_it_lives() {
        let word = 0x0123_4567_89ab_cdefu64;
        assert_eq!(heartbeat_word(&word as *const u64 as usize), word);
    }
}
