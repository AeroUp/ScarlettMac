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
/// Entries in the table.
const GROUPS: usize = 64;
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
}
