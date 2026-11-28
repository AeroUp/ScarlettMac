//! A least amount of indirect light on the native host: the constant-level stand-in for the
//! title's ray-traced one, ported from ScarlettMac (`victory-gpu/src/ambient.rs` and
//! `ambient_set.rs`). Off unless `VICTORY_AMBIENT=1`.
//!
//! Grand Theft Auto VI's indirect diffuse light comes from probes it fills by tracing rays.
//! Where those do not run, its program that resolves them (`CS_ScreenProbes_TemporalResolve_*`)
//! leaves next to nothing, and whatever the sun does not reach directly is black. Right after
//! that program, its output is replaced by one constant shade: the same light everywhere (the
//! title's own lighting applies it with the surfaces' colours, and its exposure does not move
//! it). Replaced, not added, so the result is the next frame's history.
//!
//! This is the first slice: `VICTORY_AMBIENT_LEVEL` and `VICTORY_AMBIENT_TINT` only. The
//! sky-following level, the depth shading and the reflection hold of the ScarlettMac line are
//! not here yet (see `docs/native-host.md`, "The ambient stand-in").
//!
//! shortcut: the same light everywhere, from everywhere; gone once rays run.
//!
//! Which program. The console's containers carry no program name: of the 21,596 shader
//! containers a Series S run saw, none held a `CS_` name (only ten containers, all `Clear`,
//! `Blit` and `DistantLights` graphics programs, held one), so the ScarlettMac line's way of
//! following `CS_ScreenProbes_TemporalResolve_*` by name never fires here. The stand-in follows
//! the image instead: a compute dispatch whose root tables bind an unordered view of an R9G9B9E5
//! image ([`LIGHT_FORMAT`]) writes the light, and the pass goes right after it. The tables are
//! read to the entries the dispatch's shader names, and to [`TABLE_FLOOR`] where it names none
//! (a refused pipeline has no translation to name them); the resolve's light sits at dword 176
//! of root parameter 1, its 23rd entry ([`LIGHT`]). Nothing else about the program is needed,
//! and a refused resolve still gets its light. The depth image the resolve reads is not used by
//! this slice.
//!
//! Only that place is the light. The title writes other full-screen R9G9B9E5 images through
//! unordered views, and a match anywhere in the tables took the first of them: in the 15:15 run
//! (27 November) compute `39ef10815cb825a0`, a 3,500-instruction shader of `v_dot4_i32_i8` weights
//! with four images (a neural one, a guess), had one at dword 16 of root parameter 0. The pass
//! filled its 1920x1080 output with the constant and the frame became a flat beige haze with no
//! geometry. The resolve's own light was never dispatched in that run.
//!
//! The format is the view's, not only the resource's: the resolve writes the light through an
//! R9G9B9E5 view of an R32_UINT object (the words are the same), so a match on the resource's own
//! format alone misses it. `115` is B4G4R4A4_UNORM, not R9G9B9E5 (`67`); with 115 the follow
//! never fired in a run (`VICTORY_AMBIENT=1`, 27 November, only the start-up line), which the
//! first matched dispatch's line now makes visible.

use std::sync::OnceLock;

use crate::native::list::{Binding, Root};
use victory_backend::resources::{GraphicsBackend, HeapKind};

/// Whether the stand-in is on: `VICTORY_AMBIENT=1` (off when unset, empty, `0` or `off`). Read
/// once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| is_on(var("VICTORY_AMBIENT").as_deref()))
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Whether `VICTORY_AMBIENT`'s value turns the stand-in on: it is set, and is not empty, `0` or
/// `off`. Off by default on this line (in Belobug's fork the variable is the light's level and
/// the stand-in is on unless it says `off`; the level is `VICTORY_AMBIENT_LEVEL` here).
fn is_on(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("off"))
}

/// 0.6 looked balanced in the fork's run a2.
const LEVEL: f32 = 0.6;

/// `VICTORY_AMBIENT_LEVEL=<level>`: the light's level ([`LEVEL`] by default; 0 is no indirect
/// diffuse light at all).
pub(crate) fn level() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| level_of(var("VICTORY_AMBIENT_LEVEL").as_deref()))
}

fn level_of(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|l: &f32| l.is_finite() && *l >= 0.0)
        .unwrap_or(LEVEL)
}

/// `VICTORY_AMBIENT_TINT=r,g,b`: the light's colour; warm white otherwise.
pub(crate) fn tint() -> [f32; 3] {
    static SET: OnceLock<[f32; 3]> = OnceLock::new();
    *SET.get_or_init(|| tint_of(var("VICTORY_AMBIENT_TINT").as_deref()))
}

fn tint_of(value: Option<&str>) -> [f32; 3] {
    let set: Vec<f32> = value
        .unwrap_or_default()
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    match set[..] {
        [r, g, b] => [r, g, b],
        _ => [1.0, 0.98, 0.94],
    }
}

/// The title texture format the light is: `R9G9B9E5_SHAREDEXP` (DXGI 67), one 4-byte texel an
/// entry, which the pass writes through a 32-bit integer view. (`115` is `B4G4R4A4_UNORM`.)
pub(crate) const LIGHT_FORMAT: u32 = 67;

/// Where the resolve writes the light (R9G9B9E5 as 32-bit integers) in the title's root
/// signature: root parameter 1, dword 176 of its table, its 23rd entry. Other programs write
/// R9G9B9E5 images elsewhere in their tables (compute `39ef10815cb825a0` at dword 16 of root
/// parameter 0), so the place, not the format alone, says which image is the light.
pub(crate) const LIGHT: (u8, u32) = (1, 176);

/// How many entries of each bound table are read when the dispatch's shader names none (a refused
/// pipeline has no translation to name them): 24 reaches the resolve's light, at dword 176 of root
/// parameter 1 (its 23rd entry). The pass trace reads the same 24 for a dropped dispatch.
pub(crate) const TABLE_FLOOR: usize = 24;

/// Whether an unordered view names the light: `view` is the format its description gives (`0`:
/// none, so the resource's own) and `resource` the resource's own format. An R9G9B9E5 view of an
/// R32_UINT object is the light (the resolve writes its words that way); an R32_UINT view of one
/// is not (the reflections' history is written that way).
pub(crate) fn is_light(view: u32, resource: u32) -> bool {
    let named = if view == 0 { resource } else { view };
    named == LIGHT_FORMAT
}

/// An unordered view of a light image a dispatch binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BoundLight {
    pub(crate) resource: usize,
    /// The format its view names (the resource's own where it names none).
    pub(crate) view_format: u32,
    /// The resource's own format.
    pub(crate) resource_format: u32,
    /// The root parameter whose table holds it.
    pub(crate) slot: u8,
    /// Its place in that table, in dwords from the table's start.
    pub(crate) offset: u32,
    pub(crate) size: [u32; 2],
}

/// The light image the dispatch with these root parameters writes, at the resolve's own place in
/// its root signature ([`LIGHT`]), `image(resource)` giving a resource's
/// `(format, width, height)`.
///
/// A program the title refused has no translation, so `binding` is `None` and the table is read
/// to [`TABLE_FLOOR`] entries; otherwise the entries its shader names, or the floor, whichever is
/// more. An R9G9B9E5 image anywhere else is not the light: the title's other programs write
/// full-screen R9G9B9E5 images too.
pub(crate) fn bound_lights<B: GraphicsBackend>(
    root: &[Option<Root<B>>],
    binding: Option<&Binding>,
    mut image: impl FnMut(usize) -> Option<(u32, u32, u32)>,
) -> Vec<BoundLight> {
    let (slot, offset) = LIGHT;
    let Some(Some(Root::Table(heap, at))) = root.get(usize::from(slot)) else {
        return Vec::new();
    };
    let named = binding
        .and_then(|b| {
            b.table_entries
                .iter()
                .filter(|(s, _)| *s == slot)
                .map(|(_, n)| *n)
                .max()
        })
        .unwrap_or(0);
    let count = named.max(TABLE_FLOOR).min(heap.count.saturating_sub(*at));
    let step = usize::try_from(
        crate::descriptor::increment(u32::from(heap.kind == HeapKind::Sampler)).unwrap_or(32),
    )
    .unwrap_or(32);
    // The offset is in dwords; a descriptor may be more than one dword, so only an exact place
    // on a descriptor can be the light.
    let dwords = offset as usize * 4;
    if dwords % step != 0 {
        return Vec::new();
    }
    let index = dwords / step;
    if index >= count {
        return Vec::new();
    }
    let handle = heap.start + ((*at + index) * step) as u64;
    let Some(crate::descriptor::View::Unordered(resource, _, desc)) =
        crate::descriptor::view(handle)
    else {
        return Vec::new();
    };
    let Some((format, width, height)) = image(resource) else {
        return Vec::new();
    };
    let view_format = desc.map_or(0, |d| d[0]);
    if !is_light(view_format, format) {
        return Vec::new();
    }
    vec![BoundLight {
        resource,
        view_format: if view_format == 0 {
            format
        } else {
            view_format
        },
        resource_format: format,
        slot,
        offset: (index * step / 4) as u32,
        size: [width, height],
    }]
}

/// Says what is on, once, when the host device is ready.
pub(crate) fn announce() {
    if !on() {
        return;
    }
    tracing::info!("{}", start_line(level(), tint()));
}

/// The line logged once at start when the stand-in is on.
fn start_line(level: f32, tint: [f32; 3]) -> String {
    let [r, g, b] = tint;
    format!(
        "VICTORY_AMBIENT: the screen probes' indirect diffuse light is replaced by a constant \
         ambient light of level {level}, tint {r},{g},{b}"
    )
}

/// What one pass is given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pass {
    pub(crate) level: f32,
    pub(crate) tint: [f32; 3],
    /// The image's size.
    pub(crate) size: [u32; 2],
}

/// Dwords of the pass's constant buffer: level, tint r, g, b, width, height.
pub(crate) const CONSTANTS: usize = 6;

/// The pass's constants, in the order the kernel declares them.
pub(crate) fn constants(p: &Pass) -> [u32; CONSTANTS] {
    let floats = [p.level, p.tint[0], p.tint[1], p.tint[2]];
    let mut out = [0; CONSTANTS];
    for (to, f) in out.iter_mut().zip(floats) {
        *to = f.to_bits();
    }
    out[4..].copy_from_slice(&p.size);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off unless set to something other than nothing, `0` or `off`.
    #[test]
    fn off_by_default() {
        for off in [
            None,
            Some(""),
            Some("  "),
            Some("0"),
            Some(" 0 "),
            Some("off"),
            Some("OFF"),
        ] {
            assert!(!is_on(off), "{off:?}");
        }
        for on in ["1", "on", "yes", "0.6", "2"] {
            assert!(is_on(Some(on)), "{on:?}");
        }
    }

    #[test]
    fn levels_and_colours_keep_their_defaults_when_unreadable() {
        assert_eq!(level_of(None), LEVEL);
        assert_eq!(level_of(Some("x")), LEVEL);
        assert_eq!(level_of(Some("-1")), LEVEL);
        assert_eq!(level_of(Some("inf")), LEVEL);
        assert_eq!(level_of(Some(" 0 ")), 0.0);
        assert_eq!(level_of(Some("1.4")), 1.4);
        assert_eq!(tint_of(None), [1.0, 0.98, 0.94]);
        assert_eq!(tint_of(Some("1,2")), [1.0, 0.98, 0.94]);
        assert_eq!(tint_of(Some("0.5, 0.25,1")), [0.5, 0.25, 1.0]);
    }

    /// The light is R9G9B9E5 (DXGI 67), not B4G4R4A4 (115) and not the R32_UINT object it is
    /// written through.
    #[test]
    fn the_light_has_one_format() {
        assert_eq!(LIGHT_FORMAT, 67);
        for other in [0, 10, 28, 41, 42, 87, 91, 115, 116] {
            assert_ne!(LIGHT_FORMAT, other);
        }
        assert!(is_light(67, 42));
        assert!(is_light(67, 67));
        // A view that names no format is the resource's own.
        assert!(is_light(0, 67));
        assert!(!is_light(0, 42));
        assert!(!is_light(42, 67));
        assert!(!is_light(115, 115));
    }

    /// The light is the image at the resolve's own place in the root signature.
    #[test]
    fn the_light_has_one_place() {
        assert_eq!(LIGHT, (1, 176));
    }

    /// The constants go out in the order the kernel declares them, the size after the four
    /// floats.
    #[test]
    fn constants_follow_the_kernel() {
        let p = Pass {
            level: 0.6,
            tint: [1.0, 0.98, 0.94],
            size: [1920, 1080],
        };
        let c = constants(&p);
        let floats: Vec<f32> = c[..4].iter().map(|&w| f32::from_bits(w)).collect();
        assert_eq!(floats, [0.6, 1.0, 0.98, 0.94]);
        assert_eq!(c[4..], [1920, 1080]);
    }

    #[test]
    fn the_start_line_says_what_is_on() {
        let line = start_line(0.6, [1.0, 0.98, 0.94]);
        assert!(line.starts_with("VICTORY_AMBIENT: "), "{line}");
        assert!(
            line.contains("constant ambient light of level 0.6, tint 1,0.98,0.94"),
            "{line}"
        );
    }
}
