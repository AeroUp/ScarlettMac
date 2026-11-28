//! Regressions on the containers a native run kept (`victory-gpu`'s
//! `prewarm::keep`), the shaders a world frame dropped. Point
//! `VICTORY_SHADER_STORE` at the run's folder of kept containers
//! (`<cache>/native-shaders/containers`, a file a short hash); without it
//! these tests say nothing and pass.
//!
//! The hashes are containers the series-s world log named as refused, by the
//! reason they were refused for. Each reason here has one fix and one test:
//! a descriptor's dwords the size query does not give come from the view
//! addresses (`lower::view_address_dword`), and `v_cvt_pk_u16_u32` /
//! `v_cvt_pk_i16_i32` lower to a clamp and a pack (`lower::valu_op`).

use std::path::PathBuf;

use super::{Options, translate_with};

/// A run's kept containers, as the native host translates them.
fn translate(hash: &str) -> Option<Result<(), String>> {
    let dir = PathBuf::from(std::env::var_os("VICTORY_SHADER_STORE")?);
    let bytes = std::fs::read(dir.join(hash)).ok()?;
    let options = Options {
        wave32_host: true,
        zero_copy_memory: true,
        ..Options::default()
    };
    Some(
        translate_with(&bytes, options)
            .map(|_| ())
            .map_err(|e| e.to_string()),
    )
}

fn assert_translates(reason: &str, hashes: &[&str]) {
    for &hash in hashes {
        let Some(shader) = translate(hash) else {
            return;
        };
        if let Err(e) = shader {
            panic!("{hash} ({reason}): {e}");
        }
    }
}

/// A buffer descriptor's base address, the address's high bits and stride,
/// and the control word, read as data: all four dwords come from the view
/// addresses (`t1`) now, so the programs translate where each was refused.
#[test]
fn buffer_descriptor_dwords_read_as_data_translate() {
    assert_translates(
        "a buffer descriptor's dword read as data",
        &[
            "09663050d8513ca2", // dword 0, the base address
            "008c909bee8cc93e", // dword 1, the address's high bits and stride
            "0c595a38be5eb4d4", // dword 3, the format and control word
        ],
    );
}

/// `ds_append` / `ds_consume` in a console wave the host runs as two 32-lane waves: the
/// frame's light-list compute (`0eda771eedd6fd25`, whose loss leaves the world's lights
/// unbuilt) and two more of a world run, each refused for the split wave.
#[test]
fn appends_in_a_console_wave_the_host_splits_translate() {
    assert_translates(
        "ds_append or ds_consume in a console wave the host splits",
        &["0eda771eedd6fd25", "f2cc14c2d61e7d88", "eb23425f7602debc"],
    );
}

/// `v_cvt_pk_u16_u32`: two 32-bit integers clamped to 16 bits and packed.
/// These are pixel and vertex programs of a world frame's effects.
#[test]
fn packed_unsigned_16_bit_conversions_translate() {
    assert_translates(
        "v_cvt_pk_u16_u32",
        &[
            "422f7a28d73c331e",
            "5a8a99164dce475f",
            "961a3553fd9d9a94",
            "28e8d398fb691e37",
        ],
    );
}

/// 113-120 KB compute programs that spill an image descriptor's base address to a VGPR lane
/// beside an EXEC they read back from lane 0 of the same VGPR, through a join with an edge
/// where the VGPR is not written yet: refused for the base address before each lane kept its
/// own value (`lower::Lower::spill_slot`, `lane_writes`).
#[test]
fn a_descriptor_address_spilled_beside_a_restored_exec_translates() {
    assert_translates(
        "an image's base address (dword 0) spilled to a VGPR lane",
        &["bf865cf731f6c8ff", "0318b6d2a9d5c44a", "ff6136ac963ef248"],
    );
}

/// Programs whose lowering made a lane mask's number at a join as a phi, failed to fill it
/// (an edge's mask or data has no number), and then found the empty phi through the cache on a
/// later read: they translated into a function the Metal Shader Converter crashed on, in a run
/// and offline. Now the later reads fail as the first did, so these are refused for their own
/// reasons (none for "a join whose incoming values were not all made", the guard behind it).
/// When those joins are lowered, this becomes a check that they translate.
#[test]
fn a_mask_joins_number_that_failed_is_not_read_again() {
    for hash in [
        "4997b6b9c6de757e",
        "56ee7eb1a3f7f4d5",
        "7d6341512e0fdfac",
        "91c3a450b2bf40e2",
        "a13a2417f299bb5d",
        "a3bfb55326489164",
        "ef2257f9d25f8191",
        "a75d0c3cc594f093",
    ] {
        let Some(shader) = translate(hash) else {
            return;
        };
        match shader {
            Ok(()) => panic!("{hash}: translated with a join's number left unmade"),
            Err(e) => assert!(!e.contains("not all made"), "{hash}: {e}"),
        }
    }
}
