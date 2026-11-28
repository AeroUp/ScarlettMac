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

/// An image descriptor's dword 0 (its title address in 256-byte units), read as data: served
/// from the bound texture's memory (`victory-gpu`'s `texture_words`), where the translator
/// refused it. These are the two appending-free compute shaders of the 27 November run's 114
/// that were blocked on nothing else.
#[test]
fn image_descriptor_base_addresses_read_as_data_translate() {
    assert_translates(
        "an image's base address (dword 0) read as data",
        &["cbb5aaad3dc55cc4", "ea4e30202d5918fd"],
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

/// Compute programs whose lowering left a lane mask join's number as a phi with no incoming
/// values (`Lower::mask_phi_number` cached it before an edge failed, and a later read found it):
/// they translated into DXIL that the Metal Shader Converter crashes on in SimplifyCFG
/// (the first four), taking the process with it. Refused now (`Lower::unfilled_phis`).
#[test]
fn a_join_whose_incoming_values_were_not_all_made_is_refused() {
    for hash in [
        "4791a0a64d9e325d",
        "9a95132bed6c6513",
        "b916af43ceba0a29",
        "dec5136b141ac4b2",
        "4e2be81db49aece1",
        "c22779675a2f025d",
    ] {
        let Some(shader) = translate(hash) else {
            return;
        };
        match shader {
            Ok(()) => panic!("{hash}: translated with a phi lacking an edge's value"),
            Err(e) => assert!(
                e.contains("a join whose incoming values were not all made"),
                "{hash}: {e}"
            ),
        }
    }
}
