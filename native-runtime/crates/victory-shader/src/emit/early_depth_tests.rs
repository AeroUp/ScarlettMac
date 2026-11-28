//! `DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER`: the console tests depth and
//! stencil before the pixel shader, so the host asks DXIL for the same
//! (`ForceEarlyDepthStencil`, shader flag bit 3) even where the shader writes
//! UAVs or title memory or discards. A shader exporting depth, stencil or
//! coverage keeps its depth test after the shader runs, as on the console.

use super::container::SigElement;
use super::lower::{self, Lowered, Parts};
use crate::native::{HwStage, NativeProgram, RootParam, RootType, StageIo};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;

fn element(name: &str, index: u32, kind: u8, row: u32, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

/// A pixel program that reads an image descriptor from the resource heap and
/// then runs `extra`, exports `export` and ends. `db_shader_control` is the
/// console's `DB_SHADER_CONTROL`, `outputs` its pixel signature.
fn lower_ps(
    extra: &[u32],
    exports: &[u32],
    outputs: Vec<SigElement>,
    db_shader_control: u32,
    view_marks: bool,
) -> Lowered {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302, // s_load_dwordx8 s[12:19], s[4:5], 32
        0xFA00_0020,
        S_WAITCNT_ALL,
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF000_1F08, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0x0003_0200,
        S_WAITCNT_ALL,
    ];
    words.extend_from_slice(extra);
    words.extend_from_slice(exports);
    words.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = crate::structure::structure(&p).expect("structure");
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = crate::io::raise(&p, &native).expect("io::raise");
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs,
        prims: Vec::new(),
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        rect_list: false,
        copy_params: [0; 32],
        view_marks,
    })
    .expect("lower")
}

/// A UAV store through a descriptor loaded from the resource heap, then a
/// colour export: with the bit the DXIL tests depth first, so a pixel behind
/// closer geometry never makes the store, with or without view marks.
#[test]
fn depth_before_shader_keeps_early_depth_with_uav_writes() {
    const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
    let uav_store = [
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 64
        0xFA00_0040,
        S_WAITCNT_ALL,
        0xE070_0000, // buffer_store_dword v2, off, s[8:11], 0
        0x8002_0000,
    ];
    let color = [0xF800_180F, 0x0504_0302]; // exp mrt0 v2, v3, v4, v5 done vm
    let target = || vec![element("SV_Target", 0, 16, 0, 4)];
    for view_marks in [false, true] {
        assert!(
            !lower_ps(&uav_store, &color, target(), 0, view_marks).early_depth_stencil,
            "title UAV store, view marks {view_marks}, without the bit"
        );
        let with_bit = lower_ps(&uav_store, &color, target(), DEPTH_BEFORE_SHADER, view_marks);
        assert!(
            with_bit.early_depth_stencil,
            "title UAV store, view marks {view_marks}"
        );
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut with_bit.dxil.clone()).unwrap();
        }
    }
}

/// A conditional discard (lanes that fail the compare leave for the end) and
/// a plain colour export take early depth with the bit too: on the console
/// those pixels never run and never discard.
#[test]
fn depth_before_shader_keeps_early_depth_with_a_discard() {
    const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
    let lowered = lower_ps(
        &[
            0xBE88_037E, // s_mov_b32 s8, exec_lo
            0x7C22_0480, // v_cmpx_lt_f32 exec, 0, v2
            0x8A08_7E08, // s_andn2_b32 s8, s8, exec_lo
            0xBF84_0003, // s_cbranch_scc0 -> kill
            0xBEFE_0308, // s_mov_b32 exec_lo, s8
        ],
        &[
            0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
            0xF800_1890, 0x0000_0000, // kill: exp null done vm
        ],
        vec![element("SV_Target", 0, 16, 0, 4)],
        DEPTH_BEFORE_SHADER,
        true,
    );
    assert!(lowered.early_depth_stencil);
}

/// A shader exporting `SV_Depth` cannot test depth first: the console does
/// not combine the two either, so the bit adds no early depth.
#[test]
fn depth_before_shader_leaves_a_depth_export_late() {
    const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
    let depth = lower_ps(
        &[],
        &[0xF800_1881, 0x0000_0002], // exp mrtz v2, off, off, off done vm
        vec![element("SV_Depth", 0, 17, 0, 1)],
        DEPTH_BEFORE_SHADER,
        true,
    );
    assert!(!depth.early_depth_stencil);
    // A stencil export is the same (the colour export beside it is ignored
    // for the test: the rule looks at the signature).
    let stencil = lower_ps(
        &[],
        &[0xF800_180F, 0x0504_0302],
        vec![
            element("SV_StencilRef", 0, 20, 0, 1),
            element("SV_Target", 0, 16, 1, 4),
        ],
        DEPTH_BEFORE_SHADER,
        true,
    );
    assert!(!stencil.early_depth_stencil);
}

/// The bit is read only from a pixel stage's `DB_SHADER_CONTROL`.
#[test]
fn depth_before_shader_is_off_without_the_bit() {
    let plain = lower_ps(
        &[],
        &[0xF800_180F, 0x0504_0302],
        vec![element("SV_Target", 0, 16, 0, 4)],
        0,
        true,
    );
    assert!(!plain.early_depth_stencil);
    // Other bits, including the ordered section's, are not it.
    let other = lower_ps(
        &[],
        &[0xF800_180F, 0x0504_0302],
        vec![element("SV_Target", 0, 16, 0, 4)],
        1 << 16,
        true,
    );
    assert!(!other.early_depth_stencil);
}
