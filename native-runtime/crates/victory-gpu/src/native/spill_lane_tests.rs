//! SGPR spills to VGPR lanes beside a descriptor dword the host has no value for, run on the GPU:
//! synthetic GFX10 compute programs of our own, hand-assembled, with the shape of the title's
//! 110-137 KB spill-family programs (32-lane waves of 32 threads): an image descriptor's base
//! address (dword 0) spilled to lane 12 of v6, an EXEC or a counter spilled to lane 0 of the same
//! VGPR, read back with `v_readlane_b32` in a loop or after a join, restored to EXEC and branched
//! on. Each lane stores its result to a raw UAV. The expected values are what a GFX10 wave
//! computes. Skipped without the Metal Shader Converter.

use super::addr_tests::{CANARY, Fixture, VA_A, halves, title};
use crate::native::list::NativeList;
use crate::native::shaders;
use crate::native::title::Pipe;
use victory_backend::resources::{Format, HeapKind, ResourceBackend, TextureDesc};
use victory_formats::texture::Swizzle;

const VCC_LO: u32 = 106;
const EXEC_LO: u32 = 126;
/// An inline integer constant `n` (0..=64) as a scalar source.
const fn k(n: u32) -> u32 {
    128 + n
}
const fn sop1(op: u32, dst: u32, src: u32) -> u32 {
    0xBE80_0000 | dst << 16 | op << 8 | src
}
const fn sop2(op: u32, dst: u32, a: u32, b: u32) -> u32 {
    0x8000_0000 | op << 23 | dst << 16 | b << 8 | a
}
const fn sopc(op: u32, a: u32, b: u32) -> u32 {
    0xBF00_0000 | op << 16 | b << 8 | a
}
const S_MOV_B32: u32 = 3;
const S_AND_SAVEEXEC_B32: u32 = 0x3C;
const S_ADD_U32: u32 = 0;
const S_AND_B32: u32 = 14;
const S_OR_B32: u32 = 16;
const S_ANDN2_B32: u32 = 20;
const S_LSHL_B32: u32 = 30;
const S_CMP_EQ_U32: u32 = 6;
const S_CMP_LT_U32: u32 = 10;
const S_BRANCH: u32 = 2;
const S_CBRANCH_SCC0: u32 = 4;
const S_CBRANCH_SCC1: u32 = 5;
const S_CBRANCH_EXECZ: u32 = 8;
const S_CBRANCH_EXECNZ: u32 = 9;
const S_ENDPGM: u32 = 0xBF81_0000;
/// `v_cmp_gt_u32 vcc_lo, n, v0`: the lanes below `n`.
const fn lanes_below(n: u32) -> u32 {
    0x7D88_0000 | k(n)
}
/// `v_cmp_lt_u32 vcc_lo, s<s>, v0`: the lanes above `s<s>`.
const fn lanes_above_sgpr(s: u32) -> u32 {
    0x7D82_0000 | s
}
/// `v_add_nc_u32 v<d>, <n>, v<d>` (an inline constant, or a literal when `n > 64`).
fn add(d: u32, n: u32) -> Vec<u32> {
    if n <= 64 {
        vec![0x4A00_0000 | d << 17 | d << 9 | k(n)]
    } else {
        vec![0x4A00_0000 | d << 17 | d << 9 | 255, n]
    }
}
/// `v_add_nc_u32 v<d>, s<s>, v<d>`.
const fn add_sgpr(d: u32, s: u32) -> u32 {
    0x4A00_0000 | d << 17 | d << 9 | s
}
/// `v_writelane_b32 v<v>, s<s>, <lane>`.
const fn writelane(v: u32, s: u32, lane: u32) -> [u32; 2] {
    [0xD761_0000 | v, s | k(lane) << 9]
}
/// `v_readlane_b32 s<s>, v<v>, <lane>`.
const fn readlane(s: u32, v: u32, lane: u32) -> [u32; 2] {
    [0xD760_0000 | s, (256 + v) | k(lane) << 9]
}

/// Hand assembly with labels for the scalar branches.
#[derive(Default)]
struct Asm {
    words: Vec<u32>,
    labels: Vec<(&'static str, usize)>,
    jumps: Vec<(usize, &'static str)>,
}

impl Asm {
    fn op(&mut self, w: impl AsRef<[u32]>) -> &mut Self {
        self.words.extend_from_slice(w.as_ref());
        self
    }
    fn label(&mut self, name: &'static str) -> &mut Self {
        self.labels.push((name, self.words.len()));
        self
    }
    /// A SOPP branch (`op`) to `to`.
    fn branch(&mut self, op: u32, to: &'static str) -> &mut Self {
        self.jumps.push((self.words.len(), to));
        self.words.push(0xBF80_0000 | op << 16);
        self
    }
    fn done(&mut self) -> Vec<u32> {
        let mut w = self.words.clone();
        for &(at, to) in &self.jumps {
            let &(_, target) = self.labels.iter().find(|l| l.0 == to).expect("a label");
            w[at] |= (target as i32 - at as i32 - 1) as u32 & 0xFFFF;
        }
        w
    }
}

/// The title's shape up to the spills: the raw UAV of the table at user data 4 in s[12:15], the
/// image descriptor of heap entry 1 (a 1x1 green texture) in s[16:23] from the heap base in s6,
/// its texel's green channel as an integer (1) in v3, and each lane's byte address in v1.
fn prologue() -> Asm {
    let mut a = Asm::default();
    a.op([
        0xBE8A_0304, // s_mov_b32 s10, s4
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF408_0305, 0xFA00_0000, // s_load_dwordx4 s[12:15], s[10:11], 0x0
        0xBE80_0306, // s_mov_b32 s0, s6
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF40C_0400, 0xFA00_0020, // s_load_dwordx8 s[16:23], s[0:1], 0x20
        0x7E10_0280, // v_mov_b32 v8, 0
        0x7E12_0280, // v_mov_b32 v9, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_1F08, 0x0004_0208, // image_load v[2:5], v[8:9], s[16:23] dmask:0xf dim:2D unorm
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E06_0F03, // v_cvt_u32_f32 v3, v3
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E14_0280, // v_mov_b32 v10, 0
        0x7E16_0280, // v_mov_b32 v11, 0
    ]);
    a
}

/// The descriptor's base address (dword 0, s16) to lane 12 of v6: the VGPR has no host value.
const SPILL_BASE: [u32; 2] = writelane(6, 16, 12);
/// `buffer_store_dword v<d>, v1, s[12:15], 0 offen`.
const fn store(d: u32) -> [u32; 2] {
    [0xE070_1000, 0x8003_0001 | d << 8]
}

/// `code` run as one group of 32 threads in 32-lane waves, as the host translates and runs the
/// title's: the first 32 dwords of the UAV (the canary where no lane stored). None without the
/// converter. A refusal fails the test with its reason.
fn run(code: &[u32]) -> Option<Vec<u32>> {
    run_wave(code, 32, false)
}

/// `code` translated as the host translates the title's (a raw UAV in the table at user data 4,
/// seven user SGPRs, one group of `threads`).
fn translate(code: &[u32], threads: u32, wave64: bool) -> Result<victory_shader::emit::HostShader, String> {
    // A raw UAV (kind 3) in a table (root type 3) at user data 4, slot 0.
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 1;
    uav[20..23].copy_from_slice(&[4, 4, 0]);
    shaders::translate(&crate::pipeline::test_compute_container_of(code, &[uav], 7, threads, wave64))
}

/// `code` (32 lanes) is not translated.
fn assert_refused(code: &[u32]) {
    let made = translate(code, 32, false);
    assert!(made.is_err(), "translated");
}

fn run_wave(code: &[u32], threads: u32, wave64: bool) -> Option<Vec<u32>> {
    let cs = translate(code, threads, wave64).unwrap_or_else(|e| panic!("refused: {e}"));
    let mut t = title()?;
    let f = Fixture::new(&t);
    // A resource heap of three entries: entry 1 a 1x1 RGBA8 green texture (a resource object of
    // its own in title memory, its host texture registered under it), entry 2 a raw buffer UAV
    // of 128 dwords at `VA_A` (pool offset 0x2_0000).
    let records = [0u32; 8 * 3];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 3).unwrap();
    let object = crate::resource::test_texture_at(VA_A + 0x4_0000, (1, 1), 1, 28, Swizzle::LINEAR.0, 0);
    let tex = t.host.texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1)).unwrap();
    t.host.backend.upload_texture(&tex, 0, 0, &[0, 255, 0, 255], 4, 0);
    t.host.register_texture(object as usize, &tex);
    // DXGI_FORMAT_R8G8B8A8_UNORM, TEXTURE2D, one mip.
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    crate::descriptor::create_srv(std::ptr::null(), object, &srv, start + 32);
    assert!(t.host.mirror(start + 32).unwrap());
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, 128, 1);
    [view[10], view[11]] = halves(VA_A);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, start + 64);
    assert!(t.host.mirror(start + 64).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "spill lanes") else { panic!("{:?}", t.shaders) };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap, 2);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    Some(f.get(0x2_0000, threads as usize))
}

/// What `lanes` (by lane) store, the canary for `None`.
fn expect(lanes: u32, f: impl Fn(u32) -> Option<u32>) -> Vec<u32> {
    (0..lanes).map(|l| f(l).unwrap_or(CANARY)).collect()
}

/// An EXEC saved by `s_and_saveexec_b32` (lanes below 28, the others left at the start) and
/// spilled to lane 0 beside the descriptor's address in lane 12, across an EXEC branch of the
/// lanes below 20 that clobbers the saved SGPR; after the join it is read back from lane 0,
/// moved to EXEC (`s_mov_b32 exec_lo`, or ored in) and branched on with `s_cbranch_execz`.
/// GFX10: lanes 0-19 add 100 in the branch, lanes 0-27 add 1000 after it, lanes 28-31 store
/// nothing.
#[test]
fn an_exec_spilled_beside_a_descriptor_and_restored_after_a_join_runs_its_lanes() {
    for or in [false, true] {
        let mut a = prologue();
        a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
            .op(SPILL_BASE)
            .op([lanes_below(20), sop1(S_AND_SAVEEXEC_B32, 25, VCC_LO)])
            .op(writelane(6, 25, 0))
            .branch(S_CBRANCH_EXECZ, "join")
            .op(add(3, 100))
            .op([sop1(S_MOV_B32, 25, k(0))])
            .label("join")
            .op(readlane(26, 6, 0))
            .op([if or { sop2(S_OR_B32, EXEC_LO, EXEC_LO, 26) } else { sop1(S_MOV_B32, EXEC_LO, 26) }])
            .branch(S_CBRANCH_EXECZ, "end")
            .op(add(3, 1000))
            .op(store(3))
            .label("end")
            .op([S_ENDPGM]);
        let Some(got) = run(&a.done()) else { return };
        let want = expect(32, |l| match l {
            0..20 => Some(1101),
            20..28 => Some(1001),
            _ => None,
        });
        assert_eq!(got, want, "or {or}");
    }
}

/// A uniform loop of three passes over lanes 0-27 spills the descriptor's address to lane 12
/// and EXEC to lane 0 at the top of each pass, then narrows EXEC to the lanes above `4 * pass`
/// for a branch; from the second pass on, the head restores EXEC from lane 0 (`s_mov_b32
/// exec_lo`) and leaves for the end of the program if it is empty (`s_cbranch_execz`); after the
/// loop EXEC is restored from lane 0 again. GFX10: every lane below 28 counts three passes (300)
/// and one for each pass it was above `4 * pass` in.
///
/// With the `s_cbranch_execz` going to the code after the loop instead, a lane whose restored
/// EXEC bit is clear would reach that code with lane 0 as its own last pass left it, while the
/// wave goes on and writes lane 0 again: refused.
#[test]
fn an_exec_spilled_in_a_loop_beside_a_descriptor_is_restored_each_pass() {
    let program = |exit: &'static str| {
        let mut a = prologue();
        a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
            .op([sop1(S_MOV_B32, 20, k(0))])
            .label("head")
            .op([sopc(S_CMP_EQ_U32, 20, k(0))])
            .branch(S_CBRANCH_SCC1, "body")
            .op(readlane(21, 6, 0))
            .op([sop1(S_MOV_B32, EXEC_LO, 21)])
            .branch(S_CBRANCH_EXECZ, exit)
            .label("body")
            .op(SPILL_BASE)
            .op([sop1(S_MOV_B32, 22, EXEC_LO)])
            .op(writelane(6, 22, 0))
            .op(add(10, 100))
            .op([sop2(S_LSHL_B32, 23, 20, k(2)), lanes_above_sgpr(23)])
            .op([sop2(S_AND_B32, EXEC_LO, EXEC_LO, VCC_LO)])
            .branch(S_CBRANCH_EXECZ, "skip")
            .op(add(11, 1))
            .label("skip")
            .op([sop2(S_ADD_U32, 20, 20, k(1)), sopc(S_CMP_LT_U32, 20, k(3))])
            .branch(S_CBRANCH_SCC1, "head")
            .label("out")
            .op(readlane(21, 6, 0))
            .op([sop1(S_MOV_B32, EXEC_LO, 21)])
            .op([0x4A14_1503]) // v_add_nc_u32 v10, v3, v10
            .op([0x4A14_150B]) // v_add_nc_u32 v10, v11, v10
            .op(store(10))
            .label("end")
            .op([S_ENDPGM]);
        a.done()
    };
    assert_refused(&program("out"));
    let Some(got) = run(&program("end")) else { return };
    let want = expect(32, |l| (l < 28).then(|| 1 + 300 + (0..3).filter(|p| l > 4 * p).count() as u32));
    assert_eq!(got, want);
}

/// A loop's counter spilled to lane 0 beside the descriptor's address in lane 12 at the end of
/// each pass, the SGPR then reused; the head takes it back from lane 0 from the second pass on
/// (on the first, lane 0 is not written yet and not read) and leaves after four passes; the code
/// after the loop reads it once more. GFX10: four passes of 10, the count 4 and the texel's 1.
#[test]
fn a_counter_spilled_in_a_loop_beside_a_descriptor_bounds_the_loop() {
    let mut a = prologue();
    a.op([sop1(S_MOV_B32, 25, k(1))])
        .label("head")
        .op([sopc(S_CMP_EQ_U32, 25, k(1))])
        .branch(S_CBRANCH_SCC1, "first")
        .op(readlane(21, 6, 0))
        .branch(S_BRANCH, "counted")
        .label("first")
        .op([sop1(S_MOV_B32, 21, k(0))])
        .label("counted")
        .op([sopc(S_CMP_LT_U32, 21, k(4))])
        .branch(S_CBRANCH_SCC0, "out")
        .op(SPILL_BASE)
        .op([sop2(S_ADD_U32, 21, 21, k(1))])
        .op(writelane(6, 21, 0))
        .op([sop1(S_MOV_B32, 21, k(7)), sop1(S_MOV_B32, 25, k(0))])
        .op(add(10, 10))
        .branch(S_BRANCH, "head")
        .label("out")
        .op(readlane(22, 6, 0))
        .op([add_sgpr(10, 22)])
        .op([0x4A14_1503]) // v_add_nc_u32 v10, v3, v10
        .op(store(10))
        .op([S_ENDPGM]);
    let Some(got) = run(&a.done()) else { return };
    assert_eq!(got, expect(32, |_| Some(40 + 4 + 1)));
}

/// The saved EXECs of two nested EXEC branches spilled beside the descriptor's address (lanes
/// 0 and 1), each from inside the region the outer one opens (the inner one's holds lanes the
/// branch then leaves out), the SGPR clobbered, and each restored from its lane with `s_or_b32
/// exec_lo` at its join. GFX10: lanes 0-9 add 10, 0-19 add 100 and 1000, 0-27 add 2000.
#[test]
fn nested_execs_spilled_beside_a_descriptor_are_restored_at_their_joins() {
    let mut a = prologue();
    a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
        .op(SPILL_BASE)
        .op([lanes_below(20), sop1(S_AND_SAVEEXEC_B32, 25, VCC_LO)])
        .op(writelane(6, 25, 0))
        .op([sop1(S_MOV_B32, 25, k(0))])
        .branch(S_CBRANCH_EXECZ, "outer")
        .op(add(3, 100))
        .op([lanes_below(10), sop1(S_AND_SAVEEXEC_B32, 25, VCC_LO)])
        .op(writelane(6, 25, 1))
        .op([sop1(S_MOV_B32, 25, k(0))])
        .branch(S_CBRANCH_EXECZ, "inner")
        .op(add(3, 10))
        .label("inner")
        .op(readlane(26, 6, 1))
        .op([sop2(S_OR_B32, EXEC_LO, EXEC_LO, 26)])
        .op(add(3, 1000))
        .label("outer")
        .op(readlane(26, 6, 0))
        .op([sop2(S_OR_B32, EXEC_LO, EXEC_LO, 26)])
        .op(add(3, 2000))
        .op(store(3))
        .op([S_ENDPGM]);
    let Some(got) = run(&a.done()) else { return };
    let want = expect(32, |l| match l {
        0..10 => Some(3111),
        10..20 => Some(3101),
        20..28 => Some(2001),
        _ => None,
    });
    assert_eq!(got, want);
}

/// A divergent loop's break mask kept in lane 1 of the VGPR holding the descriptor's address in
/// lane 12: written before the loop, read back at the top of each pass, the lanes whose count ran
/// out ored in, written again and the SGPR clobbered; EXEC loses the broken lanes
/// (`s_andn2_b32`) and gets them back after the loop from lane 1 (`s_or_b32`). GFX10: lane `l`
/// below 28 runs `l + 1` passes and every one of them stores after the loop.
#[test]
fn a_break_mask_spilled_in_a_divergent_loop_restores_the_lanes_it_broke() {
    let mut a = prologue();
    a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
        .op(SPILL_BASE)
        .op([sop1(S_MOV_B32, 26, k(0))])
        .op(writelane(6, 26, 1))
        .op([0x7E18_0300]) // v_mov_b32 v12, v0
        .label("loop")
        .op(readlane(26, 6, 1))
        .op(add(10, 1))
        .op([
            0x7D84_1880, // v_cmp_eq_u32 vcc_lo, 0, v12
            0x4E18_1881, // v_subrev_nc_u32 v12, 1, v12
        ])
        .op([sop2(S_OR_B32, 26, VCC_LO, 26)])
        .op(writelane(6, 26, 1))
        .op([sop2(S_ANDN2_B32, EXEC_LO, EXEC_LO, 26), sop1(S_MOV_B32, 26, k(0))])
        .branch(S_CBRANCH_EXECNZ, "loop")
        .op(readlane(27, 6, 1))
        .op([sop2(S_OR_B32, EXEC_LO, EXEC_LO, 27)])
        .op([0x4A14_1503]) // v_add_nc_u32 v10, v3, v10
        .op(store(10))
        .op([S_ENDPGM]);
    let words = a.done();
    let Some(got) = run(&words) else { return };
    assert_eq!(got, expect(32, |l| (l < 28).then_some(l + 2)));
}

/// A value restored from lane 0 spilled again to lane 5 of another VGPR that also holds the
/// descriptor's address in lane 12, then restored from there to EXEC: the title's programs spill
/// many such restored values. GFX10 as in the single spill: lanes 0-19 1101, 20-27 1001.
#[test]
fn an_exec_respilled_from_one_lane_to_another_is_restored() {
    let mut a = prologue();
    a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
        .op(SPILL_BASE)
        .op(writelane(7, 16, 12))
        .op([lanes_below(20), sop1(S_AND_SAVEEXEC_B32, 25, VCC_LO)])
        .op(writelane(6, 25, 0))
        .op([sop1(S_MOV_B32, 25, k(0))])
        .branch(S_CBRANCH_EXECZ, "join")
        .op(add(3, 100))
        .op(readlane(27, 6, 0))
        .op(writelane(7, 27, 5))
        .op([sop1(S_MOV_B32, 27, k(0))])
        .branch(S_BRANCH, "restore")
        .label("join")
        .op(readlane(27, 6, 0))
        .op(writelane(7, 27, 5))
        .label("restore")
        .op(readlane(28, 7, 5))
        .op([sop1(S_MOV_B32, EXEC_LO, 28)])
        .branch(S_CBRANCH_EXECZ, "end")
        .op(add(3, 1000))
        .op(store(3))
        .label("end")
        .op([S_ENDPGM]);
    let Some(got) = run(&a.done()) else { return };
    let want = expect(32, |l| match l {
        0..20 => Some(1101),
        20..28 => Some(1001),
        _ => None,
    });
    assert_eq!(got, want);
}

/// The single spill and restore of a saved EXEC in 64-lane waves, which the host runs as two
/// 32-lane halves: the pair saved by `s_and_saveexec_b64` spilled to lanes 0 and 1 beside the
/// descriptor's address in lane 12, restored with `s_mov_b64 exec`. The regions cross the
/// halves: lanes below 44 to start, below 36 in the branch. GFX10: lanes 0-35 1101, 36-43 1001.
#[test]
fn a_wave64_exec_spilled_beside_a_descriptor_and_restored_after_a_join_runs_its_lanes() {
    const S_AND_SAVEEXEC_B64: u32 = 0x24;
    const S_MOV_B64: u32 = 4;
    let mut a = prologue();
    a.op([lanes_below(44), sop1(S_AND_SAVEEXEC_B64, 24, VCC_LO)])
        .op(SPILL_BASE)
        .op([lanes_below(36), sop1(S_AND_SAVEEXEC_B64, 26, VCC_LO)])
        .op(writelane(6, 26, 0))
        .op(writelane(6, 27, 1))
        .branch(S_CBRANCH_EXECZ, "join")
        .op(add(3, 100))
        .op([sop1(S_MOV_B64, 26, k(0))])
        .label("join")
        .op(readlane(28, 6, 0))
        .op(readlane(29, 6, 1))
        .op([sop1(S_MOV_B64, EXEC_LO, 28)])
        .branch(S_CBRANCH_EXECZ, "end")
        .op(add(3, 1000))
        .op(store(3))
        .label("end")
        .op([S_ENDPGM]);
    let Some(got) = run_wave(&a.done(), 64, true) else { return };
    let want = expect(64, |l| match l {
        0..36 => Some(1101),
        36..44 => Some(1001),
        _ => None,
    });
    assert_eq!(got, want);
}

/// An EXEC cleared of the lanes that leave a divergent loop (`s_andn2_b32 exec_lo`) whose
/// starting EXEC (lanes below 28) is spilled to lane 0 beside the descriptor's address before the
/// loop, the saved SGPR reused in the loop, and restored from lane 0 after it (`s_or_b32
/// exec_lo`): GFX10 lane `l` below 28 runs `l + 1` passes and stores after the loop.
#[test]
fn an_exec_spilled_before_a_divergent_loop_is_restored_after_it() {
    let mut a = prologue();
    a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
        .op(SPILL_BASE)
        .op([sop1(S_MOV_B32, 22, EXEC_LO)])
        .op(writelane(6, 22, 0))
        .op([0x7E18_0300]) // v_mov_b32 v12, v0
        .label("loop")
        .op(add(10, 1))
        .op([
            0x7D84_1880, // v_cmp_eq_u32 vcc_lo, 0, v12: lanes whose count ran out
            0x4E18_1881, // v_subrev_nc_u32 v12, 1, v12
        ])
        .op([sop2(S_ANDN2_B32, EXEC_LO, EXEC_LO, VCC_LO)])
        .op([sop1(S_MOV_B32, 22, k(0))])
        .branch(S_CBRANCH_EXECNZ, "loop")
        .op(readlane(23, 6, 0))
        .op([sop2(S_OR_B32, EXEC_LO, EXEC_LO, 23)])
        .op([0x4A14_1503]) // v_add_nc_u32 v10, v3, v10
        .op(store(10))
        .op([S_ENDPGM]);
    let Some(got) = run(&a.done()) else { return };
    assert_eq!(got, expect(32, |l| (l < 28).then_some(l + 2)));
}

/// A mask written to lane 0 inside an EXEC branch (lanes below 20) that holds bits of lanes the
/// branch leaves out (`s_not_b32` of EXEC there), read back after the join and anded into EXEC.
/// On GFX10 the wave runs the branch, so after the join every lane reads that mask, and lanes
/// 20-27 store; a lane taking its own path past the branch would read the 0 written before it
/// (Metal stored nothing for lanes 20-27). The join's edges bring different spills, so lane 0
/// is not followed as a spill: beside the descriptor's address the program is refused; alone,
/// the VGPR is read whole (its lane 0 is the host lane 0's, which ran the branch).
#[test]
fn a_mask_spilled_in_a_branch_some_lanes_skip_is_not_read_as_each_lanes_own() {
    const S_NOT_B32: u32 = 7;
    for base in [false, true] {
        let mut a = prologue();
        a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)]);
        if base {
            a.op(SPILL_BASE);
        }
        a.op([sop1(S_MOV_B32, 27, k(0))])
            .op(writelane(6, 27, 0))
            .op([lanes_below(20), sop1(S_AND_SAVEEXEC_B32, 25, VCC_LO)])
            .branch(S_CBRANCH_EXECZ, "join")
            .op([sop1(S_NOT_B32, 27, EXEC_LO)])
            .op(writelane(6, 27, 0))
            .op(add(3, 100))
            .label("join")
            .op([sop2(S_OR_B32, EXEC_LO, EXEC_LO, 25)])
            .op(readlane(28, 6, 0))
            .op([sop2(S_AND_B32, EXEC_LO, EXEC_LO, 28)])
            .branch(S_CBRANCH_EXECZ, "end")
            .op(add(3, 1000))
            .op(store(3))
            .label("end")
            .op([S_ENDPGM]);
        if base {
            assert_refused(&a.done());
            continue;
        }
        let Some(got) = run(&a.done()) else { return };
        assert_eq!(got, expect(32, |l| match l {
            0..20 => None,
            20..28 => Some(1001),
            _ => None,
        }));
    }
}

/// The EXEC a branch saves (lanes below 28) spilled to lane 0 only inside the branch (lanes
/// below 20), where nothing wrote lane 0 before, and restored from it after the join. GFX10: the
/// wave runs the branch, so lanes 20-27, which skip it, read the saved EXEC too and store. Taken
/// as zero on the edge that skips the branch (the first version of this lowering), lanes 20-27
/// stored nothing on Metal: refused, as the title's programs with this shape (two each) are.
#[test]
fn an_exec_spilled_only_inside_a_branch_some_lanes_skip_is_refused() {
    let mut a = prologue();
    a.op([lanes_below(28), sop1(S_AND_SAVEEXEC_B32, 24, VCC_LO)])
        .op(SPILL_BASE)
        .op([lanes_below(20), sop1(S_AND_SAVEEXEC_B32, 25, VCC_LO)])
        .branch(S_CBRANCH_EXECZ, "join")
        .op(writelane(6, 25, 0))
        .op([sop1(S_MOV_B32, 25, k(0))])
        .op(add(3, 100))
        .label("join")
        .op(readlane(26, 6, 0))
        .op([sop2(S_OR_B32, EXEC_LO, EXEC_LO, 26)])
        .op(add(3, 1000))
        .op(store(3))
        .op([S_ENDPGM]);
    assert_refused(&a.done());
}
