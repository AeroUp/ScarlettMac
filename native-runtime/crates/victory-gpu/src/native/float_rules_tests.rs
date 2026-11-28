//! #6 on Metal: the console's float rules for NaN, infinity, zero and denormal operands, through
//! the translation, Apple's Metal Shader Converter and the GPU. Each probe is one VOP3 instruction
//! of our own (`v5 = op(v2, v3, v4)`, `v5` loaded first as the accumulator and the high half a
//! 16-bit result keeps) run over edge operands, 64 lanes a dispatch, against a CPU model of RDNA2
//! in the mode every compute program of the title sets (`COMPUTE_PGM_RSRC1`: IEEE mode off, DX10
//! clamp on, 32-bit denormals flushed, 16-bit ones kept). The model is the one
//! `victory-shader/src/emit/semantics_tests.rs` checks the lowering against on D3D12 (its sources
//! are named there); these check that the host keeps it. Skipped without the Metal Shader
//! Converter.

use super::addr_tests::{CANARY, Fixture, VA_A, dispatch, halves, title, translate_io};
use crate::native::shaders;

/// Pool offsets of the operands (`VA_A`) and of the results (`VA_A + 0x4_0000`).
const SRC: usize = 0x2_0000;
const DST: usize = 0x6_0000;

/// VOP3 modifiers: `abs`/`neg` per source (bit k), `clamp`, the output multiplier (1 `mul:2`,
/// 2 `mul:4`, 3 `div:2`).
#[derive(Clone, Copy, Default)]
struct M {
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
}

const NONE: M = M {
    abs: 0,
    neg: 0,
    clamp: false,
    omod: 0,
};
const CLAMP: M = M {
    abs: 0,
    neg: 0,
    clamp: true,
    omod: 0,
};

/// `v_lshlrev_b32 v1, 2, v0`; `v2..v5` loaded from `s[0:1]` + 0, 256, 512, 768; the instruction;
/// `v5` stored to `s[2:3]`. A compare (`op` below 0x100) writes its mask to `s[6:7]`, which
/// `v_cndmask_b32 v5, 0, 1` reads.
fn program(op: u16, m: M) -> Vec<u32> {
    let compare = op < 0x100;
    let vop3 = [
        0xD400_0000
            | u32::from(op) << 16
            | u32::from(m.clamp) << 15
            | u32::from(m.abs & 7) << 8
            | if compare { 6 } else { 5 },
        u32::from(m.neg & 7) << 29 | u32::from(m.omod & 3) << 27 | 260 << 18 | 259 << 9 | 258,
    ];
    let mut w = vec![0x3402_0082];
    for k in 0..4u32 {
        // global_load_dword v(2+k), v1, s[0:1] offset:256k
        w.extend([0xDC30_8000 | k << 8, (2 + k) << 24 | 1]);
    }
    w.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    w.extend(vop3);
    if compare {
        w.extend([0xD501_0005, 0x0019_0280]); // v_cndmask_b32_e64 v5, 0, 1, s[6:7]
    }
    // global_store_dword v1, v5, s[2:3]; s_endpgm
    w.extend([0xDC70_8000, 0x0002_0501, 0xBF81_0000]);
    w
}

// ---- the model ----

fn fl(x: u32) -> f32 {
    f32::from_bits(x)
}

fn nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

/// A 32-bit denormal as the console's mode reads and writes it: a zero of its sign.
fn ftz(x: u32) -> u32 {
    if x & 0x7F80_0000 == 0 {
        x & 0x8000_0000
    } else {
        x
    }
}

fn d(x: u32) -> f64 {
    f64::from(fl(ftz(x)))
}

fn out64(r: f64) -> u32 {
    if r.abs() < f64::from(f32::MIN_POSITIVE) {
        (r.to_bits() >> 32) as u32 & 0x8000_0000
    } else {
        (r as f32).to_bits()
    }
}

fn mods_in(x: u32, k: usize, m: M) -> u32 {
    let mut x = x;
    if m.abs >> k & 1 != 0 {
        x &= 0x7FFF_FFFF;
    }
    if m.neg >> k & 1 != 0 {
        x ^= 0x8000_0000;
    }
    x
}

/// The multiplier (a -0 product comes out +0), then the clamp to [0, 1], which gives +0 for NaN
/// (DX10 clamp) and for -0.
fn mods_out(r: u32, m: M) -> u32 {
    let mut r = r;
    if m.omod != 0 && !nan(r) {
        r = ftz((fl(r) * [1.0, 2.0, 4.0, 0.5][usize::from(m.omod)]).to_bits());
        if r == 0x8000_0000 {
            r = 0;
        }
    }
    if m.clamp {
        r = if nan(r) || fl(r) <= 0.0 {
            0
        } else if fl(r) >= 1.0 {
            0x3F80_0000
        } else {
            r
        };
    }
    r
}

fn add(a: u32, b: u32) -> u32 {
    out64(d(a) + d(b))
}

fn mul(a: u32, b: u32) -> u32 {
    out64(d(a) * d(b))
}

/// DX9's multiply: a zero operand (a denormal read as one) gives +0 against anything, infinity
/// and NaN included.
fn mul_legacy(a: u32, b: u32) -> u32 {
    if ftz(a) & 0x7FFF_FFFF == 0 || ftz(b) & 0x7FFF_FFFF == 0 {
        0
    } else {
        mul(a, b)
    }
}

fn fma(a: u32, b: u32, c: u32) -> u32 {
    out64(d(a).mul_add(d(b), d(c)))
}

/// The product rounded first: the lowering's `FMad` where the host does not fuse.
fn mad(a: u32, b: u32, c: u32) -> u32 {
    add(mul(a, b), c)
}

fn fma_legacy(a: u32, b: u32, c: u32) -> u32 {
    if ftz(a) & 0x7FFF_FFFF == 0 || ftz(b) & 0x7FFF_FFFF == 0 {
        fma(0, 0, c)
    } else {
        fma(a, b, c)
    }
}

fn mad_legacy(a: u32, b: u32, c: u32) -> u32 {
    if ftz(a) & 0x7FFF_FFFF == 0 || ftz(b) & 0x7FFF_FFFF == 0 {
        add(0, c)
    } else {
        mad(a, b, c)
    }
}

/// One NaN operand gives the other, two the first; -0 orders below +0.
fn min_max(max: bool, a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    match (nan(a), nan(b)) {
        (true, true) | (false, true) => a,
        (true, false) => b,
        _ if fl(a) == fl(b) => {
            if max {
                a & b
            } else {
                a | b
            }
        }
        _ => {
            if (fl(a) > fl(b)) == max {
                a
            } else {
                b
            }
        }
    }
}

/// The median; with a NaN among the operands their minimum.
fn med3(a: u32, b: u32, c: u32) -> u32 {
    if nan(a) || nan(b) || nan(c) {
        return min_max(false, min_max(false, a, b), c);
    }
    let lo = min_max(false, a, b);
    let hi = min_max(true, a, b);
    min_max(true, lo, min_max(false, hi, c))
}

/// The transcendentals (VOP1 opcodes): denormals in and out are zeros of their sign,
/// rcp(±0) = ±inf, rsq(-0) = -inf, sqrt(-0) = -0, log(±0) = -inf, invalid operations NaN.
fn transcendental(op: u16, x: u32) -> u32 {
    let x = ftz(x);
    let f = f64::from(fl(x));
    if nan(x) {
        return x | 0x0040_0000;
    }
    let r = match op {
        0x2A | 0x2B => 1.0 / f,
        0x2E if x == 0x8000_0000 => f64::NEG_INFINITY,
        0x2E => 1.0 / f.sqrt(),
        0x33 => f.sqrt(),
        0x27 if f == 0.0 => f64::NEG_INFINITY,
        0x27 => f.log2(),
        _ => f.exp2(),
    };
    ftz((r as f32).to_bits())
}

/// `v_fract_f32`: never 1.0, NaN for infinities, a NaN passes, a denormal reads as zero.
fn fract(x: u32) -> u32 {
    let x = ftz(x);
    if nan(x) {
        return x;
    }
    if x & 0x7FFF_FFFF == 0x7F80_0000 {
        return 0xFFC0_0000;
    }
    let r = ftz((fl(x) - fl(x).floor()).to_bits());
    if fl(r) >= 1.0 {
        0x3F7F_FFFF
    } else {
        r & 0x7FFF_FFFF
    }
}

/// Float to integer: toward zero, saturating, NaN 0, denormals read as zero.
fn to_int(x: u32, signed: bool) -> u32 {
    let x = ftz(x);
    if nan(x) {
        return 0;
    }
    let f = f64::from(fl(x)).trunc();
    if signed {
        f.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32 as u32
    } else {
        f.clamp(0.0, f64::from(u32::MAX)) as u32
    }
}

/// The 16 float predicates of `v_cmp_*_f32` (`o` ordered, `u` unordered, `n*` true when
/// unordered), of normal operands.
fn compare(op: u16, a: u32, b: u32) -> bool {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let o = fl(a).partial_cmp(&fl(b));
    let ordered = |want: &[std::cmp::Ordering]| o.is_some_and(|o| want.contains(&o));
    match op & 0xF {
        0x0 => false,
        0x1 => ordered(&[Less]),
        0x2 => ordered(&[Equal]),
        0x3 => ordered(&[Less, Equal]),
        0x4 => ordered(&[Greater]),
        0x5 => ordered(&[Less, Greater]),
        0x6 => ordered(&[Greater, Equal]),
        0x7 => o.is_some(),
        0x8 => o.is_none(),
        0x9 => !ordered(&[Greater, Equal]),
        0xA => !ordered(&[Less, Greater]),
        0xB => !ordered(&[Greater]),
        0xC => !ordered(&[Less, Equal]),
        0xD => !ordered(&[Equal]),
        0xE => !ordered(&[Less]),
        _ => true,
    }
}

/// `v_cmp_class_f32` and `_f16`: the bit of the raw operand's class (signalling NaN, quiet NaN,
/// -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf), denormals not flushed.
fn class(x: u32, f16: bool) -> u32 {
    let (sign, exp, quiet) = if f16 {
        (0x8000, 0x7C00, 0x200)
    } else {
        (0x8000_0000, 0x7F80_0000, 0x40_0000)
    };
    let mag = x & (sign - 1);
    let bit = if mag > exp {
        return u32::from(mag & quiet != 0);
    } else if mag == exp {
        9
    } else if mag & exp != 0 {
        8
    } else if mag != 0 {
        7
    } else {
        6
    };
    if x & sign != 0 { 11 - bit } else { bit }
}

fn round_even(m: f64) -> f64 {
    let r = m.round();
    if (r - m).abs() == 0.5 && r % 2.0 != 0.0 {
        r - m.signum()
    } else {
        r
    }
}

/// `x` as an f16, to nearest even.
fn to_f16(x: f64) -> u32 {
    let sign = (x.to_bits() >> 48) as u32 & 0x8000;
    if x.is_nan() {
        return sign | 0x7E00;
    }
    let a = x.abs();
    if a == f64::INFINITY {
        return sign | 0x7C00;
    }
    let (e, m) = if a < 2f64.powi(-14) {
        (0, a * f64::from(1u32 << 24))
    } else {
        let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
        (e + 15, a / 2f64.powi(e - 10))
    };
    let r = round_even(m);
    let bits = if e == 0 {
        r as u32
    } else {
        ((e as u32) << 10) + (r as u32 - 0x400)
    };
    sign | bits.min(0x7C00)
}

fn from_f16(h: u32) -> f64 {
    let (e, m) = (h >> 10 & 0x1F, h & 0x3FF);
    let v = match e {
        0 => f64::from(m) * 2f64.powi(-24),
        _ => f64::from(m | 0x400) * 2f64.powi(e as i32 - 25),
    };
    if h & 0x8000 != 0 { -v } else { v }
}

fn f16_nan(x: u32) -> bool {
    x & 0x7FFF > 0x7C00
}

/// The 16-bit minimum and maximum, denormals kept: as [`min_max`].
fn f16_min_max(max: bool, a: u32, b: u32) -> u32 {
    let (a, b) = (a & 0xFFFF, b & 0xFFFF);
    match (f16_nan(a), f16_nan(b)) {
        (true, true) | (false, true) => a,
        (true, false) => b,
        _ if from_f16(a) == from_f16(b) => {
            if max {
                a & b
            } else {
                a | b
            }
        }
        _ => {
            if (from_f16(a) > from_f16(b)) == max {
                a
            } else {
                b
            }
        }
    }
}

fn f16_med3(a: u32, b: u32, c: u32) -> u32 {
    let mm = f16_min_max;
    if f16_nan(a) || f16_nan(b) || f16_nan(c & 0xFFFF) {
        return mm(false, mm(false, a, b), c);
    }
    mm(true, mm(false, a, b), mm(false, mm(true, a, b), c))
}

/// The f16 clamp: NaN and -0 to +0.
fn f16_clamp(h: u32) -> u32 {
    if f16_nan(h) || h & 0x8000 != 0 {
        0
    } else if h >= 0x3C00 {
        0x3C00
    } else {
        h
    }
}

// ---- inputs ----

const F32_EDGES: &[u32] = &[
    0x0000_0000,
    0x8000_0000, // ±0
    0x0000_0001,
    0x8000_0001, // ±smallest denormal
    0x007F_FFFF,
    0x807F_FFFF, // ±largest denormal
    0x0080_0000,
    0x8080_0000, // ±smallest normal
    0x3F80_0000,
    0xBF80_0000, // ±1
    0x3F00_0000,
    0xBF00_0000, // ±0.5
    0x3FC0_0000,
    0xC020_0000, // 1.5, -2.5
    0x3F7F_FFFF,
    0x3F80_0001, // 1 ∓ 1 ulp
    0x7F7F_FFFF,
    0xFF7F_FFFF, // ±largest
    0x7F80_0000,
    0xFF80_0000, // ±infinity
    0x7FC0_0000,
    0xFFC0_0000,
    0x7F80_0001, // quiet, negative quiet, signalling NaN
    0xFFC0_1000, // a negative quiet NaN with a payload (#6's empty-record mark)
    0x1E3C_E508, // 1e-20 (squares to a denormal)
    0x5F80_0000, // 2^64 (squares past the largest)
    0x3DCC_CCCD, // 0.1
    0xC2C8_0000, // -100
];

const F32_THIRD: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x3F80_0000,
    0xBF80_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_1000,
    0xFF7F_FFFF,
];

const F16_EDGES: &[u32] = &[
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3800, 0xC100, 0x3BFF, 0x7BFF,
    0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFE08,
];

/// Each operand pair, the third and fourth operands `c` and `d`.
fn pairs(xs: &[u32], ys: &[u32], c: u32, d: u32) -> Vec<[u32; 4]> {
    xs.iter()
        .flat_map(|&a| ys.iter().map(move |&b| [a, b, c, d]))
        .collect()
}

fn triples(xs: &[u32], ys: &[u32], zs: &[u32]) -> Vec<[u32; 4]> {
    let mut v = Vec::new();
    for &a in xs {
        for &b in ys {
            for &c in zs {
                v.push([a, b, c, c]);
            }
        }
    }
    v
}

// ---- comparing ----

#[derive(Clone, Copy)]
enum Kind {
    /// Bit for bit.
    Int,
    /// An f32: NaN matches any NaN, zeros and infinities exactly, finite values within `ulps`.
    F32 { ulps: u32 },
    /// An f32 whose finite values may also be within `abs` (log2 near 1).
    Abs { abs: f32 },
    /// An f16 in the low half as `F32` (no ULPs); the high half bit for bit.
    F16,
}

fn close(sign: u32, exp_mask: u32, a: u32, b: u32, ulps: u32) -> bool {
    let mag = sign - 1;
    let nan = |x: u32| x & mag > exp_mask;
    if nan(a) || nan(b) {
        return nan(a) && nan(b);
    }
    if a & mag == 0 || b & mag == 0 || a & mag == exp_mask || b & mag == exp_mask {
        return a == b;
    }
    a & sign == b & sign && (a & mag).abs_diff(b & mag) <= ulps
}

fn same(kind: Kind, want: u32, got: u32) -> bool {
    match kind {
        Kind::Int => want == got,
        Kind::F32 { ulps } => close(0x8000_0000, 0x7F80_0000, want, got, ulps),
        Kind::Abs { abs } => {
            close(0x8000_0000, 0x7F80_0000, want, got, 4)
                || !nan(want)
                    && !nan(got)
                    && want & 0x7FFF_FFFF < 0x7F80_0000
                    && got & 0x7FFF_FFFF < 0x7F80_0000
                    && (fl(want) - fl(got)).abs() <= abs
        }
        Kind::F16 => {
            want >> 16 == got >> 16 && close(0x8000, 0x7C00, want & 0xFFFF, got & 0xFFFF, 0)
        }
    }
}

/// One instruction over its inputs; `model` gives the results the hardware may write (more than
/// one where the lowering's choice is not in question, as fused or not).
struct Probe {
    name: String,
    op: u16,
    m: M,
    inputs: Vec<[u32; 4]>,
    kind: Kind,
    model: Box<dyn Fn([u32; 4]) -> Vec<u32>>,
}

fn probe(
    name: impl Into<String>,
    op: u16,
    m: M,
    inputs: Vec<[u32; 4]>,
    kind: Kind,
    model: impl Fn([u32; 4]) -> Vec<u32> + 'static,
) -> Probe {
    Probe {
        name: name.into(),
        op,
        m,
        inputs,
        kind,
        model: Box::new(model),
    }
}

/// Runs the probes on the GPU; panics with the first few differing inputs of each failing one.
fn check(probes: &[Probe]) {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let [dst_lo, dst_hi] = halves(VA_A + (DST - SRC) as u64);
    let [src_lo, src_hi] = halves(VA_A);
    let mut failures = Vec::new();
    for p in probes {
        let cs = translate_io(&program(p.op, p.m), 4, 0, 64, shaders::options());
        let mut bad = Vec::new();
        for chunk in p.inputs.chunks(64) {
            for k in 0..4 {
                let mut col: Vec<u32> = chunk.iter().map(|i| i[k]).collect();
                col.resize(64, 0);
                f.put(SRC + 256 * k, &col);
            }
            f.put(DST, &[CANARY; 64]);
            dispatch(&mut t, &cs, &[src_lo, src_hi, dst_lo, dst_hi], false);
            assert_eq!(
                t.shaders.refusals(),
                0,
                "{}: {:?}",
                p.name,
                t.shaders.refused
            );
            let got = f.get(DST, 64);
            for (i, &g) in chunk.iter().zip(&got) {
                let want = (p.model)(*i);
                if !want.iter().any(|&w| same(p.kind, w, g)) {
                    bad.push(format!(
                        "  {:08x?} -> {g:#010x}, want {:#010x}",
                        &i[..3],
                        want[0]
                    ));
                }
            }
        }
        if !bad.is_empty() {
            failures.push(format!(
                "{} ({} of {}):\n{}",
                p.name,
                bad.len(),
                p.inputs.len(),
                bad[..bad.len().min(6)].join("\n")
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

const EXACT: Kind = Kind::F32 { ulps: 0 };
const FUSED: Kind = Kind::F32 { ulps: 1 };

#[test]
fn legacy_multiplies_give_zero_for_a_zero_factor_against_infinity_and_nan() {
    let ab = pairs(F32_EDGES, F32_EDGES, 0, 0);
    let abc = triples(F32_EDGES, F32_EDGES, F32_THIRD);
    let mut p = vec![
        probe("v_mul_legacy_f32", 0x107, NONE, ab.clone(), EXACT, |i| {
            vec![mul_legacy(i[0], i[1])]
        }),
        probe("v_fma_legacy_f32", 0x140, NONE, abc.clone(), FUSED, |i| {
            vec![fma_legacy(i[0], i[1], i[2]), mad_legacy(i[0], i[1], i[2])]
        }),
        // VOP3 v_fmac_legacy_f32 accumulates into v5 = d (= c here).
        probe("v_fmac_legacy_f32", 0x106, NONE, abc, FUSED, |i| {
            vec![fma_legacy(i[0], i[1], i[3]), mad_legacy(i[0], i[1], i[3])]
        }),
    ];
    for m in [
        CLAMP,
        M { neg: 1, ..NONE },
        M { abs: 2, ..NONE },
        M { omod: 1, ..NONE },
    ] {
        p.push(probe(
            "v_mul_legacy_f32 (modifiers)",
            0x107,
            m,
            ab.clone(),
            EXACT,
            move |i| {
                vec![mods_out(
                    mul_legacy(mods_in(i[0], 0, m), mods_in(i[1], 1, m)),
                    m,
                )]
            },
        ));
    }
    check(&p);
}

#[test]
fn the_clamp_gives_zero_for_nan_and_the_multiplier_keeps_nan() {
    let ab = pairs(F32_EDGES, F32_EDGES, 0, 0);
    let abc = triples(F32_EDGES, &F32_EDGES[..12], F32_THIRD);
    let mut p = Vec::new();
    for omod in 0..4u8 {
        for clamp in [false, true] {
            let m = M {
                clamp,
                omod,
                ..NONE
            };
            if !clamp && omod == 0 {
                continue;
            }
            let x2 =
                move |f: fn(u32, u32) -> u32| move |i: [u32; 4]| vec![mods_out(f(i[0], i[1]), m)];
            p.push(probe(
                format!("v_mul_f32 omod {omod} clamp {clamp}"),
                0x108,
                m,
                ab.clone(),
                EXACT,
                x2(mul),
            ));
            p.push(probe(
                format!("v_add_f32 omod {omod} clamp {clamp}"),
                0x103,
                m,
                ab.clone(),
                EXACT,
                x2(add),
            ));
            p.push(probe(
                format!("v_fma_f32 omod {omod} clamp {clamp}"),
                0x14B,
                m,
                abc.clone(),
                FUSED,
                move |i| {
                    vec![
                        mods_out(fma(i[0], i[1], i[2]), m),
                        mods_out(mad(i[0], i[1], i[2]), m),
                    ]
                },
            ));
        }
    }
    // `v_max_f32 x, x clamp`: saturate.
    let xx = F32_EDGES.iter().map(|&x| [x, x, 0, 0]).collect();
    p.push(probe(
        "v_max_f32 x, x clamp",
        0x110,
        CLAMP,
        xx,
        EXACT,
        |i| vec![mods_out(min_max(true, i[0], i[0]), CLAMP)],
    ));
    // The clamp of a transcendental's NaN (sqrt and log of negatives) and infinity.
    for op in [0x2Au16, 0x2E, 0x33, 0x27, 0x25] {
        let xs = pairs(F32_EDGES, &[0], 0, 0);
        p.push(probe(
            format!("VOP1 {op:#x} clamp"),
            0x180 + op,
            CLAMP,
            xs,
            Kind::F32 { ulps: 4 },
            move |i| vec![mods_out(transcendental(op, i[0]), CLAMP)],
        ));
    }
    check(&p);
}

#[test]
fn min_max_and_med3_take_the_number_over_nan() {
    let ab = pairs(F32_EDGES, F32_EDGES, 0, 0);
    let abc = triples(F32_EDGES, F32_EDGES, F32_THIRD);
    let max3 = |i: [u32; 4]| vec![min_max(true, min_max(true, i[0], i[1]), i[2])];
    let min3 = |i: [u32; 4]| vec![min_max(false, min_max(false, i[0], i[1]), i[2])];
    // Of mixed zeros the median's sign is the host's (`MED3_ZERO` in the semantics tests).
    let mixed = |i: [u32; 4]| {
        i[..3].iter().any(|&x| ftz(x) == 0) && i[..3].iter().any(|&x| ftz(x) == 0x8000_0000)
    };
    let p = vec![
        probe("v_max_f32", 0x110, NONE, ab.clone(), EXACT, |i| {
            vec![min_max(true, i[0], i[1])]
        }),
        probe("v_min_f32", 0x10F, NONE, ab, EXACT, |i| {
            vec![min_max(false, i[0], i[1])]
        }),
        probe("v_max3_f32", 0x154, NONE, abc.clone(), EXACT, max3),
        probe("v_min3_f32", 0x151, NONE, abc.clone(), EXACT, min3),
        probe("v_med3_f32", 0x157, NONE, abc, EXACT, move |i| {
            let r = med3(i[0], i[1], i[2]);
            if mixed(i) && r & 0x7FFF_FFFF == 0 {
                vec![0, 0x8000_0000]
            } else {
                vec![r]
            }
        }),
    ];
    check(&p);
}

#[test]
fn f16_min_max_and_med3_take_the_number_over_nan() {
    let hi = 0x5A5A_0000;
    let ab: Vec<[u32; 4]> = pairs(F16_EDGES, F16_EDGES, 0x3C00, hi);
    let abc: Vec<[u32; 4]> = triples(
        F16_EDGES,
        F16_EDGES,
        &[0x0000, 0x8000, 0x3C00, 0x7C00, 0xFC00, 0x7E00, 0xFE08],
    )
    .into_iter()
    .map(|[a, b, c, _]| [a, b, c, hi])
    .collect();
    let keep = move |r: u32| hi | r;
    let mixed = |i: [u32; 4]| i[..3].contains(&0) && i[..3].contains(&0x8000);
    let p = vec![
        probe("v_max_f16", 0x139, NONE, ab.clone(), Kind::F16, move |i| {
            vec![keep(f16_min_max(true, i[0], i[1]))]
        }),
        probe("v_min_f16", 0x13A, NONE, ab.clone(), Kind::F16, move |i| {
            vec![keep(f16_min_max(false, i[0], i[1]))]
        }),
        probe("v_max_f16 clamp", 0x139, CLAMP, ab, Kind::F16, move |i| {
            vec![keep(f16_clamp(f16_min_max(true, i[0], i[1])))]
        }),
        probe(
            "v_max3_f16",
            0x354,
            NONE,
            abc.clone(),
            Kind::F16,
            move |i| vec![keep(f16_min_max(true, f16_min_max(true, i[0], i[1]), i[2]))],
        ),
        probe(
            "v_min3_f16",
            0x351,
            NONE,
            abc.clone(),
            Kind::F16,
            move |i| {
                vec![keep(f16_min_max(
                    false,
                    f16_min_max(false, i[0], i[1]),
                    i[2],
                ))]
            },
        ),
        probe("v_med3_f16", 0x357, NONE, abc, Kind::F16, move |i| {
            let r = f16_med3(i[0], i[1], i[2]);
            if mixed(i) && r & 0x7FFF == 0 {
                vec![keep(0), keep(0x8000)]
            } else {
                vec![keep(r)]
            }
        }),
    ];
    check(&p);
}

#[test]
fn reciprocals_roots_and_logs_of_zeros_denormals_and_specials() {
    let xs = pairs(F32_EDGES, &[0], 0, 0);
    let mut p: Vec<Probe> = [0x2Au16, 0x2B, 0x2E, 0x33, 0x27, 0x25]
        .into_iter()
        .map(|op| {
            let kind = if op == 0x27 {
                Kind::Abs {
                    abs: 1.0 / 2_097_152.0,
                }
            } else {
                Kind::F32 { ulps: 4 }
            };
            probe(
                format!("VOP1 {op:#x}"),
                0x180 + op,
                NONE,
                xs.clone(),
                kind,
                move |i| vec![transcendental(op, i[0])],
            )
        })
        .collect();
    p.push(probe("v_fract_f32", 0x1A0, NONE, xs.clone(), EXACT, |i| {
        vec![fract(i[0])]
    }));
    p.push(probe(
        "v_cvt_u32_f32",
        0x187,
        NONE,
        xs.clone(),
        Kind::Int,
        |i| vec![to_int(i[0], false)],
    ));
    p.push(probe(
        "v_cvt_i32_f32",
        0x188,
        NONE,
        xs.clone(),
        Kind::Int,
        |i| vec![to_int(i[0], true)],
    ));
    // v_cvt_f16_f32 keeps the destination's high half.
    p.push(probe(
        "v_cvt_f16_f32",
        0x18A,
        NONE,
        pairs(F32_EDGES, &[0], 0, 0x5A5A_1234),
        Kind::F16,
        |i| vec![i[3] & 0xFFFF_0000 | to_f16(d(i[0]))],
    ));
    check(&p);
}

#[test]
fn compares_and_class_tests_see_nan() {
    let normal: Vec<u32> = F32_EDGES.iter().copied().filter(|&x| ftz(x) == x).collect();
    let ab = pairs(&normal, &normal, 0, 0);
    let mut p: Vec<Probe> = (1..15u16)
        .map(|op| {
            probe(
                format!("v_cmp {op:#x}"),
                op,
                NONE,
                ab.clone(),
                Kind::Int,
                move |i| vec![u32::from(compare(op, i[0], i[1]))],
            )
        })
        .collect();
    // Each class alone, and the masks a NaN or infinity test uses.
    let masks = [1, 2, 3, 0x204, 0x3FC, 0x3FF, 0x40, 0x80, 0x100, 0x200];
    let f32s: Vec<[u32; 4]> = F32_EDGES
        .iter()
        .flat_map(|&x| masks.map(|m| [x, m, 0, 0]))
        .collect();
    let f16s: Vec<[u32; 4]> = F16_EDGES
        .iter()
        .flat_map(|&x| masks.map(|m| [0xABCD_0000 | x, m, 0, 0]))
        .collect();
    p.push(probe(
        "v_cmp_class_f32",
        0x088,
        NONE,
        f32s,
        Kind::Int,
        |i| vec![i[1] >> class(i[0], false) & 1],
    ));
    p.push(probe(
        "v_cmp_class_f16",
        0x08F,
        NONE,
        f16s,
        Kind::Int,
        |i| vec![i[1] >> class(i[0] & 0xFFFF, true) & 1],
    ));
    check(&p);
}
