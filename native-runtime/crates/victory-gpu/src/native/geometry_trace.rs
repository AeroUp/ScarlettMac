//! `VICTORY_DUMP_VERTICES=<dir>,<draws>[,<seconds>]`: what a run's next draws hold, dumped from
//! `seconds` after the first draw, read the way the input assembler reads it.
//!
//! A mesh drawn as long thin strings or a field of scattered shards is a mesh whose *vertices* are
//! wrong, and the first question is whether the data is wrong or what the host says about it: the
//! layouts and strides, the addresses the views resolve to, the index width. This writes, for one
//! draw a file, the pipeline's input layout and the strides its streams step by (`Graphics::
//! stream_strides`, the rule the Metal vertex descriptor is built with), the bound views and their
//! capacities (`view size / stride`), each element's values for the first vertices, and the range
//! each channel covers over the first 4,096 vertices (with how many of them are not finite or are
//! absurdly large), then the index buffer's values and how many of them name a vertex past every
//! stream's capacity. Title memory is read by title address, the way the title has it now: the data
//! a GPU side copy or a wrong resolution would replace shows in the file as it is.
//!
//! Off unless `VICTORY_DUMP_VERTICES` is set (a run writes a few files; the format is text).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use victory_backend::resources::{DrawCall, VertexFormat};

use super::title::TargetFormats;

/// Where the dumps go, how many draws, and how many seconds after the first draw to start
/// (`None`: off).
pub(crate) fn config() -> Option<(PathBuf, usize, u64)> {
    static CONFIG: std::sync::OnceLock<Option<(PathBuf, usize, u64)>> = std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let v = std::env::var_os("VICTORY_DUMP_VERTICES")?.to_string_lossy().into_owned();
            let mut parts = v.split(',');
            let dir = PathBuf::from(parts.next().filter(|s| !s.is_empty())?);
            let draws = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            let after = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            std::fs::create_dir_all(&dir).ok()?;
            Some((dir, draws, after))
        })
        .clone()
}

/// What one draw bound: the pipeline and its list state, as `glue` has them at the draw.
pub(crate) struct Bound<'a> {
    pub pipeline: usize,
    /// The console's `IASetPrimitiveTopology` value, and what the host draws it as.
    pub topology: (u32, Option<&'static str>),
    pub strides: &'a [u32],
    pub call: DrawCall,
    pub targets: &'a TargetFormats,
    pub vertex_views: Vec<(usize, u64, u32)>,
    pub index_view: Option<(u64, u32, bool)>,
}

/// Writes draw `n` (`dir/draw<n>.txt`) and logs where. `None` when the pipeline's state is not
/// decoded yet (a guessed pipeline, or one the title built without an input layout).
pub(crate) fn dump(dir: &Path, n: usize, b: &Bound<'_>) {
    let Some((stages, graphics)) = crate::pipeline::native_stages(b.pipeline as crate::object::This) else {
        return;
    };
    let Some(g) = graphics else { return };
    let shaders: Vec<String> = stages.iter().map(|(stage, s)| format!("{stage} {}", s.hash)).collect();
    let g = g.with_strides(b.strides);
    let streams = g.stream_strides();
    let inputs = g.inputs().to_vec();
    let mut out = String::new();
    let _ = writeln!(out, "draw {n}");
    let _ = writeln!(out, "call {:?}", b.call);
    let _ = writeln!(
        out,
        "topology {} ({}) shaders {shaders:?}",
        b.topology.0,
        b.topology.1.unwrap_or("not drawn: refused")
    );
    let _ = writeln!(
        out,
        "targets {:?}",
        b.targets.color.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>()
    );
    let _ = writeln!(out, "strides (bound) {:?}", b.strides);
    let _ = writeln!(out, "streams (slot, stride, per instance) {streams:?}");
    for e in &inputs {
        let _ = writeln!(
            out,
            "  element slot {} offset {} format {:?} ({}){}",
            e.slot,
            e.offset,
            super::state::vertex_format(e.format),
            e.format,
            if e.per_instance { " (per instance)" } else { "" }
        );
    }
    // Each bound view, its capacity, and each element's values over the first vertices.
    for &(slot, addr, size) in &b.vertex_views {
        let Some(&(_, stride, per_instance)) = streams.iter().find(|s| s.0 == slot) else {
            let _ = writeln!(out, "view slot {slot} {addr:#x}+{size}: no stream (unused)");
            continue;
        };
        let capacity = if stride > 0 { u64::from(size) / stride as u64 } else { 0 };
        let _ = writeln!(
            out,
            "view slot {slot} {addr:#x}+{size} stride {stride}{} vertices {capacity}",
            if per_instance { " (per instance)" } else { "" }
        );
        let shown = capacity.min(8) as usize;
        let probe = capacity.min(4096) as usize;
        let bytes = super::textures::read_memory(addr as *const u8, probe * stride);
        for (k, e) in inputs.iter().enumerate().filter(|(_, e)| e.slot as usize == slot) {
            let Some(format) = super::state::vertex_format(e.format) else { continue };
            let mut stats = Stats::default();
            let mut first = Vec::new();
            for v in 0..probe {
                let at = v * stride + e.offset as usize;
                let Some(field) = bytes.get(at..at + format.byte_size()) else { continue };
                stats.add(format, field);
                if v < shown {
                    first.push(describe(format, field));
                }
            }
            let _ = writeln!(out, "  element {k} {format:?} over {probe} vertices: {stats}");
            let _ = writeln!(out, "    first {shown}: {}", first.join(" | "));
        }
    }
    // The indices, at the width the host bound.
    if let Some((addr, size, wide)) = b.index_view {
        let width = if wide { 4usize } else { 2 };
        let probe = (size as usize / width).min(4096);
        let bytes = super::textures::read_memory(addr as *const u8, probe * width);
        let cap = |slot: usize| {
            let s = streams.iter().find(|s| s.0 == slot)?.1;
            let (_, _, size) = *b.vertex_views.iter().find(|v| v.0 == slot)?;
            (s > 0).then(|| u64::from(size) / s as u64)
        };
        let main = cap(0);
        let mut max = 0u64;
        let mut over = 0usize;
        let mut first = Vec::new();
        for k in 0..probe {
            let at = k * width;
            let Some(field) = bytes.get(at..at + width) else { break };
            let i = if wide {
                u64::from(u32::from_le_bytes(field.try_into().expect("4 bytes")))
            } else {
                u64::from(u16::from_le_bytes(field.try_into().expect("2 bytes")))
            };
            max = max.max(i);
            if main.is_some_and(|c| i >= c) {
                over += 1;
            }
            if k < 16 {
                first.push(i.to_string());
            }
        }
        let _ = writeln!(
            out,
            "index {addr:#x}+{size} wide {wide}: {probe} read, max {max}, slot 0 capacity {main:?}, \
             {over} name a vertex past it; first: {}",
            first.join(" ")
        );
    }
    let path = dir.join(format!("draw{n}.txt"));
    match std::fs::write(&path, out) {
        Ok(()) => tracing::info!(path = %path.display(), "a draw's vertex and index data dumped"),
        Err(e) => tracing::warn!(path = %path.display(), error = %e, "a draw's data was not dumped"),
    }
}

/// What one element's channel values cover over the vertices read.
#[derive(Default)]
pub(crate) struct Stats {
    seen: usize,
    min: Vec<f64>,
    max: Vec<f64>,
    /// Values that are not finite (floats), or that no vertex position plausibly holds.
    not_finite: usize,
    huge: usize,
}

impl Stats {
    pub(crate) fn add(&mut self, format: VertexFormat, b: &[u8]) {
        self.seen += 1;
        for (c, v) in channels(format, b).into_iter().enumerate() {
            if self.min.len() <= c {
                self.min.push(f64::INFINITY);
                self.max.push(f64::NEG_INFINITY);
            }
            if !v.is_finite() {
                self.not_finite += 1;
                continue;
            }
            if v.abs() > 1e6 {
                self.huge += 1;
            }
            self.min[c] = self.min[c].min(v);
            self.max[c] = self.max[c].max(v);
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (c, (lo, hi)) in self.min.iter().zip(&self.max).enumerate() {
            if c > 0 {
                write!(f, " ")?;
            }
            write!(f, "c{c} {lo:.6}..{hi:.6}")?;
        }
        write!(f, " ({} read, {} not finite, {} over 1e6)", self.seen, self.not_finite, self.huge)
    }
}

/// One element's channel values as numbers: floats as themselves, normalized integers scaled,
/// plain integers as they are.
fn channels(format: VertexFormat, b: &[u8]) -> Vec<f64> {
    let f32_at = |k: usize| f64::from(f32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().expect("4 bytes")));
    let half = |k: usize| {
        let bits = u16::from_le_bytes([b[k * 2], b[k * 2 + 1]]);
        f64::from(f16_to_f32(bits))
    };
    let u16_at = |k: usize| u16::from_le_bytes([b[k * 2], b[k * 2 + 1]]);
    let i16_at = |k: usize| i16::from_le_bytes([b[k * 2], b[k * 2 + 1]]);
    let unorm = |v: u16| f64::from(v) / 65535.0;
    let snorm = |v: i16| (f64::from(v) / 32767.0).max(-1.0);
    use VertexFormat::*;
    match format {
        Float => vec![f32_at(0)],
        Float2 => (0..2).map(f32_at).collect(),
        Float3 => (0..3).map(f32_at).collect(),
        Float4 => (0..4).map(f32_at).collect(),
        Half => vec![half(0)],
        Half2 => (0..2).map(half).collect(),
        Half4 => (0..4).map(half).collect(),
        Unorm16 => vec![unorm(u16_at(0))],
        Unorm16x2 => (0..2).map(|k| unorm(u16_at(k))).collect(),
        Unorm16x4 => (0..4).map(|k| unorm(u16_at(k))).collect(),
        Snorm16 => vec![snorm(i16_at(0))],
        Snorm16x2 => (0..2).map(|k| snorm(i16_at(k))).collect(),
        Snorm16x4 => (0..4).map(|k| snorm(i16_at(k))).collect(),
        Uint | Uint2 | Uint3 | Uint4 | Sint | Sint2 | Sint3 | Sint4 => {
            let n = match format {
                Uint | Sint => 1,
                Uint2 | Sint2 => 2,
                Uint3 | Sint3 => 3,
                _ => 4,
            };
            let signed = matches!(format, Sint | Sint2 | Sint3 | Sint4);
            (0..n)
                .map(|k| {
                    let v = u32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().expect("4 bytes"));
                    if signed {
                        f64::from(v as i32)
                    } else {
                        f64::from(v)
                    }
                })
                .collect()
        }
        Uint16 | Uint16x2 | Uint16x4 => (0..format.byte_size() / 2).map(|k| f64::from(u16_at(k))).collect(),
        Sint16 | Sint16x2 | Sint16x4 => (0..format.byte_size() / 2).map(|k| f64::from(i16_at(k))).collect(),
        Unorm8 | Unorm8x2 | Unorm8x4 => (0..format.byte_size()).map(|k| f64::from(b[k]) / 255.0).collect(),
        Snorm8 | Snorm8x2 | Snorm8x4 => {
            (0..format.byte_size()).map(|k| (f64::from(b[k] as i8) / 127.0).max(-1.0)).collect()
        }
        Uint8 | Uint8x2 | Uint8x4 => (0..format.byte_size()).map(|k| f64::from(b[k])).collect(),
        Sint8 | Sint8x2 | Sint8x4 => (0..format.byte_size()).map(|k| f64::from(b[k] as i8)).collect(),
        Unorm1010102 | Snorm1010102 => {
            let w = u32::from_le_bytes(b[..4].try_into().expect("4 bytes"));
            let signed = format == Snorm1010102;
            let ten = |v: u32| {
                if signed {
                    let x = ((v << 22) as i32) >> 22;
                    (f64::from(x) / 511.0).max(-1.0)
                } else {
                    f64::from(v) / 1023.0
                }
            };
            let two = |v: u32| {
                if signed {
                    let x = ((v << 30) as i32) >> 30;
                    (f64::from(x) / 1.0).max(-1.0)
                } else {
                    f64::from(v) / 3.0
                }
            };
            vec![ten(w & 0x3ff), ten((w >> 10) & 0x3ff), ten((w >> 20) & 0x3ff), two((w >> 30) & 0x3)]
        }
        FloatRG11B10 => {
            let w = u32::from_le_bytes(b[..4].try_into().expect("4 bytes"));
            let f11 = |v: u32| {
                // 5 exponent bits, 6 mantissa bits, no sign.
                let e = (v >> 6) & 0x1f;
                let m = v & 0x3f;
                let f = if e == 0 {
                    f32::from(m as u16) / 64.0 * 2f32.powi(-14)
                } else if e == 0x1f {
                    f32::INFINITY
                } else {
                    (1.0 + f32::from(m as u16) / 64.0) * 2f32.powi(e as i32 - 15)
                };
                f64::from(f)
            };
            vec![f11(w & 0x7ff), f11((w >> 11) & 0x7ff), f11((w >> 22) & 0x3ff)]
        }
    }
}

fn describe(format: VertexFormat, b: &[u8]) -> String {
    channels(format, b).iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")
}

/// `f16` bits to `f32` (the usual conversion, subnormals and infinities included).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x3ff);
    let out = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize it.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(out)
}
