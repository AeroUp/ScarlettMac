//! Differential harness for the console texture layout and detile path.
//!
//! Reads a reference written by `tools/xg-layout/xg_layout_ref.c` on a Windows development PC
//! (through the Xbox Series X|S `xg_xs` library's `XGCreateTexture2DComputer`, `GetResourceLayout`,
//! `GetTexelElementOffsetBytes`, `CopyIntoSubresource`/`CopyFromSubresource` and
//! `GetResourceTiling`) and compares it, field by field, with [`victory_formats::texture`].
//!
//! ```text
//! cargo run -p victory-formats --example xg_layout_diff -- <reference>
//! cargo run -p victory-formats --example xg_layout_diff -- --pipe-config=3,4 <reference>
//! cargo run -p victory-formats --example xg_layout_diff -- --find-pipe-config <reference>
//! cargo run -p victory-formats --example xg_layout_diff -- --self-test
//! cargo run -p victory-formats --example xg_layout_diff -- --emit-self=<file>
//! ```
//!
//! The reference file is generated data and stays outside the repository. `--self-test` checks
//! this file's parser, model and writer against themselves (it cannot show that the model matches
//! the console; only a reference from the Windows library can).
//!
//! Coordinate units: texel queries use element coordinates, as `Layout::offset` does. For a
//! block-compressed format one element is a 4x4 block.
//!
//! Exit codes: 0 every compared field matched, 1 a mismatch, 2 the reference could not be read.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::process::ExitCode;

use victory_formats::texture::{Dimension, PipeConfig, Swizzle, Texture, TextureDesc};

const TILE: u64 = 64 << 10;
const EMPTY_PATTERN_SEED: u64 = 1;
const INTO_TILED_SEED: u64 = 2;
const INTO_LINEAR_SEED: u64 = 3;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(value) = args.iter().find_map(|a| a.strip_prefix("--emit-self=")) {
        let text = emit_model_reference(&self_cases());
        return match std::fs::write(value, &text) {
            Ok(()) => {
                println!(
                    "wrote {} bytes of self-test reference to {value}",
                    text.len()
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("cannot write {value}: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if args.iter().any(|a| a == "--self-test") {
        return match self_test() {
            Ok(()) => {
                println!("self-test ok");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("self-test failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let mut config = PipeConfig::DEFAULT;
    let mut find = false;
    let mut max_errors = 60usize;
    let mut path = None;
    for arg in &args {
        if let Some(value) = arg.strip_prefix("--pipe-config=") {
            let Some((packers, pipes)) = value.split_once(',') else {
                eprintln!("--pipe-config takes <packers_log2>,<pipes_log2>");
                return ExitCode::from(2);
            };
            let (Ok(packers), Ok(pipes)) = (packers.parse(), pipes.parse()) else {
                eprintln!("--pipe-config takes two numbers");
                return ExitCode::from(2);
            };
            config = PipeConfig {
                packers_log2: packers,
                pipes_log2: pipes,
            };
        } else if let Some(value) = arg.strip_prefix("--max-errors=") {
            let Ok(value) = value.parse() else {
                eprintln!("--max-errors takes a number");
                return ExitCode::from(2);
            };
            max_errors = value;
        } else if arg == "--find-pipe-config" {
            find = true;
        } else if arg.starts_with('-') {
            eprintln!("unknown switch {arg}");
            return ExitCode::from(2);
        } else {
            path = Some(arg.clone());
        }
    }
    let Some(path) = path else {
        eprintln!("usage: xg_layout_diff [--pipe-config=p,p | --find-pipe-config] <reference>");
        return ExitCode::from(2);
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let reference = match Reference::parse(&text) {
        Ok(reference) => reference,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    if find {
        return find_pipe_config(&reference);
    }
    let mut report = compare(&reference, config, false);
    report.max = max_errors;
    report.print(&reference.hardware);
    if report.mismatches == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

// ---------------------------------------------------------------- reference file

#[derive(Debug, Default)]
struct Fields(BTreeMap<String, String>);

impl Fields {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.parse().ok()
    }

    fn u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.parse().ok()
    }
}

#[derive(Debug, Default)]
struct RefCase {
    name: String,
    desc: Fields,
    /// `None` when the line is missing; `true` for `status ok`.
    ok: Option<bool>,
    result: Fields,
    planes: Vec<Fields>,
    mips: Vec<Fields>,
    texels: Vec<Fields>,
    copies: Vec<Fields>,
    tiling: Option<Fields>,
}

#[derive(Debug, Default)]
struct Reference {
    hardware: String,
    cases: Vec<RefCase>,
}

impl Reference {
    fn parse(text: &str) -> Result<Reference, String> {
        let mut lines = text.lines().enumerate();
        let (_, first) = lines.next().ok_or("empty reference")?;
        if first.trim() != "xg-layout-ref 1" {
            return Err(format!(
                "first line is not `xg-layout-ref 1`: {:?}",
                first.trim()
            ));
        }
        let mut reference = Reference::default();
        for (number, line) in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tokens = line.split_whitespace();
            let keyword = tokens.next().unwrap_or_default();
            let tokens: Vec<&str> = tokens.collect();
            let fields = |tokens: &[&str]| -> Result<Fields, String> {
                let mut fields = Fields::default();
                for token in tokens {
                    let (key, value) = token.split_once('=').ok_or_else(|| {
                        format!("line {}: `{token}` is not key=value", number + 1)
                    })?;
                    fields.0.insert(key.to_owned(), value.to_owned());
                }
                Ok(fields)
            };
            match keyword {
                "hw" => {
                    reference.hardware = tokens
                        .first()
                        .map(|s| (*s).to_owned())
                        .ok_or_else(|| format!("line {}: hw takes a name", number + 1))?;
                }
                "case" => {
                    let name = tokens
                        .first()
                        .map(|s| (*s).to_owned())
                        .ok_or_else(|| format!("line {}: case takes a name", number + 1))?;
                    reference.cases.push(RefCase {
                        name,
                        ..RefCase::default()
                    });
                }
                "desc" => case(&mut reference, number)?.desc = fields(&tokens)?,
                "status" => {
                    let c = case(&mut reference, number)?;
                    match tokens.first() {
                        Some(&"ok") => c.ok = Some(true),
                        Some(&"fail") => c.ok = Some(false),
                        _ => return Err(format!("line {}: status takes ok or fail", number + 1)),
                    }
                    c.result = fields(tokens.get(1..).unwrap_or_default())?;
                }
                "res" => case(&mut reference, number)?.result = fields(&tokens)?,
                "plane" => case(&mut reference, number)?.planes.push(fields(&tokens)?),
                "mip" => case(&mut reference, number)?.mips.push(fields(&tokens)?),
                "texel" => case(&mut reference, number)?.texels.push(fields(&tokens)?),
                "copy" => case(&mut reference, number)?.copies.push(fields(&tokens)?),
                "tiling" => case(&mut reference, number)?.tiling = Some(fields(&tokens)?),
                "end" => {}
                other => return Err(format!("line {}: unknown line {other:?}", number + 1)),
            }
        }
        if reference.cases.is_empty() {
            return Err("no cases".to_owned());
        }
        Ok(reference)
    }
}

fn case(reference: &mut Reference, number: usize) -> Result<&mut RefCase, String> {
    reference
        .cases
        .last_mut()
        .ok_or_else(|| format!("line {}: a line before the first case", number + 1))
}

// ---------------------------------------------------------------- model

/// What [`victory_formats::texture`] says a descriptor is.
struct Model {
    texture: Texture,
    desc: TextureDesc,
}

struct ModelPlane {
    usage: u32,
    size: u64,
    slice_size: u64,
    offset: u64,
    alignment: u64,
    element_bytes: u32,
    standard_mips: u32,
    tail_mips: u32,
}

struct ModelMip {
    size: u64,
    offset: u64,
    slice_size: u64,
    pitch_pixels: u32,
    pitch_bytes: u32,
    alignment: u32,
    padded: [u32; 3],
    elements: [u32; 2],
    pixels: [u32; 2],
    depth_or_array: u32,
    samples: u32,
    swizzle: u32,
    bank: (u64, u64),
    slice_depth: u32,
    in_tail: u32,
    tail_x: u32,
    tail_y: u32,
}

struct ModelTiling {
    tiles: u32,
    standard: u8,
    packed: u8,
    packed_tiles: u32,
    shape: [u32; 3],
    starts: Vec<u32>,
}

impl Model {
    fn new(desc: TextureDesc, config: PipeConfig) -> Option<Model> {
        let texture = Texture::new_for(&desc, config).ok()?;
        Some(Model { texture, desc })
    }

    fn resource(&self) -> (u64, u64, u32, u32) {
        let layout = &self.texture.planes[0].layout;
        (
            self.texture.size,
            self.texture.alignment,
            layout.mips.len() as u32,
            self.texture.planes.len() as u32,
        )
    }

    fn planes(&self) -> Vec<ModelPlane> {
        self.texture
            .planes
            .iter()
            .map(|plane| {
                let l = &plane.layout;
                let mips = l.mips.len() as u32;
                ModelPlane {
                    // The reference cases are sampled textures; a stencil or depth plane would
                    // be 8 or 7 (see `xg.rs`).
                    usage: if plane.stencil { 8 } else { 1 },
                    size: l.size,
                    slice_size: l.slice_size,
                    offset: plane.offset,
                    alignment: l.alignment,
                    element_bytes: l.desc.element_bytes,
                    standard_mips: l.first_tail_mip,
                    tail_mips: mips - l.first_tail_mip,
                }
            })
            .collect()
    }

    /// The mapping `crates/victory-hle/src/modules/xg.rs` (`mip_out`) reports.
    fn mips(&self) -> Vec<Vec<ModelMip>> {
        self.texture
            .planes
            .iter()
            .map(|plane| {
                let l = &plane.layout;
                let d = self.desc;
                (0..l.mips.len())
                    .map(|i| {
                        let m = &l.mips[i];
                        let bpe = u64::from(l.desc.element_bytes);
                        let slice = u64::from(m.pitch) * u64::from(m.rows) * bpe;
                        // `mip_out` multiplies standard mips by the sample count, not tail mips.
                        let samples = u64::from(d.samples.max(1));
                        let mut out = ModelMip {
                            size: slice * samples,
                            offset: m.offset,
                            slice_size: slice * samples,
                            pitch_pixels: m.pitch * l.desc.block,
                            pitch_bytes: m.pitch * l.desc.element_bytes,
                            alignment: l.alignment as u32,
                            padded: [m.pitch, m.rows, d.depth_or_array.max(1)],
                            elements: [m.width, m.height],
                            pixels: [(d.width >> i).max(1), (d.height >> i).max(1)],
                            depth_or_array: d.depth_or_array.max(1),
                            samples: d.samples.max(1),
                            swizzle: l.desc.swizzle.0,
                            bank: l.bank_rotation(),
                            slice_depth: l.block[2],
                            in_tail: 0,
                            tail_x: 0,
                            tail_y: 0,
                        };
                        if let Some(t) = m.tail {
                            out.in_tail = 1;
                            out.alignment = 256;
                            out.offset = u64::from(t.offset);
                            out.size = slice;
                            out.slice_size = slice;
                            out.tail_x = t.x;
                            out.tail_y = t.y;
                        }
                        out
                    })
                    .collect()
            })
            .collect()
    }

    fn tiling(&self) -> Option<ModelTiling> {
        let l = &self.texture.planes[0].layout;
        if l.alignment != TILE || l.thick {
            return None;
        }
        let mips = l.mips.len() as u32;
        let standard = l.first_tail_mip;
        let packed_tiles = match standard {
            0 => l.slice_size / TILE,
            s if s < mips => l.mips[s as usize - 1].offset / TILE,
            _ => 0,
        };
        // One entry: the first standard mip of slice 0. The harness asks the library for one
        // tiling; the host's `tiling` method serves more (see `xg.rs`).
        let starts = if standard == 0 {
            Vec::new()
        } else {
            vec![(l.mips[0].offset / TILE) as u32]
        };
        Some(ModelTiling {
            tiles: (l.size / TILE) as u32,
            standard: standard as u8,
            packed: (mips - standard) as u8,
            packed_tiles: packed_tiles as u32,
            shape: [
                l.block[0] * l.desc.block,
                l.block[1] * l.desc.block,
                l.block[2],
            ],
            starts,
        })
    }
}

// ---------------------------------------------------------------- comparison

struct Report {
    cases: u64,
    checks: u64,
    mismatches: u64,
    /// Every mismatch; `print` shows the first `max`.
    shown: Vec<String>,
    max: usize,
}

impl Default for Report {
    fn default() -> Report {
        Report {
            cases: 0,
            checks: 0,
            mismatches: 0,
            shown: Vec::new(),
            max: 60,
        }
    }
}

impl Report {
    fn check(&mut self, context: &str, what: &str, got: Option<u64>, want: u64) {
        self.checks += 1;
        match got {
            Some(value) if value == want => {}
            Some(value) => self.fail(context, format!("{what}: reference {value}, model {want}")),
            None => self.fail(
                context,
                format!("{what}: reference has no value, model {want}"),
            ),
        }
    }

    fn fail(&mut self, context: &str, message: String) {
        self.mismatches += 1;
        self.shown.push(format!("{context}: {message}"));
    }

    fn print(&self, hardware: &str) {
        let hw = if hardware.is_empty() {
            "unknown"
        } else {
            hardware
        };
        println!(
            "xg layout diff: {hw}, {} cases, {} fields, {} mismatches",
            self.cases, self.checks, self.mismatches
        );
        let shown = self.shown.len().min(self.max);
        for line in &self.shown[..shown] {
            println!("  {line}");
        }
        if self.shown.len() > shown {
            println!("  ... {} more", self.shown.len() - shown);
        }
    }
}

fn compare(reference: &Reference, config: PipeConfig, only_xor: bool) -> Report {
    compare_with(reference, config, only_xor, false)
}

fn compare_with(
    reference: &Reference,
    config: PipeConfig,
    only_xor: bool,
    skip_tiling: bool,
) -> Report {
    let mut report = Report::default();
    for case in &reference.cases {
        if only_xor && case.desc.u32("swz").unwrap_or(0) < 16 {
            continue;
        }
        report.cases += 1;
        compare_case(case, config, skip_tiling, &mut report);
    }
    report
}

fn compare_case(case: &RefCase, config: PipeConfig, skip_tiling: bool, report: &mut Report) {
    let context = &case.name;
    let Some(desc) = descriptor(&case.desc) else {
        report.fail(context, "case has no usable descriptor".to_owned());
        return;
    };
    let texture = Texture::new_for(&desc, config);
    match (case.ok, &texture) {
        (Some(false), Err(_)) => return,
        (Some(false), Ok(_)) => {
            report.fail(
                context,
                format!(
                    "reference rejected the descriptor ({}), the model accepts it",
                    hr(case)
                ),
            );
            return;
        }
        (Some(true), Err(e)) => {
            report.fail(
                context,
                format!("reference accepted the descriptor, the model: {e:?}"),
            );
            return;
        }
        (_, Err(_)) => {
            report.fail(context, "no status line".to_owned());
            return;
        }
        (None, Ok(_)) => {
            report.fail(context, "no status line".to_owned());
            return;
        }
        (Some(true), Ok(_)) => {}
    }
    let Some(model) = Model::new(desc, config) else {
        report.fail(
            context,
            "the model accepted the descriptor but built no layout".to_owned(),
        );
        return;
    };
    let (size, alignment, mips, planes) = model.resource();
    report.check(context, "res.size", case.result.u64("size"), size);
    report.check(context, "res.align", case.result.u64("align"), alignment);
    report.check(
        context,
        "res.mips",
        case.result.u64("mips"),
        u64::from(mips),
    );
    report.check(
        context,
        "res.planes",
        case.result.u64("planes"),
        u64::from(planes),
    );

    let expected_planes = model.planes();
    report.check(
        context,
        "plane count",
        Some(case.planes.len() as u64),
        expected_planes.len() as u64,
    );
    for (i, (reference, expected)) in case.planes.iter().zip(&expected_planes).enumerate() {
        let c = format!("{context} plane {i}");
        report.check(&c, "i", reference.u64("i"), i as u64);
        report.check(
            &c,
            "usage",
            reference.u64("usage"),
            u64::from(expected.usage),
        );
        report.check(&c, "size", reference.u64("size"), expected.size);
        report.check(&c, "slice", reference.u64("slice"), expected.slice_size);
        report.check(&c, "offset", reference.u64("offset"), expected.offset);
        report.check(&c, "align", reference.u64("align"), expected.alignment);
        report.check(
            &c,
            "bpe",
            reference.u64("bpe"),
            u64::from(expected.element_bytes),
        );
        report.check(
            &c,
            "std",
            reference.u64("std"),
            u64::from(expected.standard_mips),
        );
        report.check(
            &c,
            "tail",
            reference.u64("tail"),
            u64::from(expected.tail_mips),
        );
    }

    let expected_mips = model.mips();
    report.check(
        context,
        "mip line count",
        Some(case.mips.len() as u64),
        expected_mips.iter().map(Vec::len).sum::<usize>() as u64,
    );
    let mut seen = vec![0usize; expected_mips.len()];
    for reference in &case.mips {
        let (plane, mip) = (
            reference.u32("plane").unwrap_or(0) as usize,
            reference.u32("i").unwrap_or(0) as usize,
        );
        let Some(expected) = expected_mips.get(plane).and_then(|m| m.get(mip)) else {
            report.fail(
                &format!("{context} mip {mip}"),
                format!("plane {plane} or mip {mip} does not exist in the model"),
            );
            continue;
        };
        seen[plane] += 1;
        let c = format!("{context} plane {plane} mip {mip}");
        report.check(&c, "size", reference.u64("size"), expected.size);
        report.check(&c, "offset", reference.u64("offset"), expected.offset);
        report.check(&c, "slice", reference.u64("slice"), expected.slice_size);
        report.check(
            &c,
            "pitch_px",
            reference.u64("pitch_px"),
            u64::from(expected.pitch_pixels),
        );
        report.check(
            &c,
            "pitch_b",
            reference.u64("pitch_b"),
            u64::from(expected.pitch_bytes),
        );
        report.check(
            &c,
            "align",
            reference.u64("align"),
            u64::from(expected.alignment),
        );
        for (name, value) in [
            ("pw", expected.padded[0]),
            ("ph", expected.padded[1]),
            ("pd", expected.padded[2]),
            ("ew", expected.elements[0]),
            ("eh", expected.elements[1]),
            ("wp", expected.pixels[0]),
            ("hp", expected.pixels[1]),
            ("doa", expected.depth_or_array),
            ("samples", expected.samples),
            ("swz", expected.swizzle),
            ("slice_depth", expected.slice_depth),
        ] {
            report.check(&c, name, reference.u64(name), u64::from(value));
        }
        report.check(&c, "bank0", reference.u64("bank0"), expected.bank.0);
        report.check(&c, "bank1", reference.u64("bank1"), expected.bank.1);
        report.check(
            &c,
            "tail",
            reference.u64("tail"),
            u64::from(expected.in_tail),
        );
        report.check(&c, "tx", reference.u64("tx"), u64::from(expected.tail_x));
        report.check(&c, "ty", reference.u64("ty"), u64::from(expected.tail_y));
    }
    for (plane, count) in seen.iter().enumerate() {
        if *count != expected_mips[plane].len() {
            report.fail(
                &format!("{context} plane {plane}"),
                format!("{count} mip lines for {} mips", expected_mips[plane].len()),
            );
        }
    }

    compare_texels(case, &model, report);
    compare_copies(case, &model, report);
    if !skip_tiling {
        compare_tiling(case, &model, report);
    }
}

fn compare_texels(case: &RefCase, model: &Model, report: &mut Report) {
    for reference in &case.texels {
        let plane = reference.u32("plane").unwrap_or(0) as usize;
        let mip = reference.u32("mip").unwrap_or(0);
        let x = reference.u64("x").unwrap_or(0) as u32;
        let y = reference.u32("y").unwrap_or(0);
        let z = reference.u32("z").unwrap_or(0);
        let c = format!("{} texel p{plane} m{mip} ({x},{y},{z})", case.name);
        let Some(p) = model.texture.planes.get(plane) else {
            report.fail(&c, format!("plane {plane} does not exist"));
            continue;
        };
        match p.layout.offset(mip, x, y, z) {
            Ok(offset) => report.check(&c, "off", reference.u64("off"), p.offset + offset),
            Err(e) => report.fail(&c, format!("the model has no offset ({e:?})")),
        }
    }
}

fn compare_copies(case: &RefCase, model: &Model, report: &mut Report) {
    for reference in &case.copies {
        let plane = reference.u32("plane").unwrap_or(0) as usize;
        let sub = reference.u32("sub").unwrap_or(0);
        let row = reference.u32("row").unwrap_or(0);
        let depth = reference.u32("depth").unwrap_or(0);
        let c = format!("{} copy p{plane} sub{sub}", case.name);
        let Some(p) = model.texture.planes.get(plane) else {
            report.fail(&c, format!("plane {plane} does not exist"));
            continue;
        };
        let layout = &p.layout;
        let mips = layout.mips.len() as u32;
        let (mip, slice) = (sub % mips, sub / mips);
        let Some(m) = layout.mips.get(mip as usize) else {
            report.fail(&c, format!("mip {mip} does not exist"));
            continue;
        };
        let bytes = m.width as usize * layout.desc.element_bytes as usize;
        let len = (m.height as usize - 1) * row as usize + bytes;
        report.check(&c, "len", reference.u64("len"), len as u64);
        report.check(&c, "row", reference.u64("row"), u64::from(row));
        report.check(&c, "depth", reference.u64("depth"), u64::from(depth));
        let Some(size) = usize::try_from(layout.size).ok() else {
            report.fail(&c, "texture too large for this host".to_owned());
            continue;
        };
        let mut tiled = pattern(size, EMPTY_PATTERN_SEED);
        let mut linear = vec![0u8; len];
        if let Err(e) = layout.detile(
            &tiled,
            mip,
            slice,
            &mut linear,
            row as usize,
            depth as usize,
        ) {
            report.fail(&c, format!("detile failed: {e:?}"));
            continue;
        }
        report.check(&c, "from", reference.u64("from"), fnv(&linear));
        // The reverse: linear rows into a tiled buffer, hashed whole.
        tiled = pattern(size, INTO_TILED_SEED);
        linear = pattern(len, INTO_LINEAR_SEED);
        if let Err(e) = layout.tile(
            &mut tiled,
            mip,
            slice,
            &linear,
            row as usize,
            depth as usize,
        ) {
            report.fail(&c, format!("tile failed: {e:?}"));
            continue;
        }
        report.check(&c, "into", reference.u64("into"), fnv(&tiled));
    }
}

fn compare_tiling(case: &RefCase, model: &Model, report: &mut Report) {
    let expected = model.tiling();
    match (&case.tiling, &expected) {
        (None, None) => {}
        (None, Some(_)) => {
            report.fail(
                &case.name,
                "the model has a tiling the reference does not".to_owned(),
            );
        }
        (Some(reference), None) => {
            report.fail(
                &case.name,
                "the reference has a tiling the model does not (64 KiB modes only)".to_owned(),
            );
            let _ = reference;
        }
        (Some(reference), Some(expected)) => {
            let c = format!("{} tiling", case.name);
            report.check(
                &c,
                "tiles",
                reference.u64("tiles"),
                u64::from(expected.tiles),
            );
            report.check(
                &c,
                "std",
                reference.u64("std"),
                u64::from(expected.standard),
            );
            report.check(
                &c,
                "packed",
                reference.u64("packed"),
                u64::from(expected.packed),
            );
            report.check(
                &c,
                "ptiles",
                reference.u64("ptiles"),
                u64::from(expected.packed_tiles),
            );
            report.check(&c, "start", reference.u64("start"), 0);
            report.check(&c, "tw", reference.u64("tw"), u64::from(expected.shape[0]));
            report.check(&c, "th", reference.u64("th"), u64::from(expected.shape[1]));
            report.check(&c, "td", reference.u64("td"), u64::from(expected.shape[2]));
            let starts: Vec<u32> = reference
                .get("starts")
                .unwrap_or_default()
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect();
            report.check(
                &c,
                "start count",
                Some(starts.len() as u64),
                expected.starts.len() as u64,
            );
            for (i, (got, want)) in starts.iter().zip(&expected.starts).enumerate() {
                report.check(
                    &c,
                    &format!("starts[{i}]"),
                    Some(u64::from(*got)),
                    u64::from(*want),
                );
            }
        }
    }
}

fn find_pipe_config(reference: &Reference) -> ExitCode {
    let rows = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4),
        (3, 5),
        (4, 4),
        (4, 5),
        (4, 6),
        (5, 5),
        (5, 6),
    ];
    println!("XOR cases only (swizzle >= 16, tilings skipped); a row with 0 mismatches fits:");
    let mut winners = 0;
    let mut fewest = (u64::MAX, (0, 0));
    for (packers_log2, pipes_log2) in rows {
        let config = PipeConfig {
            packers_log2,
            pipes_log2,
        };
        let report = compare_with(reference, config, true, true);
        println!(
            "  packers_log2={packers_log2} pipes_log2={pipes_log2}: {} checks, {} mismatches",
            report.checks, report.mismatches
        );
        if report.mismatches == 0 {
            winners += 1;
        }
        if report.mismatches < fewest.0 {
            fewest = (report.mismatches, (packers_log2, pipes_log2));
        }
    }
    if winners == 0 {
        eprintln!(
            "no row matches; fewest mismatches: {} at packers_log2={} pipes_log2={}",
            fewest.0, fewest.1.0, fewest.1.1
        );
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn descriptor(fields: &Fields) -> Option<TextureDesc> {
    let dimension = match fields.get("dim")? {
        "1d" => Dimension::Tex1d,
        "2d" => Dimension::Tex2d,
        "3d" => Dimension::Tex3d,
        _ => return None,
    };
    Some(TextureDesc {
        dimension,
        width: fields.u32("w")?,
        height: fields.u32("h")?,
        depth_or_array: fields.u32("array")?,
        mips: fields.u32("mips")?,
        format: fields.u32("fmt")?,
        samples: fields.u32("samples")?,
        swizzle: Swizzle(fields.u32("swz")?),
    })
}

fn hr(case: &RefCase) -> String {
    case.result
        .get("hr")
        .map(str::to_owned)
        .unwrap_or_else(|| "no hr".to_owned())
}

// ---------------------------------------------------------------- shared bytes

/// The bytes both sides put in a buffer before a copy. `i` and `seed` are 64-bit.
fn pattern(len: usize, seed: u64) -> Vec<u8> {
    (0..len as u64)
        .map(|i| {
            let mut v = i
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed.wrapping_mul(0x0000_0100_0000_01B3));
            v ^= v >> 29;
            (v >> 32) as u8
        })
        .collect()
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

// ---------------------------------------------------------------- self test

fn self_cases() -> Vec<(&'static str, TextureDesc)> {
    vec![
        (
            "bc7_npot_130x68_64kb_d",
            TextureDesc {
                dimension: Dimension::Tex2d,
                width: 130,
                height: 68,
                depth_or_array: 1,
                mips: 0,
                format: 98,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
        ),
        (
            "bc1_array3_100x60_4kb_s",
            TextureDesc {
                dimension: Dimension::Tex2d,
                width: 100,
                height: 60,
                depth_or_array: 3,
                mips: 0,
                format: 71,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            },
        ),
        (
            "rgba8_256x256_64kb_s_x",
            TextureDesc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 256,
                depth_or_array: 1,
                mips: 0,
                format: 28,
                samples: 1,
                swizzle: Swizzle::SW_64KB_S_X,
            },
        ),
    ]
}

/// Writes a reference from the model, for the self test only.
fn emit_model_reference(cases: &[(&str, TextureDesc)]) -> String {
    let mut out = String::from("xg-layout-ref 1\nhw self-test\n");
    for (name, desc) in cases {
        let _ = writeln!(out, "case {name}");
        let dim = match desc.dimension {
            Dimension::Tex1d => "1d",
            Dimension::Tex2d => "2d",
            Dimension::Tex3d => "3d",
        };
        let model = Model::new(*desc, PipeConfig::DEFAULT);
        let mips = model
            .as_ref()
            .map_or(desc.mips, |m| m.texture.planes[0].layout.mips.len() as u32);
        let _ = writeln!(
            out,
            "desc dim={dim} w={} h={} array={} mips={mips} fmt={} samples={} swz={}",
            desc.width, desc.height, desc.depth_or_array, desc.format, desc.samples, desc.swizzle.0
        );
        let Some(model) = model else {
            let _ = writeln!(out, "status fail hr=0x80070057\nend");
            continue;
        };
        let _ = writeln!(out, "status ok");
        let (size, alignment, mips, planes) = model.resource();
        let _ = writeln!(
            out,
            "res size={size} align={alignment} mips={mips} planes={planes}"
        );
        for (i, plane) in model.planes().iter().enumerate() {
            let _ = writeln!(
                out,
                "plane i={i} usage={} size={} slice={} offset={} align={} bpe={} std={} tail={}",
                plane.usage,
                plane.size,
                plane.slice_size,
                plane.offset,
                plane.alignment,
                plane.element_bytes,
                plane.standard_mips,
                plane.tail_mips
            );
        }
        for (plane, mips) in model.mips().iter().enumerate() {
            for (i, mip) in mips.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "mip plane={plane} i={i} size={} offset={} slice={} pitch_px={} pitch_b={} \
                     align={} pw={} ph={} pd={} ew={} eh={} wp={} hp={} doa={} samples={} swz={} \
                     bank0={} bank1={} slice_depth={} tail={} tx={} ty={}",
                    mip.size,
                    mip.offset,
                    mip.slice_size,
                    mip.pitch_pixels,
                    mip.pitch_bytes,
                    mip.alignment,
                    mip.padded[0],
                    mip.padded[1],
                    mip.padded[2],
                    mip.elements[0],
                    mip.elements[1],
                    mip.pixels[0],
                    mip.pixels[1],
                    mip.depth_or_array,
                    mip.samples,
                    mip.swizzle,
                    mip.bank.0,
                    mip.bank.1,
                    mip.slice_depth,
                    mip.in_tail,
                    mip.tail_x,
                    mip.tail_y
                );
            }
        }
        let mip_list = model.mips();
        for (plane, mips) in mip_list.iter().enumerate() {
            for (mip, m) in mips.iter().enumerate() {
                let mut points = vec![(0, 0), (m.elements[0] - 1, 0), (0, m.elements[1] - 1)];
                if m.elements[0] > 2 && m.elements[1] > 2 {
                    points.push((m.elements[0] / 2, m.elements[1] / 2));
                }
                points.sort_unstable();
                points.dedup();
                let zs: Vec<u32> = if desc.depth_or_array > 1 {
                    vec![0, desc.depth_or_array - 1]
                } else {
                    vec![0]
                };
                for &(x, y) in &points {
                    for &z in &zs {
                        let p = &model.texture.planes[plane];
                        let Ok(offset) = p.layout.offset(mip as u32, x, y, z) else {
                            continue;
                        };
                        let _ = writeln!(
                            out,
                            "texel plane={plane} mip={mip} x={x} y={y} z={z} off={}",
                            p.offset + offset
                        );
                    }
                }
            }
        }
        let layout = &model.texture.planes[0].layout;
        let bpe = layout.desc.element_bytes as usize;
        let mips = layout.mips.len() as u32;
        let slices: Vec<u32> = if desc.depth_or_array > 1 {
            vec![0, desc.depth_or_array - 1]
        } else {
            vec![0]
        };
        for &slice in &slices {
            for mip in 0..mips {
                let m = &layout.mips[mip as usize];
                let row = m.width as usize * bpe + 16;
                let depth = row * m.height as usize + 32;
                let len = (m.height as usize - 1) * row + m.width as usize * bpe;
                let size = layout.size as usize;
                let mut tiled = pattern(size, EMPTY_PATTERN_SEED);
                let mut linear = vec![0u8; len];
                let sub = mip + slice * mips;
                if layout
                    .detile(&tiled, mip, slice, &mut linear, row, depth)
                    .is_err()
                {
                    continue;
                }
                let from = fnv(&linear);
                tiled = pattern(size, INTO_TILED_SEED);
                linear = pattern(len, INTO_LINEAR_SEED);
                if layout
                    .tile(&mut tiled, mip, slice, &linear, row, depth)
                    .is_err()
                {
                    continue;
                }
                let into = fnv(&tiled);
                let _ = writeln!(
                    out,
                    "copy plane=0 sub={sub} row={row} depth={depth} from={from} \
                     into={into} len={len}"
                );
            }
        }
        if let Some(tiling) = model.tiling() {
            let starts = tiling
                .starts
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let _ = writeln!(
                out,
                "tiling tiles={} std={} packed={} ptiles={} start=0 tw={} th={} td={} n={} \
                 starts={starts}",
                tiling.tiles,
                tiling.standard,
                tiling.packed,
                tiling.packed_tiles,
                tiling.shape[0],
                tiling.shape[1],
                tiling.shape[2],
                tiling.starts.len()
            );
        }
        let _ = writeln!(out, "end");
    }
    out
}

fn self_test() -> Result<(), String> {
    let cases = self_cases();
    let text = emit_model_reference(&cases);
    let reference = Reference::parse(&text)?;
    let report = compare(&reference, PipeConfig::DEFAULT, false);
    if report.mismatches != 0 {
        report.print("self-test");
        return Err("the model does not agree with its own reference text".to_owned());
    }
    let checks = report.checks;
    if checks < 300 {
        return Err(format!(
            "only {checks} fields compared; the self test is too small"
        ));
    }
    // Every kind of line is compared: a wrong value in each kind must be noticed.
    for keyword in ["res", "plane", "mip", "texel", "copy", "tiling"] {
        let mutated = mutate_first_line(&text, keyword)
            .ok_or_else(|| format!("self test finds no {keyword} line"))?;
        let reference = Reference::parse(&mutated)?;
        let report = compare(&reference, PipeConfig::DEFAULT, false);
        if report.mismatches == 0 {
            return Err(format!("a mutated {keyword} line was not noticed"));
        }
    }
    Ok(())
}

/// The first line starting with `keyword`, with a number after its first `=` changed by one.
fn mutate_first_line(text: &str, keyword: &str) -> Option<String> {
    let mut out = String::new();
    let mut changed = false;
    for line in text.lines() {
        if !changed && line.starts_with(keyword) && line.contains('=') {
            let (head, value) = line.split_once('=')?;
            let end = value.find(' ').unwrap_or(value.len());
            let number: u64 = value[..end].parse().ok()?;
            out.push_str(&format!("{head}={}\n", number + 1));
            changed = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    changed.then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_agrees_with_a_reference_written_from_it() {
        self_test().unwrap();
    }

    #[test]
    fn pattern_and_hash_are_stable() {
        // Guards the shared byte pattern: a change here breaks every copy comparison.
        assert_eq!(pattern(4, 7), vec![0x00, 0xbd, 0x73, 0x2a]);
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn a_reference_line_must_be_key_value() {
        let text = "xg-layout-ref 1\ncase x\ndesc dim=2 w\nend\n";
        let error = Reference::parse(text).err().unwrap();
        assert!(error.contains("not key=value"), "{error}");
    }

    #[test]
    fn a_reference_needs_the_magic_first_line() {
        assert!(Reference::parse("").is_err());
        assert!(Reference::parse("hello\n").is_err());
    }
}
