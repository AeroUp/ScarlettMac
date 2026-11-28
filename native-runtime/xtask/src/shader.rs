//! `cargo xtask shader`: a dumped shader container by the hash a run log
//! names, as GFX10 disassembly and pseudo-HLSL.
//!
//! The containers are game data and stay outside the repository: the folder
//! is given with `--corpus` or `VICTORY_SHADER_CORPUS`, and the tool writes
//! nothing unless `--dxil` names a folder. `--reference` reads RageForge's
//! `Shaders/Tables.cs` (LGPL-3.0) to cross-check our opcode tables against
//! its reference tables; nothing of it is copied into the tool.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use victory_shader::disasm;
use victory_shader::emit::Options;
use victory_shader::{Container, FourCC, NativeProgram, native_programs};

pub struct Args {
    pub hashes: Vec<String>,
    pub corpus: Option<PathBuf>,
    pub disasm: bool,
    pub hlsl: bool,
    pub dxil: Option<PathBuf>,
    pub reference: Option<PathBuf>,
    pub check: bool,
    /// Print the container's layout and the native reader's verdict only.
    pub probe: bool,
}

pub fn run(args: Args) -> Result<()> {
    if args.check && args.reference.is_none() {
        bail!("--check needs --reference");
    }
    let corpus = match args.corpus.clone() {
        Some(dir) => dir,
        None => std::env::var_os("VICTORY_SHADER_CORPUS")
            .map(PathBuf::from)
            .context("--corpus or VICTORY_SHADER_CORPUS names the folder of dumped containers")?,
    };
    let mut checked = true;
    for asked in &args.hashes {
        let hash = normalize(asked)
            .with_context(|| format!("{asked:?} does not hold a 16-digit shader hash"))?;
        let file = find(&corpus, &hash)?.with_context(|| {
            format!(
                "no container for {hash} under {} (names end with _{hash}.bin, or are {hash}.bin)",
                corpus.display()
            )
        })?;
        println!("== {hash} - {}", file.display());
        let bytes = std::fs::read(&file).with_context(|| format!("reading {}", file.display()))?;
        show(&hash, &bytes, &args)?;
    }
    if let Some(reference) = &args.reference {
        checked = check_tables(reference, args.check)?;
    }
    if !checked {
        bail!("opcode tables disagree with the reference (see above)");
    }
    Ok(())
}

/// The 16 hex digits of a hash as a log prints it, in any of the forms a log
/// or a file name uses (`compute 39ef...`, `vertex_39ef....bin`, `39ef...`).
fn normalize(asked: &str) -> Option<String> {
    let token = asked
        .split([' ', '\t', '_', '/', '\\'])
        .filter(|t| t.len() >= 16)
        .find(|t| t.as_bytes()[..16].iter().all(u8::is_ascii_hexdigit))
        .or_else(|| asked.strip_suffix(".bin").filter(|t| t.len() >= 16))?;
    Some(token[..16].to_ascii_lowercase())
}

/// The first file under `dir` whose stem is `hash` or ends with `_hash`.
fn find(dir: &Path, hash: &str) -> Result<Option<PathBuf>> {
    // The two shapes a dump writes, before walking a corpus of thousands.
    for name in [hash.to_string(), format!("{hash}.bin")] {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(Some(path));
        }
    }
    let mut found = None;
    for e in walkdir::WalkDir::new(dir).follow_links(false) {
        let e = match e {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !e.file_type().is_file() {
            continue;
        }
        let path = e.path();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if stem.eq_ignore_ascii_case(hash)
            || stem.to_ascii_lowercase().ends_with(&format!("_{hash}"))
        {
            found = Some(path.to_path_buf());
            break;
        }
    }
    Ok(found)
}

fn show(hash: &str, bytes: &[u8], args: &Args) -> Result<()> {
    if args.probe {
        print!("{}", probe_text(bytes)?);
        return Ok(());
    }
    let container = Container::parse(bytes).context("not a DXBC container")?;
    let parts: Vec<(FourCC, &[u8])> = [FourCC::DXIL, FourCC::ACCC]
        .into_iter()
        .filter_map(|fourcc| container.part(fourcc).map(|p| (fourcc, p.data)))
        .collect();
    if parts.is_empty() {
        bail!("no DXIL or ACCC part");
    }
    for (fourcc, data) in parts {
        let programs = native_programs(data).with_context(|| format!("reading {fourcc:?}"))?;
        if programs.is_empty() {
            println!("{}: no native program (a PC container)", fourcc_str(fourcc));
            continue;
        }
        for (n, program) in programs.iter().enumerate() {
            println!(
                "-- {} program {}/{}: {} bytes, {} VGPRs",
                fourcc_str(fourcc),
                n + 1,
                programs.len(),
                program.code.len(),
                program.vgprs
            );
            show_program(program);
            if args.disasm {
                println!();
                match disasm::disassembly(program, false) {
                    Ok(text) => print!("{text}"),
                    Err(e) => println!("; disassembly refused: {e}"),
                }
            }
            if args.hlsl {
                println!();
                match disasm::pseudo_hlsl(program) {
                    Ok(text) => print!("{text}"),
                    Err(e) => println!("; pseudo-HLSL refused: {e}"),
                }
            }
        }
    }
    let options = Options {
        wave32_host: true,
        zero_copy_memory: true,
        ..Options::default()
    };
    match victory_shader::emit::translate_with(bytes, options) {
        Ok(host) => {
            println!("translation: ok ({} bytes of DXIL)", host.dxil.len());
            if let Some(dir) = &args.dxil {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("making {}", dir.display()))?;
                let path = dir.join(format!("{hash}.dxil"));
                std::fs::write(&path, &host.dxil)
                    .with_context(|| format!("writing {}", path.display()))?;
                println!("wrote {}", path.display());
            }
        }
        Err(e) => {
            println!("translation: refused: {e}");
            if let Some(dir) = &args.dxil {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("making {}", dir.display()))?;
                let path = dir.join(format!("{hash}.refused.txt"));
                std::fs::write(&path, e.to_string())
                    .with_context(|| format!("writing {}", path.display()))?;
                println!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}

/// A structural reading of one container: the parts, the `DXIL` program header,
/// every `XDAT` index found (the native reader takes the one at `+0x18`) and the
/// native reader's verdict. No code, signature or container bytes are printed:
/// this is for comparing a compiler's output with the reader's assumptions.
fn probe_text(bytes: &[u8]) -> Result<String> {
    use std::fmt::Write as _;

    let c = Container::parse(bytes).context("not a DXBC container")?;
    let mut out = String::new();
    writeln!(
        out,
        "container: DXBC {}.{}, {} parts, {} bytes",
        c.version.0,
        c.version.1,
        c.parts.len(),
        bytes.len()
    )?;
    for p in &c.parts {
        writeln!(
            out,
            "  part {} at +{:#x}, {} bytes",
            p.fourcc,
            p.offset,
            p.data.len()
        )?;
    }
    let part = c
        .part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .context("no DXIL or ACCC part")?;
    let d = part.data;
    writeln!(out, "{} part: {} bytes", part.fourcc, d.len())?;

    if part.fourcc == FourCC::ACCC {
        writeln!(out, "  FSPX part: no DXIL program header (ACCC)")?;
    } else if d.len() >= 24 {
        let version = u32_at(d, 0)?;
        let size = u32_at(d, 4)?;
        let magic = &d[8..12];
        let dxil_version = u32_at(d, 12)?;
        let offset = u32_at(d, 16)?;
        let len = u32_at(d, 20)?;
        writeln!(
            out,
            "  program header: version {version:#010x} ({}), size {size} dwords, magic {}",
            profile(version),
            String::from_utf8_lossy(magic)
        )?;
        writeln!(out, "  dxil version {dxil_version:#x}")?;
        if len == 0 {
            writeln!(out, "  bitcode: none (size 0)")?;
        } else {
            let at = 8usize.saturating_add(offset as usize);
            let magic = d.get(at..at.saturating_add(4));
            let label = if magic == Some(&[0x42, 0x43, 0xc0, 0xde][..]) {
                "LLVM bitcode"
            } else if magic.is_some() {
                "not LLVM bitcode"
            } else {
                "out of bounds"
            };
            writeln!(out, "  bitcode: +{offset:#x}, {len} bytes, {label}")?;
        }
    } else {
        writeln!(out, "  program header: too short")?;
    }

    let candidates: Vec<usize> = (0..d.len().saturating_sub(3))
        .filter(|&i| &d[i..i + 4] == b"XDAT")
        .take(8)
        .collect();
    if candidates.is_empty() {
        writeln!(out, "  XDAT: none")?;
    }
    for at in candidates {
        let count = u32_at(d, at + 4).unwrap_or(u32::MAX);
        let header = u32_at(d, at + 8).unwrap_or(u32::MAX);
        writeln!(
            out,
            "  XDAT at +{at:#x}: count {count}, header {header:#x}{}",
            if at == 0x18 {
                " (the native reader's offset)"
            } else {
                " (the native reader does not look here)"
            }
        )?;
        let table = at + 12;
        for i in 0..count.min(64) as usize {
            let entry = table + 12 * i;
            let (Some(tag), Some(offset), Some(size)) = (
                u32_at(d, entry).ok(),
                u32_at(d, entry + 4).ok(),
                u32_at(d, entry + 8).ok(),
            ) else {
                writeln!(out, "    entry {i}: out of bounds")?;
                break;
            };
            let in_bounds = (offset as usize)
                .checked_add(size as usize)
                .is_some_and(|end| end <= d.len());
            write!(
                out,
                "    entry {i}: tag {tag:#04x}, +{offset:#x}, {size} bytes{}",
                if in_bounds { "" } else { " (out of bounds)" }
            )?;
            if tag == 0x0d && in_bounds {
                let blob = &d[offset as usize..offset as usize + size as usize];
                write!(out, ", magic {}", blob[..blob.len().min(4)].escape_ascii())?;
                if blob.len() >= 24 {
                    write!(
                        out,
                        ", version {}, packed {}, unpacked {}",
                        u32_at(blob, 4).unwrap_or(u32::MAX),
                        u64_at(blob, 8).unwrap_or(u64::MAX),
                        u64_at(blob, 16).unwrap_or(u64::MAX)
                    )?;
                }
            }
            writeln!(out)?;
        }
    }

    match victory_shader::native_programs(d) {
        Ok(programs) if programs.is_empty() => {
            writeln!(out, "  native reader: none (a PC layout or no tag 0x0d)")?;
        }
        Ok(programs) => {
            writeln!(out, "  native reader: {} program(s)", programs.len())?;
            for (i, p) in programs.iter().enumerate() {
                writeln!(
                    out,
                    "    {i}: stage {:?}, {}, {} code bytes, {}/{} root parameters/used resources, {} VGPRs",
                    p.stage,
                    if p.wave64 { "wave64" } else { "wave32" },
                    p.code.len(),
                    p.layout.len(),
                    p.used.len(),
                    p.vgprs
                )?;
            }
        }
        Err(e) => writeln!(out, "  native reader: refused: {e}")?,
    }
    Ok(out)
}

/// The profile a DXIL `ProgramVersion` names (`ps_6_6`, `cs_6_6`, ...).
fn profile(version: u32) -> String {
    let stage = match version >> 16 {
        0 => "ps",
        1 => "vs",
        2 => "gs",
        3 => "hs",
        4 => "ds",
        5 => "cs",
        6 => "lib",
        7 => "raygeneration",
        8 => "intersection",
        9 => "anyhit",
        10 => "closesthit",
        11 => "miss",
        12 => "callable",
        13 => "ms",
        14 => "as",
        15 => "node",
        _ => "unknown",
    };
    format!("{stage}_{}_{}", (version >> 4) & 0xf, version & 0xf)
}

fn u32_at(data: &[u8], at: usize) -> Result<u32> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .with_context(|| format!("truncated at +{at:#x}"))
}

fn u64_at(data: &[u8], at: usize) -> Result<u64> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .with_context(|| format!("truncated at +{at:#x}"))
}

fn fourcc_str(fourcc: FourCC) -> &'static str {
    match fourcc {
        FourCC::ACCC => "ACCC",
        FourCC::DXIL => "DXIL",
        _ => "part",
    }
}

fn show_program(program: &NativeProgram) {
    println!(
        "   stage {:?}, wave{}, {}",
        program.stage,
        if program.wave64 { 64 } else { 32 },
        io_summary(program)
    );
    if !program.layout.is_empty() {
        println!("   root parameters:");
        for p in &program.layout {
            println!(
                "     slot {:3} {:?} user s{} ({} SGPRs) base {}",
                p.slot, p.root, p.user_data, p.sgprs, p.base_register
            );
        }
    }
    if !program.used.is_empty() {
        println!("   used resources:");
        for u in &program.used {
            println!(
                "     {:?} slot {} dword {} register {} ({} dwords)",
                u.kind, u.slot, u.table_offset_dwords, u.lower_bound, u.dwords
            );
        }
    }
}

fn io_summary(program: &NativeProgram) -> String {
    use victory_shader::native::StageIo;
    match &program.io {
        StageIo::Cs(cs) => format!(
            "numthreads({}, {}, {}), {} user SGPRs, {} LDS bytes",
            cs.threads[0],
            cs.threads[1],
            cs.threads[2],
            cs.user_sgprs(),
            cs.lds_bytes()
        ),
        StageIo::Ms(ms) => format!(
            "{} lanes, {} vertices, {} primitives, {} outputs",
            ms.threads,
            ms.max_vertices,
            ms.max_primitives,
            ms.outputs.len()
        ),
        StageIo::Ps { inputs, .. } => format!("{} inputs", inputs.len()),
        StageIo::Vs { inputs, outputs } => {
            format!("{} inputs, {} outputs", inputs.len(), outputs.len())
        }
        StageIo::None => "no stage data".into(),
    }
}

/// Whether the reference tables agree with ours wherever both name an opcode.
/// Returns whether the check passed (`false` only when `strict` and a table
/// names another instruction).
fn check_tables(path: &Path, strict: bool) -> Result<bool> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let reference = parse_reference(&text);
    if reference.is_empty() {
        bail!(
            "{} holds no recognized opcode table (is it RageForge's Shaders/Tables.cs?)",
            path.display()
        );
    }
    let mut same = 0usize;
    let mut only_ours = 0usize;
    let mut only_reference = Vec::new();
    let mut different = Vec::new();
    for (encoding, label) in disasm::encodings() {
        let ours: BTreeMap<u16, &'static str> = victory_shader::gfx10::opcode_table(encoding)
            .iter()
            .copied()
            .collect();
        let Some(theirs) = reference.get(label) else {
            continue;
        };
        for (&op, name) in theirs {
            match ours.get(&op) {
                Some(&our) if our == name => same += 1,
                Some(&our) => different.push(format!("{label} {op:#05x}: {our} vs {name}")),
                None => only_reference.push(format!("{label} {op:#05x}: {name}")),
            }
        }
        only_ours += ours
            .values()
            .filter(|n| !theirs.values().any(|t| t == *n))
            .count();
    }
    println!();
    println!(
        "opcode cross-check against {}: {same} agree, {} differ, {} only in RageForge, {} only in ours",
        path.display(),
        different.len(),
        only_reference.len(),
        only_ours
    );
    for line in &different {
        println!("  differs: {line}");
    }
    for line in only_reference.iter().take(16) {
        println!("  only in RageForge: {line}");
    }
    if let Some(vopc) = reference.get("vopc") {
        println!(
            "  (RageForge's vopc names are computed by a function; {} fixed entries skipped)",
            vopc.len()
        );
    }
    Ok(!strict || different.is_empty())
}

/// The `Dictionary<uint, string>` tables of RageForge's `Tables.cs`: each
/// `Map((op, "name"), ...)` or `Seq(start, "name", ...)` block, by the C#
/// name in lower case.
fn parse_reference(text: &str) -> BTreeMap<String, BTreeMap<u16, String>> {
    let mut out = BTreeMap::new();
    let mut rest = text;
    while let Some(at) = rest.find("Dictionary<uint, string>") {
        rest = &rest[at + "Dictionary<uint, string>".len()..];
        let Some(name) = rest
            .trim_start()
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .next()
        else {
            continue;
        };
        let Some(eq) = rest.find('=') else { break };
        let body_at = eq + 1;
        let body = &rest[body_at..];
        let end = body
            .find("};")
            .or_else(|| body.find(";\n"))
            .unwrap_or(body.len());
        let body = &body[..end];
        let mut table = BTreeMap::new();
        if let Some(seq) = body.find("Seq(") {
            let after = &body[seq + 4..];
            let start: u16 = after
                .split(',')
                .next()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            for (i, name) in string_literals(after).into_iter().enumerate() {
                if !name.is_empty() {
                    table.insert(start + i as u16, name);
                }
            }
        } else if let Some(map) = body.find("Map(") {
            let after = &body[map + 4..];
            for (op, name) in pairs(after) {
                table.insert(op, name);
            }
        }
        if !table.is_empty() {
            out.insert(name.to_ascii_lowercase(), table);
        }
        rest = &rest[body_at + end..];
    }
    out
}

fn string_literals(s: &str) -> Vec<String> {
    s.split('"')
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, s)| s.to_string())
        .collect()
}

/// `(op, "name")` pairs, in the source's order.
fn pairs(s: &str) -> Vec<(u16, String)> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(at) = rest.find('(') {
        rest = &rest[at + 1..];
        let Some(comma) = rest.find(',') else { break };
        let op = rest[..comma].trim().parse::<u16>();
        let after = &rest[comma + 1..];
        let Some(quote) = after.find('"') else {
            rest = after;
            continue;
        };
        let quoted = &after[quote + 1..];
        let Some(end) = quoted.find('"') else { break };
        if let Ok(op) = op {
            out.push((op, quoted[..end].to_string()));
        }
        rest = &quoted[end + 1..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_found_in_a_log_line() {
        assert_eq!(
            normalize("compute 39ef10815cb825a0").as_deref(),
            Some("39ef10815cb825a0")
        );
        assert_eq!(
            normalize("vertex_39EF10815CB825A0.bin").as_deref(),
            Some("39ef10815cb825a0")
        );
        assert_eq!(normalize("not a hash"), None);
    }

    #[test]
    fn the_reference_parser_reads_map_and_seq() {
        let text = r#"
            public static readonly Dictionary<uint, string> Sop2 = Seq(0, "s_add_u32", "s_sub_u32");
            public static readonly Dictionary<uint, string> Ds = Map((0, "ds_add_u32"), (54, "ds_read_b32"));
        "#;
        let tables = parse_reference(text);
        assert_eq!(tables["sop2"][&1], "s_sub_u32");
        assert_eq!(tables["ds"][&54], "ds_read_b32");
    }

    /// A made-up reference table that names another instruction fails the
    /// strict check; ours alone does not.
    #[test]
    fn a_name_that_differs_fails_the_check() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Tables.cs");
        std::fs::write(
            &path,
            r#"public static readonly Dictionary<uint, string> Sop2 = Seq(0, "s_not_the_name");"#,
        )
        .unwrap();
        assert!(!check_tables(&path, true).unwrap());
        std::fs::write(
            &path,
            r#"public static readonly Dictionary<uint, string> Sop2 = Seq(0, "s_add_u32");"#,
        )
        .unwrap();
        assert!(check_tables(&path, true).unwrap());
        std::fs::write(&path, "").unwrap();
        assert!(check_tables(&path, true).is_err());
    }

    /// A DXIL part with an XDAT index at `at`, no entries.
    fn xdat_part(at: usize) -> Vec<u8> {
        let mut part = vec![0u8; at];
        part.extend_from_slice(b"XDAT");
        part.extend_from_slice(&0u32.to_le_bytes());
        part.extend_from_slice(&0x24u32.to_le_bytes());
        part
    }

    #[test]
    fn the_probe_names_the_reader_offset_and_its_verdict() {
        let bytes = victory_shader::dxbc::build(&[(FourCC::DXIL, &xdat_part(0x18))]);
        let text = probe_text(&bytes).unwrap();
        assert!(text.contains("XDAT at +0x18"), "{text}");
        assert!(text.contains("the native reader's offset"), "{text}");
        assert!(text.contains("native reader: none"), "{text}");
    }

    /// An XDAT anywhere else is reported as a candidate with the reader's
    /// offset named, which is how a compiler output that carries bitcode
    /// before its index shows up.
    #[test]
    fn the_probe_reports_an_xdat_the_reader_does_not_read() {
        let bytes = victory_shader::dxbc::build(&[(FourCC::DXIL, &xdat_part(0x1a))]);
        let text = probe_text(&bytes).unwrap();
        assert!(text.contains("XDAT at +0x1a"), "{text}");
        assert!(text.contains("does not look here"), "{text}");
        assert!(text.contains("native reader: none"), "{text}");
    }

    #[test]
    fn the_probe_needs_a_shader_part() {
        let bytes = victory_shader::dxbc::build(&[(FourCC::SFI0, &[0; 4])]);
        assert!(probe_text(&bytes).is_err());
    }
}
