//! Debug listings of native programs: GFX10 disassembly and a pseudo-HLSL
//! reading of the same code.
//!
//! The disassembly is the one the translator decodes ([`gfx10::decode`]),
//! printed in AMD's SP3 syntax with each instruction's raw dwords beside it.
//! The pseudo-HLSL follows the SSA form ([`ir::build`]): every register write
//! becomes a named value, so a line can be followed through the program even
//! where the compiler reuses a register. It is a reading aid, not a program:
//! lane masks, EXEC and the host's wave split are shown as operations on
//! whole-wave values, and no attempt is made to make it compile.
//!
//! `xtask shader` prints both, finds containers by the hash a run log names,
//! and cross-checks our opcode tables against RageForge's reference tables
//! (`tools/`, "Shader debugging"). Game data stays outside the repository.

use std::collections::HashMap;
use std::fmt::Write;

use crate::gfx10::{self, Encoding, Fields, Operand};
use crate::ir::{self, Block, Exit, Inst, Program, Reg, Value};
use crate::native::StageIo;
use crate::{HwStage, NativeProgram, ShaderError};

/// The program's instructions, one a line: offset, dwords, mnemonic and
/// operands. `patched` applies the pixel shader's export patches, as the
/// console does at pipeline creation; `false` lists the code as stored.
pub fn disassembly(program: &NativeProgram, patched: bool) -> Result<String, ShaderError> {
    let code = if patched {
        program.patched_code()
    } else {
        program.code.clone()
    };
    let list = gfx10::decode(&code)?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "; {} {:?} wave{} code {} bytes ({} instructions)",
        stage_name(program.stage),
        program.io,
        if program.wave64 { 64 } else { 32 },
        code.len(),
        list.len()
    );
    for i in &list {
        let dwords: Vec<String> = i.words.iter().map(|w| format!("{w:08x}")).collect();
        let _ = writeln!(
            out,
            "{:06x}:  {:<27}  {}",
            i.offset,
            dwords.join(" "),
            i.text(program.wave64)
        );
    }
    Ok(out)
}

/// The program's instructions as pseudo-HLSL: one `value = expression` line a
/// machine instruction, with labels for branch targets and the block's join
/// values. `None` for a program the tracer cannot follow.
pub fn pseudo_hlsl(program: &NativeProgram) -> Result<String, ShaderError> {
    let p = ir::build(&program.code, program.wave64, program.vgprs)?;
    let mut out = String::new();
    let entry = match program.stage {
        HwStage::Ps => "PSMain",
        HwStage::Vs => "VSMain",
        HwStage::Ms => "MSMain",
        HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs => "HSMain",
        HwStage::Cs => "CSMain",
    };
    match &program.io {
        StageIo::Cs(cs) => {
            let [x, y, z] = cs.threads;
            let _ = writeln!(out, "[numthreads({x}, {y}, {z})]");
        }
        StageIo::Ms(ms) => {
            let _ = writeln!(
                out,
                "[numthreads({}, 1, 1)]  // {} vertices, {} primitives",
                ms.threads, ms.max_vertices, ms.max_primitives
            );
        }
        _ => {}
    }
    let _ = writeln!(
        out,
        "// {} {:?} wave{}, lifted from GFX10.1 ISA",
        stage_name(program.stage),
        program.stage,
        if program.wave64 { 64 } else { 32 }
    );
    let _ = writeln!(out, "void {entry}() {{");
    let mut names = Names::new(&p);
    for (b, block) in p.blocks.iter().enumerate() {
        let _ = writeln!(out, "L{b}_{:04x}:", block.offset);
        for phi in &block.phis {
            let dst = names.define(phi.value, phi.reg);
            let args: Vec<String> = phi
                .args
                .iter()
                .zip(&block.preds)
                .map(|(&v, &q)| format!("L{q}: {}", names.of(v, None)))
                .collect();
            let _ = writeln!(out, "  {} = phi({});", dst, args.join(", "));
        }
        for inst in &block.insts {
            let line = statement(&mut names, inst);
            let _ = writeln!(
                out,
                "  {line:<52} // {:04x}: {}",
                inst.machine.offset,
                inst.machine.text(program.wave64)
            );
        }
        match block.exit {
            Exit::Jump(q) => {
                let _ = writeln!(out, "  goto L{q};");
            }
            Exit::Branch { taken, not_taken } => {
                let cond = block_cond(block).unwrap_or("scc");
                let _ = writeln!(out, "  if ({cond}) goto L{taken}; else goto L{not_taken};");
            }
            Exit::End => {
                let _ = writeln!(out, "  return;");
            }
            Exit::Indirect => {
                let _ = writeln!(
                    out,
                    "  return;  // s_setpc_b64: a handover this listing ends at"
                );
            }
        }
    }
    let _ = writeln!(out, "}}");
    Ok(out)
}

fn stage_name(stage: HwStage) -> &'static str {
    match stage {
        HwStage::Ls => "ls",
        HwStage::Hs => "hs",
        HwStage::Es => "es",
        HwStage::Gs => "gs",
        HwStage::Vs => "vs",
        HwStage::Ps => "ps",
        HwStage::Ms => "ms",
        HwStage::Cs => "cs",
    }
}

/// The condition a block's last `s_cbranch_*` tests.
fn block_cond(block: &Block) -> Option<&'static str> {
    let name = block.insts.last()?.name();
    Some(match name {
        "s_cbranch_scc0" | "s_cbranch_scc1" => "scc",
        "s_cbranch_vccz" | "s_cbranch_vccnz" => "vccz",
        "s_cbranch_execz" | "s_cbranch_execnz" => "execz",
        _ => return None,
    })
}

/// Names for the SSA values a listing prints. A value is named after the
/// register it lands in and a counter, so two writes to `v3` are `v3_0` and
/// `v3_1`; a program input keeps the register name alone.
struct Names {
    names: HashMap<Value, String>,
    next: HashMap<Reg, u32>,
}

impl Names {
    fn new(p: &Program) -> Self {
        let mut names = HashMap::new();
        for &(reg, value) in &p.inputs {
            names.insert(value, reg_name(reg));
        }
        Self {
            names,
            next: HashMap::new(),
        }
    }

    fn name_for(&mut self, reg: Reg) -> String {
        let n = self.next.entry(reg).or_insert(0);
        let name = format!("{}_{n}", reg_name(reg));
        *n += 1;
        name
    }

    fn define(&mut self, value: Value, reg: Reg) -> String {
        if let Some(n) = self.names.get(&value) {
            return n.clone();
        }
        let n = self.name_for(reg);
        self.names.insert(value, n.clone());
        n
    }

    fn of(&mut self, value: Value, reg: Option<Reg>) -> String {
        if let Some(n) = self.names.get(&value) {
            return n.clone();
        }
        let n = match reg {
            Some(r) => self.name_for(r),
            None => format!("t{}", self.next.len()),
        };
        self.names.insert(value, n.clone());
        n
    }
}

fn reg_name(reg: Reg) -> String {
    match reg {
        Reg::S(106) => "vcc".into(),
        Reg::S(126) => "exec".into(),
        Reg::S(c) => format!("s{c}"),
        Reg::Scc => "scc".into(),
        Reg::V(v) => format!("v{v}"),
    }
}

/// One instruction's writes as pseudo-HLSL assignments, and a store or an
/// export as its side effect.
fn statement(names: &mut Names, inst: &Inst) -> String {
    let name = inst.name();
    match &inst.fields {
        Fields::Sop { dst, src, imm } => {
            let args: Vec<String> = src.iter().map(|o| operand(inst, names, o, name)).collect();
            let expr = match name {
                "s_mov_b32" | "s_mov_b64" => args.first().cloned().unwrap_or_default(),
                "s_not_b32" | "s_not_b64" => format!("~{}", args[0]),
                "s_add_u32" | "s_add_i32" | "s_addc_u32" => format!("{} + {}", args[0], args[1]),
                "s_sub_u32" | "s_sub_i32" | "s_subb_u32" => format!("{} - {}", args[0], args[1]),
                "s_mul_i32" => format!("{} * {}", args[0], args[1]),
                "s_and_b32" | "s_and_b64" => format!("{} & {}", args[0], args[1]),
                "s_or_b32" | "s_or_b64" => format!("{} | {}", args[0], args[1]),
                "s_xor_b32" | "s_xor_b64" => format!("{} ^ {}", args[0], args[1]),
                "s_lshl_b32" | "s_lshl_b64" => format!("{} << {}", args[0], args[1]),
                "s_lshr_b32" | "s_lshr_b64" => format!("{} >> {}", args[0], args[1]),
                "s_ashr_i32" | "s_ashr_i64" => format!("{} >>s {}", args[0], args[1]),
                "s_cselect_b32" | "s_cselect_b64" => format!("scc ? {} : {}", args[1], args[0]),
                "s_cmp_eq_i32" | "s_cmp_eq_u32" => format!("{} == {}", args[0], args[1]),
                "s_cmp_lg_i32" | "s_cmp_lg_u32" => format!("{} != {}", args[0], args[1]),
                "s_cmp_gt_i32" | "s_cmp_gt_u32" => format!("{} > {}", args[0], args[1]),
                "s_cmp_ge_i32" | "s_cmp_ge_u32" => format!("{} >= {}", args[0], args[1]),
                "s_cmp_lt_i32" | "s_cmp_lt_u32" => format!("{} < {}", args[0], args[1]),
                "s_cmp_le_i32" | "s_cmp_le_u32" => format!("{} <= {}", args[0], args[1]),
                _ if name.starts_with("s_cmp") => format!("cmp({})", args.join(", ")),
                _ if name == "s_getpc_b64" => "pc".into(),
                _ => call(name, &args),
            };
            let _ = imm;
            assign(inst, names, dst.iter().copied(), &expr)
        }
        Fields::Sopp { imm } => match name {
            "s_endpgm" | "s_code_end" | "s_endpgm_saved" => "return".into(),
            "s_barrier" => "DeviceBarrier()".into(),
            "s_waitcnt" => format!("WaitCnt(0x{imm:04x})"),
            "s_nop" => "// nop".into(),
            "s_branch" => "// branch".into(),
            n if n.starts_with("s_cbranch_") => "// branch".into(),
            _ => call(name, &[]),
        },
        Fields::Smem {
            data,
            base,
            offset,
            soffset,
            ..
        } => {
            let addr = format!(
                "{}[{}]",
                operand(inst, names, base, name),
                scalar_off(inst, names, soffset, *offset, name)
            );
            let expr = match name {
                n if n.starts_with("s_load") => addr,
                n if n.starts_with("s_buffer_load") => format!("cbuffer{addr}"),
                n if n.starts_with("s_store") || n.starts_with("s_buffer_store") => {
                    let value = data.map_or("?".into(), |o| operand(inst, names, &o, name));
                    return format!("{addr} = {value};");
                }
                _ => call(name, &[]),
            };
            assign(inst, names, data.iter().copied(), &expr)
        }
        Fields::Valu(v) => {
            let mut args: Vec<String> = v
                .src
                .iter()
                .map(|o| operand(inst, names, o, name))
                .collect();
            // These read their destination as an accumulator before writing it.
            if matches!(
                name,
                "v_dot4c_i32_i8"
                    | "v_dot2c_f32_f16"
                    | "v_fmac_f32"
                    | "v_fmac_f16"
                    | "v_fmac_legacy_f32"
                    | "v_mac_f32"
                    | "v_pk_fmac_f16"
            ) && let Some(Operand::Vector { reg, .. }) = v.dst
                && let Some(old) = inst.read(Reg::V(reg))
            {
                args.push(names.of(old, Some(Reg::V(reg))));
            }
            let expr = valu_expr(name, &args, v);
            let dst: Vec<Operand> = v.dst.into_iter().chain(v.carry).collect();
            assign(inst, names, dst, &expr)
        }
        Fields::Vintrp { dst, src, attr } => {
            let a = format!(
                "{}[{}.{}]",
                operand(inst, names, src, name),
                "attr",
                operand(inst, names, attr, name)
            );
            assign(inst, names, [*dst], &format!("interpolate({a})"))
        }
        Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            ..
        } => {
            let base = addr.map_or("0".into(), |o| operand(inst, names, &o, name));
            // Pair operations address with two 8-bit offsets; the rest with
            // one 16-bit offset, low byte first.
            let offset = if ds_pair(name) {
                u32::from(*offset1) << 8 | u32::from(*offset0)
            } else {
                u32::from(*offset0)
            };
            let at = format!("lds[{base} + {offset}]");
            if let Some(d) = dst {
                let ty = data_type(name);
                return assign(inst, names, [*d], &format!("{at} /* {ty} */"));
            }
            let values: Vec<String> = data.iter().map(|o| operand(inst, names, o, name)).collect();
            if values.is_empty() {
                call(name, &[])
            } else {
                format!("{at} = {};  // {name}", values.join(" ").trim())
            }
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offset,
            idxen,
            offen,
            lds,
            ..
        } => {
            let idx = match (idxen, offen) {
                (true, true) => format!(
                    "{} + {}",
                    operand(inst, names, addr, name),
                    operand(inst, names, soffset, name)
                ),
                (false, true) => format!(
                    "{} + {}",
                    operand(inst, names, soffset, name),
                    operand(inst, names, addr, name)
                ),
                _ => operand(inst, names, addr, name),
            };
            let at = format!("buf{}{idx} + {offset}", operand(inst, names, rsrc, name));
            let at = if *lds { format!("lds[{at}]") } else { at };
            if name.contains("_store") {
                let value = data.map_or("?".into(), |o| operand(inst, names, &o, name));
                return format!("{at} = {value};  // {name}");
            }
            let expr = format!("{at} /* {} */", data_type(name));
            assign(inst, names, data.iter().copied(), &expr)
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            dmask,
            dim,
            ..
        } => {
            let coords: Vec<String> = addr.iter().map(|o| operand(inst, names, o, name)).collect();
            let resource = operand(inst, names, rsrc, name);
            let dims = ["1D", "2D", "3D", "CUBE", "1D[]", "2D[]", "MSAA", "MSAA[]"];
            let coords = coords.join(", ");
            let expr = if name.starts_with("image_store") {
                let value = operand(inst, names, data, name);
                return format!(
                    "tex{resource}[{coords}] = {value};  // {name} dmask {dmask:#x} {}",
                    dims[usize::from(*dim & 7)]
                );
            } else if name.starts_with("image_load") || name.starts_with("image_get_resinfo") {
                format!(
                    "tex{resource}[{coords}] /* dmask {dmask:#x} {} */",
                    dims[usize::from(*dim & 7)]
                )
            } else {
                let s = sampler.map_or("?".into(), |o| operand(inst, names, &o, name));
                format!("tex{resource}{}(sampler{s}, {coords})", image_hlsl(name))
            };
            assign(inst, names, [*data], &expr)
        }
        Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            ..
        } => {
            let a = addr.map_or("0".into(), |o| operand(inst, names, &o, name));
            let s = saddr.map_or("0".into(), |o| operand(inst, names, &o, name));
            let at = format!("mem[{a} + {s} + {offset}]");
            if let Some(d) = dst {
                return assign(
                    inst,
                    names,
                    [*d],
                    &format!("{at} /* {} */", data_type(name)),
                );
            }
            let value = data.map_or("?".into(), |o| operand(inst, names, &o, name));
            format!("{at} = {value};  // {name}")
        }
        Fields::Exp {
            target,
            src,
            compr,
            done,
            vm,
        } => {
            let target_name = match target {
                0..=7 => format!("mrt{target}"),
                8 => "mrtz".into(),
                9 => "null".into(),
                12..=15 => format!("pos{}", target - 12),
                20 => "prim".into(),
                32..=63 => format!("param{}", target - 32),
                _ => format!("target{target}"),
            };
            let values: Vec<String> = src
                .iter()
                .map(|s| s.map_or("off".into(), |o| operand(inst, names, &o, name)))
                .collect();
            let mut suffix = String::new();
            if *compr {
                suffix.push_str(" compr");
            }
            if *done {
                suffix.push_str(" done");
            }
            if *vm {
                suffix.push_str(" vm");
            }
            format!("out.{target_name} = ({}){suffix};", values.join(", "))
        }
    }
}

fn assign(
    inst: &Inst,
    names: &mut Names,
    dst: impl IntoIterator<Item = Operand>,
    expr: &str,
) -> String {
    let dst: Vec<Operand> = dst.into_iter().collect();
    if let [Operand::Vector { reg, dwords: 1 }] = dst[..]
        && let Some(v) = inst.write(Reg::V(reg))
    {
        let n = names.define(v, Reg::V(reg));
        return format!("{n} = {expr};");
    }
    if let [Operand::Scalar { code, dwords: 1 }] = dst[..]
        && code < 106
        && let Some(v) = inst.write(Reg::S(code))
    {
        let n = names.define(v, Reg::S(code));
        return format!("{n} = {expr};");
    }
    // A range or a lane mask: name each dword the instruction writes.
    let mut lines = Vec::new();
    for (k, d) in dst.iter().enumerate() {
        match *d {
            Operand::Vector { reg, .. } => {
                if let Some(v) = inst.write(Reg::V(reg + k as u8)) {
                    let n = names.define(v, Reg::V(reg + k as u8));
                    lines.push(format!("{n} = {expr}[{k}];"));
                }
            }
            Operand::Scalar { code, .. } => {
                if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                    let n = names.define(v, Reg::S(code + k as u8));
                    lines.push(format!("{n} = {expr}[{k}];"));
                }
            }
            _ => {}
        }
    }
    if lines.is_empty() {
        format!("// {expr};")
    } else {
        lines.join(" ")
    }
}

fn operand(inst: &Inst, names: &mut Names, o: &Operand, op: &str) -> String {
    match *o {
        Operand::Vector { reg, dwords: 1 } => {
            let v = inst.read(Reg::V(reg));
            names.of(v.unwrap_or(Value(u32::MAX)), v.map(|_| Reg::V(reg)))
        }
        Operand::Vector { reg, dwords } => format!("v[{reg}:{}]", reg as u32 + dwords as u32 - 1),
        Operand::Scalar { code: 125, .. } => "0".into(),
        Operand::Scalar { code: 106, .. } => {
            let v = inst.read(Reg::S(106));
            names.of(v.unwrap_or(Value(u32::MAX)), v.map(|_| Reg::S(106)))
        }
        Operand::Scalar { code: 126, .. } => {
            let v = inst.read(Reg::S(126));
            names.of(v.unwrap_or(Value(u32::MAX)), v.map(|_| Reg::S(126)))
        }
        Operand::Scalar { code, dwords: 1 } if code < 106 => {
            let v = inst.read(Reg::S(code));
            names.of(v.unwrap_or(Value(u32::MAX)), v.map(|_| Reg::S(code)))
        }
        Operand::Scalar { code, dwords } if code < 106 => {
            format!("s[{code}:{}]", code + dwords - 1)
        }
        Operand::Scalar { code, .. } => format!("s{code}"),
        Operand::Const(c) => constant(c, op),
        Operand::Literal(x) => literal(x, op),
        Operand::Attr { index, chan, .. } => {
            format!(
                "attr{index}.{}",
                ['x', 'y', 'z', 'w'][usize::from(chan & 3)]
            )
        }
    }
}

fn scalar_off(inst: &Inst, names: &mut Names, soffset: &Operand, offset: i32, op: &str) -> String {
    let s = match soffset {
        Operand::Scalar { code: 125, .. } => String::new(),
        o => format!(" + {}", operand(inst, names, o, op)),
    };
    if offset == 0 {
        s.trim_start_matches(" + ").to_string()
    } else {
        format!("{offset:#x}{s}")
    }
}

fn literal(x: u32, op: &str) -> String {
    if is_float_op(op) {
        format!("{:?}", f32::from_bits(x))
    } else {
        format!("0x{x:08x}")
    }
}

fn constant(c: u8, op: &str) -> String {
    match c {
        128..=192 => (i32::from(c) - 128).to_string(),
        193..=208 => (192 - i32::from(c)).to_string(),
        240 => "0.5".into(),
        241 => "-0.5".into(),
        242 => "1.0".into(),
        243 => "-1.0".into(),
        244 => "2.0".into(),
        245 => "-2.0".into(),
        246 => "4.0".into(),
        247 => "-4.0".into(),
        248 => "0.15915494".into(),
        251 => "vccz".into(),
        252 => "execz".into(),
        253 => "scc".into(),
        _ if is_float_op(op) => format!("f32::from_bits(0x{c:02x})"),
        _ => format!("0x{c:02x}"),
    }
}

/// Whether a DS instruction holds two addresses in its offset fields.
fn ds_pair(name: &str) -> bool {
    name.contains("read2") || name.contains("write2") || name.contains("addtid") || {
        // The other pair forms have their second operand in `offset1`.
        matches!(
            name,
            "ds_cmpst_rtn_b32"
                | "ds_cmpst_rtn_b64"
                | "ds_mskor_rtn_b32"
                | "ds_wrxchg2_rtn_b32"
                | "ds_wrxchg2st64_rtn_b32"
                | "ds_condxchg32_rtn_b64"
        )
    }
}

fn is_float_op(name: &str) -> bool {
    name.contains("_f32") || name.contains("_f16") || name.contains("_f64")
}

fn data_type(name: &str) -> &'static str {
    if name.contains("ubyte") {
        "u8"
    } else if name.contains("sbyte") {
        "i8"
    } else if name.contains("ushort") || name.contains("u16") {
        "u16"
    } else if name.contains("sshort") || name.contains("i16") {
        "i16"
    } else if name.contains("dwordx4") || name.contains("b128") {
        "uint4"
    } else if name.contains("dwordx3") || name.contains("b96") {
        "uint3"
    } else if name.contains("dwordx2") || name.contains("b64") {
        "uint2"
    } else if name.contains("_f32") {
        "float"
    } else {
        "uint"
    }
}

fn call(name: &str, args: &[String]) -> String {
    format!("{name}({})", args.join(", "))
}

fn valu_expr(name: &str, a: &[String], v: &gfx10::Valu) -> String {
    let at = |k: usize| a.get(k).map(String::as_str).unwrap_or("?");
    let expr = match name {
        "v_mov_b32" => at(0).to_string(),
        "v_nop" => "// nop".into(),
        "v_add_f32" | "v_add_nc_u32" | "v_add_f16" | "v_add_i32" | "v_add_u32" | "v_add_nc_i32"
        | "v_add_nc_i16" | "v_add_nc_u16" => format!("{} + {}", at(0), at(1)),
        "v_sub_f32" | "v_sub_nc_u32" | "v_sub_f16" | "v_sub_i32" | "v_sub_u32" | "v_sub_nc_i32"
        | "v_sub_nc_i16" | "v_sub_nc_u16" => format!("{} - {}", at(0), at(1)),
        "v_subrev_f32" | "v_subrev_nc_u32" | "v_subrev_f16" | "v_subrev_nc_u16" => {
            format!("{} - {}", at(1), at(0))
        }
        "v_mul_f32" | "v_mul_legacy_f32" | "v_mul_f16" | "v_mul_lo_u32" | "v_mul_u32_u24"
        | "v_mul_i32_i24" | "v_mul_lo_u16" => format!("{} * {}", at(0), at(1)),
        "v_mul_hi_u32" | "v_mul_hi_i32" => format!("mulhi({}, {})", at(0), at(1)),
        "v_fma_f32" | "v_fma_f16" | "v_fma_f64" => {
            format!("fma({}, {}, {})", at(0), at(1), at(2))
        }
        "v_fmac_f32" | "v_fmac_f16" | "v_mac_f32" | "v_fmac_legacy_f32" => {
            format!("{} * {} + {v}", at(0), at(1), v = at(2))
        }
        "v_mad_f32" | "v_mad_legacy_f32" | "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mad_u16"
        | "v_mad_i16" | "v_mad_u32_u16" | "v_mad_i32_i16" => {
            format!("fma({}, {}, {})", at(0), at(1), at(2))
        }
        "v_cndmask_b32" => format!("{} ? {} : {}", at(2), at(1), at(0)),
        "v_min_f32" | "v_min_i32" | "v_min_u32" | "v_min_f16" | "v_min_u16" => {
            format!("min({}, {})", at(0), at(1))
        }
        "v_max_f32" | "v_max_i32" | "v_max_u32" | "v_max_f16" | "v_max_u16" => {
            format!("max({}, {})", at(0), at(1))
        }
        "v_min3_f32" | "v_min3_i32" | "v_min3_u32" => {
            format!("min({}, {}, {})", at(0), at(1), at(2))
        }
        "v_max3_f32" | "v_max3_i32" | "v_max3_u32" => {
            format!("max({}, {}, {})", at(0), at(1), at(2))
        }
        "v_med3_f32" | "v_med3_i32" | "v_med3_u32" => {
            format!("median({}, {}, {})", at(0), at(1), at(2))
        }
        "v_rcp_f32" | "v_rcp_iflag_f32" => format!("1.0 / {}", at(0)),
        "v_rsq_f32" => format!("rsqrt({})", at(0)),
        "v_sqrt_f32" => format!("sqrt({})", at(0)),
        "v_exp_f32" => format!("exp2({})", at(0)),
        "v_log_f32" => format!("log2({})", at(0)),
        "v_sin_f32" => format!("sin({} * 6.2831853)", at(0)),
        "v_cos_f32" => format!("cos({} * 6.2831853)", at(0)),
        "v_floor_f32" => format!("floor({})", at(0)),
        "v_ceil_f32" => format!("ceil({})", at(0)),
        "v_fract_f32" => format!("frac({})", at(0)),
        "v_trunc_f32" => format!("trunc({})", at(0)),
        "v_rndne_f32" => format!("round({})", at(0)),
        "v_cvt_f32_i32" | "v_cvt_f32_u32" => format!("(float){}", at(0)),
        "v_cvt_u32_f32" => format!("(uint){}", at(0)),
        "v_cvt_i32_f32" => format!("(int){}", at(0)),
        "v_cvt_f16_f32" => format!("(half){}", at(0)),
        "v_cvt_f32_f16" => format!("(float)(half){}", at(0)),
        "v_cvt_pkrtz_f16_f32" => format!("pack_half2({}, {})", at(0), at(1)),
        "v_cvt_pk_u8_f32" => format!("pack_u8({}, {}, {})", at(0), at(1), at(2)),
        "v_and_b32" => format!("{} & {}", at(0), at(1)),
        "v_or_b32" => format!("{} | {}", at(0), at(1)),
        "v_xor_b32" => format!("{} ^ {}", at(0), at(1)),
        "v_xnor_b32" => format!("~({} ^ {})", at(0), at(1)),
        "v_lshlrev_b32" => format!("{} << {}", at(1), at(0)),
        "v_lshrrev_b32" => format!("{} >> {}", at(1), at(0)),
        "v_ashrrev_i32" => format!("{} >>s {}", at(1), at(0)),
        "v_readfirstlane_b32" => format!("readfirstlane({})", at(0)),
        "v_readlane_b32" => format!("readlane({}, {})", at(0), at(1)),
        "v_writelane_b32" => format!("writelane({}, {})", at(1), at(2)),
        "v_dot4_i32_i8" | "v_dot4c_i32_i8" => {
            format!("dot4add_i8({}, {}, {})", at(0), at(1), at(2))
        }
        "v_dot2_f32_f16" | "v_dot2c_f32_f16" => {
            format!("dot2add_f16({}, {}, {})", at(0), at(1), at(2))
        }
        n if n.starts_with("v_cmp_") && !n.starts_with("v_cmpx_") => {
            format!("vcc = cmp_{}({}, {})", &n[6..], at(0), at(1))
        }
        n if n.starts_with("v_cmpx_") => {
            format!("exec = cmp_{}({}, {})", &n[7..], at(0), at(1))
        }
        _ => call(name, a),
    };
    // A clamp or an output modifier is part of the value.
    let mut expr = expr;
    if v.clamp {
        expr = format!("saturate({expr})");
    }
    match v.omod & 3 {
        1 => format!("({expr}) * 2"),
        2 => format!("({expr}) * 4"),
        3 => format!("({expr}) * 0.5"),
        _ => expr,
    }
}

/// The HLSL-like operation a GFX10 image instruction reads as. The console
/// variants differ in ways HLSL spells differently: a comparison sample is
/// `SampleCmp`, `_lz` is level zero, and gathers are `Gather`/`GatherCmp`.
fn image_hlsl(name: &str) -> &'static str {
    if name.starts_with("image_gather4_c") {
        ".GatherCmp"
    } else if name.starts_with("image_gather4") {
        ".Gather"
    } else if name.starts_with("image_sample_c_lz") {
        ".SampleCmpLevelZero"
    } else if name.starts_with("image_sample_c_l") {
        ".SampleCmpLevel"
    } else if name.starts_with("image_sample_c_b") {
        ".SampleCmpBias"
    } else if name.starts_with("image_sample_c_d") {
        ".SampleCmpGrad"
    } else if name.starts_with("image_sample_c") {
        ".SampleCmp"
    } else if name.starts_with("image_sample_lz") || name.starts_with("image_sample_l") {
        ".SampleLevel"
    } else if name.starts_with("image_sample_b") {
        ".SampleBias"
    } else if name.starts_with("image_sample_d") {
        ".SampleGrad"
    } else {
        ".Sample"
    }
}

/// Whether `encoding` is one a listing prints by name, for the cross-check
/// with a reference disassembler's table.
pub fn encodings() -> [(Encoding, &'static str); 16] {
    [
        (Encoding::Sop2, "sop2"),
        (Encoding::Sopk, "sopk"),
        (Encoding::Sop1, "sop1"),
        (Encoding::Sopc, "sopc"),
        (Encoding::Sopp, "sopp"),
        (Encoding::Smem, "smem"),
        (Encoding::Vop2, "vop2"),
        (Encoding::Vop1, "vop1"),
        (Encoding::Vopc, "vopc"),
        (Encoding::Vop3, "vop3"),
        (Encoding::Vop3p, "vop3p"),
        (Encoding::Vintrp, "vintrp"),
        (Encoding::Ds, "ds"),
        (Encoding::Mubuf, "mubuf"),
        (Encoding::Mtbuf, "mtbuf"),
        (Encoding::Mimg, "mimg"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HwStage;
    use crate::native::{ComputeIo, StageIo};

    fn program(code: &[u32]) -> NativeProgram {
        let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 32,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code,
        }
    }

    /// s_mov_b32 s0, 3; v_mov_b32 v1, s0; v_add_f32 v2, v1, 1.0; exp mrt0 ...
    #[test]
    fn a_listing_names_each_write() {
        let p = program(&[
            0xBE80_0383, // s_mov_b32 s0, 3
            0x7E02_0200, // v_mov_b32 v1, s0
            0x0604_02F2, // v_add_f32 v2, 1.0, v1
            0xBF81_0000, // s_endpgm
        ]);
        let text = pseudo_hlsl(&p).unwrap();
        assert!(text.contains("s0_0 = 3;"), "{text}");
        assert!(text.contains("v1_0 = s0_0;"), "{text}");
        assert!(text.contains("+"), "{text}");
        let listing = disassembly(&p, false).unwrap();
        assert!(listing.contains("s_mov_b32 s0, 3"), "{listing}");
        assert!(listing.contains("v_add_f32 v2, 1.0, v1"), "{listing}");
    }

    /// An instruction the tables name is listed by that name; a branch target
    /// gets a label at its block's offset.
    #[test]
    fn branches_get_labels() {
        let p = program(&[
            0xBF84_0001, // s_cbranch_scc0 +1 (past the first s_endpgm)
            0xBF81_0000, // s_endpgm
            0xBF81_0000, // s_endpgm
        ]);
        let text = pseudo_hlsl(&p).unwrap();
        assert!(text.contains("if (scc) goto L"), "{text}");
        assert!(text.contains("return;"), "{text}");
    }

    /// A comparison sample and a comparison gather read as their HLSL
    /// operations, not as a plain `Sample`.
    #[test]
    fn image_calls_are_named_by_their_hlsl_operation() {
        // image_sample_c_lz v2, [v4,v2,v3], s[0:7], s[8:11] dim:2D
        let p = program(&[0xF0BC_010A, 0x0040_0204, 0x0000_0302, 0xBF81_0000]);
        let text = pseudo_hlsl(&p).unwrap();
        assert!(text.contains(".SampleCmpLevelZero("), "{text}");
        assert!(!text.contains(".Sample("), "{text}");

        // image_gather4_c_lz v[10:13], v[2:4], s[8:15], s[16:19] dim:2D
        let p = program(&[0xF13C_0108, 0x0082_0A02, 0xBF81_0000]);
        let text = pseudo_hlsl(&p).unwrap();
        assert!(text.contains(".GatherCmp("), "{text}");
        assert!(!text.contains(".Sample("), "{text}");
    }
}
