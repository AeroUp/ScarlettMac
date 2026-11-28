# Shader differential: Scarlett dxc output against the native reader

A small offline check that the container the GDK's Scarlett shader compiler
writes is the one `victory_shader::native_programs` expects, and that the
native programs inside it translate. It is deliberately not a game run and not
a corpus sweep: the inputs are three original HLSL shaders and one
hand-assembled GFX10 program.

The interesting questions are narrow:

- Does the compiler's output put `XDAT` at `+0x18` of the `DXIL` part, as the
  reader assumes, and does it carry a tag-`0x0d` `XBCH` payload that inflates
  to `FSSM`?
- Does a PC reference (precompilation disabled) have no native program, so a
  difference is the compiler's flags and not the reader?
- Do comparison sampling, comparison gather, depth loads and ordinary resource
  loads survive the translation?

`cargo xtask shader --probe` answers the first two from the container alone.
It prints part sizes, the `DXIL` program header, every `XDAT` offset and tag,
and the native reader's verdict. It never prints shader bytes or signatures.

## Files

| Path | What it is |
|---|---|
| `shaders/cmp_depth_ps.hlsl` | Pixel: `SampleCmpLevelZero`, `SampleCmp`, an offset, `GatherCmpRed`, a 2D array depth sample |
| `shaders/cmp_gather_cs.hlsl` | Compute: comparison gather and comparison sample |
| `shaders/depth_resource_cs.hlsl` | Compute: `Texture2D<float>` depth load, typed stencil load, raw buffer load |
| `make_fixtures.py.txt` | Compile with the GDK's Scarlett `dxc.exe`, probe, translate, report |
| `make_synthetic.py.txt` | Build a compiler-shaped container around a raw GFX10 program |
| `synthetic/depth_cmp_cs.s.txt` | Original GFX10: comparison sample, depth `image_load`, buffer store |
| `run-local.sh` | macOS/Linux: assemble the synthetic program, probe, translate |

Everything written by either workflow goes to a scratch directory outside the
repository. The shaders are original test sources; no game data, SDK file or
container is committed. The `.txt` endings on the Python tools are the
repository's asset policy, which allows text but not `.py`; Python runs them by
name regardless of the suffix.

## Real compiler: Windows or macOS through Wine

The compiler is the Scarlett copy (`...\GXDK\bin\Scarlett\dxc.exe`), not the PC
copy. `make_fixtures.py.txt` copies `dxc.exe`, `dxcompiler_xs.dll`,
`xbsc_xs.dll` and `newbe_xs.dll` into the scratch directory, copies the HLSL
beside them, and runs dxc with that directory as its cwd and relative paths.

On Windows:

```powershell
python tools/shader-diff/make_fixtures.py.txt `
  --dxc "C:/Program Files (x86)/Microsoft GDK/<version>/GXDK/bin/Scarlett/dxc.exe" `
  --out "$env:TEMP/victory-shader-diff"
```

On macOS with the local Wine:

```bash
python3 tools/shader-diff/make_fixtures.py.txt \
  --wine /Users/<you>/Documents/ScarlettMac/.toolchain/wine-cx/bin/wine \
  --dxc "<GDK>/GXDK/bin/Scarlett/dxc.exe" \
  --out "$TMPDIR/victory-shader-diff"
```

With `--wine` the script makes `<out>/wineprefix` (a fresh prefix per scratch
directory) and sets `WINEPREFIX`, `WINEDEBUG=-all` and
`WINEDLLOVERRIDES='winemac.drv=d;mscoree,mshtml='` for the dxc process only.
Use `--wineprefix` to name another prefix.

Each shader has an embedded `RootSignature`, which the Scarlett compiler needs
before it will precompile native code. The Xbox compile is:

```text
dxc.exe -T ps_6_6|cs_6_6 -E main -Fo <out> -Qstrip_reflect
        -D_GAMING_XBOX -D_GAMING_XBOX_SCARLETT
        -DWINAPI_FAMILY=WINAPI_FAMILY_GAMES <shader>.hlsl
```

The PC reference is the same command with `-D__XBOX_DISABLE_PRECOMPILE=1`
instead of the platform defines.

The script then:

1. Copies the Xbox container into `<out>/corpus/<stage>_<hash>.bin` and probes
   both forms through `cargo xtask shader ... --probe`.
2. Translates the Xbox container with `--dxil <out>/dxil`.
3. Writes `<out>/report.txt` and prints one line per shader:
   `xdat=+0x18 native=1 stages=Cs translation=ok`, or the failing part.

Use `--dry-run` to see the commands without running them, `--skip-pc` if there
is no PC reference, and `--only cmp_depth_ps` to compile one shader.

What a failure means:

- `xdat=missing`: the compiler did not precompile. Usually the shader lacks a
  `[RootSignature(...)]`, or the index is not at the reader's `+0x18`. The probe
  lists every `XDAT` candidate and says which one the reader reads.
- `native=0`: `XDAT` is there but no tag-`0x0d` entry inflates to `FSSM`, or
  the entry is not an `XBCH` stream.
- `translation=refused: ...`: the reader found the program but the translator
  refused it. The reason names the GFX10 offset; that is the next investigation.
- `pc_native=1`: the PC reference precompiled too, so the disable define did
  not take.

## macOS/Linux: the synthetic half

This assembles the original GFX10 program with `llvm-mc` (Homebrew LLVM is
fine), wraps it in the same DXBC/`DXIL`/`XDAT`/`XBCH`/`FSSM` shape a compiler
writes, and runs the real `--probe` and translation path.

```bash
LLVM_MC=/opt/homebrew/opt/llvm/bin/llvm-mc \
LLVM_OBJCOPY=/opt/homebrew/opt/llvm/bin/llvm-objcopy \
tools/shader-diff/run-local.sh
```

It also parses the three HLSL sources with `glslangValidator` when it is on
`PATH`; that is a syntax/type check by a different frontend, not DXIL and not
the GDK compiler.

## Status (2026-11-27, real Scarlett dxc)

The real compiler ran on this Mac through the local Wine. `dxc --version`
reported `dxcompiler_xs.dll!DxcCreateInstance: 1.8 - 2610.2608.12522.14643`.
All three fixture cases behaved the same way on the container path:

| Case | Xbox `XDAT` | Native reader | Translation |
|---|---|---|---|
| `cmp_depth_ps` (`ps_6_6`) | `+0x18`, 5 entries | 1 Ps, wave64, 384 code bytes | ok, 3,819 bytes |
| `cmp_gather_cs` (`cs_6_6`) | `+0x18`, 5 entries | 1 Cs, wave64, 384 code bytes | ok, 3,048 bytes |
| `depth_resource_cs` (`cs_6_6`) | `+0x18`, 5 entries | 1 Cs, wave64, 320 code bytes | ok, 2,680 bytes |

The PC references all had `XDAT` at `+0x18` but no tag-`0x0d` entry:
`native reader: none`. Recompiling the translated DXIL with `dxc -dumpbin`
showed the expected operations survived:

- PS: `dx.op.sampleCmpLevelZero`, `dx.op.sampleCmp`,
  `dx.op.textureGatherCmp` (including the `(1, -1)` offset and the array
  sample).
- Gather CS: `dx.op.textureGatherCmp` and `dx.op.sampleCmpLevelZero`.
- Depth/resource CS: `dx.op.textureLoad` twice and `dx.op.bufferLoad`.

No reader or emitter correction was needed. The only correction was in the
listing: pseudo-HLSL printed every image instruction as `.Sample`, including
comparison samples and gathers. It now prints `.SampleCmp`,
`.SampleCmpLevelZero`, `.GatherCmp` and the other variants
(`disasm::tests::image_calls_are_named_by_their_hlsl_operation`).

The existing GPU differential tests in
`crates/victory-shader/src/emit/diff_tests.rs` (`VICTORY_CONSOLE_DXC`,
`VICTORY_PC_DXC`) are the value-level complement and were not run: they need a
GPU.
