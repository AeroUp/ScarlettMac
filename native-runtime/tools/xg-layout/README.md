# Texture layouts against the development-PC `xg_xs` library

A differential harness for [`victory_formats::texture`](../../crates/victory-formats/src/texture.rs)
and the `xg_xs` HLE in [`crates/victory-hle/src/modules/xg.rs`](../../crates/victory-hle/src/modules/xg.rs).

`xg_layout_ref.c` runs on a Windows development PC (or here, under Wine) and asks the Xbox
Series X|S GDKX's development-PC `xg_xs` library for the layout of a list of synthetic textures,
writing a reference file. The example
`crates/victory-formats/examples/xg_layout_diff.rs` reads that file and compares every number with
what the Rust layout, texel-offset and copy path computes.

The cases are original and synthetic: NPOT block-compressed textures, packed mip tails, 2D
arrays, and every GFX10 swizzle mode. They find differences in the parts derived from AMD's
addrlib by reading it, especially the pipe/bank XOR modes that depend on the GPU's configuration.

## What is compared

For every case (a `D3D11`-style 2D texture or 2D array):

| Reference (dev-PC `xg_xs`) | Rust |
|---|---|
| `GetResourceLayout` `SizeBytes`, `BaseAlignmentBytes`, `MipLevels`, `Planes` | `Texture::size`, `alignment`, plane count |
| per plane `Usage`, `SizeBytes`, `SliceSizeBytes`, `BaseOffsetBytes`, `BaseAlignmentBytes`, `BytesPerElement`, `StandardMipCount`, `MipTailMipCount` | `Plane`/`Layout` fields |
| per mip `SizeBytes`, `OffsetBytes`, `Slice2DSizeBytes`, `PitchPixels`, `PitchBytes`, `AlignmentBytes`, padded/unpadded element and pixel sizes, depth/array size, sample count, swizzle mode, bank-rotation mask and size, slice depth, `IsMipTail`, `MipTailOffsetElementsX/Y` | the mapping `xg.rs`'s `mip_out` reports, from `Layout::mips` and `Layout::bank_rotation` |
| `GetTexelElementOffsetBytes` at the corners and middle of every mip, on the first and last array slice | `plane.offset + Layout::offset` |
| `CopyFromSubresource` of every mip of the first and last slice, and `CopyIntoSubresource` back, into buffers filled with a shared byte pattern, hashed whole | `Layout::detile` and `Layout::tile`, hashed the same way |
| `GetResourceTiling`: total tiles, packed-mip description, standard tile shape, the first standard tiling | `Layout::size`, `first_tail_mip`, the packed-tail arithmetic in `xg.rs::tiling` |

Coordinates are in **elements**, which is what `Layout::offset` takes: for a block-compressed
format one element is a 4x4 block.

Copies use a row pitch of `elements * bytes_per_element + 16` and a slice pitch of
`row * height_elements + 32`, so a wrong pitch, a tail placed by position, or a copy that writes
outside its subresource shows up as a hash difference. The tiled buffer starts from the same
pattern on both sides, so bytes `CopyIntoSubresource` leaves untouched are compared too.

Not covered: MSAA, 3D textures, depth/stencil planes, compression metadata planes (HTILE, CMASK,
FMASK, DCC), BCPack and HLE DirectStorage.

## Building the reference generator

### Windows / MSVC

Build with the `devpctools` (series X|S) include and lib, not the console ones:

```bat
cl /nologo /TP /EHsc /I"%GameDK%\xbox\include\devpctools\gen9" xg_layout_ref.c ^
   /link /LIBPATH:"%GameDK%\xbox\lib\devpctools\gen9" xg_xs.lib
copy "%GameDK%\xbox\bin\devpctools\gen9\xg_xs.dll" .
xg_layout_ref.exe reference.txt
xg_layout_ref.exe --hw=lockhart reference-lockhart.txt
```

The GDKX ZIP used for this work has the same files under
`Microsoft GDK/261000/xbox/{include,lib,bin}/devpctools/gen9/`.

### macOS / MinGW under Wine

The library is a Windows x64 PE DLL and needs no console or D3D device for these calls, so a
GNU-ABI build runs under Wine. MinGW cannot link the SDK's MSVC import library (its mangled names
are MSVC's), and the header declares its C entry points without `extern "C"`; the generator
therefore includes the header inside `extern "C"` for non-MSVC compilers and links `xg_xs.dll`
directly, which uses the same export table:

```bash
XG=<sdk>/xbox/include/devpctools/gen9
DLL=<sdk>/xbox/bin/devpctools/gen9/xg_xs.dll
x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wno-unknown-pragmas -I "$(dirname "$XG")" \
  xg_layout_ref.c "$DLL" -o xg_layout_ref.exe
cp "$DLL" .
mkdir -p prefix
export WINEPREFIX=$PWD/prefix
export WINEDLLOVERRIDES='winemac.drv=d;mscoree,mshtml='
export WINEDEBUG=-all
wine xg_layout_ref.exe anaconda.txt
wine xg_layout_ref.exe --hw=lockhart lockhart.txt
```

`XGSetHardwareVersion` is called explicitly (Anaconda by default, `--hw=lockhart` otherwise), so
the result does not depend on the library's default. A fresh `WINEPREFIX` works; the first run
initialises it.

The output is generated data. Keep it **outside the repository**, like a dump: only the generator
source and the comparison tool are committed.

## Comparing

```bash
export PATH="$(rustup run 1.98.1 rustc --print sysroot)/bin:$PATH"
cargo run --release -p victory-formats --example xg_layout_diff -- \
  --pipe-config=3,5 /path/to/anaconda.txt
cargo run --release -p victory-formats --example xg_layout_diff -- \
  --pipe-config=2,3 /path/to/lockhart.txt
```

Exit code 0 means every compared field matched; 1 means mismatches (the first 60 are printed, or
`--max-errors=N`); 2 means the file could not be read or parsed.

The XOR modes' addresses and bank rotations depend on the GPU's pipe configuration. The measured
rows are `PipeConfig::SERIES_X` (3 packers, 5 pipes) and `PipeConfig::SERIES_S` (2, 3; the
`DEFAULT`). To find which row a reference fits:

```bash
cargo run --release -p victory-formats --example xg_layout_diff -- --find-pipe-config reference.txt
```

It compares only the XOR cases (swizzle number 16 and above, tilings skipped) against every row the
tables define and prints the one with no mismatches.

## Measured results

Both references were produced with the GDKX 2026.10 development-PC `xg_xs.dll`, gen9 devpctools,
under Wine 11.0 (`winemac` driver disabled), and compared with this tree:

| reference | `--pipe-config` | cases | compared fields | mismatches |
|---|---|---|---|---|
| Anaconda (Series X) | 3,5 | 29 | 9179 | **0** |
| Lockhart (Series S) | 2,3 | 29 | 9179 | **0** |

`--find-pipe-config` names exactly one row per reference and no other: Anaconda `3,5`, Lockhart
`2,3`. The rows are therefore measured, not guessed, for these two profiles.

What the first run found, and what was changed:

1. **Tail mips keep the resource's swizzle mode.** The library reports `SwizzleMode = 10` (a
   64 KiB mode) on a mip whose `IsMipTail` is 1; it does not switch to a 256-byte mode. `xg.rs`'s
   `mip_out` was reporting `2` from a guess; it now reports the resource's mode. This changed no
   address or copy, only the reported field (157 fields).
2. **Bank rotation is reported.** 64 KiB `_X` modes carry `BankRotationAddressBitMask` and
   `BankRotationBytesPerSlice`: `0xE000`/`0x2000` for Series S and `0x8000`/`0x8000` for Series X;
   `_T` and 4 KiB modes carry zero. `Layout::bank_rotation` reproduces both measurements and
   `mip_out` reports it (66 fields on each reference). The formula's generalisation to other rows
   is a guess, marked in the code.
3. **Two harness cases were invalid, not layout differences.**
   * `BC7` with `64KB_Z_X` is rejected (`E_INVALIDARG`): `Z_X` is a depth mode and BC7 cannot be
     one. The case now uses `R32_FLOAT`/`D32_FLOAT` (DXGI 40), which is accepted.
   * `4KB_R_X` is rejected for every format, bind flag and sample count tried. It is not a GFX10
     mode. The case was removed.
4. **The old `DEFAULT` guess was the Series S row.** The code called (2,3) "the Series X"; the
   library says it is Lockhart. `PipeConfig::SERIES_X` (3,5) and `SERIES_S` (2,3) are now named
   constants, and `DEFAULT` stays `SERIES_S`, so the native host's behaviour is unchanged until a
   title measurement says which profile it runs.

## What remains untested

- The MSVC + `xg_xs.lib` build path (the measurements used MinGW linking the DLL).
- MSAA, 3D textures, depth/stencil and stencil planes, compression metadata planes.
- The Xbox One/gen8 hardware versions of `XGSetHardwareVersion`.
- Which profile the emulator should use at runtime. The evidence says what each profile's layout
  is; it does not say which one the title was compiled for. `DEFAULT` is unchanged.
- The model still accepts `4KB_R_X` and `256B_R`, which the library rejects. That is a validation
  difference, not exercised by these cases or (so far) by a title; it was left alone.

Reference files are generated data and stay outside the repository. The generator source uses the
SDK header at build time and copies nothing from it; no SDK binary, header or reference output is
committed.

## File format

Text, one record per line, `key=value` fields after a keyword. Lines starting with `#` are
ignored. Version 1:

```
xg-layout-ref 1
hw anaconda
case <name>
desc dim=2d w=<u32> h=<u32> array=<u32> mips=<u32> fmt=<dxgi> samples=<u32> swz=<u32>
status ok | status fail hr=0x<8 hex digits>
res size=<u64> align=<u64> mips=<u32> planes=<u32>
plane i=<u32> usage=<u32> size=<u64> slice=<u64> offset=<u64> align=<u64> bpe=<u32> std=<u32> tail=<u32>
mip plane=<u32> i=<u32> size=<u64> offset=<u64> slice=<u64> pitch_px=<u32> pitch_b=<u32> align=<u32>
    pw= ph= pd= ew= eh= wp= hp= doa= samples= swz= bank0=<u64> bank1=<u64> slice_depth= tail= tx= ty=
texel plane=<u32> mip=<u32> x=<u32> y=<u32> z=<u32> off=<u64> [bit=<u32>]
copy plane=<u32> sub=<u32> row=<u32> depth=<u32> from=<u64> into=<u64> len=<u32>
    from_hr=0x<hr> into_hr=0x<hr>
tiling tiles= std= packed= ptiles= start= tw= th= td= n=<u32> starts=<comma-separated u32, or empty>
end
```

`desc mips` is the mip count actually used (a full chain is written out). `sub` is
`mip + slice * mips`. `from`/`into` are 64-bit FNV-1a hashes written in decimal. `n` is the number
of tilings requested (1 in this version); `starts` is empty when the library reports none (a
mip chain entirely in the tail).

The byte pattern both sides share:

```
v = i * 0x9E3779B97F4A7C15 + seed * 0x100000001B3   (64-bit, wrapping)
v ^= v >> 29
byte = (v >> 32) & 0xFF
```

Seeds: `1` for the tiled buffer a `CopyFromSubresource` reads, `2` for the tiled buffer a
`CopyIntoSubresource` writes, `3` for its linear input.

`--self-test` checks the parser, model and writer against themselves and that a wrong number in
each kind of line is reported; `--emit-self=<file>` writes that self-reference. `cargo test -p
victory-formats --example xg_layout_diff` runs the same checks.
