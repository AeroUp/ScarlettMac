# ScarlettMac

[![macOS: Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-555555)](#tested-on-mac)
[![Tested on M3 Max](https://img.shields.io/badge/Tested-M3%20Max-555555)](#tested-on-mac)
[![Graphics: D3DMetal](https://img.shields.io/badge/Graphics-D3DMetal-555555)](#how-it-works)
[![GTA VI: World loading](https://img.shields.io/badge/GTA%20VI-World%20loading-9a6700)](#gta-vi-status)

ScarlettMac runs Xbox Series X|S game code on Apple Silicon. It is an experimental compatibility layer built on [XCL](https://github.com/xcl-dev/xcl) and CrossOver.

## GTA VI status

**GTA VI boots offline, shows the intro and menus, and starts loading the world. Gameplay remains unverified.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/gta-milestones-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/gta-milestones-light.png">
  <img alt="Four of seven launch stages verified: offline boot, intro graphics, setup and menus, and world loading. The first visible 3D scene, in-world controls, and first mission are still open." src="assets/gta-milestones-dark.png">
</picture>

Tested with Xbox Series build 1.0.2.1. [See the evidence for each stage](docs/compatibility.md#gta-vi-verification)

## How it works

```mermaid
flowchart LR
    Game["Xbox Series game"] --> XCL["ScarlettMac / XCL"]
    XCL --> CPU["CPU · Rosetta 2"]
    XCL --> GPU["Graphics · D3DMetal"]
```

XCL supplies Xbox system services. Rosetta 2 runs the game’s x64 code, and D3DMetal runs the translated graphics on the Mac GPU.

## Tested on Mac

Apple M3 Max with CrossOver Preview 27.0.0.41073 and D3DMetal 4.0b3.

| Test | Result |
| --- | --- |
| GTA VI | Menus and world loading reached |
| Xbox system checks | 43 of 50 passed in all three runs |
| TuxBox Lanes | Scripted replay passed with input and rendered-board checks |

[Measurements, known issues, and reproduction steps](docs/compatibility.md)

## Build and run

Use an Apple Silicon Mac with CrossOver Preview and create a dedicated Windows 11 64-bit bottle named `ScarlettLab` with D3DMetal enabled. Keep the executable and accompanying unpacked game files together in their original directory structure. [CrossOver](https://www.codeweavers.com/crossover)

Install Rust through rustup, extract LLVM-MinGW 20261117 for macOS, and extract the x64 DLLs from a DirectX Shader Compiler release. The reported build used Rust 1.92.0. [Rust](https://rustup.rs/) · [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw/releases/tag/20261117) · [DXC](https://github.com/microsoft/DirectXShaderCompiler/releases)

```sh
rustup target add --toolchain stable x86_64-pc-windows-gnullvm
export SCARLETT_LLVM_MINGW=/path/to/llvm-mingw
export SCARLETT_DXC_DIR=/path/to/dxc/bin/x64
scripts/build.sh
scripts/run.sh --inspect /path/to/unpacked/game.exe
scripts/run.sh /path/to/unpacked/game.exe
```

Set `SCARLETT_BOTTLE_DIR` to use a dedicated bottle in another location. Set `SCARLETT_OPUS_DLL` to an x64 `libopus-0.dll` to enable the Opus decoder. The launcher stores local data in the bottle under `C:\scarlett-lab-data` and runs with outbound IP traffic disabled.

The repository distributes project source, patches, and launcher scripts. Supply your own unpacked game files. [Source provenance](SOURCE-NOTICES.md)

## Credits and license

This project builds on XCL by ivall and the macOS work by utkarshxdev. [ivall/xcl](https://github.com/ivall/xcl), [utkarshxdev/xcl-mac](https://github.com/utkarshxdev/xcl-mac)

The XCL-derived patches are distributed under GPL-2.0-only. See [LICENSE](LICENSE).
