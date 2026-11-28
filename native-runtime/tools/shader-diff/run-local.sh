#!/usr/bin/env bash
# Local half of tools/shader-diff: assemble the original GFX10 test program,
# wrap it in the compiler-shaped container make_synthetic.py.txt builds, and run
# the existing debug CLI's --probe and translation on it. The Windows half
# (make_fixtures.py.txt) does the same with the GDK's Scarlett dxc.exe.
#
# The corpus is scratch data outside the repository. Needs llvm-mc and
# llvm-objcopy (Homebrew LLVM or an LLVM install), Python 3, cargo and a
# built victory-shader.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=${VICTORY_SHADER_DIFF_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/victory-shader-diff.XXXXXX")}
mkdir -p "$scratch"

# AGENTS.md: put the pinned toolchain's bin first, or Homebrew's rustc breaks
# the build with "can't find crate for std".
if command -v rustup >/dev/null 2>&1; then
    sysroot=$(rustup run 1.98.1 rustc --print sysroot 2>/dev/null || true)
    if [ -n "$sysroot" ] && [ -d "$sysroot/bin" ]; then
        PATH="$sysroot/bin:$PATH"
    fi
fi
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
fi
export PATH

find_tool() {
    local name=$1 env=$2
    if [ -n "${!env:-}" ]; then
        printf '%s\n' "${!env}"
        return
    fi
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return
    fi
    # Homebrew's keg-only LLVM keeps its tools out of PATH.
    if [ -x "/opt/homebrew/opt/llvm/bin/$name" ]; then
        printf '%s\n' "/opt/homebrew/opt/llvm/bin/$name"
        return
    fi
    return 1
}

llvm_mc=$(find_tool llvm-mc LLVM_MC) || {
    echo "llvm-mc not found; set LLVM_MC (Homebrew: /opt/homebrew/opt/llvm/bin/llvm-mc)" >&2
    exit 2
}
llvm_objcopy=$(find_tool llvm-objcopy LLVM_OBJCOPY) || {
    echo "llvm-objcopy not found; set LLVM_OBJCOPY" >&2
    exit 2
}
command -v python3 >/dev/null || {
    echo "python3 not found" >&2
    exit 2
}

# A local HLSL frontend catch for the sources. glslang's HLSL mode is not the
# GDK compiler and emits SPIR-V, but it parses and type-checks these
# declarations and intrinsic calls.
if command -v glslangValidator >/dev/null 2>&1; then
    for spec in "cmp_depth_ps ps" "cmp_gather_cs cs" "depth_resource_cs cs"; do
        set -- $spec
        stage=frag
        [ "$2" = cs ] && stage=comp
        glslangValidator -D -S "$stage" -e main -V \
            "$root/tools/shader-diff/shaders/$1.hlsl" \
            -o "$scratch/$1.spv" >/dev/null
    done
    echo "shader-diff local: HLSL parses (glslangValidator)"
fi

code="$scratch/depth_cmp_cs.bin"
"$llvm_mc" -triple=amdgcn -mcpu=gfx1030 -filetype=obj \
    -o "$scratch/depth_cmp_cs.o" "$root/tools/shader-diff/synthetic/depth_cmp_cs.s.txt"
"$llvm_objcopy" -O binary --only-section=.text \
    "$scratch/depth_cmp_cs.o" "$code"

python3 "$root/tools/shader-diff/make_synthetic.py.txt" \
    --code "$code" --out "$scratch/corpus" --name depth_cmp \
    > "$scratch/generated.txt"
hash=$(tail -n 1 "$scratch/generated.txt")

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found; put the pinned toolchain's bin on PATH (AGENTS.md)" >&2
    exit 2
fi

nice cargo run --release -j 2 --manifest-path "$root/Cargo.toml" -p xtask -- \
    shader "$hash" --corpus "$scratch/corpus" --probe \
    | tee "$scratch/probe.txt"

grep -q "XDAT at +0x18" "$scratch/probe.txt" || {
    echo "probe did not find XDAT at the native reader's offset" >&2
    exit 1
}
grep -q "native reader: 1 program" "$scratch/probe.txt" || {
    echo "the native reader did not find the synthetic program" >&2
    exit 1
}

nice cargo run --release -j 2 --manifest-path "$root/Cargo.toml" -p xtask -- \
    shader "$hash" --corpus "$scratch/corpus" --dxil "$scratch/dxil" \
    --no-disasm --no-hlsl \
    | tee "$scratch/translate.txt"

grep -q "translation: ok" "$scratch/translate.txt" || {
    echo "the synthetic comparison/depth program did not translate" >&2
    exit 1
}

echo "shader-diff local: ok (container $hash, probe + translation; scratch $scratch)"
