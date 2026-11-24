#!/bin/zsh
set -eu

root=${0:A:h:h}
revision=d0d9889514e94195ebcdbf86e23a60f4f5756189
source="$root/.build/xcl-mac"
patch="$root/patches/mac-compat.patch"

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --prepare ) ]]; then
    print -u2 'Usage: scripts/build.sh [--prepare]'
    exit 2
fi

if [[ ! -d "$source/.git" ]]; then
    git init --quiet "$source"
    git -C "$source" remote add origin https://github.com/utkarshxdev/xcl-mac.git
fi
if ! git -C "$source" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$source" fetch --depth 1 origin refs/tags/v0.1.0-stage6
    git -C "$source" checkout --detach "$revision"
fi
if [[ $(git -C "$source" rev-parse HEAD) != "$revision" ]]; then
    print -u2 "Unexpected XCL source revision in $source"
    exit 1
fi
if git -C "$source" apply --check "$patch" 2>/dev/null; then
    git -C "$source" apply "$patch"
else
    git -C "$source" apply --reverse --check "$patch"
fi

if [[ ${1:-} != --prepare ]]; then
    : "${SCARLETT_LLVM_MINGW:?Set SCARLETT_LLVM_MINGW to the extracted LLVM-MinGW directory}"
    export PATH="$SCARLETT_LLVM_MINGW/bin:$PATH"
    export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER=x86_64-w64-mingw32-clang
    cd "$source"
    rustup run "${SCARLETT_RUST_TOOLCHAIN:-stable}" cargo build --release --locked --target x86_64-pc-windows-gnullvm -p xcl
fi
