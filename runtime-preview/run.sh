#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    printf 'Usage: WINE=PATH %s RUNTIME_EXE COMMAND [ARG...]\n' "$0" >&2
    exit 2
fi
: "${WINE:?Set WINE to a compatible Wine executable}"
runtime=$1
shift
[ -x "$WINE" ] && [ -f "$runtime" ] || {
    printf '%s\n' 'Check the Wine executable and runtime paths.' >&2
    exit 2
}
runtime_dir=$(CDPATH='' cd -- "$(dirname -- "$runtime")" && pwd)
runtime=$runtime_dir/$(basename -- "$runtime")
unwind=$runtime_dir/libunwind.dll
if [ ! -f "$unwind" ]; then
    printf '%s\n' 'Place the llvm-mingw x86-64 libunwind.dll beside the runtime executable.' >&2
    exit 2
fi
export ROSETTA_ADVERTISE_AVX=1
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
if [ -n "${SCARLETTMAC_CROSSOVER_BOTTLE:-}" ]; then
    case "$SCARLETTMAC_CROSSOVER_BOTTLE" in
        .|..|*/*|*\\*|-*) printf '%s\n' 'Use a CrossOver bottle name.' >&2; exit 2 ;;
    esac
    bottles=${CX_BOTTLE_PATH:-"$HOME/Library/Application Support/CrossOver/Bottles"}
    drive_c=$bottles/$SCARLETTMAC_CROSSOVER_BOTTLE/drive_c
    if [ ! -d "$drive_c" ]; then
        printf 'Set CX_BOTTLE_PATH to the folder containing the bottle: %s\n' "$SCARLETTMAC_CROSSOVER_BOTTLE" >&2
        exit 2
    fi
    staged=$(mktemp -d "$drive_c/scarlettmac-preview.XXXXXX")
    trap 'rm -f -- "$staged/victory.exe" "$staged/libunwind.dll"; rmdir -- "$staged" || true' 0
    cp -- "$runtime" "$staged/victory.exe"
    cp -- "$unwind" "$staged/libunwind.dll"
    guest="C:\\$(basename -- "$staged")\\victory.exe"
    "$WINE" --bottle "$SCARLETTMAC_CROSSOVER_BOTTLE" --no-update --no-gui --no-convert --wait \
        --cx-app "$guest" "$@"
    exit
fi
exec "$WINE" "$runtime" "$@"
