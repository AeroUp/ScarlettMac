#!/bin/zsh
# One GTA VI run at a time, from the built ScarlettMac runtime preview.
#
#   tools/scarlett-run.sh [--seconds N] [--game /path/to/game_scarlett_master.exe]
#                         [--runtime DIR] [--tag NAME]
#
# --runtime DIR runs a build from another source tree (the directory holding
# victory.exe and libunwind.dll; default: the checkpoint diag build). --tag
# names the log file so a set of runs for one change is easy to find.
#
# It takes a lock (only one title may run: two instances on this Mac fight for
# the GPU and for the same save), sets the Wine/D3DMetal environment the
# runtime preview needs, sends the offline picker's confirmation from a file
# marker (a title-level Enter, not a synthesised OS key), and logs to
# runs/<timestamp>.log.
set -eu

root=${0:A:h:h}
seconds=600
# Your own dump of the title: SCARLETT_GAME=/path/to/game_scarlett_master.exe
game=${SCARLETT_GAME:-"$HOME/Downloads/ISOs/GTA6 Xbox personal/game_scarlett_master.exe"}
runtime=$root/.build/checkpoint-src/target/x86_64-pc-windows-gnullvm/diag
tag=
wait=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --seconds) seconds=$2; shift 2 ;;
        --game) game=$2; shift 2 ;;
        --runtime) runtime=$2; shift 2 ;;
        --tag) tag=$2; shift 2 ;;
        --wait) wait=$2; shift 2 ;;
        *) print -u2 "Usage: $0 [--seconds N] [--game PATH] [--runtime DIR] [--tag NAME] [--wait N]"; exit 2 ;;
    esac
done

runtime=${runtime:A}   # absolute: the launch below changes into it
exe=$runtime/victory.exe
wine=${SCARLETT_WINE:-${0:A:h:h}/.toolchain/wine-cx/bin/wine}
prefix=${SCARLETT_PREFIX:-$HOME/ScarlettWine}

[[ -x $wine ]] || { print -u2 "No Wine at $wine"; exit 1; }
# Several workers share the one lock and a run takes about six minutes, so a
# short queue gives up behind two runs. Any queue waits at least an hour.
[[ $wait -gt 0 && $wait -lt 3600 ]] && wait=3600

[[ -f $exe ]] || { print -u2 "No runtime at $exe (build it first)"; exit 1; }
[[ -f $runtime/libunwind.dll ]] || { print -u2 "libunwind.dll missing beside the runtime"; exit 1; }

# One title at a time. The lock is a directory: atomic, and its owner is in it.
# --wait N queues this run behind one already going, for up to N seconds, so
# several workers can share the Mac without polling the lock by hand.
lock=$root/.build/run.lock
waited=0
until mkdir "$lock" 2>/dev/null; do
    # The owner's pid is in the lock. A launcher that was killed (SIGKILL
    # skips the trap below) leaves the lock behind with nobody running: if
    # its owner is gone, the lock is stale and is taken over.
    owner=$(cat "$lock/pid" 2>/dev/null || true)
    if [[ -n $owner ]] && ! kill -0 "$owner" 2>/dev/null; then
        if [[ $(cat "$lock/pid" 2>/dev/null || true) == "$owner" ]]; then
            print "the lock's owner (pid $owner) is gone: taking the stale lock over"
            rm -f "$lock/pid" 2>/dev/null || true
            rmdir "$lock" 2>/dev/null || true
        fi
        continue
    fi
    if [[ $waited -ge $wait ]]; then
        print -u2 "A run is already going (or a stale lock at $lock)."
        print -u2 "If nothing is running: rm -rf '$lock'"
        exit 1
    fi
    sleep 2
    waited=$((waited + 2))
done
trap 'rm -rf "$lock"' EXIT
print $$ > "$lock/pid"
if [[ $waited -gt 0 ]]; then
    print "waited ${waited}s for the single-instance lock"
fi

if [[ -n ${SCARLETT_WINE:-} || ${SCARLETT_FRESH_WINESERVER:-0} == 1 ]]; then
    WINEPREFIX=$prefix "${0:A:h:h}/.toolchain/wine-cx/bin/wineserver" -k 2>/dev/null || true
    WINEPREFIX=$prefix "${wine:h}/wineserver" -k 2>/dev/null || true
    sleep 1
    print "wine: $wine (wineserver: ${wine:h}/wineserver)"
fi

# One title at a time; the lock above is the guard. (A pgrep for the guest
# image also matches any wrapper whose command line names it, including this
# script's own launch line, so it is not used.)

stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p "$root/runs"
log=$root/runs/${stamp}${tag:+-$tag}.log
marker=$root/.build/confirm.txt
: > "$marker"

export WINEPREFIX=$prefix
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d12,d3d12core,dxgi=b"
export WINEDEBUG=${WINEDEBUG:--all}
export ROSETTA_ADVERTISE_AVX=1
# The offline picker's confirmation, written to this file (see
# the picker marker in runtime-preview/checkpoint/patches/0001-Checkpoint-base-...). The file
# must be fresher than 600 ms when the title polls it.
confirm_win="Z:${marker:A}"
export VICTORY_CONFIRM_FILE=${confirm_win//\//\\}
: "${VICTORY_SNAPSHOT_POOL_MIB:=256}"
export VICTORY_SNAPSHOT_POOL_MIB
export VICTORY_DIRECT_BARYCENTRICS=${VICTORY_DIRECT_BARYCENTRICS:-1}

guest="Z:${game/#\//\\}"
guest=${guest//\//\\}

cd "$runtime"
# SCARLETT_CONSOLE=series-s runs the title as an Xbox Series S: 7.8 GB of
# title memory instead of 13.5 GB, and the title's Series S assets and
# render sizes. On a Mac with 16-18 GB that is the difference that matters.
console_args=()
[[ -n ${SCARLETT_CONSOLE:-} ]] && console_args=(--console "$SCARLETT_CONSOLE")
"$wine" "$exe" run "${console_args[@]}" --execute "$guest" > "$log" 2>&1 &
wine_pid=$!
print "running: $log (title pid $wine_pid, ${seconds}s)"

# Title-level Enter, once every 3 s for the first 3 minutes: the picker takes
# one, later writes land on whatever menu is up. Stop after that: a menu left
# alone is what we want to watch.
(
    for _ in $(seq 1 60); do
        sleep 3
        printf 'Enter' > "$marker" 2>/dev/null || true
    done
) &
writer=$!

(
    sleep "$seconds"
    kill "$writer" 2>/dev/null || true
    # Only this run's own title: its wine process, then this prefix's server
    # (one run at a time holds the lock, so the prefix is ours). Never a
    # pkill by name, which also closed other workers' titles.
    kill "$wine_pid" 2>/dev/null || true
    sleep 2
    WINEPREFIX=$prefix "${wine:h}/wineserver" -k 2>/dev/null || true
) &
limiter=$!

wait "$wine_pid" 2>/dev/null || true
kill "$writer" "$limiter" 2>/dev/null || true
if [[ -n ${SCARLETT_WINE:-} || ${SCARLETT_FRESH_WINESERVER:-0} == 1 ]]; then
    WINEPREFIX=$prefix "${wine:h}/wineserver" -k 2>/dev/null || true
fi
print "done: $log"
