#!/bin/zsh
# Run the title for a fixed time and photograph its window at given seconds.
#
#   tools/world-run.sh --runtime DIR --tag NAME [--seconds 330] [--at 150,200,250] [--wait 600]
#
# The title spends its first couple of minutes streaming the intro, then draws
# the world, then may raise its own support alert at about four minutes and
# fault, so a screenshot at the end of a run is usually the loading artwork and
# tells you nothing. This starts the run through the one-instance lock, waits
# for the window, and captures at the offsets asked for, into
# runs/<log name>/NNNs.png beside the log. It never raises, activates or
# injects input into the window.
set -eu

root=${0:A:h:h}
runtime=$root/.build/checkpoint-src/target/x86_64-pc-windows-gnullvm/diag
tag=world
seconds=330
at=150,200,250
wait=600
while [[ $# -gt 0 ]]; do
    case $1 in
        --runtime) runtime=$2; shift 2 ;;
        --tag) tag=$2; shift 2 ;;
        --seconds) seconds=$2; shift 2 ;;
        --at) at=$2; shift 2 ;;
        --wait) wait=$2; shift 2 ;;
        *) print -u2 "Usage: $0 [--runtime DIR] [--tag NAME] [--seconds N] [--at LIST] [--wait N]"; exit 2 ;;
    esac
done
[[ $wait -gt 0 && $wait -lt 3600 ]] && wait=3600

shot=$root/.build/window-shot
[[ -x $shot ]] || { print -u2 "Build the capture tool first: xcrun swiftc -O -o .build/window-shot tools/window-shot.swift"; exit 1; }

out=$(mktemp "${TMPDIR:-/tmp}/world-run.XXXXXX")
"$root/tools/scarlett-run.sh" --runtime "$runtime" --seconds "$seconds" --tag "$tag" --wait "$wait" > "$out" 2>&1 &
runner=$!

# The run prints its log and the title's pid once it holds the lock.
pid=
for _ in $(seq 1 $((wait / 2 + 60))); do
    line=$(grep -m1 '^running: ' "$out" 2>/dev/null || true)
    if [[ -n $line ]]; then
        log=${line#running: }
        log=${log%% *}
        pid=$(print -r -- "$line" | sed -n 's/.*title pid \([0-9][0-9]*\).*/\1/p')
        break
    fi
    if ! kill -0 $runner 2>/dev/null; then
        cat "$out" >&2
        print -u2 "the run did not start"
        exit 1
    fi
    sleep 2
done
[[ -n $pid && -n $log ]] || { print -u2 "no run started within ${wait}s"; exit 1; }

dir=${log%.log}
mkdir -p "$dir"
print "run: $log (title pid $pid), shots in $dir"

# Window numbers move when the title recreates its swap chain, so look again
# before every shot rather than binding one at the start.
for offset in ${(s:,:)at}; do
    while (( $(date +%s) < $(stat -f %B "$log") + offset )); do
        if ! kill -0 $pid 2>/dev/null; then
            print "title exited before ${offset}s"
            break 2
        fi
        sleep 2
    done
    kill -0 $pid 2>/dev/null || { print "title exited before ${offset}s"; break; }
    number=$($shot list | awk -F'\t' -v p="$pid" '$1 == p && $3 ~ /^[0-9]+x[0-9]+$/ { split($3, wh, "x"); if (wh[1] >= 1000) { print $2; exit } }')
    if [[ -z $number ]]; then
        print -u2 "no window for pid $pid at ${offset}s"
        continue
    fi
    "$shot" shot "$number" "$dir/${offset}s.png" || true
done

wait $runner 2>/dev/null || true
cat "$out"
rm -f "$out"
print "done: $log"
