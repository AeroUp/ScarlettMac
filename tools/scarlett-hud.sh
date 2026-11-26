#!/bin/zsh
# Open the ScarlettMac HUD in an always-on-top window (builds it on first use).
# --cli shows the same dashboard in this terminal instead.
set -eu
root=${0:A:h:h}
if [[ ${1:-} == --cli ]]; then
    exec /usr/bin/python3 -I "$root/tools/scarlett-hud.py"
fi
bin=$root/.build/hud/scarlett-hud
if [[ ! -x $bin || $root/tools/scarlett-hud.swift -nt $bin ]]; then
    mkdir -p "${bin:h}"
    swiftc -O -o "$bin" "$root/tools/scarlett-hud.swift"
fi
"$bin" "$root/tools/scarlett-hud.py" > /dev/null 2>&1 &
disown
print "HUD open (pid $!). Close it with its close button."
