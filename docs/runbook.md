# Runbook: measuring and carrying a fix

The runtime source is not in this repository. It is the pinned upstream in
`runtime-preview/checkpoint/UPSTREAM` with the patches in
`runtime-preview/checkpoint/patches/series` applied, and a fix travels back as
one more patch in that series.

## Trees

`tools/agent-worktree.sh NAME` makes `.build/wt-NAME`: a worktree of the
prepared source, the four checkpoint patches applied and committed as
"Checkpoint base", and a copy of the prepared `target/` so the first build is
about a minute. Never use a bare `git worktree add` here: it checks out the
pinned commit and silently leaves out every checkpoint patch, which produces a
runtime with no picker confirmation, no mesh limit and no ScarlettMac caption.

Build:

```sh
export PATH=$HOME/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin:$PATH
export RUSTUP_TOOLCHAIN=1.98.1-aarch64-apple-darwin
cd .build/wt-NAME
LLVM_MINGW_ROOT=/Users/aero/Documents/ScarlettMac/.toolchain/llvm-mingw-20261117-ucrt-macos-universal \
  CARGO_TARGET_DIR=$PWD/target cargo build --locked --jobs 3 \
  --target x86_64-pc-windows-gnullvm --profile diag --features diagnostics -p victory
```

## Running

One title at a time, always through `tools/scarlett-run.sh`. `--wait N` queues
behind a run already going instead of retrying by hand; the lock stays the only
guard. The lock holds its launcher's pid; a queued run takes over a lock
only when that launcher is gone (killed outright, so it never released it).

```sh
tools/scarlett-run.sh --runtime .build/wt-NAME/target/x86_64-pc-windows-gnullvm/diag \
  --seconds 330 --tag NAME --wait 900
```

Use 330 s, not 180. The title streams the intro for about two minutes, draws the
world for two more, then raises its own support alert and faults at roughly four
minutes: the alert is `mov dword ptr [0], 0x4d8bd421` at game image
`+0x23cf683`, a deliberate trap, not a bad pointer from the runtime. A 180 s
run ends during streaming, and the loading artwork it shows says nothing about
whether the world renders.

`tools/world-run.sh` does the same and photographs the window at chosen seconds:

```sh
tools/world-run.sh --runtime .build/wt-NAME/target/x86_64-pc-windows-gnullvm/diag \
  --tag NAME --seconds 330 --at 150,200,250,300 --wait 900
```

## Judging a run

```sh
python3 tools/fps-report.py runs/<log>
python3 tools/fps-report.py --compare runs/<baseline> runs/<candidate>
```

Three traps, all of which have produced a wrong conclusion here:

- Presents are logged only when the frame counter reaches a power of two, so a
  missing present line is not evidence that presents stopped.
- A log that simply ends is usually a crash. The report reads `crashes.log`
  from the Wine prefix and says whether the process died inside the run's
  window.
- Two runs that die at different frames cannot be compared by totals. Compare
  the frame spans they share, which `--compare` prints.

## Carrying a fix

```sh
tools/import-patch.sh .build/wt-NAME name-in-kebab-case
```

It takes the worker tree's difference from the pinned upstream, numbers it
after the last patch, proves it applies on top of the whole series in a scratch
checkout, and appends it to `series`. Then rebuild from a fresh tree with
`runtime-preview/build.sh --checkpoint` to prove the recipe itself still builds,
and add the measured result to `docs/measurements/`.

## Pull requests

Work happens on the `AeroUp/ScarlettMac` fork. A PR is opened only when a change
has a measured result behind it, one per fix, with the numbers and the run
logs' facts in the body. Tooling and documentation ride along with that PR
rather than arriving on their own.
