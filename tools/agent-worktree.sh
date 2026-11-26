#!/bin/zsh
# Make a build tree that is actually the ScarlettMac runtime.
#
#   tools/agent-worktree.sh NAME
#
# A plain `git worktree add` from the prepared source checks out the pinned
# upstream commit and silently leaves out the four checkpoint patches, so a
# worker builds and runs plain VIctory: no picker confirmation, no mesh limit,
# no GPU-wait diagnosis, no ScarlettMac caption. That mistake cost a round of
# runs once; this makes it impossible.
#
# It also copies the prepared target directory, so the first build is about a
# minute instead of ten.
set -eu

root=${0:A:h:h}
name=${1:?usage: $0 NAME}
src=$root/.build/checkpoint-src
patches=$root/runtime-preview/checkpoint/patches
tree=$root/.build/wt-$name

[[ -d $src ]] || { print -u2 "no prepared source at $src (run runtime-preview/build.sh --checkpoint --prepare)"; exit 1 }
[[ -e $tree ]] && { print -u2 "$tree already exists"; exit 1 }

pin=$(sed -n 2p "$root/runtime-preview/checkpoint/UPSTREAM")
git -C "$src" fetch -q upstream "$pin" 2>/dev/null || true
git -C "$src" worktree add -q --detach "$tree" "$pin"
git -C "$tree" checkout -q -b "work/$name"

while IFS= read -r p; do
    [[ -n $p ]] || continue
    git -C "$tree" apply --index "$patches/$p"
done < "$patches/series"
git -C "$tree" -c user.name=ScarlettMac -c user.email=scarlettmac@local \
    commit -q -m "Checkpoint base: runtime-preview/checkpoint/patches/series"

[[ -d $src/target ]] && cp -R "$src/target" "$tree/target"

# The checkout is correct only if the patches are visibly in it.
grep -q WAITING_ON "$tree/crates/victory-gpu/src/host.rs" || {
    print -u2 "the checkpoint patches are not in the tree"; exit 1; }
grep -q ScarlettMac "$tree/crates/victory-gpu/src/caption.rs" || {
    print -u2 "the ScarlettMac caption patch is not in the tree"; exit 1; }

cat <<EOF
$tree is ready on branch work/$name (base $(git -C "$tree" rev-parse --short HEAD)).

Build:
  export PATH=\$HOME/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin:\$PATH
  export RUSTUP_TOOLCHAIN=1.98.1-aarch64-apple-darwin
  cd $tree
  LLVM_MINGW_ROOT=$root/.toolchain/llvm-mingw-20261117-ucrt-macos-universal \\
    CARGO_TARGET_DIR=\$PWD/target cargo build --locked --jobs 3 \\
    --target x86_64-pc-windows-gnullvm --profile diag --features diagnostics -p victory

Run and photograph:
  $root/tools/world-run.sh --runtime $tree/target/x86_64-pc-windows-gnullvm/diag \\
    --tag $name --seconds 330 --at 150,200,250,300 --wait 900
EOF
