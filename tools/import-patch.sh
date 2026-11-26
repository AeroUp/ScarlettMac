#!/bin/zsh
# Turn a worker's worktree into a patch this repo can carry.
#
#   tools/import-patch.sh WORKTREE NAME [--checkpoint]
#
# The runtime source is not in this repository. A fix travels as a patch under
# runtime-preview/checkpoint/patches/ with an entry in series, exactly like the
# four that are already there, so this takes the difference between a worker's
# worktree and the pinned upstream commit, numbers it after the last one, and
# proves it applies on top of the whole series in a scratch checkout before
# anything is committed here.
#
# The worktree's base is the commit the checkpoint recipe was prepared from,
# so the diff is only the worker's own change: uncommitted edits count too.
set -eu

root=${0:A:h:h}
patches=$root/runtime-preview/checkpoint/patches

wt=${1:?usage: $0 WORKTREE NAME [--checkpoint]}
name=${2:?usage: $0 WORKTREE NAME [--checkpoint]}
[[ -d $wt ]] || { print -u2 "no worktree at $wt"; exit 2 }

# The base is the "Checkpoint base" commit, not the pinned upstream: diffing
# against the pin would carry the four checkpoint patches into the new patch.
base=$(git -C "$wt" log --format='%H %s' | grep -m1 'Checkpoint base' | cut -d' ' -f1)
[[ -n $base ]] || { print -u2 "$wt has no Checkpoint base commit"; exit 1 }
# The pinned upstream is that base's parent: the scratch checkout below starts
# there and applies the whole series plus the new patch on top.
pin=$(git -C "$wt" rev-parse "$base^")

# What changed since that base, committed or not.
diff=$(git -C "$wt" diff "$base")
if [[ -z $diff ]]; then
    print -u2 "$wt has no changes since $base"
    exit 1
fi

last=$(ls "$patches" | sed -n 's/^\([0-9]\{4\}\)-.*/\1/p' | sort -n | tail -1)
next=$(printf '%04d' $((10#$last + 1)))
file=$patches/$next-$name.patch
print -r -- "$diff" > "$file"
print "wrote $file ($(wc -l < "$file" | tr -d ' ') lines)"

# Prove it applies on top of the whole series, in a throwaway checkout of the
# pinned upstream. A patch that only works alone would break the recipe.
scratch=$root/.build/verify-$$
git -C "$root/.build/checkpoint-src" worktree add -q --detach "$scratch" "$pin"
trap 'git -C "$root/.build/checkpoint-src" worktree remove --force "$scratch" 2>/dev/null; rm -rf "$scratch"' EXIT
while IFS= read -r p; do
    [[ -n $p ]] || continue
    git -C "$scratch" apply --index "$patches/$p"
done < "$patches/series"
git -C "$scratch" apply --index "$file"
print "applies on top of $(wc -l < "$patches/series" | tr -d ' ') existing patches"

# Only now does it join the recipe.
grep -qxF "$next-$name.patch" "$patches/series" || print -r -- "$next-$name.patch" >> "$patches/series"
print "series now:"
cat "$patches/series"
