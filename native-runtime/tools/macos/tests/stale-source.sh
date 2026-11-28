#!/bin/bash
# run-native.sh names the binary a run used and warns when it is older than the checkout's
# sources: a log's commit must not be read as code the binary did not build.
# Run: /bin/bash tools/macos/tests/stale-source.sh
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/native-binary.sh"

root=$(mktemp -d "${TMPDIR:-/tmp}/victory stale.XXXXXX")
trap 'rm -rf "$root"' EXIT
mkdir -p "$root/crates/a/src" "$root/apps/b/src"
bin="$root/victory"
src="$root/crates/a/src/lib.rs"
manifest="$root/apps/b/Cargo.toml"
lock="$root/Cargo.lock"
: > "$bin"
: > "$src"
: > "$manifest"
: > "$lock"

check() {
  local got
  got=$(stale_source "$bin" "$root")
  if [ "$got" != "$2" ]; then
    echo "FAIL: $1: got '$got', expected '$2'" >&2
    exit 1
  fi
}

# The binary is newer than every source and manifest: current, nothing named.
touch -t 202001010000 "$src" "$manifest" "$lock"
touch -t 202001020000 "$bin"
check "binary newer" ""

# A source newer than the binary is named.
touch -t 202001030000 "$src"
check "source newer" "$src"

# The manifest too.
touch -t 202001010000 "$src"
touch -t 202001030000 "$manifest"
check "manifest newer" "$manifest"

# With every source and manifest older, the lock file still names the build's dependencies.
touch -t 202001010000 "$manifest"
touch -t 202001030000 "$lock"
check "lock newer" "$lock"

# A newer source wins over the lock.
touch -t 202001040000 "$src"
check "source before lock" "$src"

echo "stale source: ok"
