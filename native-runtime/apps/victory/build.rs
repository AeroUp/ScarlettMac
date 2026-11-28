// The macOS host runs an x86-64 title under Rosetta. The default `__PAGEZERO`
// of an x86-64 executable is 4 GiB: the whole range below it is then
// unmappable, yet the title's 32-bit `XMemAlloc` / `XMemVirtualAlloc`
// addresses (and anything it keeps in a `u32`) must come from there. Shrink
// it to one page so the low 4 GiB can be reserved. Null still faults.
fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "macos" && arch == "x86_64" {
        println!("cargo:rustc-link-arg-bins=-Wl,-pagezero_size,0x1000");
    }

    // The commit this binary was built from, for the window title (and any
    // log that wants to say which build it is). Empty when git is not there,
    // as in a source archive or a CI checkout without it.
    let commit = git_short_commit().unwrap_or_default();
    println!("cargo:rustc-env=VICTORY_GIT_COMMIT={commit}");
    // A new commit must rebuild this script's output, or the title would keep
    // naming the previous one.
    if let Some(head) = git_head_path() {
        println!("cargo:rerun-if-changed={head}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}

/// The abbreviated commit of `HEAD`, as `git rev-parse --short=7` prints it.
fn git_short_commit() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let commit = String::from_utf8(out.stdout).ok()?;
    let commit = commit.trim();
    // A hash is hexadecimal. Anything else is not a commit (a missing
    // executable, a tarball's replacement text) and is left out.
    let valid = (7..=40).contains(&commit.len()) && commit.chars().all(|c| c.is_ascii_hexdigit());
    valid.then(|| commit.to_ascii_lowercase())
}

/// The file `HEAD` of this checkout, absolute so Cargo watches the right one
/// in a linked worktree (where `.git` is a file pointing elsewhere).
fn git_head_path() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--absolute-git-dir"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let dir = String::from_utf8(out.stdout).ok()?;
    let head = std::path::Path::new(dir.trim()).join("HEAD");
    Some(head.to_string_lossy().into_owned())
}
