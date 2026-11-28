//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned:
//! `present` (swap chain).

#[cfg(target_os = "macos")]
mod ambient;
#[cfg(windows)]
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod clock;
/// A cache entry mapped read-only (`texture_cache`): the upload reads it without a heap copy.
mod mmap;
#[cfg(windows)]
#[cfg(windows)]
mod composite;
#[cfg(windows)]
mod compute;
pub mod console;
/// Decoded textures kept on disk (`nat`): the texture side of `shader_cache`.
pub mod texture_cache;
/// The hash title texture memory is compared and keyed by (XXH3, or blake3 when asked for).
pub mod content_hash;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
mod hold;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
pub mod list;
#[cfg(windows)]
mod memory;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod placement;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
// Plain std, so its tests run everywhere.
#[cfg(any(windows, test))]
mod ranges;
pub mod resource;
pub mod root;
mod rootser;
mod sampler;
#[cfg(any(windows, target_os = "macos"))]
pub mod shader_cache;
#[cfg(target_os = "macos")]
mod prewarm;
#[cfg(windows)]
mod shadow;
pub mod signature;
mod switch;
#[cfg(target_os = "macos")]
mod variant;
pub mod native;
#[cfg(windows)]
mod tdr;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;

#[cfg(all(windows, feature = "diagnostics"))]
pub use copystat::CountingAlloc;
#[cfg(windows)]
pub use host::on_failure;

/// Before host code writes title memory `[at, at + len)` — DirectStorage staging (a raw read, a
/// zlib or BCPack decode, a page list), a file read, XG tiling, an XMem copy — opens a released
/// decoded copy's pages, which are `PROT_NONE`. The bytes are about to be replaced, so nothing is
/// re-tiled from the texture; the GPU thread drops the copy at its next sync and reads the new
/// bytes (`Title::drop_host_written`). Cheap and lock-free (a page probe and `mprotect`), so any
/// host thread may call it before it writes. `true` when a page in the range was released.
#[cfg(target_os = "macos")]
pub fn title_memory_write(at: usize, len: usize) -> bool {
    // Whoever writes, the bytes are counted as streamed (`native::textures::host_wrote`): the memory
    // line splits the decoded copies by what their content came from, and a streamed copy's console
    // bytes are given back to the OS once they are uploaded.
    native::textures::host_wrote(at, len);
    victory_platform::faults::open_for_write(at, len)
}

/// Nothing is released on a host without the page-fault path.
#[cfg(not(target_os = "macos"))]
pub fn title_memory_write(at: usize, len: usize) -> bool {
    native::textures::host_wrote(at, len);
    false
}

/// Re-materialises the title pages of the released decoded copy that holds `addr`, from its
/// Metal texture: the callback the page-fault helper calls (`victory_platform::faults`). `false`
/// when no released copy holds `addr` (the fault is not ours) or no title exists yet.
#[cfg(target_os = "macos")]
pub fn restore_texture_page(addr: usize) -> bool {
    native::glue::with_title(|t| t.restore_at(addr)).unwrap_or(false)
}
/// Stops the host taking GPU work and then wakes and drains the GPU's queued work before the
/// process exits (`victory_metal::gfx::stop_gpu`). The order matters: work committed after the
/// drain belongs to no flight the stop sees, and would run on, and touch the GPU's objects,
/// while the process ends.
#[cfg(target_os = "macos")]
pub fn stop_gpu(wait: std::time::Duration) -> (usize, usize) {
    native::glue::stop_work();
    victory_metal::gfx::stop_gpu(wait)
}

/// How long a process end waits for the GPU's work before it exits anyway ([`stop_gpu_before_end`],
/// [`stop_gpu_and_exit`]): a stop that blocks (a title thread inside a host call, a driver that
/// will not return) must not keep the process, and its wired GPU memory, alive.
#[cfg(target_os = "macos")]
const STOP_BEFORE_EXIT: std::time::Duration = std::time::Duration::from_secs(3);

/// [`stop_gpu`] with a watchdog: if the stop itself has not finished `wait` plus two seconds later
/// (the title lock held by a thread stuck in a GPU call, a residency set that will not commit),
/// the process ends anyway. Returns nothing; the caller ends the process.
#[cfg(target_os = "macos")]
fn stop_gpu_before_exit(wait: std::time::Duration, code: i32) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let done = std::sync::Arc::new(AtomicBool::new(false));
    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }
    // SAFETY: ends the process; the watchdog only does it when the stop below it is stuck.
    {
        let done = done.clone();
        let _ = std::thread::Builder::new()
            .name("stop-gpu-deadline".into())
            .spawn(move || {
                std::thread::sleep(wait + std::time::Duration::from_secs(2));
                if !done.load(Ordering::SeqCst) {
                    unsafe { _exit(code) }
                }
            });
    }
    let (in_flight, left) = stop_gpu(wait);
    done.store(true, Ordering::SeqCst);
    tracing::warn!(
        in_flight,
        left,
        "stopped the GPU's work before the process ends (left: not finished in time, paid on the CPU)"
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

/// Wakes and drains the GPU's work on a deadline, then returns: for a path that ends the process
/// another way (an unhandled exception, which then aborts). If the stop is stuck, the process
/// exits with code 1 rather than keeping its wired GPU memory until a reboot (#357).
#[cfg(target_os = "macos")]
pub fn stop_gpu_before_end() {
    stop_gpu_before_exit(STOP_BEFORE_EXIT, 1);
}

/// Wakes and drains the GPU's work on a deadline, then ends the process with `code` (`_exit`, not
/// the C runtime's path: a thread stopped in the middle of a call must not run its teardown). Use
/// this from any path that ends the process while title work may be in flight.
#[cfg(target_os = "macos")]
pub fn stop_gpu_and_exit(code: i32, wait: std::time::Duration) -> ! {
    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }
    stop_gpu_before_exit(wait, code);
    // SAFETY: ends the process; the GPU's work was woken, drained or paid.
    unsafe { _exit(code) }
}

/// [`stop_gpu_and_exit`] with the default deadline ([`STOP_BEFORE_EXIT`]).
#[cfg(target_os = "macos")]
pub fn stop_gpu_and_exit_now(code: i32) -> ! {
    stop_gpu_and_exit(code, STOP_BEFORE_EXIT)
}

/// The window's visibility, for its presents (`victory_metal::present::set_window_hidden`).
#[cfg(target_os = "macos")]
pub use victory_metal::present::set_window_hidden;
#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, fault_tracking, title_memory_aliased, title_memory_remapped,
    writing as kernel_will_write,
};

/// A whole file, read without asking for its size or other metadata.
/// `std::fs::read` and `read_to_end` call `File::metadata()`, which Wine
/// answers with `NtQueryVolumeInformationFile`: an ioctl to its mount
/// manager whose completion runs as a system APC inside Wine's virtual
/// memory lock. Under Rosetta a thread was paused there by SIGUSR1 with
/// the lock held, and every other thread then waited on it for good (#50).
/// The shader cache read files this way on the pipeline workers.
pub fn read_file(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut out = Vec::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        match file.read(&mut chunk) {
            Ok(0) => return Ok(out),
            Ok(n) => out.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// Makes folder `dir` and its parents unless it is there. Not
/// `std::fs::create_dir_all` alone, which on a folder that is there asks for
/// its metadata (`Path::is_dir`): the same mount manager call as in
/// [`read_file`]. The pipeline library makes its folder at every save, on
/// the pipeline workers.
pub fn make_dir(dir: &std::path::Path) -> std::io::Result<()> {
    match std::fs::create_dir(dir) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir_all(dir),
        made => made,
    }
}

#[cfg(test)]
mod read_file_tests {
    #[test]
    fn whole_files_come_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        for len in [0usize, 1, 65_535, 65_536, 65_537, 300_000] {
            let path = dir.path().join(format!("f{len}"));
            let data: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
            std::fs::write(&path, &data).unwrap();
            assert_eq!(super::read_file(&path).unwrap(), data, "{len} bytes");
        }
        assert!(super::read_file(&dir.path().join("missing")).is_err());
    }
}
