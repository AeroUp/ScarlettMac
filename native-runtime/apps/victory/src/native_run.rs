//! `victory run` on macOS (x86-64, the title under Rosetta): no Wine. The
//! image is mapped by `victory_loader::native`, its imports bound against
//! `victory-hle`, and the entry point run on a title thread. Everything the
//! registry does not serve is bound to a logging stub (`--unimplemented`),
//! so loading never stops at a missing export.
//!
//! The DLLs the title ships beside its executable (the C++ runtime, Bink,
//! libHttpClient, ...) are mapped and bound the same way, through the same
//! resolver: our runtime first, then the host, then the other title DLLs.
//! Their `DllMain` runs only with `--execute`, on the title's main thread,
//! before the executable's own TLS callbacks and entry point.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};

use anyhow::{Context, Result};
use victory_hle::modules::xmem_physical::{Observer, PoolInfo};
use victory_hle::{HleResolver, HostPolicy};
use victory_loader::native::dll::{self, TitleDll, WithTitleDlls};
use victory_loader::native::{self, NativeImage, teb};
use victory_loader::{BindReport, ImportResolver, Provenance, Resolution, UnimplementedPolicy};
use victory_pe::ImportSymbol;

use crate::{Console, Unimplemented};

/// Remembers who served each import, for the listing.
struct Recording<'a> {
    inner: &'a dyn ImportResolver,
    seen: Mutex<Vec<(String, String, Option<Provenance>)>>,
}

impl<'a> Recording<'a> {
    fn new(inner: &'a dyn ImportResolver) -> Self {
        Recording {
            inner,
            seen: Mutex::default(),
        }
    }

    /// What was resolved since the last call.
    fn take(&self) -> Vec<(String, String, Option<Provenance>)> {
        std::mem::take(&mut *self.seen.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl ImportResolver for Recording<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        let resolution = self.inner.resolve(dll, symbol);
        let provenance = match &resolution {
            Resolution::Found { provenance, .. } => Some(*provenance),
            Resolution::Missing => None,
        };
        self.seen.lock().unwrap_or_else(|e| e.into_inner()).push((
            dll.to_owned(),
            symbol.to_string(),
            provenance,
        ));
        resolution
    }
}

/// DLLs our runtime serves are never taken from the title's folder; any
/// other the title ships is (the host has no C++ runtime to offer).
fn title_dll(dll: &str) -> bool {
    !victory_hle::registry().iter().any(|m| m.serves(dll))
}

/// The title DLLs that stubbed imports come from but that were never
/// mapped: not in the title's folder. Lower case, sorted, each once.
fn absent_title_dlls(
    stubbed_from: &[String],
    title: &dyn Fn(&str) -> bool,
    mapped: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let names: std::collections::BTreeSet<String> = stubbed_from
        .iter()
        .map(|d| d.to_ascii_lowercase())
        .filter(|d| title(d) && !mapped(d))
        .collect();
    names.into_iter().collect()
}

/// Binds and finalizes `image`, then prints what was bound (and, with
/// `list`, every import and who served it).
fn bind(
    name: &str,
    image: &NativeImage,
    resolver: &dyn ImportResolver,
    policy: UnimplementedPolicy,
    list: bool,
) -> Result<BindReport> {
    let recording = Recording::new(resolver);
    let report = native::bind_imports(image, &recording, policy)
        .with_context(|| format!("binding {name}"))?;
    image.finalize()?;
    println!(
        "bound          {name}: {} hle, {} host, {} title, {} stubbed",
        report.hle,
        report.host,
        report.title,
        report.missing.len()
    );
    let seen = recording.take();
    if list {
        for (dll, symbol, provenance) in &seen {
            let how = match provenance {
                Some(Provenance::Hle) => "native (hle)",
                Some(Provenance::Host) => "native (host)",
                Some(Provenance::Title) => "native (title)",
                None => "STUB",
            };
            println!("  {how:<14} {dll}!{symbol}");
        }
        // Served and stubbed per DLL imported.
        let mut by_dll: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for (dll, _, provenance) in &seen {
            let entry = by_dll.entry(dll).or_default();
            if provenance.is_some() {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
        println!("imports of     {name}, by dll (served, stub):");
        for (dll, (served, stub)) in &by_dll {
            println!("  {dll}: {served}, {stub}");
        }
    }
    if !report.missing.is_empty() {
        let mut by_dll: BTreeMap<&str, usize> = BTreeMap::new();
        for m in &report.missing {
            *by_dll.entry(&m.dll).or_default() += 1;
        }
        println!(
            "unresolved     {name}: {} imports in {} dlls:",
            report.missing.len(),
            by_dll.len()
        );
        for (dll, n) in &by_dll {
            println!("  {dll}: {n}");
        }
    }
    Ok(report)
}

/// The Windows loader lock's shape: one thread at a time, re-entered by the
/// thread that holds it (a `DllMain` that loads another DLL).
#[derive(Default)]
struct LoaderLock {
    owner: Mutex<(Option<std::thread::ThreadId>, usize)>,
    free: Condvar,
}

impl LoaderLock {
    fn hold<T>(&self, f: impl FnOnce() -> T) -> T {
        let me = std::thread::current().id();
        let lock = || self.owner.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut owner = lock();
            while owner.0.is_some_and(|o| o != me) {
                owner = self.free.wait(owner).unwrap_or_else(|e| e.into_inner());
            }
            *owner = (Some(me), owner.1 + 1);
        }
        let result = f();
        let mut owner = lock();
        owner.1 -= 1;
        if owner.1 == 0 {
            owner.0 = None;
            self.free.notify_one();
        }
        result
    }
}

/// Records where a title DLL came from, for `GetModuleFileName` and the
/// by-name lookups of `LoadLibrary`.
fn note(dll: &TitleDll) {
    println!("title dll      {} at {:#x}", dll.name, dll.image.base());
    victory_hle::runtime::set_image_path(dll.image.base(), &dll.path);
    // The BVH builder's programs are recognised by living in its image
    // (`victory_gpu::native::builder`; only that DLL is kept).
    victory_gpu::native::builder::note_image(&dll.name, dll.image.base(), dll.image.len());
}

/// `LoadLibrary` of a title DLL nothing imported (or that an import did not
/// need until now): maps it and the title DLLs it needs, binds them, runs
/// their `DllMain` on the calling thread, and returns its base; `None` if
/// the title ships no such DLL.
fn load_late(
    file: &str,
    title_dir: &Path,
    policy: UnimplementedPolicy,
    hle: &HleResolver,
    lock: &LoaderLock,
) -> Result<Option<usize>> {
    use victory_hle::runtime::image_named;
    lock.hold(|| {
        if let Some(base) = image_named(file) {
            return Ok(Some(base));
        }
        let new = dll::map_title_dll(file, title_dir, &title_dll)?;
        let resolver = WithTitleDlls { inner: hle };
        for d in &new {
            note(d);
            if d.image.tls().is_some() {
                tracing::info!(dll = d.name, "populated static TLS for late-loaded DLL");
            }
            bind(&d.name, &d.image, &resolver, policy, false)?;
        }
        for d in &new {
            // SAFETY: bound and finalized above, dependencies first.
            let started = unsafe { native::attach_dll(&d.image, 1, 0) };
            anyhow::ensure!(started, "DllMain of {} returned FALSE", d.name);
            native::register_dll_thread_notices(&d.image);
        }
        Ok(new.last().map(|d| d.image.base()))
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    exe: &Path,
    unimplemented: Unimplemented,
    strict: bool,
    execute: bool,
    list_imports: bool,
    console: Console,
    title_args: &[String],
    headless: bool,
) -> Result<u32> {
    victory_gpu::clock::initialize();
    let title_dir = exe.parent().unwrap_or(Path::new("."));
    let bytes = std::fs::read(exe).with_context(|| format!("reading {}", exe.display()))?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    match victory_game::identify(&hash) {
        Some(build) => println!("build          {} ({hash})", build.label),
        None => println!("build          unknown ({hash})"),
    }
    let image = NativeImage::map(&bytes, None)?;
    drop(bytes);
    println!(
        "mapped at      {:#x} (preferred {:#x})",
        image.base(),
        image.info().image_base
    );
    if let Some(tls) = image.tls() {
        teb::register_tls(tls);
    }
    victory_hle::runtime::set_main_image(image.base());
    victory_hle::runtime::set_image_path(image.base(), exe);

    let host = if strict {
        HostPolicy::none()
    } else {
        HostPolicy::default()
    };
    let resolver =
        victory_hle::runtime::NATIVE_TITLE_RESOLVER.get_or_init(|| HleResolver::new(host));
    let policy = match unimplemented {
        Unimplemented::Trap => UnimplementedPolicy::Trap,
        Unimplemented::Zero => UnimplementedPolicy::ReturnZero,
    };

    // The title's own DLLs: mapped (dependencies first) before anything is
    // bound, as they bind against each other. Their entry points do not run
    // here, so a dry run runs no title code.
    let dlls = dll::map_title_dlls(image.info(), title_dir, &title_dll)?;
    for d in &dlls {
        note(d);
    }
    let chain = WithTitleDlls { inner: resolver };
    let mut stubbed_from = Vec::new();
    for d in &dlls {
        let report = bind(&d.name, &d.image, &chain, policy, list_imports)?;
        stubbed_from.extend(report.missing.into_iter().map(|m| m.dll));
    }
    let exe_name = exe.file_name().unwrap_or_default().to_string_lossy();
    let report = bind(&exe_name, &image, &chain, policy, list_imports)?;
    println!(
        "bound          {} hle, {} host, {} title, {} stubbed ({} title dlls)",
        report.hle,
        report.host,
        report.title,
        report.missing.len(),
        dlls.len()
    );
    stubbed_from.extend(report.missing.iter().map(|m| m.dll.clone()));
    let absent = absent_title_dlls(&stubbed_from, &title_dll, &|d| dll::find(d).is_some());
    for name in &absent {
        println!("missing dll    {name}: not in {}", title_dir.display());
    }
    if !absent.is_empty() {
        tracing::warn!(
            dlls = absent.len(),
            dir = %title_dir.display(),
            "title DLLs are not beside the executable, so their imports are stubbed: \
             point victory at the whole game folder"
        );
    }
    if !execute {
        println!("not executing (pass --execute to start the title)");
        return Ok(0);
    }

    // The manifest names the title and its storage quota; the console's
    // fixed drives are the package as `G:` and temporary files `T:`.
    let cfg = victory_core::Config::load()?;
    victory_gpu::shader_cache::use_native_cache(cfg.cache_subdir("native-shaders")?);
    // What each detile made, keyed by the bytes it made it from (off unless
    // `VICTORY_TEXTURE_CACHE=1`: under Rosetta a hit costs more than the detile it saves).
    victory_gpu::texture_cache::use_dir(cfg.cache_subdir("native-textures")?);
    // A released decoded copy's title pages are PROT_NONE: a title access faults, and this
    // helper's callback re-materialises the console bytes from the Metal texture before the
    // access runs (the loader's signal handler calls `victory_platform::faults::on_fault`).
    victory_platform::faults::install(victory_gpu::restore_texture_page);
    let manifest = match dll::find_file(title_dir, "MicrosoftGame.config") {
        Some(path) => std::fs::read_to_string(path).unwrap_or_default(),
        None => {
            tracing::warn!(
                dir = %title_dir.display(),
                "MicrosoftGame.config is not beside the executable, so the title ID is 00000000: \
                 point victory at the whole game folder"
            );
            String::new()
        }
    };
    let title = victory_hle::runtime::Title::from_manifest(&manifest);
    let storage = cfg.data_subdir(&format!("{:08X}/storage", title.id))?;
    println!(
        "title          {:08X}, storage {}",
        title.id,
        storage.display()
    );
    victory_hle::posix::mount_at('G', title_dir);
    // The title sees its executable on the package drive, then its arguments.
    let exe_name_on_g = exe.file_name().unwrap_or_default().to_string_lossy();
    let argv: Vec<String> = std::iter::once(format!("G:\\{exe_name_on_g}"))
        .chain(title_args.iter().cloned())
        .collect();
    victory_hle::posix::set_command_line(&argv);
    let temp = cfg.data_subdir(&format!("{:08X}/temp", title.id))?;
    victory_hle::posix::mount_at('T', &temp);
    victory_hle::runtime::set_title(victory_hle::runtime::Title { storage, ..title });
    victory_hle::console::set(match console {
        Console::SeriesX => victory_hle::console::Console::SeriesX,
        Console::SeriesS => victory_hle::console::Console::SeriesS,
    });
    println!("console        {}", victory_hle::console::current().name());

    // DLLs the title loads itself (the FidelityFX loader's upscaler, ...).
    let (late_dir, lock): (PathBuf, LoaderLock) = (title_dir.to_owned(), LoaderLock::default());
    victory_hle::runtime::set_title_dll_loader(Box::new(move |file| {
        load_late(file, &late_dir, policy, resolver, &lock).map_err(|e| format!("{e:#}"))
    }));

    // The image lives as long as the process: title threads outlive `run`.
    let image: &'static NativeImage = Box::leak(Box::new(image));
    connect_gpu();
    // An unhandled exception ends the process through the loader; wake and drain the GPU's work
    // first, on a deadline, so its command buffers cannot keep the process (and their memory) alive.
    native::seh::set_on_end(victory_gpu::stop_gpu_before_end);
    // The title's graphics go to a layer: the window's, or one with no window.
    let layer = victory_gpu::native::create_layer(crate::window::WIDTH, crate::window::HEIGHT)
        .map_err(|e| anyhow::anyhow!("no layer to present into: {e}"))?;
    victory_gpu::native::set_layer(layer.as_ptr())
        .map_err(|e| anyhow::anyhow!("no layer to present into: {e}"))?;
    let layer = layer.as_ptr();
    // Debugging aid: VICTORY_FRAME_DUMP=<seconds>,<file.ppm> saves the last frame presented and
    // logs the frame count and the shader work refused so far, once.
    if let Some((secs, path)) = std::env::var("VICTORY_FRAME_DUMP")
        .ok()
        .and_then(|v| {
            v.split_once(',')
                .map(|(s, p)| (s.parse::<u64>().ok(), p.to_owned()))
        })
        .and_then(|(s, p)| Some((s?, p)))
    {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            tracing::warn!(
                frames = victory_gpu::native::frames_presented(),
                shaders = ?victory_gpu::native::shader_stats(),
                "frame dump"
            );
            if let Some((w, h, bgra)) = victory_gpu::native::last_frame() {
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend(bgra.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]));
                let _ = std::fs::write(&path, out);
                tracing::warn!(w, h, path, "frame saved");
            }
        });
    }
    // VICTORY_FRAME_DUMP_EVERY=<seconds>,<dir> saves the last frame presented as <dir>/tNNN.ppm
    // every <seconds>: screenshots that need no screen-recording permission for the window.
    if let Some((secs, dir)) = std::env::var("VICTORY_FRAME_DUMP_EVERY")
        .ok()
        .and_then(|v| frame_dump_every(&v))
    {
        std::thread::spawn(move || {
            let _ = std::fs::create_dir_all(&dir);
            let mut at = 0u64;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(secs));
                at += secs;
                if let Some((w, h, bgra)) = victory_gpu::native::last_frame() {
                    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                    out.extend(bgra.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]));
                    let _ = std::fs::write(dir.join(format!("t{at:04}.ppm")), out);
                }
            }
        });
    }
    // What the process is made of, every 10 s: the title's own committed memory beside what the
    // host adds. The 14.9 GB footprint of the run of 27 November 22:23 (1.16 GB resident, 7.8 s a
    // frame, paged out on an 18 GB Mac) could not be attributed from the log; this line does it.
    spawn_memory_report();
    // `VICTORY_STALL_SECONDS=<n>`: when no frame has been presented for n seconds, say what the
    // host's last call is doing and dump every title thread's call, and keep saying so every n
    // seconds of the same stall. A title whose frame or heartbeat passes 30 s ends the run with
    // the watchdog's fail-fast; this is the record of what held it (off unless set).
    spawn_stall_report(image.base());
    let launch = Launch(image, dlls);
    let go = move || {
        let launch = launch;
        // SAFETY: imports are bound and the images finalized; from here on we
        // run title code.
        unsafe { native::run_entry_with(launch.0, &launch.1) }
    };
    let code = if headless {
        go()?
    } else {
        // AppKit owns this (the main) thread; the title runs on its own.
        crate::window::run(layer, go)?.unwrap_or(0)
    };
    println!("title exited with {code}");
    // What the run's caches saved it; the same line a signal stop writes (`main.rs`).
    victory_gpu::shader_cache::log_summary();
    // The title's work is done, but command buffers it committed can still be in flight: wake and
    // drain them (on a deadline) before the process's teardown reaches Metal.
    victory_gpu::stop_gpu_before_end();
    Ok(code)
}

/// What the title launch thread takes along: images that live for the process.
struct Launch(&'static NativeImage, Vec<&'static TitleDll>);
// SAFETY: the images are mapped memory that is never freed; the title runs on whichever thread
// starts it.
unsafe impl Send for Launch {}

/// Wires the title-facing graphics objects to the native host: their method tables become
/// bridged ones (the title is Windows x64), the runtime's physical-memory pools feed the GPU's
/// pool table, and fence events signal the title's event handles.
/// Logs what the process is made of every 10 s (see the call site): the title's committed title
/// memory (its console's CPU and GRAPHICS kinds, and the budget it is capped at), the host's
/// texture copies and their budget, Metal's own allocated bytes, the DXIL the translations hold,
/// and what is left of the footprint -- which is what tells paging apart from a leak.
/// `VICTORY_STALL_SECONDS=<n>`: while a title clock the watchdog watches has not moved for n
/// seconds, say why and which host call each title thread is in. Three clocks are read once a
/// second, without taking the host's lock:
///
/// * the presented frames (a frame, or the render thread's heartbeat, stalled);
/// * the watchdog's own global heartbeat (`image+0x971fb8`: it fails the process 30 s after this
///   stops changing);
/// * the watchdog's scope table (`image+0x6844a20`: it fails the process 300 s after a scope a
///   title thread entered and did not leave), each entry's age against the same QPC the entries
///   are stamped with.
///
/// The first three reports of a stall include every title thread's host call and call chain (the
/// loader's thread dump); after that only the line, so a long stall does not fill the log. Off
/// unless set.
fn spawn_stall_report(base: usize) {
    let Some(seconds) = std::env::var("VICTORY_STALL_SECONDS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
    else {
        return;
    };
    let every = std::time::Duration::from_secs(seconds);
    let _ = std::thread::Builder::new().name("stall-report".into()).spawn(move || {
        let tick = std::time::Duration::from_secs(1);
        let mut frames = victory_gpu::native::presented_frames();
        let mut frames_since = std::time::Instant::now();
        let mut heartbeat = native::watchdog::heartbeat(base);
        let mut heartbeat_since = std::time::Instant::now();
        // Whether the word at `image+0x971fb8` has ever changed in this run: a build that keeps
        // something else there would otherwise report a stall from its first second.
        let mut heartbeat_moves = false;
        let mut last_report: Option<std::time::Instant> = None;
        let mut reports = 0u32;
        loop {
            std::thread::sleep(tick);
            let now = victory_gpu::native::presented_frames();
            if now != frames {
                let took = frames_since.elapsed();
                if reports > 0 {
                    tracing::warn!(
                        frame = now,
                        since_last = format_args!("{:.1} s", took.as_secs_f64()),
                        reports,
                        "the stall ended: a frame was presented"
                    );
                }
                frames = now;
                frames_since = std::time::Instant::now();
                last_report = None;
                reports = 0;
            }
            let word = native::watchdog::heartbeat(base);
            if word != heartbeat {
                if heartbeat.is_some() {
                    heartbeat_moves = true;
                }
                heartbeat = word;
                heartbeat_since = std::time::Instant::now();
            }
            let scopes = native::watchdog::active(base, victory_gpu::clock::counter());
            let oldest = scopes
                .as_ref()
                .and_then(|s| s.iter().map(|a| a.age_s).fold(None, |m: Option<f64>, a| Some(m.map_or(a, |m| m.max(a)))))
                .unwrap_or(0.0);
            // A stall is any of the watchdog's clocks not moving, past the report's interval. A
            // scope only matters at its deadline (300 s): reporting one from its tenth second
            // filled a run's log with the load's own scope, which is entered and left on purpose.
            let frame_stall = frames_since.elapsed() >= every;
            let heartbeat_stall = heartbeat_moves && heartbeat_since.elapsed() >= every;
            let scope_stall = oldest + seconds as f64 >= native::watchdog::SCOPE_DEADLINE_S;
            if !(frame_stall || heartbeat_stall || scope_stall) {
                continue;
            }
            // One report an interval while it lasts, so its growth is visible without filling
            // the log with the same line.
            if last_report.is_some_and(|t| t.elapsed() < every) {
                continue;
            }
            last_report = Some(std::time::Instant::now());
            reports += 1;
            let phase = victory_gpu::native::host_phase();
            tracing::warn!(
                frame = frames,
                no_frame_for = format_args!("{:.1} s", frames_since.elapsed().as_secs_f64()),
                heartbeat_unchanged_for = format_args!(
                    "{:.1} s{}",
                    heartbeat_since.elapsed().as_secs_f64(),
                    if heartbeat_moves { "" } else { " (never changed in this run)" }
                ),
                oldest_scope_s = format_args!("{:.1}", oldest),
                report = reports,
                host_phase = phase.map_or("no host call in a phase", |p| p.0),
                phase_for = format_args!("{:.1} s", phase.map_or(0.0, |p| p.1)),
                host_thread = phase.map_or(0, |p| p.2),
                scopes = ?scopes.map(|s| s.iter().map(|a| format!("group {} depth {} age {:.0} s", a.group, a.depth, a.age_s)).collect::<Vec<_>>()),
                "a title clock the watchdog reads has stopped"
            );
            // Every title thread's host call and call chain, while there is room in the log for it.
            if reports <= 3 {
                native::dump::report_now();
            }
        }
    });
}

fn spawn_memory_report() {
    let _ = std::thread::Builder::new().name("memory-report".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(10));
        let title = victory_hle::modules::xmem::title_memory();
        let parts = victory_gpu::native::memory_parts();
        let (copies, limit, metal, dxil, containers, twins, footprint, deferred, decoded, released) = parts.as_ref().map_or(
            (0, 0, 0, 0, 0, 0, victory_platform::mem::footprint() as u64, 0, Default::default(), 0),
            |p| (p.copies, p.copies_limit, p.metal, p.dxil, p.containers, p.twins, p.footprint, p.deferred, p.decoded, p.released),
        );
        // Faults a released decoded copy's title pages took, and ones nothing could put back.
        #[cfg(target_os = "macos")]
        let (page_faults, page_misses, page_writes) = victory_platform::faults::counts();
        #[cfg(not(target_os = "macos"))]
        let (page_faults, page_misses, page_writes) = (0u64, 0u64, 0u64);
        // The ten largest render-target shapes the title created, in bytes of title memory: at
        // 720p they are what a Series S frame is made of, and the largest is where resolution
        // shows.
        let shapes = victory_gpu::native::render_target_shapes();
        let top: Vec<String> = shapes
            .iter()
            .take(10)
            .map(|&(bytes, count, w, h, format, layers, target)| {
                let layers = if layers > 1 { format!("x{layers}") } else { String::new() };
                format!(
                    "{w}x{h}f{format}{layers}{}x{count}={}MiB",
                    if target { "rt" } else { "uav" },
                    bytes >> 20
                )
            })
            .collect();
        let targets_mib: u64 = shapes.iter().map(|s| s.0).sum::<u64>() >> 20;
        // What the title itself is told: its console's title memory, not the Mac's RAM.
        let told = victory_gpu::console::current().title_memory();
        let known = title.used() + copies + dxil;
        let resident = victory_platform::mem::resident() as u64;
        // Metal's own allocations beyond what it wraps of the title's memory and the copies: if
        // this is small, nothing of the 7.6 GB of pools is counted twice.
        let metal_own = metal.saturating_sub(title.used() + copies);
        tracing::info!(
            title_mib = title.all() >> 20,
            title_cpu_mib = title.cpu >> 20,
            title_graphics_mib = title.graphics >> 20,
            pools = title.pools,
            title_pools_mib = title.used() >> 20,
            title_heap_mib = title.heap >> 20,
            title_limit_mib = title.limit >> 20,
            copies_mib = copies >> 20,
            copies_limit_mib = limit >> 20,
            metal_mib = metal >> 20,
            metal_own_mib = metal_own >> 20,
            resident_mib = resident >> 20,
            not_resident_mib = known.saturating_sub(footprint) >> 20,
            dxil_mib = dxil >> 20,
            containers_mib = containers >> 20,
            twins_mib = twins >> 20,
            footprint_mib = footprint >> 20,
            other_mib = footprint.saturating_sub(known) >> 20,
            deferred_copies = deferred,
            told_total_mib = (told.0 + told.1) >> 20,
            told_gpu_optimal_mib = told.0 >> 20,
            told_standard_mib = told.1 >> 20,
            "memory: the title's committed title memory, the host's copies, Metal, the shader cache, and the rest of the footprint"
        );
        tracing::info!(
            targets_mib,
            shapes = top.len(),
            largest = %top.first().map_or(String::new(), |s| s.clone()),
            top = ?top,
            "the render targets the title created, largest first (bytes of title memory)"
        );
        // Why the decoded copies are needed, and how much is sampled where the title wrote it:
        // a small pool means most of the title's textures are not copied at all. Each copy has
        // one reason, and the reasons' MiB add up to `decoded_mib` (`DecodedReport::reasons_mib`).
        let [tiled, bcpack, bc_mipped, format, depth, layout, multisampled, alignment, other] = decoded.reasons_mib();
        tracing::info!(
            zero_copy_mib = decoded.zero_copy >> 20,
            zero_copy_textures = decoded.zero_copy_copies,
            decoded_mib = decoded.decoded >> 20,
            tiled_mib = tiled,
            bcpack_mib = bcpack,
            bc_mipped_mib = bc_mipped,
            format_mib = format,
            depth_mib = depth,
            layout_mib = layout,
            multisampled_mib = multisampled,
            alignment_mib = alignment,
            other_mib = other,
            gpu_written_mib = decoded.written >> 20,
            sampled_only_mib = decoded.sampled_only >> 20,
            in_place_mib = decoded.in_place >> 20,
            in_place_copies = decoded.in_place_copies,
            streamed_mib = decoded.streamed >> 20,
            streamed_detiled_mib = decoded.streamed_detiled >> 20,
            streamed_bcpack_mib = decoded.streamed_bcpack >> 20,
            streamed_total_mib = victory_gpu::native::textures::streamed_bytes() >> 20,
            under_copies_mib = decoded.under_copies >> 20,
            duplicate_copies_mib = decoded.duplicate_copies >> 20,
            released_mib = released >> 20,
            streamed_released_mib = victory_gpu::native::textures::streamed_released_bytes() >> 20,
            streamed_released_undone_mib = victory_gpu::native::textures::streamed_released_undone().1 >> 20,
            streamed_released_undone = victory_gpu::native::textures::streamed_released_undone().0,
            released_faults = page_faults,
            released_misses = page_misses,
            released_writes = page_writes,
            "texture copies: what is sampled from title memory, what the decoded copies are needed for (one reason each: tiled swizzle, BCn one image, BCn mip chain, format, depth, layout, multisampling, alignment, other), and how much could be decoded into the title's own pages in place"
        );
    });
}

fn connect_gpu() {
    victory_gpu::object::set_bridge(victory_gpu::object::Bridge {
        is_bridged: native::is_bridged,
        copy_of: native::bridged_copy_of,
        expose: |this, len, area| {
            // SAFETY: forwarded: a live object of ours whose first word is such a table.
            unsafe { native::expose_object(this, len, area) }
        },
        declare_win64: native::declare_win64,
    });
    victory_hle::modules::xmem_physical::set_observer(Some(std::sync::Arc::new(PoolFeed(
        victory_gpu::native::pool_observer(),
    ))));
    victory_gpu::native::set_event_hook(victory_hle::posix::set_event);
    // Where in the title a render target's shape came from, for `a render target the title
    // created`'s `caller`: the only way to tell a 4K target the title chose from one a host
    // figure gave it.
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    victory_gpu::native::set_caller_hook(victory_loader::native::dump::caller_chain);
}

/// `victory_gpu`'s pool observer behind the runtime's (victory-gpu does not know victory-hle).
struct PoolFeed(std::sync::Arc<dyn victory_gpu::native::memory::PoolObserver>);

fn info(pool: &PoolInfo) -> victory_gpu::native::memory::PoolInfo {
    victory_gpu::native::memory::PoolInfo {
        id: pool.id,
        base: pool.base,
        len: pool.len,
        memory_entry: pool.memory_entry,
    }
}

impl Observer for PoolFeed {
    fn pool_created(&self, pool: &PoolInfo) {
        self.0.pool_created(&info(pool));
    }
    fn pool_released(&self, pool: &PoolInfo) {
        self.0.pool_released(&info(pool));
    }
    fn mapped(&self, at: usize, len: usize, pool: &PoolInfo, offset: usize) {
        self.0.mapped(at, len, &info(pool), offset);
    }
    fn unmapped(&self, at: usize, len: usize) {
        self.0.unmapped(at, len);
    }
}

#[cfg(test)]
mod tests {
    use super::absent_title_dlls;

    #[test]
    fn absent_title_dlls_are_named_once_and_runtime_dlls_never() {
        let stubbed_from = [
            "MSVCP140.dll",
            "msvcp140.dll",
            "KERNEL32.dll",
            "Bink2Scarlett.dll",
            "VCRUNTIME140.dll",
        ]
        .map(String::from);
        let ours = |d: &str| d == "kernel32.dll";
        let mapped = |d: &str| d == "vcruntime140.dll";
        assert_eq!(
            absent_title_dlls(&stubbed_from, &|d| !ours(d), &mapped),
            ["bink2scarlett.dll", "msvcp140.dll"]
        );
    }
}

/// Parses `VICTORY_FRAME_DUMP_EVERY`: `<seconds>,<dir>`, seconds at least 1.
fn frame_dump_every(v: &str) -> Option<(u64, std::path::PathBuf)> {
    let (s, d) = v.split_once(',')?;
    let s = s.trim().parse::<u64>().ok().filter(|&s| s > 0)?;
    (!d.is_empty()).then(|| (s, std::path::PathBuf::from(d)))
}

#[cfg(test)]
mod frame_dump_every_tests {
    use super::frame_dump_every;

    #[test]
    fn a_period_and_a_directory_parse() {
        assert_eq!(
            frame_dump_every("30,/tmp/shots"),
            Some((30, std::path::PathBuf::from("/tmp/shots")))
        );
    }

    #[test]
    fn a_zero_or_missing_period_or_directory_is_off() {
        assert_eq!(frame_dump_every("0,/tmp/x"), None);
        assert_eq!(frame_dump_every("x,/tmp/x"), None);
        assert_eq!(frame_dump_every("30,"), None);
        assert_eq!(frame_dump_every("30"), None);
    }
}
