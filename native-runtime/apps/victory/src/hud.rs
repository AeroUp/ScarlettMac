//! The window's HUD: a dark, translucent, monospaced panel over the top left of the content view
//! that reads live figures from lock-free counters (frames, draws, GPU time, host texture copies)
//! and from Mach (`task_info`: resident bytes, CPU time, and `phys_footprint`).
//!
//! Why it is not the title bar: AppKit sets a window's title by asking the window server, which
//! stops answering while the frame that would update it is stalled, so a title-bar readout freezes
//! exactly when the run is in trouble. This panel is painted by an `NSTimer` on the AppKit main
//! thread from atomics alone, so it keeps updating (and can say how long the last frame was) while
//! the title's render thread is stuck in a submission.
//!
//! It never takes an event: the view is an `NSView` subclass whose `hitTest:` returns nil, so the
//! mouse looks straight through it to the content view and the title's input is unchanged. F3
//! shows and hides it; `VICTORY_HUD=0` starts without it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use objc2::rc::autoreleasepool;
use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2::{class, msg_send, sel};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

use victory_gpu::console::Console;

/// Panel size in points, its distance from the window's top-left corner, and the frame gaps the
/// sparkline shows.
const WIDTH: f64 = 560.0;
const HEIGHT: f64 = 100.0;
const MARGIN: f64 = 14.0;
const LINE_HEIGHT: f64 = 16.0;
const PADDING: f64 = 10.0;
const SPARK_SAMPLES: usize = 40;
/// How often the panel repaints (seconds).
const TICK: f64 = 0.25;
/// A gap longer than this between presents is a stall, named in the FPS line (microseconds).
const STALLED: f64 = 500_000.0;

/// The panel's live objects and the counters its last paint compared against. Only the AppKit main
/// thread touches it; the mutex is there for interior mutability, not for contention.
struct State {
    view: usize,
    labels: [usize; 5],
    last_tick: Option<Instant>,
    frames: u64,
    draws: u64,
    gpu_nanos: u64,
    cpu_micros: u64,
    draws_per_frame: f64,
}

static HUD: Mutex<Option<State>> = Mutex::new(None);
/// Whether the panel is shown (F3 flips it).
static VISIBLE: AtomicBool = AtomicBool::new(true);

/// Adds the HUD to `content` (a window's content view), unless `VICTORY_HUD=0`. Call on the main
/// thread after the content view has its layer.
pub(crate) fn install(content: *mut AnyObject) {
    if !hud_on(std::env::var("VICTORY_HUD").ok().as_deref()) {
        tracing::info!("hud: off (VICTORY_HUD=0)");
        return;
    }
    // SAFETY: the content view of the live window, on the main thread.
    unsafe {
        autoreleasepool(|_| {
            build(content);
        })
    };
    tracing::info!("hud: shown; F3 hides it, VICTORY_HUD=0 starts without it");
}

/// `VICTORY_HUD` as read: on unless it is `0`.
fn hud_on(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Shows the panel if it is hidden and hides it if it is shown (F3). A run started with
/// `VICTORY_HUD=0` has none to toggle.
pub(crate) fn toggle() {
    let view = HUD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|s| s.view);
    let Some(view) = view else { return };
    let visible = !VISIBLE.load(Ordering::Relaxed);
    VISIBLE.store(visible, Ordering::Relaxed);
    // SAFETY: the AppKit main thread, and a view this module made and the content view retains.
    unsafe {
        let _: () = msg_send![view as *mut AnyObject, setHidden: Bool::from(!visible)];
    }
    tracing::info!(visible, "hud: toggled with F3");
    if visible {
        update();
    }
}

/// The view's class: `NSView` with `hitTest:` answering nil and our timer's `tick:`.
fn view_class() -> &'static AnyClass {
    static CLASS: OnceLock<&'static AnyClass> = OnceLock::new();
    CLASS.get_or_init(|| {
        let mut builder = ClassBuilder::new(c"VictoryHudView", class!(NSView))
            .expect("the HUD view class is registered once");
        // SAFETY: `hitTest:` matches NSView's signature (a nullable object return and one point),
        // and `tick:` is ours (an NSTimer argument, no return).
        unsafe {
            builder.add_method(
                sel!(hitTest:),
                hit_test as unsafe extern "C-unwind" fn(_, _, _) -> _,
            );
            builder.add_method(sel!(tick:), tick as unsafe extern "C-unwind" fn(_, _, _));
        }
        builder.register()
    })
}

/// `-[NSView hitTest:]` returning nil: no event is ever routed to the HUD, so clicks, drags and
/// scrolls reach the view under it (and the title's input) exactly as if it were not there.
unsafe extern "C-unwind" fn hit_test(
    _this: &AnyObject,
    _cmd: Sel,
    _point: NSPoint,
) -> Option<&'static AnyObject> {
    None
}

/// The timer's selector: repaint from the counters.
unsafe extern "C-unwind" fn tick(_this: &AnyObject, _cmd: Sel, _timer: &AnyObject) {
    update();
}

/// Builds the panel, its labels and its timer on the main thread.
unsafe fn build(content: *mut AnyObject) {
    // SAFETY: documented AppKit calls on the main thread; every receiver is a class or an object
    // this function just made, and the content view retains the panel (and the panel its labels).
    unsafe {
        let bounds: NSRect = msg_send![content, bounds];
        let frame = NSRect::new(
            NSPoint::new(MARGIN, bounds.size.height - HEIGHT - MARGIN),
            NSSize::new(WIDTH, HEIGHT),
        );
        let allocated: *mut AnyObject = msg_send![view_class(), alloc];
        let hud: *mut AnyObject = msg_send![allocated, initWithFrame: frame];
        if hud.is_null() {
            return;
        }
        // Stay at the top left when the window is resized: a flexible right margin and bottom
        // margin (NSViewMaxXMargin | NSViewMinYMargin).
        let _: () = msg_send![hud, setAutoresizingMask: 12usize];
        let _: () = msg_send![hud, setWantsLayer: Bool::YES];
        let layer: *mut AnyObject = msg_send![hud, layer];
        let background: *mut AnyObject = msg_send![
            class!(NSColor),
            colorWithSRGBRed: 0.035f64,
            green: 0.045f64,
            blue: 0.065f64,
            alpha: 0.80f64
        ];
        let background: *mut std::ffi::c_void = msg_send![background, CGColor];
        let border: *mut AnyObject = msg_send![
            class!(NSColor),
            colorWithSRGBRed: 1.0f64,
            green: 1.0f64,
            blue: 1.0f64,
            alpha: 0.10f64
        ];
        let border: *mut std::ffi::c_void = msg_send![border, CGColor];
        let _: () = msg_send![layer, setBackgroundColor: background];
        let _: () = msg_send![layer, setBorderColor: border];
        let _: () = msg_send![layer, setBorderWidth: 1.0f64];
        let _: () = msg_send![layer, setCornerRadius: 10.0f64];

        let font: *mut AnyObject =
            msg_send![class!(NSFont), monospacedSystemFontOfSize: 10.5f64, weight: 0.0f64];
        let header_font: *mut AnyObject =
            msg_send![class!(NSFont), monospacedSystemFontOfSize: 10.5f64, weight: 0.3f64];
        let white: *mut AnyObject = msg_send![
            class!(NSColor),
            colorWithSRGBRed: 0.94f64,
            green: 0.96f64,
            blue: 1.0f64,
            alpha: 1.0f64
        ];
        let grey: *mut AnyObject = msg_send![
            class!(NSColor),
            colorWithSRGBRed: 0.72f64,
            green: 0.78f64,
            blue: 0.86f64,
            alpha: 1.0f64
        ];
        let green: *mut AnyObject = msg_send![
            class!(NSColor),
            colorWithSRGBRed: 0.55f64,
            green: 0.93f64,
            blue: 0.70f64,
            alpha: 1.0f64
        ];

        let console = victory_gpu::console::current();
        let commit = option_env!("VICTORY_GIT_COMMIT").unwrap_or_default();
        let lines = render(console, commit, &Readout::default());
        let mut labels = [0usize; 5];
        for (i, line) in lines.iter().enumerate() {
            let y = HEIGHT - PADDING - (i as f64 + 1.0) * LINE_HEIGHT;
            let field = label(line, font, if i == 0 { green } else { grey }, y, hud);
            if i == 0 {
                let _: () = msg_send![field as *mut AnyObject, setFont: header_font];
                let _: () = msg_send![field as *mut AnyObject, setTextColor: white];
            }
            labels[i] = field;
        }
        let _: () = msg_send![content, addSubview: hud];

        *HUD.lock().unwrap_or_else(|e| e.into_inner()) = Some(State {
            view: hud as usize,
            labels,
            last_tick: None,
            frames: 0,
            draws: 0,
            gpu_nanos: 0,
            cpu_micros: 0,
            draws_per_frame: 0.0,
        });

        let timer: *mut AnyObject = msg_send![
            class!(NSTimer),
            timerWithTimeInterval: TICK,
            target: hud,
            selector: sel!(tick:),
            userInfo: std::ptr::null::<AnyObject>(),
            repeats: Bool::YES
        ];
        // Common modes keep the panel live while the window is being dragged or resized.
        let mode = NSString::from_str("kCFRunLoopCommonModes");
        let run_loop: *mut AnyObject = msg_send![class!(NSRunLoop), mainRunLoop];
        let _: () = msg_send![run_loop, addTimer: timer, forMode: &*mode];
    }
}

/// One label in `parent`, its address back (the panel retains it).
unsafe fn label(
    text: &str,
    font: *mut AnyObject,
    color: *mut AnyObject,
    y: f64,
    parent: *mut AnyObject,
) -> usize {
    // SAFETY: an NSTextField label, configured with AppKit's own setters.
    unsafe {
        let field: *mut AnyObject =
            msg_send![class!(NSTextField), labelWithString: &*NSString::from_str(text)];
        if field.is_null() {
            return 0;
        }
        let _: () = msg_send![field, setFont: font];
        let _: () = msg_send![field, setTextColor: color];
        let _: () = msg_send![field, setFrame: NSRect::new(NSPoint::new(12.0, y), NSSize::new(WIDTH - 24.0, LINE_HEIGHT - 1.0))];
        let _: () = msg_send![parent, addSubview: field];
        field as usize
    }
}

/// Repaints the panel from the counters. Runs on the AppKit main thread and takes no lock the
/// render thread can hold: only atomics and Mach. The first tick only takes a baseline, so CPU and
/// GPU percentages are not the whole process's history divided by a timer interval.
fn update() {
    let mut guard = HUD.lock().unwrap_or_else(|e| e.into_inner());
    let Some(state) = guard.as_mut() else { return };
    let now = Instant::now();
    let pacing = victory_gpu::native::frame_pacing();
    let (resident, cpu_micros) = mach::resident_and_cpu();
    let footprint = victory_platform::mem::footprint() as u64;
    let (copy_bytes, budget_bytes) = victory_gpu::native::texture_budget_bytes();
    let draws_total = victory_gpu::native::draws();
    let gpu_total = victory_gpu::native::gpu_busy_nanos();

    let mut readout = Readout {
        footprint,
        resident,
        copy_bytes,
        budget_bytes,
        ..Readout::default()
    };
    let recent = &pacing.recent[pacing.recent.len().saturating_sub(SPARK_SAMPLES)..];
    readout.frame_ms = recent.last().map_or(0.0, |&us| us as f64 / 1e3);
    if !recent.is_empty() {
        readout.avg_ms = recent.iter().map(|&us| us as f64).sum::<f64>() / recent.len() as f64 / 1e3;
    }
    readout.spark = sparkline(recent);
    readout.stalled_ms = pacing.since_last_micros as f64 / 1e3;

    if let Some(last) = state.last_tick {
        let wall = now.duration_since(last).as_secs_f64().max(1e-9);
        let frames = pacing.frames.saturating_sub(state.frames);
        readout.fps = frames as f64 / wall;
        readout.cpu_percent =
            cpu_micros.saturating_sub(state.cpu_micros) as f64 / (wall * 1e6) * 100.0;
        readout.gpu_percent =
            (gpu_total.saturating_sub(state.gpu_nanos) as f64 / (wall * 1e9) * 100.0)
                .clamp(0.0, 100.0);
        if frames > 0 {
            state.draws_per_frame = draws_total.saturating_sub(state.draws) as f64 / frames as f64;
        }
    }
    readout.draws_per_frame = state.draws_per_frame;
    state.last_tick = Some(now);
    state.frames = pacing.frames;
    state.draws = draws_total;
    state.gpu_nanos = gpu_total;
    state.cpu_micros = cpu_micros;

    if !VISIBLE.load(Ordering::Relaxed) {
        return;
    }
    let console = victory_gpu::console::current();
    let commit = option_env!("VICTORY_GIT_COMMIT").unwrap_or_default();
    let lines = render(console, commit, &readout);
    for (i, line) in lines.iter().enumerate() {
        // SAFETY: labels the panel retains, on the AppKit main thread.
        unsafe {
            let _: () =
                msg_send![state.labels[i] as *mut AnyObject, setStringValue: &*NSString::from_str(line)];
        }
    }
}

/// What one paint shows.
#[derive(Default, Clone, Debug, PartialEq)]
struct Readout {
    fps: f64,
    frame_ms: f64,
    avg_ms: f64,
    spark: String,
    stalled_ms: f64,
    footprint: u64,
    resident: u64,
    copy_bytes: u64,
    budget_bytes: u64,
    cpu_percent: f64,
    gpu_percent: f64,
    draws_per_frame: f64,
}

/// The panel's five lines: the identity header and one line each for frames, RAM, textures and
/// processors.
fn render(console: Console, commit: &str, r: &Readout) -> [String; 5] {
    let header = crate::window::window_title(console, commit);
    let frames = if r.fps < 0.1 && r.stalled_ms >= STALLED / 1e3 {
        // No sparkline here: during a stall it would show the frames from before it, and the
        // stall is the figure that matters.
        format!(
            "FPS  {:.1}   STALLED {:.1} s   last {:.1} ms   avg {:.1} ms",
            r.fps,
            r.stalled_ms / 1e3,
            r.frame_ms,
            r.avg_ms
        )
    } else {
        format!(
            "FPS {:>5.1}  frame {:>5.1} ms  avg {:>5.1} ms  {}",
            r.fps, r.frame_ms, r.avg_ms, r.spark
        )
    };
    // The title's own committed title memory beside the process's, so a footprint that outgrows
    // the machine shows which side grew (the console budget is the cap).
    let title = victory_hle::modules::xmem::title_memory();
    let ram = format!(
        "RAM {} footprint   {} resident   title {} of {}",
        size(r.footprint),
        size(r.resident),
        size(title.all()),
        size(title.limit),
    );
    let textures = if r.budget_bytes > 0 {
        format!(
            "TEX {:.0} MB in host copies   budget {}",
            r.copy_bytes as f64 / (1u64 << 20) as f64,
            size(r.budget_bytes)
        )
    } else {
        format!(
            "TEX {:.0} MB in host copies   no budget",
            r.copy_bytes as f64 / (1u64 << 20) as f64
        )
    };
    let cpu = format!(
        "CPU {:>4.0}%   GPU {:>3.0}%   {:>5.0} draws/frame",
        r.cpu_percent, r.gpu_percent, r.draws_per_frame
    );
    [header, frames, ram, textures, cpu]
}

/// Bytes as the HUD writes them: GiB with two decimals from 1 GiB up, else MiB.
fn size(bytes: u64) -> String {
    let mib = bytes as f64 / (1u64 << 20) as f64;
    if mib >= 1024.0 {
        format!("{:.2} GB", mib / 1024.0)
    } else {
        format!("{mib:.0} MB")
    }
}

/// Frame gaps (microseconds) as blocks, tall for slow frames: the shape of the pacing over the last
/// few seconds. Eight levels of U+2581..U+2588, boundaries at common frame-time grades.
fn sparkline(micros: &[u64]) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    micros
        .iter()
        .map(|&us| {
            let ms = us as f64 / 1e3;
            let level = match ms {
                m if m < 10.0 => 0,
                m if m < 16.0 => 1,
                m if m < 20.0 => 2,
                m if m < 25.0 => 3,
                m if m < 34.0 => 4,
                m if m < 50.0 => 5,
                m if m < 100.0 => 6,
                _ => 7,
            };
            BLOCKS[level]
        })
        .collect()
}

/// Mach queries for the HUD: the process's resident bytes and the CPU time its threads have used.
mod mach {
    type KernReturn = i32;
    type MachPort = u32;
    const KERN_SUCCESS: KernReturn = 0;
    /// `TASK_BASIC_INFO`, whose data is a `mach_task_basic_info`.
    const TASK_BASIC_INFO: i32 = 20;

    #[repr(C)]
    #[derive(Default)]
    struct TimeValue {
        seconds: i32,
        microseconds: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct TaskBasicInfo {
        _virtual_size: u64,
        resident_size: u64,
        _resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        _policy: i32,
        _suspend_count: i32,
    }

    unsafe extern "C" {
        static mach_task_self_: MachPort;
        fn task_info(task: MachPort, flavor: i32, info: *mut i32, count: *mut u32) -> KernReturn;
    }

    /// The process's resident bytes and the CPU microseconds all its threads have used (user and
    /// system), or zeros if the kernel refuses the query.
    pub(super) fn resident_and_cpu() -> (u64, u64) {
        let mut info = TaskBasicInfo::default();
        let mut count = (std::mem::size_of::<TaskBasicInfo>() / 4) as u32;
        // SAFETY: `info` holds `count` 32-bit words, which `TASK_BASIC_INFO` fills.
        let kr = unsafe {
            task_info(
                mach_task_self_,
                TASK_BASIC_INFO,
                (&mut info as *mut TaskBasicInfo).cast::<i32>(),
                &mut count,
            )
        };
        if kr != KERN_SUCCESS {
            return (0, 0);
        }
        let time = |t: &TimeValue| {
            u64::from(t.seconds.max(0) as u32) * 1_000_000
                + u64::from(t.microseconds.max(0) as u32)
        };
        // The resident figure from the platform's one implementation of it (flavor 20's sizes);
        // the CPU time is here, where the panel needs it.
        (victory_platform::mem::resident() as u64, time(&info.user_time) + time(&info.system_time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_is_never_the_view_an_event_reaches() {
        autoreleasepool(|_| {
            // SAFETY: AppKit's view factory on a live class; the view is released at the end.
            unsafe {
                let allocated: *mut AnyObject = msg_send![view_class(), alloc];
                let view: *mut AnyObject = msg_send![
                    allocated,
                    initWithFrame: NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(10.0, 10.0))
                ];
                assert!(!view.is_null());
                // `hitTest:` is nil for every point, inside the frame or not.
                let hit: *mut AnyObject = msg_send![view, hitTest: NSPoint::new(5.0, 5.0)];
                assert!(hit.is_null());
                let hit: *mut AnyObject = msg_send![view, hitTest: NSPoint::new(500.0, 500.0)];
                assert!(hit.is_null());
                let _: () = msg_send![view, release];
            }
        });
    }

    #[test]
    fn the_hud_is_on_unless_switched_off() {
        assert!(hud_on(None));
        assert!(hud_on(Some("1")));
        assert!(hud_on(Some("")));
        assert!(!hud_on(Some("0")));
        assert!(!hud_on(Some(" 0\n")));
    }

    #[test]
    fn the_header_is_the_window_title_and_every_figure_is_named() {
        let readout = Readout {
            fps: 58.9,
            frame_ms: 17.0,
            avg_ms: 16.8,
            spark: "▁▂▃".into(),
            footprint: 6 << 30,
            resident: 5 << 30,
            copy_bytes: 1200 << 20,
            budget_bytes: 1536 << 20,
            cpu_percent: 143.0,
            gpu_percent: 61.0,
            draws_per_frame: 412.0,
            stalled_ms: 0.0,
        };
        let lines = render(Console::SeriesS, "24c26f0", &readout);
        assert_eq!(lines[0], "ScarlettMac Native | Series S | 24c26f0");
        assert!(lines[1].contains("58.9"), "{}", lines[1]);
        assert!(lines[1].contains("17.0 ms"), "{}", lines[1]);
        assert!(lines[1].contains("16.8 ms"), "{}", lines[1]);
        assert!(lines[1].contains("▁▂▃"), "{}", lines[1]);
        assert!(lines[2].contains("6.00 GB footprint"), "{}", lines[2]);
        assert!(lines[2].contains("5.00 GB resident"), "{}", lines[2]);
        assert!(lines[3].contains("1200 MB"), "{}", lines[3]);
        assert!(lines[3].contains("1.50 GB"), "{}", lines[3]);
        assert!(lines[4].contains("143%"), "{}", lines[4]);
        assert!(lines[4].contains("61%"), "{}", lines[4]);
        assert!(lines[4].contains("412 draws/frame"), "{}", lines[4]);
    }

    #[test]
    fn a_stall_is_named_in_the_fps_line() {
        let readout = Readout {
            fps: 0.0,
            frame_ms: 16.7,
            avg_ms: 16.8,
            spark: "▁▁".into(),
            stalled_ms: 2400.0,
            ..Readout::default()
        };
        let lines = render(Console::SeriesX, "", &readout);
        assert!(lines[1].contains("STALLED 2.4 s"), "{}", lines[1]);
        assert!(lines[1].contains("last 16.7 ms"), "{}", lines[1]);
        assert_eq!(lines[0], "ScarlettMac Native | Series X");
    }

    #[test]
    fn no_budget_reads_as_none_and_sizes_scale() {
        let readout = Readout {
            copy_bytes: 640 << 20,
            ..Readout::default()
        };
        let lines = render(Console::SeriesS, "", &readout);
        assert!(lines[3].contains("640 MB"), "{}", lines[3]);
        assert!(lines[3].contains("no budget"), "{}", lines[3]);
        assert_eq!(size(1536 << 20), "1.50 GB");
        assert_eq!(size(1 << 20), "1 MB");
    }

    #[test]
    fn the_sparkline_rises_with_slower_frames() {
        let spark = sparkline(&[5_000, 15_000, 18_000, 22_000, 30_000, 40_000, 80_000, 400_000]);
        assert_eq!(spark, "▁▂▃▄▅▆▇█");
        assert_eq!(sparkline(&[]), "");
        assert_eq!(sparkline(&[10_000]), "▂");
    }
}
