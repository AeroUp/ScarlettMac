//! Present: the title's back buffer into the next drawable of a `CAMetalLayer` the app owns.
//!
//! The app creates the layer (a window's view layer, or [`Metal::create_layer`] for one that is
//! shown by something else, or by nothing, in tests) and hands it over by its address
//! ([`Metal::create_surface`]); the backend draws the title's texture over the drawable with a
//! full-screen triangle, so that any size and channel order (the title's RGBA, the layer's BGRA) works,
//! and presents it after the work already submitted to the queue.

use crate::metal::{Metal, Queue};
use crate::gfx::Texture;
use crate::resources::{Format, PresentBackend, ResourceBackend, TextureDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::msg_send;
use objc2_foundation::NSSize as CGSize;
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use std::cell::RefCell;
use std::collections::HashMap;

/// Draws a texture over the whole target. Compiled by Metal's own compiler at run time (this is
/// the one shader of the backend that is not converted DXIL).
const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; float2 uv; };
vertex Out present_vertex(uint i [[vertex_id]]) {
    float2 p = float2((i << 1) & 2, i & 2);
    Out o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}
fragment float4 present_fragment(Out in [[stage_in]], texture2d<float> source [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return source.sample(s, in.uv);
}
"#;

type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

/// MetalFX's spatial upscaler: the title's 720p frame into the window's 1080p one, edge-aware
/// instead of a plain linear sample. Bound by hand (`objc2` has no MetalFX crate): the framework is
/// opened once, and anything missing (an older macOS, no class, a refused descriptor) leaves
/// [`spatial`] `None`, and the linear blit draws the frame as before.
mod fx {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, AnyObject};
    use objc2::msg_send;

    /// The framework, opened once; false when it is not there.
    fn loaded() -> bool {
        static LOADED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *LOADED.get_or_init(|| {
            let name = c"/System/Library/Frameworks/MetalFX.framework/MetalFX";
            // SAFETY: dlopen of a system framework; the handle is kept for the process's life.
            let h = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            h.is_null() || AnyClass::get(c"MTLFXSpatialScalerDescriptor").is_none()
        }) == false
    }

    /// A spatial scaler from `in_w` x `in_h` to `out_w` x `out_h` of `format`, if MetalFX has one.
    /// Small (it is a compute pipeline and its sizes), made once per pair by the caller.
    pub(super) struct Spatial {
        scaler: Retained<AnyObject>,
    }

    // SAFETY: the scaler is used only from the present path, on the queue's thread.
    unsafe impl Send for Spatial {}

    pub(super) fn spatial(
        device: &ProtocolObject<dyn MTLDevice>,
        in_w: usize,
        in_h: usize,
        out_w: usize,
        out_h: usize,
        format: MTLPixelFormat,
    ) -> Option<Spatial> {
        if !loaded() {
            return None;
        }
        let cls = AnyClass::get(c"MTLFXSpatialScalerDescriptor")?;
        // SAFETY: `new` on a class MetalFX provides; the +1 reference is taken by `Retained`.
        let descriptor: *mut AnyObject = unsafe { msg_send![cls, new] };
        let descriptor = unsafe { Retained::from_raw(descriptor)? };
        // SAFETY: the descriptor's documented setters (sizes in pixels, the input's pixel format,
        // perceptual colour processing); a selector MetalFX does not have would raise, so only
        // these are sent.
        unsafe {
            let _: () = msg_send![&*descriptor, setInputWidth: in_w];
            let _: () = msg_send![&*descriptor, setInputHeight: in_h];
            let _: () = msg_send![&*descriptor, setOutputWidth: out_w];
            let _: () = msg_send![&*descriptor, setOutputHeight: out_h];
            let _: () = msg_send![&*descriptor, setColorTextureFormat: format.0];
            let _: () = msg_send![&*descriptor, setColorProcessingMode: 0usize]; // Perceptual
        }
        // SAFETY: MetalFX's constructor; null when the device or the sizes are refused, and the
        // +1 reference is taken by `Retained`.
        let scaler: *mut AnyObject = unsafe { msg_send![&*descriptor, newSpatialScalerWithDevice: device] };
        let scaler = unsafe { Retained::from_raw(scaler)? };
        Some(Spatial { scaler })
    }

    impl Spatial {
        /// Encodes the upscale into `cb`: `input` into `output` (a `ShaderWrite` texture of the
        /// scaler's output size).
        pub(super) fn encode(
            &self,
            cb: &ProtocolObject<dyn MTLCommandBuffer>,
            input: &ProtocolObject<dyn MTLTexture>,
            output: &ProtocolObject<dyn MTLTexture>,
        ) {
            // SAFETY: the scaler's documented properties and encoder call; `output` is the size
            // and format the descriptor asked for.
            unsafe {
                let _: () = msg_send![&*self.scaler, setColorTexture: input];
                let _: () = msg_send![&*self.scaler, setOutputTexture: output];
                let _: () = msg_send![&*self.scaler, encodeToCommandBuffer: cb];
            }
        }
    }
}

/// Whether the window showing the surfaces is minimised or wholly covered (its event loop says,
/// [`set_window_hidden`]). macOS then hands out few drawables or none, and a present that waited
/// for one held the title's presents back until the title stopped (#333). A hidden window's
/// presents count as shown with nothing drawn, as a headless run's do.
static HIDDEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Tells the presents whether their window can be seen (see [`HIDDEN`]).
pub fn set_window_hidden(hidden: bool) {
    if HIDDEN.swap(hidden, std::sync::atomic::Ordering::Relaxed) != hidden {
        tracing::info!(hidden, "present: the window's visibility changed");
    }
}

// CAMetalLayer supports drawable acquisition on a background rendering thread. Only the
// worker calls nextDrawable; ownership of each retained drawable crosses the channel once.
struct AcquisitionLayer(Retained<CAMetalLayer>);
// SAFETY: the retained layer is used only for thread-safe drawable acquisition on the worker.
unsafe impl Send for AcquisitionLayer {}
struct Drawable(Retained<ProtocolObject<dyn CAMetalDrawable>>);
// SAFETY: Metal drawables may cross rendering threads; the channel transfers ownership.
unsafe impl Send for Drawable {}

impl AcquisitionLayer {
    fn next(&self) -> Option<Drawable> {
        self.0.nextDrawable().map(Drawable)
    }
}

/// A `CAMetalLayer` presented to.
pub struct Surface {
    acquisition: std::sync::mpsc::Sender<()>,
    drawables: std::sync::mpsc::Receiver<Option<Drawable>>,
    acquiring: std::cell::Cell<bool>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// The copy pipeline for each drawable format seen, and whether it blends.
    pipelines: RefCell<HashMap<(usize, bool), Pipeline>>,
    /// The texture of the last drawable presented, readable when the layer is not frame-buffer
    /// only (the tests read their pixels from it).
    last: RefCell<Option<Retained<ProtocolObject<dyn MTLTexture>>>>,
    /// Frames presented.
    frames: std::cell::Cell<u64>,
    /// `VICTORY_COUNT_GREEN` state: pixel probes of frames the GPU has not finished, and pacing.
    green: RefCell<GreenCount>,
    /// MetalFX spatial upscalers by `(in_w, in_h, out_w, out_h, format)`: what turned the title's
    /// 720p frame into the window's 1080p one, made once per pair (`fx::spatial`). Empty when
    /// MetalFX is not there or refused one, and the linear blit does it then.
    scalers: RefCell<HashMap<(usize, usize, usize, usize, usize), Option<std::rc::Rc<fx::Spatial>>>>,
    /// The `outputTexture` a scaler writes, by `(w, h, format)`: MetalFX writes a texture with
    /// `ShaderWrite`, which the drawable has not, so the blit copies this into it.
    scaled: RefCell<HashMap<(usize, usize, usize), Retained<ProtocolObject<dyn MTLTexture>>>>,
}

/// `VICTORY_COUNT_GREEN=1`: each presented frame has nine pixels (a 3 by 3 grid at a quarter, a
/// half and three quarters of its size) copied out after it is drawn, and is counted green when
/// most are pure green (what a YCbCr-to-RGB pass makes of zero planes: a movie frame sampled
/// blank). Once a second stderr gets a `present pacing` line: presents, the longest gap between
/// two, and the green frames in that second and so far. Costs one small blit per frame.
fn counting_green() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COUNT_GREEN").is_some())
}

type Probe = (Retained<ProtocolObject<dyn MTLCommandBuffer>>, Retained<ProtocolObject<dyn MTLBuffer>>, bool);

#[derive(Default)]
struct GreenCount {
    probes: std::collections::VecDeque<Probe>,
    start: Option<std::time::Instant>,
    second: u64,
    last: Option<std::time::Instant>,
    presents: u32,
    longest: std::time::Duration,
    green: u32,
    partly: u32,
    total: u64,
}

/// Whether nine probed pixels (`r, g, b, a` or `b, g, r, a`) are a green frame (five or more pure
/// green), and how many of them are.
pub(crate) fn green_pixels(px: &[u8], bgra: bool) -> (bool, usize) {
    let n = px
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| {
            let (r, g, b) = if bgra { (p[2], p[1], p[0]) } else { (p[0], p[1], p[2]) };
            g >= 64 && r < 32 && b < 32
        })
        .count();
    (n >= 5, n)
}

impl GreenCount {
    /// A frame was presented: pacing, and the probes the GPU finished.
    fn presented(&mut self) {
        let now = std::time::Instant::now();
        let start = *self.start.get_or_insert(now);
        if let Some(last) = self.last.replace(now) {
            self.longest = self.longest.max(now - last);
        }
        self.presents += 1;
        while let Some((cb, _, _)) = self.probes.front() {
            let status = cb.status();
            if status != MTLCommandBufferStatus::Completed && status != MTLCommandBufferStatus::Error {
                break;
            }
            let Some((_, buf, bgra)) = self.probes.pop_front() else { break };
            if status == MTLCommandBufferStatus::Completed {
                // SAFETY: the buffer holds the 36 bytes the finished blit wrote.
                let px = unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<u8>(), 36) };
                match green_pixels(px, bgra) {
                    (true, _) => self.green += 1,
                    (false, n) if n > 0 => self.partly += 1,
                    _ => {}
                }
            }
        }
        let second = (now - start).as_secs();
        if second != self.second {
            self.total += u64::from(self.green);
            eprintln!(
                "present pacing t={} presents={} longest_ms={:.1} green={} partly_green={} green_total={}",
                self.second, self.presents, self.longest.as_secs_f64() * 1e3, self.green, self.partly, self.total
            );
            *self = GreenCount { probes: std::mem::take(&mut self.probes), start: self.start, second, last: self.last, total: self.total, ..Default::default() };
        }
    }
}

impl Surface {
    pub fn frames(&self) -> u64 {
        self.frames.get()
    }

    /// The pixels of the last presented drawable (width, height, BGRA rows), read back from its
    /// texture: complete once the GPU has drawn the frame (a fence signalled after the present).
    pub fn read_last(&self) -> Option<(usize, usize, Vec<u8>)> {
        let t = self.last.borrow().clone()?;
        let (w, h) = (t.width(), t.height());
        let mut out = vec![0u8; w * h * 4];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w,
                height: h,
                depth: 1,
            },
        };
        // SAFETY: `out` holds `h` rows of `w * 4` bytes, the region's size.
        unsafe {
            t.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(out.as_mut_ptr().cast())?,
                w * 4,
                region,
                0,
            )
        };
        Some((w, h, out))
    }

    /// The last presented drawable's texture, with its size.
    pub fn last_texture(&self) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        self.last.borrow().clone()
    }

    /// The last presented drawable's size, without reading its pixels back: [`Surface::read_last`]
    /// waits for the GPU and must not be used where a command buffer may never complete.
    pub fn last_size(&self) -> Option<(usize, usize)> {
        self.last.borrow().as_ref().map(|t| (t.width(), t.height()))
    }
}

/// A layer made by [`Metal::create_layer`]: its address is what [`Metal::create_surface`] takes,
/// and what an app puts into a view (`NSView.layer`).
pub struct Layer(pub Retained<CAMetalLayer>);

impl Layer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        Retained::as_ptr(&self.0).cast_mut().cast()
    }
}

impl Metal {
    /// A layer drawing to this device: `width` by `height` drawables in BGRA, readable (not
    /// frame-buffer only), vsync off.
    pub fn create_layer(&self, width: usize, height: usize) -> Layer {
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&self.dev));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setFramebufferOnly(false);
        layer.setDrawableSize(CGSize {
            width: width as f64,
            height: height as f64,
        });
        // SAFETY: a plain BOOL property of CAMetalLayer.
        let _: () = unsafe { msg_send![&*layer, setDisplaySyncEnabled: false] };
        Layer(layer)
    }
}

fn pipeline(
    m: &Metal,
    surface: &Surface,
    format: MTLPixelFormat,
    blend: bool,
) -> Result<Pipeline, String> {
    if let Some(p) = surface.pipelines.borrow().get(&(format.0, blend)) {
        return Ok(p.clone());
    }
    let d = MTLRenderPipelineDescriptor::new();
    let name = |s: &str| NSString::from_str(s);
    d.setVertexFunction(surface.library.newFunctionWithName(&name("present_vertex")).as_deref());
    d.setFragmentFunction(
        surface
            .library
            .newFunctionWithName(&name("present_fragment"))
            .as_deref(),
    );
    // SAFETY: colour attachment 0 always exists.
    let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
    a.setPixelFormat(format);
    if blend {
        // Straight alpha: an overlay plane over the planes under it.
        a.setBlendingEnabled(true);
        a.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        a.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        a.setSourceAlphaBlendFactor(MTLBlendFactor::One);
        a.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }
    let p = m
        .dev
        .newRenderPipelineStateWithDescriptor_error(&d)
        .map_err(|e| e.localizedDescription().to_string())?;
    surface.pipelines.borrow_mut().insert((format.0, blend), p.clone());
    Ok(p)
}

impl PresentBackend for Metal {
    type Surface = Surface;

    fn create_surface(&self, layer: *mut std::ffi::c_void) -> Result<Surface, String> {
        // SAFETY: the caller passes a live CAMetalLayer (retained here).
        let layer = unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }
            .ok_or("a null CAMetalLayer")?;
        if layer.device().is_none() {
            layer.setDevice(Some(&self.dev));
        }
        // Drawn into, so the layer cannot be frame-buffer only for reading back; presenting
        // works either way.
        let library = self
            .dev
            .newLibraryWithSource_options_error(&NSString::from_str(SHADER), None)
            .map_err(|e| e.localizedDescription().to_string())?;
        let (acquisition, requests) = std::sync::mpsc::channel();
        let (results, drawables) = std::sync::mpsc::sync_channel(1);
        let layer = AcquisitionLayer(layer);
        std::thread::Builder::new().name("victory drawables".into()).spawn(move || {
            while requests.recv().is_ok() {
                let drawable = objc2::rc::autoreleasepool(|_| layer.next());
                if results.send(drawable).is_err() {
                    break;
                }
            }
        }).map_err(|e| e.to_string())?;
        Ok(Surface {
            acquisition,
            drawables,
            acquiring: std::cell::Cell::new(false),
            library,
            pipelines: RefCell::default(),
            last: RefCell::default(),
            frames: Default::default(),
            green: Default::default(),
            scalers: RefCell::default(),
            scaled: RefCell::default(),
        })
    }

    fn present(
        &self,
        queue: &Queue,
        surface: &Surface,
        sources: &[&Texture],
    ) -> Result<(), String> {
        // Standalone backend callers may block, but an unavailable drawable must not keep
        // them here forever. The title pump uses try_present instead and never sleeps.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if self.try_present(queue, surface, sources)? {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err("timed out waiting for a drawable".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn try_present(&self, queue: &Queue, surface: &Surface, sources: &[&Texture]) -> Result<bool, String> {
        let Some(first) = sources.first() else { return Err("nothing to present".into()) };
        // A hidden window shows nothing; a drawable asked for already is still taken below.
        if HIDDEN.load(std::sync::atomic::Ordering::Relaxed) && !surface.acquiring.get() {
            surface.frames.set(surface.frames.get() + 1);
            crate::pace::presented();
            if counting_green() {
                surface.green.borrow_mut().presented();
            }
            return Ok(true);
        }
        if !surface.acquiring.replace(true) {
            surface.acquisition.send(()).map_err(|_| "drawable worker stopped")?;
        }
        let drawable = match surface.drawables.try_recv() {
            Ok(drawable) => {
                surface.acquiring.set(false);
                drawable.ok_or("no drawable")?.0
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(false),
            Err(_) => return Err("drawable worker stopped".into()),
        };
        objc2::rc::autoreleasepool(|_| {
            let target = drawable.texture();
            let (in_w, in_h) = first.texel_size();
            let (out_w, out_h) = (target.width(), target.height());
            // What is shown and into what: the title's present plane against the layer's drawable.
            // One line per size pair, so a run says whether the title presents its 720p frames or
            // already-upscaled 1080p ones.
            {
                static SIZES: std::sync::Mutex<Vec<(usize, usize, usize, usize, bool)>> =
                    std::sync::Mutex::new(Vec::new());
                let pair = (in_w, in_h, out_w, out_h, sources.len() > 1);
                let mut seen = SIZES.lock().unwrap_or_else(|e| e.into_inner());
                if seen.len() < 8 && !seen.contains(&pair) {
                    seen.push(pair);
                    // Every plane's size, bottom first: a 1080p picture under a 3840x2160
                    // interface is the title's own choice of plane size, and says whether the
                    // mode it was told reaches all of them.
                    let sizes: Vec<(usize, usize)> =
                        sources.iter().map(|t| t.texel_size()).collect();
                    tracing::info!(
                        w = pair.0,
                        h = pair.1,
                        out_w = pair.2,
                        out_h = pair.3,
                        planes = sources.len(),
                        plane_sizes = ?sizes,
                        "present: the title's plane into the window's drawable"
                    );
                }
            }
            // An sRGB source is shown as the bytes it holds when the layer is not sRGB (the title's
            // swap chain holds display values).
            let src = first.view_without_srgb(self);
            // MetalFX spatial upscale, when the title presents at a smaller size than the window's
            // drawable (Series S: 1280x720 into 1920x1080) and MetalFX has a scaler for the pair.
            // One plane only: a second plane is blended by the linear path below. The drawable has
            // no `ShaderWrite`, so the scaler writes an intermediate texture that is then blitted
            // into it.
            let fx_format = matches!(
                first.texel_format(),
                MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::BGRA8Unorm
            );
            if sources.len() == 1
                && metal_fx()
                && fx_format
                && (in_w, in_h) != (out_w, out_h)
                && target.pixelFormat() == MTLPixelFormat::BGRA8Unorm
            {
                if let Some((scaler, scaled)) = spatial_scaler(self, surface, &src, in_w, in_h, out_w, out_h) {
                    let cb = queue.0.commandBuffer().ok_or("no command buffer")?;
                    scaler.encode(&cb, &src, &scaled);
                    let blit = cb.blitCommandEncoder().ok_or("no blit encoder")?;
                    // SAFETY: full-size copy between two textures of the same size and format.
                    unsafe {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            &scaled,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                            MTLSize { width: out_w, height: out_h, depth: 1 },
                            &target,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                        )
                    };
                    blit.endEncoding();
                    note_scaler(in_w, in_h, out_w, out_h, "MetalFX spatial");
                    let probe = if counting_green() { probe_pixels(self, &cb, &src) } else { None };
                    cb.presentDrawable(ProtocolObject::from_ref(&*drawable));
                    cb.commit();
                    if counting_green() {
                        let mut g = surface.green.borrow_mut();
                        if let Some((buf, bgra)) = probe {
                            g.probes.push_back((cb.clone(), buf, bgra));
                        }
                        g.presented();
                    }
                    *surface.last.borrow_mut() = Some(target);
                    surface.frames.set(surface.frames.get() + 1);
                    crate::pace::presented();
                    return Ok(true);
                }
            }
            note_scaler(in_w, in_h, out_w, out_h, "a linear blit");
            let pass = MTLRenderPassDescriptor::new();
            // SAFETY: colour attachment 0 always exists.
            let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(&target));
            a.setLoadAction(MTLLoadAction::DontCare);
            a.setStoreAction(MTLStoreAction::Store);
            let cb = queue.0.commandBuffer().ok_or("no command buffer")?;
            let e = cb
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or("no render encoder")?;
            for (i, source) in sources.iter().enumerate() {
                let view = if i == 0 { src.clone() } else { source.view_without_srgb(self) };
                let state = pipeline(self, surface, target.pixelFormat(), i > 0)?;
                e.setRenderPipelineState(&state);
                // SAFETY: slot 0 of the fragment stage takes the source texture.
                unsafe { e.setFragmentTexture_atIndex(Some(&view), 0) };
                // SAFETY: three vertices, no buffers.
                unsafe {
                    e.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3)
                };
            }
            e.endEncoding();
            let probe = if counting_green() { probe_pixels(self, &cb, &src) } else { None };
            cb.presentDrawable(ProtocolObject::from_ref(&*drawable));
            cb.commit();
            if counting_green() {
                let mut g = surface.green.borrow_mut();
                if let Some((buf, bgra)) = probe {
                    g.probes.push_back((cb.clone(), buf, bgra));
                }
                g.presented();
            }
            *surface.last.borrow_mut() = Some(target);
            surface.frames.set(surface.frames.get() + 1);
            crate::pace::presented();
            Ok(true)
        })
    }
}

/// Whether the present path may use MetalFX: **opt-in** (`VICTORY_METALFX=1`) until a run has
/// shown it on the title's real 720p -> 1080p present. It is off by default because this crate is
/// compiled without `cfg(test)` when another crate's tests link it, so a test presentation would
/// otherwise reach MetalFX (a 16x16 test frame into a 32x32 test layer), where the scaler is not
/// something to depend on.
fn metal_fx() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_METALFX").is_some_and(|v| v == "1"))
}

/// The MetalFX scaler for this size pair and its `ShaderWrite` output texture, made on first use.
/// `None` when MetalFX has none (an older macOS, a refused descriptor): the linear blit draws the
/// frame then.
#[allow(clippy::type_complexity)]
fn spatial_scaler(
    m: &Metal,
    surface: &Surface,
    src: &ProtocolObject<dyn MTLTexture>,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Option<(std::rc::Rc<fx::Spatial>, Retained<ProtocolObject<dyn MTLTexture>>)> {
    let key = (in_w, in_h, out_w, out_h, src.pixelFormat().0);
    let scaler = {
        let mut scalers = surface.scalers.borrow_mut();
        let made = scalers.entry(key).or_insert_with(|| {
            fx::spatial(&m.dev, in_w, in_h, out_w, out_h, src.pixelFormat()).map(std::rc::Rc::new)
        });
        made.clone()?
    };
    let key = (out_w, out_h, MTLPixelFormat::BGRA8Unorm.0);
    let scaled = {
        let mut scaled = surface.scaled.borrow_mut();
        match scaled.get(&key) {
            Some(t) => t.clone(),
            None => {
                // MetalFX writes a `ShaderWrite` texture; the layer's drawables have no such
                // usage, and this one is the size and format (BGRA8) the scaler produces.
                let desc = TextureDesc {
                    render_target: true,
                    unordered_access: true,
                    ..TextureDesc::d2(Format::Bgra8Unorm, out_w, out_h)
                };
                let tex = m.create_texture(&desc).ok()?;
                let raw = tex.raw();
                scaled.insert(key, raw.clone());
                raw
            }
        }
    };
    Some((scaler, scaled))
}

/// Says once per size pair which path shows the frame: MetalFX's spatial upscale or the linear
/// blit.
fn note_scaler(in_w: usize, in_h: usize, out_w: usize, out_h: usize, by: &'static str) {
    static SAID: std::sync::Mutex<Vec<(usize, usize, usize, usize)>> = std::sync::Mutex::new(Vec::new());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.len() < 8 && !said.contains(&(in_w, in_h, out_w, out_h)) {
        said.push((in_w, in_h, out_w, out_h));
        tracing::info!(w = in_w, h = in_h, out_w, out_h, by, "present: the upscale path");
    }
}

/// Copies the nine probe pixels of `src` (8-bit RGBA or BGRA) into a new shared buffer, in `cb`.
fn probe_pixels(
    m: &Metal,
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, bool)> {
    let bgra = match src.pixelFormat() {
        MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::RGBA8Unorm_sRGB => false,
        MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB => true,
        _ => return None,
    };
    let buf = m.dev.newBufferWithLength_options(36, MTLResourceOptions::StorageModeShared)?;
    let blit = cb.blitCommandEncoder()?;
    let (w, h) = (src.width(), src.height());
    for (i, (fy, fx)) in [1, 2, 3].iter().flat_map(|y| [1, 2, 3].map(|x| (*y, x))).enumerate() {
        // SAFETY: a 1 by 1 region inside the texture into 4 bytes inside the 36-byte buffer.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                src, 0, 0,
                MTLOrigin { x: w * fx / 4, y: h * fy / 4, z: 0 },
                MTLSize { width: 1, height: 1, depth: 1 },
                &buf, i * 4, 4, 4,
            )
        };
    }
    blit.endEncoding();
    Some((buf, bgra))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ComputeBackend;
    use crate::resources::{Format, ResourceBackend, TextureDesc};

    #[test]
    fn zero_planes_through_ycbcr_count_as_green() {
        let green = [10u8, 135, 0, 255];
        let grey = [90u8, 90, 90, 255];
        let frame = |n: usize| -> Vec<u8> { (0..9).flat_map(|i| if i < n { green } else { grey }).collect() };
        assert_eq!(green_pixels(&frame(9), false), (true, 9));
        assert_eq!(green_pixels(&frame(4), false), (false, 4));
        assert_eq!(green_pixels(&frame(0), false), (false, 0));
        // BGRA order.
        let bgra: Vec<u8> = (0..9).flat_map(|_| [0u8, 135, 10, 255]).collect();
        assert_eq!(green_pixels(&bgra, true), (true, 9));
    }

    /// Two planes: the first (a smaller picture) scaled to the drawable, the second blended over
    /// it by its alpha: clear where alpha is 0, its own colour where alpha is 1.
    #[test]
    fn a_second_plane_is_blended_over_the_first_by_its_alpha() {
        let _turn = PRESENTING.lock().unwrap_or_else(|e| e.into_inner());
        use crate::backend::Submission;
        let Ok(metal) = Metal::new() else { return };
        let layer = metal.create_layer(4, 4);
        let surface = metal.create_surface(layer.as_ptr()).unwrap();
        let queue = metal.create_queue().unwrap();
        let picture = metal.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        metal.upload_texture(&picture, 0, 0, &[255, 0, 0, 255].repeat(4), 8, 16);
        // The overlay: transparent on the left half, opaque blue on the right half.
        let overlay = metal.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 4, 4)).unwrap();
        let row: Vec<u8> = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 255, 255], [0, 0, 255, 255]].concat();
        metal.upload_texture(&overlay, 0, 0, &row.repeat(4), 16, 64);
        metal.present(&queue, &surface, &[&picture, &overlay]).unwrap();
        let fence = metal.create_fence(0).unwrap();
        metal.submit(&queue, &Submission { waits: &[], dispatches: &[], signals: &[(&fence, 1)] }).unwrap();
        assert!(metal.wait(&fence, 1, 5000));
        let (w, h, bgra) = surface.read_last().unwrap();
        assert_eq!((w, h), (4, 4));
        for y in 0..4 {
            let px = |x: usize| &bgra[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
            assert_eq!(px(0), [0, 0, 255, 255], "red picture at x 0, row {y}");
            assert_eq!(px(3), [255, 0, 0, 255], "blue overlay at x 3, row {y}");
        }
    }

    /// [`HIDDEN`] is the process's: the tests that present take turns.
    static PRESENTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_hidden_window_presents_without_a_drawable() {
        let _turn = PRESENTING.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(metal) = Metal::new() else { return };
        let layer = metal.create_layer(4, 4);
        let mut surface = metal.create_surface(layer.as_ptr()).unwrap();
        let queue = metal.create_queue().unwrap();
        let source = metal.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 4, 4)).unwrap();
        // A worker that never hands out a drawable, as macOS does for a minimised window.
        let (requests, _requests_rx) = std::sync::mpsc::channel();
        let (_results, drawables) = std::sync::mpsc::channel();
        surface.acquisition = requests;
        surface.drawables = drawables;
        set_window_hidden(true);
        let shown = (0..3).map(|_| metal.try_present(&queue, &surface, &[&source])).collect::<Vec<_>>();
        set_window_hidden(false);
        assert_eq!(shown, vec![Ok(true), Ok(true), Ok(true)]);
        assert_eq!(surface.frames(), 3);
        // Seen again: the present waits for a drawable as before.
        assert_eq!(metal.try_present(&queue, &surface, &[&source]), Ok(false));
        assert_eq!(surface.frames(), 3);
    }

    #[test]
    fn blocking_present_times_out_when_acquisition_never_returns() {
        let _turn = PRESENTING.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(metal) = Metal::new() else { return };
        let layer = metal.create_layer(4, 4);
        let mut surface = metal.create_surface(layer.as_ptr()).unwrap();
        let queue = metal.create_queue().unwrap();
        let source = metal.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 4, 4)).unwrap();
        // A connected acquisition worker that never produces a drawable. Disconnect after
        // four seconds as a test watchdog, so removing the deadline fails rather than hangs.
        let (requests, _requests_rx) = std::sync::mpsc::channel();
        let (results, drawables) = std::sync::mpsc::channel();
        surface.acquisition = requests;
        surface.drawables = drawables;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(4));
            drop(results);
        });
        let started = std::time::Instant::now();
        assert_eq!(metal.present(&queue, &surface, &[&source]).unwrap_err(), "timed out waiting for a drawable");
        assert!(started.elapsed() >= std::time::Duration::from_secs(2));
        assert_eq!(surface.frames(), 0);
    }
}
