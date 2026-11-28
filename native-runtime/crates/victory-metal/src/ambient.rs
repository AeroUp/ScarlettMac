//! The ambient stand-in's kernel (`VICTORY_AMBIENT=1`): after the title's indirect diffuse
//! resolve, replace its image with one constant shade (R9G9B9E5 words), as ScarlettMac's
//! `victory-gpu/src/ambient.rs` does with HLSL. The kernel is Metal's own compiler's work, like
//! the present pass (`present.rs`), and the pipeline is made once, like the indirect gate
//! (`gfx.rs`).
//!
//! The image is bound through a 32-bit integer view (`Texture::ambient_uint`): the pack below
//! writes R9G9B9E5 exactly as the console's resolve left it, so the title's own lighting reads
//! the same texels it would have.
//!
//! shortcut: one shade everywhere, from everywhere; gone once rays run. The sky-following level,
//! the depth shading and the reflection hold of the ScarlettMac line are not here yet.

use crate::metal::Metal;
use crate::resources::AmbientPass;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

/// Compiled by Metal's own compiler at run time, like the present pass.
const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Ambient {
    float level;
    float tint_r;
    float tint_g;
    float tint_b;
    uint width;
    uint height;
};

// R9G9B9E5: the console's shared-exponent RGB, as the resolve's image holds it.
uint pack_rgb9e5(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp((int)floor(log2(max(most, 1e-9))) + 16, 0, 31);
    float scale = exp2((float)(24 - e));
    if (most * scale >= 511.5) {
        e = min(e + 1, 31);
        scale = exp2((float)(24 - e));
    }
    uint3 m = min(uint3(c * scale + 0.5), uint3(511u));
    return m.r | (m.g << 9) | (m.b << 18) | ((uint)e << 27);
}

kernel void ambient_fill(texture2d<uint, access::write> light [[texture(0)]],
                         constant Ambient &p [[buffer(0)]],
                         uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.width || at.y >= p.height) return;
    float3 c = float3(p.tint_r, p.tint_g, p.tint_b) * p.level;
    light.write(pack_rgb9e5(c), at);
}
"#;

/// Threads a group runs: the resolve's own `numthreads(8, 8, 1)`.
const GROUP: usize = 8;

impl Metal {
    /// The ambient kernel, made at first use; `None`'s reason if Metal refuses it (said once by
    /// the caller).
    fn ambient_pipeline(
        &self,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        self.ambient
            .get_or_init(|| {
                let lib = self
                    .dev
                    .newLibraryWithSource_options_error(&NSString::from_str(SHADER), None)
                    .map_err(|e| e.localizedDescription().to_string())?;
                let f = lib
                    .newFunctionWithName(&NSString::from_str("ambient_fill"))
                    .ok_or("no ambient_fill function")?;
                self.dev
                    .newComputePipelineStateWithFunction_error(&f)
                    .map_err(|e| e.localizedDescription().to_string())
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Encodes the ambient pass into `cb`, after whatever the command buffer has run: one
    /// dispatch over `light`'s texels.
    pub(crate) fn encode_ambient(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        pass: &AmbientPass<'_, Self>,
        order: Option<(&ProtocolObject<dyn MTLFence>, &std::cell::Cell<bool>)>,
    ) -> Result<(), String> {
        let light = pass.light.ambient_uint()?;
        let pipeline = self.ambient_pipeline()?.clone();
        let (width, height) = (light.width(), light.height());
        if width == 0 || height == 0 {
            return Ok(());
        }
        let e = cb.computeCommandEncoder().ok_or("no compute encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        if crate::frametime::passes() {
            crate::frametime::pass(format!("ambient {width}x{height}"));
        }
        e.setComputePipelineState(&pipeline);
        // SAFETY: slot 0 takes the light image and buffer 0 the six constant words (24 bytes);
        // Metal copies the bytes before the call returns.
        unsafe {
            e.setTexture_atIndex(Some(&light), 0);
            e.setBytes_length_atIndex(
                NonNull::new(pass.constants.as_ptr().cast_mut().cast()).unwrap(),
                std::mem::size_of_val(&pass.constants),
                0,
            );
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: width.div_ceil(GROUP),
                height: height.div_ceil(GROUP),
                depth: 1,
            },
            MTLSize {
                width: GROUP,
                height: GROUP,
                depth: 1,
            },
        );
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        e.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's compiler takes the kernel and the host takes the pipeline.
    #[test]
    fn the_ambient_kernel_builds() {
        let Ok(metal) = Metal::new() else { return };
        metal.ambient_pipeline().expect("the ambient pipeline");
    }

    /// The R9G9B9E5 image the title resolves into can be written as 32-bit integers.
    #[test]
    fn the_light_is_viewed_as_32_bit_integers() {
        use crate::resources::{Format, ResourceBackend, TextureDesc};
        let Ok(metal) = Metal::new() else { return };
        let desc = TextureDesc {
            unordered_access: true,
            ..TextureDesc::d2(Format::Rgb9e5Float, 4, 4)
        };
        let light = metal.create_texture(&desc).expect("an R9G9B9E5 texture");
        let view = light.ambient_uint().expect("a 32-bit integer view");
        assert_eq!(view.pixelFormat(), MTLPixelFormat::R32Uint);
    }
}
