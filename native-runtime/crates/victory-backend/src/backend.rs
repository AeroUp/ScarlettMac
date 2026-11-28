//! The seam victory-gpu's `host.rs` / `queue.rs` / `fence.rs` would call instead of D3D12, first
//! slice: queues, fences, compute pipelines from DXIL, compute dispatch over title memory.
//!
//! Platform neutral on purpose (no Metal types), so the D3D12 host can implement it too and the
//! title-facing objects (`queue.rs`, `fence.rs`, `hold.rs`) stay as they are. One submission is one
//! `ExecuteCommandLists` worth of work: waits, then dispatches in order, then signals. Nothing in it
//! waits for the GPU; completion is observed through fences.

/// How a dispatch's top-level argument buffer is laid out: root constants inline, root SRV/UAV/CBV
/// as 64-bit GPU addresses, in root-parameter order (what `IRCompiler` makes of a root signature).
#[derive(Default, Clone, Debug)]
pub struct RootArgs(pub Vec<u8>);

impl RootArgs {
    pub fn constants(mut self, words: &[u32]) -> Self {
        words.iter().for_each(|w| self.0.extend(w.to_le_bytes()));
        self
    }

    /// A root view (SRV, UAV or CBV): a GPU address, 8-byte aligned.
    pub fn address(mut self, gpu: u64) -> Self {
        self.0.resize(self.0.len().next_multiple_of(8), 0);
        self.0.extend(gpu.to_le_bytes());
        self
    }
}

/// One root parameter of a [`RootLayout`]: what a translated shader declares at a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RootParam {
    /// `dwords` root constants at `b<register>` of `space`: inline in the argument buffer.
    Constants { register: u32, space: u32, dwords: u32 },
    /// A root CBV (`b`), SRV (`t`) or UAV (`u`): a 64-bit GPU address in the argument buffer.
    Cbv { register: u32, space: u32 },
    Srv { register: u32, space: u32 },
    Uav { register: u32, space: u32 },
}

/// The root signature of translated shaders, which carry none: parameters in the order
/// [`RootArgs`] is filled in, descriptor heaps indexed directly. Without one a backend takes the
/// signature the shader itself embeds (DXIL compiled by dxc), or its own default layout.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RootLayout {
    pub params: Vec<RootParam>,
    /// The shader reads the memory of a raw buffer view out of range as 0 and drops its writes
    /// (`D3D12` hardware does; Metal needs the converter told).
    pub bounds_checked: bool,
    /// Static samplers, `s<n>` at position n in space 0: the words of a
    /// `D3D12_STATIC_SAMPLER_DESC` up to `MaxLOD` (filter, address U, V, W, LOD bias as f32 bits,
    /// anisotropy, comparison, border, minimum and maximum LOD as f32 bits).
    pub static_samplers: Vec<[u32; 10]>,
}

/// One `Dispatch` call: the pipeline, its root arguments, the memory it touches (for residency),
/// and the group counts and sizes.
pub struct Dispatch<'a, B: ComputeBackend + ?Sized> {
    pub pipeline: &'a B::Pipeline,
    pub root: RootArgs,
    /// Every title buffer the shader reaches through `root` (heap-referenced resources are made
    /// resident by the heaps themselves).
    pub resources: Vec<&'a B::Buffer>,
    pub heaps: Heaps<'a, B>,
    pub groups: [usize; 3],
    pub threads_per_group: [usize; 3],
    /// `ExecuteIndirect`: the group counts are read on the GPU from this title address (three
    /// words) instead of `groups`. A record that asks for more than 2^22 groups in all
    /// dispatches nothing.
    pub indirect: Option<(&'a B::Buffer, *const u8)>,
    /// `ExecuteIndirect` with a count buffer: the count's title address and this record's index.
    /// The record runs only if the index is below the count, which the GPU reads when it runs.
    pub count: Option<(&'a B::Buffer, *const u8, u32)>,
}

/// The descriptor heaps a draw or dispatch binds (`SetDescriptorHeaps`): slot 0 and slot 1.
pub struct Heaps<'a, B: ComputeBackend + ?Sized> {
    pub resource: Option<&'a B::Heap>,
    pub sampler: Option<&'a B::Heap>,
}

impl<B: ComputeBackend + ?Sized> Default for Heaps<'_, B> {
    fn default() -> Self {
        Heaps {
            resource: None,
            sampler: None,
        }
    }
}

/// Work for one queue: wait for fence values, run dispatches in order, signal fence values.
pub struct Submission<'a, B: ComputeBackend + ?Sized> {
    pub waits: &'a [(&'a B::Fence, u64)],
    pub dispatches: &'a [Dispatch<'a, B>],
    pub signals: &'a [(&'a B::Fence, u64)],
}

/// The most thread groups one dispatch or mesh dispatch may run: the product of its three group
/// counts. The title's own records ask for far less (9,192 groups at most seen on the native host;
/// other ports report up to 115,200 x 1 x 1), so this only catches a count read from garbage. A
/// record that asks for more is clamped (never run as asked): 510 x 1,182,027,346 x 1,314,076,063
/// groups was what an unfinished record in title memory held, and the command buffer running it
/// never ended.
pub const MAX_DISPATCH_GROUPS: u64 = 1 << 18;

/// The most vertices or indices one draw may run in all its instances, together. The title's own
/// draws are far below it; a record read from garbage asks for billions and hangs the GPU.
pub const MAX_DRAW_COUNTS: u64 = 1 << 22;

/// The most instances one draw may run. A full-screen draw with millions of instances is as
/// unbounded as billions of vertices; the title's instanced draws are far below this.
pub const MAX_DRAW_INSTANCES: u64 = 1 << 16;

/// The most groups one dimension of a dispatch may name (D3D12's `Dispatch` limit).
pub const MAX_GROUPS_PER_DIMENSION: u64 = 65_535;

/// `groups` clamped so no dimension is past [`MAX_GROUPS_PER_DIMENSION`] and their product is at
/// most [`MAX_DISPATCH_GROUPS`]. A zero in any dimension stays zero (no work).
pub fn clamp_dispatch_groups(mut groups: [usize; 3]) -> [usize; 3] {
    for g in &mut groups {
        *g = (*g).min(MAX_GROUPS_PER_DIMENSION as usize);
    }
    let mut product = groups.iter().fold(1u128, |p, &g| p.saturating_mul(g as u128));
    // Halve the largest dimension until the product fits. Every step halves a count of at most 16
    // bits, so 16 steps per dimension (48 here) are enough for anything the caps allow.
    for _ in 0..48 {
        if product <= u128::from(MAX_DISPATCH_GROUPS) {
            break;
        }
        let Some(d) = (0..3).max_by_key(|&d| groups[d]) else {
            break;
        };
        if groups[d] <= 1 {
            break;
        }
        let was = groups[d];
        groups[d] = (was / 2).max(1);
        product = product / was as u128 * groups[d] as u128;
    }
    groups
}

/// `(count, instances)` clamped so `count` is at most [`MAX_DRAW_COUNTS`], `instances` at most
/// [`MAX_DRAW_INSTANCES`], and their product at most [`MAX_DRAW_COUNTS`]. Zero stays zero (an
/// empty draw).
pub fn clamp_draw_counts(count: u32, instances: u32) -> (u32, u32) {
    let max = u32::try_from(MAX_DRAW_COUNTS).unwrap_or(u32::MAX);
    let max_instances = u32::try_from(MAX_DRAW_INSTANCES).unwrap_or(u32::MAX);
    let count = count.min(max);
    let instances = instances.min(max_instances);
    if count == 0 || instances == 0 {
        return (count, instances);
    }
    if u64::from(count) * u64::from(instances) <= MAX_DRAW_COUNTS {
        return (count, instances);
    }
    let room = (MAX_DRAW_COUNTS / u64::from(count)).max(1);
    (count, instances.min(room as u32))
}

pub trait ComputeBackend {
    /// `ID3D12CommandQueue`.
    type Queue;
    /// `ID3D12Fence`: a 64-bit value that only rises.
    type Fence;
    /// A GPU-visible view of title memory (a pool, wrapped once).
    type Buffer;
    /// A compute pipeline state.
    type Pipeline;
    /// A descriptor heap (`ID3D12DescriptorHeap`): resource (CBV/SRV/UAV) or sampler.
    type Heap;

    fn create_queue(&self) -> Result<Self::Queue, String>;
    fn create_fence(&self, initial: u64) -> Result<Self::Fence, String>;
    /// Wrap `ptr..ptr+len` (page-aligned) without copying; it stays mapped while the buffer lives.
    fn wrap_title_memory(&self, ptr: *mut u8, len: usize) -> Result<Self::Buffer, String>;
    /// The bytes of GPU memory the device holds for this process now, where the backend knows
    /// (Metal's `currentAllocatedSize`), for memory measurements.
    fn allocated_bytes(&self) -> u64 {
        0
    }
    /// A zeroed buffer of the host's own (`len` bytes) and where the CPU writes it. It lives
    /// while the caller or work submitted with it holds it: memory the GPU reads after the
    /// caller let go (a page table that was replaced) must not be the caller's to free.
    fn host_buffer(&self, len: usize) -> Result<(Self::Buffer, *mut u8), String>;
    /// Makes `buffers` resident for everything submitted to `queue` from now on, so that dispatches
    /// and draws need not name them one by one (Metal: an `MTLResidencySet` of the title's pools,
    /// made once, committed when a pool joins, added to each queue once). Idempotent; a buffer
    /// leaves the set when it is dropped. The default does nothing (the backend names buffers per
    /// dispatch).
    fn make_resident(&self, _queue: &Self::Queue, _buffers: &[&Self::Buffer]) -> Result<(), String> {
        Ok(())
    }
    /// The GPU address of a title address inside `buffer` (`gpu = title + delta`).
    fn gpu_address(&self, buffer: &Self::Buffer, title: *const u8) -> u64;
    /// DXIL container (`victory_shader::emit` output) to a compute pipeline.
    fn create_compute_pipeline(&self, dxil: &[u8]) -> Result<Self::Pipeline, String>;
    /// The same for DXIL without a root signature, laid out as `layout` says.
    fn create_compute_pipeline_with(
        &self,
        dxil: &[u8],
        layout: &RootLayout,
    ) -> Result<Self::Pipeline, String>;
    /// As the two above, waiting at most `wait` for the pipeline: `None` while it is still being
    /// made, and the same call later picks it up. The default makes it at once, however long.
    fn create_compute_pipeline_within(
        &self,
        dxil: &[u8],
        layout: Option<&RootLayout>,
        _wait: std::time::Duration,
    ) -> Result<Option<Self::Pipeline>, String> {
        match layout {
            Some(l) => self.create_compute_pipeline_with(dxil, l),
            None => self.create_compute_pipeline(dxil),
        }
        .map(Some)
    }
    /// Names a pipeline for diagnostics (`VICTORY_TRACE_GPU_FAILS`); the default does nothing.
    fn name_pipeline(&self, _p: &Self::Pipeline, _name: &str) {}
    /// Queue the work and return; never waits for the GPU.
    fn submit(&self, queue: &Self::Queue, work: &Submission<'_, Self>) -> Result<(), String>;
    /// `ID3D12Fence::Signal` from the CPU: queues waiting for `value` go on.
    fn signal(&self, fence: &Self::Fence, value: u64);
    /// `GetCompletedValue`.
    fn completed(&self, fence: &Self::Fence) -> u64;
    /// Block until the fence reaches `value`; false on timeout.
    fn wait(&self, fence: &Self::Fence, value: u64, timeout_ms: u64) -> bool;
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_DISPATCH_GROUPS, MAX_DRAW_COUNTS, MAX_DRAW_INSTANCES, MAX_GROUPS_PER_DIMENSION,
        clamp_dispatch_groups, clamp_draw_counts,
    };

    #[test]
    fn a_dispatch_past_the_caps_is_clamped() {
        assert_eq!(clamp_dispatch_groups([16, 16, 6]), [16, 16, 6]);
        assert_eq!(clamp_dispatch_groups([0, 9, 9]), [0, 9, 9], "an empty dispatch stays empty");
        // One dimension over D3D12's limit.
        assert_eq!(
            clamp_dispatch_groups([100_000, 1, 1]),
            [MAX_GROUPS_PER_DIMENSION as usize, 1, 1]
        );
        // A garbage record: the product is halved until it fits, and no count goes to zero.
        let asked = [510usize, 1_182_027_346, 1_314_076_063];
        let ran = clamp_dispatch_groups(asked);
        assert!(
            ran.iter().map(|&n| n as u128).product::<u128>() <= u128::from(MAX_DISPATCH_GROUPS),
            "{ran:?}"
        );
        assert!(ran.iter().all(|&n| n >= 1), "{ran:?}");
        assert!(ran.iter().all(|&n| n <= asked[0].max(asked[1]).max(asked[2])), "{ran:?}");
    }

    #[test]
    fn a_draw_past_the_cap_is_clamped() {
        assert_eq!(clamp_draw_counts(3, 4), (3, 4));
        assert_eq!(
            clamp_draw_counts(0, u32::MAX),
            (0, MAX_DRAW_INSTANCES as u32),
            "an empty draw stays empty; its instance count is still capped"
        );
        // A full-screen draw with millions of instances is clamped too.
        assert_eq!(
            clamp_draw_counts(3, u32::MAX),
            (3, MAX_DRAW_INSTANCES as u32)
        );
        let (count, instances) = clamp_draw_counts(3_190_467_240, 1_051_372_204);
        assert_eq!(count, MAX_DRAW_COUNTS as u32);
        assert_eq!(instances, 1);
        let (count, instances) = clamp_draw_counts(1_000, 1_000_000);
        assert_eq!(count, 1_000);
        assert_eq!(
            instances,
            (MAX_DRAW_COUNTS / 1_000).min(MAX_DRAW_INSTANCES) as u32
        );
    }
}
