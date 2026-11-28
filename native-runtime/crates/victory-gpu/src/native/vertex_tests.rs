//! Capture a native host submission at the backend boundary without a GPU or shader converter.

use super::host::NativeHost;
use super::list::NativeList;
use super::memory::{PoolInfo, PoolObserver};
use std::cell::RefCell;
use std::rc::Rc;
use victory_backend::backend::{ComputeBackend, RootLayout, Submission};
use victory_backend::resources::*;

#[test]
fn a_non_indexed_draw_does_not_resolve_a_stale_index_buffer() {
    let host = NativeHost::new(Capture::default());
    let pool = PoolInfo {
        id: 1,
        base: 0x40000,
        len: 0x4000,
        memory_entry: None,
    };
    host.memory().mapped(0x80000, pool.len, &pool, 0);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.set_vertex_buffer(2, 0x80100);
    // An index buffer left bound whose pages the title has since unmapped: a non-indexed draw
    // reads no index, so it must not be resolved (and dropped) for it.
    list.set_index_buffer_view(0x9000_0000, 16, false);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    assert_eq!(*host.backend.draws.borrow(), [vec![(2, 0x40000, 0x40100)]]);
}

#[test]
fn an_empty_index_buffer_view_clears_the_binding() {
    let mut list = NativeList::<Capture>::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_index_buffer_view(0x80100, 0, false);
    assert!(list.draw_indexed_instanced(3, 1, 0, 0, 0).is_err());
    list.set_index_buffer_view(0, 16, false);
    assert!(list.draw_indexed_instanced(3, 1, 0, 0, 0).is_err());
}

#[test]
fn an_empty_vertex_buffer_view_unbinds_the_slot() {
    let mut list = NativeList::<Capture>::default();
    list.set_vertex_buffer_view(2, 0x80100, 0);
    assert_eq!(list.vertex_buffer_address(2), None);
    list.set_vertex_buffer_view(2, 0, 16);
    assert_eq!(list.vertex_buffer_address(2), None);
}

#[test]
fn a_draw_with_only_vertex_slot_two_submits_its_buffer() {
    let host = NativeHost::new(Capture::default());
    let pool = PoolInfo {
        id: 1,
        base: 0x40000,
        len: 0x4000,
        memory_entry: None,
    };
    host.memory().mapped(0x80000, pool.len, &pool, 0);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.set_vertex_buffer(2, 0x80100);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    assert_eq!(host.unmapped(), 0, "empty vertex slots dropped the draw");
    assert_eq!(*host.backend.draws.borrow(), [vec![(2, 0x40000, 0x40100)]]);
}

#[derive(Default)]
pub(super) struct Capture {
    pub(super) views: RefCell<Vec<ViewDesc>>,
    // For each draw, the (slot, pool base, resolved address) bindings received.
    draws: RefCell<Vec<Vec<(usize, usize, usize)>>>,
    host_buffers: RefCell<Vec<Box<[u8]>>>,
    // (heap index, pool base, resolved address, size) for each buffer descriptor write.
    pub(super) buffer_writes: RefCell<Vec<(usize, usize, usize, u32)>>,
    // The light format and constants of each ambient pass received.
    pub(super) ambient: RefCell<Vec<(Format, [u32; 6])>>,
    // When set, `submit_passes` fails with it: (whether the command buffer committed, the pass
    // indices dropped from it).
    pub(super) refuse: std::cell::RefCell<Option<(bool, Vec<usize>)>>,
}

impl GraphicsBackend for Capture {
    type RenderPipeline = ();

    fn submit_passes(&self, _: &(), work: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        if let Some((committed, refused)) = self.refuse.borrow().as_ref() {
            return Err(SubmitFailure {
                committed: *committed,
                refused: refused.clone(),
                error: "refused for the test".to_string(),
            });
        }
        for pass in work.passes {
            match pass {
                Pass::Render(render) => {
                    for draw in &render.draws {
                        self.draws.borrow_mut().push(
                            draw.vertex_buffers
                                .iter()
                                .map(|&(slot, base, address)| (slot, *base, address as usize))
                                .collect(),
                        );
                    }
                }
                Pass::Ambient(a) => self.ambient.borrow_mut().push((a.light.format, a.constants)),
                _ => {}
            }
        }
        Ok(())
    }

    fn create_render_pipeline(&self, _: &RenderPipelineDesc<'_>) -> Result<(), String> {
        unreachable!()
    }
}

impl ComputeBackend for Capture {
    type Queue = ();
    type Fence = ();
    type Buffer = usize;
    type Pipeline = ();
    type Heap = RefCell<Vec<Option<TextureDesc>>>;

    fn wrap_title_memory(&self, base: *mut u8, _: usize) -> Result<usize, String> {
        Ok(base as usize)
    }

    fn create_queue(&self) -> Result<(), String> {
        unreachable!()
    }
    fn create_fence(&self, _: u64) -> Result<(), String> {
        unreachable!()
    }
    fn host_buffer(&self, len: usize) -> Result<(usize, *mut u8), String> {
        let mut bytes = vec![0; len].into_boxed_slice();
        let ptr = bytes.as_mut_ptr();
        self.host_buffers.borrow_mut().push(bytes);
        Ok((ptr as usize, ptr))
    }
    fn gpu_address(&self, _: &usize, _: *const u8) -> u64 {
        unreachable!()
    }
    fn create_compute_pipeline(&self, _: &[u8]) -> Result<(), String> {
        unreachable!()
    }
    fn create_compute_pipeline_with(&self, _: &[u8], _: &RootLayout) -> Result<(), String> {
        unreachable!()
    }
    fn submit(&self, _: &(), _: &Submission<'_, Self>) -> Result<(), String> {
        unreachable!()
    }
    fn signal(&self, _: &(), _: u64) {
        unreachable!()
    }
    fn completed(&self, _: &()) -> u64 {
        unreachable!()
    }
    fn wait(&self, _: &(), _: u64, _: u64) -> bool {
        unreachable!()
    }
}

impl ResourceBackend for Capture {
    type Texture = TextureDesc;
    type Sampler = ();

    fn create_heap(&self, _: HeapKind, count: usize) -> Result<Self::Heap, String> {
        Ok(RefCell::new(vec![None; count]))
    }
    fn heap_address(&self, _: &Self::Heap, index: usize) -> u64 {
        (index * 32) as u64
    }
    fn write_buffer(&self, _: &Self::Heap, index: usize, base: &usize, address: *const u8, size: u32) {
        self.buffer_writes.borrow_mut().push((index, *base, address as usize, size));
    }
    fn write_texture(&self, heap: &Self::Heap, slot: usize, tex: &TextureDesc, _: f32) {
        heap.borrow_mut()[slot] = Some(*tex);
    }
    fn write_sampler(&self, _: &Self::Heap, _: usize, _: &(), _: f32) {
        unreachable!()
    }
    fn clear_entry(&self, _: &Self::Heap, _: usize) {
        unreachable!()
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureDesc, String> {
        Ok(*desc)
    }
    fn create_texture_view(&self, tex: &TextureDesc, view: &ViewDesc) -> Result<TextureDesc, String> {
        self.views.borrow_mut().push(*view);
        Ok(*tex)
    }
    fn create_texture_over_memory(
        &self,
        _: &usize,
        _: *const u8,
        _: &TextureDesc,
        _: usize,
    ) -> Result<TextureDesc, String> {
        unreachable!()
    }
    fn linear_alignment(&self, _: Format) -> usize {
        unreachable!()
    }
    fn upload_texture(&self, _: &TextureDesc, _: usize, _: usize, _: &[u8], _: usize, _: usize) {
        unreachable!()
    }
    fn read_texture(&self, _: &TextureDesc, _: usize, _: usize, _: &mut [u8], _: usize, _: usize) {
        unreachable!()
    }
    fn create_sampler(&self, _: &SamplerDesc) -> Result<(), String> {
        unreachable!()
    }
}

/// A pool of real memory shown to the "title" at an alias, the way `title_tests` makes one.
fn aliased_pool(host: &NativeHost<Capture>, mem: &[u8]) -> u64 {
    let base = mem.as_ptr() as usize;
    let alias = (base + 0x8000_0000) & !0xfff;
    let pool = PoolInfo { id: 1, base, len: mem.len(), memory_entry: None };
    host.memory().mapped(alias, pool.len, &pool, 0);
    alias as u64
}

#[test]
fn a_submission_that_never_committed_pays_its_value_writes_on_the_cpu() {
    let backend = Capture::default();
    backend.refuse.replace(Some((false, Vec::new())));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    list.fill(alias + 0x200, 4, 0x5A);
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused the submission");
    assert_eq!(&mem[0x100..0x108], &[0xAB; 8], "the lost submission's first fill was not paid");
    assert_eq!(&mem[0x200..0x204], &[0x5A; 4], "the lost submission's second fill was not paid");
}

#[test]
fn only_the_fills_of_refused_passes_are_paid_on_the_cpu() {
    let backend = Capture::default();
    // The command buffer commits, but pass 0 (the first fill) and pass 1 (the draw) are dropped.
    backend.refuse.replace(Some((true, vec![0, 1])));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.fill(alias + 0x100, 8, 0xAB); // pass 0: refused
    list.set_vertex_buffer(2, alias + 0x300);
    list.draw_instanced(3, 1, 0, 0).unwrap(); // pass 1: refused
    list.fill(alias + 0x200, 4, 0x5A); // pass 2: kept, runs on the GPU
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused passes 0 and 1");
    assert_eq!(&mem[0x100..0x108], &[0xAB; 8], "the refused pass's fill was not paid");
    assert_eq!(&mem[0x200..0x204], &[0; 4], "a kept pass's fill must stay with the GPU");
}

#[test]
fn a_submitted_piece_leaves_its_value_writes_to_the_gpu() {
    let host = NativeHost::new(Capture::default());
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    host.execute(&(), &[&list], &[], &[]).unwrap();
    assert_eq!(&mem[0x100..0x108], &[0; 8], "the host wrote a fill the GPU was given");
}

/// The ambient stand-in's command becomes a pass of its own at the backend, with the light image
/// and the constants the list recorded (`crate::ambient`, `VICTORY_AMBIENT=1`).
#[test]
fn an_ambient_command_becomes_its_own_pass() {
    let host = NativeHost::new(Capture::default());
    let mut list = NativeList::default();
    let light = Rc::new(TextureDesc::d2(Format::Rgb9e5Float, 8, 4));
    list.ambient(&light, [1, 2, 3, 4, 8, 4]);
    host.execute(&(), &[&list], &[], &[]).unwrap();
    assert_eq!(
        *host.backend.ambient.borrow(),
        [(Format::Rgb9e5Float, [1, 2, 3, 4, 8, 4])]
    );
}

/// A descriptor table a compute dispatch binds, with room for `entries` 32-byte entries.
#[cfg(target_os = "macos")]
struct Table {
    _slots: Box<[u32]>,
    start: u64,
}

#[cfg(target_os = "macos")]
impl Table {
    fn new(
        host: &NativeHost<Capture>,
        entries: usize,
    ) -> (Self, Rc<super::list::NativeHeap<Capture>>) {
        let slots = vec![0u32; entries * 8].into_boxed_slice();
        let start = slots.as_ptr() as u64;
        let heap = host.heap(HeapKind::Resource, start, entries).unwrap();
        (
            Table {
                _slots: slots,
                start,
            },
            heap,
        )
    }

    /// An unordered access view of `resource` in `entry`, as `CreateUnorderedAccessView` leaves it.
    fn uav(&mut self, entry: usize, resource: usize, format: u32) {
        // The description's first word is the DXGI format: the view's own, or the resource's.
        let desc = [format, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        crate::descriptor::create_uav(
            std::ptr::null_mut(),
            resource as _,
            std::ptr::null_mut(),
            &desc,
            self.start + (entry * 32) as u64,
        );
    }

    /// An unordered access view of `resource` with no description (the resource's own format).
    fn uav_plain(&mut self, entry: usize, resource: usize) {
        crate::descriptor::create_uav(
            std::ptr::null_mut(),
            resource as _,
            std::ptr::null_mut(),
            std::ptr::null(),
            self.start + (entry * 32) as u64,
        );
    }

    /// A shader resource view of `resource` in `entry`.
    fn srv(&mut self, entry: usize, resource: usize, format: u32) {
        let desc = [format, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        crate::descriptor::create_srv(
            std::ptr::null_mut(),
            resource as _,
            &desc,
            self.start + (entry * 32) as u64,
        );
    }
}

/// The ambient light is found by the image a dispatch writes at the resolve's own place, not by a
/// program name the console's containers do not carry: a dispatch binding an unordered R9G9B9E5
/// view at dword 176 of root parameter 1 has a light at that entry. A refused resolve has no
/// binding, so its table is read to the floor, which reaches it.
#[cfg(target_os = "macos")]
#[test]
fn a_light_is_found_at_the_resolves_place() {
    let host = NativeHost::new(Capture::default());
    let (mut table, heap) = Table::new(&host, 32);
    table.uav(22, 0xA1, crate::ambient::LIGHT_FORMAT);
    let mut list = NativeList::<Capture>::default();
    list.set_compute_root_table(1, &heap, 0);
    let info =
        |resource: usize| (resource == 0xA1).then_some((crate::ambient::LIGHT_FORMAT, 1280, 720));
    assert_eq!(
        crate::ambient::bound_lights(list.compute_args(), None, info),
        vec![crate::ambient::BoundLight {
            resource: 0xA1,
            view_format: crate::ambient::LIGHT_FORMAT,
            resource_format: crate::ambient::LIGHT_FORMAT,
            slot: 1,
            offset: 176,
            size: [1280, 720],
        }]
    );
}

/// The resolve writes the light through an R9G9B9E5 view of an R32_UINT object (the words are
/// its own); the view names the light, not the object's format.
#[cfg(target_os = "macos")]
#[test]
fn the_lights_view_format_names_it_over_the_resources_own() {
    let host = NativeHost::new(Capture::default());
    let (mut table, heap) = Table::new(&host, 32);
    table.uav(22, 0xA2, crate::ambient::LIGHT_FORMAT); // R9G9B9E5 view of an R32_UINT object
    let mut list = NativeList::<Capture>::default();
    list.set_compute_root_table(1, &heap, 0);
    let info = |resource: usize| (resource == 0xA2).then_some((42, 1920, 1080));
    assert_eq!(
        crate::ambient::bound_lights(list.compute_args(), None, info),
        vec![crate::ambient::BoundLight {
            resource: 0xA2,
            view_format: crate::ambient::LIGHT_FORMAT,
            resource_format: 42,
            slot: 1,
            offset: 176,
            size: [1920, 1080],
        }]
    );
}

/// A view with no description takes the resource's format, so an R9G9B9E5 object there is the
/// light too.
#[cfg(target_os = "macos")]
#[test]
fn a_view_with_no_description_takes_the_resources_format() {
    let host = NativeHost::new(Capture::default());
    let (mut table, heap) = Table::new(&host, 32);
    table.uav_plain(22, 0xA3);
    let mut list = NativeList::<Capture>::default();
    list.set_compute_root_table(1, &heap, 0);
    let info =
        |resource: usize| (resource == 0xA3).then_some((crate::ambient::LIGHT_FORMAT, 1280, 720));
    assert_eq!(
        crate::ambient::bound_lights(list.compute_args(), None, info),
        vec![crate::ambient::BoundLight {
            resource: 0xA3,
            view_format: crate::ambient::LIGHT_FORMAT,
            resource_format: crate::ambient::LIGHT_FORMAT,
            slot: 1,
            offset: 176,
            size: [1280, 720],
        }]
    );
}

/// An R9G9B9E5 image at another place in the tables is not the light. In the 15:15 run on 27
/// November compute `39ef10815cb825a0` bound one at dword 16 of root parameter 0; the pass filled
/// its 1920x1080 output with the constant and the frame became a flat beige haze with no geometry.
#[cfg(target_os = "macos")]
#[test]
fn a_full_screen_rgb9e5_image_in_parameter_0_is_not_the_light() {
    let host = NativeHost::new(Capture::default());
    let (mut table, heap) = Table::new(&host, 32);
    table.uav(2, 0xE1, crate::ambient::LIGHT_FORMAT); // dword 16 of root parameter 0
    let mut list = NativeList::<Capture>::default();
    list.set_compute_root_table(0, &heap, 0);
    let info =
        |resource: usize| (resource == 0xE1).then_some((crate::ambient::LIGHT_FORMAT, 1920, 1080));
    assert!(crate::ambient::bound_lights(list.compute_args(), None, info).is_empty());
}

/// Only an unordered view of an R9G9B9E5 image at the light's place is a light: an R32_UINT
/// object there (the reflections' history, which the lighting reads through R9G9B9E5 placed
/// views) and a shader resource view of the light's own texels are not written by the pass, so
/// they are not filled.
#[cfg(target_os = "macos")]
#[test]
fn only_an_unordered_rgb9e5_view_at_the_lights_place_is_a_light() {
    let host = NativeHost::new(Capture::default());
    let (mut uint, uint_heap) = Table::new(&host, 32);
    uint.uav(22, 0xB1, 42); // R32_UINT
    let mut uav = NativeList::<Capture>::default();
    uav.set_compute_root_table(1, &uint_heap, 0);
    let info = |resource: usize| (resource == 0xB1).then_some((42, 1920, 1080));
    assert!(crate::ambient::bound_lights(uav.compute_args(), None, info).is_empty());

    let (mut seen, seen_heap) = Table::new(&host, 32);
    seen.srv(22, 0xC1, crate::ambient::LIGHT_FORMAT);
    let mut srv = NativeList::<Capture>::default();
    srv.set_compute_root_table(1, &seen_heap, 0);
    let info =
        |resource: usize| (resource == 0xC1).then_some((crate::ambient::LIGHT_FORMAT, 1280, 720));
    assert!(crate::ambient::bound_lights(srv.compute_args(), None, info).is_empty());
}

/// A translated dispatch names the table entries it reads, and the light at its place is found
/// there; the floor is for a refused program, not a limit on the ones that translate.
#[cfg(target_os = "macos")]
#[test]
fn a_translated_dispatch_finds_the_light_at_its_place() {
    use super::list::Binding;
    let host = NativeHost::new(Capture::default());
    let (mut table, heap) = Table::new(&host, 40);
    table.uav(22, 0xA2, crate::ambient::LIGHT_FORMAT);
    let mut list = NativeList::<Capture>::default();
    list.set_compute_root_table(1, &heap, 0);
    let binding = Binding {
        table_entries: vec![(1, 23)],
        ..Default::default()
    };
    let info =
        |resource: usize| (resource == 0xA2).then_some((crate::ambient::LIGHT_FORMAT, 1280, 720));
    assert_eq!(
        crate::ambient::bound_lights(list.compute_args(), Some(&binding), info),
        vec![crate::ambient::BoundLight {
            resource: 0xA2,
            view_format: crate::ambient::LIGHT_FORMAT,
            resource_format: crate::ambient::LIGHT_FORMAT,
            slot: 1,
            offset: 176,
            size: [1280, 720],
        }]
    );
}
