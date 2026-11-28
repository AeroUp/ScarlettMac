//! `VICTORY_TRACE_GPU_VERTICES` (`gpu_vertex_trace`): a dispatch dropped for want of a host
//! compute pipeline names, by its buffer unordered-access views, root UAVs and heap, memory a
//! later draw reads as vertices; the trace reports the overlap at the draw's submission. On Metal
//! through a translated stage-in vertex shader, and for a late sample of what a dispatch that ran
//! wrote; without a GPU (the backend boundary) for a vertex shader's table views, its null views,
//! their statistics, the kinds of writer and the verdict.
use super::scaled_vertex_tests::{translate, vertex};
use super::{Pool, host};
use crate::descriptor::{View, record};
use crate::native::gpu_vertex_trace::{self as trace, Config, FrameCounts, Via, WriteProfile};
use crate::native::host::NativeHost;
use crate::native::list::{Binding, NativeList};
use crate::native::memory::{PoolInfo, PoolObserver};
use crate::native::title::{Pipe, Title};
use crate::native::{state, vertex_tests::Capture};
use std::rc::Rc;
use victory_backend::backend::ComputeBackend;
use victory_backend::resources::{Format, HeapKind, TextureDesc, Topology};
use victory_shader::native::HwStage;

const DROPPED: &str = "compute 0123456789abcdef";

/// A placed buffer view (`res[0]` 1) of `count` 4-byte elements at title address `at`.
fn placed_buffer(unordered: bool, at: u64, count: u32) -> View {
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut d = [0u32; 16];
    d[1] = 1; // BUFFER
    if unordered {
        (d[4], d[5]) = (count, 4);
    } else {
        (d[6], d[7]) = (count, 4);
    }
    (d[10], d[11]) = (at as u32, (at >> 32) as u32);
    View::Placed(unordered, resource, d)
}

#[test]
fn a_dropped_dispatchs_buffer_uav_read_as_a_vertex_stream_is_reported() {
    let Some(h) = host() else { return };
    trace::force_on_this_thread();
    let pool = Pool::new(&h, 1);
    let memory = h.memory();
    let mut t = Title::new(h.backend, memory);
    let ps = translate(
        HwStage::Ps,
        &[
            0xBEFC_0300, // s_mov_b32 m0, s0
            0xC806_0002,
            0xC80A_0102,
            0xC80E_0202,
            0xC812_0302, // flat attr0.xyzw
            0xF800_180F,
            0x0403_0201, // exp mrt0 v1, v2, v3, v4 done vm
            0xBF81_0000,
        ],
        false,
    );
    let vs = vertex(4, 1, false);
    let reads = state::vertex_inputs(&vs.dxil).unwrap();
    let g = state::Graphics::guess_base().for_draw(
        vec![Format::Rgba32Float],
        None,
        vec![20],
        &reads,
        false,
    );
    let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &g, "vertex 1111 pixel 2222")
    else {
        panic!("{:?}", t.shaders);
    };
    assert!(
        binding.as_ref().is_some_and(|b| b.vertex_attributes > 0),
        "a stage-in pipeline"
    );
    let mut desc = TextureDesc::d2(Format::Rgba32Float, 4, 4);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let queue = t.host.backend.create_queue().unwrap();
    let fence = t.host.backend.create_fence(0).unwrap();
    for (i, xy) in [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]]
        .iter()
        .enumerate()
    {
        pool.write(i * 20, &[xy[0], xy[1], 0.0, 1.0, 0.5]);
    }

    // The dispatch of a refused compute shader: a placed buffer UAV over the three vertices in
    // its table, and a root view elsewhere. Dropped as `glue::dropped_dispatch` drops it.
    let views = vec![0u32; 8];
    let start = views.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 1).unwrap();
    record(start, placed_buffer(true, pool.va(0), 15));
    let mut compute = NativeList::default();
    compute.set_heaps(Some(&heap), None);
    compute.set_compute_root_table(0, &heap, 0);
    compute.set_compute_root_view(1, pool.va(0x1000));
    trace::dropped_dispatch(
        &t.host,
        compute.compute_args(),
        compute.compute_heap(),
        DROPPED,
        "refused",
        None,
    );

    let draw = |serial: u64| {
        let mut list = NativeList::default();
        list.set_render_targets(std::slice::from_ref(&color), None);
        list.set_graphics_pipeline(&pso, Topology::Triangles);
        list.set_graphics_binding(binding.clone());
        list.set_vertex_buffer_view(0, pool.va(0), 60);
        list.draw_instanced(3, 1, 0, 0).unwrap();
        t.host
            .execute(&queue, &[&list], &[], &[(&fence, serial)])
            .unwrap();
        assert!(
            t.host.backend.wait(&fence, serial, 10_000),
            "GPU did not finish"
        );
    };
    draw(1);
    let logged = trace::logged_for_test();
    assert_eq!(logged.hits.len(), 1, "{logged:?}");
    let hit = &logged.hits[0];
    assert_eq!(&*hit.pipeline, "vertex 1111 pixel 2222");
    assert_eq!(hit.read.via, Via::Stream(0));
    assert_eq!(&*hit.writer.shader, DROPPED);
    assert_eq!(
        (hit.writer.through, hit.writer.lo, hit.writer.hi),
        ("table uav", pool.va(0), pool.va(60))
    );
    assert_eq!(hit.bytes, 60);
    assert!(
        logged.summaries.is_empty(),
        "no statistics for a stage-in pipeline"
    );

    // Once per pipeline; the frame's line counts every draw; a present forgets the writers.
    draw(2);
    assert_eq!(trace::logged_for_test().hits.len(), 1);
    assert_eq!(
        trace::present_for_test(),
        FrameCounts {
            dispatches: 1,
            ranges: 2,
            draws: 2,
            draws_hit: 2,
            reads_hit: 2,
            ..FrameCounts::default()
        }
    );
    draw(3);
    assert_eq!(
        trace::present_for_test(),
        FrameCounts {
            draws: 1,
            ..FrameCounts::default()
        }
    );
    drop(views);
}

#[test]
fn a_vertex_shaders_table_views_are_checked_summarised_and_flagged_when_null() {
    trace::force_on_this_thread();
    let host = NativeHost::new(Capture::default());
    // Title memory with a hole: [0, 0x1000) and [0x2000, 0x3000) mapped, at their own addresses.
    let memory = vec![0u8; 0x3000];
    let base = memory.as_ptr() as u64;
    let pool = PoolInfo {
        id: 9,
        base: base as usize,
        len: memory.len(),
        memory_entry: None,
    };
    host.memory().pool_created(&pool);
    host.memory().mapped(base as usize, 0x1000, &pool, 0);
    host.memory()
        .mapped(base as usize + 0x2000, 0x1000, &pool, 0x2000);
    // 64 dwords: 16 zeros, then floats, one of them not finite.
    let floats: Vec<f32> = (0..64)
        .map(|k| {
            if k < 16 {
                0.0
            } else if k == 40 {
                f32::INFINITY
            } else {
                k as f32
            }
        })
        .collect();
    // SAFETY: inside `memory`, which nothing else reads yet.
    unsafe { std::ptr::copy_nonoverlapping(floats.as_ptr().cast::<u8>(), base as *mut u8, 256) };

    // Entries 0 and 1: what the vertex shader reads (the vertices, and a view over the hole);
    // entry 2: the dropped dispatch's UAV over the vertices.
    let views = vec![0u32; 8 * 3];
    let start = views.as_ptr() as u64;
    let heap = host.heap(HeapKind::Resource, start, 3).unwrap();
    record(start, placed_buffer(false, base, 64));
    record(start + 32, placed_buffer(false, base + 0x800, 0x400));
    record(start + 64, placed_buffer(true, base + 0x80, 4));
    let mut compute = NativeList::<Capture>::default();
    compute.set_compute_root_table(0, &heap, 2);
    trace::dropped_dispatch(
        &host,
        compute.compute_args(),
        compute.compute_heap(),
        DROPPED,
        "refused",
        None,
    );

    let binding = Binding {
        name: "vertex 3333 pixel 4444".into(),
        table_entries: vec![(0, 2)],
        ..Binding::default()
    };
    assert_eq!(
        binding.vertex_attributes, 0,
        "no input layout: the vertex shader fetches"
    );
    let mut list = NativeList::default();
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.set_heaps(Some(&heap), None);
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_graphics_binding(Some(Rc::new(binding)));
    list.set_graphics_root_table(0, &heap, 0);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    list.draw_instanced(3, 1, 0, 0).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    let logged = trace::logged_for_test();
    assert_eq!(logged.hits.len(), 1, "{logged:?}");
    assert_eq!(logged.hits[0].read.via, Via::Table(0, 0));
    assert_eq!(logged.hits[0].bytes, 16);
    assert_eq!(
        logged.null,
        [(
            Via::Table(0, 1),
            base + 0x800,
            0x1000,
            "a gap: the null view"
        )],
        "flagged once"
    );
    // Each view summarised once: the vertices' zeros, and only the mapped half of the other.
    assert_eq!(
        logged.summaries,
        [
            (Via::Table(0, 0), 256, 64, 16),
            (Via::Table(0, 1), 0x800, 0x200, 0x200)
        ]
    );
    drop((views, memory));
}

/// A vertex shader reading one table view, as the backend-boundary tests bind it.
fn table_draw(name: &str) -> Binding {
    Binding {
        name: name.into(),
        table_entries: vec![(0, 1)],
        ..Binding::default()
    }
}

#[test]
fn root_uavs_heap_uavs_and_computed_writers_are_told_apart_and_end_in_a_verdict() {
    trace::force_on_this_thread();
    trace::configure_for_test(Config {
        draws: 1,
        after: 0,
        late: 8,
        later: 1,
        verdict: 0,
    });
    let host = NativeHost::new(Capture::default());
    let memory = vec![0u8; 0x3000];
    let base = memory.as_ptr() as u64;
    let pool = PoolInfo {
        id: 9,
        base: base as usize,
        len: memory.len(),
        memory_entry: None,
    };
    host.memory().pool_created(&pool);
    host.memory().mapped(base as usize, 0x3000, &pool, 0);

    // Entry 0: the vertex shader's view of 16 dwords at +0x200; entry 1: a buffer UAV at +0x220
    // that no table of the dropped dispatches holds (only a heap-indexing program reaches it).
    let views = vec![0u32; 8 * 2];
    let start = views.as_ptr() as u64;
    let heap = host.heap(HeapKind::Resource, start, 2).unwrap();
    record(start, placed_buffer(false, base + 0x200, 16));
    record(start + 32, placed_buffer(true, base + 0x220, 4));
    let mut compute = NativeList::<Capture>::default();
    compute.set_heaps(Some(&heap), None);
    compute.set_compute_root_view(1, base + 0x100);
    compute.set_compute_root_view(2, base + 0x800);
    let root_uav = WriteProfile {
        root_uavs: vec![1],
        decoded: true,
        ..WriteProfile::default()
    };
    let heap_uav = WriteProfile {
        heap: true,
        descriptor_stores: true,
        decoded: true,
        ..WriteProfile::default()
    };
    let computed = WriteProfile {
        computed: true,
        decoded: true,
        ..WriteProfile::default()
    };
    let args = compute.compute_args();
    for (hash, p) in [("a1", &root_uav), ("a2", &heap_uav), ("a3", &computed)] {
        trace::dropped_dispatch(
            &host,
            args,
            compute.compute_heap(),
            &format!("compute {hash}"),
            "refused",
            Some(p),
        );
    }

    let mut list = NativeList::default();
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.set_heaps(Some(&heap), None);
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_graphics_binding(Some(Rc::new(table_draw("vertex 7777 pixel 8888"))));
    list.set_graphics_root_table(0, &heap, 0);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    // The root UAV, to the end of the mapping holding it (no buffer resource does): estimated.
    // Root view 2 is not one of the program's UAVs, so it is not noted. The heap's UAV: coarse.
    let logged = trace::logged_for_test();
    let hits: Vec<_> = logged
        .hits
        .iter()
        .map(|h| {
            (
                h.writer.through,
                trace::precision(h.writer.through),
                h.writer.lo - base,
                h.writer.hi - base,
                h.bytes,
            )
        })
        .collect();
    assert_eq!(
        hits,
        [
            (trace::ROOT_UAV, "estimated", 0x100, 0x3000, 0x40),
            (trace::HEAP_UAV, "coarse", 0x220, 0x230, 0x10)
        ]
    );
    assert_eq!(
        logged.summaries,
        [(Via::Table(0, 0), 0x40, 16, 16)],
        "zeros at submission"
    );

    // The dropped writers would have written the vertices: they are there by the late sample.
    let ones = [1.0f32; 16];
    // SAFETY: inside `memory`, which nothing else reads now.
    unsafe {
        std::ptr::copy_nonoverlapping(ones.as_ptr().cast::<u8>(), (base + 0x200) as *mut u8, 64)
    };
    assert_eq!(
        trace::present_for_test(),
        FrameCounts {
            dispatches: 3,
            ranges: 1,
            heap_writers: 1,
            heap_ranges: 1,
            computed_writers: 1,
            draws: 1,
            draws_hit: 1,
            reads_hit: 1,
            ..FrameCounts::default()
        }
    );
    assert!(
        trace::logged_for_test().verdicts.is_empty(),
        "a later sample is still due"
    );
    trace::present_for_test();
    let logged = trace::logged_for_test();
    assert_eq!(
        logged.late,
        [
            (Via::Table(0, 0), false, 16, 0),
            (Via::Table(0, 0), true, 16, 0)
        ]
    );
    assert_eq!(
        logged.verdicts,
        [trace::Verdict {
            pipeline: "vertex 7777 pixel 8888".into(),
            views: "1/1/1".into(),
            zero_early: "1.0000".into(),
            zero_late: "0.0000".into(),
            zero_later: "0.0000".into(),
            overlaps: "table uav:0 root uav:1 root view:0 heap uav (any):1".into(),
            writers: "compute a1 (refused; root uav), compute a2 (refused; heap uav (any))".into(),
            unbounded: "compute a3 (refused; title memory (computed))".into(),
        }],
        "once, when the window is done"
    );
    trace::present_for_test();
    assert_eq!(trace::logged_for_test().verdicts.len(), 1);
    drop((views, memory));
}

#[test]
fn a_late_sample_sees_what_a_dispatch_that_ran_wrote() {
    use super::addr_tests::{dispatch, halves, title, translate_io};
    let Some(mut t) = title() else { return };
    trace::force_on_this_thread();
    trace::configure_for_test(Config {
        draws: 512,
        after: 0,
        late: 8,
        later: 0,
        verdict: 0,
    });
    // A pool shown to the title at its own address (as the title's memory is in a run), on a
    // whole 64 KiB page (the page table's), its first 256 bytes zero.
    let mem = victory_metal::mem::Alias::new(0x2_0000).unwrap();
    let va = (mem.primary as u64).next_multiple_of(0x1_0000);
    let at = (va - mem.primary as u64) as usize;
    let info = PoolInfo {
        id: 3,
        base: mem.primary as usize,
        len: 0x2_0000,
        memory_entry: None,
    };
    t.host.memory().pool_created(&info);
    t.host.memory().mapped(va as usize, 0x1_0000, &info, at);
    // SAFETY: the pool's own mapping; the GPU is idle.
    unsafe { std::ptr::write_bytes(va as *mut u8, 0, 256) };
    let views = vec![0u32; 8];
    let start = views.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 1).unwrap();
    record(start, placed_buffer(false, va, 64));
    let mut list = NativeList::default();
    list.set_graphics_root_table(0, &heap, 0);
    let binding = table_draw("vertex 5555 pixel 6666");
    // The draw is submitted before the GPU has run the dispatch that writes its vertices.
    trace::draw(
        &t.host,
        list.graphics_args().0,
        Some(&binding),
        &[],
        None,
        false,
    );
    assert_eq!(
        trace::logged_for_test().summaries,
        [(Via::Table(0, 0), 256, 64, 64)]
    );

    // Every lane stores 1.0 at s[0:1] + 4 * lane (`v_lshlrev_b32 v1, 2, v0`, `v_mov_b32 v2,
    // 1.0`, `global_store_dword v1, v2, s[0:1]`), run to completion on the GPU.
    let cs = translate_io(
        &[
            0x3402_0082,
            0x7E04_02FF,
            0x3F80_0000,
            0xDC70_8000,
            0x0000_0201,
            0xBF81_0000,
        ],
        2,
        0,
        64,
        crate::native::shaders::options(),
    );
    dispatch(&mut t, &cs, &halves(va), false);
    // SAFETY: as above; the dispatch has completed.
    assert_eq!(unsafe { (va as *const f32).read() }, 1.0);
    // Its command buffer has completed (`dispatch` waited): the next present reads it again.
    trace::present_with_for_test(victory_metal::gfx::committed_serial(), |_| true);
    assert_eq!(
        trace::logged_for_test().late,
        [(Via::Table(0, 0), false, 64, 0)]
    );
    drop((views, mem));
}
