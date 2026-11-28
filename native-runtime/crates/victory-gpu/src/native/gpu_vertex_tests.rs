//! `VICTORY_TRACE_GPU_VERTICES` (`gpu_vertex_trace`): a dispatch dropped for want of a host
//! compute pipeline names, by its buffer unordered-access views, memory a later draw reads as
//! vertices; the trace reports the overlap at the draw's submission. On Metal through a translated
//! stage-in vertex shader, and without a GPU (the backend boundary) for a vertex shader's table
//! views, its null views and their statistics.
use super::scaled_vertex_tests::{translate, vertex};
use super::{Pool, host};
use crate::descriptor::{View, record};
use crate::native::gpu_vertex_trace::{self as trace, FrameCounts, Via};
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
    trace::dropped_dispatch(&t.host, compute.compute_args(), DROPPED);

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
            over: 0,
            draws: 2,
            draws_hit: 2,
            reads_hit: 2
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
    trace::dropped_dispatch(&host, compute.compute_args(), DROPPED);

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
