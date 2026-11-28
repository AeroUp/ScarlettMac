# Roadmap and open problems

Updated 27 November 2026. Each problem lists what is already known, so you don't have to rediscover it. The deep detail, with addresses and every diagnostic switch, is in [native-host.md](native-host.md).

**Where it stands:** the game boots, plays its boot movie, shows the start screen and every setup screen, loads story mode and reaches the prologue. It crashes a few seconds into the prologue. Getting the prologue to play through is the top priority.

## 1. The prologue crash (top priority, [#15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15))

Two problems stop it, in this order:

1. **A stall on the first world frame trips the game's own watchdog.**
   - At the first world frame, the game binds about 1,100 pipelines for the first time.
   - About 69 of them are large compute shaders. Each took 0.2–0.6 s to translate, almost all of it in structurisation (`range::build`), and is then refused. Each was done in turn on the game's thread.
   - The game's watchdog fires 30 s into a stalled frame (its fallback when no watchdog scope is active; a scope entered and not left fails after 300 s): a deliberate null write at `image+0x23cf683`, then the fault at `image+0x55e56f8`.
   - **Fixed on branch `fix/prologue-crash`:** every shader is translated ahead on host worker threads as the game creates it (`VICTORY_NO_SHADER_PREWARM=1` turns it off). Being checked for memory cost before it lands.
   - The frame budget was only checked *before* a translation, never during one: a translation started inside the first 10 s of a frame still held it for as long as it took. Four took 47–52 s in the run of 27 November 17:44, and the watchdog fired 30 s after the frame's last bind (the run's crash report names an `Unsignaled CrossQueueBarrier`; the queues' heads wait for the next frame's barrier value the frame never wrote). Title threads now start no translation at all while the prewarm pool is on: the pipeline (or its rectangle-list, extra-parameter, integer-target or thread-count form) is put off and its work dropped until the pool has it. The BVH builder's programs stay inline.
   - The same budget was still handed to *compiles*: a title's draw or dispatch that first needed a pipeline waited up to the frame's remaining 10 s for Metal, on the title's thread and under the host's lock, so the pump that hands held work to the GPU could not run for that long. The run of 27 November 21:32 shows the shape: `work held for a wait that does not hold` with `0x18437f9d8 cmp 7 ref 0x14a7 ... now Some(5286)` (the next frame's barrier, as above), heads held 20+ s, `signal of 30 queued ... GPU at 29`, no slow command buffer, and `metal: compute pipeline compiled: 28993 ms` in the same frames. A title call now waits only the 20 ms a pipeline the disk cache and the archive hold needs (`glue::compile_wait`); a pipeline not made then is put off and a later draw picks it up. Tests: `native::glue::a_frame_never_waits_out_its_budget_for_a_compile`, `hold::a_wait_for_the_next_frames_barrier_is_released_by_its_writer`.
   - **A `PresentX` that waited for a drawable held the queue** (run of 27 November 22:08, `87388f80`): frames of 6 s with the whole process at 0.28 CPU cores and the GPU at 22%, and the warning naming a queue head with `waits []` whose fence signal was queued behind it 7.4 s. The head was a present: the display had stopped handing out drawables, and our FIFO let that stop the queue's later work, including the frame's fence signal and its cross-queue barrier writes, so every other queue waited on its barrier. A present now waits 500 ms at most, is dropped past that (and at once while the display is starved), and the warning names the head when it is not a segment. Tests: `title::scheduling_tests::a_present_waits_for_a_drawable_only_its_window`, `hold::tests::a_head_that_is_no_segment_is_named_with_its_token`.
   - A slowdown in queue submission after merging #76 (resources named at every draw) is being fixed on `fix/fps-decay`.
2. **Pointers read from data no shader filled in.**
   - Once in the world, the game reads arrays of *empty* half-float bounding boxes as pointers: `0xfc007c007c007c00`, `0xfc00fc00fc007c00`, or the float32 form `0x7f8000007f800000`. It then reads null+0x10 at `image+0x5592694`.
   - In both runs that showed it, a Metal assertion came just before: "bytes_per_row >= used_bytes_per_row". It followed the first use of a compute shader refused for "a cross-lane operation between host waves in divergent code".
   - **Leading guess (not proven):** a refused shader's output buffer keeps its initial empty boxes, and the CPU reads it.
   - `VICTORY_TRACE_OWNERS=1` names who owns the memory, and `VICTORY_DUMP_REFUSED=<dir>` dumps the refused shaders.
   - **Ruled out:** linear root views (those shaders go through the page table), pool pages owned twice (fixed by #108; a pool audit finds 0 differences), and GPU command buffers that fail (their signals are now paid on the CPU).

**Start here:** `crates/victory-gpu/src/native/shaders.rs`, `crates/victory-gpu/src/pipeline.rs`, `crates/victory-metal/src/gfx.rs`, and the run-game skill. `tools/macos/run-summary.sh <log>` shows how a run ended. `tools/macos/tester-report.sh` turns a log into a scrubbed attachment and a tester-report body.

## 2. Frame rate and memory in the world

- In the world, fps falls from about 15 to about 7 over 4 minutes. Something grows every frame; host texture copies are the main suspect (one tester counted 26,000 by frame 4096). Being investigated on `fix/fps-decay`.
- Series X can run out of GPU memory at the world load, even on 24 GB ([#8](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/8), [#79](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/79), [#110](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/110)). A byte budget on host texture copies ([#294](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/294); `VICTORY_METAL_TEXTURE_BUDGET_MB`, off unless set) gives up copies nothing uses past a byte limit; whether that frees enough on 16 GB is not measured yet. On a 24 GB M4 Pro the world load stopped with the budget off because copies overlap: 8.5 GB of copies over 1.7 GB of distinct title memory (#301). Series S is the default on Macs with 24 GB or less. The native host now logs Metal's own recommended working set size alongside the title's declared budget at startup, so a tester's log carries the numbers needed to set a real threshold.
- The game rewrites about 40,000 descriptor records a frame with plain stores. Patching its store sites instead of remirroring is being measured ([#43](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/43); `VICTORY_TRACE_DESCRIPTORS=sites`).
- Textures the GPU writes and the CPU reads back (the water simulation) are copied back behind `VICTORY_TEXTURE_BACK=1`, off by default ([#46](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/46)). A world run on an M5 Pro names 22 textures at no measurable cost, then crashes before boats can be seen. Needed: a run past that crash.
- The screen reflections' history is a title texture written as R32_UINT (render target and unordered access) and read through R9G9B9E5 placed views. An automatic layout resolves to `SW_64KB_R_X` for the written texture but `SW_4KB_S` for the views. Such an R9G9B9E5 view now names the written texture whatever the two layouts are (`alias_rgb9e5`), so the tiled lighting and the denoiser read what compute wrote; `VICTORY_PLACED_TWIN=1` (off unless set) extends the sharing to other formats. Unmeasured in a run.
- **The process outgrew the machine, and now says what it is (runs of 27 November 22:23 and 22:50).**
  The first died paged out: 14.9 GB footprint on the 18 GB Mac, 1.16 GB resident, frames of 7.8 s
  with the CPU and the GPU at 0%. The memory line of the second run closes the accounting: title
  committed 7,607 MiB (4,712 CPU, 2,895 GRAPHICS, 12 pools, capped at the console's 7,776), host
  texture copies 4,489 MiB of a 3,072 budget, Metal 12,034 MiB (the pools it wraps plus the copies),
  DXIL 832 MiB, footprint 12,996 MiB -- so there is no hidden leak, and the copies are the host's
  own addition. Two cuts so far: a sampled descriptor no longer realizes a copy under a spent budget
  (`realize_sampled`; 5.4 -> 4.5 GB of copies), and the budget's second tier takes a GPU-written
  copy no submission names (it waited for a presented frame, which in a world presenting every 7.8 s
  never came: 2.1 GB written, 39 MiB evictable).
  - The 832 MiB of DXIL went with the prewarm change (only a shader an earlier run bound is
    translated ahead): the run of 23:25 read `dxil_mib=128`.
  - `metal_own` (Metal's allocations beyond the pools it wraps and the copies) and `other` (the
    rest of the footprint) now have parts in the line: `twins_mib` (depth twins, one extra host
    texture per depth a compute pass writes, given up after `TWIN_IDLE` submissions without a
    write) and `containers_mib` (the copies the title's `Shader`s keep of their DXIL containers,
    ~322 MiB: vertex and pixel keep them because a rectangle-list, extra-parameter or
    integer-target variant is translated from them, and the title may have freed the container by
    then; compute shaders let theirs go after translation). What is left of `other` after the
    containers is the x86-64 title's Rosetta JIT and translation caches (~240 MB in the vmmap of
    the run of 23:00), the app's AppKit/window state, and host bookkeeping (heap entry maps, the
    pool table's ~97,000 mappings, texture records). Next: a run with `twins_mib`,
    `containers_mib` and `title_heap_mib` decides which of those to take.
  - Untested after reboot: the depth-twin sweep (`TWIN_IDLE`), the `twins_bytes`/`container_bytes`
    accounting, and MetalFX's opt-in (`VICTORY_METALFX=1`) were committed while the Mac was wedged
    (processes stuck in state U), so no run or test has seen them.

## 3. Shader translation gaps ([#4](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/4))

Refused shaders drop their draws and dispatches. Unique refused shaders in the story load fell from 526 to about 150 on 27 November, and more fixes have landed since. Refusals now name their exact cause. The largest groups left:
- a descriptor loaded from title memory at an address in a root constant (bindless; needs the host to map the address to a heap index);
- a lane-mask join of an untracked mask: mostly closed on 27 November (EXEC-skip edges carry no lanes, a join read as data is a join of its edges' numbers, 64-bit scalar pairs in 32-lane programs, EXEC phis at mask joins; `docs/native-host.md`, "Lane masks joined with data"). What remains is masks a VALU compare wrote and XOR swaps move as data, and six vertex or pixel programs joining arbitrary data on the host's lanes;
- cross-lane operations and LDS shared between the two host waves of a console wave, inside real per-lane branches (needs if-conversion so both host waves always arrive);
- a whole-wave section step the simulation does not follow.

Fixed on 27 November: compute root constants, sized from the root signature's `Num32BitValues` (#114); `v_pk_fmac_f16` constants; wave64 lane-mask joins; `s_absdiff_i32`; signed BC6H; ExecuteIndirect with a count buffer. A batch on `fix/shader-refusals-2` also covers samplers built from constant bit masks, the thread-group info SGPR, and LDS behind uniform branches; it is being merged.

Since then: image stores of some channels (#94), whole-wave float products (`WaveActiveProduct`, `v_mul_f32`), and wave64 cross-half exchanges in loops every host wave runs alike (`in_step`, `halves::branch_regions`; the ray-tracing builders' shape). Also descriptor dwords the compiler kept in VGPR lanes (`v_writelane_b32`/`v_readlane_b32`), which were "neither loaded nor built", and the last three dwords of an image descriptor read as data (the view's last mip level and no compression metadata).

Translated shaders, pipelines and refusals are cached on disk between runs.

**Start here:** `crates/victory-shader/src/emit/`. Tests use hand-assembled GFX10 code (`gfx_refusal_tests.rs`), so a refusal shape can be fixed without the game.

## 4. Running without Rosetta (arm64 recompilation)

The whole game executable (about 440,000 functions) lifts to C in seconds and compiles to arm64 (about 80 minutes at 6 jobs). A contributor's branch ([#115](https://github.com/LakshanSivakumar/Native-Victory-macOS/pull/115)) boots it to the intro at 30 fps. It is being tested here, and it is 85 commits behind main.

## 5. Everything after that

- Reverb: the audio DSP's convolution has a synthetic CPU fallback behind `VICTORY_GUESS_DSP=1` (a fixed decaying-noise tail, not the title's own impulse response); default behaviour is unchanged (silence) until a real run judges whether it sounds right ([#17](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/17)). The hardware's impulse-response layout is still undocumented; `VICTORY_TRACE_DSP=1` logs the title's own `ir` data to help settle it.
- The Brightness logo draws as half a rectangle ([#16](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/16)).
- Mesh-shader pipelines: direct and indirect mesh dispatches run on the native host (#353 for the direct ones), under a guessed record layout and a guessed `DispatchMeshX` group count.
- Culling, depth, stencil and write mask in guessed pipelines are still partly guessed.

## Solved recently

- **26–27 November:**
  - The Metal GPU timeout at about 53 s ([#2](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/2)): waits stay on the CPU until the GPU work is observed done.
  - Failed command buffers now still signal their fences ([#80](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/80)).
  - The black window on M1/M2 ([#26](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/26), #91).
  - The boot movie's green frames and stutter.
  - Pool pages owned twice after `VirtualFree` (#108); physical page numbers reused (#124); decommit keeps page views (#126).
  - World geometry's shards and streaks: stage-in vertex and index buffers (and blits) across title pages mapped out of pool order resolved only their first page; ranges now wrap the title's own pages (#442; `docs/native-host.md`, "World geometry drawn as shards").
  - The remaining shards through the page table: an access of more than one dword looked a page up once at its first byte, so a vertex position, a skinning matrix or an instance transform that straddled a 64 KiB title page read the pool past that page. Each dword now reads and writes through the page that holds it (`docs/native-host.md`, "An access whose dwords cross a page").
  - Two more cases of the same family, found by reading the code: a page the title reports one mapping at a time was a null page in the page table (now one run of mappings shows it), and a described pipeline's stage-in stream stepped by its layout's elements' end rather than the bound vertex view's stride (now the longer of the two). A trace of the world (`VICTORY_TRACE_PAGES`, `VICTORY_TRACE_STRIDES`) says neither case occurs in today's world: every page is readable and no stream pads. What the world still lacks is dropped draws for refused pipelines (`docs/native-host.md`, "World geometry drawn as shards", last bullet).
- **Earlier:**
  - Black content draws.
  - The present pass.
  - Boxed glyphs (#6).
  - The "press A" start screen, now with keyboard and controller input.
  - The story-load deadlock.

## Good first contributions

- Run the game on your Mac and file a tester report; different chips and RAM sizes help, especially runs that reach the prologue.
- Pick a refused shader shape from section 3. It's self-contained and testable without the game.
- Turn a finding in native-host.md into a unit test.
- Improve `tools/macos/native-setup.sh` and `run-native.sh` for setups we haven't seen.
