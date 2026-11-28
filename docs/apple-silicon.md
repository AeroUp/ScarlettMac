# Apple Silicon testing guide

The estimates below are for the **native Series S opening**, with a warm shader cache and the 1080p console output profile. World rendering is still incomplete. Frame rates are for frames that reach the window, including frames with visible defects. They are low-confidence planning bands based on one short measurement on this fork. The bands overlap because texture copies, memory pressure, shader translation and queue waits vary across runs. There is no reliable Series X FPS estimate yet.

| System memory | Estimated world FPS today | What to expect |
| --- | ---: | --- |
| 8 GB | No reliable figure | Boot or menu testing. The measured world footprint already exceeds this capacity. |
| 16–18 GB | Under 2, with long stalls possible | Series S bring-up. The 18 GB M3 Pro reached 1.34 FPS over one in-world interval. |
| 24–32 GB | Roughly 0.5–2 | More room for macOS and the runtime. Base chips and fanless Macs may fall toward the bottom of the range. |
| 36–64 GB | Roughly 0.5–3 | More headroom for profiling. CPU submission and graphics correctness still limit runs. |
| 96 GB and up | Roughly 0.5–3 | Useful for long traces and Series X experiments. Sparse measurements leave Ultra scaling unknown. |

## Chip configurations

CPU and GPU figures are core counts. Memory is in Apple's marketed GB. Ranges cover different Mac models and chip bins. Individual Macs ship in specific combinations.

| Generation | Base chip · CPU / GPU · memory | Pro · CPU / GPU · memory | Max · CPU / GPU · memory | Ultra · CPU / GPU · memory |
| --- | --- | --- | --- | --- |
| M1 | 8 / 7–8 · 8–16 GB | 8–10 / 14–16 · 16–32 GB | 10 / 24–32 · 32–64 GB | 20 / 48–64 · 64–128 GB |
| M2 | 8 / 8–10 · 8–24 GB | 10–12 / 16–19 · 16–32 GB | 12 / 30–38 · 32–96 GB | 24 / 60–76 · 64–192 GB |
| M3 | 8 / 8–10 · 8–24 GB | 11–12 / 14–18 · 18–36 GB | 14–16 / 30–40 · 36–128 GB | 28–32 / 60–80 · 96–512 GB |
| M4 | 8–10 / 8–10 · 16–32 GB | 12–14 / 16–20 · 24–64 GB | 14–16 / 32–40 · 36–128 GB | — |
| M5 | 10 / 8–10 · 16–32 GB | 15–18 / 16–20 · 24–64 GB | 18 / 32–40 · 36–128 GB | 30–36 / 64–80 · 96–512 GB |
| M6 | 12 / 12 · 16–32 GB | — | — | — |

Apple specifications for the notebook and desktop configurations. [Mac model comparison](https://www.apple.com/mac/compare/) · [M1 Studio](https://support.apple.com/en-us/111900) · [M3 MacBook Pro](https://support.apple.com/en-us/117736) · [M3 Ultra launch specs](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/) · [M4 iMac](https://support.apple.com/en-us/121556) · [M4 Mac mini](https://support.apple.com/en-ie/121555) · [M5 MacBook Pro](https://support.apple.com/en-us/126318) · [M6 Mac mini](https://support.apple.com/en-us/128108) · [M5 Ultra Studio](https://www.apple.com/mac-studio/specs/)

Start comparisons with the same console profile, cache state and scene. Record the runtime commit, macOS version, Mac model, CPU/GPU core counts, memory, median world frame time, number of world intervals, and whether the run ended early. The launcher defaults to Series S on systems with up to 24 GiB of memory and Series X above that. Pass `--console series-s` on every test machine for like-for-like results.

The Series S target for this project is around **8 GiB of process footprint**, including title memory and graphics allocations. Apple Silicon shares memory across the CPU and GPU, but the runtime can still hold console-format pages alongside decoded Metal textures and render targets. Extra GPU cores and hardware ray tracing do not fix invalid geometry, skipped lighting passes, or CPU-side texture refreshes.
