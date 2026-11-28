# Native Series S status · November 28, 2026

## Measured opening

A windowed run of native revision `462ea4e2` on an **M3 Pro with 18 GiB of unified memory** reached the opening. The capture showed a sky and character silhouette with dark surfaces and stretched geometry. The run was observed for 10.6 minutes without a crash line. The log contains **one** qualifying world interval, which averaged **745 ms per frame (1.34 FPS)**. That interval is too short to establish a sustained rate.

At the last memory sample, the process footprint was **11.48 GiB**. The run report attributed **7.43 GiB** to title memory and **2.89 GiB** to host texture copies. The footprint includes other categories and already reflects the host's accounting. The two component figures describe allocations and should not be added to the footprint.

Series S was configured to advertise 1080p output. Traces identify 1280×720 scene targets as well as larger auxiliary surfaces. The profile can use MetalFX for upscale. Render-target dimensions vary by pass.

## Included in the source snapshot

| Change | Evidence |
| --- | --- |
| Split-wave implementation of `ds_append` and `ds_consume` | The native branch records successful Metal pipeline creation for three affected compute shaders. World lighting still needs an end-to-end run on this revision. |
| Per-submission budget for changed textures | The scanner-driven refresh path is capped at 384 MiB per submission and can defer excess work. The opening measurement above predates this cap. |
| Index and stream capacity diagnostics | `VICTORY_TRACE_INDEX` records index data and vertex stream capacities for selected draws to investigate stretched triangles. A geometry fix is still open. |
| Shadow depth and residency work | Earlier traces showed geometry in shadow maps. The lighting pass and final presentation still need verification in the same run. |

Snapshot source revision `73e75045b2b26279bdeb1af34094262721f1500b`. See the [native host notes](../native-runtime/docs/native-host.md) and [roadmap](../native-runtime/docs/ROADMAP.md).
