# Runtime preview baseline, 26 November 2026

Measured on the development Mac (Apple M3 Pro, 18 GB, 4K display) with the
November 26 checkpoint runtime, Wine 11 and Apple D3DMetal. One title instance
at a time, always through `tools/scarlett-run.sh`, which takes a lock and sends
the offline picker's confirmation from a file marker.

## Runs

| Run | Log | Seconds | Presents | Mean fps | Median segment fps |
| --- | --- | --- | --- | --- | --- |
| A | `runs/20261126-123001.log` | 44 | 256 | 8.66 | 30.00 |
| B | `runs/20261126-124043.log` | 176 | 2048 | 19.41 | 24.79 |

Run B is the current baseline: `--seconds 180 --tag baseline`.

```sh
tools/scarlett-run.sh --seconds 180 --tag baseline
python3 tools/fps-report.py runs/<stamp>-baseline.log
```

## What the baseline shows

- The title presents through the opening and the illustrated loading artwork
  only. No 3D world and no gameplay frame was captured in either run.
- The 30 fps cap holds in the menus (A measured 30.00 fps over six segments).
  Run B spends most of its time between 23 and 26 fps, below the cap, because
  the host, not the cap, is the limit.
- Presents are logged only when the frame counter reaches a power of two, so a
  missing present line is not proof that presents stopped. The frame counter
  reached 3273 in run B, 65 s after the last present line; the next present line
  would have been frame 4096.
- Title-memory refresh is the largest CPU cost: 3741 submissions in 176.5 s,
  9287 ranges, 635 GB copied, 169.8 MB per submission, 3.6 GB/s. Fault tracking
  is off, so a whole copy is refreshed rather than the pages the title wrote.
- Dropped work is counted, not inferred: 81 shader translation refusals
  (`shader does not translate`), 4 black frames, 1 unordered view the host
  cannot store, 1 ignored depth-bounds test, 4 slow host submissions.

## Comparing runs

`tools/fps-report.py --compare BASELINE LOG` prints one number per quantity for
both runs with the change, including the counts of dropped work above. A change
counts as a win only when the quantity it targets moves and the presented image
does not regress.

## What this baseline does not establish

Sustained 3D rendering, walking, camera control and gameplay remain unverified
on this track. Earlier experimental runtimes on an M3 Max with CrossOver
Preview reached 3D frames; this runtime-preview track has not.
