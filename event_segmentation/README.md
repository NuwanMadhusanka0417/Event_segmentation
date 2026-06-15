# Event-stream instance segmentation (streaming, motion-coherence)

Online (streaming) **instance segmentation of an event-camera stream by motion
coherence**, plus visualization-ready outputs. Every event is assigned an
integer `instance_id` (`0,1,2,…` for moving objects, `-1` for background /
unassigned) so that events from the same independently-moving object share an
id. The scene content is unknown — nothing about object type is hard-coded; the
grouping comes entirely from space-time motion.

```
segment.py     stream -> labeled_events + tracks + colormap + frames + summary
visualize.py   those files -> preview video + trajectory & speed plots
```

---

## 1. Install

```bash
pip install -r requirements.txt
```

Only `numpy` is strictly required for the core. `pandas`+`pyarrow` give Parquet
output (a CSV fallback is used otherwise); `matplotlib`+`Pillow` are used by
`visualize.py` and `--render-frames`.

On this machine the interpreter is `C:\ProgramData\anaconda3\python.exe`
(the bare `python` on PATH is a Microsoft Store stub).

---

## 2. Quick start (acceptance test)

The reference sample is the EED `what_is_background` stream (~847k events,
0.9 s, 190×180 sensor):

```bash
python segment.py ^
  --input "..\Data\EED\what_is_background\events_filtered.txt" ^
  --out out\ --render-frames
python visualize.py --out out\
```

This parses the file, segments it, writes all outputs, and renders
`out/preview.mp4` (or `out/preview.gif` if ffmpeg is missing) plus
`trajectories.png` and `speed_profiles.png`. The run prints the resolved config,
progress every 100k events, the confirmed-instance count, and the runtime.

Any of the bundled streams under `..\Data` work as `--input` — they are all the
same `t x y p` text format (EED, hkust_EMS, DistSurf, EV-IMO). Sensor size is
auto-detected, so the larger-sensor sets work too.

---

## 3. How it works

Events are processed in time order with **bounded** working state:

* a **time-window ring buffer** of recent events (horizon `dt_window`);
* a **spatial hash grid** (cell `grid_cell` px) for O(1) neighbour lookup;
* a small set of **active tracks** — each a running velocity `(vx,vy)`,
  centroid `(cx,cy)`, last-update time and event count.

For each event `e = (t,x,y,p)`:

1. **Insert** into the grid + ring buffer; stale entries (older than
   `t − dt_window`, or overwritten by ring wrap) are evicted lazily.
2. **Graph step** — gather candidates from the 3×3 cells around `(x,y)`, keep the
   `k` nearest under the causal space-time metric
   `d = sqrt(dx² + dy² + (beta·dt)²)` (edges point past → `e`).
3. **Motion estimate** — fit a local velocity `(vx,vy)` by trimmed
   least-squares of `(dx,dy)` against `dt`. Bursts with no temporal spread, or
   implausibly large estimates, are guarded/clamped (`v_max`). With fewer than
   `min_nbrs` neighbours the event waits in a `lag`-second buffer for more
   causal context before it is labeled.
4. **Assignment (motion-coherent connectivity)** — the event joins the track
   that dominates its neighbours, with votes weighted toward velocity-coherent
   tracks (`|v_e − v_track| < vel_tol`); a differently-moving object that merely
   touches `e` therefore cannot capture it. If no neighbour is labeled yet, a
   new candidate track is **spawned**.
5. **Update** — EMA fold of the event into the track (weight `lambda`); a
   candidate is **confirmed** once it reaches `min_track_events`.
6. **Housekeeping** — idle tracks (`track_timeout`) are retired; coherent tracks
   whose predicted centroids and velocities agree are **merged**
   (`merge_px`, `merge_vel_tol`) to fix articulation over-segmentation.
7. **Spatial split** — every `split_every` events a union-find over buffered
   same-label neighbours within `split_px` breaks a label that covers two
   spatially-disconnected blobs into separate instances.

At the end, provisional ids are resolved through the merge union-find, tiny
tracks (`< min_track_lifetime_events`) are dropped to background, and the
survivors are remapped to compact ids `0..M-1` ordered by start time.

**Pluggable descriptor.** The affinity lives behind `MotionDescriptor`
(`describe` + `affinity`); a learnable HD/VSA descriptor can replace it without
touching the pipeline.

---

## 4. Parameters

All are `argparse` flags; the resolved config is printed at startup. Defaults
are tuned to the EED sample.

| flag | default | meaning |
|---|---|---|
| `--input` | `events_filtered.txt` | input `t x y p` text stream |
| `--out` | `out` | output directory |
| `--csv` | off | write `labeled_events.csv` instead of `.parquet` |
| `--render-frames` | off | also write `frames/` PNG slices |
| `--limit` | `0` | process at most N events (0 = all); handy for quick tests |
| `--seed` | `0` | RNG seed (kept deterministic) |
| `--dt-window` | `0.010` s | neighbour time window / buffer horizon |
| `--k` | `12` | neighbours kept per event |
| `--beta` | `2000.0` | s→px time scaling in the metric (1 ms ≈ 2 px) |
| `--grid-cell` | `6` px | spatial hash cell size |
| `--min-nbrs` | `5` | min neighbours before an event is labeled |
| `--lag` | `0.002` s | look-back lag buffer |
| `--gate-px` | `8` px | spatial gate radius |
| `--vel-tol` | `1500` px/s | velocity-agreement tolerance for assignment |
| `--lam` / `--lambda` | `0.15` | EMA weight for a new event |
| `--min-track-events` | `40` | events to confirm a candidate track |
| `--track-timeout` | `0.05` s | kill an idle track |
| `--v-max` | `8000` px/s | clamp on the local velocity estimate |
| `--split-px` | `6` px | spatial-split connectivity radius |
| `--split-every` | `5000` | events between spatial splits |
| `--min-track-lifetime-events` | `200` | drop tiny noise tracks at the end |
| `--merge-vel-tol` | `800` px/s | velocity agreement to merge two tracks |
| `--merge-px` | `6` px | centroid agreement to merge two tracks |
| `--traj-dt` | `0.005` s | trajectory resample step in `tracks.json` |
| `--frame-dt` | `0.02` s | PNG slice duration for `frames/` |
| `--width`, `--height` | auto | sensor size (0 = auto-detect) |

`beta = 2000` means **1 ms of time counts as 2 px of distance** in the
neighbour metric — it sets how strongly time vs. space binds the graph.

---

## 5. Tuning: over- vs. under-segmentation

The three knobs to reach for first:

* **`--vel-tol`** — the motion gate. **Too many instances** (one object split
  into several colours)? *Raise* it (e.g. `2500`) so noisier velocities still
  count as the same motion. **Different objects merged into one**? *Lower* it
  (e.g. `800`) so only tightly-agreeing motion groups together.
* **`--gate-px` / `--merge-px`** — spatial scale. Raise `--merge-px`
  (e.g. `12`) to consolidate fragments of one object that sit close together;
  keep it small to hold neighbouring objects apart.
* **`--dt-window`** — temporal context. Larger (e.g. `0.02`) gives each event
  more neighbours → smoother velocities and fewer spurious spawns (fewer
  instances), at more compute per event. Smaller reacts faster but fragments more.

Other useful moves:

* **Fewer, larger instances:** raise `--min-track-lifetime-events`
  (e.g. `1000`) to drop small fragments into background, and/or raise
  `--merge-px`.
* **Fast objects fragmenting:** raise `--grid-cell` (e.g. `8`) and/or
  `--dt-window` so the leading edge still finds its own recent events.
* **Background leaking in:** raise `--min-nbrs` (e.g. `8`) so sparse events stay
  `-1`.

The streams under `..\Data` vary a lot (clean single object vs. dense
multi-motion scenes), so expect to adjust `--vel-tol` and
`--min-track-lifetime-events` per scene.

---

## 6. Outputs (written to `--out`)

1. **`labeled_events.parquet`** — columns `t:float64, x:int16, y:int16, p:int8,
   instance_id:int32`, original order preserved. `--csv` writes
   `labeled_events.csv` instead.
2. **`tracks.json`** — one entry per instance:
   `{id, color_rgb, t_start, t_end, n_events, mean_speed_px_s,
   trajectory:[{t,cx,cy,vx,vy}]}` resampled at `traj_dt`.
3. **`colormap.json`** — `instance_id → [r,g,b]`, stable & colourblind-friendly
   (Okabe–Ito / Tol); `-1 → [110,110,110]` gray.
4. **`frames/`** — (with `--render-frames`) one PNG per `frame_dt` slice, events
   drawn at their pixel in their instance colour on black. Assembles into video.
5. **`summary.json`** — resolved params, total events, confirmed-instance count,
   wall-clock runtime, throughput (events/s), sensor size, assigned vs.
   background counts.

### `visualize.py`

```bash
python visualize.py --out out\ --fps 25 --frame-dt 0.02 --max-time 0
```

Produces, from the files above:

* **`preview.mp4`** — `FuncAnimation` replay of events coloured by instance over
  time (falls back to **`preview.gif`** when ffmpeg is unavailable);
* **`trajectories.png`** — centroid path per instance (image coordinates);
* **`speed_profiles.png`** — per-instance speed vs. time.

`--max-time` caps the animation to the first N seconds (0 = whole stream).

---

## 7. Engineering notes

* **Streaming & memory-bounded.** The input is read line-by-line; a small
  look-ahead heap repairs minor timestamp disorder. The ring buffer + grid +
  active tracks are the only working state — their size is bounded by the events
  inside `dt_window`, not by the stream length. The per-event labels and raw
  columns are accumulated because they *are* the output; for an unbounded stream
  you would flush these to disk in chunks (the algorithm state stays bounded).
* **Determinism.** Given the same `--seed` and inputs the result is reproducible.
* **Defensive parsing.** Malformed lines are skipped; timestamps are sorted
  within a small look-ahead window so the core loop always sees non-decreasing
  time.
* **Throughput.** The pure-Python causal core processes the 847k-event sample in
  roughly two minutes on a laptop (≈6k events/s; it logs progress every 100k).
  For interactive iteration use `--limit` to segment a prefix. Raising
  `--vel-tol` / fixing velocities reduces the active-track count, which is the
  main cost driver.
