# Training-free VSA grouping: results

Branch `VSA_flow_NOCNN`. Module `hdems/vsa_grouping.py`; eval `--vsa-group`; baseline
`--threshold-baseline`; tuning `scripts/tune_vsa_grouping.py`; PBS `MODE=vsa_group`.

## 1. Unit tests on synthetic flow fields (`tests/test_vsa_grouping.py`, all pass)

| Test | Result |
|---|---|
| encoding kernel ≈ exp(-Δpos²/2σs²)·exp(-Δvel²/2σv²) (d = 4096) | within 0.05 |
| one object on a static background | 1 object, IoU > 0.9 |
| two objects, different velocities | 2 objects, IoU > 0.85 each |
| rotating disc (0.05 rad / interval) | 1 object, IoU > 0.85 |
| moving-camera background (affine field, 1.0–2.3 px) + one object | background = 1 cluster (from > 3), object IoU > 0.85 |
| large mover (60 % of the image) with the temporal prior | prior picks the true background; `extent` picks the mover (wrong) |
| prior per sequence, expires after 5 frames | ✓ |
| hull IoU of sparse events of a box vs the dense box | > 0.9 |
| a mixed object + background cluster is split; a smooth camera field is not | ✓ |
| determinism with a fixed seed | ✓ |

Three design corrections came out of these tests:
- **Split mixed clusters after k-means.** On a synthetic moving-camera frame at
  ratio-2 size, k-means left clusters that were 41 % and 66 % object. They bridged the
  object into the background, so with a border gap the object vanished (0 objects).
  After the split: object IoU 0.985 with and without the gap.
- **Merge by LOCAL affine fits compared at the same point.** One median velocity per
  side of a border failed on rotation. Local means at a few px distance failed too:
  0.23–0.27 px jump inside a noiseless rotating disc, because the flow changes with
  position.
- **"Moves like the background" by per-pixel magnitudes.** A rotating object's median
  velocity vector is ~0, so a vector test absorbed it into the background.

## 2. Synthetic EVIMO2-format scenes (`tools/synthetic/`)

**Not run.** The harness `hdems_review_tests.zip` is not in the project folder.
Provide it to run section 6 of the task.

## 3. Real data: does the continuity merge separate background from objects?

Validation scene (scene 9, `holdout`), 8 frames with a visible mover, ratio 2, before
tuning. For every pair of neighbouring k-means clusters, the continuity jump (full-res
px per 12.5 ms), labelled by ground truth (cluster = mover if > 50 % of its pixels are
moving). A good merge threshold merges most background/background pairs and few
object/background pairs.

| Mixed-cluster split | Flow smoothing (Eq. 12) | Border gap | bg/bg jump median | obj/bg jump median | at 0.5 px: bg/bg merged vs obj/bg merged | at 1.0 px |
|---|---|---|---|---|---|---|
| no | 71 (default) | 0 | 0.29 | 0.44 | 70 % vs **57 %** | 87 % vs 82 % |
| no | 71 (default) | 36 | 1.64 | 3.16 | 7 % vs 0 % | 30 % vs **6 %** |
| no | 17 | 0 | 0.60 | 0.81 | 38 % vs 20 % | 82 % vs 68 % |
| **yes** | 71 (default) | 0 | 0.33 | 0.54 | 64 % vs **48 %** | 81 % vs 68 % |
| **yes** | 71 (default) | 36 | 2.68 | 5.98 | 5 % vs 0 % | 21 % vs **1.3 %** |

End result per frame (the true answer is 1 object):
- **no gap:** 0 objects before the split fix (everything merges into 1 cluster), 0–4
  objects after it;
- **gap 36:** the background stays in pieces, so 10–22 "objects" before the fix and
  21–46 after it;
- **smoothing 17, no gap (before the fix):** 0–12 objects.

The border gap makes object borders almost never merge, but then the background's own
pieces don't merge either. No single threshold gives one background and separate
objects.

**Why:** in this flow the background is not one smooth motion field. There are
surfaces at different depths (parallax), flow errors on low-texture areas, and untracked
moving things (e.g. a hand) labelled as background. The 71 px Eq. 12 pooling also
blurs each object's motion ~35 px into the background, which hides object borders
unless a gap is skipped.

## 4. EVIMO2 comparison (eval split, once)

To be filled from the Gadi run (`MODE=vsa_group`, then `MODE=threshold`, plus the
trained `mfunet` job). Same scoring mask, ratio 2:

| Method | FG IoU | Hull IoU | Object mIoU | P / R | Objects/frame | Static false-object rate | ms/frame |
|---|---|---|---|---|---|---|---|
| VSA grouping (tuned on scene 9) | | | | | | | |
| Threshold baseline (tuned on scene 9) | | | | | | | |
| Events-only control | | | | | | | |
| `mfunet` (trained) | | | | | | | |

## 5. Parameters

Defaults: `configs/evimo_seg.yaml` → `vsa_grouping:`. Tuned values:
`results/<run>/vsa_grouping_tuned.yaml` (+ `.txt` with the full grid), written by
the PBS `TUNE=yes` step.

## 6. Known failure cases

- **Large moving objects:** handled by the temporal prior only if an earlier frame of
  the same sequence had a correct background.
- **Background with parallax / depth steps:** splits into fragments (section 3).
- **Sub-pixel movers:** α = 0.85 sets their flow to 0, so they look static.
- **Untracked moving things** (hands) are labelled static in the ground truth.
