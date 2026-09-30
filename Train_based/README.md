# HD-EMS

**Hyperdimensional Event Motion Segmentation** — a VSA/VFA (Vector-Symbolic /
Vector-Function Architecture) pipeline for event-camera **motion segmentation**,
built on the VSA-Flow feature-matching method (You et al. 2025) with an added
ego-motion + segmentation stage.

## Pipeline

```
events → multi-time surfaces (F0,F1,F2,F4)
       → VFA HD descriptors        F = T ∗ K   (paper Eq. 4/6, FPE over positions)
       → two-time multi-scale cost volume       (paper Eq. 10–11)
       → probability-volume flow estimator      (paper Eq. 12–14)
       → ego-motion residual velocity           (robust affine, IRLS)
       → residual velocity → FPE hypervector, combined with the event field Φ
       → classifier → per-pixel object classes
```

The whole VSA front-end (kernel, descriptors, cost volume, flow estimator,
ego-fit) is **parameter-free**. Only the (optional) CNN heads train.

## Labels: what "moving" means

EVIMO2 masks annotate **every tracked surface** — the table and the static props
included — so `mask > 0` means "on a tracked object", not "moving". Measured on
`samsung_mono/imo`: `mask > 0` covers 61% (train) / 72% (eval) of pixels, while
objects that actually move cover 0.9% / 4.1%.

`label_mode: motion` therefore derives labels from the **pose metadata**: each
object's camera-frame pose is composed with the camera pose to get its WORLD
pose, and per frame its speed is converted to image-plane displacement
(`fx·v/Z·Δt`, plus a rotation term so spin-in-place counts):

| displacement per window | label |
|---|---|
| ≥ `motion_label.move_px` (1.0) | moving (1) |
| ≤ `motion_label.static_px` (0.3) | static (0) |
| in between | ignore (255) |
| within `boundary_ignore_px` of a mask edge | ignore (255) |

Ignore pixels are excluded from the loss, from the ridge/prototype fit and from
every metric. `label_mode: tracked` keeps the old `mask > 0` rule as a baseline.

**Report foreground IoU, not mIoU** — moving pixels are a few percent of event
pixels, so mIoU stays near 0.5 for a model that predicts "static" everywhere.
`hdems.eval` prints FG IoU as the headline.

## Gates before you train

Both need no GPU and no training. If gate 1 fails, a retrain tells you nothing.

```bash
# Gate 1 (signal) + Gate 2 (events-only control), writes four-panel figures
python scripts/diagnose_motion.py --config configs/evimo_seg.yaml \
  --split train --frames 6 --d 256 --out diagnostics

# Gate 2 on the full eval set, through the normal eval path
python -m hdems.eval --config configs/evimo_seg.yaml --head cnn \
  --checkpoint <ckpt> --events-only-baseline
```

- Gate 1 passes when AUC(|residual|) ≥ 0.6: the ego-compensated flow is larger on
  moving objects than on static ones.
- Gate 2 passes when the events-only foreground IoU is poor (≲ 0.35): the task is
  no longer solvable by "there are events here".

## Heads (`segmentation.head`, or `--head`)

With `dataset.time_frames` set, all heads use the paper front-end.

- **`mfcnn`** (default) — motion-first CNN. Inputs: a 1×1 readout of the velocity
  code `Mv` (fixed units), 6 explicit motion channels (`rx, ry, |r|` in px per
  12.5 ms, `|r|` divided by the frame's background noise, the raw `|flow|` before ego
  compensation, and the camera's own speed in this frame), and only 8 channels
  of appearance (1×1 projection of unit-RMS `X`), which are zeroed for a whole
  sample with probability 0.5 while training. `X` and `Mv` are not fused, so
  `event_combine` is not used.
- **`mfunet`** — the same inputs as `mfcnn` plus two evidence channels, in a small
  U-Net (1, ½, ¼, ⅛ resolution, ~1M parameters) that sees 109 working px around each
  pixel (measured) instead of 5×5. A moving object is recognised by its motion *differing from
  its surroundings*, and objects are 50–200 px wide, so the head must see both.
  - event density: log of the decayed event count of the label-time surface — where
    the evidence is (dropped together with appearance while training);
  - flow confidence: how concentrated the Eq. 12 probability volume is (Σ P²,
    `flow_from_cost(..., return_confidence=True)`, cached with the flow). Measured
    against GT flow it is only a weak predictor of flow error (Spearman +0.27, the
    best of four cost-volume measures), so the head may learn to ignore it.
  No absolute pixel coordinates: they would let the head learn where objects usually
  are in the training scenes.
- **`cnn`** — HV-channels CNN on the fused `X`/`Mv` feature (`event_combine`).
- **`prototype`** — VSA nearest-centroid; class prototypes are the
  normalized bundle of training features. **Parameter-free**, no backprop.
- **`ridge`** — closed-form linear readout (least squares). No backprop.
- **`motion`** — older motion-primary CNN (residual + magnitude + 32 Φ channels).

All heads are fit/trained with `scripts/fit_ridge_head.py` (what the PBS runs).

**Augmentation** (`train.augment_flip`, `--augment yes|no`, PBS `AUGMENT`): each
training sample is randomly kept, mirrored left-right, upside-down or rotated 180°.
The event surfaces are flipped *before* the front end, so the flow is re-measured on
the mirrored scene and the velocity mirrors with the image (a left-right flip negates
vx). The flow cache then holds up to 4 versions of a frame. Validation and eval are
never augmented.

### Does the head use motion? (`--ablation-check`)

Measured 2026-09-29 on real eval frames: in the `cnn` head, zeroing the velocity input
changed only 1.3% (Φ + concat) to 4.6% (F + bundle) of the predictions, while zeroing
appearance changed 15–25%. It had learned what moving objects LOOK like in the
training scenes, which does not transfer to new scenes. So every eval can now check it:

```bash
python -m hdems.eval --checkpoint <ckpt> --ablation-check
```

prints the FG IoU with and without each input, and a verdict (`USES MOTION` when
removing motion costs ≥ 50% of the FG IoU, `IGNORES MOTION` below 10%). Accept a
head only if it uses motion. `run_segmentation.pbs` does this with `ABLATION=yes`.

### Phase-1 fixes (2026-09-29, from the code review + the ablation)

| Setting | New | Old | Why |
|---|---|---|---|
| `dataset.time_frames` | `[1.0, 0.75, 0.5, 0.0]` | `[0.0, 0.25, 0.5, 1.0]` | the reference surface now ends at the label time (it ended 50 ms before) |
| `dataset.score_window_ms` | `12.5` | all events | train/score only on events near the label time, not on the trail |
| `dataset.val_split` | `holdout` (`val_scenes: [scene9]`) | `eval` | the best epoch was chosen on the test set |
| `matching.phi_window` / `phi_pad` | `7` / `zero` | `M` (31) / wrap | Φ bundled 961 terms in d=512 and wrapped around the borders |
| `velocity.vel_norm` | `fixed` (`vel_unit_px 0.5`) | per-frame 95th percentile | the same code must mean the same speed in every frame |
| `velocity.x_norm` | `rms` | none | X was 7–370× larger than the velocity code |
| `velocity.ego_fit` | `stack` (option: `reference`) | `stack` | tested, `reference` was no better |

Measured on 48 eval frames with a mover (ratio 2, no training), AUC of moving vs
static on the scored events:

| | forward (old) | reversed (new) |
|---|---|---|
| raw \|flow\| | 0.683 | **0.738** |
| \|residual\| after ego compensation | 0.704 | 0.700 |

The reversed order gives better flow; the ego fit is now the limit. On still-camera
frames the ego fit can lock onto a large mover and wreck the residual, while on
moving-camera frames the raw flow is useless. So `mfcnn` gets both, plus the
camera's speed, and learns which to trust.

Checkpoints record these settings, and older checkpoints are evaluated with the old
behaviour automatically (`hdems/config.py: _LEGACY_FRONTEND`). `.pt` shards are now
ignored unless `dataset.use_shards: true` (they bypass the split and mover filters).

## Velocity → event combination (selectable)

The residual velocity is FPE-encoded (`Vx`, `Vy`) into a velocity hypervector `Mv`
and fused with an event hypervector `X`. Three choices, set in the config **or** on
the command line:

| Option | Values | Meaning |
|---|---|---|
| `velocity.event_feature` / `--event-feature` | `phi` · `f` | event HV `X`: `phi` = 7×7 bundled neighbourhood field Φ (default), `f` = VFA descriptor F0 (paper Eq.4) |
| `velocity.axis_combine` / `--axis-combine` | `bind` · `bundle` | combine `Vx, Vy` → `Mv` |
| `velocity.event_combine` / `--event-combine` | `bind` · `bundle` · `bindbundle` · `concat` | combine `X` with `Mv` |

`event_combine`:
- `bind` → `X ⊙ Mv`
- `bundle` → `X + Mv`
- `bindbundle` → `(X̂ ⊙ Mv) + Mv`, where `X̂ = X / RMS(X)` per frame. The rescale is
  needed because `|Mv| = 1` while F is ~5–11× and Φ ~100–350× larger on event pixels,
  so an unscaled bundle would drown the velocity term. (With `velocity.x_norm: rms`,
  now the default, every mode gets the unit-RMS `X`.)
- `concat` → `[X | Mv]`

Feature dim: `4d` for `concat`, `2d` for the others. Old checkpoints without
`event_feature` load as `phi`.

## Layout

```
Train_based/
├── hdems/
│   ├── data/            events → accumulative time surfaces, EVIMO2 reader/dataset
│   │   ├── time_surface.py      vectorized accumulative TS
│   │   ├── evimo2_reader.py     raw NPZ/NPY reader; multi-time surface builder
│   │   └── evimo.py             EVIMODataset (raw or cached .pt shards)
│   ├── vsa/
│   │   ├── fpe.py               FPE, HRR bind/bundle/similarity
│   │   ├── velocity.py          residual-velocity hypervector + combine (bind/bundle/bindbundle/concat)
│   │   └── kernel.py, temporal.py
│   ├── models/
│   │   ├── encoder.py           paper VFA kernel  F = T ∗ K  (FPE over positions)
│   │   ├── matching.py          bundled matching field (Φ context)
│   │   ├── paper_flow.py        two-time multi-scale cost volume + Eq.12–14 flow
│   │   ├── motion.py            ego-motion residual (+ single-frame decode)
│   │   ├── prototype_head.py    VSA nearest-centroid head
│   │   ├── segmentation.py      MotionSegHead / SegmentationHead (CNN heads)
│   │   └── hdems.py             HDEMS assembly + forward + paper_features
│   ├── ridge_head.py / ridge_fit.py / seg_features.py / feature_extract.py
│   ├── losses/          seg (CE + Dice), flow (EPE)
│   ├── train.py         trains the CNN heads (motion/cnn); saves last.pt/best.pt
│   └── eval.py          mIoU + events|GT|prediction PNG panels
├── scripts/
│   ├── nci_env.sh       redirect torch/matplotlib caches to scratch (NCI quota)
│   ├── prepare_evimo.py precompute (surface, mask) → .pt shards
│   └── fit_ridge_head.py  fit prototype/ridge readouts
├── configs/            evimo_seg.yaml (main), base.yaml, dsec_flow.yaml
├── tests/              pytest unit tests
├── docs/               SPEC.md, RIDGE_RESULTS.md
└── pyproject.toml, requirements.txt, pytest.ini
```

## Setup (NCI Gadi)

```bash
module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/Train_based
pip install -r requirements.txt      # first time only

# Interactive GPU node:
qsub -I -l walltime=12:00:00,mem=190GB,ncpus=12,ngpus=1,jobfs=50GB \
  -q gpuvolta -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

# ALWAYS source first — sends caches to scratch ($HOME quota is tiny):
source scripts/nci_env.sh
```

## Dataset layout

```
../Data/EVIMO2/
├── train/scene10_dyn_train_00_000000/
│   ├── dataset_events_t.npy  dataset_events_xy.npy  dataset_events_p.npy
│   ├── dataset_mask.npz      dataset_info.npz
│   └── dataset_classical.npz   # fallback when events are empty
└── eval/scene13_dyn_test_00_000000/ ...
```
Set `dataset.root: ../Data/EVIMO2` in `configs/evimo_seg.yaml`.

## Run

```bash
source scripts/nci_env.sh

# 1) Build the multi-time cache ONCE (see cache note below).
python scripts/prepare_evimo.py --config configs/evimo_seg.yaml

# 2) Fit a readout. Feature + combine + head are CLI options; the checkpoint is
#    auto-named checkpoints/[head]_[feature]_[axis]_[event]_[label_mode]_[N].pt (omit --out).
python scripts/fit_ridge_head.py --config configs/evimo_seg.yaml \
  --head prototype --event-feature phi --axis-combine bind --event-combine bind \
  --device cuda --max-train-samples 500 --max-val-samples 50
#  -> checkpoints/prototype_phi_bind_bind_motion_500.pt

# 3) Evaluate: mIoU + PNG panels. Feature + combine are read from the checkpoint.
python -m hdems.eval --config configs/evimo_seg.yaml --head prototype \
  --checkpoint checkpoints/prototype_phi_bind_bind_motion_500.pt \
  --device cuda --save-images eval_out --max-images 50
```

Ridge example, F0 with bind+bundle:
```bash
python scripts/fit_ridge_head.py --config configs/evimo_seg.yaml \
  --head ridge --event-feature f --axis-combine bind --event-combine bindbundle \
  --max-train-samples 500
#  -> checkpoints/ridge_f_bind_bindbundle_motion_500.pt
python -m hdems.eval --config configs/evimo_seg.yaml --head ridge \
  --checkpoint checkpoints/ridge_f_bind_bindbundle_motion_500.pt --save-images eval_out/ridge
```

Trained CNN motion head (uses `hdems.train`, not the fit script):
```bash
python -m hdems.train --config configs/evimo_seg.yaml --device cuda --max-samples 500   # set head: motion first
python -m hdems.eval  --config configs/evimo_seg.yaml --checkpoint checkpoints/last.pt --save-images eval_out
```

Notes on the flags:
- `--max-train-samples N` caps the fit to N frames; that N appears in the auto name.
- `--max-samples` (train/eval) caps frames; `--max-images` only caps saved PNGs.
- Re-fit after changing a combine option (**no cache rebuild** needed — combining
  happens in the model, not in the cached surfaces).

**⚠️ Rebuild the cache** only when you change `dataset` resolution, `window_ms`,
`time_frames`, or `decay` (the `.pt` shards bake those in):
```bash
rm ../Data/EVIMO2/train/*.pt ../Data/EVIMO2/eval/*.pt
python scripts/prepare_evimo.py --config configs/evimo_seg.yaml
```

## Key config knobs (`configs/evimo_seg.yaml`)

| Key | Meaning |
|-----|---------|
| `d` | hypervector dimension (512) |
| `encoder.patch_size` | VFA kernel size N (21; lower ≈11 for speed, σ=1.5 support is ±4–5 px) |
| `matching.M` | cost-volume window (7 → ±3 px per scale) |
| `matching.scales` | `[0,1,2]` two-time pairs F0↔F1/F2/F4 at pool 1/2/4 |
| `matching.alpha` | Eq.12 probability-volume threshold (0.3) |
| `matching.smooth` | Eq.12 cost-volume average pooling, stride 1 (paper `sc`: 71) |
| `dataset.time_frames` | `[0,0.25,0.5,1.0]` multi-time (paper); `[]` = single-time |
| `dataset.height/width` | working resolution (240×320) |
| `dataset.label_mode` | `motion` (pose-derived) · `tracked` (legacy mask>0) · `objects` |
| `dataset.scene_disjoint` | drop train sequences whose scene also appears in `eval/` |
| `dataset.require_mover` | train only on frames with ≥1 moving object |
| `dataset.negative_ratio` | share of all-static frames kept as negatives |
| `dataset.interleave` | round-robin across sequences, so `--max-train-samples N` is balanced |
| `motion_label.move_px` / `static_px` | moving / static thresholds, px per window |
| `motion_label.boundary_ignore_px` | ignore band around mask edges |
| `velocity.event_feature` | `phi` · `f` |
| `velocity.axis_combine` | `bind` · `bundle` |
| `velocity.event_combine` | `bind` · `bundle` · `bindbundle` · `concat` |
| `segmentation.head` | `prototype` (default) · `ridge` · `motion` · `cnn` |

## Front end = the paper (Table A1)

| | paper | setting |
|---|---|---|
| Time surface decay τ (Eq. 3) | 35 ms | `time_surface.tau_ms: 35` |
| HD kernel `K = D ∗ G` (Eq. 6) | convolution, N = 21, σ = 1.5 | `encoder.kernel: conv` |
| Per-polarity kernels + role binding (Eq. 7–8) | yes | `encoder.polarity_binding: true` |
| Multi-scale descriptor (Eq. 9) | S = 2 | `encoder.scales: 2` |
| Cost-volume search window | M = 31 | `matching.M: 31` |
| Eq. 12 threshold / pooling | α = 0.85, sc = 71 | `matching.alpha`, `matching.smooth` |

Measured against the previous implementation: the old Gaussian-*window* kernel
held 0% of its energy beyond a 4 px radius (the "21×21" kernel was effectively
9×9) — the paper's convolution keeps 87%. Summing polarities made a positive and
a negative edge identical (cosine +1.00); role-binding makes them orthogonal.
`kernel: window`, `polarity_binding: false`, `scales: 1` rebuild the old encoder.

### Resolution ratio

`dataset.resolution_ratio` (or `--resolution-ratio`): 1 = 480×640, 2 = 240×320,
4 = 120×160. The pixel-sized settings are divided with it so they cover the same
physical area — N 21→11→5, M 31→15→7, sc 71→35→17 — which makes the cost volume
roughly ratio⁴ cheaper. Surfaces are downsampled by area averaging (bilinear at
1/4 would ignore 12 of every 16 pixels). Motion labels stay in sensor pixels, so
every ratio is scored on the same task. Checkpoints store their front end
(including the ratio), and eval rebuilds exactly that.

### Check the flow before training

```bash
qsub run_flow_check.pbs                       # ~minutes, no training
python scripts/flow_epe.py --config configs/evimo_seg.yaml --resolution-ratio 2 --device cuda
```

Compares the VSA flow with ground-truth flow (EVIMO2 depth + poses,
`hdems/data/gt_flow.py`): EPE, zero-flow EPE and correlation per split. PASS =
correlation ≥ 0.7 and EPE < 0.6 × zero-flow EPE on **both** splits. Train the
classifier only at a ratio that passes — otherwise it learns from noise.

## Training speed: flow cache + early stopping

The cost volume is ~80% of a training step and has no trainable parameters, so
recomputing it every epoch was what made training take >24 h. With
`flow_cache.enabled: true` the flow is computed **once per frame** and stored in
`cache/flow/` (~1.2 MB/frame fp16); every later epoch, validation pass, eval run
and any later run that shares the same front end (e.g. a different head or
combine mode) reads it back instead.

- The cache key covers the encoder's actual kernel weights, `M`, `alpha`,
  `smooth`, `vel_scale` and the time surface itself, so changing any of them
  recomputes automatically — it cannot serve stale flow. Delete the folder any time.
- A cache hit and a cache miss give bit-identical features (both go through fp16).
- The latency printed by `hdems.eval` bypasses the cache, so it is the real speed.

`train.patience` stops training after that many validations without improvement,
and `train.val_every` validates every N epochs. The best epoch is selected on
foreground IoU in motion mode.

## One colour per moving object

The CNN answers *"is this pixel moving?"*. `hdems/grouping.py` then answers
*"which object?"*, the way EMSGC and cascaded multi-model fitting do: every
independently moving object has one rigid motion, so its pixels share one
parametric (affine) flow model — even when the flow **direction** varies across
the object (rotation, scaling). Pixels are grouped by the model they fit:

1. sequential RANSAC on the ego-compensated VSA flow of the CNN's moving pixels
2. each pixel joins the model it fits best
3. one model covering two separate regions → two objects
4. neighbouring groups whose joint motion still fits one model are merged (so one
   object does not split into several colours); small leftovers join a neighbour

Settings: `grouping:` in the config (full-resolution units, scaled by
`resolution_ratio`). It runs in `hdems.eval` only — training is unchanged.

**Ground truth.** EVIMO2 gives every tracked part its own id, but motion
segmentation can only separate things that move *independently*. Parts whose
relative pose stays constant (from the poses) are merged into one object
(`rigid_groups` → `gt_instances`), matching the papers' definition.

**Reported:** object mIoU / precision / recall over frames that contain a moving
object, the same grouping run on the ground-truth moving pixels (best case — the
gap is the CNN's share), and how often a static frame gets a false object.

## Colouring the moving events

```bash
python -m hdems.eval --config configs/evimo_seg.yaml --head cnn \
  --checkpoint checkpoints/<run>.pt --color-events results/<run>/colour
```

Writes, per frame: events | CNN moving (red) vs static (grey) | **objects: CNN +
motion grouping, one colour each** | the same grouping on the ground-truth moving
pixels (best case) | ground-truth objects. Everything is drawn **only at event
pixels** — the rest is untrained guesswork.

## Notes

- EVIMO2 masks cover every tracked surface, including the static table — see
  "Labels" above. `label_mode: motion` is the only mode that means motion.
- Record results in `docs/RIDGE_RESULTS.md`; `docs/SPEC.md` has the VSA constraints.
- CLI combine/head options override the config; the chosen combo is stored inside
  each checkpoint so eval auto-matches it.
