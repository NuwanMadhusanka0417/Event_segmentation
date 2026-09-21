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

## Heads (`segmentation.head`, or `--head`)

With `dataset.time_frames` set, all heads use the paper front-end.

- **`prototype`** (default) — VSA nearest-centroid; class prototypes are the
  normalized bundle of training features. **Parameter-free**, no backprop.
- **`ridge`** — closed-form linear readout (least squares). No backprop.
- **`motion`** — small trained CNN on motion-primary features (`hdems.train`).
- **`cnn`** — raw single-frame Φ baseline (`hdems.train`).

`prototype`/`ridge` are **fit** with `scripts/fit_ridge_head.py`.

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
  so an unscaled bundle would drown the velocity term.
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
| `dataset.time_frames` | `[0,0.25,0.5,1.0]` multi-time (paper); `[]` = single-time |
| `dataset.height/width` | working resolution (240×320) |
| `velocity.event_feature` | `phi` · `f` |
| `velocity.axis_combine` | `bind` · `bundle` |
| `velocity.event_combine` | `bind` · `bundle` · `bindbundle` · `concat` |
| `segmentation.head` | `prototype` (default) · `ridge` · `motion` · `cnn` |

## Notes

- Masks are remapped to consecutive class IDs (background = 0); EVIMO masks are
  the independently-moving objects, so this is genuinely motion segmentation.
- Record results in `docs/RIDGE_RESULTS.md`; `docs/SPEC.md` has the VSA constraints.
- CLI combine/head options override the config; the chosen combo is stored inside
  each checkpoint so eval auto-matches it.
