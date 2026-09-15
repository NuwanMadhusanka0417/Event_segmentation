# HD-EMS

**Hyperdimensional Event Motion Segmentation** — a VSA/VFA (Vector-Symbolic /
Vector-Function Architecture) pipeline for event-camera **motion segmentation**,
built on the VSA-Flow feature-matching method (You et al. 2025) with an added
ego-motion + segmentation stage.

## Pipeline

```
events → multi-time surfaces → VFA HD descriptors (F0,F1,F2,F4)
       → two-time multi-scale cost volume  (paper Eq. 10–11)
       → probability-volume flow estimator (paper Eq. 12–14)
       → ego-motion residual (robust affine, IRLS)
       → motion segmentation head → per-pixel object classes
```

Only the small segmentation head is trained; the entire VSA front-end (kernel,
descriptors, cost volume, flow estimator, ego-fit) is **parameter-free**.

## Layout

```
Train_based/
├── hdems/
│   ├── data/            events → accumulative time surfaces, EVIMO2 reader/dataset
│   │   ├── time_surface.py      vectorized accumulative TS + pyramid
│   │   ├── evimo2_reader.py     raw NPZ/NPY reader; multi-time surface builder
│   │   └── evimo.py             EVIMODataset (raw or cached .pt shards)
│   ├── vsa/             FPE, HRR bind/bundle, VFA kernel eigen-basis, temporal
│   ├── models/
│   │   ├── encoder.py           VFA HD feature descriptor (rank-r, frozen)
│   │   ├── matching.py          bundled matching field (HV context)
│   │   ├── paper_flow.py        two-time multi-scale cost volume + Eq.12–14 flow
│   │   ├── motion.py            single-frame flow decode + ego-motion residual
│   │   ├── segmentation.py      MotionSegHead (motion-primary) / SegmentationHead
│   │   ├── decoder.py           learned flow decoder (flow task only)
│   │   └── hdems.py             HDEMS assembly + forward
│   ├── ridge_head.py / ridge_fit.py / seg_features.py / feature_extract.py
│   │                    closed-form Ridge readout head (no backprop)
│   ├── losses/          seg (CE + Dice), flow (EPE)
│   ├── train.py         training entry point (saves checkpoints/last.pt, best.pt)
│   ├── eval.py          metrics (mIoU) + events|GT|prediction PNG panels
│   └── metrics.py / benchmark.py
├── scripts/
│   ├── nci_env.sh       redirect torch/matplotlib caches to scratch (NCI quota)
│   ├── prepare_evimo.py precompute (surface, mask) → .pt shards
│   └── fit_ridge_head.py
├── configs/            evimo_seg.yaml (main), base.yaml, dsec_flow.yaml
├── tests/              pytest unit tests (FPE, kernel rank, field exactness, …)
├── docs/               SPEC.md, RIDGE_RESULTS.md
└── pyproject.toml, requirements.txt, pytest.ini
```

## Setup (NCI Gadi)

```bash
module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/Train_based

pip install -r requirements.txt      # first time only
pytest tests/ -v                     # optional sanity check

# Interactive GPU node:
qsub -I -l walltime=12:00:00,mem=190GB,ncpus=12,ngpus=1,jobfs=50GB \
  -q gpuvolta -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

# ALWAYS source this first — sends torch/matplotlib caches to scratch
# (NCI $HOME quota is tiny and otherwise causes "Disk quota exceeded"):
source scripts/nci_env.sh
```

## EVIMO2 dataset layout

```
../Data/EVIMO2/
├── train/scene10_dyn_train_00_000000/
│   ├── dataset_events_t.npy   dataset_events_xy.npy   dataset_events_p.npy
│   ├── dataset_mask.npz       dataset_info.npz
│   └── dataset_classical.npz  # fallback when events are empty (e.g. flea3_7)
└── eval/scene13_dyn_test_00_000000/ ...
```

Set `dataset.root: ../Data/EVIMO2` in `configs/evimo_seg.yaml`.

## Run

```bash
source scripts/nci_env.sh

# 1) Precompute cached shards ONCE (fast per-epoch loading).
#    Shards are resolution- and time_frames-baked — see note below.
python scripts/prepare_evimo.py --config configs/evimo_seg.yaml

# 2) Train (only the seg head learns; saves checkpoints/last.pt + best.pt)
python -m hdems.train --config configs/evimo_seg.yaml --device cuda

# 3) Evaluate: mIoU + events|GT|prediction panels
python -m hdems.eval --config configs/evimo_seg.yaml --checkpoint checkpoints/last.pt \
  --device cuda --save-images eval_out --max-images 50
```

**⚠️ Rebuild the cache when you change `dataset` or `time_frames`.** The `.pt`
shards bake in resolution and the multi-time stack. After changing
`height/width/window_ms/time_frames/decay`, delete and rebuild:

```bash
rm ../Data/EVIMO2/train/*.pt ../Data/EVIMO2/eval/*.pt
python scripts/prepare_evimo.py --config configs/evimo_seg.yaml
```

## Key config knobs (`configs/evimo_seg.yaml`)

| Key | Meaning |
|-----|---------|
| `d` | hypervector dimension (512) |
| `encoder.rank` | VFA kernel eigen-basis rank (16) — quality vs speed |
| `matching.M` | cost-volume window (7 → ±3 px per scale) |
| `matching.scales` | `[0,1,2]` two-time pairs F0↔F1/F2/F4 at pool 1/2/4 |
| `matching.alpha` | Eq.12 probability-volume threshold (0.3) |
| `dataset.time_frames` | `[0,0.25,0.5,1.0]` multi-time surfaces; `[]` = single-time |
| `dataset.height/width` | working resolution (240×320) |
| `segmentation.head` | `motion` (default) · `cnn` · `ridge` |
| `train.use_dice` | add Dice to CE for class imbalance (true) |

Set `segmentation.head: motion` for the paper motion pipeline (default), `cnn`
for the raw-HV baseline, or `ridge` for the closed-form readout.

## Ridge readout (closed-form, no backprop)

Ridge and CNN heads use single-time features — set `time_frames: []` first.

```bash
python scripts/fit_ridge_head.py --config configs/evimo_seg.yaml --out checkpoints/ridge_head.pt
python -m hdems.eval --config configs/evimo_seg.yaml --head ridge \
  --ridge-checkpoint checkpoints/ridge_head.pt --device cuda --save-images eval_out/ridge

# Side-by-side CNN vs Ridge:
python -m hdems.eval --config configs/evimo_seg.yaml --checkpoint checkpoints/best.pt \
  --compare-heads --ridge-checkpoint checkpoints/ridge_head.pt --save-images output/ridge_compare
```

See `docs/RIDGE_RESULTS.md` for the metrics table and `docs/SPEC.md` for VSA
constraints.

## Notes

- Raw NPZ/NPY sequences load **directly**; `.pt` shards are an optional cache.
- Masks are remapped to consecutive class IDs (background = 0); EVIMO masks are
  the independently-moving objects, so this is genuinely motion segmentation.
- The VSA front-end is deterministic — precomputing the residual flow to cache is
  the next speed optimization (not yet implemented).
