# VSA Motion Segmentation (`Train_gpt`)

Class-agnostic **motion segmentation** for event cameras using **Vector Symbolic Architectures (VSA)**, extending the feature-matching / optical-flow ideas from **You et al., “Vector-Symbolic Architecture for Event-Based Optical Flow” (2025)**. The primary pipeline is **training-free**: frozen VSA encoder, VSA cost-volume flow, HDC prototypes, and CPU-friendly clustering. **E-RAFT and CNN segmentation heads are not used** in the default path (Ridge/OnlineHD are optional comparisons only).

## Pipeline

```text
Event stream → time surface → frozen multi-scale VSA encoder → hypervector field
  → VSA cost-volume flow → ego compensation → residual flow
  → motion hypervectors → static/dynamic prototypes
  → region growing / DBSCAN → local VSA refinement → temporal segment IDs → metrics
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for design notes and relation to other folders in the repo.

## Installation

```bash
cd Train_gpt
pip install -r requirements.txt
```

Use a **CUDA-enabled PyTorch** wheel if you want GPU acceleration (`pip install torch` matching your CUDA version from [pytorch.org](https://pytorch.org)).

## Dataset (EVIMO2)

Configure paths in `configs/evimo.yaml` (default):

```yaml
dataset:
  root: ../Data/EVIMO2
  split: train
  eval_split: eval
```

- Training data: `Data/EVIMO2/train/<sequence>/`
- Evaluation: `Data/EVIMO2/eval/<sequence>/` (same file layout as train)

Each sequence folder needs `dataset_events_*.npy`, `dataset_mask.npz`, and `dataset_info.npz`.

## Quick start (CPU)

```bash
# 1) Prototypes from train split (limited frames by default)
python scripts/train_prototypes.py --config configs/evimo.yaml --output checkpoints/prototypes.pt

# 2) Segment one sequence
python scripts/segment_sequence.py --config configs/evimo.yaml --input ../Data/EVIMO2 --output outputs/evimo_vsa_motionseg

# 3) Evaluate on eval split
python scripts/evaluate.py --config configs/evimo.yaml --split eval
```

## Running with CUDA

GPU accelerates **VSA encoding**, **cost-volume flow**, and **prototype readout**. **Clustering, refinement, and temporal matching stay on CPU** (SciKit-learn / SciPy / Python loops).

### Check GPU

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

On **Gadi**, request a GPU node and load CUDA first — see [`../GADI_CUDA_README.md`](../GADI_CUDA_README.md) for `qsub`, `module load cuda`, and venv activation (adjust paths to `Train_gpt` instead of `Train_based`).

### Option A — CLI (overrides config)

```bash
python scripts/train_prototypes.py --config configs/evimo.yaml --device cuda --max-frames 10

python scripts/segment_sequence.py --config configs/evimo.yaml --input ../Data/EVIMO2 --output outputs/evimo_vsa_cuda --device cuda

python scripts/evaluate.py --config configs/evimo.yaml --split eval --device cuda --max-frames 15
```

Use `cuda:0`, `cuda:1`, etc. if you have multiple GPUs. If CUDA is unavailable, the pipeline **falls back to CPU** automatically.

### Option B — config file

In `configs/evimo.yaml` or `configs/default.yaml`:

```yaml
runtime:
  device: cuda
```

Scripts print `Device: cuda:0` (or `cpu`) at startup.

## Limiting training and testing samples

Limits apply to **frames** (timestamps) from **one sequence folder**, not to individual event pixels.

| Setting | Config key | CLI flag | Default (from merged config) |
|--------|------------|----------|------------------------------|
| Prototype training (frames) | `dataset.max_train_frames` | `--max-frames` | 20 |
| Prototype training (pixels) | `dataset.max_train_samples` | `--max-samples` | unlimited (`null`) |
| Evaluation | `dataset.max_eval_frames` | `--max-frames` | 10 |
| Segmentation / inference | `dataset.max_infer_frames` | `--max-frames` | all frames (`null`) |
| Which sequence | `dataset.sequence_index` | `--sequence` | 0 (first sorted folder under split) |

**Examples — small smoke runs:**

```bash
# Train prototypes on 5 frames from sequence 0
python scripts/train_prototypes.py --max-frames 5 --sequence 0

# Cap pixel-level training rows (static/dynamic labels at active event pixels)
python scripts/train_prototypes.py --max-samples 5000 --max-frames 20 --device cuda

# Eval on 8 frames from eval split, sequence 1
python scripts/evaluate.py --split eval --max-frames 8 --sequence 1

# Segment only 30 frames with CUDA
python scripts/segment_sequence.py \
  --output outputs/smoke \
  --max-frames 30 \
  --device cuda
```

**Notes:**

- `--max-frames` limits **timestamps**; `--max-samples` limits **pixel rows** used for prototypes (stratified static/dynamic when class balancing is on).
- With `--max-samples`, processing **stops early** once enough active pixels were collected (still at least one full frame).
- Frames without instance masks are skipped for training.

## Tests

```bash
pytest tests/ -q
```

Tests always use CPU-friendly settings in code; they do not require a GPU.

## Configuration

- `configs/default.yaml` — full parameter surface (`runtime.device`, frame caps, VSA, flow, clustering)
- `configs/evimo.yaml` — EVIMO2 experiment (D=512, rank=16)

Important knobs: `runtime.device`, `dataset.max_*_frames`, `vsa.dimension`, `flow.search_radius`, `events.window_ms`, `clustering.method`.

## Reproducibility

VSA kernels and role vectors are saved under `checkpoints/vsa_artifacts/` when `vsa.artifact_dir` is set. Random seeds are controlled via `vsa.seed`.

## Known limitations

- Pixels without events cannot support reliable motion or segmentation.
- Normal flow (optional) suffers from the aperture problem.
- Full translational ego-motion without depth is only approximated (RANSAC translation).
- Nearby objects with similar motion may merge.
- Sparse events yield unstable hypervectors.
- Temporal smoothing / tracking can lag during fast motion.
- HDC prototypes alone do not guarantee instance identity — explicit segment matching is used for IDs.

## Computational notes

Default **CPU** execution is supported everywhere. With **`runtime.device: cuda`**, expect the largest speedups on encoding and cost-volume stages; end-to-end time still includes CPU clustering. Per-stage timings are printed in each result’s `benchmark` field. Real-time performance is **not** claimed until you benchmark on your hardware.

## Optional / future

- Ridge readout: `vsa_motionseg/segmentation/ridge_readout.py`
- Self-supervised similarity loss: disabled in config (`self_supervised.enabled: false`)
- DSEC adapter stub: `vsa_motionseg/data/evimo_adapter.py` (`DSECAdapterStub`)
