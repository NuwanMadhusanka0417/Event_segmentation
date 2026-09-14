# VSA Motion Segmentation (`Train_gpt`)

Class-agnostic **motion segmentation** for event cameras using **Vector Symbolic Architectures (VSA)**, extending the feature-matching / optical-flow ideas from **You et al., “Vector-Symbolic Architecture for Event-Based Optical Flow” (2025)**. The primary pipeline is **training-free**: frozen VSA encoder, VSA cost-volume flow, HDC prototypes, and CPU-friendly clustering. **E-RAFT and CNN segmentation heads are not used** in the default path (Ridge/OnlineHD are optional comparisons only).

## Pipeline

```text
Event stream → time surface → frozen multi-scale VSA encoder → hypervector field
  → VSA cost-volume flow → ego compensation → residual flow
  → motion hypervectors → static/dynamic prototypes
  → region growing / DBSCAN → local VSA refinement → temporal segment IDs → metrics
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for how this relates to existing code under `Train_based/` and `GVFA/`.

## Installation

```bash
cd Train_gpt
pip install -r requirements.txt
```

## Dataset (EVIMO2)

Configure paths in `configs/evimo.yaml` (default):

```yaml
dataset:
  root: ../Data/EVIMO2
  split: train
  eval_split: eval
```

- Training data: `Data/EVIMO2/train`
- Evaluation: add sequences under `Data/EVIMO2/eval` (same layout as train)

## Commands

**Train static/dynamic prototypes** (uses instance masks as weak motion proxy until motion GT is wired):

```bash
python scripts/train_prototypes.py --config configs/evimo.yaml --output checkpoints/prototypes.pt
```

**Run segmentation on a sequence:**

```bash
python scripts/segment_sequence.py \
  --config configs/evimo.yaml \
  --input ../Data/EVIMO2 \
  --output outputs/evimo_vsa_motionseg
```

**Evaluate** (event-masked IoU/F1/boundary/temporal stability when GT exists):

```bash
python scripts/evaluate.py --config configs/evimo.yaml --split eval --max-frames 20
```

**Tests (synthetic + unit):**

```bash
pytest tests/ -q
```

## Configuration

- `configs/default.yaml` — full parameter surface
- `configs/evimo.yaml` — EVIMO2 experiment (D=512, rank=16, aligned with existing seg config)

Important knobs: `vsa.dimension`, `flow.search_radius`, `events.window_ms`, `ego_motion.mode`, `clustering.method`, `classifier.type`.

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

The pipeline targets **CPU** (PyTorch CPU tensors). Runtime per stage is printed in `benchmark` output. Real-time performance is **not** claimed until you benchmark on your hardware.

## Optional / future

- Ridge readout: `vsa_motionseg/segmentation/ridge_readout.py`
- Self-supervised similarity loss: disabled in config (`self_supervised.enabled: false`)
- DSEC adapter stub: `vsa_motionseg/data/evimo_adapter.py` (`DSECAdapterStub`)
