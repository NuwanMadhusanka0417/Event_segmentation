# VSA Motion Segmentation — Implementation Plan

## Repository inspection (2026-09-14)

### Top-level layout

| Path | Role |
|------|------|
| `Train_based/hdems/` | Primary HDEMS stack: VSA encoder, matching, EVIMO2 reader, time surfaces, optional CNN/Ridge segmentation |
| `GVFA/` | Graph/VSA segmentation prototype: `node_flow`, `_fit_plane`, ego motion, diagnostics |
| `event_segmentation/` | Lightweight segmentation scripts |
| `web/` | Visualization UI |
| `Data/` | User data (EVIMO2 under `train/` and future `eval/`) — not in git |

### Existing VSA code (reuse in `Train_gpt`, do not modify originals)

- `Train_based/hdems/vsa/fpe.py` — FPE, bind/unbind, bundle, cosine similarity
- `Train_based/hdems/vsa/kernel.py` — VFA Gram matrix, rank-64 eigen basis (configurable `r`)
- `Train_based/hdems/vsa/field.py` — bundled matching field, explicit cost volume
- `Train_based/hdems/vsa/temporal.py` — temporal binding helpers
- `Train_based/hdems/models/encoder.py` — frozen rank-r VSA encoder, default **D=1024**, rank **64**
- `Train_based/hdems/models/matching.py` — hierarchical matcher
- `Train_based/hdems/data/time_surface.py` — polarity surfaces; **decay=0.8** per-event multiplier (not physical τ)
- `Train_based/hdems/data/evimo2_reader.py` — EVIMO2 NPZ/NPY sequences

### Hypervector dimensionality

- Encoder default: **D=1024** (`VSAEncoder`)
- EVIMO seg config: **D=512** (`Train_based/configs/evimo_seg.yaml`)
- Basis rank: **64** default in encoder; **16** in evimo_seg.yaml (separate from D)

### Event representation

- Columns: `[t, x, y, p]` in time-surface code; EVIMO2 loads `[t, x, y, p]` from separate `.npy` files
- Accumulative decay: `decay ** (t - t0)` in existing code

### Dataset format (EVIMO2)

Per sequence folder: `dataset_events_{t,xy,p}.npy`, `dataset_mask.npz`, `dataset_info.npz`, frame list in meta.

### Tests / config in repo

- `Train_based/tests/test_fpe.py`, `test_field_exact.py`, `test_kernel_rank.py`, etc.
- Configs: `Train_based/configs/evimo_seg.yaml`, `dsec_flow.yaml`

### Optional modules elsewhere

- Normal flow: `GVFA/segment.py` (`node_flow`, `_fit_plane`) — optional only in this project

---

## `Train_gpt` goals

New package **`vsa_motionseg`** implements a **training-free, CPU-first** pipeline:

```
Events → time surface → frozen VSA encoder → hypervector field
  → VSA cost-volume flow → ego compensation → residual flow
  → motion hypervectors → HDC static/dynamic prototypes
  → region growing / DBSCAN → local VSA refinement → temporal segment IDs
  → metrics (event-masked IoU, F1, boundary, temporal stability)
```

**No E-RAFT or CNN head in the primary path.** Ridge / OnlineHD are optional comparisons.

---

## Planned modules (this folder)

| Module | Source / notes |
|--------|----------------|
| `vsa_motionseg/vsa/*` | Adapted from `Train_based/hdems/vsa` + multi-scale polarity/scale encoder |
| `vsa_motionseg/data/*` | EVIMO2 adapter + generic base; paths via YAML |
| `vsa_motionseg/motion/*` | Cost volume, flow, RANSAC ego, optional normal flow stub |
| `vsa_motionseg/segmentation/*` | Prototypes, clustering, refinement, tracking |
| `vsa_motionseg/evaluation/*` | Event-masked metrics, benchmark timing |
| `scripts/segment_sequence.py` | Main entry point |

---

## Phases

1. **Done in initial commit**: plan, VSA core, time surface (τ + legacy decay), encoder, cost-volume flow, prototypes, region growing, refinement, temporal matching, EVIMO adapter, configs, tests, README.
2. **Next**: full EVIMO2 eval runs, prototype training script on `Data/EVIMO2/train`, ablation scripts.
3. **Later**: self-supervised similarity loss, DSEC adapter, graph-cut refinement, depth-based rigid flow.

---

## Data paths (configurable)

```yaml
dataset:
  root: ../Data/EVIMO2
  split: train
  eval_split: eval
```

User training data: `Data/EVIMO2/train`. Testing: `Data/EVIMO2/eval` (to be added).

---

## Preservation policy

- No files under `Train_based/`, `GVFA/`, or `event_segmentation/` are modified by this work.
- Shared logic is **copied and extended** inside `Train_gpt/vsa_motionseg/`.
