# GVFA event segmentation

Events-only motion segmentation with a frozen GVFA/VSA encoder. The pipeline:

1. Build spatial + temporal graph
2. Estimate normal flow → **aperture resolve** (VSA / LK / affine / none)
3. Ego-motion fit + residual split (background / IMO)
4. FPE node encoding + GVFA
5. **IMO labeling** — EM soft assignment (default) or sequential VSA pool+assign
6. Graph label smoothing + tiny-cluster cleanup

---

## Gadi (NCI) setup

```bash
ssh nk8155@gadi.nci.org.au

qsub -I -l walltime=12:00:00,mem=190GB,ncpus=12,jobfs=50GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/GVFA
```

### Pip cache (if installs fail)

```bash
echo $PIP_CACHE_DIR
echo $TMPDIR
which pip   # should be .../seg/bin/pip

export PIP_CACHE_DIR=/scratch/jq77/nk8155/.pip_cache
export TMPDIR=/scratch/jq77/nk8155/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR
pip install --upgrade pip
pip install pandas scipy matplotlib scikit-learn
```

---

## VSA only (recommended)

Use `--motion-resolver vsa` to run **only** the VSA aperture resolver. Without this flag, `--stream-segments` runs all four resolvers (`none`, `lk`, `affine`, `vsa`).

### Single window (first N ms)

Full diagnostics (`01`–`07`, `11`–`12`, `13`–`14`, `17`–`19` on first segment) plus `events_labeled.parquet` and `seg.png`:

```bash
python segment.py \
  --input events_filtered.txt \
  --window-ms 60 \
  --motion-resolver vsa \
  --out-dir diag/vsa
```

EM assignment is the default. To use VSA prototype clusters as the EM warm start:

```bash
python segment.py \
  --input events_filtered.txt \
  --window-ms 60 \
  --motion-resolver vsa \
  --em-init vsa \
  --out-dir diag/vsa_em_init
```

Ablation baseline (old sequential pool → assign path, no EM):

```bash
python segment.py \
  --input events_filtered.txt \
  --window-ms 60 \
  --motion-resolver vsa \
  --assignment sequential \
  --out-dir diag/vsa_sequential
```

### Stream full recording (60 ms tiles)

**VSA only** across every segment — writes `stream_segments.csv` and segmentation PNGs under one folder:

```bash
python segment.py \
  --input events_filtered.txt \
  --stream-segments \
  --segment-ms 60 \
  --motion-resolver vsa \
  --out-dir diag_stream/vsa
```

| Segment | Diagnostics saved |
| --- | --- |
| **First** (0–60 ms) | Full set: `01`–`07`, `11`–`12`, `13`–`14`, `17`–`19` |
| **Later** (60–120, …) | `06_segmentation_{start}_{end}.png` only (unless `--diag-all-segments`) |

Output layout (flat folder, time in filename):

```text
diag_stream/vsa/
  stream_segments.csv
  01_flow_raw_0_60.png
  02_flow_smoothed_0_60.png
  …
  06_segmentation_0_60.png
  07_summary_0_60.png
  13_constraint_votes_0_60.png
  14_vsa_vs_hough_0_60.png
  17_em_convergence_0_60.png
  18_soft_memberships_0_60.png
  19_em_vs_sequential_0_60.png
  06_segmentation_60_120.png
  …
```

Filename pattern: `{figure}_{start_ms}_{end_ms}.png` (e.g. `06_segmentation_180_240.png`).

### Stream VSA — full diagnostics on every segment (slow)

```bash
python segment.py \
  --input events_filtered.txt \
  --stream-segments \
  --segment-ms 60 \
  --motion-resolver vsa \
  --diag-all-segments \
  --out-dir diag_stream/vsa_full
```

### Stream VSA — faster later segments

Skip Hough validation on segments after the first (saves time; first segment still validates):

```bash
python segment.py \
  --input events_filtered.txt \
  --stream-segments \
  --segment-ms 60 \
  --motion-resolver vsa \
  --vsa-validate false \
  --out-dir diag_stream/vsa
```

---

## Motion resolvers (ablation)

| Resolver | Flag | Description |
| --- | --- | --- |
| **vsa** | `--motion-resolver vsa` | Track B: VSA constraint superposition |
| **lk** | `--motion-resolver lk` | Track A: Lucas–Kanade local fit (script default if flag omitted) |
| **affine** | `--motion-resolver affine` | Track A: affine flow with LK fallback |
| **none** | `--motion-resolver none` | Raw normal flow (baseline) |

Run all four on one window:

```bash
python segment.py \
  --input events_filtered.txt \
  --window-ms 60 \
  --all-resolvers \
  --out-dir diag
```

Compare `none` / `lk` / `vsa` on the same window (`compare_resolvers.csv`, `15_resolver_comparison.png`):

```bash
python segment.py \
  --input events_filtered.txt \
  --window-ms 60 \
  --compare-resolvers \
  --out-dir diag
```

---

## IMO assignment modes

| Mode | Flag | Description |
| --- | --- | --- |
| **EM** (default) | `--assignment em` | Alternating soft assignment + motion-model refit |
| **Sequential** | `--assignment sequential` | VSA pool → prototype assign (ablation baseline) |

EM warm-start (`--em-init`):

| Init | Flag | Description |
| --- | --- | --- |
| Sequential greedy | `--em-init sequential` | Default — `fit_object_models` warm start |
| K-means | `--em-init kmeans` | K-means on residual velocity |
| VSA prototypes | `--em-init vsa` | Supernode VSA clusters → motion models |

Example — VSA resolver + VSA-init EM:

```bash
python segment.py \
  --input events_filtered.txt \
  --window-ms 60 \
  --motion-resolver vsa \
  --assignment em \
  --em-init vsa \
  --em-n-clusters 6 \
  --em-model-kind affine \
  --out-dir diag/vsa
```

---

## Diagnostic figures

| File | Stage |
| --- | --- |
| `01_flow_raw` | Raw normal flow |
| `02_flow_smoothed` | Smoothed resolved flow |
| `03_ego_fit` | Ego-motion fit |
| `04_residual_split` | Background vs IMO |
| `05_supernodes` | Motion-coherent pooling |
| `06_segmentation` | Final segmentation |
| `07_summary` | Mosaic of stages |
| `11_flow_resolved` | Aperture-resolved flow |
| `12_orientation_check` | Edge-normal vs flow angle |
| `13_constraint_votes` | VSA only — vote maps |
| `14_vsa_vs_hough` | VSA only — HV vs explicit Hough |
| `15_resolver_comparison` | Compare mode only |
| `17_em_convergence` | EM: confidence / live clusters / residual vs iter |
| `18_soft_memberships` | EM: alpha = max membership + histogram |
| `19_em_vs_sequential` | EM vs sequential ablation (first segment) |

---

## Useful CLI flags

```bash
--out-dir PATH              # output root (default: diag)
--window-ms 60              # single-window length (ignored with --stream-segments)
--segment-ms 60             # tile size for streaming
--motion-resolver vsa       # run VSA only (required to skip other resolvers when streaming)

# GVFA / clustering
--tau 0.15                  # VSA prototype merge threshold (sequential / em-init vsa)
--num-layers 3              # GVFA depth
--lam 1.5                   # label smoothing strength
--min-cluster-size 400      # objects smaller than this -> background

# EM assignment (default: em)
--assignment em             # em | sequential
--em-init sequential        # sequential | kmeans | vsa
--em-n-clusters 6
--em-iters 15
--em-tol 1e-3
--em-sigma-min 20.0
--em-min-weight 50.0
--em-model-kind affine      # similarity | affine
--no-em-use-conf-in-smoothing

# Track B (VSA)
--d-vel 2048
--vel-grid-n 48
--band-sigma-px 40.0
--cleanup-topk 5
--cleanup-min-conf 0.05
--vsa-validate false        # faster on later stream segments

# Node encoding
--w-node-motion 1.0
--w-node-t 0.1
```

### Segmentation tuning (VSA)

```bash
python segment.py --motion-resolver vsa --tau 0.08 --window-ms 60 --num-layers 2
python segment.py --motion-resolver vsa --tau 0.18 --window-ms 60
```

---

## Outputs besides PNGs

| Mode | Extra files |
| --- | --- |
| Single resolver, single window | `events_labeled.parquet`, `seg.png` (cwd) |
| `--compare-resolvers` | `compare_resolvers.csv`, `15_resolver_comparison.png` |
| `--stream-segments` | `stream_segments.csv` |
