# GVFA event segmentation

Events-only motion segmentation with a frozen GVFA/VSA encoder. The pipeline:

1. Build spatial + temporal graph
2. Estimate normal flow → **aperture resolve** (optional)
3. Ego-motion fit + residual split (background / IMO)
4. FPE node encoding + GVFA
5. Motion-coherent pooling + label smoothing

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
pip install pandas scipy matplotlib
```

---



## Motion resolvers


| Resolver   | Flag                       | Description                               |
| ---------- | -------------------------- | ----------------------------------------- |
| **none**   | `--motion-resolver none`   | Raw normal flow (ablation baseline)       |
| **lk**     | `--motion-resolver lk`     | Track A: Lucas–Kanade local fit (default) |
| **affine** | `--motion-resolver affine` | Track A: affine flow with LK fallback     |
| **vsa**    | `--motion-resolver vsa`    | Track B: VSA constraint superposition     |


---



## Single time window

Processes the **first** `--window-ms` milliseconds from the file start (default 1000 ms).

```bash
python segment.py --input events_filtered.txt --window-ms 1000 --out-dir diag
```



### One resolver

```bash
# Ablation — raw normal flow
python segment.py --input events_filtered.txt --window-ms 1000  --motion-resolver none --out-dir diag/none

# Track A (default)
python segment.py --input events_filtered.txt --window-ms 1000 
  --motion-resolver lk --out-dir diag/lk

python segment.py --input events_filtered.txt --window-ms 1000 
  --motion-resolver affine --out-dir diag/affine

# Track B — VSA (+ Hough validation figures when --vsa-validate true)
python segment.py --input events_filtered.txt --window-ms 1000 
  --motion-resolver vsa --out-dir diag/vsa
```



### All four resolvers (one window)

```bash
python segment.py --input events_filtered.txt --window-ms 1000 
  --all-resolvers --out-dir diag
```

Creates `diag/none/`, `diag/lk/`, `diag/affine/`, `diag/vsa/` with full diagnostic PNGs.

### Compare resolvers (paper ablation)

Runs `none`, `lk`, `vsa` on the **same** window; writes `compare_resolvers.csv` and `15_resolver_comparison.png`.

```bash
python segment.py --input events_filtered.txt --window-ms 1000 
  --compare-resolvers --out-dir diag
```

---



## Stream 60 ms segments (full recording)

Tiles the **entire file** into non-overlapping segments (default **60 ms**).  
Use `--out-dir` to choose where PNGs are saved.

```bash
python segment.py --input events_filtered.txt \
  --stream-segments --segment-ms 60 --out-dir diag_stream
```



### Default behaviour


| Segment                        | Diagnostics saved                                                   |
| ------------------------------ | ------------------------------------------------------------------- |
| **First** (0–60 ms)            | Full set per resolver: `01`–`07`, `11`, `12` (+ `13`, `14` for VSA) |
| **Later** (60–120, 120–180, …) | `06_segmentation_{start}_{end}.png` **only**                        |


Also written: `stream_segments.csv` (metrics per segment × method).

### Output layout (flat folders, time in filename)

No `t0000_0060/` subfolders — files sit directly under each resolver folder:

```text
diag_stream/
  stream_segments.csv
  none/
    01_flow_raw_0_60.png
    02_flow_smoothed_0_60.png
    …
    06_segmentation_0_60.png
    07_summary_0_60.png
    06_segmentation_60_120.png
    06_segmentation_120_180.png
    …
  lk/          … same pattern …
  affine/      …
  vsa/
    …
    13_constraint_votes_0_60.png
    14_vsa_vs_hough_0_60.png
    06_segmentation_60_120.png
    …
```

Filename pattern: `{figure}_{start_ms}_{end_ms}.png` (e.g. `06_segmentation_180_240.png`).

### Stream — one resolver only

**Without** `--motion-resolver`: runs all four resolvers (default).

**With** `--motion-resolver`: runs only that resolver across all segments.

```bash
# VSA only, all 60 ms segments
python segment.py --input events_filtered.txt \
  --stream-segments --segment-ms 60 \
  --motion-resolver vsa --out-dir diag_stream/vsa

# LK only
python segment.py --input events_filtered.txt \
  --stream-segments --segment-ms 60 \
  --motion-resolver lk --out-dir diag_stream/lk
```



### Stream — full diagnostics on every segment (slow)

```bash
python segment.py --input events_filtered.txt \
  --stream-segments --segment-ms 60 --diag-all-segments \
  --out-dir diag_stream
```

---



## Diagnostic figures


| File                     | Stage                           |
| ------------------------ | ------------------------------- |
| `01_flow_raw`            | Raw normal flow                 |
| `02_flow_smoothed`       | Smoothed resolved flow          |
| `03_ego_fit`             | Ego-motion fit                  |
| `04_residual_split`      | Background vs IMO               |
| `05_supernodes`          | Motion-coherent pooling         |
| `06_segmentation`        | Final segmentation              |
| `07_summary`             | Mosaic of stages                |
| `11_flow_resolved`       | Aperture-resolved flow          |
| `12_orientation_check`   | Edge-normal vs flow angle       |
| `13_constraint_votes`    | VSA only — vote maps            |
| `14_vsa_vs_hough`        | VSA only — HV vs explicit Hough |
| `15_resolver_comparison` | Compare mode only               |


---



## Useful CLI flags

```bash
--out-dir PATH          # output root (default: diag)
--window-ms 1000        # single-window length (ignored when --stream-segments)
--segment-ms 60         # tile size for streaming
--tau 0.15              # clustering merge threshold
--num-layers 3          # GVFA depth
--lam 1.5               # label smoothing strength

# Track A (LK / affine)
--lk-min-support 6
--condition-min-eig 1e-3

# Track B (VSA)
--d-vel 512
--vel-grid-n 48
--band-sigma 0.08
--cleanup-topk 5
--cleanup-min-conf 0.05
--vsa-validate false    # skip Hough validation (faster on later segments)

# Node encoding
--w-node-motion 1.0
--w-node-t 0.1
```



### Segmentation tuning examples

```bash
python segment.py --tau 0.08 --window-ms 30 --num-layers 2   # fewer objects
python segment.py --tau 0.18 --window-ms 30                  # more objects
```

---



## Outputs besides PNGs


| Mode                           | Extra files                                           |
| ------------------------------ | ----------------------------------------------------- |
| Single resolver, single window | `events_labeled.parquet`, `seg.png` (cwd)             |
| `--compare-resolvers`          | `compare_resolvers.csv`, `15_resolver_comparison.png` |
| `--stream-segments`            | `stream_segments.csv`                                 |


