# GVFA-YOLO — Event-Camera Object Detection

Frozen **GVFA** (graph hypervectors) + trainable **point-wise YOLO** head on Prophesee Gen1 data.

Each Prophesee **recording** (`*_td.dat` + `*_bbox.npy`) is split into many short **time windows** (default **5 ms**). One window = one training/eval **sample**.

```
One recording (~60 s)
├── window 0:  0–5 ms   → sample 0
├── window 1:  5–10 ms  → sample 1
└── ...
```

Pipeline per window: events → causal graphs → frozen GVFA encoder → YOLO head → NMS.

---

## Gadi setup (interactive job)

```bash
qsub -I -l walltime=15:00:00,mem=40GB,ncpus=7,jobfs=50GB \
  -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

```bash
source /scratch/jq77/nk8155/nv2/bin/activate
module load python3/3.11.0
cd /g/data/jq77/nuwan/Event_segmentation/GVFA_YOLO
```

**Pip on scratch** (if installing packages):

```bash
export TMPDIR=/scratch/nv2/tmp
export PIP_CACHE_DIR=/scratch/nv2/pip_cache
mkdir -p $TMPDIR $PIP_CACHE_DIR

pip install <LIBRARY> \
  --target=/scratch/jq77/nk8155/nv2/lib/python3.11/site-packages \
  --no-cache-dir
```

**RAM:** ~32–40 GB is enough with `--max_events 15000`. Use 100 GB only for very large runs without subsampling.

---

## 1. Extract data (once)

```bash
7z x ../Data/Prophesee/train_a.7z -o../Data/train_a
7z x ../Data/Prophesee/val_a.7z   -o../Data/val_a
```

Each folder must contain matching `*_td.dat` and `*_bbox.npy` pairs.

---

## 2. Commands

### Smoke test — pipeline check (no labels)

Runs **one** time window through GVFA + detection head. Good first run; no Prophesee data needed.

```bash
python train.py smoke --events events_filtered.txt --max_events 5000
```

Output: `boxes_on_events.png` (override with `--out`).

---

### Quick debug train — few samples

Trains only on a **small subset** of windows. Use this before a full run.

```bash
python train.py train \
  --data_dir ../Data/val_a \
  --max_windows 5 \
  --max_events 15000 \
  --epochs 1 \
  --ckpt checkpoints/head_debug.pt
```

- Processes **5 time windows** total (usually from the first recording).
- **1 epoch** = one pass over those 5 windows.

With validation:

```bash
python train.py train \
  --data_dir ../Data/train_a \
  --val_dir ../Data/val_a \
  --max_recordings 2 \
  --max_windows 10 \
  --max_events 15000 \
  --epochs 1 \
  --ckpt checkpoints/head_debug.pt
```

---

### Full train — Prophesee train_a

Streams one window at a time (low RAM). Full `train_a` has hundreds of thousands of windows; expect long runtime on CPU.

```bash
python train.py train \
  --data_dir ../Data/train_a \
  --val_dir ../Data/val_a \
  --max_events 15000 \
  --epochs 10 \
  --ckpt checkpoints/head.pt
```

Optional longer time slices (default window is 5 ms):

```bash
python train.py train \
  --data_dir ../Data/train_a \
  --val_dir ../Data/val_a \
  --window_ms 50 \
  --max_events 15000 \
  --epochs 10 \
  --ckpt checkpoints/head.pt
```

---

### Eval — mAP on validation set

Runs inference on all windows (or limited subset) and reports COCO-style mAP.

```bash
python train.py eval \
  --data_dir ../Data/val_a \
  --ckpt checkpoints/head.pt \
  --score_thr 0.3
```

Quick eval on few windows:

```bash
python train.py eval \
  --data_dir ../Data/val_a \
  --max_windows 10 \
  --max_events 15000 \
  --ckpt checkpoints/head.pt
```

---

### Test — save detection frames

Runs detection and saves event visualizations with boxes to disk.

```bash
python train.py test \
  --data_dir ../Data/val_a \
  --ckpt checkpoints/head.pt \
  --out_dir runs/test_frames \
  --score_thr 0.3
```

Few frames only:

```bash
python train.py test \
  --data_dir ../Data/val_a \
  --max_windows 5 \
  --max_events 15000 \
  --ckpt checkpoints/head.pt \
  --out_dir runs/test_few
```

Output: `runs/test_few/<recording_name>/win_*.png`

---

## 3. Important flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--data_dir` | — | Folder with Prophesee `*_td.dat` + `*_bbox.npy` pairs |
| `--val_dir` | — | Validation folder (train mode only; reports mAP each epoch) |
| `--window_ms` | `5` | Length of each time slice in milliseconds |
| `--max_recordings` | all | Use only the first N `.dat` files (sorted by name) |
| `--max_windows` | all | Cap total windows across all recordings (debug / quick runs) |
| `--max_events` | `40000` | Max events **per window**; subsamples if exceeded (speed/RAM) |
| `--epochs` | `10` | Training passes over the indexed windows |
| `--ckpt` | `checkpoints/head.pt` | Where to save/load the YOLO head weights |
| `--score_thr` | `0.3` | Detection confidence threshold (eval/test) |
| `--reader_cache_size` | `1` | Mmap readers kept open; `1` = lowest RAM |

**How limits combine**

- `max_recordings=2` → only 2 scene files.
- `max_windows=10` → only 10 time slices total (from recording 1, then 2 if needed).
- `max_events=15000` → each slice still covers the same time span, but at most 15k events are kept.

Empty windows (no events in that slice) are skipped during training.

---

## 4. Modes summary

| Mode | Purpose |
|------|---------|
| `smoke` | One unlabeled window; verify encoder + head run |
| `train` | Train YOLO head (GVFA frozen); optional val mAP |
| `eval` | mAP on labeled data |
| `test` | Save PNG frames with detections |
