# GVFA_HEAD — Supervised per-event FG/BG segmentation

Frozen GVFA encoder (`src/`) + trainable `Adapter` / `SegHead` only.
Data path is **lazy memmap**: never holds a full clip in RAM.

## Data

Format **(B)**: Prophesee `*_td.dat` + `*_bbox.npy` (pixel top-left xywh).
Windows are 50 ms causal; labels derived per-window from boxes.
Index sidecars go to `--index_cache_dir` (default `output/index_cache`).
`--max_events_per_window` (default 20000) caps graph size via uniform subsample.

## Train / test

```bash
cd GVFA_HEAD

python run.py --mode train --train_dir ../Data/train \
  --out_dir output/runs/seg_v1 --epochs 1 --window_ms 50 \
  --index_cache_dir output/index_cache --max_events_per_window 20000

python run.py --mode test --test_dir ../Data/test \
  --ckpt output/runs/seg_v1/best.pt \
  --out_dir output/seg_v1/test_images --window_ms 50
```

Headline metric: **foreground IoU** (`best.pt` by val fg-IoU). RSS logs should stay flat (a few GiB), not climb with clips seen.

```bash
qsub -I  -l walltime=5:00:00,mem=190GB,ncpus=12,jobfs=50GB -P mi23 \
  -l storage=gdata/jq77+scratch/jq77+scratch/mi23
module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/GVFA_HEAD
```
