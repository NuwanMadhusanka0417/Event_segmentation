# GVFA_BG — Supervised per-event FG/BG segmentation

Frozen GVFA encoder (`src/`) + trainable `Adapter` / `SegHead` only.
Unsupervised VSA / ego-motion path removed.

## Data

Defaults: `Data/Yolo/train`, `Data/Yolo/test` (paths accept `\` or `/`).

Auto-detected formats:
- **(A)** text `t x y p label` (0=bg, 1=fg)
- **(B)** Prophesee `*_td.dat` + `*_bbox.npy` (pixel top-left xywh + timestamp) → per-event labels via point-in-box

## Train / test

```bash
cd GVFA_BG

python run.py --mode train \
  --train_dir ../Data/Yolo/train \
  --out_dir /scratch/<proj>/runs/seg_v1 \
  --epochs 50 --lr 1e-3 --window_ms 50 --width 304 --height 240

python run.py --mode test \
  --test_dir ../Data/Yolo/test \
  --ckpt /scratch/<proj>/runs/seg_v1/best.pt \
  --out_dir /scratch/<proj>/runs/seg_v1/test_images \
  --window_ms 50 --width 304 --height 240
```

Headline metric: **foreground IoU** (`best.pt` selected by val fg-IoU).


python run.py --mode train --train_dir ../Data/Yolo/train  --out_dir output/runs/seg_v1 --epochs 50 --lr 1e-3 --window_ms 50

python run.py --mode test --test_dir ../Data/Yolo/test --ckpt /scratch/<proj>/runs/seg_v1/best.pt --out_dir output/seg_v1/test_images --window_ms 50