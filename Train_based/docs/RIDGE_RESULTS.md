# Ridge readout vs CNN segmentation head

Fill in after running fit + eval on EVIMO2.

## Setup

```bash
cd Train_based
python scripts/fit_ridge_head.py --config configs/evimo_seg.yaml --out checkpoints/ridge_head.pt

CUDA_VISIBLE_DEVICES="" python -m hdems.eval --config configs/evimo_seg.yaml \
  --checkpoint checkpoints/best.pt --head ridge \
  --ridge-checkpoint checkpoints/ridge_head.pt --device cpu

python -m hdems.eval --config configs/evimo_seg.yaml \
  --checkpoint checkpoints/best.pt --compare-heads \
  --ridge-checkpoint checkpoints/ridge_head.pt --save-images output/ridge_compare
```

## Metrics (eval split)

| Head | mIoU | CE loss | Latency (ms/frame, CPU) | Head params |
|------|------|---------|-------------------------|-------------|
| CNN  |      |         |                         | ~500k       |
| Ridge|      |         |                         | D×C         |

## Ridge fit

- Chosen `lambda` (alpha):
- Imbalance strategy:
- Feature dim D:
- mean_center / motion_features:

## One-line conclusion

_(Is the closed-form linear readout competitive with the trained CNN on the same frozen HD features?)_
