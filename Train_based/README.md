# HD-EMS

Hyperdimensional Event Motion Segmentation — VSA-based event camera flow and segmentation.

## Quick start

```bash
ssh nk8155@gadi.nci.org.au

cd Train_based
pip install -r requirements.txt
pytest tests/ -v

module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/Train_based


qsub -I -l walltime=12:00:00,mem=190GB,ncpus=12,jobfs=50GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

## EVIMO2 dataset layout

Place extracted EVIMO2v2 sequences here:

```
../Data/EVIMO2/
├── train/
│   └── scene10_dyn_train_00_000000/
│       ├── dataset_events_t.npy
│       ├── dataset_events_xy.npy
│       ├── dataset_events_p.npy
│       ├── dataset_mask.npz
│       ├── dataset_classical.npz   # used when events empty (e.g. flea3_7)
│       └── dataset_info.npz
└── eval/
    └── scene13_dyn_test_00_000000/
        └── ...
```

Config: `configs/evimo_seg.yaml` → `dataset.root: ../Data/EVIMO2`

## Train and evaluate

### CNN segmentation head (trained)

```bash
source scripts/nci_env.sh
python -m hdems.train --config configs/evimo_seg.yaml --device cuda

# Eval metrics + PNG panels (events | GT | prediction)
python -m hdems.eval --config configs/evimo_seg.yaml --checkpoint checkpoints/last.pt \
  --device cuda --save-images eval_out --max-images 50
```

Eval image options (all segmentation eval commands):

- `--save-images <dir>` — output folder for `eval_00000.png`, … (created if missing). Omit to skip PNGs.
- `--max-images N` — cap how many panels to write (default `50`).

Use a scratch path on Gadi if quota is tight, e.g. `--save-images /scratch/mi23/nuwan/eval_out`.

### Ridge readout (closed-form, no backprop)

```bash
python scripts/fit_ridge_head.py --config configs/evimo_seg.yaml --out checkpoints/ridge_head.pt

# Fast smoke run (first N frames only; also set ridge.alphas: [1.0] in yaml)
python scripts/fit_ridge_head.py --config configs/evimo_seg.yaml --device cuda \
  --max-train-samples 30 --max-val-samples 10 --out checkpoints/ridge_smoke.pt

python -m hdems.eval --config configs/evimo_seg.yaml --head ridge \
  --ridge-checkpoint checkpoints/ridge_head.pt --device cuda \
  --save-images eval_out/ridge --max-images 50

# Side-by-side CNN vs Ridge metrics + panels → <dir>/cnn/ and <dir>/ridge/
python -m hdems.eval --config configs/evimo_seg.yaml --checkpoint checkpoints/best.pt \
  --compare-heads --ridge-checkpoint checkpoints/ridge_head.pt --device cuda \
  --save-images output/ridge_compare --max-images 50
```

Set `segmentation.head: ridge` in `configs/evimo_seg.yaml` to make ridge the default eval head.
See `docs/RIDGE_RESULTS.md` for the supervisor-facing metrics table.



### Fair VSA-first settings (unchanged from before)

For the GVFA event pipeline (GVFA/segment.py), not HD-EMS:

```bash
python segment.py --input events_filtered.txt --stream-segments --segment-ms 60 --motion-resolver vsa --assignment em --em-init vsa --out-dir diag_stream/vsa_best
```



## Notes

- Raw NPZ/NPY sequences are loaded **directly** — no manual `.pt` conversion required.
- Event cameras: builds time surfaces from events in a 50 ms window.
- RGB-only sequences (`flea3_7`): falls back to classical frames as a 2-channel pseudo-surface.
- Masks are remapped to consecutive class IDs (background = 0).

See `docs/SPEC.md` and `.cursorrules` for VSA constraints.