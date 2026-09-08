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

```bash
python -m hdems.train --config configs/evimo_seg.yaml
python -m hdems.eval --config configs/evimo_seg.yaml
```

Optional `.pt` cache (faster reload):

```bash
python scripts/prepare_evimo.py --root ../Data/EVIMO2
```

## Notes

- Raw NPZ/NPY sequences are loaded **directly** — no manual `.pt` conversion required.
- Event cameras: builds time surfaces from events in a 50 ms window.
- RGB-only sequences (`flea3_7`): falls back to classical frames as a 2-channel pseudo-surface.
- Masks are remapped to consecutive class IDs (background = 0).

See `docs/SPEC.md` and `.cursorrules` for VSA constraints.
