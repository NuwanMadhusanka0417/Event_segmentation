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

## Commands (from Train_gpt/)

```bash
pip install -r requirements.txt

# 1) Train static/dynamic prototypes (needs masks under train/)
python scripts/train_prototypes.py --config configs/evimo.yaml --output checkpoints/prototypes.pt

# 2) Main inference
python scripts/segment_sequence.py \
  --config configs/evimo.yaml \
  --input ../Data/EVIMO2 \
  --output outputs/evimo_vsa_motionseg

# 3) Evaluation (uses eval split, falls back to train if eval missing)
python scripts/evaluate.py --config configs/evimo.yaml --split eval

# 4) Synthetic / unit tests
pytest tests/ -q
```