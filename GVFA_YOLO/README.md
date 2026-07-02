qsub -I  -l walltime=15:00:00,mem=100GB,ncpus=7,jobfs=100GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

--------------------------------------------------------------
    python --version
    module avail python
    module load python3/3.11.0 
    pip --version

    # 1. Force pip to use scratch for temporary build artifacts
    export TMPDIR=/scratch/nv2/tmp
    mkdir -p $TMPDIR

    # 2. Force pip to store or look for caches inside your scratch space
    export PIP_CACHE_DIR=/scratch/nv2/pip_cache
    mkdir -p $PIP_CACHE_DIR

    pip install <LIBRARY> --target=/scratch/jq77/nk8155/nv2/lib/python3.11/site-packages --no-cache-dir

------------------------------------------------------------------
source /scratch/jq77/nk8155/nv2/bin/activate

cd "/g/data/jq77/nuwan/Event_segmentation/GVFA_YOLO/"

Extract data 
7z x Data/Prophesee/val_a.7z   -oData/Prophesee/val_a


Step 2 — Train (streaming loader — full train_a OK at ~30–40GB with max_events 15000)
python train.py train --data_dir ../Data/train_a --val_dir ../Data//val_a --window_ms 50 --epochs 10 --max_events 15000 --ckpt checkpoints/head.pt 

Debug run
python train.py train --data_dir ../Data/Prophesee/train_a --max_recordings 2 --max_windows 20 --max_events 15000 --epochs 2

# Request ~40GB should be enough now
python train.py train --data_dir ../Data/train_a --val_dir ..Data/val_a --window_ms 50 --max_events 15000 --epochs 10 --ckpt checkpoints/head.pt

Step 3 — Eval (mAP)
python train.py eval --data_dir ../Data/Prophesee/val_a --ckpt checkpoints/head.pt --score_thr 0.3

Step 4 — Test + save detection frames
python train.py test --data_dir ../Data/Prophesee/val_a --ckpt checkpoints/head.pt --out_dir runs/test_frames --score_thr 0.3


