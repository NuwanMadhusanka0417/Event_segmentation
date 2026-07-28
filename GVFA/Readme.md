ssh nk8155@gadi.nci.org.au

qsub -I  -l walltime=3:00:00,mem=190GB,ncpus=12,jobfs=50GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/GVFA

python segment.py


python segment.py --tau 0.08  --window-ms 30  --num-layers 2  # even fewer objects (~5)

python segment.py --tau 0.18  --window-ms 30    # slightly more objects (~8)


---------------------------------------------------------------
CHECK CACHE

echo $PIP_CACHE_DIR
echo $TMPDIR

which pip
# should show /scratch/jq77/nk8155/seg/bin/pip


IF NOT 
rm -rf ~/.cache/pip
export PIP_CACHE_DIR=/scratch/jq77/nk8155/.pip_cache
export TMPDIR=/scratch/jq77/nk8155/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR
pip install --upgrade pip
pip install pandas
--------------------------------------------------------------------

```
cd /g/data/jq77/nuwan/Event_segmentation/GVFA
# or your local GVFA/ after syncing

# Ablation: raw normal flow (must match old flow behaviour)
python segment.py --input events_filtered.txt --window-ms 1000 --motion-resolver none

# Track A default — Lucas-Kanade local fit
python segment.py --input events_filtered.txt --window-ms 1000 --motion-resolver lk

# Track A upgrade — affine with LK fallback
python segment.py --input events_filtered.txt --window-ms 1000 --motion-resolver affine

# Track B — VSA constraint superposition (+ Hough validation when VSA_VALIDATE)
python segment.py --input events_filtered.txt --window-ms 1000 --motion-resolver vsa

# Paper ablation table: none vs lk vs vsa on ONE window
python segment.py --input events_filtered.txt --window-ms 1000 --compare-resolvers
```

```
--lk-min-support 6
--condition-min-eig 1e-3
--d-vel 512 --vel-grid-n 48
--band-sigma 0.08
--cleanup-topk 5 --cleanup-min-conf 0.05
--vsa-validate false   # skip Hough agreement check (faster)
--w-node-motion 1.0 --w-node-t 0.1
```