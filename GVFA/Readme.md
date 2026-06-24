ssh nk8155@gadi.nci.org.au

qsub -I  -l walltime=15:00:00,mem=30GB,ncpus=12,jobfs=30GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /g/data/jq77/nuwan/Event_segmentation/GVFA

python segment.py


python segment.py --tau 0.12  --window-ms 30

python segment.py --tau 0.18  --window-ms 30


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