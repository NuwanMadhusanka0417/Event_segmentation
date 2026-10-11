#!/bin/bash
# Reinstall the Python packages that lost files to the NCI /scratch 100-day file
# expiry (e.g. "Unable to find torch_shm_manager"), keeping the exact installed
# versions. Only the broken packages are touched (--no-deps --force-reinstall).
#
# Run on a Gadi LOGIN node (compute nodes have no internet), from Train_nocnn/:
#     bash scripts/repair_env.sh [path/to/venv/bin/activate]
#
# torch is a 1-2.5 GB download. If the login node's limits stop it, run this same
# command in a copyq job (qsub -I -q copyq -P mi23 -l walltime=1:00:00,mem=8GB
# -l storage=gdata/jq77+scratch/jq77+scratch/mi23).
#
# Files quarantined less than 14 days ago can instead be restored as they were:
#     nci-file-expiry list-quarantined
#     nci-file-expiry recover <UUID>

set -eo pipefail
VENV="${1:-/scratch/jq77/nk8155/seg1/bin/activate}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

module load python3/3.9.2 2>/dev/null || true
source "${VENV}"
source scripts/nci_env.sh                 # pip's cache off the small $HOME quota
export TMPDIR="${CACHE_ROOT}/pip-tmp"     # unpacking torch needs several GB
mkdir -p "${TMPDIR}"

# --venv: stop if `python` is not this venv's own (a renamed venv's bin/activate
# still points at its old folder) -- never reinstall into the wrong venv
python scripts/check_env.py --venv "${VENV}" --pip-specs > "${TMPDIR}/broken.txt"
mapfile -t SPECS < "${TMPDIR}/broken.txt"
if [ "${#SPECS[@]}" -eq 0 ]; then
  echo ">>> no package has missing files -- nothing to reinstall"
fi
for line in "${SPECS[@]}"; do
  spec="${line%%$'\t'*}"                  # name==version
  index="${line#*$'\t'}"                  # extra index for +cuXXX builds ('' = PyPI)
  echo ">>> reinstalling ${spec}${index:+  (from ${index})}"
  python -m pip install --no-deps --force-reinstall --no-cache-dir \
    ${index:+--extra-index-url "${index}"} "${spec}"
done

echo ">>> checking again"
python scripts/check_env.py --venv "${VENV}" --config configs/evimo_seg.yaml
