# Gadi GPU / CUDA — Quick Reference

HD-EMS training needs a **GPU node** and `--device cuda`. A CPU-only `qsub` (no `ngpus`) will not use CUDA.

## 1. Request an interactive GPU session

**V100 (gpuvolta):**

```bash
qsub -I -q gpuvolta -l ncpus=12,mem=64GB,ngpus=1,walltime=12:00:00 -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

**A100 (gpuhopper):**

```bash
qsub -I -q gpuhopper -l ncpus=12,mem=64GB,ngpus=1,walltime=12:00:00 -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

## 2. Load environment

```bash
module load cuda
module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/Train_based
```

## 3. Check CUDA is available

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True` and a GPU name (e.g. `Tesla V100`).

## 4. Train and evaluate (GPU)

```bash
python -m hdems.train --config configs/evimo_seg.yaml --device cuda
python -m hdems.eval  --config configs/evimo_seg.yaml --device cuda
```

## 5. CPU only (no GPU — slow, may OOM)

```bash
qsub -I -l walltime=12:00:00,mem=190GB,ncpus=12,jobfs=50GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

```bash
python -m hdems.train --config configs/evimo_seg.yaml --device cpu
```

## 6. Batch job (optional)

Save as `train_gpu.pbs`:

```bash
#!/bin/bash
#PBS -P mi23
#PBS -q gpuvolta
#PBS -l ncpus=12,mem=64GB,ngpus=1,walltime=12:00:00
#PBS -l storage=gdata/jq77+scratch/jq77+scratch/mi23
#PBS -j oe

module load cuda
module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/Event_segmentation/Train_based
python -m hdems.train --config configs/evimo_seg.yaml --device cuda
```

Submit:

```bash
qsub train_gpu.pbs
```
