#!/bin/bash
# Redirect Python / PyTorch / Matplotlib caches off the tiny NCI $HOME quota
# onto scratch, so runs don't fail with "Disk quota exceeded".
#
# Usage (before training/eval, or near the top of your PBS job script):
#     source scripts/nci_env.sh
#
# Override the location by exporting CACHE_ROOT first, e.g.
#     CACHE_ROOT=/scratch/jq77/nk8155/.cache source scripts/nci_env.sh

: "${CACHE_ROOT:=/scratch/mi23/nuwan/.cache}"

mkdir -p "$CACHE_ROOT"/matplotlib "$CACHE_ROOT"/torch/kernels "$CACHE_ROOT"/torch_extensions

export XDG_CACHE_HOME="$CACHE_ROOT"                        # generic ~/.cache (pip, etc.)
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"              # matplotlib font/config cache
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch/kernels"  # CUDA JIT kernel cache
export TORCH_HOME="$CACHE_ROOT/torch"                     # torch hub / weights
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"

echo "caches -> $CACHE_ROOT"
