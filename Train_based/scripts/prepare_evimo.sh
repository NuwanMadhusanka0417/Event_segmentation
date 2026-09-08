#!/usr/bin/env bash
# Prepare EVIMO2 for HD-EMS training.
#
# Expected raw layout (after extracting NPZ from better-flow.github.io/evimo):
#   ../Data/EVIMO2/train/scene_name/dataset_mask.npz
#   ../Data/EVIMO2/eval/scene_name/...
#
# Optional: cache to .pt shards for faster loading:
#   python scripts/prepare_evimo.py --root ../Data/EVIMO2

set -euo pipefail

ROOT="${1:-../Data/EVIMO2}"
mkdir -p "$ROOT/train" "$ROOT/eval"

echo "EVIMO2 data root: $ROOT"
echo "  train/  - training sequences (one folder per scene)"
echo "  eval/   - evaluation sequences"
echo ""
echo "Then run:"
echo "  python -m hdems.train --config configs/evimo_seg.yaml"
echo "  python -m hdems.eval --config configs/evimo_seg.yaml"
