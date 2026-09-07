#!/usr/bin/env bash
# Prepare EV-IMO / EV-IMO2 dataset for segmentation training.

set -euo pipefail

ROOT="${1:-data/evimo}"
VERSION="${2:-evimo2}"
mkdir -p "$ROOT/$VERSION/train" "$ROOT/$VERSION/val"

echo "EV-IMO preparation stub."
echo "  1. Download EV-IMO2 from the official repository"
echo "  2. Convert events + instance masks to time surfaces"
echo "  3. Save as \$ROOT/$VERSION/{train,val}/*.pt"
echo ""
echo "Target layout per sample:"
echo "  { surface: (C,H,W), mask: (H,W) }"
