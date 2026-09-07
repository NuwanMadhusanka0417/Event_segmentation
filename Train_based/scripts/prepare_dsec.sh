#!/usr/bin/env bash
# Prepare DSEC dataset for HD-EMS training.
# Download from https://dsec.ifi.uzh.ch/ and convert to .pt shards.

set -euo pipefail

ROOT="${1:-data/dsec}"
mkdir -p "$ROOT/train" "$ROOT/val"

echo "DSEC preparation stub."
echo "  1. Download DSEC optical flow sequences"
echo "  2. Convert events + flow labels to time surfaces"
echo "  3. Save as \$ROOT/{train,val}/*.pt"
echo ""
echo "Target layout per sample:"
echo "  { surface: (C,H,W), flow: (2,H,W) }"
