"""Precompute EVIMO2 (surface, mask) samples into .pt shards for fast training.

Run this ONCE. The training dataset auto-detects the shards
(``EVIMODataset`` -> ``find_cached_samples``) and then every epoch just does
``torch.load`` instead of decoding events and rebuilding time surfaces, so the
per-epoch data cost is paid a single time here.

Shards are written to ``<out>/<split>/<sequence>_<frameidx>.pt`` at the
resolution / window from the config. If you later change height/width/window_ms
or the time-surface decay, DELETE the shards and re-run (cached shards are
resolution-baked and take priority over raw sequences).

Usage
-----
    python scripts/prepare_evimo.py --config configs/evimo_seg.yaml
    python scripts/prepare_evimo.py --config configs/evimo_seg.yaml --splits train eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Make the package importable when run as a plain script (python scripts/...),
# not just via `python -m`. Adds the project root (parent of scripts/) to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hdems.data.evimo2_reader import build_sample_index, load_frame_sample, load_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    ap.add_argument("--out", default=None,
                    help="output root (default: the dataset root from the config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = cfg.get("dataset", {})
    root = Path(ds["root"])
    out_root = Path(args.out) if args.out else root

    height = ds.get("height", 480)
    width = ds.get("width", 640)
    window_s = ds.get("window_ms", 50.0) / 1000.0
    decay = cfg.get("time_surface", {}).get("decay", 0.8)
    remap = ds.get("remap_mask", True)
    fallback = ds.get("use_classical_fallback", True)

    for split in args.splits:
        index = build_sample_index(root, split)
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{split}] {len(index)} samples -> {out_dir}  ({height}x{width})")

        meta_cache: dict[Path, dict] = {}
        for k, (seq_dir, fi) in enumerate(index):
            if seq_dir not in meta_cache:                 # read info.npz once per seq
                meta_cache[seq_dir] = load_meta(seq_dir)
            frame = meta_cache[seq_dir]["frames"][fi]

            sample = load_frame_sample(
                seq_dir, frame,
                out_height=height, out_width=width,
                window_s=window_s, decay=decay,
                remap_mask=remap, use_classical_fallback=fallback,
            )
            out_path = out_dir / f"{seq_dir.name}_{fi:06d}.pt"
            torch.save({"surface": sample["surface"], "mask": sample["mask"]}, out_path)

            if (k + 1) % 200 == 0 or (k + 1) == len(index):
                print(f"  {k + 1}/{len(index)}")

    print("done. Shards ready; training will load them automatically.")


if __name__ == "__main__":
    main()
