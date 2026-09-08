#!/usr/bin/env python3
"""Convert EVIMO2v2 sequence folders to cached .pt training shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hdems.data.evimo2_reader import build_sample_index, load_frame_sample, load_meta


def convert_split(
    root: Path,
    split: str,
    *,
    height: int,
    width: int,
    window_ms: float,
    overwrite: bool = False,
) -> int:
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_sample_index(root, split)
    if not index:
        print(f"No sequences found in {root / split}")
        return 0

    count = 0
    for seq_dir, frame_idx in index:
        meta = load_meta(seq_dir)
        frame = meta["frames"][frame_idx]
        out_name = f"{seq_dir.name}_f{frame['id']:06d}.pt"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            count += 1
            continue

        sample = load_frame_sample(
            seq_dir,
            frame,
            out_height=height,
            out_width=width,
            window_s=window_ms / 1000.0,
        )
        torch.save(sample, out_path)
        count += 1
        if count % 50 == 0:
            print(f"  {split}: wrote {count} samples...")

    print(f"{split}: {count} samples in {out_dir}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EVIMO2 .pt cache")
    parser.add_argument("--root", type=str, default="../Data/EVIMO2")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    for split in ("train", "eval"):
        convert_split(
            root,
            split,
            height=args.height,
            width=args.width,
            window_ms=args.window_ms,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
