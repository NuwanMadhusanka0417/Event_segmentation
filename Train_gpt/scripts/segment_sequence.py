#!/usr/bin/env python3
"""Run VSA motion segmentation on an EVIMO2 sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vsa_motionseg.cli_helpers import apply_device_override, resolve_max_frames
from vsa_motionseg.config import load_config, resolve_device
from vsa_motionseg.data.evimo_adapter import EVIMO2Adapter
from vsa_motionseg.pipeline import VSAMotionSegPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "evimo.yaml"))
    p.add_argument("--input", type=str, default=None, help="Dataset root override")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames (dataset.max_infer_frames or unlimited)")
    p.add_argument("--sequence", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    apply_device_override(cfg, args.device)
    if args.input:
        cfg["dataset"]["root"] = args.input
    split = args.split or cfg["dataset"].get("split", "train")
    seq = args.sequence
    if seq is None:
        seq = int(cfg.get("dataset", {}).get("sequence_index", 0))
    max_frames = resolve_max_frames(args.max_frames, cfg, "max_infer_frames", None)

    adapter = EVIMO2Adapter(cfg["dataset"]["root"], split=split, sequence=seq)
    pipe = VSAMotionSegPipeline(cfg)
    print(f"Device: {resolve_device(cfg.get('runtime', {}).get('device', 'cpu'))}")
    results = pipe.run_sequence(adapter, max_frames=max_frames)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": args.config, "split": split, "frames": len(results)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for i, r in enumerate(results):
        torch_save = {
            "dynamic_mask": r["dynamic_mask"],
            "segment_labels": r["segment_labels"],
            "residual_flow": r["residual_flow"],
            "benchmark": r["benchmark"],
        }
        torch.save(torch_save, out / f"frame_{i:05d}.pt")

    if results:
        print(results[-1]["benchmark"])


if __name__ == "__main__":
    main()
