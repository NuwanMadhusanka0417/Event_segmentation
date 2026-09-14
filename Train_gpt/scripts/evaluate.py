#!/usr/bin/env python3
"""Evaluate motion segmentation against instance/motion masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vsa_motionseg.config import load_config
from vsa_motionseg.data.evimo_adapter import EVIMO2Adapter
from vsa_motionseg.evaluation.metrics import (
    boundary_f1,
    event_masked_valid,
    iou,
    precision_recall_f1,
    temporal_stability,
)
from vsa_motionseg.pipeline import VSAMotionSegPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "evimo.yaml"))
    p.add_argument("--split", default=None)
    p.add_argument("--max-frames", type=int, default=10)
    args = p.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg["dataset"].get("eval_split", "eval")
    try:
        adapter = EVIMO2Adapter(cfg["dataset"]["root"], split=split)
    except FileNotFoundError:
        print(f"Split '{split}' not found — falling back to train for smoke eval.")
        adapter = EVIMO2Adapter(cfg["dataset"]["root"], split="train")

    pipe = VSAMotionSegPipeline(cfg)
    window_s = float(cfg["events"]["window_ms"]) / 1000.0
    event_masked = cfg["dataset"].get("event_masked_evaluation", True)

    ious, f1s = [], []
    prev_ids = None
    stab = []
    for i in range(min(len(adapter), args.max_frames)):
        fr = adapter.get_frame(i)
        ev = adapter.events_in_window(fr.timestamp - window_s, fr.timestamp)
        r = pipe.process_window(ev, fr.image_height, fr.image_width, fr.timestamp, window_s)
        pred = r["segment_labels"] > 0
        if fr.instance_masks is None:
            continue
        gt = fr.instance_masks > 0
        valid = event_masked_valid(r["event_count"]) if event_masked else torch.ones_like(gt, dtype=torch.bool)
        ious.append(iou(pred, gt, valid))
        f1s.append(precision_recall_f1(pred, gt, valid)["f1"])
        if prev_ids is not None:
            stab.append(temporal_stability(r["segment_labels"], prev_ids, valid))
        prev_ids = r["segment_labels"]
        print(f"frame {i}: IoU={ious[-1]:.3f} F1={f1s[-1]:.3f} boundary_f1={boundary_f1(pred, gt, valid):.3f}")

    if ious:
        print(f"mean IoU={sum(ious)/len(ious):.3f} mean F1={sum(f1s)/len(f1s):.3f}")
        if stab:
            print(f"temporal stability={sum(stab)/len(stab):.3f}")


if __name__ == "__main__":
    main()
