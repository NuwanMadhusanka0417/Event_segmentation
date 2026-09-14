#!/usr/bin/env python3
"""Train static/dynamic HDC prototypes from labeled frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vsa_motionseg.config import load_config
from vsa_motionseg.data.evimo_adapter import EVIMO2Adapter
from vsa_motionseg.pipeline import VSAMotionSegPipeline
from vsa_motionseg.vsa.prototypes import save_prototypes, train_prototypes


def motion_label_from_instance(inst: torch.Tensor) -> torch.Tensor:
    """Dynamic if instance id > 0 (simplified; user may replace with motion GT)."""
    return (inst.reshape(-1) > 0).long()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "evimo.yaml"))
    p.add_argument("--output", default=str(ROOT / "checkpoints" / "prototypes.pt"))
    p.add_argument("--max-frames", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args.config)
    adapter = EVIMO2Adapter(cfg["dataset"]["root"], split=cfg["dataset"]["split"])
    pipe = VSAMotionSegPipeline(cfg)

    Q_all = []
    y_all = []
    for i in range(min(len(adapter), args.max_frames)):
        fr = adapter.get_frame(i)
        if fr.instance_masks is None:
            continue
        window_s = float(cfg["events"]["window_ms"]) / 1000.0
        ev = adapter.events_in_window(fr.timestamp - window_s, fr.timestamp)
        r = pipe.process_window(ev, fr.image_height, fr.image_width, fr.timestamp, window_s)
        d = r["motion_hypervector"].shape[0]
        Q = r["motion_hypervector"].reshape(d, -1).T
        y = motion_label_from_instance(fr.instance_masks)
        m = r["active_mask"].reshape(-1)
        Q_all.append(Q[m])
        y_all.append(y[m])

    if not Q_all:
        raise SystemExit("No training samples — check Data/EVIMO2/train path and masks.")

    Q_cat = torch.cat(Q_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)
    protos = train_prototypes(Q_cat, y_cat, balance=cfg["classifier"].get("balance_classes", True))
    save_prototypes(args.output, protos, meta={"n_samples": int(Q_cat.shape[0])})
    print(f"Saved prototypes to {args.output} ({Q_cat.shape[0]} samples)")


if __name__ == "__main__":
    main()
