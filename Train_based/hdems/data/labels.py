"""Label modes: how an EVIMO raw mask (object_id * 1000) becomes training labels.

  motion  -> 0/1  (background vs any moving object)   ... Option A, motion segmentation
  objects -> object_id = mask // 1000 (globally consistent ids) ... Option B, identity
  remap   -> legacy: objects renumbered 1..K PER FRAME (ids are not comparable
             across frames/sequences; kept only for backward compatibility)

The mode also fixes the number of classes, so the head size and the metric can
never disagree with the labels.
"""

from __future__ import annotations

import torch

LABEL_MODES = ("motion", "objects", "remap")


def to_labels(raw: torch.Tensor, mode: str = "motion") -> torch.Tensor:
    """Raw EVIMO mask -> integer labels (background = 0)."""
    raw = raw.long()
    if mode == "motion":
        return (raw > 0).long()
    if mode == "objects":
        return raw // 1000
    if mode == "remap":                       # legacy per-frame renumbering
        labels = torch.zeros_like(raw)
        uniq = [int(v) for v in torch.unique(raw).tolist() if v > 0]
        for idx, value in enumerate(sorted(uniq), start=1):
            labels[raw == value] = idx
        return labels
    raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {mode!r}")


def num_classes_for(mode: str, configured: int = 32) -> int:
    """Class count implied by the label mode (motion is always binary)."""
    if mode == "motion":
        return 2
    return int(configured)


def resolve_label_mode(cfg: dict) -> str:
    """Read dataset.label_mode, falling back to the legacy remap_mask flag."""
    mode = cfg.get("dataset", {}).get("label_mode")
    return str(mode).lower() if mode else "motion"
