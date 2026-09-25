"""Label modes: how an EVIMO raw mask (object_id * 1000) becomes training labels.

  motion  -> 0/1, MOVING vs not, from the per-frame object motion in the pose
             metadata (see hdems.data.motion_labels). Objects whose speed falls
             between the two thresholds, and a band around every mask boundary,
             become ignore (255) instead of being forced into a class.
  tracked -> 0/1 with the OLD rule mask > 0 = "on a tracked surface". Kept only
             as a baseline/ablation: on EVIMO2 it marks the static table as
             foreground, so it is NOT motion segmentation.
  objects -> object_id = mask // 1000 (globally consistent ids) ... Option B, identity
  remap   -> legacy: objects renumbered 1..K PER FRAME (ids are not comparable
             across frames/sequences; kept only for backward compatibility)

The mode also fixes the number of classes, so the head size and the metric can
never disagree with the labels.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from hdems.data.motion_labels import IGNORE_LABEL

LABEL_MODES = ("motion", "tracked", "objects", "remap")


def _boundary_band(obj_ids: torch.Tensor, width_px: int) -> torch.Tensor:
    """True within ``width_px`` of any object-id change.

    Mask edges are a few pixels imprecise and the motion of a boundary pixel is
    ambiguous (it belongs to both surfaces during the window), so these pixels
    are ignored rather than counted for or against the model.
    """
    if width_px <= 0:
        return torch.zeros_like(obj_ids, dtype=torch.bool)
    k = 2 * int(width_px) + 1
    x = obj_ids.float().unsqueeze(0).unsqueeze(0)
    hi = F.max_pool2d(x, k, stride=1, padding=k // 2)
    lo = -F.max_pool2d(-x, k, stride=1, padding=k // 2)
    return (hi != lo).squeeze(0).squeeze(0)


def _isin(obj_ids: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
    ids = [int(i) for i in ids]
    if not ids:
        return torch.zeros_like(obj_ids, dtype=torch.bool)
    return torch.isin(obj_ids, torch.tensor(ids, dtype=obj_ids.dtype, device=obj_ids.device))


def to_labels(
    raw: torch.Tensor,
    mode: str = "motion",
    *,
    moving_ids: Iterable[int] | None = None,
    ambiguous_ids: Iterable[int] | None = None,
    boundary_ignore_px: int = 0,
) -> torch.Tensor:
    """Raw EVIMO mask -> integer labels (background = 0, ignore = 255)."""
    raw = raw.long()
    if mode == "motion":
        if moving_ids is None:
            raise ValueError(
                "label_mode='motion' needs the per-frame moving object ids from the "
                "pose metadata (hdems.data.motion_labels.frame_motion). EVIMO2 masks "
                "label the static table too, so 'mask > 0' is not motion -- use "
                "label_mode='tracked' if you really want the old rule."
            )
        obj = raw // 1000
        labels = _isin(obj, moving_ids).long()
        if ambiguous_ids:
            labels = labels.masked_fill(_isin(obj, ambiguous_ids), IGNORE_LABEL)
        if boundary_ignore_px:
            labels = labels.masked_fill(_boundary_band(obj, boundary_ignore_px), IGNORE_LABEL)
        return labels
    if mode == "tracked":                     # legacy "on a tracked surface"
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
    """Class count implied by the label mode (motion/tracked are always binary)."""
    if mode in ("motion", "tracked"):
        return 2
    return int(configured)


def resolve_label_mode(cfg: dict) -> str:
    """Read dataset.label_mode, defaulting to motion."""
    mode = cfg.get("dataset", {}).get("label_mode")
    return str(mode).lower() if mode else "motion"
