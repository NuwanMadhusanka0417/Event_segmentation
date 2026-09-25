"""EV-IMO / EVIMO2 segmentation dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from hdems.data.evimo2_reader import (
    build_sample_index,
    find_cached_samples,
    load_frame_sample,
    load_meta,
)
from hdems.data.labels import to_labels
from hdems.data.motion_labels import MotionParams, frame_motion


class EVIMODataset(Dataset):
    """EVIMO2 segmentation dataset.

    Layout under ``root``::

        root/train/scene_name/dataset_mask.npz   # raw sequences
        root/eval/scene_name/...
        root/train/sample.pt                     # optional cached shards

    Each sample provides ``surface`` (2, H, W) and ``mask`` (H, W).

    In ``label_mode='motion'`` the labels come from the per-frame object motion in
    the pose metadata, not from ``mask > 0`` (EVIMO2 labels the static table too).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        version: str | None = None,
        *,
        height: int = 480,
        width: int = 640,
        window_ms: float = 50.0,
        decay: float = 0.8,
        time_frames: list[float] | None = None,
        label_mode: str = "motion",
        motion_params: MotionParams | None = None,
        boundary_ignore_px: int = 2,
        exclude_scenes: Iterable[str] = (),
        require_mover: bool = False,
        negative_ratio: float = 0.0,
        interleave: bool = True,
        min_moving_px: int = 100,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.version = version  # unused, kept for config compatibility
        self.height = height
        self.width = width
        self.window_s = window_ms / 1000.0
        self.decay = decay
        self.time_frames = time_frames or None  # [] -> None (single-time)
        self.label_mode = str(label_mode).lower()
        self.motion_params = motion_params or MotionParams(window_s=self.window_s)
        self.boundary_ignore_px = int(boundary_ignore_px)

        self.cached: list[Path] = find_cached_samples(self.root, split)
        self.index: list[tuple[Path, int]] = (
            [] if self.cached else build_sample_index(
                self.root, split,
                exclude_scenes=exclude_scenes,
                require_mover=require_mover and self.label_mode == "motion",
                negative_ratio=negative_ratio,
                interleave=interleave,
                motion_params=self.motion_params,
                min_moving_px=min_moving_px,
                window_s=self.window_s,
            )
        )

    def __len__(self) -> int:
        return len(self.cached) if self.cached else len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.cached and not self.index:
            raise FileNotFoundError(
                f"No EVIMO2 samples under {self.root / self.split}. "
                "Place sequence folders in root/train and root/eval, "
                "or run: python scripts/prepare_evimo.py --config configs/evimo_seg.yaml"
            )

        if self.cached:
            sample = torch.load(self.cached[idx], weights_only=True)
            return self._apply_labels(
                sample,
                moving=sample.get("moving_ids"),
                ambiguous=sample.get("ambiguous_ids"),
            )

        seq_dir, frame_idx = self.index[idx]
        meta = load_meta(seq_dir)
        frame = meta["frames"][frame_idx]
        moving, ambiguous = frame_motion(seq_dir, frame_idx, self.motion_params, meta)
        return self._apply_labels(
            load_frame_sample(
                seq_dir,
                frame,
                out_height=self.height,
                out_width=self.width,
                window_s=self.window_s,
                decay=self.decay,
                time_fracs=self.time_frames,
            ),
            moving=moving,
            ambiguous=ambiguous,
        )

    def _apply_labels(
        self,
        sample: dict[str, Any],
        *,
        moving: Iterable[int] | None,
        ambiguous: Iterable[int] | None,
    ) -> dict[str, Any]:
        """Raw mask -> labels for the chosen label_mode; keep raw ids as gt_raw.

        Shards written before this change already hold derived labels (no
        ``mask_raw`` flag) and are passed through unchanged.
        """
        if not sample.get("mask_raw", False):
            if self.label_mode == "motion":
                raise RuntimeError(
                    "Cached shards hold pre-derived labels but label_mode='motion' needs "
                    "the raw object ids and the per-frame moving set. Delete the .pt shards "
                    "and re-run scripts/prepare_evimo.py."
                )
            return sample
        raw = sample["mask"].long()
        if self.label_mode == "motion" and moving is None:
            raise RuntimeError(
                "label_mode='motion' needs moving ids; rebuild the cached shards "
                "(scripts/prepare_evimo.py) so they carry the per-frame motion state."
            )
        labels = to_labels(
            raw,
            self.label_mode,
            moving_ids=moving,
            ambiguous_ids=ambiguous,
            boundary_ignore_px=self.boundary_ignore_px,
        )
        out = {
            "surface": sample["surface"],
            "mask": labels,
            "gt_raw": raw,                 # every tracked object id (table included)
        }
        if moving is not None:
            # Raw ids of the MOVING objects only, 0 elsewhere. Instance and detection
            # metrics must not count the static table as an object to be found.
            ids = torch.tensor(sorted(int(i) for i in moving), dtype=raw.dtype)
            keep = torch.isin(raw // 1000, ids) if ids.numel() else torch.zeros_like(raw, dtype=torch.bool)
            out["gt_moving"] = raw.masked_fill(~keep, 0)
        return out
