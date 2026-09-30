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
from hdems.data.motion_labels import MotionParams, frame_motion, rigid_groups


FLIPS = ((), (-1,), (-2,), (-2, -1))      # none | horizontal | vertical | 180-degree rotation


def flip_sample(sample: dict[str, Any], dims: tuple[int, ...]) -> dict[str, Any]:
    """Flip every image-shaped tensor of a sample (surfaces, labels, masks, GT maps).

    The INPUT surfaces are flipped, not the computed flow: the front end then measures
    the flow of the mirrored scene, so the velocity is mirrored with the image by
    construction (a horizontal flip negates vx). Flipping the flow afterwards would
    not work for the appearance input -- the VSA descriptor of a mirrored surface is
    not the mirrored descriptor, because the random kernel D is not symmetric.
    """
    if not dims:
        return sample
    hw = tuple(sample["mask"].shape[-2:])
    return {k: (torch.flip(v, dims) if isinstance(v, torch.Tensor) and v.dim() >= 2
                and tuple(v.shape[-2:]) == hw else v)
            for k, v in sample.items()}


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
        only_scenes: Iterable[str] | None = None,
        require_mover: bool = False,
        negative_ratio: float = 0.0,
        interleave: bool = True,
        min_moving_px: int = 100,
        score_window_ms: float | None = None,
        use_shards: bool = False,
        augment: bool = False,
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
        self.min_moving_px = int(min_moving_px)
        self.score_window_s = (float(score_window_ms) / 1000.0) if score_window_ms else None
        # random flip / 180-degree rotation per sample (training only): 3 training
        # scenes are little data, and mirrored motion is still valid motion
        self.augment = bool(augment)

        # Pre-built .pt shards bypass EVERY index filter (scene-disjoint split, held-out
        # validation scenes, require_mover, rigid groups) and freeze the time frames,
        # so a stray shard folder would silently change what is trained and scored.
        # They are only used when dataset.use_shards is set.
        shards = find_cached_samples(self.root, split)
        if shards and not use_shards:
            print(f"[data] ignoring {len(shards)} .pt shard(s) in {self.root / split}: they "
                  f"bypass the split/mover filters (set dataset.use_shards: true to use them)")
        self.cached: list[Path] = shards if use_shards else []
        self.index: list[tuple[Path, int]] = (
            [] if self.cached else build_sample_index(
                self.root, split,
                exclude_scenes=exclude_scenes,
                only_scenes=only_scenes,
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
        sample = self._load(idx)
        if self.augment:
            sample = flip_sample(sample, FLIPS[int(torch.randint(len(FLIPS), (1,)))])
        return sample

    def _load(self, idx: int) -> dict[str, Any]:
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
        # moving objects that move TOGETHER are one object (independent motion)
        rigid = rigid_groups(meta, frame_idx, moving) if moving else {}
        return self._apply_labels(
            load_frame_sample(
                seq_dir,
                frame,
                out_height=self.height,
                out_width=self.width,
                window_s=self.window_s,
                decay=self.decay,
                time_fracs=self.time_frames,
                score_window_s=self.score_window_s,
            ),
            moving=moving,
            ambiguous=ambiguous,
            rigid=rigid,
        )

    def _apply_labels(
        self,
        sample: dict[str, Any],
        *,
        moving: Iterable[int] | None,
        ambiguous: Iterable[int] | None,
        rigid: dict[int, int] | None = None,
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
        if sample.get("score_mask") is not None:   # events near the label time
            out["score_mask"] = sample["score_mask"]
        if moving is not None:
            # Raw ids of the MOVING objects only, 0 elsewhere. Instance and detection
            # metrics must not count the static table as an object to be found.
            ids = torch.tensor(sorted(int(i) for i in moving), dtype=raw.dtype)
            keep = torch.isin(raw // 1000, ids) if ids.numel() else torch.zeros_like(raw, dtype=torch.bool)
            out["gt_moving"] = raw.masked_fill(~keep, 0)
            # Independently moving OBJECTS: parts that move rigidly together share
            # one instance id (1..G). This is what object colouring is scored against.
            inst = torch.zeros_like(raw)
            obj = raw // 1000
            for oid, gid in (rigid or {}).items():
                inst[obj == int(oid)] = int(gid)
            if not rigid:                       # e.g. cached shards: fall back to raw ids
                inst = out["gt_moving"] // 1000
            out["gt_instances"] = inst
            # Objects moving too slowly to call (between static_px and move_px): their
            # pixels are IGNORED in training and scoring, but the figures should still
            # show them -- otherwise a slowly moving object looks like "nothing".
            slow_ids = torch.tensor(sorted(int(i) for i in (ambiguous or ())), dtype=raw.dtype)
            slow = torch.isin(obj, slow_ids) if slow_ids.numel() else torch.zeros_like(raw, dtype=torch.bool)
            out["gt_slow"] = obj.masked_fill(~slow, 0)
        return out
