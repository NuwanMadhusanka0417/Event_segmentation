"""Read EVIMO2v2 NPZ/NPY sequence folders into training samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hdems.data.time_surface import events_to_time_surface


def find_sequence_dirs(root: Path, split: str) -> list[Path]:
    """Return sequence directories under ``root/split/``."""
    split_dir = root / split
    if not split_dir.exists():
        return []
    seqs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
    return [p for p in seqs if (p / "dataset_mask.npz").exists()]


def find_cached_samples(root: Path, split: str) -> list[Path]:
    """Return preprocessed ``.pt`` shards under ``root/split/``."""
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("*.pt"))


def load_meta(seq_dir: Path) -> dict[str, Any]:
    info = np.load(seq_dir / "dataset_info.npz", allow_pickle=True)
    meta = info["meta"].item()
    if not isinstance(meta, dict):
        raise ValueError(f"Unexpected meta type in {seq_dir}")
    return meta


def sensor_size(meta: dict[str, Any], mask_shape: tuple[int, ...]) -> tuple[int, int]:
    inner = meta.get("meta", {})
    res_x = inner.get("res_x")
    res_y = inner.get("res_y")
    if res_x and res_y:
        return int(res_y), int(res_x)
    return int(mask_shape[0]), int(mask_shape[1])


def mask_to_classes(mask: np.ndarray, remap: bool = True) -> np.ndarray:
    """Convert EVIMO mask (object_id * 1000) to integer class labels."""
    mask = mask.astype(np.int64)
    if not remap:
        return mask // 1000

    labels = np.zeros_like(mask, dtype=np.int64)
    unique = sorted(v for v in np.unique(mask) if v > 0)
    for idx, value in enumerate(unique, start=1):
        labels[mask == value] = idx
    return labels


def classical_to_surface(rgb: np.ndarray) -> torch.Tensor:
    """RGB frame -> (2, H, W) pseudo time surface for classical-only sequences."""
    rgb = rgb.astype(np.float32) / 255.0
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray_t = torch.from_numpy(gray)
    return torch.stack([gray_t, gray_t.clone()], dim=0)


def events_window_to_surface(
    seq_dir: Path,
    t_start: float,
    t_end: float,
    height: int,
    width: int,
    *,
    decay: float = 0.8,
) -> torch.Tensor:
    """Build time surface from events in ``[t_start, t_end]``."""
    t_path = seq_dir / "dataset_events_t.npy"
    if not t_path.exists():
        return torch.zeros(2, height, width, dtype=torch.float32)

    t = np.load(t_path, mmap_mode="r").reshape(-1)
    if t.size == 0:
        return torch.zeros(2, height, width, dtype=torch.float32)

    xy = np.load(seq_dir / "dataset_events_xy.npy", mmap_mode="r")
    p = np.load(seq_dir / "dataset_events_p.npy", mmap_mode="r").reshape(-1)

    i0 = int(np.searchsorted(t, t_start, side="left"))
    i1 = int(np.searchsorted(t, t_end, side="right"))
    if i1 <= i0:
        return torch.zeros(2, height, width, dtype=torch.float32)

    events = np.stack(
        [
            t[i0:i1].astype(np.float64),
            xy[i0:i1, 0].astype(np.float64),
            xy[i0:i1, 1].astype(np.float64),
            p[i0:i1].astype(np.float64),
        ],
        axis=1,
    )
    return events_to_time_surface(
        torch.from_numpy(events),
        height,
        width,
        polarity=True,
        decay=decay,
    )


def load_frame_sample(
    seq_dir: Path,
    frame: dict[str, Any],
    *,
    out_height: int | None = None,
    out_width: int | None = None,
    window_s: float = 0.05,
    decay: float = 0.8,
    remap_mask: bool = True,
    use_classical_fallback: bool = True,
) -> dict[str, torch.Tensor]:
    """Load one aligned (surface, mask) training sample from a sequence."""
    masks = np.load(seq_dir / "dataset_mask.npz")
    meta = load_meta(seq_dir)

    frame_id = int(frame["id"])
    ts = float(frame["ts"])
    mask_key = f"mask_{frame_id:010d}"
    if mask_key not in masks.files:
        raise KeyError(f"{mask_key} not found in {seq_dir / 'dataset_mask.npz'}")

    mask_np = masks[mask_key]
    height, width = sensor_size(meta, mask_np.shape)

    t_path = seq_dir / "dataset_events_t.npy"
    has_events = t_path.exists() and np.load(t_path, mmap_mode="r").size > 0

    if has_events:
        surface = events_window_to_surface(
            seq_dir,
            ts - window_s,
            ts,
            height,
            width,
            decay=decay,
        )
    elif use_classical_fallback:
        classical = np.load(seq_dir / "dataset_classical.npz")
        c_key = f"classical_{frame_id:010d}"
        if c_key not in classical.files:
            raise KeyError(f"{c_key} not found in {seq_dir / 'dataset_classical.npz'}")
        surface = classical_to_surface(classical[c_key])
    else:
        raise RuntimeError(f"No events and classical fallback disabled for {seq_dir}")

    mask = torch.from_numpy(mask_to_classes(mask_np, remap=remap_mask)).long()

    if out_height and out_width and (
        surface.shape[-2] != out_height or surface.shape[-1] != out_width
    ):
        surface = F.interpolate(
            surface.unsqueeze(0),
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(out_height, out_width),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

    return {"surface": surface.float(), "mask": mask}


def build_sample_index(root: Path, split: str) -> list[tuple[Path, int]]:
    """List of (sequence_dir, frame_index) for all GT frames in a split."""
    index: list[tuple[Path, int]] = []
    for seq_dir in find_sequence_dirs(root, split):
        meta = load_meta(seq_dir)
        for fi, _frame in enumerate(meta["frames"]):
            index.append((seq_dir, fi))
    return index
