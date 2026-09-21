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


def events_multitime_surface(
    seq_dir: Path,
    ts: float,
    window_s: float,
    height: int,
    width: int,
    fracs: list[float],
    *,
    decay: float = 0.8,
) -> torch.Tensor:
    """Stack of accumulative time surfaces at several times inside the window.

    For the window ``[ts - window_s, ts]`` and each fraction ``fr`` in ``fracs``,
    build an accumulative TS ending at ``t_end = ts - window_s + fr*window_s``
    over a lookback of one window. Recent events (near ``t_end``) dominate. The
    first frame (fr=0) is the reference F0; later fractions are the targets the
    paper cost volume matches against. Returns ``(len(fracs), 2, H, W)``.
    """
    n = len(fracs)
    t_path = seq_dir / "dataset_events_t.npy"
    if not t_path.exists():
        return torch.zeros(n, 2, height, width, dtype=torch.float32)
    t = np.load(t_path, mmap_mode="r").reshape(-1)
    if t.size == 0:
        return torch.zeros(n, 2, height, width, dtype=torch.float32)
    xy = np.load(seq_dir / "dataset_events_xy.npy", mmap_mode="r")
    p = np.load(seq_dir / "dataset_events_p.npy", mmap_mode="r").reshape(-1)

    surfaces = []
    for fr in fracs:
        t_end = ts - window_s + fr * window_s
        i0 = int(np.searchsorted(t, t_end - window_s, side="left"))
        i1 = int(np.searchsorted(t, t_end, side="right"))
        if i1 <= i0:
            surfaces.append(torch.zeros(2, height, width, dtype=torch.float32))
            continue
        age = (t_end - t[i0:i1]).astype(np.float64)          # recent -> small age
        events = np.stack(
            [age, xy[i0:i1, 0].astype(np.float64),
             xy[i0:i1, 1].astype(np.float64), p[i0:i1].astype(np.float64)],
            axis=1,
        )
        surfaces.append(
            events_to_time_surface(torch.from_numpy(events), height, width,
                                   polarity=True, decay=decay)
        )
    return torch.stack(surfaces, dim=0)


def load_frame_sample(
    seq_dir: Path,
    frame: dict[str, Any],
    *,
    out_height: int | None = None,
    out_width: int | None = None,
    window_s: float = 0.05,
    decay: float = 0.8,
    time_fracs: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Load one aligned (surface, mask) training sample from a sequence.

    EVENT DATA ONLY — there is no RGB/classical fallback. A sequence without
    events is an error, so a wrong (frame-camera) folder can never be trained on
    silently.

    If ``time_fracs`` is given, ``surface`` is a multi-time stack
    ``(len(time_fracs), 2, H, W)`` for the paper two-time cost volume; otherwise
    a single ``(2, H, W)`` time surface.
    """
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
    if not (t_path.exists() and np.load(t_path, mmap_mode="r").size > 0):
        raise RuntimeError(
            f"No events in {seq_dir}. This pipeline is event-only — point "
            "dataset.root at an event camera folder (e.g. samsung_mono/imo)."
        )

    if time_fracs:
        surface = events_multitime_surface(
            seq_dir, ts, window_s, height, width, time_fracs, decay=decay,
        )
    else:
        surface = events_window_to_surface(
            seq_dir, ts - window_s, ts, height, width, decay=decay,
        )

    # Keep the RAW mask (object_id * 1000). Labels are derived at load time from
    # dataset.label_mode, so switching mode never requires a cache rebuild.
    mask = torch.from_numpy(mask_np.astype(np.int64))

    if out_height and out_width and (
        surface.shape[-2] != out_height or surface.shape[-1] != out_width
    ):
        if surface.dim() == 3:                               # (2, H, W)
            surface = F.interpolate(
                surface.unsqueeze(0), size=(out_height, out_width),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        else:                                                # (T, 2, H, W)
            surface = F.interpolate(
                surface, size=(out_height, out_width),
                mode="bilinear", align_corners=False,
            )
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(out_height, out_width),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

    # mask_raw marks shards that store raw ids (older shards hold derived labels).
    return {"surface": surface.float(), "mask": mask, "mask_raw": True}


def build_sample_index(root: Path, split: str, min_match: float = 0.5) -> list[tuple[Path, int]]:
    """List of (sequence_dir, frame_index) for frames that HAVE a GT mask.

    ``meta["frames"]`` lists every camera frame, but ``dataset_mask.npz`` only
    stores masks for frames with segmentation ground truth. Frames without a
    matching ``mask_<id>`` key are skipped so ``load_frame_sample`` never raises
    a KeyError and ``len(dataset)`` reflects only loadable samples.
    """
    index: list[tuple[Path, int]] = []
    for seq_dir in find_sequence_dirs(root, split):
        meta = load_meta(seq_dir)
        with np.load(seq_dir / "dataset_mask.npz") as masks:
            present = set(masks.files)
        frames = meta["frames"]
        hits = [fi for fi, fr in enumerate(frames)
                if f"mask_{int(fr['id']):010d}" in present]
        # Some EVIMO exports number masks from 0 while the frame ids start from an
        # offset. Then a handful of ids collide by coincidence and would pair a
        # surface with the WRONG mask, so drop the whole sequence.
        ratio = len(hits) / max(len(frames), 1)
        if ratio < min_match:
            print(f"[data] skipping {seq_dir.name}: only {len(hits)}/{len(frames)} "
                  f"frames match a mask ({ratio:.0%}) — inconsistent mask indexing")
            continue
        index.extend((seq_dir, fi) for fi in hits)
    return index
