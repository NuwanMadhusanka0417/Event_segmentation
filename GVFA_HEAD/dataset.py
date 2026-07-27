"""Causal 50 ms event windows for supervised FG/BG segmentation (clip-level).

Supports two label formats (auto-detected):
  (A) Per-event label column: rows `t x y p label`
  (B) Prophesee / YOLO-style: `*_td.dat` events + `*_bbox.npy` boxes
      (pixel top-left xywh + timestamp) → per-event labels via point-in-box
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from seg_model import GraphCfg, build_causal_graph

EVENT_EXTS = {".txt", ".csv", ".dat", ".npy"}
TEXT_EXTS = {".txt", ".csv"}

# Prophesee DAT v2 packed-field masks
_DAT_X_MASK = 2**14 - 1
_DAT_Y_MASK = 2**28 - 2**14
_DAT_P_MASK = 2**29 - 2**28
_DAT_EV_DTYPE = [("t", "u4"), ("_", "i4")]

FORMAT_A = "A_per_event_label"
FORMAT_B = "B_yolo_bbox"
FORMAT_UNKNOWN = "unknown"


def _norm_path(p: str | Path) -> Path:
    """Accept both \\ and / path separators."""
    return Path(str(p).replace("\\", "/")).expanduser().resolve()


# ---------------------------------------------------------------------------
# Prophesee DAT / bbox I/O (format B)
# ---------------------------------------------------------------------------
def _parse_dat_header(f) -> tuple[int, int, int, dict]:
    """Return (body_offset_after_type, event_type, event_size, meta)."""
    meta: dict = {"width": None, "height": None}
    num_comment = 0
    while True:
        bod = f.tell()
        line = f.readline()
        if not line.startswith(b"% "):
            f.seek(bod)
            break
        num_comment += 1
        text = line.decode("ascii", "replace").strip()
        if "Height" in text:
            try:
                meta["height"] = int(text.split()[-1])
            except ValueError:
                pass
        if "Width" in text:
            try:
                meta["width"] = int(text.split()[-1])
            except ValueError:
                pass
    if num_comment > 0:
        ev_type = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        ev_type = 0
        ev_size = 8
    return f.tell(), ev_type, ev_size, meta


def load_prophesee_dat(path: str | Path) -> tuple[dict, tuple[int, int]]:
    """Load Prophesee *_td.dat → events with t in seconds, + (width, height)."""
    path = _norm_path(path)
    with path.open("rb") as f:
        _, ev_type, _, meta = _parse_dat_header(f)
        if ev_type not in (0, 12):
            raise ValueError(f"{path}: unsupported DAT event type {ev_type}")
        raw = np.fromfile(f, dtype=np.dtype(_DAT_EV_DTYPE))
    if raw.size == 0:
        raise ValueError(f"{path}: empty DAT body")
    t_us = raw["t"].astype(np.float64)
    x = np.bitwise_and(raw["_"], _DAT_X_MASK).astype(np.float64)
    y = np.right_shift(np.bitwise_and(raw["_"], _DAT_Y_MASK), 14).astype(np.float64)
    p = np.right_shift(np.bitwise_and(raw["_"], _DAT_P_MASK), 28).astype(np.float64)
    # Relative seconds for the rest of the pipeline
    t = (t_us - t_us[0]) * 1e-6
    w = int(meta["width"] or 304)
    h = int(meta["height"] or 240)
    order = np.argsort(t, kind="stable")
    return {
        "t": t[order], "x": x[order], "y": y[order], "p": p[order],
        "t_us0": float(t_us[0]),
        "path": path,
        "clip": path.stem.replace("_td", ""),
    }, (w, h)


def load_prophesee_bbox(path: str | Path, t_us0: float = 0.0) -> np.ndarray:
    """Load structured bbox.npy; return array with relative-second intervals.

    Boxes are pixel top-left xywh (Prophesee Gen1). Timestamps are µs in the
    same relative frame as the paired DAT (events start near 0).
    """
    path = _norm_path(path)
    boxes = np.load(str(path), allow_pickle=False)
    if boxes.dtype.names is None:
        raise ValueError(f"{path}: expected structured bbox array, got {boxes.dtype}")
    names = set(boxes.dtype.names)
    need = {"x", "y", "w", "h"}
    if not need.issubset(names):
        raise ValueError(f"{path}: bbox fields {boxes.dtype.names}; need {need}")
    ts_key = "ts" if "ts" in names else ("t" if "t" in names else None)
    if ts_key is None:
        raise ValueError(f"{path}: no timestamp field (ts/t) in {boxes.dtype.names}")

    # Detect units: normalized vs pixel
    x = boxes["x"].astype(np.float64)
    y = boxes["y"].astype(np.float64)
    w = boxes["w"].astype(np.float64)
    h = boxes["h"].astype(np.float64)
    if np.nanmax(w) <= 1.5 and np.nanmax(h) <= 1.5 and np.nanmax(np.abs(x)) <= 1.5:
        raise ValueError(
            f"{path}: boxes look normalized (cxcywh in [0,1]); "
            "this loader expects Prophesee pixel top-left xywh"
        )

    ts_us = boxes[ts_key].astype(np.float64)
    # Same relative frame as DAT events (both start near 0 within the clip)
    t_s = ts_us * 1e-6

    # Build per-box time intervals via midpoints of unique timestamps
    order = np.argsort(t_s, kind="stable")
    t_sorted = t_s[order]
    uniq_t, first_idx = np.unique(t_sorted, return_index=True)
    if len(uniq_t) == 1:
        half = 1.0 / 60.0  # ~one 60 Hz frame
        intervals = {float(uniq_t[0]): (float(uniq_t[0] - half), float(uniq_t[0] + half))}
    else:
        mids = np.empty(len(uniq_t) + 1, dtype=np.float64)
        mids[0] = uniq_t[0] - 0.5 * (uniq_t[1] - uniq_t[0])
        mids[1:-1] = 0.5 * (uniq_t[:-1] + uniq_t[1:])
        mids[-1] = uniq_t[-1] + 0.5 * (uniq_t[-1] - uniq_t[-2])
        intervals = {
            float(uniq_t[i]): (float(mids[i]), float(mids[i + 1]))
            for i in range(len(uniq_t))
        }

    # Pack as plain float array: t0, t1, x, y, w, h  (top-left xywh, seconds)
    out = np.empty((len(boxes), 6), dtype=np.float64)
    for i, b in enumerate(boxes):
        ts = float(b[ts_key]) * 1e-6
        t0, t1 = intervals[ts]
        out[i] = (t0, t1, float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
    return out


def labels_from_boxes(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, boxes: np.ndarray,
) -> np.ndarray:
    """Per-event FG=1 if (x,y) inside any box whose [t0,t1) contains t."""
    labels = np.zeros(len(t), dtype=np.int64)
    if boxes is None or len(boxes) == 0 or len(t) == 0:
        return labels
    # Group boxes by identical (t0, t1) so we searchsorted once per interval
    order = np.argsort(boxes[:, 0], kind="stable")
    boxes = boxes[order]
    i = 0
    n = len(boxes)
    while i < n:
        t0, t1 = boxes[i, 0], boxes[i, 1]
        j = i + 1
        while j < n and boxes[j, 0] == t0 and boxes[j, 1] == t1:
            j += 1
        group = boxes[i:j]
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="left"))
        if i1 > i0:
            xi = x[i0:i1]
            yi = y[i0:i1]
            inside = np.zeros(i1 - i0, dtype=bool)
            for _, _, bx, by, bw, bh in group:
                inside |= (
                    (xi >= bx) & (xi < bx + bw) & (yi >= by) & (yi < by + bh)
                )
            labels[i0:i1][inside] = 1
        i = j
    return labels


# ---------------------------------------------------------------------------
# Format detection + loading
# ---------------------------------------------------------------------------
def detect_data_format(directory: str | Path) -> str:
    """Return FORMAT_A / FORMAT_B / FORMAT_UNKNOWN after inspecting files."""
    root = _norm_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    td_files = sorted(root.glob("*_td.dat"))
    bbox_files = sorted(root.glob("*_bbox.npy"))
    if td_files and bbox_files:
        # Confirm a pairing exists
        stems = {p.name.replace("_td.dat", "") for p in td_files}
        bbox_stems = {p.name.replace("_bbox.npy", "") for p in bbox_files}
        if stems & bbox_stems:
            return FORMAT_B

    text_files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in TEXT_EXTS
    )
    for path in text_files[:3]:
        try:
            sample = np.loadtxt(str(path), max_rows=5)
        except Exception:
            continue
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        if sample.shape[1] >= 5:
            return FORMAT_A
        if sample.shape[1] == 4:
            # unlabeled events — still format A family without labels
            return FORMAT_A

    # Lone .dat without bbox?
    dat_files = sorted(p for p in root.iterdir() if p.suffix.lower() == ".dat")
    if dat_files and not bbox_files:
        return FORMAT_UNKNOWN
    return FORMAT_UNKNOWN


def describe_unknown_format(directory: str | Path) -> str:
    root = _norm_path(directory)
    files = sorted(p.name for p in root.iterdir() if p.is_file())[:30]
    lines = [f"Unrecognized data format in {root}", f"Files (first {len(files)}):"]
    for name in files:
        lines.append(f"  - {name}")
    lines.append(
        "Expected either:\n"
        "  (A) text files with columns `t x y p [label]`\n"
        "  (B) paired `*_td.dat` + `*_bbox.npy` (Prophesee / YOLO boxes)"
    )
    return "\n".join(lines)


def list_clips(directory: str | Path, fmt: str | None = None) -> list[Path]:
    """List clip event files for the detected format."""
    root = _norm_path(directory)
    fmt = fmt or detect_data_format(root)
    if fmt == FORMAT_B:
        pairs = []
        for td in sorted(root.glob("*_td.dat")):
            stem = td.name.replace("_td.dat", "")
            bbox = root / f"{stem}_bbox.npy"
            if bbox.is_file():
                pairs.append(td)
        if not pairs:
            raise FileNotFoundError(f"no *_td.dat + *_bbox.npy pairs in {root}")
        return pairs
    if fmt == FORMAT_A:
        files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in TEXT_EXTS
        )
        if not files:
            raise FileNotFoundError(f"no text event files in {root}")
        return files
    raise RuntimeError(describe_unknown_format(root))


def list_event_files(directory: str | Path) -> list[Path]:
    """Back-compat alias used by summarize / test loops."""
    fmt = detect_data_format(directory)
    if fmt == FORMAT_UNKNOWN:
        raise RuntimeError(describe_unknown_format(directory))
    return list_clips(directory, fmt)


def load_event_file(
    path: str | Path,
    *,
    require_labels: bool = False,
    derive_labels: bool = True,
) -> dict:
    """Load one clip. Auto-detects Prophesee DAT vs text `t x y p [label]`.

    For format B, boxes are always loaded when present. Per-event labels are
    derived immediately only if derive_labels=True (expensive on long clips);
    otherwise callers can label window slices via labels_from_boxes.
    """
    path = _norm_path(path)

    if path.name.endswith("_td.dat") or path.suffix.lower() == ".dat":
        try:
            with path.open("rb") as f:
                head = f.read(2)
        except OSError as e:
            raise ValueError(f"{path}: cannot read ({e})") from e
        if head == b"% " or path.name.endswith("_td.dat"):
            ev, sensor = load_prophesee_dat(path)
            bbox_path = path.with_name(
                path.name.replace("_td.dat", "_bbox.npy")
                if path.name.endswith("_td.dat")
                else f"{path.stem}_bbox.npy"
            )
            if bbox_path.is_file():
                boxes = load_prophesee_bbox(bbox_path, t_us0=ev.get("t_us0", 0.0))
                ev["boxes"] = boxes
                if derive_labels:
                    ev["label"] = labels_from_boxes(
                        ev["t"], ev["x"], ev["y"], boxes)
            elif require_labels:
                raise ValueError(f"{path}: missing paired bbox {bbox_path.name}")
            ev["sensor"] = sensor
            ev["format"] = FORMAT_B
            return ev

    if path.suffix.lower() not in TEXT_EXTS:
        raise ValueError(f"{path}: unsupported event file type")

    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path}: expected >=4 columns, got {data.shape[1]}")
    t = data[:, 0].astype(np.float64)
    if np.nanmax(t) > 1e5:
        t = t * 1e-6
    x = data[:, 1].astype(np.float64)
    y = data[:, 2].astype(np.float64)
    p = data[:, 3].astype(np.float64)
    labels = data[:, 4].astype(np.int64) if data.shape[1] >= 5 else None
    if require_labels and labels is None:
        raise ValueError(f"{path}: training requires a label column")
    order = np.argsort(t, kind="stable")
    out = {
        "t": t[order], "x": x[order], "y": y[order], "p": p[order],
        "path": path, "clip": path.stem, "format": FORMAT_A,
    }
    if labels is not None:
        out["label"] = labels[order]
    return out


def window_indices(
    t: np.ndarray,
    window_ms: float = 50.0,
    stride_ms: float | None = None,
) -> list[tuple[int, int, float, float]]:
    """Non-overlapping (or strided) causal windows. Returns (i0, i1, t0, t1)."""
    if len(t) == 0:
        return []
    w_s = window_ms * 1e-3
    s_s = w_s if stride_ms is None else stride_ms * 1e-3
    t0 = float(t[0])
    t_end = float(t[-1])
    windows = []
    start = t0
    while start <= t_end + 1e-12:
        stop = start + w_s
        i0 = int(np.searchsorted(t, start, side="left"))
        i1 = int(np.searchsorted(t, stop, side="left"))
        if i1 > i0:
            windows.append((i0, i1, start, stop))
        start += s_s
        if s_s <= 0:
            break
    return windows


def slice_events(ev: dict, i0: int, i1: int) -> dict:
    out = {
        "t": ev["t"][i0:i1].copy(),
        "x": ev["x"][i0:i1].copy(),
        "y": ev["y"][i0:i1].copy(),
        "p": ev["p"][i0:i1].copy(),
    }
    if "label" in ev:
        out["label"] = ev["label"][i0:i1].copy()
    elif "boxes" in ev:
        out["label"] = labels_from_boxes(
            out["t"], out["x"], out["y"], ev["boxes"])
    return out


def fg_bg_ratio(labels: np.ndarray | None) -> tuple[int, int, float]:
    if labels is None or len(labels) == 0:
        return 0, 0, float("nan")
    n_fg = int((labels == 1).sum())
    n_bg = int((labels == 0).sum())
    ratio = n_fg / max(n_bg, 1)
    return n_fg, n_bg, ratio


def summarize_dir(directory: str | Path, window_ms: float = 50.0) -> str:
    """Detect format, print per-clip window counts and fg/bg ratios. Returns fmt."""
    root = _norm_path(directory)
    fmt = detect_data_format(root)
    if fmt == FORMAT_UNKNOWN:
        print(describe_unknown_format(root))
        raise RuntimeError(f"unsupported data format in {root}")

    if fmt == FORMAT_A:
        print(f"[data] detected format (A) per-event label column  in {root}")
    else:
        print(
            f"[data] detected format (B) YOLO/Prophesee bounding boxes  "
            f"(*_td.dat + *_bbox.npy, pixel top-left xywh)  in {root}"
        )

    files = list_clips(root, fmt)
    print(f"[data] {root}: {len(files)} clip(s)  (split is by clip/file)")
    if len(files) == 1:
        print("[data] WARNING: single clip only — treat as dev/sanity, "
              "not a generalization result")
    for path in files:
        # Defer full-clip labeling; label a stride of windows for the ratio.
        ev = load_event_file(path, require_labels=False, derive_labels=False)
        wins = window_indices(ev["t"], window_ms=window_ms)
        labels = ev.get("label")
        if labels is None and "boxes" in ev:
            # Sample up to 40 windows for a quick fg/bg estimate
            sample_wins = wins[:: max(1, len(wins) // 40)][:40]
            chunks = []
            for i0, i1, _, _ in sample_wins:
                chunks.append(labels_from_boxes(
                    ev["t"][i0:i1], ev["x"][i0:i1], ev["y"][i0:i1], ev["boxes"]))
            labels = np.concatenate(chunks) if chunks else None
        n_fg, n_bg, ratio = fg_bg_ratio(labels)
        lab = "labeled" if (labels is not None or "boxes" in ev) else "unlabeled"
        sensor = ev.get("sensor")
        sens = f" sensor={sensor[0]}x{sensor[1]}" if sensor else ""
        print(
            f"  {path.name}: {len(ev['t'])} events, {len(wins)} windows "
            f"({window_ms:.0f} ms), {lab}, fg={n_fg} bg={n_bg} "
            f"fg/bg={ratio:.4f}{sens}"
        )
    return fmt


class EventWindowDataset(Dataset):
    """One item = one causal window graph (+ labels when present)."""

    def __init__(
        self,
        directory: str | Path,
        cfg: GraphCfg,
        window_ms: float = 50.0,
        stride_ms: float | None = None,
        require_labels: bool = True,
        min_events: int = 8,
        clip_files: list[Path] | None = None,
    ):
        self.cfg = cfg
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.require_labels = require_labels
        self.min_events = min_events
        root = _norm_path(directory)
        self.fmt = detect_data_format(root)
        if self.fmt == FORMAT_UNKNOWN:
            raise RuntimeError(describe_unknown_format(root))
        self.files = clip_files if clip_files is not None else list_clips(root, self.fmt)
        self.index: list[tuple[int, int, int, float, float]] = []
        self._cache: dict[int, dict] = {}

        for fi, path in enumerate(self.files):
            ev = load_event_file(
                path, require_labels=require_labels, derive_labels=False)
            has_labels = ("label" in ev) or ("boxes" in ev)
            if require_labels and not has_labels:
                raise ValueError(f"{path}: training requires labels")
            if "sensor" in ev:
                sw, sh = ev["sensor"]
                if (sw, sh) != (cfg.width, cfg.height):
                    print(
                        f"[data] WARNING: clip sensor {sw}x{sh} != "
                        f"cfg {cfg.width}x{cfg.height} ({path.name})"
                    )
            self._cache[fi] = ev
            for i0, i1, t0, t1 in window_indices(ev["t"], window_ms, stride_ms):
                if (i1 - i0) >= min_events:
                    self.index.append((fi, i0, i1, t0, t1))

        if not self.index:
            raise RuntimeError(f"no windows with >= {min_events} events in {directory}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        fi, i0, i1, t0, t1 = self.index[idx]
        ev = self._cache[fi]
        sl = slice_events(ev, i0, i1)
        graph = build_causal_graph(sl, self.cfg)
        item = {
            "graph": graph,
            "clip": self.files[fi].stem.replace("_td", ""),
            "window_index": idx,
            "file_window": (fi, i0, i1),
            "t_start_ms": (t0 - float(ev["t"][0])) * 1e3,
            "t0": t0,
            "t1": t1,
            "n_events": i1 - i0,
        }
        if "label" in sl:
            order = graph["order"]
            item["label"] = torch.from_numpy(sl["label"][order].astype(np.float32))
        return item


def iter_windows_streaming(
    path: str | Path,
    cfg: GraphCfg,
    window_ms: float = 50.0,
    stride_ms: float | None = None,
    min_events: int = 1,
) -> Iterator[dict]:
    """Yield windows in time order without buffering all predictions."""
    ev = load_event_file(path, require_labels=False, derive_labels=False)
    wins = window_indices(ev["t"], window_ms, stride_ms)
    for wi, (i0, i1, t0, t1) in enumerate(wins):
        if (i1 - i0) < min_events:
            continue
        sl = slice_events(ev, i0, i1)
        graph = build_causal_graph(sl, cfg)
        out = {
            "graph": graph,
            "clip": Path(path).stem.replace("_td", ""),
            "window_index": wi,
            "t_start_ms": (t0 - float(ev["t"][0])) * 1e3,
            "t0": t0,
            "t1": t1,
            "has_labels": "label" in sl,
        }
        if "label" in sl:
            out["label"] = sl["label"][graph["order"]].astype(np.int64)
        yield out
