"""
Prophesee Gen1 Automotive Detection I/O.

Reads paired files:
  * *_td.dat   — binary Event2D stream (t_us, x, y, p)
  * *_bbox.npy — structured numpy boxes (ts/t, x, y, w, h, class_id)

Based on prophesee-automotive-dataset-toolbox (dat_events_tools.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Gen1 Automotive Detection defaults (overridden from .dat header when present)
PROPHESEE_SENSOR_W = 304
PROPHESEE_SENSOR_H = 240
PROPHESEE_CLASS_NAMES = ["car", "pedestrian"]

_EV_DTYPE = [("t", "u4"), ("_", "i4")]
_EV_ITEMSIZE = np.dtype(_EV_DTYPE).itemsize
_X_MASK = 2**14 - 1
_Y_MASK = 2**28 - 2**14
_P_MASK = 2**29 - 2**28


def _raw_to_events(raw, t0):
    """Decode a raw Event2D chunk -> float32 [N,4] with relative timestamps."""
    if raw.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    t_abs = raw["t"].astype(np.float64)
    x = np.bitwise_and(raw["_"], _X_MASK).astype(np.float64)
    y = np.right_shift(np.bitwise_and(raw["_"], _Y_MASK), 14).astype(np.float64)
    p = np.right_shift(np.bitwise_and(raw["_"], _P_MASK), 28).astype(np.float64)
    return np.stack([t_abs - t0, x, y, p.clip(0, 1)], axis=1).astype(np.float32)


def parse_dat_header(f):
    """Return (body_offset, ev_type, ev_size, (height, width))."""
    bod = None
    num_comment_line = 0
    size = [None, None]
    while True:
        bod = f.tell()
        line = f.readline()
        if not line.startswith(b"% "):
            break
        words = line.split()
        if len(words) > 2:
            key = words[1].decode("latin-1") if isinstance(words[1], bytes) else words[1]
            if key == "Height":
                size[0] = int(words[2])
            elif key == "Width":
                size[1] = int(words[2])
        num_comment_line += 1
    f.seek(bod)
    if num_comment_line > 0:
        ev_type = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        ev_type = 0
        ev_size = sum(int(n[-1]) for _, n in _EV_DTYPE)
    bod = f.tell()
    return bod, ev_type, ev_size, tuple(size)


def probe_dat(dat_path):
    """Read only metadata from a .dat file (mmap; no full load into RAM).

    Returns dict with keys: t0, t_last, sensor (w,h), n_events, bod, stem.
    """
    with open(dat_path, "rb") as f:
        bod, ev_type, ev_size, size = parse_dat_header(f)
        if ev_type != 0:
            raise ValueError(f"Unsupported DAT event type {ev_type} in {dat_path}")
    w = int(size[1] or PROPHESEE_SENSOR_W)
    h = int(size[0] or PROPHESEE_SENSOR_H)
    stem = Path(dat_path).stem.replace("_td", "")
    fsize = os.path.getsize(dat_path)
    n_events = max(0, (fsize - bod) // _EV_ITEMSIZE)
    if n_events == 0:
        return dict(t0=0.0, t_last=0.0, sensor=(w, h), n_events=0, bod=bod, stem=stem)
    mm = np.memmap(dat_path, dtype=np.dtype(_EV_DTYPE), mode="r", offset=bod,
                   shape=(n_events,))
    t0 = float(mm["t"][0])
    t_last = float(mm["t"][-1])
    del mm
    return dict(t0=t0, t_last=t_last, sensor=(w, h), n_events=n_events, bod=bod, stem=stem)


class DatWindowReader:
    """Memory-mapped .dat reader; loads one time window at a time."""

    def __init__(self, dat_path):
        self.dat_path = dat_path
        self.meta = probe_dat(dat_path)
        self._mm = np.memmap(
            dat_path, dtype=np.dtype(_EV_DTYPE), mode="r",
            offset=self.meta["bod"], shape=(self.meta["n_events"],),
        )

    @property
    def t0(self):
        return self.meta["t0"]

    @property
    def sensor(self):
        return self.meta["sensor"]

    @property
    def stem(self):
        return self.meta["stem"]

    def load_window(self, t_start_rel, t_end_rel):
        """Load events with relative timestamps in [t_start_rel, t_end_rel) us."""
        if self.meta["n_events"] == 0:
            return np.zeros((0, 4), dtype=np.float32)
        t_lo = self.t0 + float(t_start_rel)
        t_hi = self.t0 + float(t_end_rel)
        i0 = int(np.searchsorted(self._mm["t"], t_lo, side="left"))
        i1 = int(np.searchsorted(self._mm["t"], t_hi, side="left"))
        if i1 <= i0:
            return np.zeros((0, 4), dtype=np.float32)
        return _raw_to_events(np.asarray(self._mm[i0:i1]), self.t0)

    def close(self):
        mm = getattr(self, "_mm", None)
        if mm is not None:
            del self._mm
            self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_prophesee_window(dat_path, t_start_rel, t_end_rel, reader=None):
    """Load one event window without reading the full recording."""
    if reader is not None:
        return reader.load_window(t_start_rel, t_end_rel)
    r = DatWindowReader(dat_path)
    try:
        return r.load_window(t_start_rel, t_end_rel)
    finally:
        r.close()


def window_ranges(t0, t_last, window_ms):
    """Yield (win_i, t_start_rel, t_end_rel) for relative-us windows."""
    if t_last <= t0:
        yield 0, 0.0, window_ms * 1000.0
        return
    win_us = window_ms * 1000.0
    span = t_last - t0
    n = max(1, int(np.ceil(span / win_us)))
    for wi in range(n):
        ts = wi * win_us
        te = min((wi + 1) * win_us, span + 1.0)
        yield wi, ts, te


def load_prophesee_recording(dat_path, bbox_path):
    """Load paired *_td.dat + *_bbox.npy with aligned relative timestamps.

    Returns
        events : (N, 4) float32  [t_us_rel, x, y, p]
        boxes  : structured numpy with field `t` in the same relative-us frame
        sensor : (width, height)
        stem   : recording basename without extension
    """
    with open(dat_path, "rb") as f:
        _, ev_type, _, size = parse_dat_header(f)
        if ev_type != 0:
            raise ValueError(f"Unsupported DAT event type {ev_type} in {dat_path}")
        raw = np.fromfile(f, dtype=np.dtype(_EV_DTYPE))

    w = int(size[1] or PROPHESEE_SENSOR_W)
    h = int(size[0] or PROPHESEE_SENSOR_H)
    stem = Path(dat_path).stem.replace("_td", "")

    if raw.size == 0:
        boxes = load_prophesee_bbox(bbox_path, t0_us=0.0)
        return np.zeros((0, 4), dtype=np.float32), boxes, (w, h), stem

    t0 = float(raw["t"][0])
    events = _raw_to_events(raw, t0)
    boxes = load_prophesee_bbox(bbox_path, t0_us=t0)
    return events, boxes, (w, h), stem


def load_prophesee_dat(path):
    """Load a Prophesee *_td.dat file (events only).

    Returns
        events : (N, 4) float32  [t_us_rel, x, y, p]
        sensor : (width, height)
    """
    with open(path, "rb") as f:
        _, ev_type, _, size = parse_dat_header(f)
        if ev_type != 0:
            raise ValueError(f"Unsupported DAT event type {ev_type} in {path}")
        raw = np.fromfile(f, dtype=np.dtype(_EV_DTYPE))
    w = int(size[1] or PROPHESEE_SENSOR_W)
    h = int(size[0] or PROPHESEE_SENSOR_H)
    if raw.size == 0:
        return np.zeros((0, 4), dtype=np.float32), (w, h)
    t0 = float(raw["t"][0])
    return _raw_to_events(raw, t0), (w, h)


def load_prophesee_bbox(path, t0_us=None):
    """Load *_bbox.npy annotations."""
    boxes = np.load(path, allow_pickle=False)
    names = boxes.dtype.names or ()
    if "ts" in names and "t" not in names:
        boxes = boxes.copy()
        new = np.empty(boxes.shape, dtype=[(n if n != "ts" else "t", boxes.dtype[n])
                                           for n in names])
        for n in names:
            new[n if n != "ts" else "t"] = boxes[n]
        boxes = new
    if t0_us is not None and boxes.size:
        boxes = boxes.copy()
        boxes["t"] = boxes["t"].astype(np.float64) - float(t0_us)
    return boxes


def find_prophesee_pairs(data_dir):
    """Find all (dat_path, bbox_path) pairs under `data_dir`."""
    root = Path(data_dir)
    pairs = []
    for dat in sorted(root.rglob("*_td.dat")):
        bbox = dat.with_name(dat.name.replace("_td.dat", "_bbox.npy"))
        if bbox.exists():
            pairs.append((str(dat), str(bbox)))
    return pairs


def boxes_in_window(boxes, w_start_us, w_end_us):
    """Filter bbox array to a time window; return (M,5) [class, xc, yc, w, h]."""
    if boxes is None or boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    names = boxes.dtype.names or ()
    t_arr = boxes["t"] if "t" in names else boxes["ts"]
    mask = (t_arr >= w_start_us) & (t_arr < w_end_us)
    b = boxes[mask]
    if b.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xc = b["x"].astype(np.float32) + b["w"].astype(np.float32) / 2
    yc = b["y"].astype(np.float32) + b["h"].astype(np.float32) / 2
    cls = b["class_id"].astype(np.float32)
    return np.stack([cls, xc, yc, b["w"].astype(np.float32),
                     b["h"].astype(np.float32)], axis=1)


def render_event_frame(events, width, height, bg=32):
    """Render events in a window to an RGB frame (ON=green, OFF=blue)."""
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    if events.shape[0] == 0:
        return img
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    p = events[:, 3].astype(np.int32)
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[ok], y[ok], p[ok]
    on = p >= 1
    off = ~on
    img[y[on], x[on], 1] = 255
    img[y[off], x[off], 2] = 255
    return img


def save_detection_frame(path, events, width, height, pred_boxes, pred_labels,
                         pred_scores, gt_boxes=None, gt_labels=None,
                         class_names=None):
    """Save one annotated event frame as PNG (matplotlib, no OpenCV required)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if class_names is None:
        class_names = PROPHESEE_CLASS_NAMES

    frame = render_event_frame(events, width, height)
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=80)
    ax.imshow(frame, origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    def _to_np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    if gt_boxes is not None and len(gt_boxes):
        for i, box in enumerate(_to_np(gt_boxes)):
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="cyan", lw=1.2))
            li = int(_to_np(gt_labels)[i]) if gt_labels is not None else 0
            name = class_names[li] if li < len(class_names) else str(li)
            ax.text(x1, max(y1 - 2, 8), f"GT:{name}", color="cyan", fontsize=6)

    if pred_boxes is not None and len(pred_boxes):
        pb = _to_np(pred_boxes)
        pl = _to_np(pred_labels)
        ps = _to_np(pred_scores)
        for i, box in enumerate(pb):
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="lime", lw=1.2))
            li = int(pl[i])
            name = class_names[li] if li < len(class_names) else str(li)
            ax.text(x1, y1 + 10, f"{name} {ps[i]:.2f}", color="lime", fontsize=6)

    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
