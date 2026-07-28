"""Causal 50 ms event windows — minimal-memory (lazy memmap, no full-clip cache).

Format (B): Prophesee ``*_td.dat`` + ``*_bbox.npy`` (pixel top-left xywh).
Only the current window's rows are decoded; clips are never held in RAM.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from seg_model import GraphCfg, build_causal_graph

TEXT_EXTS = {".txt", ".csv"}

# Prophesee DAT v2 packed-field masks (unchanged)
_DAT_X_MASK = 2**14 - 1
_DAT_Y_MASK = 2**28 - 2**14
_DAT_P_MASK = 2**29 - 2**28
_DAT_EV_DTYPE = np.dtype([("t", "<u4"), ("_", "<i4")])

FORMAT_A = "A_per_event_label"
FORMAT_B = "B_yolo_bbox"
FORMAT_UNKNOWN = "unknown"


def _norm_path(p: str | Path) -> Path:
    """Accept both \\ and / path separators."""
    return Path(str(p).replace("\\", "/")).expanduser().resolve()


# ---------------------------------------------------------------------------
# RSS / resident-size helpers
# ---------------------------------------------------------------------------
def process_rss_gb() -> float:
    """Best-effort process RSS in GiB."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        pass
    try:
        import resource
        # Linux: ru_maxrss is KiB; macOS: bytes
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 ** 3)
        return rss / (1024 ** 2) / 1024.0
    except Exception:
        return float("nan")


def _approx_nbytes(obj, _seen: set[int] | None = None) -> int:
    """Rough deep nbytes for numpy arrays / containers (no full-clip expected)."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj) * 2
    if isinstance(obj, Path):
        return len(str(obj)) * 2
    if isinstance(obj, dict):
        return sum(_approx_nbytes(k, _seen) + _approx_nbytes(v, _seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(_approx_nbytes(x, _seen) for x in obj)
    if isinstance(obj, (int, float, bool, type(None))):
        return 32
    return 64


# ---------------------------------------------------------------------------
# Prophesee DAT header / memmap (format B)
# ---------------------------------------------------------------------------
def _parse_dat_header(f) -> tuple[int, int, int, dict]:
    """Return (body_offset, event_type, event_size, meta)."""
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


def peek_dat_meta(path: str | Path) -> dict:
    """Header-only: body_offset, n_events, sensor, mtime, size — no event decode."""
    path = _norm_path(path)
    st = path.stat()
    with path.open("rb") as f:
        offset, ev_type, ev_size, meta = _parse_dat_header(f)
        if ev_type not in (0, 12):
            raise ValueError(f"{path}: unsupported DAT event type {ev_type}")
        body = st.st_size - offset
        if body % _DAT_EV_DTYPE.itemsize != 0:
            raise ValueError(f"{path}: body size {body} not multiple of record")
        n_events = body // _DAT_EV_DTYPE.itemsize
    w = int(meta["width"] or 304)
    h = int(meta["height"] or 240)
    return {
        "path": path,
        "offset": int(offset),
        "n_events": int(n_events),
        "sensor": (w, h),
        "mtime": float(st.st_mtime),
        "size": int(st.st_size),
        "ev_type": int(ev_type),
        "ev_size": int(ev_size),
    }


def open_dat_memmap(path: str | Path, offset: int | None = None) -> np.memmap:
    """Memmap the binary event record array after the %-header."""
    path = _norm_path(path)
    if offset is None:
        meta = peek_dat_meta(path)
        offset = meta["offset"]
    return np.memmap(
        str(path), mode="r", dtype=_DAT_EV_DTYPE, offset=int(offset),
    )


def decode_dat_rows(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Decode a *slice* of DAT records → lean arrays (t int64 µs, x/y/p float32)."""
    if raw.size == 0:
        return {
            "t_us": np.zeros(0, dtype=np.int64),
            "x": np.zeros(0, dtype=np.float32),
            "y": np.zeros(0, dtype=np.float32),
            "p": np.zeros(0, dtype=np.float32),
        }
    # Materialise only this slice (copy out of memmap)
    t_us = np.asarray(raw["t"], dtype=np.int64).copy()
    packed = np.asarray(raw["_"], dtype=np.int32)
    x = np.bitwise_and(packed, _DAT_X_MASK).astype(np.float32)
    y = np.right_shift(np.bitwise_and(packed, _DAT_Y_MASK), 14).astype(np.float32)
    p = np.right_shift(np.bitwise_and(packed, _DAT_P_MASK), 28).astype(np.float32)
    return {"t_us": t_us, "x": x, "y": y, "p": p}


def events_for_graph(t_us: np.ndarray, x, y, p) -> dict[str, np.ndarray]:
    """Convert window µs → seconds relative to window start (model expects seconds)."""
    if len(t_us) == 0:
        return {
            "t": np.zeros(0, dtype=np.float32),
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "p": np.asarray(p, dtype=np.float32),
        }
    t0 = int(t_us[0])
    # float32 exact for integer µs offsets up to 2^24 (~16 s); windows are 50 ms
    t = (t_us.astype(np.int64) - t0).astype(np.float32) * np.float32(1e-6)
    return {
        "t": t,
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "p": np.asarray(p, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# BBoxes + per-window labels
# ---------------------------------------------------------------------------
def load_prophesee_bbox(path: str | Path) -> np.ndarray:
    """Load bbox.npy → float32 array [t0_s, t1_s, x, y, w, h] (pixel TL xywh)."""
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

    x = boxes["x"].astype(np.float32)
    y = boxes["y"].astype(np.float32)
    w = boxes["w"].astype(np.float32)
    h = boxes["h"].astype(np.float32)
    if float(np.nanmax(w)) <= 1.5 and float(np.nanmax(h)) <= 1.5 and float(np.nanmax(np.abs(x))) <= 1.5:
        raise ValueError(
            f"{path}: boxes look normalized (cxcywh in [0,1]); "
            "this loader expects Prophesee pixel top-left xywh"
        )

    t_s = boxes[ts_key].astype(np.float64) * 1e-6
    order = np.argsort(t_s, kind="stable")
    t_sorted = t_s[order]
    uniq_t = np.unique(t_sorted)
    if len(uniq_t) == 1:
        half = 1.0 / 60.0
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

    out = np.empty((len(boxes), 6), dtype=np.float32)
    for i, b in enumerate(boxes):
        ts = float(b[ts_key]) * 1e-6
        t0, t1 = intervals[ts]
        out[i] = (t0, t1, float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
    return out


def labels_from_boxes(
    t_s: np.ndarray, x: np.ndarray, y: np.ndarray, boxes: np.ndarray,
) -> np.ndarray:
    """Per-event FG=1 if (x,y) inside any box whose [t0,t1) contains t (seconds)."""
    labels = np.zeros(len(t_s), dtype=np.int64)
    if boxes is None or len(boxes) == 0 or len(t_s) == 0:
        return labels
    order = np.argsort(boxes[:, 0], kind="stable")
    boxes = boxes[order]
    i = 0
    n = len(boxes)
    while i < n:
        t0, t1 = float(boxes[i, 0]), float(boxes[i, 1])
        j = i + 1
        while j < n and float(boxes[j, 0]) == t0 and float(boxes[j, 1]) == t1:
            j += 1
        group = boxes[i:j]
        i0 = int(np.searchsorted(t_s, t0, side="left"))
        i1 = int(np.searchsorted(t_s, t1, side="left"))
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


def labels_from_boxes_us(
    t_us: np.ndarray, x: np.ndarray, y: np.ndarray, boxes_s: np.ndarray,
) -> np.ndarray:
    """Label using int64 µs event times + boxes stored in seconds."""
    if len(t_us) == 0:
        return np.zeros(0, dtype=np.int64)
    # Small window only — convert once for comparison with second-scale boxes
    t_s = t_us.astype(np.float64) * 1e-6
    return labels_from_boxes(t_s, x, y, boxes_s)


# ---------------------------------------------------------------------------
# Format detection / clip listing
# ---------------------------------------------------------------------------
def detect_data_format(directory: str | Path) -> str:
    root = _norm_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    td_files = sorted(root.glob("*_td.dat"))
    bbox_files = sorted(root.glob("*_bbox.npy"))
    if td_files and bbox_files:
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
        if sample.shape[1] >= 4:
            return FORMAT_A
    return FORMAT_UNKNOWN


def describe_unknown_format(directory: str | Path) -> str:
    root = _norm_path(directory)
    files = sorted(p.name for p in root.iterdir() if p.is_file())[:30]
    lines = [f"Unrecognized data format in {root}", f"Files (first {len(files)}):"]
    for name in files:
        lines.append(f"  - {name}")
    lines.append(
        "Expected (B) paired `*_td.dat` + `*_bbox.npy` (Prophesee / YOLO boxes)"
    )
    return "\n".join(lines)


def list_clips(directory: str | Path, fmt: str | None = None) -> list[Path]:
    root = _norm_path(directory)
    fmt = fmt or detect_data_format(root)
    if fmt == FORMAT_B:
        pairs = []
        for td in sorted(root.glob("*_td.dat")):
            stem = td.name.replace("_td.dat", "")
            if (root / f"{stem}_bbox.npy").is_file():
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
    fmt = detect_data_format(directory)
    if fmt == FORMAT_UNKNOWN:
        raise RuntimeError(describe_unknown_format(directory))
    return list_clips(directory, fmt)


def bbox_path_for(td_path: Path) -> Path:
    td_path = _norm_path(td_path)
    if td_path.name.endswith("_td.dat"):
        return td_path.with_name(td_path.name.replace("_td.dat", "_bbox.npy"))
    return td_path.with_name(td_path.stem + "_bbox.npy")


# ---------------------------------------------------------------------------
# Window index (memmap timestamps only) + sidecar cache
# ---------------------------------------------------------------------------
def window_row_ranges_memmap(
    t_us_mm: np.ndarray,
    window_ms: float = 50.0,
    stride_ms: float | None = None,
    min_events: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build window row ranges via searchsorted on memmapped timestamps (µs).

    Returns row_start, row_end, t0_us, t1_us (int64 arrays). Windows match the
    previous seconds-based indexer: fixed length from the first event time.
    """
    if t_us_mm.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty

    t0_abs = int(t_us_mm[0])
    t_end = int(t_us_mm[-1])
    w_us = int(round(window_ms * 1e3))
    s_us = w_us if stride_ms is None else int(round(stride_ms * 1e3))
    if s_us <= 0:
        raise ValueError("stride_ms must be > 0")

    starts, ends, t0s, t1s = [], [], [], []
    start = t0_abs
    while start <= t_end:
        stop = start + w_us
        i0 = int(np.searchsorted(t_us_mm, start, side="left"))
        i1 = int(np.searchsorted(t_us_mm, stop, side="left"))
        if (i1 - i0) >= min_events:
            starts.append(i0)
            ends.append(i1)
            t0s.append(start)
            t1s.append(stop)
        start += s_us

    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(t0s, dtype=np.int64),
        np.asarray(t1s, dtype=np.int64),
    )


def _sidecar_path(index_cache_dir: Path, td_path: Path, window_ms: float,
                  stride_ms: float | None, min_events: int) -> Path:
    key = f"{td_path.resolve()}|{window_ms}|{stride_ms}|{min_events}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    stem = td_path.name.replace("_td.dat", "")
    return index_cache_dir / f"{stem}_{h}.npz"


def load_or_build_window_index(
    td_path: Path,
    *,
    window_ms: float,
    stride_ms: float | None,
    min_events: int,
    index_cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (row_start, row_end, t0_us, t1_us, meta). Uses .npz sidecar when valid."""
    td_path = _norm_path(td_path)
    meta = peek_dat_meta(td_path)

    if index_cache_dir is not None:
        index_cache_dir = _norm_path(index_cache_dir)
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        side = _sidecar_path(index_cache_dir, td_path, window_ms, stride_ms, min_events)
        if side.is_file():
            try:
                z = np.load(str(side))
                if (float(z["dat_mtime"]) == meta["mtime"]
                        and int(z["dat_size"]) == meta["size"]
                        and float(z["window_ms"]) == float(window_ms)
                        and int(z["min_events"]) == int(min_events)):
                    print(f"[index] cache hit {side.name}")
                    return (
                        z["row_start"].astype(np.int64),
                        z["row_end"].astype(np.int64),
                        z["t0_us"].astype(np.int64),
                        z["t1_us"].astype(np.int64),
                        meta,
                    )
                print(f"[index] cache stale {side.name} — rebuilding")
            except Exception as e:
                print(f"[index] cache unreadable ({e}) — rebuilding")

    print(f"[index] scanning timestamps {td_path.name} "
          f"(n~{meta['n_events']:,}, timestamps only) ...")
    mm = open_dat_memmap(td_path, offset=meta["offset"])
    try:
        # Materialise *only* the uint32 timestamp column (~4 bytes/event),
        # never x/y/p. Freed immediately after building the row index.
        t_us = np.array(mm["t"], dtype=np.uint32)
    finally:
        del mm
    rs, re, t0, t1 = window_row_ranges_memmap(
        t_us, window_ms=window_ms, stride_ms=stride_ms, min_events=min_events,
    )
    del t_us

    if index_cache_dir is not None:
        side = _sidecar_path(index_cache_dir, td_path, window_ms, stride_ms, min_events)
        np.savez(
            str(side),
            row_start=rs, row_end=re, t0_us=t0, t1_us=t1,
            dat_mtime=np.float64(meta["mtime"]),
            dat_size=np.int64(meta["size"]),
            window_ms=np.float64(window_ms),
            min_events=np.int64(min_events),
            n_events=np.int64(meta["n_events"]),
        )
        print(f"[index] wrote {side.name}  ({len(rs)} windows)")

    return rs, re, t0, t1, meta


# ---------------------------------------------------------------------------
# Subsample + fg/bg helpers
# ---------------------------------------------------------------------------
def subsample_window(
    t_us, x, y, p, label,
    max_events: int,
    *,
    seed: int,
    deterministic: bool,
) -> tuple:
    """Uniform subsample if over cap; keep time order. Returns arrays + (n_pre, n_post)."""
    n = int(len(t_us))
    if max_events is None or max_events <= 0 or n <= max_events:
        return t_us, x, y, p, label, n, n
    rng = np.random.default_rng(seed if deterministic else seed)
    idx = rng.choice(n, size=int(max_events), replace=False)
    idx.sort()  # preserve causal / time order
    lab = None if label is None else label[idx]
    return t_us[idx], x[idx], y[idx], p[idx], lab, n, int(max_events)


def fg_bg_ratio(labels: np.ndarray | None) -> tuple[int, int, float]:
    if labels is None or len(labels) == 0:
        return 0, 0, float("nan")
    n_fg = int((labels == 1).sum())
    n_bg = int((labels == 0).sum())
    return n_fg, n_bg, n_fg / max(n_bg, 1)


def summarize_dir(
    directory: str | Path,
    window_ms: float = 50.0,
    *,
    index_cache_dir: str | Path | None = None,
    max_events_per_window: int = 20000,
) -> str:
    """Detect format; print window counts / sampled fg/bg without loading full clips."""
    root = _norm_path(directory)
    fmt = detect_data_format(root)
    if fmt == FORMAT_UNKNOWN:
        print(describe_unknown_format(root))
        raise RuntimeError(f"unsupported data format in {root}")
    if fmt != FORMAT_B:
        raise RuntimeError(
            f"[data] format {fmt} is not supported by the lazy memmap loader; "
            "expected (B) *_td.dat + *_bbox.npy"
        )
    print(
        f"[data] detected format (B) YOLO/Prophesee bounding boxes  "
        f"(*_td.dat + *_bbox.npy, pixel top-left xywh)  in {root}"
    )
    print(f"[data] RSS at summarize start: {process_rss_gb():.2f} GiB")

    files = list_clips(root, fmt)
    print(f"[data] {root}: {len(files)} clip(s)  (split is by clip/file)")
    if len(files) == 1:
        print("[data] WARNING: single clip only — treat as dev/sanity, "
              "not a generalization result")

    cache = _norm_path(index_cache_dir) if index_cache_dir else None
    for path in files:
        rs, re, t0, t1, meta = load_or_build_window_index(
            path, window_ms=window_ms, stride_ms=None, min_events=8,
            index_cache_dir=cache,
        )
        boxes = load_prophesee_bbox(bbox_path_for(path))
        # Sample up to 20 windows for fg/bg estimate (decode those rows only)
        n_w = len(rs)
        step = max(1, n_w // 20)
        sample_idx = list(range(0, n_w, step))[:20]
        chunks = []
        mm = open_dat_memmap(path, offset=meta["offset"])
        try:
            for k in sample_idx:
                raw = mm[int(rs[k]): int(re[k])]
                dec = decode_dat_rows(raw)
                chunks.append(labels_from_boxes_us(
                    dec["t_us"], dec["x"], dec["y"], boxes))
        finally:
            del mm
        labels = np.concatenate(chunks) if chunks else None
        n_fg, n_bg, ratio = fg_bg_ratio(labels)
        sw, sh = meta["sensor"]
        print(
            f"  {path.name}: {meta['n_events']:,} events (memmap), "
            f"{n_w} windows ({window_ms:.0f} ms), labeled, "
            f"fg={n_fg} bg={n_bg} fg/bg={ratio:.4f} sensor={sw}x{sh}"
        )
    print(f"[data] RSS after summarize: {process_rss_gb():.2f} GiB")
    return fmt


# ---------------------------------------------------------------------------
# Lazy dataset
# ---------------------------------------------------------------------------
class EventWindowDataset(Dataset):
    """One item = one causal window. Holds only the window index + tiny bboxes."""

    def __init__(
        self,
        directory: str | Path,
        cfg: GraphCfg,
        window_ms: float = 50.0,
        stride_ms: float | None = None,
        require_labels: bool = True,
        min_events: int = 8,
        clip_files: list[Path] | None = None,
        index_cache_dir: str | Path | None = None,
        max_events_per_window: int = 20000,
        subsample_seed: int = 0,
        deterministic_subsample: bool = False,
    ):
        self.cfg = cfg
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.require_labels = require_labels
        self.min_events = min_events
        self.max_events_per_window = int(max_events_per_window)
        self.subsample_seed = int(subsample_seed)
        self.deterministic_subsample = bool(deterministic_subsample)

        root = _norm_path(directory)
        self.fmt = detect_data_format(root)
        if self.fmt != FORMAT_B:
            raise RuntimeError(
                f"lazy memmap dataset requires format (B); got {self.fmt}"
            )
        self.files = [
            _norm_path(p) for p in (
                clip_files if clip_files is not None else list_clips(root, self.fmt)
            )
        ]
        cache = _norm_path(index_cache_dir) if index_cache_dir else None

        # Per-clip: meta + tiny boxes only (never event arrays)
        self._meta: list[dict] = []
        self._boxes: list[np.ndarray | None] = []
        self._t0_abs: list[int] = []  # first-event µs for t_start_ms
        # Flat index: (clip_i, row_start, row_end, t0_us, t1_us)
        self.index: list[tuple[int, int, int, int, int]] = []

        for ci, path in enumerate(self.files):
            rs, re, t0, t1, meta = load_or_build_window_index(
                path,
                window_ms=window_ms,
                stride_ms=stride_ms,
                min_events=min_events,
                index_cache_dir=cache,
            )
            bp = bbox_path_for(path)
            if bp.is_file():
                boxes = load_prophesee_bbox(bp)
            else:
                boxes = None
                if require_labels:
                    raise ValueError(f"{path}: missing paired bbox {bp.name}")

            sw, sh = meta["sensor"]
            if (sw, sh) != (cfg.width, cfg.height):
                print(
                    f"[data] WARNING: clip sensor {sw}x{sh} != "
                    f"cfg {cfg.width}x{cfg.height} ({path.name})"
                )

            # First-event timestamp (one scalar from memmap)
            mm = open_dat_memmap(path, offset=meta["offset"])
            try:
                t0_abs = int(mm["t"][0]) if meta["n_events"] else 0
            finally:
                del mm

            self._meta.append(meta)
            self._boxes.append(boxes)
            self._t0_abs.append(t0_abs)
            for a, b, u0, u1 in zip(rs, re, t0, t1):
                self.index.append((ci, int(a), int(b), int(u0), int(u1)))

        if not self.index:
            raise RuntimeError(
                f"no windows with >= {min_events} events in {directory}"
            )

        self._assert_lean()
        # One sample window: print pre/post cap counts
        self._print_sample_cap()

    def _assert_lean(self) -> None:
        """Refuse to retain anything resembling a full-clip event array."""
        n_win = max(len(self.index), 1)
        limit = 10 * n_win
        # Walk only retained state
        retained = {
            "index": self.index,
            "files": self.files,
            "meta": self._meta,
            "boxes": self._boxes,
            "t0_abs": self._t0_abs,
        }
        for name, obj in retained.items():
            if isinstance(obj, np.ndarray) and obj.size > limit and name == "boxes":
                # boxes are tiny; skip
                continue
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, np.ndarray) and item.size > 1_000_000:
                        raise AssertionError(
                            f"dataset retained large array {name}[{i}] "
                            f"size={item.size} (full-clip leak?)"
                        )
        nbytes = _approx_nbytes(retained)
        print(
            f"[data] EventWindowDataset lean check OK  "
            f"windows={len(self.index)}  approx_resident={nbytes / 1e6:.2f} MB  "
            f"RSS={process_rss_gb():.2f} GiB"
        )
        # Hard assert: no array with > 10 * n_windows elements except tiny boxes
        max_elems = 0
        for boxes in self._boxes:
            if boxes is not None:
                max_elems = max(max_elems, int(boxes.size))
        # index tuples don't count as big arrays; boxes << limit always
        assert max_elems < max(limit, 100_000), (
            f"unexpected large retained array elems={max_elems}"
        )

    def _print_sample_cap(self) -> None:
        if not self.index:
            return
        item = self._load_window_arrays(0, apply_cap=False)
        n_pre = int(len(item["t_us"]))
        _, _, _, _, _, _, n_post = subsample_window(
            item["t_us"], item["x"], item["y"], item["p"], item.get("label"),
            self.max_events_per_window,
            seed=self.subsample_seed,
            deterministic=True,
        )
        print(
            f"[data] sample window[0] events pre-cap={n_pre}  "
            f"post-cap={min(n_pre, n_post if self.max_events_per_window > 0 else n_pre)}  "
            f"(max_events_per_window={self.max_events_per_window})"
        )

    def _load_window_arrays(self, idx: int, *, apply_cap: bool) -> dict:
        ci, i0, i1, t0_us, t1_us = self.index[idx]
        path = self.files[ci]
        meta = self._meta[ci]
        mm = open_dat_memmap(path, offset=meta["offset"])
        try:
            raw = np.array(mm[i0:i1])  # copy slice out of memmap
        finally:
            del mm
        dec = decode_dat_rows(raw)
        boxes = self._boxes[ci]
        label = None
        if boxes is not None:
            label = labels_from_boxes_us(dec["t_us"], dec["x"], dec["y"], boxes)
        elif self.require_labels:
            raise ValueError(f"{path}: labels required but no boxes")

        n_pre = len(dec["t_us"])
        if apply_cap:
            seed = self.subsample_seed + (0 if self.deterministic_subsample else idx)
            t_us, x, y, p, label, n_pre, n_post = subsample_window(
                dec["t_us"], dec["x"], dec["y"], dec["p"], label,
                self.max_events_per_window,
                seed=seed,
                deterministic=self.deterministic_subsample,
            )
        else:
            t_us, x, y, p = dec["t_us"], dec["x"], dec["y"], dec["p"]
            n_post = n_pre
        return {
            "t_us": t_us, "x": x, "y": y, "p": p, "label": label,
            "ci": ci, "i0": i0, "i1": i1,
            "t0_us": t0_us, "t1_us": t1_us,
            "n_pre": n_pre, "n_post": n_post,
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        w = self._load_window_arrays(idx, apply_cap=True)
        sl = events_for_graph(w["t_us"], w["x"], w["y"], w["p"])
        if w["label"] is not None:
            sl["label"] = w["label"]
        graph = build_causal_graph(sl, self.cfg)
        ci = w["ci"]
        t0_abs = self._t0_abs[ci]
        item = {
            "graph": graph,
            "clip": self.files[ci].stem.replace("_td", ""),
            "window_index": idx,
            "file_window": (ci, w["i0"], w["i1"]),
            "t_start_ms": (w["t0_us"] - t0_abs) * 1e-3,
            "t0": w["t0_us"] * 1e-6,
            "t1": w["t1_us"] * 1e-6,
            "n_events": w["n_post"],
            "n_events_pre_cap": w["n_pre"],
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
    *,
    index_cache_dir: str | Path | None = None,
    max_events_per_window: int = 20000,
    subsample_seed: int = 0,
) -> Iterator[dict]:
    """Yield windows in time order; decode only each window's memmap rows."""
    path = _norm_path(path)
    rs, re, t0s, t1s, meta = load_or_build_window_index(
        path,
        window_ms=window_ms,
        stride_ms=stride_ms,
        min_events=min_events,
        index_cache_dir=_norm_path(index_cache_dir) if index_cache_dir else None,
    )
    bp = bbox_path_for(path)
    boxes = load_prophesee_bbox(bp) if bp.is_file() else None
    mm = open_dat_memmap(path, offset=meta["offset"])
    try:
        t0_abs = int(mm["t"][0]) if meta["n_events"] else 0
        for wi, (i0, i1, t0_us, t1_us) in enumerate(zip(rs, re, t0s, t1s)):
            raw = np.array(mm[int(i0): int(i1)])
            dec = decode_dat_rows(raw)
            label = None
            if boxes is not None:
                label = labels_from_boxes_us(dec["t_us"], dec["x"], dec["y"], boxes)
            t_us, x, y, p, label, n_pre, n_post = subsample_window(
                dec["t_us"], dec["x"], dec["y"], dec["p"], label,
                max_events_per_window,
                seed=subsample_seed + wi,
                deterministic=True,
            )
            sl = events_for_graph(t_us, x, y, p)
            if label is not None:
                sl["label"] = label
            graph = build_causal_graph(sl, cfg)
            out = {
                "graph": graph,
                "clip": path.stem.replace("_td", ""),
                "window_index": wi,
                "t_start_ms": (int(t0_us) - t0_abs) * 1e-3,
                "t0": int(t0_us) * 1e-6,
                "t1": int(t1_us) * 1e-6,
                "has_labels": label is not None,
                "n_events_pre_cap": n_pre,
                "n_events": n_post,
            }
            if label is not None:
                out["label"] = label[graph["order"]].astype(np.int64)
            yield out
    finally:
        del mm


# Back-compat: sensor peek without full load (used by run.py)
def load_event_file(path: str | Path, *, require_labels: bool = False,
                    derive_labels: bool = False) -> dict:
    """Header/meta only for format B — does **not** load event arrays."""
    path = _norm_path(path)
    if not (path.name.endswith("_td.dat") or path.suffix.lower() == ".dat"):
        raise ValueError(f"{path}: lazy loader expects Prophesee *_td.dat")
    meta = peek_dat_meta(path)
    out = {
        "path": path,
        "clip": path.stem.replace("_td", ""),
        "format": FORMAT_B,
        "sensor": meta["sensor"],
        "n_events": meta["n_events"],
    }
    bp = bbox_path_for(path)
    if bp.is_file():
        out["boxes"] = load_prophesee_bbox(bp)
    elif require_labels:
        raise ValueError(f"{path}: missing paired bbox {bp.name}")
    return out
