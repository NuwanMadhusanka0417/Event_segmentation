"""
Event-camera 3D graph visualizer.

Loads .mat event files (SNKTH-style and similar), exposes them to the browser
as a JSON event stream that a Three.js front-end animates into a directed
spatiotemporal graph (x, y, t).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

try:
    from scipy.io import loadmat as _scipy_loadmat
except Exception:  # pragma: no cover
    _scipy_loadmat = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


BASE_DIR = Path(__file__).resolve().parent
EVENTS_DIR = BASE_DIR / "events"
EVENTS_DIR.mkdir(exist_ok=True)

# Cap how many events we ship to the browser. A typical sample is <1M events;
# more than this brings WebGL/JS to its knees regardless of how fast we send it.
MAX_EVENTS = 200_000

# Field-name candidates we look for when introspecting a .mat structure.
T_KEYS = ("t", "ts", "time", "times", "timestamp", "timestamps")
X_KEYS = ("x", "xs", "col", "cols", "u")
Y_KEYS = ("y", "ys", "row", "rows", "v")
P_KEYS = ("p", "pol", "polarity", "polarities")
BUNDLE_KEYS = ("events", "ev", "evs", "TD", "td", "data", "Events")


app = Flask(__name__, static_folder="static", template_folder="templates")


# ---------------------------------------------------------------------------
# .mat loading
# ---------------------------------------------------------------------------

def _coerce_1d(a) -> np.ndarray | None:
    if a is None:
        return None
    arr = np.asarray(a).squeeze()
    if arr.ndim == 0:
        return None
    return arr.astype(np.float64, copy=False).ravel()


def _from_field_map(fields: dict) -> tuple[np.ndarray, ...] | None:
    """Given a dict-like with possible t/x/y/p keys, extract (t,x,y,p)."""
    lower = {k.lower(): k for k in fields.keys() if isinstance(k, str)}

    def pick(cands: Iterable[str]) -> np.ndarray | None:
        for c in cands:
            if c in lower:
                return _coerce_1d(fields[lower[c]])
        return None

    t = pick(T_KEYS)
    x = pick(X_KEYS)
    y = pick(Y_KEYS)
    p = pick(P_KEYS)
    if t is None or x is None or y is None:
        return None
    if p is None:
        p = np.ones_like(t)
    n = min(len(t), len(x), len(y), len(p))
    return t[:n], x[:n], y[:n], p[:n]


def _from_2d_array(arr: np.ndarray) -> tuple[np.ndarray, ...] | None:
    """Try to interpret an (N,4) or (4,N) array as event columns."""
    a = np.asarray(arr)
    if a.ndim != 2:
        return None
    if a.shape[1] == 4:
        cols = [a[:, i].astype(np.float64) for i in range(4)]
    elif a.shape[0] == 4:
        cols = [a[i, :].astype(np.float64) for i in range(4)]
    else:
        return None
    # Heuristic: the column with the largest range and monotonic-ish trend
    # is the timestamp. Fall back to assuming order (t, x, y, p) or (x, y, t, p).
    ranges = [c.max() - c.min() for c in cols]
    t_idx = int(np.argmax(ranges))
    remaining = [i for i in range(4) if i != t_idx]
    # Polarity is the smallest-range column among the remaining three.
    p_idx = remaining[int(np.argmin([cols[i].max() - cols[i].min() for i in remaining]))]
    xy_idx = [i for i in remaining if i != p_idx]
    # Whichever of those varies more is x by convention (wider sensor).
    if cols[xy_idx[0]].max() >= cols[xy_idx[1]].max():
        x_idx, y_idx = xy_idx[0], xy_idx[1]
    else:
        x_idx, y_idx = xy_idx[1], xy_idx[0]
    return cols[t_idx], cols[x_idx], cols[y_idx], cols[p_idx]


def _walk_scipy(obj, depth: int = 0):
    """Yield candidate field-maps and 2-D arrays from a scipy-loaded structure."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        # The dict itself may already hold t/x/y/p at the top level.
        yield ("map", obj)
        for k, v in obj.items():
            if k.startswith("__"):
                continue
            yield from _walk_scipy(v, depth + 1)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names:  # structured array (MATLAB struct)
            fmap = {n: obj[n] for n in obj.dtype.names}
            yield ("map", fmap)
            for n in obj.dtype.names:
                yield from _walk_scipy(obj[n], depth + 1)
        elif obj.dtype == object:
            for item in obj.ravel():
                yield from _walk_scipy(item, depth + 1)
        else:
            if obj.ndim == 2 and (obj.shape[0] == 4 or obj.shape[1] == 4):
                yield ("arr", obj)


def _try_scipy(path: Path) -> tuple[np.ndarray, ...] | None:
    if _scipy_loadmat is None:
        return None
    try:
        mat = _scipy_loadmat(str(path), squeeze_me=False, struct_as_record=True)
    except NotImplementedError:
        return None  # v7.3 -> h5py
    except Exception:
        return None
    # Prefer named bundles first.
    for key in BUNDLE_KEYS:
        if key in mat:
            for kind, blob in _walk_scipy(mat[key]):
                got = _from_field_map(blob) if kind == "map" else _from_2d_array(blob)
                if got:
                    return got
    # Top-level t/x/y/p variables (a common SNKTH layout).
    got = _from_field_map(mat)
    if got:
        return got
    # Any structured / 2-D leaf.
    for kind, blob in _walk_scipy(mat):
        got = _from_field_map(blob) if kind == "map" else _from_2d_array(blob)
        if got:
            return got
    return None


def _walk_h5(group):
    """Yield (path, dataset) entries from an h5py group recursively."""
    for name, item in group.items():
        if isinstance(item, h5py.Dataset):
            yield name, item
        elif isinstance(item, h5py.Group):
            for sub_name, sub in _walk_h5(item):
                yield f"{name}/{sub_name}", sub


def _try_h5(path: Path) -> tuple[np.ndarray, ...] | None:
    if h5py is None:
        return None
    try:
        f = h5py.File(str(path), "r")
    except Exception:
        return None
    with f:
        datasets = dict(_walk_h5(f))
        if not datasets:
            return None
        # Field-map by basename of the dataset path.
        fmap = {k.split("/")[-1]: np.array(v) for k, v in datasets.items()}
        got = _from_field_map(fmap)
        if got:
            return got
        # Try any (N,4) / (4,N) dataset.
        for v in datasets.values():
            arr = np.array(v)
            if arr.ndim == 2 and (arr.shape[0] == 4 or arr.shape[1] == 4):
                got = _from_2d_array(arr)
                if got:
                    return got
    return None


def load_events(path: Path) -> dict:
    """Return a normalized event payload for the browser."""
    parsed = _try_scipy(path) or _try_h5(path)
    if parsed is None:
        raise ValueError(
            f"Could not interpret '{path.name}' as an event stream. "
            "Expected fields like t/x/y/p or a 4-column array."
        )
    t, x, y, p = parsed

    # Sort by time and drop NaNs.
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]
    if t.size == 0:
        raise ValueError("Event stream is empty after parsing.")

    # Downsample if absurdly large (uniform sampling preserves time density).
    n = t.size
    if n > MAX_EVENTS:
        idx = np.linspace(0, n - 1, MAX_EVENTS).astype(np.int64)
        t, x, y, p = t[idx], x[idx], y[idx], p[idx]
        downsampled_from = int(n)
    else:
        downsampled_from = None

    # Normalize polarity to {-1, +1}; original may be {0,1} or {-1,+1}.
    p_norm = np.where(p > 0.5, 1, -1).astype(np.int8)

    t0 = float(t.min())
    payload = {
        "count": int(t.size),
        "downsampled_from": downsampled_from,
        "t_min": t0,
        "t_max": float(t.max()),
        "x_min": float(x.min()), "x_max": float(x.max()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        # Send ints where possible; t kept as float (units unknown — could be us/ms/s).
        "t": (t - t0).astype(np.float32).tolist(),
        "x": x.astype(np.float32).tolist(),
        "y": y.astype(np.float32).tolist(),
        "p": p_norm.tolist(),
    }
    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/files")
def list_files():
    files = sorted(
        f.name for f in EVENTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".mat", ".h5", ".hdf5"}
    )
    return jsonify({"files": files, "dir": str(EVENTS_DIR)})


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file"}), 400
    name = Path(f.filename).name  # strip any path
    if Path(name).suffix.lower() not in {".mat", ".h5", ".hdf5"}:
        return jsonify({"error": "only .mat / .h5 supported"}), 400
    dest = EVENTS_DIR / name
    f.save(dest)
    return jsonify({"ok": True, "name": name})


@app.get("/api/events")
def get_events():
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "name is required"}), 400
    path = EVENTS_DIR / Path(name).name
    if not path.exists():
        return jsonify({"error": f"file not found: {name}"}), 404
    try:
        payload = load_events(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(payload)


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
