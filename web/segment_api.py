"""Flask routes for event_segmentation pipeline (web UI tab)."""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
from flask import jsonify, request, send_from_directory

# Project paths
WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
SEG_DIR = REPO_ROOT / "event_segmentation"
WEB_RUNS_DIR = SEG_DIR / "out" / "web_runs"
WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)

if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

from run_pipeline import run_full_pipeline  # noqa: E402
from segment import Config  # noqa: E402

# Import event loaders from the main app module (set by register_segment_routes).
_load_raw_events = None
_events_dir: Path | None = None

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

MAX_SEG_EVENTS = 200_000

ALLOWED_ASSETS = {
    "preview.gif",
    "preview.mp4",
    "trajectories.png",
    "speed_profiles.png",
    "summary.json",
    "colormap.json",
    "tracks.json",
}

PARAM_GROUPS: list[dict] = [
    {
        "id": "io",
        "label": "I/O",
        "fields": [
            {"key": "limit", "label": "Event limit (0 = all)", "type": "int"},
            {"key": "seed", "label": "Random seed", "type": "int"},
            {"key": "csv", "label": "Write CSV (not parquet)", "type": "bool"},
            {"key": "render_frames", "label": "Render frame PNGs", "type": "bool"},
        ],
    },
    {
        "id": "graph",
        "label": "Neighbour graph",
        "fields": [
            {"key": "dt_window", "label": "Time window (s)", "type": "float"},
            {"key": "k", "label": "Neighbours per event", "type": "int"},
            {"key": "beta", "label": "Time→space beta", "type": "float"},
            {"key": "grid_cell", "label": "Grid cell (px)", "type": "int"},
            {"key": "min_nbrs", "label": "Min neighbours", "type": "int"},
            {"key": "lag", "label": "Lag buffer (s)", "type": "float"},
        ],
    },
    {
        "id": "tracks",
        "label": "Track assignment",
        "fields": [
            {"key": "gate_px", "label": "Spatial gate (px)", "type": "float"},
            {"key": "vel_tol", "label": "Velocity tolerance (px/s)", "type": "float"},
            {"key": "lam", "label": "EMA weight (lambda)", "type": "float"},
            {"key": "min_track_events", "label": "Min track events", "type": "int"},
            {"key": "track_timeout", "label": "Track timeout (s)", "type": "float"},
            {"key": "v_max", "label": "Max velocity (px/s)", "type": "float"},
        ],
    },
    {
        "id": "split",
        "label": "Split / merge",
        "fields": [
            {"key": "split_px", "label": "Split radius (px)", "type": "float"},
            {"key": "split_every", "label": "Split every N events", "type": "int"},
            {"key": "min_track_lifetime_events", "label": "Min lifetime events", "type": "int"},
            {"key": "merge_vel_tol", "label": "Merge vel tol (px/s)", "type": "float"},
            {"key": "merge_px", "label": "Merge centroid (px)", "type": "float"},
        ],
    },
    {
        "id": "sensor",
        "label": "Sensor (0 = auto)",
        "fields": [
            {"key": "width", "label": "Width", "type": "int"},
            {"key": "height", "label": "Height", "type": "int"},
        ],
    },
    {
        "id": "timing",
        "label": "Output timing",
        "fields": [
            {"key": "traj_dt", "label": "Trajectory step (s)", "type": "float"},
            {"key": "frame_dt", "label": "Frame duration (s)", "type": "float"},
        ],
    },
]


def _coerce_value(key: str, val, field_type: str):
    if field_type == "bool":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("1", "true", "yes", "on")
        return bool(val)
    if field_type == "int":
        return int(val)
    if field_type == "float":
        return float(val)
    return val


def _build_config(params: dict, input_txt: Path, out_dir: Path) -> tuple[Config, int, float, float]:
    """Merge user params into Config; return (cfg, viz_fps, viz_frame_dt, viz_max_time)."""
    defaults = Config()
    merged = asdict(defaults)
    viz_fps = int(params.get("viz_fps", 25))
    viz_max_time = float(params.get("viz_max_time", 0.0))
    for group in PARAM_GROUPS:
        for field in group["fields"]:
            key = field["key"]
            if key in ("viz_fps", "viz_max_time"):
                continue
            if key in params:
                merged[key] = _coerce_value(key, params[key], field["type"])
    merged["input"] = str(input_txt)
    merged["out"] = str(out_dir)
    cfg = Config(**{k: merged[k] for k in asdict(Config()).keys()})
    return cfg, viz_fps, cfg.frame_dt, viz_max_time


def load_labeled_payload(run_dir: Path) -> dict:
    """Load labeled events + colormap for browser playback."""
    from visualize import load_labeled_events, load_colormap  # noqa: WPS433

    run_dir = Path(run_dir)
    t, x, y, inst = load_labeled_events(run_dir)
    raw_cmap = load_colormap(run_dir)
    colormap = {str(k): list(v) for k, v in raw_cmap.items()}

    order = np.argsort(t, kind="stable")
    t, x, y, inst = t[order], x[order], y[order], inst[order]
    n = int(t.size)
    downsampled_from = None
    if n > MAX_SEG_EVENTS:
        idx = np.linspace(0, n - 1, MAX_SEG_EVENTS).astype(np.int64)
        t, x, y, inst = t[idx], x[idx], y[idx], inst[idx]
        downsampled_from = n
        n = MAX_SEG_EVENTS

    t0 = float(t[0])
    return {
        "count": n,
        "downsampled_from": downsampled_from,
        "t_min": t0,
        "t_max": float(t[-1]),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "t": (t - t0).astype(np.float32).tolist(),
        "x": x.astype(np.float32).tolist(),
        "y": y.astype(np.float32).tolist(),
        "instance_id": inst.astype(np.int32).tolist(),
        "colormap": colormap,
    }


def export_events_txt(source: Path, dest: Path) -> int:
    """Convert any supported event file to t x y p text for segment.py."""
    parsed = _load_raw_events(source)
    if parsed is None:
        raise ValueError(f"Could not parse '{source.name}' as an event stream.")
    t, x, y, p = parsed
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]
    if t.size == 0:
        raise ValueError("Event stream is empty after parsing.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", buffering=1 << 20) as fh:
        for i in range(t.size):
            pi = int(p[i]) if np.isfinite(p[i]) else 0
            fh.write(f"{t[i]:.6f} {int(x[i])} {int(y[i])} {pi}\n")
    return int(t.size)


def _run_job(job_id: str, cfg: Config, viz_fps: int, viz_frame_dt: float, viz_max_time: float) -> None:
    def progress(msg: str, pct: float | None) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job["message"] = msg
                if pct is not None:
                    job["progress"] = pct

    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["message"] = "Starting…"
        result = run_full_pipeline(
            cfg,
            viz_fps=viz_fps,
            viz_frame_dt=viz_frame_dt,
            viz_max_time=viz_max_time,
            progress=progress,
            skip_animation=True,
        )
        preview_name = None
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "done",
                "progress": 1.0,
                "message": "Done.",
                "summary": result.get("summary"),
                "preview": preview_name,
                "result": {
                    "n_events": result.get("n_events"),
                    "n_instances": result.get("n_instances"),
                    "runtime_s": round(result.get("runtime_s", 0), 2),
                },
            })
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "message": str(e),
                "error": str(e),
            })


def register_segment_routes(app, load_raw_events_fn, events_dir: Path) -> None:
    """Attach segmentation API routes to the Flask app."""
    global _load_raw_events, _events_dir
    _load_raw_events = load_raw_events_fn
    _events_dir = events_dir

    @app.get("/api/segment/defaults")
    def segment_defaults():
        cfg = Config()
        defaults = asdict(cfg)
        return jsonify({"defaults": defaults, "groups": PARAM_GROUPS})

    @app.post("/api/segment/run")
    def segment_run():
        body = request.get_json(silent=True) or {}
        name = body.get("input", "")
        params = body.get("params") or {}
        if not name:
            return jsonify({"error": "input is required"}), 400
        src = _events_dir / Path(name).name
        if not src.exists():
            return jsonify({"error": f"file not found: {name}"}), 404

        job_id = uuid.uuid4().hex[:12]
        out_dir = WEB_RUNS_DIR / job_id
        txt_path = out_dir / "input_events.txt"
        try:
            n_events = export_events_txt(src, txt_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        try:
            cfg, viz_fps, viz_frame_dt, viz_max_time = _build_config(params, txt_path, out_dir)
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"invalid parameters: {e}"}), 400

        with _jobs_lock:
            _jobs[job_id] = {
                "status": "queued",
                "progress": 0.0,
                "message": f"Queued ({n_events:,} events)…",
                "job_id": job_id,
                "input": name,
                "created": time.time(),
            }

        thread = threading.Thread(
            target=_run_job,
            args=(job_id, cfg, viz_fps, viz_frame_dt, viz_max_time),
            daemon=True,
        )
        thread.start()
        return jsonify({"ok": True, "job_id": job_id, "n_events": n_events})

    @app.get("/api/segment/status")
    def segment_status():
        job_id = request.args.get("job_id", "")
        if not job_id:
            return jsonify({"error": "job_id is required"}), 400
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return jsonify({"error": "job not found"}), 404
            return jsonify({
                "job_id": job_id,
                "status": job["status"],
                "progress": job.get("progress", 0),
                "message": job.get("message", ""),
                "error": job.get("error"),
                "summary": job.get("summary"),
                "result": job.get("result"),
            })

    @app.get("/api/segment/events")
    def segment_events():
        job_id = request.args.get("job_id", "")
        if not job_id:
            return jsonify({"error": "job_id is required"}), 400
        run_dir = WEB_RUNS_DIR / job_id
        if not run_dir.is_dir():
            return jsonify({"error": "job not found"}), 404
        try:
            payload = load_labeled_payload(run_dir)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(payload)

    @app.get("/api/segment/assets/<job_id>/<path:filename>")
    def segment_asset(job_id: str, filename: str):
        safe_name = Path(filename).name
        if safe_name not in ALLOWED_ASSETS:
            return jsonify({"error": "asset not allowed"}), 403
        run_dir = WEB_RUNS_DIR / job_id
        if not run_dir.is_dir():
            return jsonify({"error": "job not found"}), 404
        path = run_dir / safe_name
        if not path.exists():
            return jsonify({"error": f"asset not found: {safe_name}"}), 404
        return send_from_directory(run_dir, safe_name)
