"""Config helpers shared by every entry point.

resolution_ratio
----------------
``dataset.resolution_ratio: r`` (or ``--resolution-ratio r``) processes the event
data at 1/r of the configured resolution: r=1 full, r=2 half, r=4 quarter.

The paper's pixel-sized settings describe a PHYSICAL extent at full resolution,
so they are divided by r to cover the same area of the scene:

    encoder.patch_size  (kernel aperture N)      21 -> 11 (r=2) -> 5 (r=4)
    matching.M          (cost-volume search)     31 -> 15 (r=2) -> 7 (r=4)
    matching.smooth     (Eq.12 pooling, "sc")    71 -> 35 (r=2) -> 17 (r=4)

Per-pixel similarity settings (sigma_k, FPE bandwidth) stay in working pixels: the
downsampled image is already area-averaged, so a working pixel is the natural unit.
Motion LABELS are unaffected: "moving" is defined in sensor pixels, so every ratio
is scored on exactly the same task.

Cost: the image has r^2 fewer pixels and the search window (M^2) shrinks by ~r^2,
so the cost volume is roughly r^4 cheaper.
"""

from __future__ import annotations

import copy
import math
from typing import Any

# Settings that define the frozen front end. A trained head is only valid for the
# front end it was trained on, so checkpoints store these and eval restores them.
_FRONTEND_KEYS = {
    "encoder": ("patch_size", "sigma_k", "kernel", "polarity_binding", "scales"),
    "matching": ("M", "scales", "alpha", "smooth", "vel_scale"),
    "time_surface": ("tau_ms", "decay"),
    "dataset": ("height", "width", "window_ms", "time_frames", "resolution_ratio"),
}


def _odd_scaled(value: int, ratio: int, minimum: int) -> int:
    """Divide a pixel extent by ratio and keep it odd (centred windows)."""
    return max(minimum, int(value) // ratio | 1)


def resolution_ratio_of(cfg: dict[str, Any]) -> int:
    r = int(cfg.get("dataset", {}).get("resolution_ratio", 1) or 1)
    if r < 1:
        raise ValueError(f"resolution_ratio must be >= 1, got {r}")
    return r


def apply_resolution_ratio(
    cfg: dict[str, Any], ratio: int | None = None, *, verbose: bool = False,
) -> dict[str, Any]:
    """Return a copy of cfg with the resolution ratio applied (idempotent).

    ``ratio`` overrides ``dataset.resolution_ratio`` when given (CLI flag).
    Entry points call this once with verbose=True; library code calls it
    defensively, which is a silent no-op when it has already been applied.
    """
    cfg = copy.deepcopy(cfg)
    if cfg.get("_resolution_applied"):
        return cfg
    ds = cfg.setdefault("dataset", {})
    if ratio is not None:
        ds["resolution_ratio"] = int(ratio)
    r = resolution_ratio_of(cfg)

    enc = cfg.setdefault("encoder", {})
    match = cfg.setdefault("matching", {})
    full = {
        "height": int(ds.get("height", 480)), "width": int(ds.get("width", 640)),
        "patch_size": int(enc.get("patch_size", 21)), "M": int(match.get("M", 7)),
        "smooth": int(match.get("smooth", 1)),
    }
    ds["height"] = full["height"] // r
    ds["width"] = full["width"] // r
    enc["patch_size"] = _odd_scaled(full["patch_size"], r, 3)
    match["M"] = _odd_scaled(full["M"], r, 3)
    match["smooth"] = _odd_scaled(full["smooth"], r, 1) if full["smooth"] > 1 else 1

    cfg["_resolution_applied"] = True
    cfg["_resolution_full"] = full
    if verbose:
        print(f"[config] resolution_ratio={r}: {ds['height']}x{ds['width']}  "
              f"patch_size {full['patch_size']}->{enc['patch_size']}  "
              f"M {full['M']}->{match['M']}  smooth {full['smooth']}->{match['smooth']}")
    return cfg


def frontend_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """The un-scaled front-end settings, for storing in a checkpoint."""
    full = cfg.get("_resolution_full", {})
    out: dict[str, Any] = {"d": cfg.get("d")}
    for section, keys in _FRONTEND_KEYS.items():
        src = cfg.get(section, {}) or {}
        sec = {k: src[k] for k in keys if k in src}
        for k in keys:                      # store full-resolution values, not scaled ones
            if k in full:
                sec[k] = full[k]
        out[section] = sec
    return out


def restore_frontend(cfg: dict[str, Any], saved: dict[str, Any] | None) -> dict[str, Any]:
    """Overwrite the front-end sections of cfg with those saved in a checkpoint."""
    if not saved:
        return cfg
    cfg = copy.deepcopy(cfg)
    # saved values are FULL resolution: the ratio must be applied again afterwards
    cfg.pop("_resolution_applied", None)
    cfg.pop("_resolution_full", None)
    if saved.get("d") is not None:
        cfg["d"] = saved["d"]
    for section in _FRONTEND_KEYS:
        if saved.get(section):
            cfg[section] = {**cfg.get(section, {}), **saved[section]}
    return cfg


def surface_decay(cfg: dict[str, Any]) -> float:
    """Per-second decay factor for the accumulative time surface (paper Eq. 3).

    ``time_surface.tau_ms`` (paper: 35 ms) gives weight exp(-age / tau); the code
    applies ``decay ** age_seconds``, so decay = exp(-1 / tau_seconds). Falls back
    to the legacy ``time_surface.decay`` when tau_ms is not set.
    """
    ts = cfg.get("time_surface", {}) or {}
    tau_ms = ts.get("tau_ms")
    if tau_ms:
        return math.exp(-1000.0 / float(tau_ms))
    return float(ts.get("decay", 0.8))
