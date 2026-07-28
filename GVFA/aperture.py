"""
aperture.py — Shared aperture constraint extraction + flow-resolver dispatcher.

Normal flow constrains velocity to a line n_hat · v = u. Resolvers intersect
neighbour constraints to recover full flow (Adelson & Movshon 1982).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def extract_constraints(vx_perp, vy_perp, *, u_min=1e-6):
    """Extract (u, n_hat, valid) from normal flow.

    u = ||v_perp||, n_hat = v_perp / u.  u < u_min => valid=False.
    """
    vx_perp = np.asarray(vx_perp, dtype=np.float64)
    vy_perp = np.asarray(vy_perp, dtype=np.float64)
    u = np.hypot(vx_perp, vy_perp)
    valid = u >= float(u_min)
    n_hat = np.zeros((len(u), 2), dtype=np.float64)
    n_hat[valid, 0] = vx_perp[valid] / u[valid]
    n_hat[valid, 1] = vy_perp[valid] / u[valid]
    frac = 100.0 * float(valid.mean()) if len(valid) else 0.0
    print(f"[aperture] constraints: valid={int(valid.sum())}/{len(valid)} "
          f"({frac:.1f}%)  u_min={u_min}")
    return u, n_hat, valid


def orientation_correlation(n_hat, vx_res, vy_res, valid):
    """Circular correlation between edge-normal angle and resolved-flow angle.

    High => flow still locked to edge orientation (bad).
    Low  => flow reflects object motion (good).
    """
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return float("nan")
    ang_n = np.arctan2(n_hat[valid, 1], n_hat[valid, 0])
    ang_v = np.arctan2(vy_res[valid], vx_res[valid])
    # circular correlation via complex exponentials
    zn = np.exp(1j * ang_n)
    zv = np.exp(1j * ang_v)
    zn = zn - zn.mean()
    zv = zv - zv.mean()
    denom = np.sqrt(np.mean(np.abs(zn) ** 2) * np.mean(np.abs(zv) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.real(np.mean(np.conj(zn) * zv)) / denom)


def _median_speed(vx, vy, mask=None):
    spd = np.hypot(vx, vy)
    if mask is not None:
        spd = spd[mask]
    spd = spd[spd > 1e-12]
    return float(np.median(spd)) if spd.size else 0.0


def _base_info(method, vx_before, vy_before, vx_after, vy_after,
               resolved_mask, valid, n_hat, runtime_s, **extra):
    info = {
        "method": method,
        "resolved_frac": float(resolved_mask.mean()) if len(resolved_mask) else 0.0,
        "median_speed_before": _median_speed(vx_before, vy_before, valid),
        "median_speed_after": _median_speed(vx_after, vy_after, valid),
        "orientation_corr": orientation_correlation(
            n_hat, vx_after, vy_after, valid),
        "runtime_s": float(runtime_s),
        "n_unresolved": int((~resolved_mask & valid).sum()),
        "n_valid_constraints": int(valid.sum()),
    }
    info.update(extra)
    return info


def resolve_flow(x, y, vx_perp, vy_perp, edge_index_list, *, method, cfg):
    """Dispatch aperture resolvers. Returns (vx, vy, resolved_mask, info).

    method: "none" | "lk" | "affine" | "vsa"
    Unresolved events keep raw normal flow (never NaN).
    """
    method = str(method).lower()
    vx_perp = np.asarray(vx_perp, dtype=np.float64)
    vy_perp = np.asarray(vy_perp, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    u_min = float(cfg.get("U_MIN", 1e-6))
    u, n_hat, valid = extract_constraints(vx_perp, vy_perp, u_min=u_min)

    t0 = time.perf_counter()

    if method == "none":
        # Ablation baseline — reproduce raw normal flow exactly
        resolved_mask = valid.copy()
        info = _base_info(
            "none", vx_perp, vy_perp, vx_perp, vy_perp,
            resolved_mask, valid, n_hat, time.perf_counter() - t0,
        )
        print(f"[resolve] method=none  resolved_frac={info['resolved_frac']:.3f}  "
              f"orient_corr={info['orientation_corr']:.4f}  "
              f"runtime={info['runtime_s']:.3f}s")
        return vx_perp.copy(), vy_perp.copy(), resolved_mask, info

    if method == "lk":
        from aperture_lk import resolve_flow_lk
        vx, vy, resolved_mask, extra = resolve_flow_lk(
            x, y, u, n_hat, valid, edge_index_list,
            vx_perp=vx_perp, vy_perp=vy_perp,
            min_support=int(cfg.get("LK_MIN_SUPPORT", 6)),
            min_eig=float(cfg.get("CONDITION_MIN_EIG", 1e-3)),
            huber_iters=int(cfg.get("LK_HUBER_ITERS", 3)),
        )
    elif method == "affine":
        from aperture_lk import resolve_flow_affine
        vx, vy, resolved_mask, extra = resolve_flow_affine(
            x, y, u, n_hat, valid, edge_index_list,
            vx_perp=vx_perp, vy_perp=vy_perp,
            min_support=int(cfg.get("AFFINE_MIN_SUPPORT", 12)),
            min_eig=float(cfg.get("CONDITION_MIN_EIG", 1e-3)),
            huber_iters=int(cfg.get("LK_HUBER_ITERS", 3)),
            lk_min_support=int(cfg.get("LK_MIN_SUPPORT", 6)),
        )
    elif method == "vsa":
        from vsa_aperture import resolve_flow_vsa
        vx, vy, resolved_mask, extra = resolve_flow_vsa(
            x, y, u, n_hat, valid, edge_index_list,
            vx_perp=vx_perp, vy_perp=vy_perp, cfg=cfg,
        )
    else:
        raise ValueError(f"unknown MOTION_RESOLVER={method!r}")

    # Safety: no NaNs; unresolved keep raw normal flow
    bad = ~np.isfinite(vx) | ~np.isfinite(vy)
    if bad.any():
        vx[bad] = vx_perp[bad]
        vy[bad] = vy_perp[bad]
        resolved_mask = resolved_mask.copy()
        resolved_mask[bad] = False
    keep_raw = ~resolved_mask
    vx = vx.copy()
    vy = vy.copy()
    vx[keep_raw] = vx_perp[keep_raw]
    vy[keep_raw] = vy_perp[keep_raw]

    runtime = time.perf_counter() - t0
    info = _base_info(
        method, vx_perp, vy_perp, vx, vy,
        resolved_mask, valid, n_hat, runtime, **extra,
    )
    print(f"[resolve] method={method}  resolved_frac={info['resolved_frac']:.3f}  "
          f"orient_corr={info['orientation_corr']:.4f}  "
          f"unresolved={info['n_unresolved']}  runtime={info['runtime_s']:.3f}s")
    return vx, vy, resolved_mask, info


def undirected_edges_from_list(edge_index_list, n_nodes):
    """Unique undirected pairs + self-loops as (src, dst) arrays for add.at."""
    pairs = []
    for ei in edge_index_list:
        if ei is None:
            continue
        if hasattr(ei, "numel") and ei.numel() == 0:
            continue
        if hasattr(ei, "numpy"):
            a = ei[0].numpy().astype(np.int64)
            b = ei[1].numpy().astype(np.int64)
        else:
            a = np.asarray(ei[0], dtype=np.int64)
            b = np.asarray(ei[1], dtype=np.int64)
            if a.size == 0:
                continue
        pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    if pairs:
        uniq = np.unique(np.concatenate(pairs, axis=0), axis=0)
        # undirected both ways
        src = np.concatenate([uniq[:, 0], uniq[:, 1]])
        dst = np.concatenate([uniq[:, 1], uniq[:, 0]])
    else:
        src = np.empty(0, np.int64)
        dst = np.empty(0, np.int64)
    # self-loops
    self_i = np.arange(n_nodes, dtype=np.int64)
    src = np.concatenate([src, self_i])
    dst = np.concatenate([dst, self_i])
    return src, dst  # accumulate at dst from src (neighbour = src, node = dst)
