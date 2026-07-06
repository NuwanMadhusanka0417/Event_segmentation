"""Causal ellipsoid multigraph + FPE node encoding (aligned with GVFA/segment.py)."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

SENSOR = (346, 260)

SPATIAL_R_XY_FRAC = 0.04
SPATIAL_R_T_MS = 5.0
SPATIAL_MMAX = 16

TEMPORAL_R_XY_FRAC = 0.01
TEMPORAL_R_T_MS = 40.0
TEMPORAL_MMAX = 12

POS_BW = 1.0
TIME_BW = 0.5
SEED = 0


def _ellipsoid_metric(dx, dy, dt_ms, r_xy, r_t):
    return np.hypot(dx, dy) / r_xy + np.abs(dt_ms) / r_t


def edge_features_spatial(rec, src, x, y, t):
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    return np.stack([dx, dy, dt], axis=1).astype(np.float64)


def edge_features_temporal(rec, src, x, y, t, p):
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    dp = p[rec] - p[src]
    dt_safe = np.where(np.abs(dt) > 1e-12, dt, np.sign(dt) * 1e-12 + 1e-12)
    return np.stack([dx, dy, dt, dx / dt_safe, dy / dt_safe, dp], axis=1).astype(np.float64)


def _build_causal_ellipsoid_edges(t, x, y, r_xy, r_t, mmax):
    if len(t) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return edge_index, np.empty(0, np.int64), np.empty(0, np.int64)
    t_ms = (t - t[0]) * 1e3
    xy = np.stack([x, y], axis=1).astype(np.float64)

    nn = NearestNeighbors(radius=r_xy).fit(xy)
    dists_xy, idxs = nn.radius_neighbors(xy, return_distance=True)

    rec, src = [], []
    for j in range(len(t)):
        cand = idxs[j]
        if cand.size == 0:
            continue
        past_mask = t[cand] < t[j]
        past = cand[past_mask]
        if past.size == 0:
            continue

        dx = x[j] - x[past]
        dy = y[j] - y[past]
        dt = t_ms[j] - t_ms[past]
        metric = _ellipsoid_metric(dx, dy, dt, r_xy, r_t)
        keep = metric < 1.0
        past = past[keep]
        metric = metric[keep]
        if past.size == 0:
            continue
        if past.size > mmax:
            order = np.argsort(metric)[:mmax]
            past = past[order]

        rec.append(np.full(past.size, j, dtype=np.int64))
        src.append(past.astype(np.int64))

    if rec:
        rec = np.concatenate(rec)
        src = np.concatenate(src)
    else:
        rec = np.empty(0, np.int64)
        src = np.empty(0, np.int64)

    edge_index = torch.from_numpy(np.stack([rec, src], axis=0)).long()
    return edge_index, rec, src


def build_multigraph(t, x, y, p, sensor=SENSOR,
                     spatial_r_xy_frac=SPATIAL_R_XY_FRAC,
                     spatial_r_t_ms=SPATIAL_R_T_MS,
                     spatial_mmax=SPATIAL_MMAX,
                     temporal_r_xy_frac=TEMPORAL_R_XY_FRAC,
                     temporal_r_t_ms=TEMPORAL_R_T_MS,
                     temporal_mmax=TEMPORAL_MMAX):
    """Build spatial (E_s) and temporal (E_t) causal ellipsoid graphs.

    Spatial edge features  — (Δx, Δy, Δt).
    Temporal edge features — (Δx, Δy, Δt, Δx/Δt, Δy/Δt, Δp).

    `t` must be in seconds (monotone).  Returns edge indices and edge attrs.
    """
    if len(t) == 0:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea_s = np.zeros((0, 3), dtype=np.float64)
        ea_t = np.zeros((0, 6), dtype=np.float64)
        vx_e = np.empty(0, np.float64)
        vy_e = np.empty(0, np.float64)
        return ei, ei, ea_s, ea_t, np.empty(0, np.int64), np.empty(0, np.int64), vx_e, vy_e

    w, _ = sensor
    r_xy_s = spatial_r_xy_frac * w
    r_xy_t = temporal_r_xy_frac * w

    edge_spatial, rec_s, src_s = _build_causal_ellipsoid_edges(
        t, x, y, r_xy_s, spatial_r_t_ms, spatial_mmax)
    edge_temporal, rec_t, src_t = _build_causal_ellipsoid_edges(
        t, x, y, r_xy_t, temporal_r_t_ms, temporal_mmax)

    attr_spatial = edge_features_spatial(rec_s, src_s, x, y, t)
    attr_temporal = edge_features_temporal(rec_t, src_t, x, y, t, p)
    # temporal cols 3,4 = Δx/Δt, Δy/Δt in px/s -> px/ms for codebook
    vx_edge = (attr_temporal[:, 3] / 1000.0).astype(np.float64) if attr_temporal.size else np.empty(0)
    vy_edge = (attr_temporal[:, 4] / 1000.0).astype(np.float64) if attr_temporal.size else np.empty(0)

    return (edge_spatial, edge_temporal,
            attr_spatial, attr_temporal,
            rec_s, src_s, vx_edge, vy_edge)
