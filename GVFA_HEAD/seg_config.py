"""Shared graph / FPE config extracted from segment.py for supervised FG/BG.

Contains constants and helpers used by seg_model.py / dataset.py:
  make_codebooks, build_multigraph, node_flow, smooth_flow.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from fpe_codebook import FPECodebook, PHASE_INT_KMAX

# ----------------------------------------------------------------------------
# PARAMETERS (from segment.py)
# ----------------------------------------------------------------------------
SENSOR = (346, 260)  # (W, H) in pixels — CLI / DAT header may override

# Spatial graph: ellipsoid elongated in (x, y) — local spatial structure
SPATIAL_R_XY_FRAC = 0.04  # R_XY = this fraction of sensor width
SPATIAL_R_T_MS = 5.0      # semi-minor axis along time (ms)
SPATIAL_MMAX = 16         # max past spatial neighbours per node

# Temporal graph: ellipsoid elongated in t — motion over longer horizons
TEMPORAL_R_XY_FRAC = 0.01  # R_XY = this fraction of sensor width
TEMPORAL_R_T_MS = 40.0     # semi-major axis along time (ms)
TEMPORAL_MMAX = 12         # max past temporal neighbours per node

D = 4000          # hypervector dimensionality per graph branch
NUM_LAYERS = 3    # GraphCNN layers incl. input
SEED = 0

# === STAGE 1: flow regularization ===
FLOW_SMOOTH_ITERS = 4
FLOW_KEEP = 0.5

# === FPE CODEBOOK CONFIG ===
BW_X, BW_Y = 0.0333, 0.0333
BW_T = 1e-5
BW_P = 1.43
BW_DX, BW_DY = 0.1, 0.1
BW_DT = 3.3e-4
BW_VX, BW_VY = 0.1, 0.1
BW_DP = 1.43
BW_SPEED = 0.1
SPEED_V0 = 50.0
N_ANGLE_BINS = 720

# Node bundle weights — motion weighted ~2x position
W_NODE_X, W_NODE_Y = 0.3, 0.3
W_NODE_T = 0.5
W_NODE_P = 0.0
W_NODE_MOTION = 2.0

# Spatial edge bundle weights
W_EDGE_S_DX, W_EDGE_S_DY, W_EDGE_S_DT = 0.5, 0.5, 0.3

# Temporal edge bundle weights
W_EDGE_T_DX, W_EDGE_T_DY, W_EDGE_T_DT = 0.5, 0.5, 0.3
W_EDGE_T_VX, W_EDGE_T_VY, W_EDGE_T_DP = 5.0, 5.0, 0.5


# ----------------------------------------------------------------------------
# Causal ellipsoid multigraph
# ----------------------------------------------------------------------------
def _ellipsoid_metric(dx, dy, dt_ms, r_xy, r_t):
    """||dxy||/R_XY + |dt|/R_t  (must be < 1 for an edge)."""
    return np.hypot(dx, dy) / r_xy + np.abs(dt_ms) / r_t


def edge_features_spatial(rec, src, x, y, t):
    """Spatial edge features (Eq. 5): e_ij = (Δx, Δy, Δt)."""
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    return np.stack([dx, dy, dt], axis=1).astype(np.float64)


def edge_features_temporal(rec, src, x, y, t, p):
    """Temporal edge features (Eq. 6): (Δx, Δy, Δt, Δx/Δt, Δy/Δt, Δp)."""
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    dp = p[rec] - p[src]
    dt_safe = np.where(np.abs(dt) > 1e-12, dt, np.sign(dt) * 1e-12 + 1e-12)
    return np.stack([dx, dy, dt, dx / dt_safe, dy / dt_safe, dp], axis=1).astype(np.float64)


def _build_causal_ellipsoid_edges(t, x, y, r_xy, r_t, mmax):
    """Build one directed causal graph under the ellipsoid constraint.

    A directed edge (i -> j) is kept when t_i < t_j and
        ||v_i^xy - v_j^xy|| / R_XY + |t_i - t_j| / R_t < 1.

    edge_index[0] = receiver j (later), edge_index[1] = source i (earlier).
    """
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


def build_multigraph(
    t, x, y, p, sensor=SENSOR,
    spatial_r_xy_frac=SPATIAL_R_XY_FRAC,
    spatial_r_t_ms=SPATIAL_R_T_MS,
    spatial_mmax=SPATIAL_MMAX,
    temporal_r_xy_frac=TEMPORAL_R_XY_FRAC,
    temporal_r_t_ms=TEMPORAL_R_T_MS,
    temporal_mmax=TEMPORAL_MMAX,
):
    """Build spatial (E_s) and temporal (E_t) causal ellipsoid graphs.

    Returns
        edge_spatial, edge_temporal : torch.LongTensor [2, E]
        edge_attr_spatial  : np.ndarray [E, 3]
        edge_attr_temporal : np.ndarray [E, 6]
        rec_s, src_s : int arrays for spatial connected components
    """
    w, _ = sensor
    r_xy_s = spatial_r_xy_frac * w
    r_xy_t = temporal_r_xy_frac * w

    edge_spatial, rec_s, src_s = _build_causal_ellipsoid_edges(
        t, x, y, r_xy_s, spatial_r_t_ms, spatial_mmax)
    edge_temporal, rec_t, src_t = _build_causal_ellipsoid_edges(
        t, x, y, r_xy_t, temporal_r_t_ms, temporal_mmax)

    attr_spatial = edge_features_spatial(rec_s, src_s, x, y, t)
    attr_temporal = edge_features_temporal(rec_t, src_t, x, y, t, p)

    return (edge_spatial, edge_temporal,
            attr_spatial, attr_temporal,
            rec_s, src_s)


# ----------------------------------------------------------------------------
# Node flow (causal normal optical flow)
# ----------------------------------------------------------------------------
def _solve3(M, rhs):
    """Closed-form 3x3 solve (no BLAS) with ridge assumed already in M."""
    a00, a01, a02 = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    a10, a11, a12 = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    a20, a21, a22 = float(M[2, 0]), float(M[2, 1]), float(M[2, 2])
    b0, b1, b2 = float(rhs[0]), float(rhs[1]), float(rhs[2])
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if abs(det) < 1e-18:
        return np.zeros(3, dtype=np.float64)
    inv = 1.0 / det
    x0 = inv * (
        b0 * (a11 * a22 - a12 * a21)
        - a01 * (b1 * a22 - a12 * b2)
        + a02 * (b1 * a21 - a11 * b2)
    )
    x1 = inv * (
        a00 * (b1 * a22 - a12 * b2)
        - b0 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * b2 - b1 * a20)
    )
    x2 = inv * (
        a00 * (a11 * b2 - b1 * a21)
        - a01 * (a10 * b2 - b1 * a20)
        + b0 * (a10 * a21 - a11 * a20)
    )
    return np.array([x0, x1, x2], dtype=np.float64)


def _fit_plane(N, rec, src, x, y, tt, active, ridge):
    """Least-squares fit t ~= a*x + b*y + c per node over {self} U {active past
    neighbours}. Per-node 3x3 solve (avoids OpenBLAS batched-solve crashes)."""
    Sxx = x * x; Sxy = x * y; Syy = y * y
    Sx = x.copy(); Sy = y.copy(); S1 = np.ones(N, dtype=np.float64)
    Stx = tt * x; Sty = tt * y; St = tt.copy()
    if np.any(active):
        ri = rec[active]; xj = x[src[active]]; yj = y[src[active]]; tj = tt[src[active]]
        np.add.at(Sxx, ri, xj * xj); np.add.at(Sxy, ri, xj * yj); np.add.at(Syy, ri, yj * yj)
        np.add.at(Sx, ri, xj);       np.add.at(Sy, ri, yj);       np.add.at(S1, ri, 1.0)
        np.add.at(Stx, ri, tj * xj); np.add.at(Sty, ri, tj * yj); np.add.at(St, ri, tj)

    coef = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        M = np.array([
            [Sxx[i] + ridge, Sxy[i],         Sx[i]],
            [Sxy[i],         Syy[i] + ridge, Sy[i]],
            [Sx[i],          Sy[i],          S1[i] + ridge],
        ], dtype=np.float64)
        rhs = np.array([Stx[i], Sty[i], St[i]], dtype=np.float64)
        coef[i] = _solve3(M, rhs)
    return coef, S1


def node_flow(t, x, y, edge_index, min_pts=5, ridge=1e-9, clip_pct=99.0):
    """Per-event normal optical flow by fitting a local plane to the time surface.

    Returns vx, vy in px/s; nodes with too few / aperture-degenerate support get 0.
    """
    rec = edge_index[0].numpy(); src = edge_index[1].numpy()
    N = len(t)
    tt = (t - t[0])
    if rec.shape[0] == 0:
        return np.zeros(N), np.zeros(N)

    active = np.ones(rec.shape[0], dtype=bool)
    coef, _ = _fit_plane(N, rec, src, x, y, tt, active, ridge)

    a, b, c = coef[rec, 0], coef[rec, 1], coef[rec, 2]
    resid = tt[src] - (a * x[src] + b * y[src] + c)
    rss = np.zeros(N); cnt = np.zeros(N)
    np.add.at(rss, rec, resid * resid); np.add.at(cnt, rec, 1.0)
    scale = np.sqrt(rss / np.maximum(cnt, 1.0))
    active = np.abs(resid) <= (2.5 * scale[rec] + 1e-12)
    coef, npts = _fit_plane(N, rec, src, x, y, tt, active, ridge)

    a, b = coef[:, 0], coef[:, 1]
    g2 = a * a + b * b
    vx = np.zeros(N); vy = np.zeros(N)
    ok = (npts >= min_pts) & (g2 > 1e-12)
    vx[ok] = a[ok] / g2[ok]
    vy[ok] = b[ok] / g2[ok]

    spd = np.hypot(vx, vy)
    if ok.any():
        cap = np.percentile(spd[ok], clip_pct)
        scl = np.where((spd > cap) & (spd > 0), cap / np.maximum(spd, 1e-12), 1.0)
        vx *= scl; vy *= scl
    return vx, vy


def smooth_flow(vx, vy, edge_index_spatial, n_iters=4, keep=0.5, verbose=False):
    """Graph-neighbour regularization of normal flow (causal Stage-1 style).

    Returns (vx_s, vy_s, n_valid_before, n_valid_after).
    """
    vx_s = np.asarray(vx, dtype=np.float64).copy()
    vy_s = np.asarray(vy, dtype=np.float64).copy()
    n = len(vx_s)
    n_valid_before = int((np.hypot(vx_s, vy_s) > 1e-12).sum())

    if edge_index_spatial is None or edge_index_spatial.numel() == 0 or n_iters <= 0:
        return vx_s, vy_s, n_valid_before, n_valid_before

    rec = edge_index_spatial[0].numpy().astype(np.int64)
    src = edge_index_spatial[1].numpy().astype(np.int64)
    a = np.concatenate([rec, src])
    b = np.concatenate([src, rec])

    for _ in range(n_iters):
        valid = np.hypot(vx_s, vy_s) > 1e-12
        nbr_ok = valid[b]
        aa, bb = a[nbr_ok], b[nbr_ok]
        sx = np.zeros(n, dtype=np.float64)
        sy = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        if aa.size:
            np.add.at(sx, aa, vx_s[bb])
            np.add.at(sy, aa, vy_s[bb])
            np.add.at(cnt, aa, 1.0)
        has = cnt > 0
        mean_x = np.zeros(n, dtype=np.float64)
        mean_y = np.zeros(n, dtype=np.float64)
        mean_x[has] = sx[has] / cnt[has]
        mean_y[has] = sy[has] / cnt[has]
        new_x = vx_s.copy()
        new_y = vy_s.copy()
        new_x[has] = keep * vx_s[has] + (1.0 - keep) * mean_x[has]
        new_y[has] = keep * vy_s[has] + (1.0 - keep) * mean_y[has]
        vx_s, vy_s = new_x, new_y

    n_valid_after = int((np.hypot(vx_s, vy_s) > 1e-12).sum())
    if verbose:
        print(f"[smooth_flow] iters={n_iters} keep={keep}  "
              f"valid {n_valid_before} -> {n_valid_after}")
    return vx_s, vy_s, n_valid_before, n_valid_after


# ----------------------------------------------------------------------------
# FPE codebooks
# ----------------------------------------------------------------------------
def make_codebooks(sensor=SENSOR, t_span_s=0.06, seed=SEED):
    """Build node and edge FPE codebooks for one processing window."""
    W, H = sensor
    dt_s_max = int(SPATIAL_R_T_MS * 1000) + 1
    dt_t_max = int(TEMPORAL_R_T_MS * 1000) + 1

    node = {
        "x": FPECodebook("x", D, BW_X, "integer", vmin=0, vmax=W - 1,
                         phase_dist="gaussian", seed=seed + 1),
        "y": FPECodebook("y", D, BW_Y, "integer", vmin=0, vmax=H - 1,
                         phase_dist="gaussian", seed=seed + 2),
        "t": FPECodebook("t", D, BW_T, "radix", radix_S=1000, vmin=0,
                         vmax=1_000_000, value_grid_step=1.0,
                         phase_dist="gaussian", seed=seed + 3),
        "p": FPECodebook("p", D, BW_P, "integer", vmin=0, vmax=1,
                         phase_dist="gaussian", seed=seed + 4),
        "speed": FPECodebook(
            "speed", D, BW_SPEED, "signed_log_radix", radix_S=16,
            vmin=0, vmax=60, value_grid_step=0.1,
            signed_log_v0=SPEED_V0, phase_dist="gaussian", seed=seed + 20,
        ),
        "dir": FPECodebook(
            "dir", D, 1.0, "periodic", n_angle_bins=N_ANGLE_BINS,
            phase_dist="integer", phase_int_kmax=PHASE_INT_KMAX,
            seed=seed + 21,
        ),
    }
    edge_dx = FPECodebook("dx", D, BW_DX, "integer", vmin=-W, vmax=W,
                          phase_dist="gaussian", seed=seed + 10)
    edge_dy = FPECodebook("dy", D, BW_DY, "integer", vmin=-H, vmax=H,
                          phase_dist="gaussian", seed=seed + 11)
    edge_spatial = {
        "dx": edge_dx,
        "dy": edge_dy,
        "dt": FPECodebook("dt_s", D, BW_DT, "radix", radix_S=100,
                          vmin=0, vmax=dt_s_max, value_grid_step=1.0,
                          phase_dist="gaussian", seed=seed + 12),
    }
    edge_temporal = {
        "dx": edge_dx,
        "dy": edge_dy,
        "dt": FPECodebook("dt_t", D, BW_DT, "radix", radix_S=200,
                          vmin=0, vmax=dt_t_max, value_grid_step=1.0,
                          phase_dist="gaussian", seed=seed + 13),
        "vx": FPECodebook("vx", D, BW_VX, "signed_log_radix", radix_S=16,
                          vmin=-25, vmax=25, value_grid_step=0.1,
                          signed_log_v0=100.0, phase_dist="gaussian", seed=seed + 14),
        "vy": FPECodebook("vy", D, BW_VY, "signed_log_radix", radix_S=16,
                          vmin=-25, vmax=25, value_grid_step=0.1,
                          signed_log_v0=100.0, phase_dist="gaussian", seed=seed + 15),
        "dp": FPECodebook("dp", D, BW_DP, "integer", vmin=-1, vmax=1,
                          phase_dist="gaussian", seed=seed + 16),
    }
    w_spatial = {"dx": W_EDGE_S_DX, "dy": W_EDGE_S_DY, "dt": W_EDGE_S_DT}
    w_temporal = {
        "dx": W_EDGE_T_DX, "dy": W_EDGE_T_DY, "dt": W_EDGE_T_DT,
        "vx": W_EDGE_T_VX, "vy": W_EDGE_T_VY, "dp": W_EDGE_T_DP,
    }
    return node, edge_spatial, edge_temporal, w_spatial, w_temporal
