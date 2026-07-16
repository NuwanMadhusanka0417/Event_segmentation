"""
ego_motion.py — Stage 2: 4-param similarity ego-motion fit + residual IMO split.

Fits a global camera motion model to smoothed normal flow via IRLS, then splits
events into background (residual-consistent) vs IMO candidates (outliers).
Motion is used ONLY for this split / later affinity — never as a node feature.
"""

from __future__ import annotations

import numpy as np


def ego_field(x, y, params, cx, cy):
    """4-param similarity flow at (x, y).

    v = [tx - w*(y-cy) + s*(x-cx),
         ty + w*(x-cx) + s*(y-cy)]
    params = (tx, ty, w, s)
    """
    tx, ty, w, s = params
    dx = x - cx
    dy = y - cy
    vx = tx - w * dy + s * dx
    vy = ty + w * dx + s * dy
    return vx, vy


def _design_matrix(x, y, cx, cy):
    """Stack 2N x 4 design matrix for linear similarity flow."""
    dx = x - cx
    dy = y - cy
    n = len(x)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    # vx row: [1, 0, -(y-cy), (x-cx)]
    A[0::2, 0] = 1.0
    A[0::2, 2] = -dy
    A[0::2, 3] = dx
    # vy row: [0, 1, (x-cx), (y-cy)]
    A[1::2, 1] = 1.0
    A[1::2, 2] = dx
    A[1::2, 3] = dy
    return A


def _mad(arr):
    """Median absolute deviation (scalar)."""
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def fit_ego_motion(x, y, vx, vy, sensor, n_iters=10, valid_mask=None):
    """IRLS fit of 4-param similarity ego-motion to smoothed flow.

    Parameters
    ----------
    x, y, vx, vy : arrays [N]
    sensor : (W, H)
    n_iters : IRLS iterations (Huber)
    valid_mask : optional bool [N]; default |v| > 0

    Returns
    -------
    params : (tx, ty, w, s)
    residual : [N, 2]  r_i = v_i - v_pred(x_i, y_i)  (0 where invalid)
    info : dict with inlier_rms, mad, n_valid, cx, cy
    """
    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    n = len(x)
    if valid_mask is None:
        valid_mask = (np.hypot(vx, vy) > 1e-12)
    valid = np.asarray(valid_mask, dtype=bool)
    n_valid = int(valid.sum())

    residual = np.zeros((n, 2), dtype=np.float64)
    if n_valid < 4:
        print("[ego] WARNING: too few valid flow events for ego fit")
        return (0.0, 0.0, 0.0, 0.0), residual, {
            "inlier_rms": 0.0, "mad": 0.0, "n_valid": n_valid,
            "cx": cx, "cy": cy,
        }

    xv, yv = x[valid], y[valid]
    vxv, vyv = vx[valid], vy[valid]
    A = _design_matrix(xv, yv, cx, cy)
    b = np.empty(2 * n_valid, dtype=np.float64)
    b[0::2] = vxv
    b[1::2] = vyv

    # OLS init
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    weights = np.ones(n_valid, dtype=np.float64)

    for _ in range(n_iters):
        # expand per-event weights to 2N rows
        w2 = np.repeat(weights, 2)
        Aw = A * w2[:, None]
        bw = b * w2
        params, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

        pvx, pvy = ego_field(xv, yv, params, cx, cy)
        rn = np.hypot(vxv - pvx, vyv - pvy)
        mad = _mad(rn)
        delta = 1.4826 * max(mad, 1e-9)
        # Huber weight: 1 if |r|<=delta else delta/|r|
        weights = np.where(rn <= delta, 1.0, delta / np.maximum(rn, 1e-12))

    tx, ty, w, s = [float(p) for p in params]
    pvx_all, pvy_all = ego_field(x, y, (tx, ty, w, s), cx, cy)
    residual[:, 0] = vx - pvx_all
    residual[:, 1] = vy - pvy_all
    residual[~valid] = 0.0

    rn_all = np.hypot(residual[valid, 0], residual[valid, 1])
    # inliers: residual within Huber delta from last iter
    mad = _mad(rn_all)
    delta = 1.4826 * max(mad, 1e-9)
    inlier = rn_all <= delta
    if inlier.any():
        inlier_rms = float(np.sqrt(np.mean(rn_all[inlier] ** 2)))
    else:
        inlier_rms = float(np.sqrt(np.mean(rn_all ** 2))) if rn_all.size else 0.0

    print(f"[ego] params tx={tx:.4g} ty={ty:.4g} w={w:.4g} s={s:.4g}  "
          f"inlier_RMS={inlier_rms:.4g}  MAD={mad:.4g}  n_valid={n_valid}")
    return (tx, ty, w, s), residual, {
        "inlier_rms": inlier_rms,
        "mad": mad,
        "n_valid": n_valid,
        "cx": cx,
        "cy": cy,
        "delta": delta,
    }


def residual_split(residual, edge_index_spatial, res_k=3.0, valid_mask=None):
    """Split events into background vs IMO by residual magnitude + hysteresis.

    thresh = RES_K * (1.4826 * MAD(|r| over inliers-ish / all valid))
    |r| <= thresh -> background; > thresh -> IMO.
    One hysteresis pass on the spatial graph for boundary stability.

    Returns
    -------
    is_imo : bool [N]
    residual : [N, 2] (unchanged, returned for convenience)
    thresh : float
    """
    n = residual.shape[0]
    rn = np.hypot(residual[:, 0], residual[:, 1])
    if valid_mask is None:
        valid_mask = rn > 1e-12
    valid = np.asarray(valid_mask, dtype=bool) & (rn > 1e-12)

    if not valid.any():
        print("[ego] WARNING: no valid residuals for split — all background")
        return np.zeros(n, dtype=bool), residual, 0.0

    mad = _mad(rn[valid])
    sigma = 1.4826 * max(mad, 1e-9)
    thresh = float(res_k * sigma)

    is_imo = np.zeros(n, dtype=bool)
    is_imo[valid] = rn[valid] > thresh

    # hysteresis over spatial graph (undirected)
    if edge_index_spatial is not None and edge_index_spatial.numel() > 0:
        rec = edge_index_spatial[0].numpy().astype(np.int64)
        src = edge_index_spatial[1].numpy().astype(np.int64)
        # build undirected neighbour lists via CSR-like counts
        # vote: for each node, count IMO vs BG neighbours
        imo_votes = np.zeros(n, dtype=np.float64)
        bg_votes = np.zeros(n, dtype=np.float64)
        # edge i--j: j votes on i and i votes on j
        for a, b in ((rec, src), (src, rec)):
            # neighbour is IMO?
            nbr_imo = is_imo[b].astype(np.float64)
            np.add.at(imo_votes, a, nbr_imo)
            np.add.at(bg_votes, a, 1.0 - nbr_imo)

        # borderline below thresh but majority IMO neighbours -> IMO
        borderline_lo = valid & (~is_imo) & (rn > 0.7 * thresh)
        flip_up = borderline_lo & (imo_votes > bg_votes) & ((imo_votes + bg_votes) > 0)
        # just above thresh but majority BG neighbours -> background
        borderline_hi = valid & is_imo & (rn <= thresh / 0.7)
        flip_down = borderline_hi & (bg_votes > imo_votes) & ((imo_votes + bg_votes) > 0)

        is_imo = is_imo.copy()
        is_imo[flip_up] = True
        is_imo[flip_down] = False

    n_imo = int(is_imo.sum())
    n_bg = n - n_imo
    frac = 100.0 * n_imo / max(n, 1)
    print(f"[ego] residual split: thresh={thresh:.4g} px/s  "
          f"background={n_bg}  IMO={n_imo} ({frac:.1f}%)")
    if frac < 2.0 or frac > 40.0:
        print(f"[ego] *** WARNING: IMO fraction {frac:.1f}% outside expected "
              f"2–40% — check RES_K / flow quality ***")
    return is_imo, residual, thresh
