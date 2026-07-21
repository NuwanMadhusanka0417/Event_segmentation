"""
ego_motion.py — Stage 2: 4-param similarity ego-motion fit + residual IMO split.

Fits a global camera motion model to smoothed normal flow via IRLS, then splits
events into background (residual-consistent) vs IMO candidates (outliers).
Residual velocity is also used for multi-model object fitting and (downstream)
enters the node hypervector as (|r|, angle(r)).
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


def residual_split(residual, edge_index_spatial, res_k=2.0, valid_mask=None):
    """Split events into background vs IMO by residual magnitude + hysteresis.

    thresh = RES_K * (1.4826 * MAD(|r| over valid))
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
        imo_votes = np.zeros(n, dtype=np.float64)
        bg_votes = np.zeros(n, dtype=np.float64)
        for a, b in ((rec, src), (src, rec)):
            nbr_imo = is_imo[b].astype(np.float64)
            np.add.at(imo_votes, a, nbr_imo)
            np.add.at(bg_votes, a, 1.0 - nbr_imo)

        borderline_lo = valid & (~is_imo) & (rn > 0.7 * thresh)
        flip_up = borderline_lo & (imo_votes > bg_votes) & ((imo_votes + bg_votes) > 0)
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


def _undirected_neighbour_counts(edge_index, is_imo):
    """Return (imo_nbr_count, total_nbr_count) via np.add.at over undirected edges."""
    n = len(is_imo)
    imo_votes = np.zeros(n, dtype=np.float64)
    tot_votes = np.zeros(n, dtype=np.float64)
    if edge_index is None or edge_index.numel() == 0:
        return imo_votes, tot_votes
    rec = edge_index[0].numpy().astype(np.int64)
    src = edge_index[1].numpy().astype(np.int64)
    for a, b in ((rec, src), (src, rec)):
        np.add.at(imo_votes, a, is_imo[b].astype(np.float64))
        np.add.at(tot_votes, a, 1.0)
    return imo_votes, tot_votes


def dilate_imo_mask(is_imo, edge_index_list, n_iters=2, frac=0.5):
    """Grow IMO into background events with enough IMO neighbours (aperture fix).

    For each iteration, flip BACKGROUND -> IMO when >= frac of neighbours are IMO.
    ``edge_index_list`` may be a single [2,E] tensor or a list of them.
    Returns (is_imo_new, info) with per-iteration added counts.
    """
    is_imo = np.asarray(is_imo, dtype=bool).copy()
    if not isinstance(edge_index_list, (list, tuple)):
        edge_index_list = [edge_index_list]

    added_per_iter = []
    for it in range(int(n_iters)):
        imo_votes = np.zeros(len(is_imo), dtype=np.float64)
        tot_votes = np.zeros(len(is_imo), dtype=np.float64)
        for ei in edge_index_list:
            iv, tv = _undirected_neighbour_counts(ei, is_imo)
            imo_votes += iv
            tot_votes += tv
        bg = ~is_imo
        eligible = bg & (tot_votes > 0) & (imo_votes >= frac * tot_votes)
        n_add = int(eligible.sum())
        is_imo[eligible] = True
        added_per_iter.append(n_add)
        print(f"[ego] dilate iter {it + 1}/{n_iters}: +{n_add} events -> IMO")

    info = {"added_per_iter": added_per_iter, "n_added": int(sum(added_per_iter))}
    return is_imo, info


def erode_imo_mask(is_imo, edge_index_list, frac=0.5):
    """Revert isolated IMO events with < frac IMO neighbours back to background."""
    is_imo = np.asarray(is_imo, dtype=bool).copy()
    if not isinstance(edge_index_list, (list, tuple)):
        edge_index_list = [edge_index_list]

    imo_votes = np.zeros(len(is_imo), dtype=np.float64)
    tot_votes = np.zeros(len(is_imo), dtype=np.float64)
    for ei in edge_index_list:
        iv, tv = _undirected_neighbour_counts(ei, is_imo)
        imo_votes += iv
        tot_votes += tv

    # isolated: currently IMO, has neighbours, but IMO fraction below frac
    isolated = is_imo & (tot_votes > 0) & (imo_votes < frac * tot_votes)
    # also strip IMO with zero neighbours (salt)
    isolated |= is_imo & (tot_votes == 0)
    n_removed = int(isolated.sum())
    is_imo[isolated] = False
    print(f"[ego] erode: -{n_removed} isolated IMO events -> background")
    return is_imo, {"n_removed": n_removed}


def refine_imo_mask(is_imo, edge_index_list, n_dilate=2, frac=0.5):
    """Dilate then erode; report final IMO fraction with loud warning if out of band."""
    is_imo, dinfo = dilate_imo_mask(is_imo, edge_index_list, n_iters=n_dilate, frac=frac)
    is_imo, einfo = erode_imo_mask(is_imo, edge_index_list, frac=frac)
    n = len(is_imo)
    n_imo = int(is_imo.sum())
    frac_pct = 100.0 * n_imo / max(n, 1)
    print(f"[ego] after dilate/erode: IMO={n_imo}/{n} ({frac_pct:.1f}%)")
    if frac_pct < 2.0 or frac_pct > 40.0:
        print(f"[ego] *** WARNING: final IMO fraction {frac_pct:.1f}% outside "
              f"expected 2–40% ***")
    info = {
        **dinfo,
        **einfo,
        "n_imo": n_imo,
        "imo_frac_pct": frac_pct,
        "n_dilate": n_dilate,
        "frac": frac,
    }
    return is_imo, info


def fit_object_models(
    x, y, vx, vy, is_imo,
    sensor,
    max_models=4,
    min_inliers=300,
    res_k=2.5,
    n_iters=10,
):
    """Recursive multi-model similarity fitting on IMO events only.

    Loop up to max_models: fit 4-param IRLS on unassigned IMO, claim inliers
    with |v - v_pred| <= res_k * robust_sigma, remove, continue.
    Stops when inlier count < min_inliers (upper bound, not a fixed object count).

    Returns
    -------
    model_id : int [N]  (-1 = background / unassigned)
    models : list of dicts with params, n_inliers, rms, ...
    """
    n = len(x)
    model_id = np.full(n, -1, dtype=np.int64)
    is_imo = np.asarray(is_imo, dtype=bool)
    unassigned = is_imo.copy()
    models = []

    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)

    for m in range(int(max_models)):
        n_pool = int(unassigned.sum())
        if n_pool < min_inliers:
            print(f"[models] stop: unassigned IMO={n_pool} < min_inliers={min_inliers}")
            break

        params, _res, info = fit_ego_motion(
            x, y, vx, vy, sensor, n_iters=n_iters, valid_mask=unassigned,
        )
        pvx, pvy = ego_field(x, y, params, cx, cy)
        rn = np.hypot(vx - pvx, vy - pvy)
        pool_rn = rn[unassigned]
        if pool_rn.size == 0:
            break
        mad = _mad(pool_rn)
        sigma = 1.4826 * max(mad, 1e-9)
        thresh = float(res_k * sigma)
        inliers = unassigned & (rn <= thresh)
        n_in = int(inliers.sum())
        if n_in < min_inliers:
            print(f"[models] stop model {m}: inliers={n_in} < {min_inliers} "
                  f"(thresh={thresh:.4g})")
            break

        model_id[inliers] = m
        unassigned[inliers] = False
        rms = float(np.sqrt(np.mean(rn[inliers] ** 2)))
        tx, ty, w, s = params
        models.append({
            "id": m,
            "params": params,
            "tx": tx, "ty": ty, "w": w, "s": s,
            "n_inliers": n_in,
            "rms": rms,
            "thresh": thresh,
            "cx": cx, "cy": cy,
        })
        print(f"[models] model {m}: #inliers={n_in}  "
              f"tx={tx:.4g} ty={ty:.4g} w={w:.4g} s={s:.4g}  RMS={rms:.4g}")

    n_assigned = int((model_id >= 0).sum())
    print(f"[models] fitted {len(models)} models; assigned {n_assigned} / "
          f"{int(is_imo.sum())} IMO events")
    return model_id, models
