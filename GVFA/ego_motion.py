"""
ego_motion.py — Stage 2: 4-param similarity ego-motion fit + residual IMO split.

Fits a global camera motion model to smoothed flow via RANSAC+IRLS (default) or
plain IRLS, then splits events into background vs IMO candidates.
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
    A[0::2, 0] = 1.0
    A[0::2, 2] = -dy
    A[0::2, 3] = dx
    A[1::2, 1] = 1.0
    A[1::2, 2] = dx
    A[1::2, 3] = dy
    return A


def _mad(arr):
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _solve_params(x, y, vx, vy, cx, cy):
    """Minimal-sample 4-param solve. Returns params [4] or None."""
    n = len(x)
    if n < 4:
        return None
    A = _design_matrix(x, y, cx, cy)
    b = np.empty(2 * n, dtype=np.float64)
    b[0::2] = vx
    b[1::2] = vy
    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        return params
    except np.linalg.LinAlgError:
        return None


def _residual_norms(x, y, vx, vy, params, cx, cy):
    pvx, pvy = ego_field(x, y, params, cx, cy)
    return np.hypot(vx - pvx, vy - pvy)


def fit_ego_motion(x, y, vx, vy, sensor, n_iters=10, valid_mask=None):
    """IRLS fit of 4-param similarity ego-motion to smoothed flow."""
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
            "cx": cx, "cy": cy, "fitter": "irls",
        }

    xv, yv = x[valid], y[valid]
    vxv, vyv = vx[valid], vy[valid]
    A = _design_matrix(xv, yv, cx, cy)
    b = np.empty(2 * n_valid, dtype=np.float64)
    b[0::2] = vxv
    b[1::2] = vyv

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    weights = np.ones(n_valid, dtype=np.float64)

    for _ in range(n_iters):
        w2 = np.repeat(weights, 2)
        Aw = A * w2[:, None]
        bw = b * w2
        params, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

        pvx, pvy = ego_field(xv, yv, params, cx, cy)
        rn = np.hypot(vxv - pvx, vyv - pvy)
        mad = _mad(rn)
        delta = 1.4826 * max(mad, 1e-9)
        weights = np.where(rn <= delta, 1.0, delta / np.maximum(rn, 1e-12))

    tx, ty, w, s = [float(p) for p in params]
    pvx_all, pvy_all = ego_field(x, y, (tx, ty, w, s), cx, cy)
    residual[:, 0] = vx - pvx_all
    residual[:, 1] = vy - pvy_all
    residual[~valid] = 0.0

    rn_all = np.hypot(residual[valid, 0], residual[valid, 1])
    mad = _mad(rn_all)
    delta = 1.4826 * max(mad, 1e-9)
    inlier = rn_all <= delta
    inlier_rms = (
        float(np.sqrt(np.mean(rn_all[inlier] ** 2))) if inlier.any()
        else float(np.sqrt(np.mean(rn_all ** 2))) if rn_all.size else 0.0
    )

    print(f"[ego] IRLS params tx={tx:.4g} ty={ty:.4g} w={w:.4g} s={s:.4g}  "
          f"inlier_RMS={inlier_rms:.4g}  MAD={mad:.4g}  n_valid={n_valid}")
    return (tx, ty, w, s), residual, {
        "inlier_rms": inlier_rms,
        "mad": mad,
        "n_valid": n_valid,
        "cx": cx,
        "cy": cy,
        "delta": delta,
        "fitter": "irls",
        "ransac_inlier_mask": valid.copy(),
    }


def fit_ego_motion_ransac(x, y, vx, vy, sensor, *, n_hypotheses=200,
                          sample_size=8, inlier_k=2.5, n_iters_polish=10,
                          valid_mask=None, seed=0):
    """RANSAC ego fit: largest consensus background motion, then IRLS polish.

    RANSAC recovers the background as the largest consensus motion, so a large
    moving person cannot bias the fit the way least squares can.
    """
    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    n = len(x)
    if valid_mask is None:
        valid_mask = np.hypot(vx, vy) > 1e-12
    valid = np.asarray(valid_mask, dtype=bool)
    idx_valid = np.where(valid)[0]
    n_valid = len(idx_valid)

    if n_valid < 4:
        print("[ego] WARNING: too few valid flow events for RANSAC ego fit")
        residual = np.zeros((n, 2), dtype=np.float64)
        return (0.0, 0.0, 0.0, 0.0), residual, {
            "inlier_rms": 0.0, "mad": 0.0, "n_valid": n_valid,
            "cx": cx, "cy": cy, "fitter": "ransac",
            "n_ransac_inliers": 0, "best_hypo_rms": 0.0,
        }

    rng = np.random.default_rng(seed)
    ss = min(int(sample_size), n_valid)
    best_count = -1
    best_rms = np.inf
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_hypo_rms = np.inf

    xv_all, yv_all = x[valid], y[valid]
    vxv_all, vyv_all = vx[valid], vy[valid]

    for _ in range(int(n_hypotheses)):
        sample_idx = rng.choice(idx_valid, size=ss, replace=False)
        params = _solve_params(
            x[sample_idx], y[sample_idx],
            vx[sample_idx], vy[sample_idx], cx, cy,
        )
        if params is None:
            continue
        rn = _residual_norms(xv_all, yv_all, vxv_all, vyv_all, params, cx, cy)
        mad = _mad(rn)
        delta = float(inlier_k) * 1.4826 * max(mad, 1e-9)
        inliers_local = rn <= delta
        count = int(inliers_local.sum())
        if count < 4:
            continue
        rms = float(np.sqrt(np.mean(rn[inliers_local] ** 2)))
        if count > best_count or (count == best_count and rms < best_rms):
            best_count = count
            best_rms = rms
            best_hypo_rms = rms
            best_inlier_mask = np.zeros(n, dtype=bool)
            best_inlier_mask[idx_valid[inliers_local]] = True

    if best_count < 0:
        print("[ego] RANSAC failed — falling back to plain IRLS")
        params, residual, info = fit_ego_motion(
            x, y, vx, vy, sensor, n_iters=n_iters_polish, valid_mask=valid)
        info["fitter"] = "ransac_fallback_irls"
        return params, residual, info

    print(f"[ego] RANSAC best hypo: inliers={best_count}/{n_valid}  "
          f"hypo_RMS={best_hypo_rms:.4g}")
    params, residual, info = fit_ego_motion(
        x, y, vx, vy, sensor, n_iters=n_iters_polish,
        valid_mask=best_inlier_mask,
    )
    tx, ty, w, s = params
    info.update({
        "fitter": "ransac",
        "n_ransac_inliers": int(best_inlier_mask.sum()),
        "best_hypo_rms": float(best_hypo_rms),
        "n_hypotheses": int(n_hypotheses),
        "ransac_inlier_mask": best_inlier_mask.copy(),
    })
    print(f"[ego] RANSAC polished tx={tx:.4g} ty={ty:.4g} w={w:.4g} s={s:.4g}  "
          f"inlier_RMS={info['inlier_rms']:.4g}  "
          f"RANSAC_inliers={info['n_ransac_inliers']}/{n_valid}")
    return params, residual, info


def _otsu_threshold(values, n_bins=128):
    """Otsu threshold on |r| values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[v > 1e-12]
    if v.size < 2:
        return float(np.max(v)) if v.size else 0.0
    vmax = float(np.percentile(v, 99.5))
    vmax = max(vmax, float(v.max()), 1e-6)
    hist, edges = np.histogram(v, bins=n_bins, range=(0.0, vmax))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(vmax * 0.5)
    prob = hist / total
    omega = np.cumsum(prob)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b))
    return float(centers[k])


def _valley_threshold(values, smooth=3, n_bins=128):
    """Deepest valley between the two largest histogram peaks."""
    v = np.asarray(values, dtype=np.float64)
    v = v[v > 1e-12]
    if v.size < 2:
        return float(np.max(v)) if v.size else 0.0
    vmax = float(np.percentile(v, 99.5))
    vmax = max(vmax, float(v.max()), 1e-6)
    hist, edges = np.histogram(v, bins=n_bins, range=(0.0, vmax))
    hist = hist.astype(np.float64)
    if smooth > 1:
        k = int(smooth)
        kernel = np.exp(-0.5 * (np.arange(k) - (k - 1) / 2) ** 2)
        kernel /= kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")
    if hist.max() <= 0:
        return float(vmax * 0.5)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return _otsu_threshold(v, n_bins=n_bins)
    peaks = sorted(peaks, key=lambda i: hist[i], reverse=True)[:2]
    p0, p1 = sorted(peaks)
    if p1 <= p0 + 1:
        return float(0.5 * (edges[p0] + edges[p0 + 1]))
    valley_slice = hist[p0:p1 + 1]
    j = int(np.argmin(valley_slice)) + p0
    return float(0.5 * (edges[j] + edges[j + 1]))


def compute_threshold_candidates(rn_valid, res_k=2.0, valley_smooth=3):
    """Return sigma / otsu / valley thresholds (auto modes clamped)."""
    mad = _mad(rn_valid)
    sigma = 1.4826 * max(mad, 1e-9)
    lo, hi = 0.5 * sigma, 4.0 * sigma
    thresh_sigma = float(res_k * sigma)
    thresh_otsu = float(np.clip(_otsu_threshold(rn_valid), lo, hi))
    thresh_valley = float(np.clip(_valley_threshold(rn_valid, smooth=valley_smooth), lo, hi))
    return {
        "sigma": thresh_sigma,
        "otsu": thresh_otsu,
        "valley": thresh_valley,
        "mad": mad,
        "robust_sigma": sigma,
    }


def residual_split(residual, edge_index_spatial, res_k=2.0, valid_mask=None,
                   *, thresh_mode="otsu", valley_smooth=3):
    """Split events into background vs IMO by residual magnitude + hysteresis."""
    n = residual.shape[0]
    rn = np.hypot(residual[:, 0], residual[:, 1])
    if valid_mask is None:
        valid_mask = rn > 1e-12
    valid = np.asarray(valid_mask, dtype=bool) & (rn > 1e-12)

    if not valid.any():
        print("[ego] WARNING: no valid residuals for split — all background")
        return np.zeros(n, dtype=bool), residual, 0.0, {
            "thresh_sigma": 0.0, "thresh_otsu": 0.0, "thresh_valley": 0.0,
            "thresh_mode": thresh_mode, "thresh_active": 0.0,
        }

    cands = compute_threshold_candidates(rn[valid], res_k=res_k, valley_smooth=valley_smooth)
    mode = str(thresh_mode).lower()
    if mode not in cands:
        mode = "otsu"
    thresh = float(cands[mode])

    print(f"[ego] threshold candidates: sigma={cands['sigma']:.4g}  "
          f"otsu={cands['otsu']:.4g}  valley={cands['valley']:.4g}  "
          f"active={mode} -> {thresh:.4g} px/s")

    is_imo = np.zeros(n, dtype=bool)
    is_imo[valid] = rn[valid] > thresh

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
    print(f"[ego] residual split: thresh={thresh:.4g} px/s ({mode})  "
          f"background={n_bg}  IMO={n_imo} ({frac:.1f}%)")
    if frac < 2.0 or frac > 40.0:
        print(f"[ego] *** WARNING: IMO fraction {frac:.1f}% outside expected "
              f"2–40% — check RES_K / flow quality ***")

    thresh_info = {
        "thresh_sigma": cands["sigma"],
        "thresh_otsu": cands["otsu"],
        "thresh_valley": cands["valley"],
        "thresh_mode": mode,
        "thresh_active": thresh,
    }
    return is_imo, residual, thresh, thresh_info


# ---------------------------------------------------------------------------
# Object motion models (similarity 4-param / affine 6-param)
# ---------------------------------------------------------------------------

def affine_field(x, y, params):
    """6-param affine flow: vx = p0 + p1*x + p2*y, vy = p3 + p4*x + p5*y."""
    p0, p1, p2, p3, p4, p5 = params
    vx = p0 + p1 * x + p2 * y
    vy = p3 + p4 * x + p5 * y
    return vx, vy


def _design_matrix_affine(x, y):
    """Stack 2N x 6 design matrix for linear affine flow."""
    n = len(x)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    A[0::2, 0] = 1.0
    A[0::2, 1] = x
    A[0::2, 2] = y
    A[1::2, 3] = 1.0
    A[1::2, 4] = x
    A[1::2, 5] = y
    return A


def predict_flow(x, y, models, sensor, model_kind="similarity"):
    """Predict flow for N events under K models.

    Parameters
    ----------
    x, y : [N]
    models : [K, P]
    sensor : (W, H)

    Returns
    -------
    vx_pred, vy_pred : [N, K]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    models = np.asarray(models, dtype=np.float64)
    K = models.shape[0]
    N = len(x)
    vx_pred = np.zeros((N, K), dtype=np.float64)
    vy_pred = np.zeros((N, K), dtype=np.float64)

    if K == 0:
        return vx_pred, vy_pred

    if model_kind == "similarity":
        W, H = sensor
        cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
        dx = x[:, None] - cx
        dy = y[:, None] - cy
        tx = models[:, 0]
        ty = models[:, 1]
        w = models[:, 2]
        s = models[:, 3]
        vx_pred = tx[None, :] - w[None, :] * dy + s[None, :] * dx
        vy_pred = ty[None, :] + w[None, :] * dx + s[None, :] * dy
    elif model_kind == "affine":
        p0 = models[:, 0]
        p1 = models[:, 1]
        p2 = models[:, 2]
        p3 = models[:, 3]
        p4 = models[:, 4]
        p5 = models[:, 5]
        vx_pred = p0[None, :] + p1[None, :] * x[:, None] + p2[None, :] * y[:, None]
        vy_pred = p3[None, :] + p4[None, :] * x[:, None] + p5[None, :] * y[:, None]
    else:
        raise ValueError(f"unknown model_kind={model_kind!r}")
    return vx_pred, vy_pred


def _fit_model_ls(x, y, vx, vy, sensor, model_kind, weights=None):
    """Weighted least-squares fit for one motion model."""
    n = len(x)
    min_pts = 4 if model_kind == "similarity" else 6
    if n < min_pts:
        return None

    if model_kind == "similarity":
        W, H = sensor
        cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
        A = _design_matrix(x, y, cx, cy)
    else:
        A = _design_matrix_affine(x, y)

    b = np.empty(2 * n, dtype=np.float64)
    b[0::2] = vx
    b[1::2] = vy

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w2 = np.repeat(np.sqrt(np.maximum(w, 0.0)), 2)
        A = A * w2[:, None]
        b = b * w2

    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        return params
    except np.linalg.LinAlgError:
        return None


def _fit_model_irls(x, y, vx, vy, sensor, model_kind, n_iters=5, weights=None):
    """IRLS fit for one motion model (Huber on flow residuals)."""
    params = _fit_model_ls(x, y, vx, vy, sensor, model_kind, weights=weights)
    if params is None:
        return None

    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    w_base = np.ones(len(x), dtype=np.float64)
    if weights is not None:
        w_base = np.asarray(weights, dtype=np.float64).copy()

    for _ in range(max(1, int(n_iters))):
        if model_kind == "similarity":
            pvx, pvy = ego_field(x, y, params, cx, cy)
        else:
            pvx, pvy = affine_field(x, y, params)
        rn = np.hypot(vx - pvx, vy - pvy)
        mad = _mad(rn)
        delta = 1.4826 * max(mad, 1e-9)
        huber = np.where(rn <= delta, 1.0, delta / np.maximum(rn, 1e-12))
        w = w_base * huber
        params = _fit_model_ls(x, y, vx, vy, sensor, model_kind, weights=w)
        if params is None:
            return None
    return params


def _residual_rms(x, y, vx, vy, params, sensor, model_kind, weights=None):
    """Membership-weighted RMS flow residual for a model."""
    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    if model_kind == "similarity":
        pvx, pvy = ego_field(x, y, params, cx, cy)
    else:
        pvx, pvy = affine_field(x, y, params)
    rn2 = (vx - pvx) ** 2 + (vy - pvy) ** 2
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        wt = w.sum()
        if wt <= 0:
            return float(np.sqrt(np.mean(rn2))) if rn2.size else 0.0
        return float(np.sqrt(np.sum(w * rn2) / wt))
    return float(np.sqrt(np.mean(rn2)))


def fit_object_models(x, y, vx, vy, sensor, *, max_models=6, model_kind="affine",
                      inlier_k=2.5, n_irls=5, min_inliers=20, seed=0):
    """Greedy sequential object-motion fit (hard assignment, one pass).

    Each iteration fits the largest remaining inlier set, removes them, repeats.
    Returns models [K, P] and hard labels 1..K (0 = unassigned).
    """
    _ = seed  # reserved for future stochastic variants
    n = len(x)
    remaining = np.ones(n, dtype=bool)
    models = []
    labels = np.zeros(n, dtype=np.int64)
    min_pts = 4 if model_kind == "similarity" else 6

    for _ in range(int(max_models)):
        if int(remaining.sum()) < max(min_pts, min_inliers):
            break
        idx = np.where(remaining)[0]
        params = _fit_model_irls(
            x[idx], y[idx], vx[idx], vy[idx], sensor, model_kind, n_iters=n_irls,
        )
        if params is None:
            break

        pvx, pvy = predict_flow(x, y, params[None, :], sensor, model_kind)
        rn = np.hypot(vx - pvx[:, 0], vy - pvy[:, 0])
        rn_rem = rn[remaining]
        mad = _mad(rn_rem)
        delta = float(inlier_k) * 1.4826 * max(mad, 1e-9)
        inliers = remaining & (rn <= delta)
        if int(inliers.sum()) < min_inliers:
            break

        models.append(params)
        labels[inliers] = len(models)
        remaining &= ~inliers

    if not models:
        return np.zeros((0, 6 if model_kind == "affine" else 4), dtype=np.float64), labels

    return np.stack(models, axis=0), labels
