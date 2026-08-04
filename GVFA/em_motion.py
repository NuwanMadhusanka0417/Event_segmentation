"""
em_motion.py — EM-style soft assignment + alternating motion-model refinement.

Implements Stoffregen et al. (ICCV 2019) Algorithm 1: E-step (soft memberships)
and M-step (membership-weighted least squares), with cluster death for surplus
models.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from ego_motion import (
    _design_matrix,
    _design_matrix_affine,
    _mad,
    _fit_model_ls,
    fit_object_models,
    predict_flow,
)


def _n_params(model_kind):
    return 4 if model_kind == "similarity" else 6


def _empty_models(model_kind):
    return np.zeros((0, _n_params(model_kind)), dtype=np.float64)


def _pad_models(models, n_clusters, sensor, model_kind, seed):
    """Pad model list to n_clusters by perturbing the largest-weight model."""
    K = models.shape[0]
    if K >= n_clusters:
        return models[:n_clusters].copy(), "sequential"

    rng = np.random.default_rng(seed)
    P = _n_params(model_kind)
    out = np.zeros((n_clusters, P), dtype=np.float64)
    out[:K] = models
    ref = models[0].copy()
    if K > 0:
        ref = models[0].copy()
    else:
        ref = np.zeros(P, dtype=np.float64)
        if model_kind == "similarity":
            ref[0] = ref[1] = 0.0
        else:
            ref[0] = ref[3] = 0.0

    scale = 0.05 * (np.abs(ref) + 1.0)
    for j in range(K, n_clusters):
        noise = rng.normal(0.0, 1.0, size=P) * scale
        out[j] = ref + noise
    return out, "sequential_padded"


def _init_sigmas(x, y, vx, vy, models, sensor, model_kind, sigma_init=None):
    """Per-cluster sigma from init residuals (1.4826 * MAD), or global fallback."""
    K = models.shape[0]
    if K == 0:
        return np.array([], dtype=np.float64)

    vx_p, vy_p = predict_flow(x, y, models, sensor, model_kind)
    res2 = (vx[:, None] - vx_p) ** 2 + (vy[:, None] - vy_p) ** 2
    rn = np.sqrt(res2)

    sigmas = np.zeros(K, dtype=np.float64)
    for j in range(K):
        mad = _mad(rn[:, j])
        sigmas[j] = 1.4826 * max(mad, 1e-9)

    if sigma_init is not None:
        sigmas = np.full(K, float(sigma_init), dtype=np.float64)
    return sigmas


def initialize_models(x, y, vx, vy, is_imo, sensor, n_clusters, *, seed,
                      method="sequential", model_kind="affine",
                      vsa_labels=None, sigma_init=None):
    """Warm-start motion models for EM.

    Parameters
    ----------
    method : "sequential" | "kmeans" | "vsa"
    vsa_labels : int [N_imo] cluster ids from VSA prototype assign (for method=vsa)

    Returns
    -------
    models : [K, P]
    init_method : str actually used
    sigmas : [K] initial per-cluster scales
    """
    mask = np.asarray(is_imo, dtype=bool)
    xi, yi = x[mask], y[mask]
    vxi, vyi = vx[mask], vy[mask]
    n_imo = len(xi)

    if n_imo == 0:
        return _empty_models(model_kind), method, np.array([])

    method = str(method).lower()
    P = _n_params(model_kind)

    if method == "kmeans":
        k = min(int(n_clusters), max(1, n_imo))
        km = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
        km.fit(np.stack([vxi, vyi], axis=1))
        centroids = km.cluster_centers_
        models = np.zeros((k, P), dtype=np.float64)
        for j in range(k):
            tx, ty = centroids[j]
            if model_kind == "similarity":
                models[j] = (tx, ty, 0.0, 0.0)
            else:
                models[j] = (tx, 0.0, 0.0, ty, 0.0, 0.0)
        init_method = "kmeans"
        if k < n_clusters:
            models, _ = _pad_models(models, n_clusters, sensor, model_kind, seed)
            init_method = "kmeans_padded"
        else:
            models = models[:n_clusters]
        sigmas = _init_sigmas(xi, yi, vxi, vyi, models, sensor, model_kind, sigma_init)
        return models, init_method, sigmas

    if method == "vsa":
        if vsa_labels is None:
            raise ValueError("vsa_labels required for method='vsa'")
        vsa_labels = np.asarray(vsa_labels, dtype=np.int64)
        unique = [u for u in np.unique(vsa_labels) if u >= 0]
        models_list = []
        for lid in unique:
            m = vsa_labels == lid
            if int(m.sum()) < (_n_params(model_kind)):
                continue
            w = np.ones(int(m.sum()), dtype=np.float64)
            params = _fit_model_ls(
                xi[m], yi[m], vxi[m], vyi[m], sensor, model_kind, weights=w,
            )
            if params is not None:
                models_list.append(params)
        if models_list:
            models = np.stack(models_list, axis=0)
        else:
            models = _empty_models(model_kind)
        init_method = "vsa"
        if models.shape[0] < n_clusters:
            models, pad_tag = _pad_models(models, n_clusters, sensor, model_kind, seed)
            init_method = f"vsa_{pad_tag}"
        else:
            models = models[:n_clusters]
        sigmas = _init_sigmas(xi, yi, vxi, vyi, models, sensor, model_kind, sigma_init)
        return models, init_method, sigmas

    # default: sequential greedy warm start
    models, _ = fit_object_models(
        xi, yi, vxi, vyi, sensor,
        max_models=n_clusters, model_kind=model_kind, seed=seed,
    )
    init_method = "sequential"
    if models.shape[0] < n_clusters:
        models, _ = _pad_models(models, n_clusters, sensor, model_kind, seed)
        init_method = "sequential_padded"
    sigmas = _init_sigmas(xi, yi, vxi, vyi, models, sensor, model_kind, sigma_init)
    return models, init_method, sigmas


def e_step(x, y, vx, vy, models, sensor, *, sigma, model_kind):
    """Soft memberships P [N, K] via Gaussian likelihood in log space."""
    models = np.asarray(models, dtype=np.float64)
    K = models.shape[0]
    N = len(x)
    if K == 0 or N == 0:
        return np.zeros((N, 0), dtype=np.float64), 0

    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)

    vx_p, vy_p = predict_flow(x, y, models, sensor, model_kind)
    res2 = (vx[:, None] - vx_p) ** 2 + (vy[:, None] - vy_p) ** 2
    log_L = -0.5 * res2 / (sigma[None, :] ** 2)

    log_norm = logsumexp(log_L, axis=1, keepdims=True)
    bad = ~np.isfinite(log_norm) | (log_norm <= -1e300)
    n_underflow = int(bad.sum())

    P = np.exp(log_L - log_norm)
    if n_underflow:
        P[bad] = 1.0 / K

    # rows that still don't sum to 1
    row_sum = P.sum(axis=1, keepdims=True)
    bad2 = row_sum <= 1e-12
    if bad2.any():
        P[bad2.ravel()] = 1.0 / K
        n_underflow += int(bad2.sum())

    return P, n_underflow


def m_step(x, y, vx, vy, P, sensor, *, model_kind, min_weight=50.0,
           sigma_min=20.0):
    """Membership-weighted refit; drop clusters below min_weight."""
    P = np.asarray(P, dtype=np.float64)
    N, K = P.shape
    if K == 0:
        return _empty_models(model_kind), np.array([]), np.array([])

    weights = P.sum(axis=0)
    live = weights >= float(min_weight)
    n_deaths = int((~live).sum())

    if not live.any():
        return _empty_models(model_kind), np.array([]), np.zeros(K)

    idx = np.where(live)[0]
    models = np.zeros((len(idx), _n_params(model_kind)), dtype=np.float64)
    sigmas = np.zeros(len(idx), dtype=np.float64)

    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)

    for out_j, j in enumerate(idx):
        w = P[:, j]
        params = _fit_model_ls(x, y, vx, vy, sensor, model_kind, weights=w)
        if params is None:
            params = np.zeros(_n_params(model_kind), dtype=np.float64)
        models[out_j] = params

        if model_kind == "similarity":
            from ego_motion import ego_field
            pvx, pvy = ego_field(x, y, params, cx, cy)
        else:
            from ego_motion import affine_field
            pvx, pvy = affine_field(x, y, params)
        rn2 = (vx - pvx) ** 2 + (vy - pvy) ** 2
        wt = w.sum()
        rms = float(np.sqrt(np.sum(w * rn2) / max(wt, 1e-12)))
        sigmas[out_j] = max(rms, float(sigma_min))

    return models, sigmas, weights


def fit_em(x, y, vx, vy, is_imo, sensor, *, n_clusters, n_iters, sigma_init,
           model_kind, tol, seed, min_weight=50.0, sigma_min=20.0,
           init_method="sequential", vsa_labels=None, param_eps=1e-4):
    """Alternating EM loop on IMO events.

    Returns
    -------
    P : [N_imo, K_live] final soft memberships (live clusters only)
    models : [K_live, P]
    info : dict with convergence diagnostics
    """
    mask = np.asarray(is_imo, dtype=bool)
    xi, yi = x[mask], y[mask]
    vxi, vyi = vx[mask], vy[mask]
    n_imo = len(xi)

    info = {
        "init_method": init_method,
        "n_clusters_requested": n_clusters,
        "n_imo": n_imo,
        "iterations": 0,
        "converged": False,
        "converge_reason": "",
        "n_underflow_total": 0,
        "deaths_per_iter": [],
        "mean_max_membership": [],
        "n_live_clusters": [],
        "cluster_weights": [],
        "mean_weighted_residual": [],
        "models_history": [],
    }

    if n_imo == 0:
        info["n_live"] = 0
        return np.zeros((0, 0)), _empty_models(model_kind), info

    models, used_init, sigmas = initialize_models(
        x, y, vx, vy, mask, sensor, n_clusters,
        seed=seed, method=init_method, model_kind=model_kind,
        vsa_labels=vsa_labels, sigma_init=sigma_init,
    )
    info["init_method"] = used_init

    P = np.zeros((n_imo, models.shape[0]), dtype=np.float64)
    if models.shape[0] > 0:
        P, n_u = e_step(xi, yi, vxi, vyi, models, sensor,
                        sigma=sigmas, model_kind=model_kind)
        info["n_underflow_total"] += n_u
        max_mem = P.max(axis=1)
        info["mean_max_membership"].append(float(max_mem.mean()))
        info["n_live_clusters"].append(int(models.shape[0]))
        info["cluster_weights"].append(P.sum(axis=0).tolist())
        vx_p, vy_p = predict_flow(xi, yi, models, sensor, model_kind)
        mwr0 = []
        for j in range(models.shape[0]):
            res2 = (vxi - vx_p[:, j]) ** 2 + (vyi - vy_p[:, j]) ** 2
            wt = P[:, j].sum()
            mwr0.append(float(np.sqrt(np.sum(P[:, j] * res2) / max(wt, 1e-12))))
        info["mean_weighted_residual"].append(mwr0)

    prev_P = P.copy()
    prev_models = models.copy()

    for it in range(int(n_iters)):
        info["iterations"] = it + 1

        # M-step
        models, sigmas, cluster_w = m_step(
            xi, yi, vxi, vyi, P, sensor,
            model_kind=model_kind, min_weight=min_weight, sigma_min=sigma_min,
        )
        deaths = max(0, P.shape[1] - models.shape[0])
        info["deaths_per_iter"].append(deaths)

        if models.shape[0] == 0:
            info["converged"] = True
            info["converge_reason"] = "all_clusters_dead"
            P = np.zeros((n_imo, 0))
            break

        # remap P to live clusters if some died
        if P.shape[1] != models.shape[0]:
            live = cluster_w >= float(min_weight)
            P = P[:, live]
            row_sum = P.sum(axis=1, keepdims=True)
            nz = row_sum.ravel() > 1e-12
            P[nz] /= row_sum[nz]

        # E-step
        P, n_u = e_step(xi, yi, vxi, vyi, models, sensor,
                        sigma=sigmas, model_kind=model_kind)
        info["n_underflow_total"] += n_u

        max_mem = P.max(axis=1) if P.size else np.array([])
        info["mean_max_membership"].append(
            float(max_mem.mean()) if max_mem.size else 0.0,
        )
        info["n_live_clusters"].append(int(models.shape[0]))
        info["cluster_weights"].append(cluster_w[cluster_w >= min_weight].tolist()
                                      if cluster_w.size else [])
        mwr = []
        vx_p, vy_p = predict_flow(xi, yi, models, sensor, model_kind)
        for j in range(models.shape[0]):
            res2 = (vxi - vx_p[:, j]) ** 2 + (vyi - vy_p[:, j]) ** 2
            wt = P[:, j].sum()
            mwr.append(float(np.sqrt(np.sum(P[:, j] * res2) / max(wt, 1e-12))))
        info["mean_weighted_residual"].append(mwr)

        # convergence checks
        dP = float(np.mean(np.abs(P - prev_P))) if prev_P.shape == P.shape else np.inf
        d_model = 0.0
        if prev_models.shape == models.shape and models.size:
            d_model = float(np.max(np.abs(models - prev_models)))

        if dP < float(tol):
            info["converged"] = True
            info["converge_reason"] = f"mean_|dP|={dP:.2e}<{tol}"
            break
        if d_model < float(param_eps) and it > 0:
            info["converged"] = True
            info["converge_reason"] = f"max_|dmodel|={d_model:.2e}<{param_eps}"
            break

        prev_P = P.copy()
        prev_models = models.copy()

    if not info["converged"]:
        info["converge_reason"] = f"max_iters={n_iters}"

    info["n_live"] = int(models.shape[0])
    info["sigmas"] = sigmas.tolist() if sigmas.size else []
    if P.size:
        info["final_mean_max_membership"] = float(P.max(axis=1).mean())
        info["pct_above_0.9"] = float(100.0 * (P.max(axis=1) > 0.9).mean())
        info["pct_below_0.5"] = float(100.0 * (P.max(axis=1) < 0.5).mean())
        row_sums = P.sum(axis=1)
        bad_rows = np.abs(row_sums - 1.0) > 1e-6
        if bad_rows.any():
            print(f"[em] WARNING: {int(bad_rows.sum())} rows of P do not sum to 1")
        if not np.isfinite(P).all():
            print("[em] WARNING: NaN/Inf detected in P")
    else:
        info["final_mean_max_membership"] = 0.0
        info["pct_above_0.9"] = 0.0
        info["pct_below_0.5"] = 0.0

    return P, models, info


def em_hard_labels(P):
    """Argmax soft memberships -> labels 1..K (0 if no cluster)."""
    if P.size == 0:
        return np.zeros(P.shape[0], dtype=np.int64)
    best = np.argmax(P, axis=1)
    labels = best + 1
    weak = P.max(axis=1) <= 1e-12
    labels[weak] = 0
    return labels


def em_confidence(P):
    """Per-event max membership."""
    if P.size == 0:
        return np.zeros(P.shape[0], dtype=np.float64)
    return P.max(axis=1)
