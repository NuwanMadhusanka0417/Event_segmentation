"""
ego_motion.py — Stage 2: ego-motion fit + residual IMO split + multi-model objects.

Global ego uses 4-param similarity (RANSAC consensus + IRLS polish) so a large
moving person cannot bias the background model. Per-object fits use 6-param
affine flow (optional similarity) with post-fit merge of near-duplicate models.
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


def affine_field(x, y, params6, cx, cy):
    """6-param affine flow at (x, y).

    v = [a0 + a1*(x-cx) + a2*(y-cy),
         a3 + a4*(x-cx) + a5*(y-cy)]
    """
    a0, a1, a2, a3, a4, a5 = params6
    dx = x - cx
    dy = y - cy
    vx = a0 + a1 * dx + a2 * dy
    vy = a3 + a4 * dx + a5 * dy
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


def _design_matrix_affine(x, y, cx, cy):
    """Stack 2N x 6 design matrix for linear affine flow."""
    dx = x - cx
    dy = y - cy
    n = len(x)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    A[0::2, 0] = 1.0
    A[0::2, 1] = dx
    A[0::2, 2] = dy
    A[1::2, 3] = 1.0
    A[1::2, 4] = dx
    A[1::2, 5] = dy
    return A


def _mad(arr):
    """Median absolute deviation (scalar)."""
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _predict_field(x, y, params, cx, cy, model_kind="similarity"):
    if model_kind == "affine":
        return affine_field(x, y, params, cx, cy)
    return ego_field(x, y, params, cx, cy)


def _design(x, y, cx, cy, model_kind="similarity"):
    if model_kind == "affine":
        return _design_matrix_affine(x, y, cx, cy)
    return _design_matrix(x, y, cx, cy)


def fit_ego_motion(x, y, vx, vy, sensor, n_iters=10, valid_mask=None,
                   *, model_kind="similarity"):
    """IRLS fit of similarity (4-param) or affine (6-param) flow.

    model_kind : "similarity" | "affine"  (keyword-only; default preserves old API)
    """
    if model_kind not in ("similarity", "affine"):
        raise ValueError(f"unknown model_kind={model_kind!r}")

    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    n = len(x)
    n_params = 6 if model_kind == "affine" else 4
    zero_params = tuple(0.0 for _ in range(n_params))

    if valid_mask is None:
        valid_mask = (np.hypot(vx, vy) > 1e-12)
    valid = np.asarray(valid_mask, dtype=bool)
    n_valid = int(valid.sum())

    residual = np.zeros((n, 2), dtype=np.float64)
    if n_valid < n_params:
        print(f"[ego] WARNING: too few valid flow events for {model_kind} fit")
        return zero_params, residual, {
            "inlier_rms": 0.0, "mad": 0.0, "n_valid": n_valid,
            "cx": cx, "cy": cy, "model_kind": model_kind,
        }

    xv, yv = x[valid], y[valid]
    vxv, vyv = vx[valid], vy[valid]
    A = _design(xv, yv, cx, cy, model_kind)
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

        pvx, pvy = _predict_field(xv, yv, params, cx, cy, model_kind)
        rn = np.hypot(vxv - pvx, vyv - pvy)
        mad = _mad(rn)
        delta = 1.4826 * max(mad, 1e-9)
        weights = np.where(rn <= delta, 1.0, delta / np.maximum(rn, 1e-12))

    params = tuple(float(p) for p in params)
    pvx_all, pvy_all = _predict_field(x, y, params, cx, cy, model_kind)
    residual[:, 0] = vx - pvx_all
    residual[:, 1] = vy - pvy_all
    residual[~valid] = 0.0

    rn_all = np.hypot(residual[valid, 0], residual[valid, 1])
    mad = _mad(rn_all)
    delta = 1.4826 * max(mad, 1e-9)
    inlier = rn_all <= delta
    if inlier.any():
        inlier_rms = float(np.sqrt(np.mean(rn_all[inlier] ** 2)))
    else:
        inlier_rms = float(np.sqrt(np.mean(rn_all ** 2))) if rn_all.size else 0.0

    if model_kind == "similarity":
        tx, ty, w, s = params
        print(f"[ego] params tx={tx:.4g} ty={ty:.4g} w={w:.4g} s={s:.4g}  "
              f"inlier_RMS={inlier_rms:.4g}  MAD={mad:.4g}  n_valid={n_valid}")
    else:
        print(f"[ego] affine params={[f'{p:.4g}' for p in params]}  "
              f"inlier_RMS={inlier_rms:.4g}  MAD={mad:.4g}  n_valid={n_valid}")

    return params, residual, {
        "inlier_rms": inlier_rms,
        "mad": mad,
        "n_valid": n_valid,
        "cx": cx,
        "cy": cy,
        "delta": delta,
        "model_kind": model_kind,
    }


def fit_ego_motion_ransac(
    x, y, vx, vy, sensor, *,
    n_hypotheses=200,
    sample_size=8,
    inlier_k=2.5,
    n_iters_polish=10,
    valid_mask=None,
    seed=0,
):
    """RANSAC consensus on 4-param similarity, then IRLS polish on inliers.

    RANSAC recovers the background as the LARGEST consensus motion, so a big
    moving person no longer biases the ego fit (Huber-IRLS alone fails when a
    coherent outlier set is numerous).
    """
    W, H = sensor
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    n = len(x)
    if valid_mask is None:
        valid_mask = (np.hypot(vx, vy) > 1e-12)
    valid = np.asarray(valid_mask, dtype=bool)
    valid_idx = np.where(valid)[0]
    n_valid = int(valid_idx.size)

    if n_valid < sample_size:
        print("[ego-ransac] WARNING: too few valid events — falling back to IRLS")
        return fit_ego_motion(x, y, vx, vy, sensor, n_iters=n_iters_polish,
                              valid_mask=valid)

    rng = np.random.default_rng(seed)
    best_count = -1
    best_rms = np.inf
    best_inliers_valid = None

    xv_all, yv_all = x[valid_idx], y[valid_idx]
    vxv_all, vyv_all = vx[valid_idx], vy[valid_idx]

    for _ in range(int(n_hypotheses)):
        samp = rng.choice(n_valid, size=sample_size, replace=False)
        A = _design_matrix(xv_all[samp], yv_all[samp], cx, cy)
        b = np.empty(2 * sample_size, dtype=np.float64)
        b[0::2] = vxv_all[samp]
        b[1::2] = vyv_all[samp]
        try:
            params, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(params)):
            continue
        params = tuple(float(p) for p in params)
        pvx, pvy = ego_field(xv_all, yv_all, params, cx, cy)
        rn = np.hypot(vxv_all - pvx, vyv_all - pvy)
        mad = _mad(rn)
        sigma = 1.4826 * max(mad, 1e-9)
        inl = rn <= (inlier_k * sigma)
        count = int(inl.sum())
        if count == 0:
            continue
        rms = float(np.sqrt(np.mean(rn[inl] ** 2)))
        if count > best_count or (count == best_count and rms < best_rms):
            best_count = count
            best_rms = rms
            best_inliers_valid = inl

    if best_inliers_valid is None or best_count < 4:
        print("[ego-ransac] WARNING: no consensus — falling back to plain IRLS")
        return fit_ego_motion(x, y, vx, vy, sensor, n_iters=n_iters_polish,
                              valid_mask=valid)

    polish_mask = np.zeros(n, dtype=bool)
    polish_mask[valid_idx[best_inliers_valid]] = True

    print(f"[ego-ransac] hypotheses={n_hypotheses}  best inliers="
          f"{best_count}/{n_valid} ({100.0 * best_count / max(n_valid, 1):.1f}%)  "
          f"hypo_RMS={best_rms:.4g}")

    params, _res_pol, info = fit_ego_motion(
        x, y, vx, vy, sensor,
        n_iters=n_iters_polish,
        valid_mask=polish_mask,
    )
    # residuals for ALL events under polished model
    pvx, pvy = ego_field(x, y, params, cx, cy)
    residual = np.zeros((n, 2), dtype=np.float64)
    residual[:, 0] = vx - pvx
    residual[:, 1] = vy - pvy
    residual[~valid] = 0.0

    rn_valid = np.hypot(residual[valid, 0], residual[valid, 1])
    mad = _mad(rn_valid)
    delta = 1.4826 * max(mad, 1e-9)
    final_inl = np.zeros(n, dtype=bool)
    final_inl[valid] = rn_valid <= delta

    info = dict(info)
    info.update({
        "n_ransac_inliers": int(best_count),
        "best_hypo_rms": float(best_rms),
        "ransac": True,
        "n_hypotheses": int(n_hypotheses),
        "ransac_sample": int(sample_size),
        "ransac_inlier_k": float(inlier_k),
        "ransac_inlier_mask": polish_mask.copy(),
        "n_valid": n_valid,
        "mad": mad,
        "delta": delta,
    })
    if final_inl.any():
        info["inlier_rms"] = float(np.sqrt(np.mean(
            np.hypot(residual[final_inl, 0], residual[final_inl, 1]) ** 2)))

    tx, ty, w, s = params
    print(f"[ego-ransac] polished tx={tx:.4g} ty={ty:.4g} w={w:.4g} s={s:.4g}  "
          f"inlier_RMS={info['inlier_rms']:.4g}")
    return params, residual, info


def residual_split(residual, edge_index_spatial, res_k=2.0, valid_mask=None):
    """Split events into background vs IMO by residual magnitude + hysteresis."""
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

    return is_imo, {"added_per_iter": added_per_iter, "n_added": int(sum(added_per_iter))}


def erode_imo_mask(is_imo, edge_index_list, frac=0.5):
    is_imo = np.asarray(is_imo, dtype=bool).copy()
    if not isinstance(edge_index_list, (list, tuple)):
        edge_index_list = [edge_index_list]

    imo_votes = np.zeros(len(is_imo), dtype=np.float64)
    tot_votes = np.zeros(len(is_imo), dtype=np.float64)
    for ei in edge_index_list:
        iv, tv = _undirected_neighbour_counts(ei, is_imo)
        imo_votes += iv
        tot_votes += tv

    isolated = is_imo & (tot_votes > 0) & (imo_votes < frac * tot_votes)
    isolated |= is_imo & (tot_votes == 0)
    n_removed = int(isolated.sum())
    is_imo[isolated] = False
    print(f"[ego] erode: -{n_removed} isolated IMO events -> background")
    return is_imo, {"n_removed": n_removed}


def refine_imo_mask(is_imo, edge_index_list, n_dilate=2, frac=0.5):
    is_imo, dinfo = dilate_imo_mask(is_imo, edge_index_list, n_iters=n_dilate, frac=frac)
    is_imo, einfo = erode_imo_mask(is_imo, edge_index_list, frac=frac)
    n = len(is_imo)
    n_imo = int(is_imo.sum())
    frac_pct = 100.0 * n_imo / max(n, 1)
    print(f"[ego] after dilate/erode: IMO={n_imo}/{n} ({frac_pct:.1f}%)")
    if frac_pct < 2.0 or frac_pct > 40.0:
        print(f"[ego] *** WARNING: final IMO fraction {frac_pct:.1f}% outside "
              f"expected 2–40% ***")
    return is_imo, {
        **dinfo, **einfo,
        "n_imo": n_imo, "imo_frac_pct": frac_pct,
        "n_dilate": n_dilate, "frac": frac,
    }


def _model_mean_flow(model, gx, gy, cx, cy):
    kind = model.get("kind", "similarity")
    return _predict_field(gx, gy, model["params"], cx, cy, kind)


def merge_similar_models(models, model_id, x, y, sensor, *,
                         merge_cos=0.9, merge_speed_ratio=0.5):
    """Merge near-duplicate object models (0-MMS-style; Parameshwara ICRA 2021)."""
    n_pre = len(models)
    if n_pre <= 1:
        print(f"[models] merge: pre={n_pre} — nothing to merge")
        return model_id, models, {"n_pre": n_pre, "n_post": n_pre, "n_merges": 0,
                                  "merge_cos": merge_cos,
                                  "merge_speed_ratio": merge_speed_ratio}

    W, H = sensor
    cx = models[0].get("cx", 0.5 * (W - 1))
    cy = models[0].get("cy", 0.5 * (H - 1))
    gx = np.linspace(0, W - 1, 12)
    gy = np.linspace(0, H - 1, 10)
    GX, GY = np.meshgrid(gx, gy)
    gxf, gyf = GX.ravel(), GY.ravel()

    flows, speeds = [], []
    for m in models:
        pvx, pvy = _model_mean_flow(m, gxf, gyf, cx, cy)
        flows.append(np.stack([pvx, pvy], axis=1))
        speeds.append(float(np.mean(np.hypot(pvx, pvy))))

    parent = np.arange(n_pre, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n_merges = 0
    for i in range(n_pre):
        for j in range(i + 1, n_pre):
            fi, fj = flows[i], flows[j]
            ni = np.linalg.norm(fi, axis=1)
            nj = np.linalg.norm(fj, axis=1)
            ok = (ni > 1e-9) & (nj > 1e-9)
            if not ok.any():
                continue
            cos = np.sum(fi[ok] * fj[ok], axis=1) / (ni[ok] * nj[ok])
            mean_cos = float(np.mean(cos))
            si, sj = speeds[i], speeds[j]
            smax = max(si, sj, 1e-9)
            speed_ok = (smax - min(si, sj)) / smax <= merge_speed_ratio
            if mean_cos > merge_cos and speed_ok:
                print(f"[models] merge m{models[i]['id']}+m{models[j]['id']}: "
                      f"cos={mean_cos:.3f}")
                union(i, j)
                n_merges += 1

    roots = np.array([find(i) for i in range(n_pre)])
    _, inv = np.unique(roots, return_inverse=True)
    n_post = int(inv.max()) + 1

    old_to_new = {models[i]["id"]: int(inv[i]) for i in range(n_pre)}
    new_model_id = np.full_like(model_id, -1)
    for old, new in old_to_new.items():
        new_model_id[model_id == old] = new

    merged_models = []
    for new_id in range(n_post):
        members = [models[i] for i in range(n_pre) if inv[i] == new_id]
        rep = max(members, key=lambda m: m["n_inliers"])
        mask = new_model_id == new_id
        entry = dict(rep)
        entry["id"] = new_id
        entry["n_inliers"] = int(mask.sum())
        entry["merged_from"] = [m["id"] for m in members]
        if entry.get("kind", "similarity") == "affine":
            p = entry["params"]
            entry["tx"], entry["ty"] = float(p[0]), float(p[3])
            entry["w"], entry["s"] = float(p[1]), float(p[2])
        merged_models.append(entry)

    print(f"[models] merge: pre={n_pre}  pair_merges={n_merges}  post={n_post}")
    return new_model_id, merged_models, {
        "n_pre": n_pre, "n_post": n_post, "n_merges": n_merges,
        "merge_cos": merge_cos, "merge_speed_ratio": merge_speed_ratio,
    }


def fit_object_models(
    x, y, vx, vy, is_imo,
    sensor,
    max_models=4,
    min_inliers=600,
    res_k=2.5,
    n_iters=10,
    *,
    model_kind="affine",
    merge_cos=0.9,
    merge_speed_ratio=0.5,
):
    """Recursive multi-model fitting on IMO (affine by default) + merge."""
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
            model_kind=model_kind,
        )
        pvx, pvy = _predict_field(x, y, params, cx, cy, model_kind)
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
        entry = {
            "id": m, "params": params, "kind": model_kind,
            "n_inliers": n_in, "rms": rms, "thresh": thresh,
            "cx": cx, "cy": cy,
        }
        if model_kind == "similarity":
            tx, ty, w, s = params
            entry.update({"tx": tx, "ty": ty, "w": w, "s": s})
            print(f"[models] model {m} (similarity): #inliers={n_in}  "
                  f"tx={tx:.4g} ty={ty:.4g}  RMS={rms:.4g}")
        else:
            a0, a1, a2, a3, a4, a5 = params
            entry.update({"tx": a0, "ty": a3, "w": a1, "s": a2,
                          "a4": a4, "a5": a5})
            print(f"[models] model {m} (affine): #inliers={n_in}  "
                  f"a0={a0:.4g} a3={a3:.4g}  RMS={rms:.4g}")
        models.append(entry)

    print(f"[models] fitted {len(models)} models (pre-merge); assigned "
          f"{int((model_id >= 0).sum())} / {int(is_imo.sum())} IMO")

    model_id, models, merge_info = merge_similar_models(
        models, model_id, x, y, sensor,
        merge_cos=merge_cos, merge_speed_ratio=merge_speed_ratio,
    )

    # drop merged models still below min_inliers
    keep, new_id_map, next_id = [], {}, 0
    for m in models:
        if m["n_inliers"] < min_inliers:
            print(f"[models] drop merged m{m['id']}: n={m['n_inliers']} "
                  f"< {min_inliers}")
            model_id[model_id == m["id"]] = -1
            continue
        new_id_map[m["id"]] = next_id
        m = dict(m)
        m["id"] = next_id
        keep.append(m)
        next_id += 1
    if new_id_map:
        remapped = np.full_like(model_id, -1)
        for old, new in new_id_map.items():
            remapped[model_id == old] = new
        model_id = remapped
    models = keep
    merge_info["n_post"] = len(models)
    print(f"[models] final {len(models)} models after merge/filter")
    return model_id, models, merge_info
