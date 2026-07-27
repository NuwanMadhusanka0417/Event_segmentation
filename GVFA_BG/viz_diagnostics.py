"""
viz_diagnostics.py — Stage 1–3 diagnostic figures with parameter annotation boxes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _ensure_dir(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def _annotate(ax, lines, loc="upper right"):
    """Semi-transparent monospace parameter/result box."""
    from matplotlib.offsetbox import AnchoredText
    text = "\n".join(lines)
    at = AnchoredText(
        text, loc=loc, prop=dict(family="monospace", size=8),
        frameon=True, borderpad=0.4,
    )
    at.patch.set_boxstyle("round,pad=0.3")
    at.patch.set_alpha(0.85)
    at.patch.set_facecolor("white")
    ax.add_artist(at)


def _subsample_idx(n, k=2000, seed=0):
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def _scatter_events(ax, x, y, c="0.7", s=1, **kwargs):
    ax.scatter(x, y, c=c, s=s, linewidths=0, **kwargs)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


def plot_flow_raw(x, y, vx, vy, out_dir, n_sample=2000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "01_flow_raw.png")
    spd = np.hypot(vx, vy)
    valid = spd > 1e-12
    frac = 100.0 * valid.mean()
    med = float(np.median(spd[valid])) if valid.any() else 0.0

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_events(ax, x, y, c="0.85", s=1)
    idx = _subsample_idx(len(x), n_sample, seed)
    idx = idx[valid[idx]]
    if idx.size:
        ax.quiver(x[idx], y[idx], vx[idx], -vy[idx],  # -vy: screen y inverted
                  angles="xy", scale_units="xy", scale=None,
                  width=0.0015, color="crimson", alpha=0.7)
    ax.set_title("Stage 1 input: raw normal flow", fontweight="bold")
    _annotate(ax, [
        "n_iters: - (raw)",
        f"valid: {frac:.1f}%",
        f"median |v|: {med:.4g} px/s",
        f"N: {len(x)}",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_flow_smoothed(x, y, vx_s, vy_s, out_dir, n_iters, keep,
                       n_valid_before, n_valid_after, n_sample=2000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "02_flow_smoothed.png")
    spd = np.hypot(vx_s, vy_s)
    valid = spd > 1e-12
    frac = 100.0 * valid.mean()
    med = float(np.median(spd[valid])) if valid.any() else 0.0
    n = len(x)

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_events(ax, x, y, c="0.85", s=1)
    idx = _subsample_idx(n, n_sample, seed)
    idx = idx[valid[idx]]
    if idx.size:
        ax.quiver(x[idx], y[idx], vx_s[idx], -vy_s[idx],
                  angles="xy", scale_units="xy", scale=None,
                  width=0.0015, color="darkblue", alpha=0.7)
    ax.set_title("Stage 1: regularized flow", fontweight="bold")
    _annotate(ax, [
        f"n_iters: {n_iters}",
        f"keep: {keep}",
        f"valid before: {100.0 * n_valid_before / max(n, 1):.1f}%",
        f"valid after:  {100.0 * n_valid_after / max(n, 1):.1f}%",
        f"median |v|: {med:.4g} px/s",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_ego_fit(x, y, params, residual, sensor, out_dir, res_k, thresh,
                 inlier_rms, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ego_motion import ego_field

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "03_ego_fit.png")
    W, H = sensor
    cx, cy = info.get("cx", 0.5 * (W - 1)), info.get("cy", 0.5 * (H - 1))
    tx, ty, w, s = params
    ransac = bool(info.get("ransac", False))
    n_ransac = int(info.get("n_ransac_inliers", 0))
    n_valid = int(info.get("n_valid", max(len(x), 1)))
    n_hyp = int(info.get("n_hypotheses", 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    # optional RANSAC inlier/outlier scatter under quiver
    mask = info.get("ransac_inlier_mask")
    if ransac and mask is not None and np.any(mask):
        out = ~np.asarray(mask, dtype=bool)
        if out.any():
            ax.scatter(x[out], y[out], c="0.85", s=1, linewidths=0, zorder=0)
        ax.scatter(x[mask], y[mask], c="steelblue", s=1, linewidths=0,
                   alpha=0.5, zorder=1, label="RANSAC inliers")
    gx = np.linspace(0, W - 1, 18)
    gy = np.linspace(0, H - 1, 14)
    GX, GY = np.meshgrid(gx, gy)
    pvx, pvy = ego_field(GX.ravel(), GY.ravel(), params, cx, cy)
    ax.quiver(GX.ravel(), GY.ravel(), pvx, -pvy,
              angles="xy", scale_units="xy", scale=None,
              width=0.003, color="crimson", zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("fitted ego field v_pred")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    rn = np.hypot(residual[:, 0], residual[:, 1])
    rn_pos = rn[rn > 1e-12]
    ax.hist(rn_pos, bins=60, color="gray", edgecolor="none", alpha=0.85)
    ax.axvline(thresh, color="crimson", lw=2, label=f"thresh={thresh:.3g}")
    ax.set_xlabel("|r| (px/s)")
    ax.set_ylabel("count")
    ax.set_title("residual magnitude histogram")
    ax.legend(fontsize=8)

    title = ("Stage 2: ego-motion fit (RANSAC + IRLS polish)" if ransac
             else "Stage 2: ego-motion fit (4-param IRLS)")
    fig.suptitle(title, fontweight="bold")
    box = [
        f"EGO_RANSAC: {ransac}",
        f"n_hypotheses: {n_hyp}",
        f"RANSAC inliers: {n_ransac}/{n_valid}",
        f"tx={tx:.4g}  ty={ty:.4g}",
        f"w={w:.4g}  s={s:.4g}",
        f"polished inlier RMS={inlier_rms:.4g}",
        f"RES_K={res_k}",
        f"thresh={thresh:.4g} px/s",
    ]
    _annotate(axes[0], box)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_ego_inliers(x, y, params, info, sensor, out_dir):
    """10_ego_inliers.png — RANSAC consensus (background) vs outliers + ego quiver."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ego_motion import ego_field

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "10_ego_inliers.png")
    W, H = sensor
    cx, cy = info.get("cx", 0.5 * (W - 1)), info.get("cy", 0.5 * (H - 1))
    tx, ty, w, s = params
    mask = info.get("ransac_inlier_mask")
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    n_inl = int(mask.sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    out = ~mask
    if out.any():
        ax.scatter(x[out], y[out], c="tomato", s=1, linewidths=0,
                   alpha=0.6, label="outlier / IMO")
    if mask.any():
        ax.scatter(x[mask], y[mask], c="steelblue", s=1, linewidths=0,
                   alpha=0.7, label="ego consensus (BG)")
    gx = np.linspace(0, W - 1, 16)
    gy = np.linspace(0, H - 1, 12)
    GX, GY = np.meshgrid(gx, gy)
    pvx, pvy = ego_field(GX.ravel(), GY.ravel(), params, cx, cy)
    ax.quiver(GX.ravel(), GY.ravel(), pvx, -pvy,
              angles="xy", scale_units="xy", scale=None,
              width=0.003, color="black", alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ego consensus inliers (should be BACKGROUND, not a person)",
                 fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8, markerscale=4)
    _annotate(ax, [
        f"tx={tx:.4g}  ty={ty:.4g}",
        f"w={w:.4g}  s={s:.4g}",
        f"inliers: {n_inl}/{len(x)}",
        f"inlier RMS: {info.get('inlier_rms', float('nan')):.4g}",
        f"RANSAC: {bool(info.get('ransac', False))}",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_residual_split(x, y, is_imo, residual, thresh, out_dir,
                        res_k=None, dilate_info=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "04_residual_split.png")
    n = len(x)
    n_imo = int(is_imo.sum())
    n_bg = n - n_imo
    frac = 100.0 * n_imo / max(n, 1)
    dilate_info = dilate_info or {}

    fig, ax = plt.subplots(figsize=(9, 7))
    bg = ~is_imo
    if bg.any():
        ax.scatter(x[bg], y[bg], c="0.75", s=1, linewidths=0, label="background")
    if is_imo.any():
        ang = np.arctan2(residual[is_imo, 1], residual[is_imo, 0])
        hue = (ang + np.pi) / (2 * np.pi)
        colors = np.stack([hue, np.ones_like(hue) * 0.85, np.ones_like(hue) * 0.9],
                         axis=1)
        rgb = mcolors.hsv_to_rgb(colors)
        ax.scatter(x[is_imo], y[is_imo], c=rgb, s=2, linewidths=0, label="IMO")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Stage 2: residual split (background vs IMO)", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    box = [
        f"thresh: {thresh:.4g} px/s",
        f"RES_K: {res_k if res_k is not None else '?'}",
        f"dilate iters: {dilate_info.get('n_dilate', '?')}",
        f"dilate frac: {dilate_info.get('frac', '?')}",
        f"#added by dilation: {dilate_info.get('n_added', 0)}",
        f"#removed by erosion: {dilate_info.get('n_removed', 0)}",
        f"#background: {n_bg}",
        f"#IMO: {n_imo}",
        f"IMO %: {frac:.1f}",
    ]
    _annotate(ax, box)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_supernodes(x, y, cluster_id, is_imo, out_dir, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "05_supernodes.png")
    fig, ax = plt.subplots(figsize=(9, 7))
    bg = ~is_imo
    if bg.any():
        ax.scatter(x[bg], y[bg], c="0.85", s=1, linewidths=0)
    if is_imo.any():
        cid = cluster_id
        colors = (cid.astype(np.int64) * 2654435761) % (2 ** 32)
        sc = ax.scatter(x[is_imo], y[is_imo], c=colors, s=2,
                        cmap="gist_ncar", linewidths=0)
        fig.colorbar(sc, ax=ax, label="hashed cluster_id")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Stage 3: motion-coherent supernodes (residual affinity)",
                 fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    sizes = info.get("sizes", np.array([0]))
    _annotate(ax, [
        f"sigma_v: {info.get('sigma_v', float('nan')):.4g}",
        f"w_min: {info.get('w_min', float('nan'))}",
        f"n_levels: {info.get('n_levels', '?')}",
        f"C: {info.get('C', int(cluster_id.max()) + 1 if is_imo.any() else 0)}",
        f"size min/med/max: "
        f"{int(sizes.min()) if len(sizes) else 0}/"
        f"{int(np.median(sizes)) if len(sizes) else 0}/"
        f"{int(sizes.max()) if len(sizes) else 0}",
        f"rejections: {info.get('n_rejected_total', 0)}",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_segmentation(x, y, labels, out_dir, tau, num_layers, lam, smooth_iters,
                      runtime_s, w_node_motion=None, n_models=None,
                      object_counts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "06_segmentation.png")
    ids, counts = np.unique(labels, return_counts=True)
    n_obj = int((ids > 0).sum())
    count_lines = [f"  id {i}: {c}" for i, c in zip(ids, counts)]
    if object_counts is not None:
        count_lines = [f"  id {i}: {c}" for i, c in object_counts]

    fig, ax = plt.subplots(figsize=(9, 7))
    bg = labels == 0
    if bg.any():
        ax.scatter(x[bg], y[bg], c="0.82", s=1, linewidths=0)
    fg = labels > 0
    if fg.any():
        sc = ax.scatter(x[fg], y[fg], c=labels[fg], s=2, cmap="tab10",
                        linewidths=0)
        fig.colorbar(sc, ax=ax, label="object_id")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Stage 3: final segmentation", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    box = [
        f"tau: {tau}",
        f"num_layers: {num_layers}",
        f"lam: {lam}",
        f"smooth_iters: {smooth_iters}",
        f"W_NODE_MOTION: {w_node_motion if w_node_motion is not None else '?'}",
        f"#models found: {n_models if n_models is not None else '?'}",
        f"#objects: {n_obj}",
        f"runtime: {runtime_s:.1f}s",
    ] + count_lines[:8]
    _annotate(ax, box)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_motion_kernels(dir_cb, speed_cb, out_dir, bw_speed, speed_v0,
                        n_angle_bins, phase_int_kmax):
    """08_motion_kernels.png — periodic direction + log-speed codebook sims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "08_motion_kernels.png")

    # direction heatmap over [0, 2pi]
    n_th = 72
    thetas = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    Zdir = dir_cb.encode(thetas, interpolate=True)
    Zdir = torch.nn.functional.normalize(Zdir, p=2, dim=1)
    Gdir = (Zdir @ Zdir.T).detach().cpu().numpy()

    # speed heatmap over a useful range
    n_s = 48
    speeds = np.linspace(0.0, 200.0, n_s)
    Zsp = speed_cb.encode(speeds, interpolate=True)
    Zsp = torch.nn.functional.normalize(Zsp, p=2, dim=1)
    Gsp = (Zsp @ Zsp.T).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    im0 = ax.imshow(Gdir, origin="lower", cmap="viridis",
                    extent=[0, 360, 0, 360], vmin=-0.2, vmax=1.0)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("theta (deg)")
    ax.set_title("dir codebook cosine")
    fig.colorbar(im0, ax=ax, fraction=0.046)

    ax = axes[1]
    im1 = ax.imshow(Gsp, origin="lower", cmap="magma",
                    extent=[speeds[0], speeds[-1], speeds[0], speeds[-1]],
                    vmin=-0.2, vmax=1.0)
    ax.set_xlabel("|r| (px/s)")
    ax.set_ylabel("|r| (px/s)")
    ax.set_title("speed codebook cosine")
    fig.colorbar(im1, ax=ax, fraction=0.046)

    fig.suptitle("Motion codebooks: periodic direction + log speed",
                 fontweight="bold")
    _annotate(axes[0], [
        f"BW_SPEED: {bw_speed}",
        f"SPEED_V0: {speed_v0}",
        f"N_ANGLE_BINS: {n_angle_bins}",
        f"PHASE_INT_KMAX: {phase_int_kmax}",
        "bandwidth_dir: 1.0",
    ], loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_motion_models(x, y, is_imo, model_id, models, out_dir,
                       max_models, model_res_k, min_model_inliers,
                       merge_info=None, model_kind="affine"):
    """09_motion_models.png — IMO coloured by model_id + centroid quivers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "09_motion_models.png")
    merge_info = merge_info or {}

    fig, ax = plt.subplots(figsize=(9, 7))
    bg = ~is_imo
    if bg.any():
        ax.scatter(x[bg], y[bg], c="0.82", s=1, linewidths=0)
    fg = is_imo
    if fg.any():
        mid = model_id[fg].astype(np.float64)
        sc = ax.scatter(x[fg], y[fg], c=mid, s=2, cmap="tab10",
                        linewidths=0, vmin=-1)
        fig.colorbar(sc, ax=ax, label="model_id")

    for m in models:
        mid = m["id"]
        mask = model_id == mid
        if not mask.any():
            continue
        cx = float(np.mean(x[mask]))
        cy = float(np.mean(y[mask]))
        tx, ty = m.get("tx", 0.0), m.get("ty", 0.0)
        ax.quiver([cx], [cy], [tx], [-ty],
                  angles="xy", scale_units="xy", scale=None,
                  width=0.006, color="black", zorder=5)
        ax.scatter([cx], [cy], c="black", s=30, zorder=6)

    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Multi-model fitting ({model_kind}, merged)", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

    box = [
        f"kind: {model_kind}",
        f"MAX_MODELS: {max_models}",
        f"MODEL_RES_K: {model_res_k}",
        f"MIN_MODEL_INLIERS: {min_model_inliers}",
        f"MERGE_COS: {merge_info.get('merge_cos', '?')}",
        f"pre-merge: {merge_info.get('n_pre', '?')}",
        f"post-merge: {merge_info.get('n_post', len(models))}",
        f"#final models: {len(models)}",
    ]
    for m in models:
        box.append(
            f"m{m['id']}: n={m['n_inliers']}  "
            f"tx={m.get('tx', 0):.3g} ty={m.get('ty', 0):.3g}  "
            f"RMS={m['rms']:.3g}"
        )
    _annotate(ax, box, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_summary(paths, out_dir):
    """3x3 grid of stages + motion diagnostics for one-glance reporting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "07_summary.png")
    # pad to 9 slots
    slots = list(paths) + [None] * max(0, 9 - len(paths))
    slots = slots[:9]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for ax, p in zip(axes.ravel(), slots):
        if p and os.path.isfile(p):
            img = mpimg.imread(p)
            ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Ego-motion-compensated VSA segmentation — summary",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path
