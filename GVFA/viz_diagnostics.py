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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) ego field on grid
    ax = axes[0]
    gx = np.linspace(0, W - 1, 18)
    gy = np.linspace(0, H - 1, 14)
    GX, GY = np.meshgrid(gx, gy)
    pvx, pvy = ego_field(GX.ravel(), GY.ravel(), params, cx, cy)
    ax.quiver(GX.ravel(), GY.ravel(), pvx, -pvy,
              angles="xy", scale_units="xy", scale=None,
              width=0.003, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("fitted ego field v_pred")
    ax.set_aspect("equal", adjustable="box")

    # (b) residual histogram
    ax = axes[1]
    rn = np.hypot(residual[:, 0], residual[:, 1])
    rn_pos = rn[rn > 1e-12]
    ax.hist(rn_pos, bins=60, color="gray", edgecolor="none", alpha=0.85)
    ax.axvline(thresh, color="crimson", lw=2, label=f"thresh={thresh:.3g}")
    ax.set_xlabel("|r| (px/s)")
    ax.set_ylabel("count")
    ax.set_title("residual magnitude histogram")
    ax.legend(fontsize=8)

    fig.suptitle("Stage 2: ego-motion fit (4-param IRLS)", fontweight="bold")
    _annotate(axes[0], [
        f"tx={tx:.4g}  ty={ty:.4g}",
        f"w={w:.4g}  s={s:.4g}",
        f"inlier RMS={inlier_rms:.4g}",
        f"RES_K={res_k}",
        f"thresh={thresh:.4g} px/s",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_residual_split(x, y, is_imo, residual, thresh, out_dir):
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
    _annotate(ax, [
        f"thresh: {thresh:.4g} px/s",
        f"#background: {n_bg}",
        f"#IMO: {n_imo}",
        f"IMO %: {frac:.1f}",
    ])
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
                      runtime_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "06_segmentation.png")
    ids, counts = np.unique(labels, return_counts=True)
    n_obj = int((ids > 0).sum())
    count_lines = [f"  id {i}: {c}" for i, c in zip(ids, counts)]

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
        f"#objects: {n_obj}",
        f"runtime: {runtime_s:.1f}s",
    ] + count_lines[:8]
    _annotate(ax, box)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_summary(paths, out_dir):
    """2x3 grid of stages 1–6 for one-glance reporting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "07_summary.png")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, p in zip(axes.ravel(), paths):
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
