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


def _fig_path(out_dir, stem, name_suffix=None):
    """Build output PNG path; stem is e.g. '06_segmentation'."""
    if name_suffix:
        return os.path.join(out_dir, f"{stem}_{name_suffix}.png")
    return os.path.join(out_dir, f"{stem}.png")


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


def plot_flow_raw(x, y, vx, vy, out_dir, n_sample=2000, seed=0, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "01_flow_raw", name_suffix)
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
                       n_valid_before, n_valid_after, n_sample=2000, seed=0,
                       *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "02_flow_smoothed", name_suffix)
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
                 inlier_rms, info, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ego_motion import ego_field

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "03_ego_fit", name_suffix)
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


def plot_residual_split(x, y, is_imo, residual, thresh, out_dir, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "04_residual_split", name_suffix)
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


def plot_supernodes(x, y, cluster_id, is_imo, out_dir, info, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "05_supernodes", name_suffix)
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
                      runtime_s, extra_box=None, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "06_segmentation", name_suffix)
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
    ]
    if extra_box:
        box.extend(f"{k}: {v}" for k, v in extra_box.items())
    box.extend(count_lines[:8])
    _annotate(ax, box)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_summary(paths, out_dir, *, name_suffix=None):
    """Flexible grid summary of diagnostic images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "07_summary", name_suffix)
    paths = [p for p in paths if p]
    n = max(len(paths), 1)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, paths + [None] * (len(axes) - len(paths))):
        if p and os.path.isfile(p):
            ax.imshow(mpimg.imread(p))
        ax.axis("off")
    fig.suptitle("Aperture-resolved VSA segmentation — summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_flow_resolved(x, y, vx, vy, resolved_mask, out_dir, info, n_sample=2000,
                       *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "11_flow_resolved", name_suffix)
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_events(ax, x, y, c="0.9", s=1)
    idx = _subsample_idx(len(x), n_sample, 0)
    conf = info.get("confidence")
    if conf is not None:
        c = conf[idx]
        q = ax.quiver(x[idx], y[idx], vx[idx], -vy[idx], c,
                      angles="xy", scale_units="xy", scale=None,
                      width=0.0015, cmap="viridis", alpha=0.8)
        fig.colorbar(q, ax=ax, label="confidence")
    else:
        colors = np.where(resolved_mask[idx], "darkgreen", "crimson")
        for color in ("darkgreen", "crimson"):
            m = colors == color
            if m.any():
                ax.quiver(x[idx][m], y[idx][m], vx[idx][m], -vy[idx][m],
                          angles="xy", scale_units="xy", scale=None,
                          width=0.0015, color=color, alpha=0.7)
    ax.set_title(f"Aperture-resolved flow ({info.get('method', '?')})",
                 fontweight="bold")
    _annotate(ax, [
        f"method: {info.get('method')}",
        f"resolved: {100*info.get('resolved_frac', 0):.1f}%",
        f"|v| before: {info.get('median_speed_before', 0):.4g}",
        f"|v| after:  {info.get('median_speed_after', 0):.4g}",
        f"orient_corr: {info.get('orientation_corr', float('nan')):.4f}",
        f"unresolved: {info.get('n_unresolved', '?')}",
        f"runtime: {info.get('runtime_s', 0):.3f}s",
        f"gate: {info.get('n_conditioning_gate', '-')}",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_orientation_check(n_hat, vx, vy, valid, out_dir, info, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "12_orientation_check", name_suffix)
    ang_n = np.arctan2(n_hat[valid, 1], n_hat[valid, 0])
    ang_v = np.arctan2(vy[valid], vx[valid])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ang_n, ang_v, s=2, alpha=0.3, c="steelblue", linewidths=0)
    ax.set_xlabel("edge-normal angle")
    ax.set_ylabel("resolved-flow angle")
    ax.set_title("Orientation check (flat=resolved, diagonal=locked)",
                 fontweight="bold")
    ax.set_aspect("equal")
    _annotate(ax, [
        f"method: {info.get('method')}",
        f"orientation_corr: {info.get('orientation_corr', float('nan')):.4f}",
        f"N valid: {int(valid.sum())}",
        "(low corr = aperture resolved)",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_constraint_votes(x, y, is_imo, info, out_dir, *, name_suffix=None):
    """13: vote maps before/after bundling for 3 hand-picked events."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = info.get("C_sparse")
    V = info.get("V_sparse")
    grid = info.get("grid_coords")
    if C is None or V is None or grid is None:
        return None
    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "13_constraint_votes", name_suffix)
    G = grid.shape[0]
    gn = int(np.sqrt(G))
    # pick events: strong vertical edge, horizontal, corner-like among IMO
    cand = np.where(is_imo)[0]
    if cand.size < 3:
        cand = np.arange(len(x))
    # heuristic: use events with diverse positions
    picks = [cand[0], cand[len(cand)//3], cand[2*len(cand)//3]]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for row, i in enumerate(picks):
        before = C[i].toarray().reshape(gn, gn)
        after = V[i].toarray().reshape(gn, gn)
        axes[row, 0].imshow(before, origin="lower", cmap="magma")
        axes[row, 0].set_title(f"event {i} BEFORE bundle (ridge)")
        axes[row, 1].imshow(after, origin="lower", cmap="magma")
        axes[row, 1].set_title(f"event {i} AFTER bundle (peak)")
        # mark resolved velocity if available
        conf = info.get("confidence")
        if conf is not None:
            axes[row, 1].set_xlabel(f"conf={conf[i]:.3f}")
    fig.suptitle("VSA constraint votes: ridges → peaks", fontweight="bold")
    _annotate(axes[0, 1], [
        f"VEL_GRID_N: {info.get('VEL_GRID_N')}",
        f"BAND_SIGMA: {info.get('BAND_SIGMA')}",
        f"D_VEL: {info.get('D_VEL')}",
        f"G: {info.get('G')}",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_vsa_vs_hough(v_vsa, v_hough, mask, out_dir, info, *, name_suffix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    path = _fig_path(out_dir, "14_vsa_vs_hough", name_suffix)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    m = np.asarray(mask, dtype=bool)
    for ax, dim, name in zip(axes, (0, 1), ("vx", "vy")):
        ax.scatter(v_hough[m, dim], v_vsa[m, dim], s=2, alpha=0.3, linewidths=0)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel(f"Hough {name}")
        ax.set_ylabel(f"VSA {name}")
        ax.set_title(name)
        ax.set_aspect("equal")
    fig.suptitle("VSA superposition vs explicit Hough", fontweight="bold")
    _annotate(axes[1], [
        f"corr_vx: {info.get('hough_corr_vx', float('nan')):.4f}",
        f"corr_vy: {info.get('hough_corr_vy', float('nan')):.4f}",
        f"RMS: {info.get('hough_rms', float('nan')):.4g}",
        f"D_VEL: {info.get('D_VEL')}",
        f"G: {info.get('G')}",
    ])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_resolver_comparison(fig_paths_by_method, out_dir):
    """15: columns=methods, rows=quiver / residual split / segmentation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, "15_resolver_comparison.png")
    methods = list(fig_paths_by_method.keys())
    nrows, ncols = 3, len(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if ncols == 1:
        axes = np.array(axes).reshape(nrows, 1)
    row_keys = ["quiver", "split", "seg"]
    row_titles = ["resolved flow", "residual split", "segmentation"]
    for c, m in enumerate(methods):
        pack = fig_paths_by_method[m]
        for r, key in enumerate(row_keys):
            ax = axes[r, c]
            p = pack.get(key)
            if p and os.path.isfile(p):
                ax.imshow(mpimg.imread(p))
            ax.axis("off")
            if r == 0:
                ax.set_title(m, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_titles[r])
            row = pack.get("row", {})
            if r == 0 and row:
                _annotate(ax, [
                    f"resol%: {100*row.get('resolved_frac', 0):.1f}",
                    f"orient: {row.get('orientation_corr', float('nan')):.3f}",
                    f"imo%: {100*row.get('imo_frac', 0):.1f}",
                    f"#obj: {row.get('n_objects', 0)}",
                    f"t: {row.get('total_runtime_s', 0):.1f}s",
                ], loc="lower left")
    fig.suptitle("Resolver comparison (none / lk / vsa)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path
