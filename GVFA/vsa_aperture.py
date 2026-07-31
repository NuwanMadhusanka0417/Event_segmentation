"""
vsa_aperture.py — Track B: VSA constraint-line superposition aperture resolver.

Each event's normal-flow constraint is a Hough vote pattern over a velocity grid.
Neighbour bundling = superposition (VSA); cleanup recovers the intersection.
An explicit Hough readout validates that HV voting matches classical voting.

Constraints are evaluated in LINEAR px/s (n_hat · v = u). SSP codes are built
from signed-log grid coordinates for dense low-speed sampling.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch
from scipy import sparse

from fpe_codebook import FPECodebook, bind_hv
from aperture import undirected_edges_from_list


def _signed_log(w, v0):
    w = np.asarray(w, dtype=np.float64)
    return np.sign(w) * np.log1p(np.abs(w) / v0)


def _inv_signed_log(s, v0):
    s = np.asarray(s, dtype=np.float64)
    return np.sign(s) * v0 * np.expm1(np.abs(s))


def _local_spacing_linear(grid_linear, grid_n):
    """Min linear px/s spacing to an adjacent grid neighbour, per cell [G]."""
    lin = grid_linear.reshape(grid_n, grid_n, 2)
    spacing = np.full((grid_n, grid_n), np.inf, dtype=np.float64)
    if grid_n > 1:
        dx = np.linalg.norm(lin[:, 1:, :] - lin[:, :-1, :], axis=2)
        dy = np.linalg.norm(lin[1:, :, :] - lin[:-1, :, :], axis=2)
        spacing[:, :-1] = np.minimum(spacing[:, :-1], dx)
        spacing[:, 1:] = np.minimum(spacing[:, 1:], dx)
        spacing[:-1, :] = np.minimum(spacing[:-1, :], dy)
        spacing[1:, :] = np.minimum(spacing[1:, :], dy)
    med = float(np.median(spacing[np.isfinite(spacing) & (spacing < np.inf)]))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    spacing[~np.isfinite(spacing)] = med
    return spacing.ravel()


def build_velocity_codebook(D_vel, grid_n, v_max, *, signed_log=True, v0=50.0,
                            bw=0.15, seed=0):
    """SSP velocity codebook on a signed-log grid.

    Returns
    -------
    codebook    [G, D_vel]  L2-normalized SSP codes (built from grid_slog)
    grid_linear [G, 2]    actual velocities px/s (constraint evaluation)
    grid_slog   [G, 2]    signed-log sampling coords (code construction)
    local_spacing [G]     linear spacing to adjacent grid cells (px/s)
    """
    D_vel = int(D_vel)
    grid_n = int(grid_n)
    if signed_log:
        s_max = float(_signed_log(v_max, v0))
        s_axis = np.linspace(-s_max, s_max, grid_n)
        SX, SY = np.meshgrid(s_axis, s_axis, indexing="xy")
        grid_slog = np.stack([SX.ravel(), SY.ravel()], axis=1)
        grid_linear = np.stack([
            _inv_signed_log(grid_slog[:, 0], v0),
            _inv_signed_log(grid_slog[:, 1], v0),
        ], axis=1)
        s_vals = grid_slog
        vmax_idx = int(np.ceil(s_max / 0.1)) + 1
    else:
        axis = np.linspace(-v_max, v_max, grid_n)
        VX, VY = np.meshgrid(axis, axis, indexing="xy")
        grid_linear = np.stack([VX.ravel(), VY.ravel()], axis=1)
        grid_slog = grid_linear / max(v0, 1.0)
        s_vals = grid_slog
        vmax_idx = int(np.ceil(v_max / 0.1)) + 1

    cb_x = FPECodebook(
        "vel_x", D_vel, bw, "radix", radix_S=16,
        vmin=-vmax_idx, vmax=vmax_idx, value_grid_step=0.1,
        phase_dist="gaussian", seed=seed + 101,
    )
    cb_y = FPECodebook(
        "vel_y", D_vel, bw, "radix", radix_S=16,
        vmin=-vmax_idx, vmax=vmax_idx, value_grid_step=0.1,
        phase_dist="gaussian", seed=seed + 102,
    )
    zx = cb_x.encode(s_vals[:, 0], interpolate=True)
    zy = cb_y.encode(s_vals[:, 1], interpolate=True)
    codebook = bind_hv(zx, zy)
    codebook = torch.nn.functional.normalize(codebook, p=2, dim=1)
    local_spacing = _local_spacing_linear(grid_linear, grid_n)
    return (
        codebook.detach().cpu().numpy().astype(np.float64),
        grid_linear.astype(np.float64),
        grid_slog.astype(np.float64),
        local_spacing.astype(np.float64),
    )


def build_constraint_matrix(u, n_hat, grid_linear, *, band_sigma_px=40.0,
                            local_spacing=None, weight_floor=1e-2, chunk=2048):
    """Sparse [N, G] Hough weights using LINEAR px/s constraint distance."""
    n = len(u)
    G = grid_linear.shape[0]
    if local_spacing is None:
        local_spacing = np.ones(G, dtype=np.float64)
    sigma_eff = np.maximum(float(band_sigma_px), 0.5 * local_spacing)
    sig2 = 2.0 * sigma_eff * sigma_eff  # [G]

    rows, cols, data = [], [], []
    valid = np.isfinite(u) & (np.linalg.norm(n_hat, axis=1) > 0.5)
    idx_all = np.where(valid)[0]

    for start in range(0, len(idx_all), chunk):
        ids = idx_all[start:start + chunk]
        # d[i,g] = n_i · v_g - u_i  in px/s
        d = n_hat[ids] @ grid_linear.T - u[ids, None]
        w = np.exp(-(d * d) / sig2[None, :])
        mask = w >= weight_floor
        ii, gg = np.where(mask)
        if ii.size == 0:
            continue
        rows.append(ids[ii])
        cols.append(gg)
        data.append(w[ii, gg])

    if not rows:
        return sparse.csr_matrix((n, G), dtype=np.float64)
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, G), dtype=np.float64,
    )


def _boundary_mask(grid_n, G, boundary_ring=1):
    """Bool [G] — cells within `boundary_ring` of the grid edge."""
    ring = int(boundary_ring)
    idx = np.arange(G)
    iy = idx // grid_n
    ix = idx % grid_n
    return (
        (ix < ring) | (ix >= grid_n - ring)
        | (iy < ring) | (iy >= grid_n - ring)
    )


def _pick_verification_events(n_hat, valid, u, n_pick=3, seed=0):
    """Deterministic: horizontal, vertical, corner-like normals among valid."""
    cand = np.where(valid)[0]
    if cand.size == 0:
        return np.array([], dtype=np.int64)
    nh = n_hat[cand]
    horiz = cand[np.argmax(np.abs(nh[:, 0]))]
    vert = cand[np.argmax(np.abs(nh[:, 1]))]
    corner_score = np.abs(np.abs(nh[:, 0]) - np.abs(nh[:, 1]))
    corner = cand[np.argmin(corner_score)]
    picks = np.unique(np.array([horiz, vert, corner], dtype=np.int64))
    return picks[:n_pick]


def _print_active_cell_fractions(C, picks, G, weight_floor):
    for i in picks:
        row = C[i]
        nnz = row.nnz
        frac = 100.0 * nnz / max(G, 1)
        mass = float(row.sum()) if nnz else 0.0
        print(f"[vsa] event {i}: active cells={nnz}/{G} ({frac:.2f}%)  "
              f"vote_mass={mass:.4g}  (floor={weight_floor})")


def bundle_constraints(C, edge_index_list, n_nodes, *, self_weight=1.0):
    """Neighbour superposition: V = A @ C."""
    src, dst = undirected_edges_from_list(edge_index_list, n_nodes)
    if src.size == 0:
        A = sparse.eye(n_nodes, format="csr", dtype=np.float64) * self_weight
    else:
        w = np.ones(src.shape[0], dtype=np.float64)
        is_self = src == dst
        w[is_self] = float(self_weight)
        A = sparse.csr_matrix(
            (w, (dst, src)), shape=(n_nodes, n_nodes), dtype=np.float64,
        )
    t0 = time.perf_counter()
    V = A @ C
    t_bundle = time.perf_counter() - t0
    return V, t_bundle


def mean_active_codes(V):
    """Mean non-zero entries per row of sparse accumulator V."""
    if V.shape[0] == 0:
        return 0.0
    return float(np.diff(V.indptr).mean())


def readout_velocity(Z, codebook, grid_linear, V, *, topk=5, min_conf=0.05,
                     vote_mass_min=1e-6, boundary_ring=1, boundary_mass_frac=0.5,
                     grid_n=48, chunk=4096):
    """HV readout with three independent rejection tests."""
    n, _D = Z.shape
    G = codebook.shape[0]
    topk = min(int(topk), G)
    v_res = np.zeros((n, 2), dtype=np.float64)
    conf = np.zeros(n, dtype=np.float64)
    peak_idx = np.zeros(n, dtype=np.int64)

    on_boundary = _boundary_mask(grid_n, G, boundary_ring)
    bnd_cols = np.where(on_boundary)[0]

    vote_mass = np.asarray(V.sum(axis=1)).ravel()
    if bnd_cols.size:
        boundary_mass = np.asarray(V[:, bnd_cols].sum(axis=1)).ravel()
    else:
        boundary_mass = np.zeros(n, dtype=np.float64)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        S = Z[start:end] @ codebook.T
        if topk < G:
            part = np.argpartition(S, -topk, axis=1)[:, -topk:]
            Sk = np.take_along_axis(S, part, axis=1)
            order = np.argsort(Sk, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(part, order, axis=1)
            top_s = np.take_along_axis(S, top_idx, axis=1)
        else:
            top_idx = np.argsort(S, axis=1)[:, ::-1]
            top_s = np.take_along_axis(S, top_idx, axis=1)

        conf[start:end] = S.max(axis=1) - S.mean(axis=1)
        peak_idx[start:end] = S.argmax(axis=1)

        w = np.maximum(top_s, 0.0)
        wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        coords = grid_linear[top_idx]
        v_res[start:end] = (w[:, :, None] * coords).sum(axis=1) / wsum

    reject_empty = vote_mass < float(vote_mass_min)
    reject_lowconf = conf < float(min_conf)
    reject_boundary = on_boundary[peak_idx]
    mass_frac = boundary_mass / np.maximum(vote_mass, 1e-12)
    reject_boundary |= mass_frac > float(boundary_mass_frac)

    resolved = ~(reject_empty | reject_lowconf | reject_boundary)
    counters = {
        "reject_empty": int(reject_empty.sum()),
        "reject_lowconf": int(reject_lowconf.sum()),
        "reject_boundary": int(reject_boundary.sum()),
    }
    return v_res, conf, resolved, counters, peak_idx


def readout_velocity_explicit(V, grid_linear, topk=5, chunk=2048):
    """Explicit Hough readout on sparse accumulator V [N,G]."""
    n = V.shape[0]
    G = V.shape[1]
    topk = min(int(topk), G)
    v_ref = np.zeros((n, 2), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        S = V[start:end].toarray()
        if topk < G:
            part = np.argpartition(S, -topk, axis=1)[:, -topk:]
            Sk = np.take_along_axis(S, part, axis=1)
            order = np.argsort(Sk, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(part, order, axis=1)
            top_s = np.take_along_axis(S, top_idx, axis=1)
        else:
            top_idx = np.argsort(S, axis=1)[:, ::-1]
            top_s = np.take_along_axis(S, top_idx, axis=1)
        w = np.maximum(top_s, 0.0)
        wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        coords = grid_linear[top_idx]
        v_ref[start:end] = (w[:, :, None] * coords).sum(axis=1) / wsum
    return v_ref


def pick_sample_events(n_hat, valid, u, seed=0):
    """Events for constraint-vote figure: horizontal, vertical, corner."""
    return _pick_verification_events(n_hat, valid, u, seed=seed)


def resolve_flow_vsa(x, y, u, n_hat, valid, edge_index_list, *,
                     vx_perp, vy_perp, cfg):
    """Full Track B pipeline."""
    n = len(u)
    D_vel = int(cfg.get("D_VEL", 2048))
    grid_n = int(cfg.get("VEL_GRID_N", 48))
    v_max = float(cfg.get("VEL_MAX", 1500.0))
    v0 = float(cfg.get("SPEED_V0", 50.0))
    bw = float(cfg.get("BW_VEL", 0.15))
    band_sigma_px = float(cfg.get("BAND_SIGMA_PX", 40.0))
    weight_floor = float(cfg.get("WEIGHT_FLOOR", 1e-2))
    self_weight = float(cfg.get("SELF_WEIGHT", 1.0))
    topk = int(cfg.get("CLEANUP_TOPK", 5))
    min_conf = float(cfg.get("CLEANUP_MIN_CONF", 0.05))
    vote_mass_min = float(cfg.get("VOTE_MASS_MIN", 1e-6))
    boundary_ring = int(cfg.get("BOUNDARY_RING", 1))
    boundary_mass_frac = float(cfg.get("BOUNDARY_MASS_FRAC", 0.5))
    chunk = int(cfg.get("READOUT_CHUNK", 4096))
    validate = bool(cfg.get("VSA_VALIDATE", True))
    seed = int(cfg.get("SEED", 0))

    t0 = time.perf_counter()
    codebook, grid_linear, grid_slog, local_spacing = build_velocity_codebook(
        D_vel, grid_n, v_max, signed_log=bool(cfg.get("VEL_SIGNED_LOG", True)),
        v0=v0, bw=bw, seed=seed,
    )
    G = grid_linear.shape[0]
    t_cb = time.perf_counter() - t0

    t1 = time.perf_counter()
    u_use = u.copy()
    n_use = n_hat.copy()
    u_use[~valid] = 0.0
    n_use[~valid] = 0.0
    C = build_constraint_matrix(
        u_use, n_use, grid_linear,
        band_sigma_px=band_sigma_px,
        local_spacing=local_spacing,
        weight_floor=weight_floor,
        chunk=chunk,
    )
    t_C = time.perf_counter() - t1

    verify_picks = _pick_verification_events(n_hat, valid, u, seed=seed)
    if verify_picks.size:
        _print_active_cell_fractions(C, verify_picks, G, weight_floor)

    t2 = time.perf_counter()
    V, t_bundle = bundle_constraints(
        C, edge_index_list, n, self_weight=self_weight,
    )
    M_eff = mean_active_codes(V)
    print(f"[vsa] M_eff={M_eff:.1f}  D_VEL={D_vel}  ratio={D_vel / max(M_eff, 1e-9):.2f}")
    if D_vel < 4.0 * M_eff:
        warnings.warn(
            f"[vsa] D_VEL={D_vel} < 4*M_eff={4*M_eff:.1f}: readout is capacity-limited",
            stacklevel=2,
        )
        print(f"[vsa] *** WARNING: D_VEL={D_vel} < 4*M_eff={4*M_eff:.1f} — "
              f"readout capacity-limited ***")

    Z = V @ codebook
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(norms, 1e-12)
    t_Z = time.perf_counter() - t2

    t3 = time.perf_counter()
    v_res, conf, resolved, rej, peak_idx = readout_velocity(
        Z, codebook, grid_linear, V,
        topk=topk, min_conf=min_conf,
        vote_mass_min=vote_mass_min,
        boundary_ring=boundary_ring,
        boundary_mass_frac=boundary_mass_frac,
        grid_n=grid_n, chunk=chunk,
    )
    resolved = resolved & valid
    t_read = time.perf_counter() - t3

    valid_frac = float(valid.mean()) if n else 0.0
    resolved_frac_valid = float(resolved[valid].mean()) if valid.any() else 0.0
    total_rej = rej["reject_empty"] + rej["reject_lowconf"] + rej["reject_boundary"]
    print(f"[vsa] reject: empty={rej['reject_empty']}  lowconf={rej['reject_lowconf']}  "
          f"boundary={rej['reject_boundary']}  resolved={int(resolved.sum())}/{n}  "
          f"resolved|valid={resolved_frac_valid:.4f}  valid_frac={valid_frac:.4f}")
    if valid.any() and resolved_frac_valid >= valid_frac - 1e-9:
        print("[vsa] *** WARNING: resolved_frac >= valid_frac — gate may be inert ***")
    if total_rej == 0 and valid.any():
        print("[vsa] *** WARNING: all rejection counters zero — gate inert ***")

    vx = np.where(resolved, v_res[:, 0], vx_perp)
    vy = np.where(resolved, v_res[:, 1], vy_perp)

    sample_events = pick_sample_events(n_hat, valid, u, seed=seed)
    vote_mass = np.asarray(V.sum(axis=1)).ravel()

    extra = {
        "D_VEL": D_vel,
        "VEL_GRID_N": grid_n,
        "G": G,
        "BAND_SIGMA_PX": band_sigma_px,
        "M_eff": M_eff,
        "CLEANUP_TOPK": topk,
        "CLEANUP_MIN_CONF": min_conf,
        "VOTE_MASS_MIN": vote_mass_min,
        "confidence": conf,
        "vote_mass": vote_mass,
        "reject_empty": rej["reject_empty"],
        "reject_lowconf": rej["reject_lowconf"],
        "reject_boundary": rej["reject_boundary"],
        "sample_events": sample_events,
        "grid_linear": grid_linear,
        "grid_slog": grid_slog,
        "t_codebook_s": t_cb,
        "t_constraint_s": t_C,
        "t_bundle_s": t_bundle,
        "t_superpose_s": t_Z,
        "t_readout_s": t_read,
        "V_sparse": V if validate else None,
        "grid_coords": grid_linear if validate else None,
        "C_sparse": C if validate else None,
        "Z": Z if validate else None,
        "codebook": codebook if validate else None,
    }

    if validate:
        v_ref = readout_velocity_explicit(V, grid_linear, topk=topk)
        mask = resolved
        if mask.sum() > 10:
            r_x = np.corrcoef(v_res[mask, 0], v_ref[mask, 0])[0, 1]
            r_y = np.corrcoef(v_res[mask, 1], v_ref[mask, 1])[0, 1]
            rms = float(np.sqrt(np.mean((v_res[mask] - v_ref[mask]) ** 2)))
        else:
            r_x = r_y = rms = float("nan")
        extra.update({
            "hough_corr_vx": float(r_x) if np.isfinite(r_x) else float("nan"),
            "hough_corr_vy": float(r_y) if np.isfinite(r_y) else float("nan"),
            "hough_rms": rms,
            "v_hough": v_ref,
        })
        print(f"[vsa] vs explicit Hough: corr_vx={r_x:.4f} corr_vy={r_y:.4f} "
              f"RMS={rms:.4g}  D_vel={D_vel} G={G} M_eff={M_eff:.1f}")
        print(f"[vsa] timings: C={t_C:.3f}s  bundle+Z={t_Z:.3f}s  readout={t_read:.3f}s")

    return vx, vy, resolved, extra
