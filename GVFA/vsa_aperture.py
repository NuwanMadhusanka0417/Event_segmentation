"""
vsa_aperture.py — Track B: VSA constraint-line superposition aperture resolver.

Each event's normal-flow constraint is a Hough vote pattern over a velocity grid.
Neighbour bundling = superposition (VSA); cleanup recovers the intersection.
An explicit Hough readout validates that HV voting matches classical voting.
"""

from __future__ import annotations

import time

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
    return np.sign(s) * v0 * (np.expm1(np.abs(s)))


def build_velocity_codebook(D_vel, grid_n, v_max, *, signed_log=True, v0=50.0,
                            bw=0.15, seed=0):
    """2D SSP velocity codebook on a signed-log (or linear) grid.

    Returns codebook [G, D_vel] (L2-normalized) and grid_coords [G, 2] in px/s.
    """
    D_vel = int(D_vel)
    grid_n = int(grid_n)
    # Build grid in signed-log space spanning ±slog(v_max)
    if signed_log:
        s_max = float(_signed_log(v_max, v0))
        s_axis = np.linspace(-s_max, s_max, grid_n)
        SX, SY = np.meshgrid(s_axis, s_axis, indexing="xy")
        grid_slog = np.stack([SX.ravel(), SY.ravel()], axis=1)
        grid_coords = np.stack([
            _inv_signed_log(grid_slog[:, 0], v0),
            _inv_signed_log(grid_slog[:, 1], v0),
        ], axis=1)
        # Codebooks encode signed-log coordinates
        s_vals = grid_slog
        vmax_idx = int(np.ceil(s_max / 0.1)) + 1
    else:
        axis = np.linspace(-v_max, v_max, grid_n)
        VX, VY = np.meshgrid(axis, axis, indexing="xy")
        grid_coords = np.stack([VX.ravel(), VY.ravel()], axis=1)
        s_vals = grid_coords / max(v0, 1.0)  # crude scale
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
    return codebook.detach().cpu().numpy().astype(np.float64), grid_coords.astype(np.float64)


def build_constraint_matrix(u, n_hat, grid_coords, *, band_sigma=0.08,
                            weight_floor=1e-3, chunk=2048):
    """Sparse [N, G] Hough weights: exp(-(n·v_g - u)^2 / (2 σ^2))."""
    n = len(u)
    G = grid_coords.shape[0]
    # Work in same units as u (px/s): distance n·v - u
    rows, cols, data = [], [], []
    valid = np.isfinite(u) & (np.linalg.norm(n_hat, axis=1) > 0.5)
    idx_all = np.where(valid)[0]
    sig2 = 2.0 * band_sigma * band_sigma

    for start in range(0, len(idx_all), chunk):
        ids = idx_all[start:start + chunk]
        # d[i,g] = n_i · v_g - u_i   -> [chunk, G]
        d = n_hat[ids] @ grid_coords.T - u[ids, None]
        w = np.exp(-(d * d) / sig2)
        mask = w >= weight_floor
        ii, gg = np.where(mask)
        if ii.size == 0:
            continue
        rows.append(ids[ii])
        cols.append(gg)
        data.append(w[ii, gg])

    if not rows:
        return sparse.csr_matrix((n, G), dtype=np.float64)
    C = sparse.csr_matrix(
        (np.concatenate(data),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, G), dtype=np.float64,
    )
    return C


def bundle_constraints(C, edge_index_list, n_nodes, *, self_weight=1.0):
    """Neighbour superposition: V = A @ C (sparse), then caller does Z = V @ codebook."""
    src, dst = undirected_edges_from_list(edge_index_list, n_nodes)
    # Build A: dst <- src with weight 1, self already included once — boost self
    if src.size == 0:
        A = sparse.eye(n_nodes, format="csr", dtype=np.float64) * self_weight
    else:
        # undirected_edges_from_list already adds self-loops once
        w = np.ones(src.shape[0], dtype=np.float64)
        # identify self-loops and scale
        is_self = src == dst
        w[is_self] = float(self_weight)
        A = sparse.csr_matrix(
            (w, (dst, src)), shape=(n_nodes, n_nodes), dtype=np.float64,
        )
    t0 = time.perf_counter()
    V = A @ C
    t_bundle = time.perf_counter() - t0
    return V, t_bundle


def readout_velocity(Z, codebook, grid_coords, *, topk=5, min_conf=0.05,
                     chunk=4096, v0=50.0, signed_log_grid=False):
    """Similarity-weighted centroid of top-k grid velocities + confidence."""
    n, D = Z.shape
    G = codebook.shape[0]
    v_res = np.zeros((n, 2), dtype=np.float64)
    conf = np.zeros(n, dtype=np.float64)
    topk = min(int(topk), G)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        S = Z[start:end] @ codebook.T  # [c, G]
        # top-k
        if topk < G:
            part = np.argpartition(S, -topk, axis=1)[:, -topk:]
            # sort those
            row = np.arange(end - start)[:, None]
            Sk = np.take_along_axis(S, part, axis=1)
            order = np.argsort(Sk, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(part, order, axis=1)
            top_s = np.take_along_axis(S, top_idx, axis=1)
        else:
            top_idx = np.argsort(S, axis=1)[:, ::-1]
            top_s = np.take_along_axis(S, top_idx, axis=1)

        # confidence = peak - mean
        conf[start:end] = S.max(axis=1) - S.mean(axis=1)

        # weighted centroid in px/s
        w = np.maximum(top_s, 0.0)
        wsum = w.sum(axis=1, keepdims=True)
        wsum = np.maximum(wsum, 1e-12)
        coords = grid_coords[top_idx]  # [c, k, 2]
        v_res[start:end] = (w[:, :, None] * coords).sum(axis=1) / wsum

    resolved = conf >= float(min_conf)
    return v_res, conf, resolved


def readout_velocity_explicit(V, grid_coords, topk=5):
    """Explicit Hough readout on sparse accumulator V [N,G] (no hypervectors)."""
    n = V.shape[0]
    # densify in chunks if needed — for CSR, use toarray per chunk
    G = V.shape[1]
    topk = min(int(topk), G)
    v_ref = np.zeros((n, 2), dtype=np.float64)
    # Convert to dense by rows in chunks
    chunk = 2048
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
        coords = grid_coords[top_idx]
        v_ref[start:end] = (w[:, :, None] * coords).sum(axis=1) / wsum
    return v_ref


def resolve_flow_vsa(x, y, u, n_hat, valid, edge_index_list, *,
                     vx_perp, vy_perp, cfg):
    """Full Track B pipeline."""
    n = len(u)
    D_vel = int(cfg.get("D_VEL", 512))
    grid_n = int(cfg.get("VEL_GRID_N", 48))
    v_max = float(cfg.get("VEL_MAX", 1500.0))
    v0 = float(cfg.get("SPEED_V0", 50.0))
    bw = float(cfg.get("BW_VEL", 0.15))
    band_sigma = float(cfg.get("BAND_SIGMA", 0.08))
    weight_floor = float(cfg.get("WEIGHT_FLOOR", 1e-3))
    self_weight = float(cfg.get("SELF_WEIGHT", 1.0))
    topk = int(cfg.get("CLEANUP_TOPK", 5))
    min_conf = float(cfg.get("CLEANUP_MIN_CONF", 0.05))
    chunk = int(cfg.get("READOUT_CHUNK", 4096))
    validate = bool(cfg.get("VSA_VALIDATE", True))
    seed = int(cfg.get("SEED", 0))

    # Optional: only resolve valid constraint events (others keep normal flow)
    t0 = time.perf_counter()
    codebook, grid_coords = build_velocity_codebook(
        D_vel, grid_n, v_max, signed_log=bool(cfg.get("VEL_SIGNED_LOG", True)),
        v0=v0, bw=bw, seed=seed,
    )
    t_cb = time.perf_counter() - t0

    # Constraint band in px/s: scale band_sigma from signed-log units roughly
    # Task specifies band_sigma in same units as signed-log; convert via local slope
    # Use px/s band ≈ band_sigma * (v0 + median_u) for practical Hough width
    med_u = float(np.median(u[valid])) if valid.any() else v0
    band_px = float(band_sigma) * (v0 + med_u)

    t1 = time.perf_counter()
    # Zero-out invalid constraints
    u_use = u.copy()
    n_use = n_hat.copy()
    u_use[~valid] = 0.0
    n_use[~valid] = 0.0
    C = build_constraint_matrix(
        u_use, n_use, grid_coords,
        band_sigma=band_px, weight_floor=weight_floor, chunk=chunk,
    )
    t_C = time.perf_counter() - t1

    t2 = time.perf_counter()
    V, t_bundle = bundle_constraints(
        C, edge_index_list, n, self_weight=self_weight,
    )
    # Z = V @ codebook
    Z = V @ codebook  # [N, D_vel]
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(norms, 1e-12)
    t_Z = time.perf_counter() - t2

    t3 = time.perf_counter()
    v_res, conf, resolved = readout_velocity(
        Z, codebook, grid_coords, topk=topk, min_conf=min_conf, chunk=chunk,
    )
    resolved = resolved & valid
    t_read = time.perf_counter() - t3

    vx = np.where(resolved, v_res[:, 0], vx_perp)
    vy = np.where(resolved, v_res[:, 1], vy_perp)

    extra = {
        "D_VEL": D_vel,
        "VEL_GRID_N": grid_n,
        "G": int(grid_coords.shape[0]),
        "BAND_SIGMA": band_sigma,
        "band_px": band_px,
        "CLEANUP_TOPK": topk,
        "CLEANUP_MIN_CONF": min_conf,
        "confidence": conf,
        "t_codebook_s": t_cb,
        "t_constraint_s": t_C,
        "t_bundle_s": t_bundle,
        "t_superpose_s": t_Z,
        "t_readout_s": t_read,
        "V_sparse": V if validate else None,
        "grid_coords": grid_coords if validate else None,
        "C_sparse": C if validate else None,
        "Z": Z if validate else None,
        "codebook": codebook if validate else None,
    }

    if validate:
        v_ref = readout_velocity_explicit(V, grid_coords, topk=topk)
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
              f"RMS={rms:.4g}  D_vel={D_vel} G={grid_coords.shape[0]}")
        print(f"[vsa] timings: C={t_C:.3f}s  bundle+Z={t_Z:.3f}s  "
              f"readout={t_read:.3f}s")

    return vx, vy, resolved, extra
