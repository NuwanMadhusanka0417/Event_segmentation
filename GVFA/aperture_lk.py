"""
aperture_lk.py — Track A: classical aperture resolvers (LK + affine).

Vectorized Lucas-Kanade-style intersection of normal-flow constraints over
graph neighbourhoods, with Harris/LK conditioning gate and Huber IRLS.
"""

from __future__ import annotations

import numpy as np

from aperture import undirected_edges_from_list


def _accumulate_lk(src, dst, nx, ny, u, w, n):
    """Accumulate 2x2 normal equations at dst from neighbour src with weights w."""
    M00 = np.zeros(n, dtype=np.float64)
    M01 = np.zeros(n, dtype=np.float64)
    M11 = np.zeros(n, dtype=np.float64)
    b0 = np.zeros(n, dtype=np.float64)
    b1 = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    if src.size == 0:
        return M00, M01, M11, b0, b1, cnt

    wx = w * nx[src]
    wy = w * ny[src]
    np.add.at(M00, dst, wx * nx[src])
    np.add.at(M01, dst, wx * ny[src])
    np.add.at(M11, dst, wy * ny[src])
    np.add.at(b0, dst, w * u[src] * nx[src])
    np.add.at(b1, dst, w * u[src] * ny[src])
    np.add.at(cnt, dst, 1.0)
    return M00, M01, M11, b0, b1, cnt


def _solve_2x2(M00, M01, M11, b0, b1):
    """Closed-form 2x2 solve + smaller eigenvalue (Harris/LK test)."""
    det = M00 * M11 - M01 * M01
    # eigenvalues of [[a,b],[b,c]]: (tr ± sqrt(tr^2 - 4 det))/2
    tr = M00 + M11
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    lam_min = 0.5 * (tr - np.sqrt(disc))
    vx = np.zeros_like(M00)
    vy = np.zeros_like(M00)
    ok = np.abs(det) > 1e-18
    vx[ok] = (M11[ok] * b0[ok] - M01[ok] * b1[ok]) / det[ok]
    vy[ok] = (M00[ok] * b1[ok] - M01[ok] * b0[ok]) / det[ok]
    return vx, vy, lam_min, ok


def resolve_flow_lk(x, y, u, n_hat, valid, edge_index_list, *,
                    vx_perp, vy_perp,
                    min_support=6, min_eig=1e-3, huber_iters=3):
    """Constant local flow by weighted LS on n_hat·v = u (vectorized)."""
    n = len(u)
    nx, ny = n_hat[:, 0], n_hat[:, 1]
    src, dst = undirected_edges_from_list(edge_index_list, n)

    # only use valid constraint neighbours
    if src.size:
        keep = valid[src] & valid[dst]
        src, dst = src[keep], dst[keep]

    w = np.ones(src.shape[0], dtype=np.float64)
    vx = vx_perp.copy()
    vy = vy_perp.copy()
    resolved = np.zeros(n, dtype=bool)
    n_gate = 0

    for it in range(max(1, int(huber_iters))):
        M00, M01, M11, b0, b1, cnt = _accumulate_lk(src, dst, nx, ny, u, w, n)
        vx_s, vy_s, lam_min, ok_det = _solve_2x2(M00, M01, M11, b0, b1)
        resolved = valid & ok_det & (cnt >= min_support) & (lam_min >= min_eig)
        n_gate = int((valid & ok_det & (cnt >= min_support) & (lam_min < min_eig)).sum())
        vx = np.where(resolved, vx_s, vx_perp)
        vy = np.where(resolved, vy_s, vy_perp)

        if it + 1 >= huber_iters or src.size == 0:
            break
        # Huber reweight on residual of neighbour constraints at each edge
        # r_e = n_src · v_dst - u_src
        r = nx[src] * vx[dst] + ny[src] * vy[dst] - u[src]
        # robust scale
        mad = np.median(np.abs(r - np.median(r))) if r.size else 0.0
        delta = 1.4826 * max(mad, 1e-9)
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))

    extra = {
        "lk_min_support": min_support,
        "condition_min_eig": min_eig,
        "huber_iters": huber_iters,
        "n_conditioning_gate": n_gate,
        "frac_lk": float(resolved.mean()),
        "frac_affine": 0.0,
    }
    print(f"[lk] resolved={int(resolved.sum())}/{n}  "
          f"conditioning_gate={n_gate}  min_eig={min_eig}")
    return vx, vy, resolved, extra


def resolve_flow_affine(x, y, u, n_hat, valid, edge_index_list, *,
                        vx_perp, vy_perp,
                        min_support=12, min_eig=1e-3, huber_iters=3,
                        lk_min_support=6):
    """6-param local affine flow; fall back to LK when under-determined."""
    # First get LK as fallback
    vx_lk, vy_lk, resolved_lk, extra_lk = resolve_flow_lk(
        x, y, u, n_hat, valid, edge_index_list,
        vx_perp=vx_perp, vy_perp=vy_perp,
        min_support=lk_min_support, min_eig=min_eig, huber_iters=huber_iters,
    )

    n = len(u)
    nx, ny = n_hat[:, 0], n_hat[:, 1]
    src, dst = undirected_edges_from_list(edge_index_list, n)
    if src.size:
        keep = valid[src] & valid[dst]
        src, dst = src[keep], dst[keep]

    # Relative coords of neighbour (src) w.r.t. node (dst)
    dx = x[src] - x[dst]
    dy = y[src] - y[dst]
    # Equation: nx*(a0 + a1*dx + a2*dy) + ny*(a3 + a4*dx + a5*dy) = u
    # Feature row phi = [nx, nx*dx, nx*dy, ny, ny*dx, ny*dy]
    phi = np.stack([
        nx[src],
        nx[src] * dx,
        nx[src] * dy,
        ny[src],
        ny[src] * dx,
        ny[src] * dy,
    ], axis=1)  # [E, 6]
    rhs = u[src]

    w = np.ones(src.shape[0], dtype=np.float64)
    used_affine = np.zeros(n, dtype=bool)
    vx = vx_lk.copy()
    vy = vy_lk.copy()

    for it in range(max(1, int(huber_iters))):
        # Accumulate M[N,6,6] and b[N,6]
        M = np.zeros((n, 6, 6), dtype=np.float64)
        b = np.zeros((n, 6), dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        if src.size:
            wp = w[:, None] * phi  # [E, 6]
            # M += phi^T diag(w) phi  via outer products
            for p in range(6):
                for q in range(p, 6):
                    contrib = wp[:, p] * phi[:, q]
                    np.add.at(M[:, p, q], dst, contrib)
                    if p != q:
                        np.add.at(M[:, q, p], dst, contrib)
                np.add.at(b[:, p], dst, w * phi[:, p] * rhs)
            np.add.at(cnt, dst, 1.0)

        # Conditioning: smallest eigenvalue of each 6x6 (via eigh on valid candidates)
        cand = valid & (cnt >= min_support)
        n_aff = 0
        if cand.any():
            idx = np.where(cand)[0]
            # Batched solve where matrix is well-conditioned
            Mc = M[idx]
            bc = b[idx]
            # eigenvalues
            try:
                evals = np.linalg.eigvalsh(Mc)  # [K, 6]
                lam_min = evals[:, 0]
            except np.linalg.LinAlgError:
                lam_min = np.zeros(len(idx))
            good = lam_min >= min_eig
            # also check finite det-ish via condition
            solve_idx = idx[good]
            if solve_idx.size:
                try:
                    params = np.linalg.solve(M[solve_idx], b[solve_idx])  # [K,6]
                    # velocity at event location (dx=dy=0): (a0, a3)
                    vx[solve_idx] = params[:, 0]
                    vy[solve_idx] = params[:, 3]
                    used_affine[solve_idx] = True
                    n_aff = int(solve_idx.size)
                except np.linalg.LinAlgError:
                    pass

        if it + 1 >= huber_iters or src.size == 0:
            break
        # Huber: residual of affine model at neighbour
        # v_src_pred using dst params — approximate with constant (a0,a3) for weight
        r = nx[src] * vx[dst] + ny[src] * vy[dst] - u[src]
        mad = np.median(np.abs(r - np.median(r))) if r.size else 0.0
        delta = 1.4826 * max(mad, 1e-9)
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))

    resolved = used_affine | resolved_lk
    # where affine failed, keep LK result
    extra = {
        "lk_min_support": lk_min_support,
        "affine_min_support": min_support,
        "condition_min_eig": min_eig,
        "huber_iters": huber_iters,
        "n_conditioning_gate": extra_lk.get("n_conditioning_gate", 0),
        "frac_affine": float(used_affine.mean()),
        "frac_lk": float((resolved_lk & ~used_affine).mean()),
        "n_affine": int(used_affine.sum()),
        "n_lk_fallback": int((resolved_lk & ~used_affine).sum()),
    }
    print(f"[affine] used_affine={int(used_affine.sum())}  "
          f"lk_fallback={extra['n_lk_fallback']}  resolved={int(resolved.sum())}")
    return vx, vy, resolved, extra
