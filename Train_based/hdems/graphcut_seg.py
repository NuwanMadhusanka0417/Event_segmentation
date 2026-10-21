"""Region-level motion segmentation of the events of one window by multi-model graph cuts.

Why (2026-10-21): with the SAME flow, thresholding the motion cue scores ~36% event IoU on
the Hua et al. benchmark (VSA 35.9, VecKM 36.1), our per-pixel U-Net 46.8, and Hua et al.'s
region + temporal pipeline 77.4. The gap is in the decision stage after the flow, not in
the flow. This module is that decision stage, in the form of EMSGC (Zhou et al. 2021,
arXiv 2012.08730) and EvMotionSeg (Zhong et al., ICRA 2026), without ROS:

  nodes      the pixels that fired in the window (full sensor resolution)
  edges      k nearest neighbouring pixels, weight exp(-d^2 / 2 s^2)
  flow       per node, constraint rows  D_r . u(x) = b_r  in sensor px per 12.5 ms (toward
             EARLIER, rotation already removed): a full flow vector gives 2 rows (VSA), a
             normal flow 1 row per event (VecKM, several per pixel)
  models     affine motions u(x) = [a0 + a1 X + a2 Y, a3 + a4 X + a5 Y]; label 0 = the static
             background, started at "no motion" (the IMU removed the rotation) and kept
             near it with a ridge term; ~n_init more from spatial k-means clusters (the
             over-segmentation of Hua et al.)
  energy     sum_i min(mean_r residual^2 / sigma^2, trunc)  +  lam * sum_ij w_ij [l_i != l_j]
  solver     alpha-expansion (PyMaxflow), alternated with refitting each model on its
             nodes; models with fewer than min_nodes nodes are dropped (label cost)
  decision   a node moves if its model differs from the background model by more than
             delta px (mean over the region) -- ``motion_score`` returns that difference,
             so delta is chosen afterwards on validation

The result is region-level: an object is moving or static as a whole, not pixel by pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GCParams:
    k: int = 8                  # neighbours per node
    max_link_px: float = 4.0    # no edge longer than this (sensor px)
    edge_sigma_px: float = 2.0  # edge weight exp(-d^2 / 2 s^2)
    lam: float = 1.0            # smoothness weight
    sigma_px: float = 0.5       # flow noise (sensor px per 12.5 ms)
    trunc: float = 4.0          # robust cap of the data cost
    n_init: int = 30            # initial motion models from spatial clusters
    iters: int = 4              # expansion + refit rounds
    min_nodes: int = 30         # smaller regions lose their model
    bg_ridge: float = 50.0      # keeps the background model near "no motion"
    seed: int = 0


def _basis(x: np.ndarray, y: np.ndarray, H: int, W: int) -> np.ndarray:
    s = max(H, W) / 2.0
    return np.stack([np.ones_like(x, dtype=np.float64), (x - W / 2.0) / s, (y - H / 2.0) / s], 1)


def _row_design(Bn: np.ndarray, row_node: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(R, 6) rows of D_r . u(x) for the affine model, u = [B a_x, B a_y]."""
    b = Bn[row_node]
    return np.concatenate([D[:, :1] * b, D[:, 1:2] * b], 1)


def _fit(G: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    A = G.T @ G + ridge * np.eye(G.shape[1])
    return np.linalg.solve(A, G.T @ b)


def _costs(G, b, row_node, n_nodes, thetas, p: GCParams) -> np.ndarray:
    """(N, L) robust data cost of every node under every model."""
    res = G @ thetas.T - b[:, None]                              # (R, L)
    cnt = np.bincount(row_node, minlength=n_nodes).astype(np.float64)
    sq = np.zeros((n_nodes, thetas.shape[0]))
    np.add.at(sq, row_node, res * res)
    mean = sq / np.maximum(cnt, 1)[:, None]
    return np.minimum(mean / (p.sigma_px ** 2), p.trunc)


def knn_graph(x: np.ndarray, y: np.ndarray, p: GCParams) -> tuple[np.ndarray, np.ndarray]:
    """Edges (E, 2) between neighbouring pixels and their weights (E,)."""
    from scipy.spatial import cKDTree
    pts = np.stack([x, y], 1).astype(np.float64)
    if len(pts) < 2:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    d, j = cKDTree(pts).query(pts, k=min(p.k + 1, len(pts)), distance_upper_bound=p.max_link_px)
    i = np.repeat(np.arange(len(pts)), j.shape[1])
    j, d = j.reshape(-1), d.reshape(-1)
    keep = (j < len(pts)) & (j != i)
    i, j, d = i[keep], j[keep], d[keep]
    a, b = np.minimum(i, j), np.maximum(i, j)
    e, idx = np.unique(np.stack([a, b], 1), axis=0, return_index=True)
    return e, np.exp(-d[idx] ** 2 / (2 * p.edge_sigma_px ** 2))


def alpha_expansion(cost: np.ndarray, edges: np.ndarray, w: np.ndarray, labels: np.ndarray,
                    sweeps: int = 1) -> np.ndarray:
    """Potts alpha-expansion (Boykov et al. 2001) with PyMaxflow; w already includes lam."""
    import maxflow
    N, L = cost.shape
    labels = labels.copy()
    i, j = (edges[:, 0], edges[:, 1]) if len(edges) else (np.zeros(0, int), np.zeros(0, int))

    def energy(lab):
        return float(cost[np.arange(N), lab].sum() + (w * (lab[i] != lab[j])).sum())

    e_cur = energy(labels)
    for _ in range(sweeps):
        for a in range(L):
            li, lj = labels[i], labels[j]
            A = w * (li != lj)                 # E(keep, keep)
            B = w * (li != a)                  # E(keep, alpha)
            C = w * (a != lj)                  # E(alpha, keep)
            # E = A + (C-A) x_i + (0-C) x_j + (B+C-A) (1-x_i) x_j,   x = 1: switch to alpha
            u0 = cost[np.arange(N), labels].copy()
            u1 = cost[:, a] + np.bincount(i, C - A, N) + np.bincount(j, -C, N)
            m = np.minimum(u0, u1)
            g = maxflow.Graph[float]()
            nodes = g.add_nodes(N)
            if len(edges):
                g.add_edges(i, j, np.maximum(B + C - A, 0.0), np.zeros(len(i)))
            g.add_grid_tedges(nodes, u1 - m, u0 - m)
            g.maxflow()
            switch = g.get_grid_segments(nodes)
            cand = np.where(switch, a, labels)
            e_new = energy(cand)
            if e_new < e_cur - 1e-9:
                labels, e_cur = cand, e_new
    return labels


def segment(x: np.ndarray, y: np.ndarray, row_node: np.ndarray, D: np.ndarray, b: np.ndarray,
            H: int, W: int, p: GCParams = GCParams()) -> dict:
    """Nodes = pixels (x, y); constraint rows (row_node, D, b). -> {"labels" (N,) region id
    (0 = background), "motion_score" (N,) |model - background model| in sensor px per 12.5 ms
    (mean over the region), "thetas", "edges"}."""
    N = len(x)
    out = {"labels": np.zeros(N, np.int64), "motion_score": np.zeros(N), "thetas": np.zeros((1, 6))}
    if N == 0:
        return out
    Bn = _basis(x.astype(np.float64), y.astype(np.float64), H, W)
    G = _row_design(Bn, row_node, D)
    edges, w = knn_graph(x, y, p)
    w = p.lam * w

    # initial models: background (zero motion) + one per spatial cluster
    thetas = [np.zeros(6)]
    if N >= 2 * p.min_nodes:
        from scipy.cluster.vq import kmeans2
        k0 = int(min(p.n_init, max(1, N // p.min_nodes)))
        _c, lab0 = kmeans2(np.stack([x, y], 1).astype(np.float64), k0, minit="++", seed=p.seed)
        node_rows = lab0[row_node]
        for c in range(k0):
            sel = node_rows == c
            if sel.sum() >= 12:
                thetas.append(_fit(G[sel], b[sel], 1e-3))
    thetas = np.stack(thetas)
    labels = np.argmin(_costs(G, b, row_node, N, thetas, p), 1)

    for _ in range(p.iters):
        cost = _costs(G, b, row_node, N, thetas, p)
        labels = alpha_expansion(cost, edges, w, labels)
        # refit, drop small regions (label 0 is always kept)
        keep, new = [], []
        for l in range(thetas.shape[0]):
            nodes_l = labels == l
            if l != 0 and nodes_l.sum() < p.min_nodes:
                continue
            sel = nodes_l[row_node]
            if sel.sum() >= 6:
                new.append(_fit(G[sel], b[sel], p.bg_ridge if l == 0 else 1e-3))
            else:
                new.append(thetas[l])
            keep.append(l)
        thetas = np.stack(new)
        remap = {old: k for k, old in enumerate(keep)}
        cost = _costs(G, b, row_node, N, thetas, p)
        labels = np.array([remap.get(int(l), -1) for l in labels])
        lost = labels < 0
        if lost.any():
            labels[lost] = np.argmin(cost[lost], 1)

    # how different each region's motion is from the background, over its own pixels
    def predict(th):
        return np.stack([Bn @ th[:3], Bn @ th[3:]], 1)
    bg = predict(thetas[0])
    score = np.zeros(N)
    for l in range(1, thetas.shape[0]):
        nodes_l = labels == l
        if nodes_l.any():
            score[nodes_l] = float(np.linalg.norm(predict(thetas[l])[nodes_l] - bg[nodes_l], axis=1).mean())
    out.update(labels=labels, motion_score=score, thetas=thetas, edges=edges)
    return out
