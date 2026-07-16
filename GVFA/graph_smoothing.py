"""
graph_smoothing.py — Label cleanup by graph-neighbour energy minimization.

Minimizes E = sum_i D_i(l_i) + lam * sum_ij 1[l_i != l_j]
with data term D_i(l) = 1 - cos(H_i, proto_l).

Label 0 (background) is protected only via the data term — no special casing.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_numpy(H):
    if isinstance(H, torch.Tensor):
        return H.detach().cpu().numpy().astype(np.float64)
    return np.asarray(H, dtype=np.float64)


def _normalize_rows(M):
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(nrm, 1e-12)


def compute_prototypes(H, labels):
    """L2-normalized mean HV per label id (including 0). Returns dict id->vec."""
    Hn = _normalize_rows(_as_numpy(H))
    protos = {}
    for lid in np.unique(labels):
        m = Hn[labels == lid].mean(axis=0)
        n = np.linalg.norm(m)
        protos[int(lid)] = (m / n) if n > 1e-12 else m
    return protos


def _merged_undirected_edges(edge_index_list):
    """Unique undirected pairs from a list of [2, E] tensors / arrays."""
    pairs = []
    for ei in edge_index_list:
        if ei is None:
            continue
        if isinstance(ei, torch.Tensor):
            if ei.numel() == 0:
                continue
            a = ei[0].numpy()
            b = ei[1].numpy()
        else:
            a = np.asarray(ei[0])
            b = np.asarray(ei[1])
            if a.size == 0:
                continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pairs.append(np.stack([lo, hi], axis=1))
    if not pairs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    uniq = np.unique(np.concatenate(pairs, axis=0), axis=0)
    return uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64)


def smooth_labels(labels, H, protos, edge_index_list, lam=1.5, n_iters=5):
    """ICM label smoothing: data term + lam * disagreeing neighbours.

    Parameters
    ----------
    labels : int [N]
    H : [N, D] hypervectors
    protos : dict {label_id: unit vector [D]} or None (recomputed each iter)
    edge_index_list : list of edge_index [2, E]
    lam, n_iters : smoothing strength / iterations

    Returns
    -------
    labels : int [N] (copy, possibly updated)
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    Hn = _normalize_rows(_as_numpy(H))
    n = len(labels)
    ei, ej = _merged_undirected_edges(edge_index_list)
    # undirected adjacency as two directed copies for voting
    if ei.size:
        src = np.concatenate([ei, ej])
        dst = np.concatenate([ej, ei])
    else:
        src = np.empty(0, np.int64)
        dst = np.empty(0, np.int64)

    for it in range(n_iters):
        if protos is None:
            P = compute_prototypes(Hn, labels)
        else:
            P = protos
            # refresh means lightly from current labels
            P = compute_prototypes(Hn, labels)

        ids = sorted(P.keys())
        if not ids:
            break
        id_to_col = {lid: c for c, lid in enumerate(ids)}
        K = len(ids)
        Pmat = np.stack([P[lid] for lid in ids], axis=0)  # [K, D]

        # data cost [N, K]: 1 - cos
        cos = Hn @ Pmat.T
        data = 1.0 - cos

        # neighbour disagreement votes: for each node, count of neighbours per label
        # cost_pair[i,k] = lam * (#nbrs whose label != ids[k])
        #               = lam * (deg_i - count_nbrs_with_label_k)
        deg = np.zeros(n, dtype=np.float64)
        if src.size:
            np.add.at(deg, dst, 1.0)
            counts = np.zeros((n, K), dtype=np.float64)
            nbr_lab = labels[src]
            # only count known labels
            for lid, col in id_to_col.items():
                mask = nbr_lab == lid
                if mask.any():
                    np.add.at(counts[:, col], dst[mask], 1.0)
            pair = lam * (deg[:, None] - counts)
        else:
            pair = np.zeros((n, K), dtype=np.float64)

        cost = data + pair
        best_col = np.argmin(cost, axis=1)
        new_labels = np.array([ids[c] for c in best_col], dtype=np.int64)
        n_flip = int((new_labels != labels).sum())
        labels = new_labels
        print(f"[smooth] iter {it + 1}/{n_iters}: flipped {n_flip} labels")
        if n_flip == 0:
            break

    return labels


def drop_tiny_clusters(labels, min_size, background=0):
    """Relabel clusters with fewer than min_size events to background."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    ids, counts = np.unique(labels, return_counts=True)
    for lid, cnt in zip(ids.tolist(), counts.tolist()):
        if lid == background:
            continue
        if cnt < min_size:
            labels[labels == lid] = background
    # compact remaining non-background ids to 1..K
    keep = labels != background
    if keep.any():
        _, compact = np.unique(labels[keep], return_inverse=True)
        labels[keep] = compact + 1  # 1..K
    return labels
