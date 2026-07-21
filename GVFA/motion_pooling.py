"""
motion_pooling.py — Training-free motion-coherent hierarchical pooling for GVFA.

Groups events into supernodes by local normal-flow agreement (Graclus / heavy-edge
matching), bundles member hypervectors, and unpools supernode labels back to events.

Motion (vx, vy) is used ONLY as the pooling criterion — never as a node feature.
"""

from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Motion affinity
# ----------------------------------------------------------------------------
def motion_affinity_weights(vx, vy, ei, ej, sigma_v=None, model_id=None):
    """w_ij = exp(-||v_i - v_j||^2 / (2 sigma_v^2)) for each undirected edge.

    If sigma_v is None, use the median of ||v_i - v_j|| over edges (printed by caller).
    If model_id is given, hard-gate: w_ij = 0 whenever model_id[i] != model_id[j]
    (different independently-moving-object models must never merge).
    Returns (weights [E], sigma_v_used).
    """
    dv = np.hypot(vx[ei] - vx[ej], vy[ei] - vy[ej])
    if sigma_v is None:
        pos = dv[dv > 0]
        sigma_v = float(np.median(pos)) if pos.size else 1.0
        sigma_v = max(sigma_v, 1e-6)
    w = np.exp(-(dv * dv) / (2.0 * sigma_v * sigma_v))
    if model_id is not None:
        mid = np.asarray(model_id)
        w = np.where(mid[ei] == mid[ej], w, 0.0)
    return w.astype(np.float64), float(sigma_v)


def _undirected_edge_union(edge_a, edge_b):
    """Merge two directed [2, E] edge_index tensors into unique undirected pairs."""
    pairs = []
    for ei in (edge_a, edge_b):
        if ei is None or ei.numel() == 0:
            continue
        rec = ei[0].numpy()
        src = ei[1].numpy()
        a = np.minimum(rec, src)
        b = np.maximum(rec, src)
        pairs.append(np.stack([a, b], axis=1))
    if not pairs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    all_pairs = np.concatenate(pairs, axis=0)
    uniq = np.unique(all_pairs, axis=0)
    return uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64)


def induce_subgraph(edge_index, node_mask):
    """Keep edges whose both ends are in node_mask; remap endpoints to 0..n_sub-1.

    Returns
        edge_sub : LongTensor [2, E'] (empty if none)
        sub_idx  : int [n_sub] original indices of kept nodes
        old_to_new : int [N] map original -> new (-1 if not kept)
    """
    node_mask = np.asarray(node_mask, dtype=bool)
    n = len(node_mask)
    sub_idx = np.where(node_mask)[0].astype(np.int64)
    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[sub_idx] = np.arange(len(sub_idx), dtype=np.int64)
    if edge_index is None or (
            isinstance(edge_index, torch.Tensor) and edge_index.numel() == 0):
        return torch.zeros(2, 0, dtype=torch.long), sub_idx, old_to_new
    rec = edge_index[0].numpy()
    src = edge_index[1].numpy()
    keep = node_mask[rec] & node_mask[src]
    if not keep.any():
        return torch.zeros(2, 0, dtype=torch.long), sub_idx, old_to_new
    rec_n = old_to_new[rec[keep]]
    src_n = old_to_new[src[keep]]
    edge_sub = torch.from_numpy(np.stack([rec_n, src_n], axis=0)).long()
    return edge_sub, sub_idx, old_to_new


# ----------------------------------------------------------------------------
# Graclus / heavy-edge matching
# ----------------------------------------------------------------------------
def _graclus_one_level(n, ei, ej, w, w_min, rng):
    """One Graclus matching pass. Returns cluster_id [n] in 0..C-1 and n_rejected."""
    # adjacency: list of (nbr, weight)
    adj = [[] for _ in range(n)]
    for a, b, wij in zip(ei.tolist(), ej.tolist(), w.tolist()):
        if a == b:
            continue
        adj[a].append((b, wij))
        adj[b].append((a, wij))

    matched = np.full(n, -1, dtype=np.int64)
    order = rng.permutation(n)
    n_rejected = 0
    next_id = 0

    for i in order:
        if matched[i] >= 0:
            continue
        best_j, best_w = -1, -1.0
        best_gated = -1.0  # heaviest unmatched nbr below w_min
        for j, wij in adj[i]:
            if matched[j] >= 0:
                continue
            if wij < w_min:
                if wij > best_gated:
                    best_gated = wij
                continue
            if wij > best_w:
                best_w, best_j = wij, j
        if best_j >= 0:
            matched[i] = next_id
            matched[best_j] = next_id
            next_id += 1
        else:
            # had motion-disagreeing neighbours but gate refused every merge
            if best_gated >= 0.0:
                n_rejected += 1
            matched[i] = next_id
            next_id += 1

    return matched, n_rejected


def _aggregate_model_id(model_id, cluster_id, n_super):
    """Majority model_id per supernode (members share id when gate is on)."""
    out = np.full(n_super, -1, dtype=np.int64)
    mid = np.asarray(model_id)
    for c in range(n_super):
        members = mid[cluster_id == c]
        if members.size == 0:
            continue
        vals, counts = np.unique(members, return_counts=True)
        out[c] = int(vals[np.argmax(counts)])
    return out


def _aggregate_velocities(vx, vy, cluster_id, n_super):
    """Mean (vx, vy) per supernode."""
    vx_s = np.zeros(n_super, dtype=np.float64)
    vy_s = np.zeros(n_super, dtype=np.float64)
    cnt = np.zeros(n_super, dtype=np.float64)
    np.add.at(vx_s, cluster_id, vx)
    np.add.at(vy_s, cluster_id, vy)
    np.add.at(cnt, cluster_id, 1.0)
    cnt = np.maximum(cnt, 1.0)
    return vx_s / cnt, vy_s / cnt


def _project_edges(ei, ej, cluster_id):
    """Map fine undirected edges onto coarse supernode pairs (drop self-loops)."""
    a = cluster_id[ei]
    b = cluster_id[ej]
    keep = a != b
    if not keep.any():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a, b = a[keep], b[keep]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    uniq = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64)


def motion_coarsen(n_events, edge_spatial, edge_temporal, vx, vy,
                   n_levels=5, sigma_v=None, w_min=0.3, seed=0,
                   model_id=None, velocity_name="residual"):
    """Multilevel motion-coherent Graclus coarsening.

    Parameters
    ----------
    vx, vy : velocity field used for affinity (must be residual on IMO subset)
    model_id : optional int [N]; hard-gates merges across different models
    velocity_name : label printed for verification (expect \"residual\")

    Returns
        cluster_id : int [N] event -> supernode index in 0..C-1
        info       : dict with per-level counts, rejections, sigma_v, sizes
    """
    assert len(vx) == n_events and len(vy) == n_events, (
        f"[coarsen] velocity length mismatch: n={n_events} "
        f"vx={len(vx)} vy={len(vy)}"
    )
    if model_id is not None:
        assert len(model_id) == n_events, (
            f"[coarsen] model_id length {len(model_id)} != n_events {n_events}"
        )

    print(f"[coarsen] VERIFY: coarsening {n_events} nodes with velocity="
          f"'{velocity_name}' (expect residual on IMO subset)"
          + (f"; model_id gate ON ({len(np.unique(model_id))} ids)"
             if model_id is not None else "; model_id gate OFF"))

    ei0, ej0 = _undirected_edge_union(edge_spatial, edge_temporal)
    if ei0.size == 0:
        print("[coarsen] no edges — every event is its own supernode")
        return np.arange(n_events, dtype=np.int64), {
            "sigma_v": float(sigma_v) if sigma_v is not None else 1.0,
            "levels": [n_events],
            "n_rejected_total": 0,
            "sizes": np.ones(n_events, dtype=np.int64),
            "w_min": w_min,
            "n_levels": n_levels,
            "C": n_events,
            "velocity_name": velocity_name,
        }

    # sigma_v from fine-level edge motion distances
    w0, sigma_v_used = motion_affinity_weights(
        vx, vy, ei0, ej0, sigma_v, model_id=model_id)
    print(f"[coarsen] sigma_v = {sigma_v_used:.6g}  "
          f"({'auto=median edge ||v_i-v_j||' if sigma_v is None else 'user'})  "
          f"w_min={w_min}  levels={n_levels}")

    rng = np.random.default_rng(seed)
    # event -> current-level node
    event_to_node = np.arange(n_events, dtype=np.int64)
    vx_c, vy_c = vx.copy(), vy.copy()
    mid_c = None if model_id is None else np.asarray(model_id).copy()
    ei, ej = ei0, ej0
    level_counts = [n_events]
    n_rejected_total = 0

    for lev in range(n_levels):
        n_cur = int(event_to_node.max()) + 1
        w, _ = motion_affinity_weights(
            vx_c, vy_c, ei, ej, sigma_v_used, model_id=mid_c)
        match, n_rej = _graclus_one_level(n_cur, ei, ej, w, w_min, rng)
        n_rejected_total += n_rej
        n_next = int(match.max()) + 1
        event_to_node = match[event_to_node]
        level_counts.append(n_next)
        print(f"[coarsen] level {lev + 1}/{n_levels}: {n_cur} -> {n_next} nodes  "
              f"(rejected matches this level: {n_rej})")
        if n_next >= n_cur or n_next <= 1:
            print(f"[coarsen] stopping early (no further reduction)")
            break
        vx_c, vy_c = _aggregate_velocities(vx, vy, event_to_node, n_next)
        if mid_c is not None:
            mid_c = _aggregate_model_id(model_id, event_to_node, n_next)
        ei, ej = _project_edges(ei0, ej0, event_to_node)

    C = int(event_to_node.max()) + 1
    sizes = np.bincount(event_to_node, minlength=C)
    print(f"[coarsen] final supernodes: {C}  "
          f"size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}  "
          f"total w_min rejections (across levels): {n_rejected_total}")
    info = {
        "sigma_v": sigma_v_used,
        "levels": level_counts,
        "n_rejected_total": n_rejected_total,
        "sizes": sizes,
        "w_min": w_min,
        "n_levels": n_levels,
        "C": C,
        "velocity_name": velocity_name,
    }
    return event_to_node.astype(np.int64), info


# ----------------------------------------------------------------------------
# Bundling + aggregates + unpool
# ----------------------------------------------------------------------------
def bundle_hypervectors(H, cluster_id):
    """H_super[c] = normalize(sum_{i in c} H[i]). H: torch [N, D] or ndarray."""
    if isinstance(H, torch.Tensor):
        H_np = H.detach().cpu().numpy().astype(np.float64)
    else:
        H_np = np.asarray(H, dtype=np.float64)
    C = int(cluster_id.max()) + 1
    D = H_np.shape[1]
    acc = np.zeros((C, D), dtype=np.float64)
    np.add.at(acc, cluster_id, H_np)
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = acc / np.maximum(norms, 1e-12)
    return torch.from_numpy(acc.astype(np.float32))


def supernode_aggregates(x, y, t, p, vx, vy, cluster_id):
    """Mean (x,y,t,p,vx,vy) and member counts per supernode."""
    C = int(cluster_id.max()) + 1
    counts = np.bincount(cluster_id, minlength=C).astype(np.int64)

    def _mean(arr):
        out = np.zeros(C, dtype=np.float64)
        np.add.at(out, cluster_id, arr.astype(np.float64))
        return out / np.maximum(counts, 1)

    return {
        "x": _mean(x),
        "y": _mean(y),
        "t": _mean(t),
        "p": _mean(p),
        "vx": _mean(vx),
        "vy": _mean(vy),
        "counts": counts,
    }


def unpool(labels_super, cluster_id):
    """event_labels = labels_super[cluster_id]."""
    return labels_super[cluster_id]


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def diagnose_node_flow(vx, vy, label="node_flow"):
    """Print fraction valid, median |v|, percentiles of vx/vy."""
    spd = np.hypot(vx, vy)
    valid = spd > 1e-12
    frac = float(valid.mean())
    med = float(np.median(spd[valid])) if valid.any() else 0.0
    print(f"[{label}] called — valid flow: {frac * 100:.1f}% of events  "
          f"median |v|={med:.4g} px/s")
    if valid.any():
        for name, arr in (("vx", vx[valid]), ("vy", vy[valid])):
            qs = np.percentile(arr, [5, 25, 50, 75, 95])
            print(f"[{label}] {name} p5/p25/p50/p75/p95: "
                  f"{qs[0]:.4g} {qs[1]:.4g} {qs[2]:.4g} {qs[3]:.4g} {qs[4]:.4g}")
    else:
        print(f"[{label}] WARNING: no valid flow estimates")


def mean_pairwise_cosine(H, n_sample=256, seed=0):
    """Mean pairwise cosine of a random sample of rows of H (assumed roughly unit)."""
    if isinstance(H, torch.Tensor):
        Hn = H.detach().cpu().numpy().astype(np.float64)
    else:
        Hn = np.asarray(H, dtype=np.float64)
    n = Hn.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    S = Hn[idx]
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    G = S @ S.T
    m = G.shape[0]
    iu = np.triu_indices(m, k=1)
    return float(G[iu].mean()) if iu[0].size else float("nan")


def diagnose_bundling_cosine(H_super, seed=0, label="bundling"):
    """Mean pairwise cosine before/after mean-centering; warn if collapsed."""
    before = mean_pairwise_cosine(H_super, seed=seed)
    if isinstance(H_super, torch.Tensor):
        Hc = H_super - H_super.mean(dim=0, keepdim=True)
    else:
        Hc = H_super - H_super.mean(axis=0, keepdims=True)
    after = mean_pairwise_cosine(Hc, seed=seed)
    print(f"[{label}] mean pairwise cosine BEFORE mean-center: {before:.4f}")
    print(f"[{label}] mean pairwise cosine AFTER  mean-center: {after:.4f}")
    if before > 0.9:
        print(f"[{label}] *** COLLAPSED REPRESENTATION *** "
              "mean pairwise cosine before centering > 0.9 — "
              "supernode HVs are nearly identical; clustering will struggle.")
    return before, after


def diagnose_clustering(H_super, labels_super):
    """Print cluster count, sizes, and prototype pairwise cosine matrix."""
    if isinstance(H_super, torch.Tensor):
        Hn = H_super.detach().cpu().numpy().astype(np.float64)
    else:
        Hn = np.asarray(H_super, dtype=np.float64)
    ids = np.unique(labels_super)
    ids_fg = ids[ids >= 0]
    print(f"[clustering] {len(ids_fg)} clusters "
          f"(+ {int((labels_super < 0).sum())} background supernodes)")
    for oid in ids:
        n = int((labels_super == oid).sum())
        tag = "  <- background" if oid < 0 else ""
        print(f"[clustering]   id {oid:3d}: {n:5d} supernodes{tag}")

    if len(ids_fg) == 0:
        return
    # prototype = L2-normalized mean of (mean-centered) member HVs
    Hc = Hn - Hn.mean(axis=0, keepdims=True)
    protos = []
    for oid in ids_fg:
        m = Hc[labels_super == oid].mean(axis=0)
        nrm = np.linalg.norm(m) + 1e-12
        protos.append(m / nrm)
    P = np.stack(protos, axis=0)
    G = P @ P.T
    print("[clustering] proto cosine matrix (rows/cols = cluster ids "
          f"{ids_fg.tolist()}):")
    with np.printoptions(precision=3, suppress=True):
        print(G)


def save_supernodes_png(x, y, cluster_id, path="supernodes.png", title=None):
    """Colour events by coarsening cluster_id (before object clustering)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = int(cluster_id.max()) + 1
    fig, ax = plt.subplots(figsize=(9, 7))
    # hash cluster ids into a cyclic colormap so adjacent ids differ
    colors = (cluster_id.astype(np.int64) * 2654435761) % (2 ** 32)
    sc = ax.scatter(x, y, c=colors, s=2, cmap="gist_ncar", linewidths=0)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"coarsening supernodes (C={C})")
    fig.colorbar(sc, ax=ax, label="hashed cluster_id")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}  ({C} supernodes)")
