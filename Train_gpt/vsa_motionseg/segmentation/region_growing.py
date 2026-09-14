"""CPU-friendly region growing on dynamic pixels."""

from __future__ import annotations

import torch


def region_growing(
    dynamic_mask: torch.Tensor,
    residual_flow: torch.Tensor,
    hv_field: torch.Tensor,
    *,
    flow_threshold: float = 2.0,
    hv_threshold: float = 0.3,
    neighborhood: int = 8,
    min_cluster_size: int = 20,
) -> torch.Tensor:
    """
    Returns segment_id map (H,W), 0=background.
    hv_field: (d,H,W) complex.
    """
    H, W = dynamic_mask.shape
    labels = torch.zeros(H, W, dtype=torch.int64)
    if not dynamic_mask.any():
        return labels

    d = hv_field.shape[0]
    hv = hv_field.reshape(d, -1).T
    hv = hv / (hv.abs().norm(dim=1, keepdim=True) + 1e-8)
    flow = residual_flow.reshape(2, -1).T

    dyn_idx = torch.where(dynamic_mask.reshape(-1))[0]
    visited = torch.zeros(H * W, dtype=torch.bool)
    current_label = 0
    offsets = _neighbors(neighborhood)

    for start in dyn_idx.tolist():
        if visited[start]:
            continue
        current_label += 1
        stack = [start]
        cluster = []
        while stack:
            p = stack.pop()
            if visited[p]:
                continue
            visited[p] = True
            cluster.append(p)
            y, x = p // W, p % W
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                q = ny * W + nx
                if not dynamic_mask.view(-1)[q] or visited[q]:
                    continue
                fd = (flow[p] - flow[q]).norm().item()
                if fd > flow_threshold:
                    continue
                sim = (hv[p].conj() * hv[q]).sum().real.item()
                if sim < hv_threshold:
                    continue
                stack.append(q)
        if len(cluster) < min_cluster_size:
            current_label -= 1
            for p in cluster:
                visited[p] = False
            continue
        for p in cluster:
            labels.view(-1)[p] = current_label
    return labels


def _neighbors(n: int) -> list[tuple[int, int]]:
    if n == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
