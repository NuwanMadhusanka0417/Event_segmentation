"""AEGNN-style spatial voxel max-pooling on hypervectors (post-GVFA, pre-head)."""

import torch

POOL_CELL = 4


def voxel_max_pool(H: torch.Tensor, pos: torch.Tensor, cell: float = POOL_CELL):
    """Max-pool nodes in (x,y) voxels. Returns H_p [M,D], pos_p [M,2] (centroids)."""
    n = H.shape[0]
    if n == 0:
        return H, pos
    ix = torch.floor(pos[:, 0] / cell).long()
    iy = torch.floor(pos[:, 1] / cell).long()
    keys = ix * 100_000 + iy
    uniq, inv = torch.unique(keys, return_inverse=True)
    m = uniq.shape[0]
    H_p = torch.empty(m, H.shape[1], device=H.device, dtype=H.dtype)
    pos_p = torch.empty(m, 2, device=H.device, dtype=pos.dtype)
    for ui in range(m):
        sel = inv == ui
        H_p[ui] = H[sel].max(dim=0).values
        pos_p[ui] = pos[sel].mean(dim=0)
    return H_p, pos_p
