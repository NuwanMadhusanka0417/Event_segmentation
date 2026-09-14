"""VFA kernel Gram matrix and rank-r eigen basis."""

from __future__ import annotations

import numpy as np
import torch


def vfa_gram(N: int = 21, sigma_k: float = 1.5) -> torch.Tensor:
    n = N // 2
    c = torch.arange(-n, n + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * (np.sqrt(2) * sigma_k) ** 2))


def eigen_basis(
    N: int = 21,
    sigma_k: float = 1.5,
    r: int = 64,
) -> tuple[torch.Tensor, float]:
    G = vfa_gram(N, sigma_k)
    evals, evecs = torch.linalg.eigh(G)
    evals, evecs = evals.flip(0), evecs.flip(1)
    evals = evals.clamp_min(0)
    K = (evals[:r].sqrt()[:, None] * evecs[:, :r].T)
    return K.reshape(r, N, N).float(), (evals[:r].sum() / evals.sum()).item()


def random_vsa_gram(N: int, seed: int = 0) -> torch.Tensor:
    """Unsmoothed random spatial basis for VFA vs basic kernel comparison tests."""
    g = torch.Generator().manual_seed(seed)
    d = N * N
    W = torch.randn(d, d, generator=g, dtype=torch.float64)
    W = W / W.norm(dim=1, keepdim=True)
    return W @ W.T


def local_similarity_comparison(N: int = 21, sigma_k: float = 1.5, seed: int = 0) -> float:
    """Mean adjacent-pixel Gram similarity: VFA should exceed random."""
    G_vfa = vfa_gram(N, sigma_k)
    G_rand = random_vsa_gram(N, seed)
    idx = N // 2 * N + N // 2
    nbr = idx + 1
    vfa_sim = G_vfa[idx, nbr].item()
    rand_sim = G_rand[idx, nbr].item()
    return vfa_sim - rand_sim
