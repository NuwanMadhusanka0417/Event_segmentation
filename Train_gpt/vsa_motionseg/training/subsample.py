"""Subsample labeled pixel features for prototype training."""

from __future__ import annotations

import torch


def subsample_labeled(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int | None,
    *,
    stratified: bool = True,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cap the number of rows in (features, labels).

    features : (N, d)
    labels   : (N,) int 0/1
    """
    n = features.shape[0]
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return features, labels

    gen = torch.Generator().manual_seed(seed)
    if not stratified:
        pick = torch.randperm(n, generator=gen)[:max_samples]
        return features[pick], labels[pick]

    idx0 = torch.where(labels == 0)[0]
    idx1 = torch.where(labels == 1)[0]
    if idx0.numel() == 0 or idx1.numel() == 0:
        pick = torch.randperm(n, generator=gen)[:max_samples]
        return features[pick], labels[pick]

    n0 = min(idx0.numel(), max_samples // 2)
    n1 = min(idx1.numel(), max_samples - n0)
    n0 = min(idx0.numel(), max_samples - n1)
    pick0 = idx0[torch.randperm(idx0.numel(), generator=gen)[:n0]]
    pick1 = idx1[torch.randperm(idx1.numel(), generator=gen)[:n1]]
    pick = torch.cat([pick0, pick1])
    pick = pick[torch.randperm(pick.numel(), generator=gen)]
    return features[pick], labels[pick]
