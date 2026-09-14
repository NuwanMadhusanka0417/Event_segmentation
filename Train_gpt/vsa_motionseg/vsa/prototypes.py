"""HDC prototype training and inference."""

from __future__ import annotations

from pathlib import Path

import torch

from vsa_motionseg.vsa.fpe import similarity


def train_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    bipolar: bool = True,
    balance: bool = True,
) -> torch.Tensor:
    """
    features: (N, d) complex or real
    labels: (N,) int 0=static, 1=dynamic
    """
    K = 2
    d = features.shape[-1]
    is_complex = features.is_complex()
    protos = []
    for k in range(K):
        mask = labels == k
        if balance and k == 0 and mask.sum() > labels.numel() // 2:
            idx = torch.where(mask)[0]
            n_dyn = (labels == 1).sum().item()
            if n_dyn > 0 and idx.numel() > 2 * n_dyn:
                perm = torch.randperm(idx.numel())[: 2 * int(n_dyn)]
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[idx[perm]] = True
        if not mask.any():
            protos.append(torch.zeros(d, dtype=features.dtype, device=features.device))
            continue
        bundle = features[mask].sum(dim=0)
        if bipolar:
            if is_complex:
                bundle = torch.sign(bundle.real).to(torch.float32)
            else:
                bundle = torch.sign(bundle)
        else:
            bundle = bundle / (bundle.abs().norm() + 1e-8)
        protos.append(bundle)
    return torch.stack(protos, dim=0)


def predict_prototypes(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns pred (N,) and confidence (N,)."""
    if features.is_complex():
        scores = torch.stack(
            [similarity(features, prototypes[k].expand_as(features), dim=-1) for k in range(2)],
            dim=-1,
        )
    else:
        fn = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        pn = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-8)
        scores = fn @ pn.T
    conf, pred = scores.max(dim=-1)
    pred = pred.clone()
    pred[conf < threshold] = -1
    return pred, conf


def save_prototypes(path: str | Path, prototypes: torch.Tensor, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototypes, "meta": meta or {}}, path)


def load_prototypes(path: str | Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["prototypes"]
