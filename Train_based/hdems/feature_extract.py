"""Extract frozen encoder features for ridge fitting."""

from __future__ import annotations

import torch

from hdems.models.hdems import HDEMS
from hdems.seg_features import flatten_valid_features, prepare_seg_features


@torch.no_grad()
def extract_phi(model: HDEMS, surface: torch.Tensor) -> torch.Tensor:
    if surface.dim() == 3:
        surface = surface.unsqueeze(0)
    f = model.encoder(surface)
    return model.matcher([f])[0]


@torch.no_grad()
def extract_flat_batch(
    model: HDEMS,
    surface: torch.Tensor,
    mask: torch.Tensor,
    *,
    feature_mean: torch.Tensor | None,
    mean_center: bool,
    motion_features: bool,
    event_threshold: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    phi = extract_phi(model, surface)
    feats = prepare_seg_features(
        phi,
        surface,
        feature_mean=feature_mean,
        mean_center=mean_center,
        motion_features=motion_features,
    )
    return flatten_valid_features(
        feats, mask, surface, event_threshold=event_threshold,
    )


@torch.no_grad()
def accumulate_feature_mean(
    model: HDEMS,
    loader,
    device: torch.device,
    *,
    motion_features: bool,
    event_threshold: float = 1e-6,
) -> torch.Tensor:
    total = None
    count = 0
    for batch in loader:
        surface = batch["surface"].to(device)
        mask = batch["mask"].to(device)
        phi = extract_phi(model, surface)
        feats = prepare_seg_features(
            phi, surface, feature_mean=None,
            mean_center=False, motion_features=motion_features,
        )
        x, _ = flatten_valid_features(
            feats, mask, surface, event_threshold=event_threshold,
        )
        if x.numel() == 0:
            continue
        if total is None:
            total = x.double().sum(dim=0)
        else:
            total += x.double().sum(dim=0)
        count += x.shape[0]
    if total is None or count == 0:
        raise RuntimeError("No valid pixels for feature mean")
    return (total / count).float()
