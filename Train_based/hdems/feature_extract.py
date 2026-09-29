"""Extract frozen encoder features for ridge fitting."""

from __future__ import annotations

import torch

from hdems.models.hdems import HDEMS
from hdems.seg_features import flatten_valid_features, prepare_seg_features, score_pixel_mask


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
def extract_paper_flat_batch(
    model: HDEMS,
    surface: torch.Tensor,
    mask: torch.Tensor,
    *,
    feature_mean: torch.Tensor | None,
    mean_center: bool,
    event_threshold: float = 1e-6,
    score_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten paper front-end features (multi-time cost volume) over valid pixels.

    ``score_mask``: pixels to fit on (seg_features.score_pixel_mask); default =
    the event pixels of the reference surface.
    """
    feats, _ = model.paper_features(surface)                 # (B, 2d+3, H, W)
    if mean_center and feature_mean is not None and feature_mean.numel():
        feats = feats - feature_mean.to(device=feats.device, dtype=feats.dtype).view(1, -1, 1, 1)
    return flatten_valid_features(feats, mask, surface[:, 0], event_threshold=event_threshold,
                                  score_mask=score_mask)


@torch.no_grad()
def accumulate_paper_feature_mean(
    model: HDEMS,
    loader,
    device: torch.device,
    *,
    event_threshold: float = 1e-6,
) -> torch.Tensor:
    """Mean of paper front-end features over valid pixels (for mean-centering)."""
    total = None
    count = 0
    for batch in loader:
        surface = batch["surface"].to(device)
        mask = batch["mask"].to(device)
        feats, _ = model.paper_features(surface)
        x, _ = flatten_valid_features(
            feats, mask, surface[:, 0], event_threshold=event_threshold,
            score_mask=score_pixel_mask(batch, surface),
        )
        if x.numel() == 0:
            continue
        total = x.double().sum(dim=0) if total is None else total + x.double().sum(dim=0)
        count += x.shape[0]
    if total is None or count == 0:
        raise RuntimeError("No valid pixels for paper feature mean")
    return (total / count).float()


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
