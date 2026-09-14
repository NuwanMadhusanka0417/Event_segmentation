"""Lightweight RANSAC background motion."""

from __future__ import annotations

import torch


def ransac_global_translation(
    flow: torch.Tensor,
    valid: torch.Tensor,
    *,
    inlier_threshold: float = 1.5,
    min_inliers: int = 50,
    max_iterations: int = 200,
) -> dict[str, torch.Tensor | float | bool]:
    """
    flow: (2, H, W), valid: (H, W)
    Returns background (2,), inlier_mask (H,W), success bool.
    """
    H, W = valid.shape
    ys, xs = torch.where(valid)
    if ys.numel() < min_inliers:
        return {
            "background_flow": torch.zeros(2),
            "inlier_mask": torch.zeros(H, W, dtype=torch.bool),
            "success": False,
        }
    vx = flow[0, ys, xs]
    vy = flow[1, ys, xs]
    pts = torch.stack([vx, vy], dim=1)
    best_inliers = None
    best_count = -1
    bg = torch.zeros(2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(max_iterations):
        idx = torch.randint(0, pts.shape[0], (1,), generator=gen).item()
        candidate = pts[idx]
        dist = (pts - candidate).norm(dim=1)
        inl = dist < inlier_threshold
        c = inl.sum().item()
        if c > best_count:
            best_count = c
            best_inliers = inl
            bg = pts[inl].mean(dim=0)
    success = best_count >= min_inliers
    inlier_mask = torch.zeros(H, W, dtype=torch.bool)
    if success and best_inliers is not None:
        inlier_mask[ys[best_inliers], xs[best_inliers]] = True
    return {
        "background_flow": bg,
        "inlier_mask": inlier_mask,
        "success": success,
    }


def compensate_ego(
    flow: torch.Tensor,
    mode: str,
    valid: torch.Tensor,
    ransac_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns rigid_background (2,H,W) and inlier_mask."""
    H, W = flow.shape[-2:]
    if mode == "stationary_camera":
        return torch.zeros_like(flow), torch.ones(H, W, dtype=torch.bool)
    if mode in ("ransac_global_model", "rotation_only"):
        res = ransac_global_translation(flow, valid, **ransac_cfg)
        bg = res["background_flow"].view(2, 1, 1).expand_as(flow)
        return bg, res["inlier_mask"]
    return torch.zeros_like(flow), valid
