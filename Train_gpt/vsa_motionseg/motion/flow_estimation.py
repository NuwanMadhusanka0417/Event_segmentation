"""Optical flow from VSA cost volume."""

from __future__ import annotations

import torch

from vsa_motionseg.motion.cost_volume import compute_cost_volume_pair, cost_to_flow


def estimate_flow_vsa(
    F_t0: torch.Tensor,
    F_t1: torch.Tensor,
    cfg: dict,
    active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    r = int(cfg.get("search_radius", 3))
    C = compute_cost_volume_pair(
        F_t0,
        F_t1,
        r,
        active_mask=active_mask,
        chunked=bool(cfg.get("chunked", False)),
    )
    stats = cost_to_flow(
        C,
        temperature=float(cfg.get("temperature", 0.05)),
        delta_t=float(cfg.get("delta_t_ms", 50)) / 1000.0,
    )
    conf_thr = float(cfg.get("confidence_threshold", 0.15))
    valid = stats["max_prob"] >= conf_thr
    if active_mask is not None:
        valid = valid & active_mask
    stats["valid_mask"] = valid
    stats["cost_volume"] = C
    return stats
