"""Full HD-EMS model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hdems.data.time_surface import build_pyramid
from hdems.models.decoder import FlowDecoder
from hdems.models.encoder import VSAEncoder
from hdems.models.matching import HierarchicalMatcher
from hdems.models.segmentation import SegmentationHead
from hdems.vsa.temporal import bind_trajectory, make_time_phases


class HDEMS(nn.Module):
    """Hyperdimensional Event Motion Segmentation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        d = cfg.get("d", 1024)
        enc = cfg.get("encoder", {})
        match = cfg.get("matching", {})
        dec = cfg.get("decoder", {})
        seg = cfg.get("segmentation", {})
        temp = cfg.get("temporal", {})

        self.encoder = VSAEncoder(
            d=d,
            patch_size=enc.get("patch_size", 21),
            sigma_k=enc.get("sigma_k", 1.5),
            rank=enc.get("rank", 64),
            separable_terms=enc.get("separable_terms", 2),
        )
        self.matcher = HierarchicalMatcher(
            d=d,
            M=match.get("M", 7),
            pyramid_levels=match.get("pyramid_levels", 4),
        )
        self.decoder = FlowDecoder(
            d=d,
            hidden_channels=dec.get("hidden_channels", 64),
            gru_layers=dec.get("gru_layers", 1),
        )
        self.seg_head = SegmentationHead(
            d=d,
            embedding_dim=seg.get("embedding_dim", 32),
            num_classes=seg.get("num_classes", 16),
        )
        self.pyramid_levels = match.get("pyramid_levels", 4)
        self.temporal_window = temp.get("window", 8)
        self.register_buffer("time_phases", make_time_phases(d))

    def encode_pyramid(self, surface: torch.Tensor) -> list[torch.Tensor]:
        """Time surface -> pyramid of complex descriptor fields."""
        pyr_surfaces = build_pyramid(surface, self.pyramid_levels)
        return [self.encoder(s.unsqueeze(0) if s.dim() == 3 else s) for s in pyr_surfaces]

    def forward(
        self,
        surface: torch.Tensor,
        *,
        task: str = "flow",
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        surface : (B, C, H, W) time surface
        task : "flow" or "segmentation"

        Returns
        -------
        dict with "flow" and/or "seg_logits"
        """
        if surface.dim() == 3:
            surface = surface.unsqueeze(0)
        B = surface.shape[0]

        # Encode finest level (extend to full pyramid in training loop)
        F = self.encoder(surface)
        Phi = self.matcher([F])[0]

        outputs: dict[str, torch.Tensor] = {}
        # Only run the flow decoder when flow is actually needed; for the
        # segmentation task it is unused and just wastes GPU memory/compute.
        if task != "segmentation":
            outputs["flow"] = self.decoder(Phi)
        if task == "segmentation":
            outputs["seg_logits"] = self.seg_head(Phi)
        return outputs

    def bind_temporal(
        self,
        field_sequence: list[torch.Tensor],
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Bind temporal trajectory from field sequence."""
        stacked = torch.stack(field_sequence, dim=0)
        return bind_trajectory(stacked, self.time_phases, times)

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
