"""Full HD-EMS model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hdems.data.time_surface import build_pyramid
from hdems.models.decoder import FlowDecoder
from hdems.models.encoder import VSAEncoder
from hdems.models.matching import HierarchicalMatcher
from hdems.models.motion import decode_flow, ego_residual
from hdems.models.segmentation import MotionSegHead, SegmentationHead
from hdems.ridge_head import RidgeHead
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
        self.head_type = str(seg.get("head", "cnn")).lower()
        num_classes = seg.get("num_classes", 16)
        mean_center = bool(seg.get("mean_center", False))
        motion_features = bool(seg.get("motion_features", False))
        if self.head_type == "ridge":
            self.seg_head = RidgeHead(
                num_classes=num_classes,
                mean_center=seg.get("ridge_mean_center", True),
                motion_features=seg.get("ridge_motion_features", True),
            )
        elif self.head_type == "motion":
            self.seg_head = MotionSegHead(
                d=d,
                embedding_dim=seg.get("embedding_dim", 32),
                num_classes=num_classes,
                ctx_dim=seg.get("ctx_dim", 32),
            )
        else:
            self.seg_head = SegmentationHead(
                d=d,
                embedding_dim=seg.get("embedding_dim", 32),
                num_classes=num_classes,
                mean_center=mean_center,
                motion_features=motion_features,
            )
        # motion-head decode params
        self.flow_beta = float(seg.get("flow_beta", 1.0))
        self.ego_iters = int(seg.get("ego_iters", 3))
        self.pyramid_levels = match.get("pyramid_levels", 4)
        self.temporal_window = temp.get("window", 8)
        self.register_buffer("time_phases", make_time_phases(d))

    def encode_pyramid(self, surface: torch.Tensor) -> list[torch.Tensor]:
        pyr_surfaces = build_pyramid(surface, self.pyramid_levels)
        return [self.encoder(s.unsqueeze(0) if s.dim() == 3 else s) for s in pyr_surfaces]

    def forward(
        self,
        surface: torch.Tensor,
        *,
        task: str = "flow",
    ) -> dict[str, torch.Tensor]:
        if surface.dim() == 3:
            surface = surface.unsqueeze(0)

        f = self.encoder(surface)
        phi = self.matcher([f])[0]

        outputs: dict[str, torch.Tensor] = {}
        if task != "segmentation":
            outputs["flow"] = self.decoder(phi)
        if task == "segmentation":
            if self.head_type == "motion":
                # Phase 1: decode velocity from Phi; Phase 3: ego-compensate.
                flow = decode_flow(
                    f, phi, self.matcher.phx, self.matcher.phy,
                    M=self.matcher.M, beta=self.flow_beta,
                )
                residual, mag = ego_residual(flow, iters=self.ego_iters)
                motion = torch.cat([residual, mag], dim=1)
                outputs["flow"] = flow
                outputs["seg_logits"] = self.seg_head(phi, motion=motion, surface=surface)
            else:
                outputs["seg_logits"] = self.seg_head(phi, surface=surface)
        return outputs

    def bind_temporal(
        self,
        field_sequence: list[torch.Tensor],
        times: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack(field_sequence, dim=0)
        return bind_trajectory(stacked, self.time_phases, times)

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def seg_head_param_count(self) -> int:
        if self.head_type == "ridge":
            return self.seg_head.num_readout_params()
        return sum(p.numel() for p in self.seg_head.parameters())
