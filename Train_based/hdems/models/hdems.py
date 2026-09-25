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
from hdems.models.paper_flow import flow_from_cost, multiscale_cost_volume
from hdems.models.prototype_head import PrototypeHead
from hdems.models.segmentation import HVConvHead, MotionSegHead, SegmentationHead
from hdems.ridge_head import RidgeHead
from hdems.seg_features import event_pixel_mask
from hdems.vsa.fpe import make_base_phases
from hdems.vsa.temporal import bind_trajectory, make_time_phases
from hdems.vsa.velocity import combine_event_velocity, encode_velocity


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

        # velocity-hypervector combination (ridge / prototype / cnn heads)
        vel = cfg.get("velocity", {})
        self.vel_bw = float(vel.get("vel_bw", 6.0))
        self.axis_combine = str(vel.get("axis_combine", "bind"))     # bind | bundle
        self.event_combine = str(vel.get("event_combine", "bind"))   # bind | bundle | bindbundle | concat
        self.event_feature = str(vel.get("event_feature", "phi"))    # phi | f
        self.register_buffer("phi_vx", make_base_phases(d, seed=7))
        self.register_buffer("phi_vy", make_base_phases(d, seed=8))
        # paper feature dim fed to a per-pixel head: concat -> 4d, else 2d
        combine_dim = 4 * d if self.event_combine == "concat" else 2 * d

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
        elif self.head_type == "prototype":
            self.seg_head = PrototypeHead(num_classes)
        else:  # "cnn" — HV-as-channels CNN on the paper feature tensor
            self.seg_head = HVConvHead(
                in_ch=combine_dim,
                embedding_dim=seg.get("embedding_dim", 32),
                num_classes=num_classes,
            )
        # motion-head decode params
        self.flow_beta = float(seg.get("flow_beta", 1.0))
        self.ego_iters = int(seg.get("ego_iters", 3))
        # paper two-time / multi-scale cost-volume params
        self.match_scales = tuple(match.get("scales", [0, 1, 2]))
        self.flow_alpha = float(match.get("alpha", 0.3))
        self.vel_scale = float(match.get("vel_scale", 1.0))
        # Eq.12 cost-volume pre-smoothing (paper: removes event-stochasticity noise)
        self.flow_smooth = int(match.get("smooth", 1))
        self.pyramid_levels = match.get("pyramid_levels", 4)
        self.temporal_window = temp.get("window", 8)
        self.register_buffer("time_phases", make_time_phases(d))

    def encode_pyramid(self, surface: torch.Tensor) -> list[torch.Tensor]:
        pyr_surfaces = build_pyramid(surface, self.pyramid_levels)
        return [self.encoder(s.unsqueeze(0) if s.dim() == 3 else s) for s in pyr_surfaces]

    def encode_times(self, surfaces: torch.Tensor) -> list[torch.Tensor]:
        """Encode each time-frame of a multi-time stack (B, T, 2, H, W) -> [F_t]."""
        return [self.encoder(surfaces[:, t]) for t in range(surfaces.shape[1])]

    def event_hv(self, f0: torch.Tensor) -> torch.Tensor:
        """Event hypervector fused with velocity (velocity.event_feature).

        phi -> bundled 7x7 neighbourhood of F0, each neighbour bound to its offset code
        f   -> the VFA descriptor F0 itself (paper Eq.4, F = T * K)
        """
        if self.event_feature == "phi":
            return self.matcher([f0])[0]
        if self.event_feature == "f":
            return f0
        raise ValueError(f"event_feature must be phi|f, got {self.event_feature!r}")

    def paper_features(self, surfaces: torch.Tensor):
        """Paper front-end features for a linear (Ridge) readout.

        (B, T, 2, H, W) -> (feats, flow):
          feats = event HV X (Phi or F, event_feature) fused with the residual-velocity
                  HV Mv (event_combine): (B, 2d, H, W), or (B, 4d, H, W) for concat
          flow  = decoded optical flow (B, 2, H, W)
        """
        fields = self.encode_times(surfaces)
        cost = multiscale_cost_volume(fields, self.matcher.M, self.match_scales)
        flow = flow_from_cost(cost, self.matcher.M, alpha=self.flow_alpha,
                              vel_scale=self.vel_scale, smooth=self.flow_smooth)
        # Fit the global model on EVENT pixels only -- empty pixels have flow ~0 and
        # would drag the fit to zero, leaving the "residual" equal to the raw flow.
        residual, _mag = ego_residual(flow, iters=self.ego_iters,
                                      valid=event_pixel_mask(surfaces))
        x = self.event_hv(fields[0])
        # residual velocity -> hypervector (axis_combine), fused with X (event_combine)
        mv = encode_velocity(residual[:, 0], residual[:, 1], self.phi_vx, self.phi_vy,
                             vel_bw=self.vel_bw, axis_combine=self.axis_combine)
        feats = combine_event_velocity(x, mv, self.event_combine)
        return feats, flow

    def forward(
        self,
        surface: torch.Tensor,
        *,
        task: str = "flow",
    ) -> dict[str, torch.Tensor]:
        if surface.dim() == 3:                       # (2, H, W) -> (1, 2, H, W)
            surface = surface.unsqueeze(0)
        multitime = surface.dim() == 5              # (B, T, 2, H, W)

        outputs: dict[str, torch.Tensor] = {}

        # ---- paper two-time / multi-scale motion path -----------------------
        if task == "segmentation" and self.head_type == "motion" and multitime:
            fields = self.encode_times(surface)                      # [F0, F1, F2, F4]
            cost = multiscale_cost_volume(fields, self.matcher.M, self.match_scales)
            flow = flow_from_cost(cost, self.matcher.M, alpha=self.flow_alpha,
                                  vel_scale=self.vel_scale, smooth=self.flow_smooth)
            residual, mag = ego_residual(flow, iters=self.ego_iters,
                                         valid=event_pixel_mask(surface))
            motion = torch.cat([residual, mag], dim=1)
            phi = self.event_hv(fields[0])                           # HV context (Phi or F)
            outputs["flow"] = flow
            outputs["seg_logits"] = self.seg_head(phi, motion=motion, surface=surface[:, -1])
            return outputs

        # ---- paper front-end with a per-pixel readout (ridge/prototype/cnn) --
        if task == "segmentation" and self.head_type in ("ridge", "prototype", "cnn") and multitime:
            feats, flow = self.paper_features(surface)
            outputs["flow"] = flow
            if self.head_type == "cnn":                      # trained HV-channels CNN
                outputs["seg_logits"] = self.seg_head(feats)
            else:                                            # closed-form linear / prototype
                outputs["seg_logits"] = self.seg_head.logits_from_features(feats)
            return outputs

        # ---- single-time paths (fall back to the last frame if a stack) -----
        if multitime:
            surface = surface[:, -1]
        f = self.encoder(surface)
        phi = self.matcher([f])[0]

        if task != "segmentation":
            outputs["flow"] = self.decoder(phi)
        elif self.head_type == "motion":
            flow = decode_flow(f, phi, self.matcher.phx, self.matcher.phy,
                               M=self.matcher.M, beta=self.flow_beta)
            residual, mag = ego_residual(flow, iters=self.ego_iters,
                                         valid=event_pixel_mask(surface))
            motion = torch.cat([residual, mag], dim=1)
            outputs["flow"] = flow
            outputs["seg_logits"] = self.seg_head(phi, motion=motion, surface=surface)
        elif self.head_type == "cnn":                        # single-frame fallback (event_combine != concat)
            feats = torch.cat([phi.real, phi.imag], dim=1).float()
            outputs["seg_logits"] = self.seg_head(feats)
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
