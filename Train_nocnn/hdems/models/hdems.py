"""Full HD-EMS model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hdems.config import apply_resolution_ratio, frontend_option, resolution_ratio_of
from hdems.data.time_surface import build_pyramid
from hdems.flow_cache import FlowCache
from hdems.models.decoder import FlowDecoder
from hdems.models.encoder import VSAEncoder
from hdems.models.matching import HierarchicalMatcher
from hdems.models.motion import decode_flow, ego_residual
from hdems.models.paper_flow import flow_from_cost, multiscale_cost_volume
from hdems.models.prototype_head import PrototypeHead
from hdems.models.segmentation import (
    HVConvHead,
    MotionFirstHead,
    MotionSegHead,
    MotionUNetHead,
    SegmentationHead,
)
from hdems.ridge_head import RidgeHead
from hdems.seg_features import event_pixel_mask
from hdems.vsa.field import bundled_field
from hdems.vsa.fpe import make_base_phases
from hdems.vsa.temporal import bind_trajectory, make_time_phases
from hdems.vsa.velocity import (
    MOTION_SCALARS,
    combine_event_velocity,
    encode_velocity,
    motion_scalars,
    unit_rms,
)

ABLATIONS = ("motion", "appearance")
MOTION_FIRST_HEADS = ("mfcnn", "mfunet")
EVIDENCE_CHANNELS = 2          # mfunet: event density + flow confidence


class HDEMS(nn.Module):
    """Hyperdimensional Event Motion Segmentation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        # patch_size / M / smooth are scaled by dataset.resolution_ratio (no-op if the
        # entry point already applied it)
        cfg = apply_resolution_ratio(cfg)
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
            # paper defaults; set kernel=window, polarity_binding=false, scales=1 to
            # rebuild the pre-2026-09-28 encoder for old checkpoints
            kernel=str(enc.get("kernel", "conv")),
            polarity_binding=bool(enc.get("polarity_binding", True)),
            scales=int(enc.get("scales", 2)),
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

        # Front-end options added 2026-09-29 (old checkpoints restore the old values,
        # see hdems.config._LEGACY_FRONTEND):
        self.res_ratio = resolution_ratio_of(cfg)
        # index of the surface that ends at the label time (time_frames entry 1.0)
        tf = (cfg.get("dataset", {}) or {}).get("time_frames") or [1.0]
        self.label_t = max(range(len(tf)), key=lambda i: float(tf[i]))
        # Phi's own window, in working pixels. It used to be the cost-volume M
        # (31 -> 961 bundled terms in d=512, far past the capacity of the bundle).
        self.phi_window = int(frontend_option(cfg, "matching", "phi_window"))
        self.phi_pad = str(frontend_option(cfg, "matching", "phi_pad"))
        # velocity code in fixed units: vel_unit_px SENSOR px per interval
        self.vel_norm = str(frontend_option(cfg, "velocity", "vel_norm"))
        self.vel_unit_px = float(frontend_option(cfg, "velocity", "vel_unit_px"))
        self.x_norm = str(frontend_option(cfg, "velocity", "x_norm"))       # rms | none
        self.ego_fit = str(frontend_option(cfg, "velocity", "ego_fit"))     # reference | stack
        # Evaluation switch (hdems.eval --ablation-check): "motion" zeroes the residual
        # velocity, "appearance" zeroes the event HV X. None = normal.
        self.ablate: str | None = None

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
        elif self.head_type == "mfcnn":  # motion-first: Mv + motion channels, small appearance
            self.seg_head = MotionFirstHead(
                d,
                num_classes,
                embedding_dim=seg.get("embedding_dim", 32),
                motion_dim=int(seg.get("mf_motion_dim", 32)),
                app_dim=int(seg.get("mf_app_dim", 8)),
                app_dropout=float(seg.get("mf_app_dropout", 0.5)),
                scalar_ch=MOTION_SCALARS,
            )
        elif self.head_type == "mfunet":  # the same inputs + evidence channels, U-Net view
            self.seg_head = MotionUNetHead(
                d,
                num_classes,
                motion_dim=int(seg.get("mf_motion_dim", 32)),
                app_dim=int(seg.get("mf_app_dim", 8)),
                app_dropout=float(seg.get("mf_app_dropout", 0.5)),
                scalar_ch=MOTION_SCALARS + EVIDENCE_CHANNELS,
                app_scalars=(MOTION_SCALARS,),      # event density goes with appearance
                widths=tuple(int(w) for w in seg.get("mf_widths", (32, 64, 96, 128))),
            )
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

        # The cost volume is ~80% of a training step and has no trainable parameters,
        # so its flow is cached per sample instead of recomputed every epoch.
        fc = cfg.get("flow_cache", {}) or {}
        self.flow_cache: FlowCache | None = (
            FlowCache(fc.get("dir", "cache/flow")) if fc.get("enabled", False) else None
        )

    def encode_pyramid(self, surface: torch.Tensor) -> list[torch.Tensor]:
        pyr_surfaces = build_pyramid(surface, self.pyramid_levels)
        return [self.encoder(s.unsqueeze(0) if s.dim() == 3 else s) for s in pyr_surfaces]

    def encode_times(self, surfaces: torch.Tensor) -> list[torch.Tensor]:
        """Encode each time-frame of a multi-time stack (B, T, 2, H, W) -> [F_t]."""
        return [self.encoder(surfaces[:, t]) for t in range(surfaces.shape[1])]

    def event_hv(self, f0: torch.Tensor) -> torch.Tensor:
        """Event hypervector fused with velocity (velocity.event_feature).

        phi -> bundled phi_window x phi_window neighbourhood of F0, each neighbour
               bound to its offset code (phi_window 0 = the cost-volume M, old)
        f   -> the VFA descriptor F0 itself (paper Eq.4, F = T * K)
        """
        if self.event_feature == "phi":
            return bundled_field(f0, self.matcher.phx, self.matcher.phy,
                                 M=self.phi_window or self.matcher.M, pad=self.phi_pad)
        if self.event_feature == "f":
            return f0
        raise ValueError(f"event_feature must be phi|f, got {self.event_feature!r}")

    def ego_mask(self, surfaces: torch.Tensor) -> torch.Tensor:
        """Pixels the ego-motion model is fitted on -> (B, H, W) bool.

        stack     : events of any surface (default)
        reference : events of the reference surface only (index 0), where the flow is
                    measured from. Measured 2026-09-30: no better than stack (AUC 0.696
                    vs 0.700 on 48 eval frames) -- the Eq.12 pooling spreads the flow
                    beyond the reference events anyway.
        """
        if self.ego_fit == "reference":
            return event_pixel_mask(surfaces[:, :1])
        return event_pixel_mask(surfaces)

    def residual_from_flow(self, flow: torch.Tensor, surfaces: torch.Tensor):
        """Ego-compensated (residual) velocity, working px per interval."""
        return ego_residual(flow, iters=self.ego_iters, valid=self.ego_mask(surfaces))

    def motion_inputs(self, surfaces: torch.Tensor):
        """Shared front end of every multi-time head -> (flow, confidence, residual, X).

        X is the event HV of the reference surface (Phi or F), unit-RMS when
        velocity.x_norm = rms. ``self.ablate`` zeroes one of the two inputs.
        """
        flow, conf = self.compute_motion(surfaces)
        residual, _ = self.residual_from_flow(flow, surfaces)
        # Only the reference surface is needed for X, so a cache hit skips the other
        # three encodes as well as the cost volume.
        x = self.event_hv(self.encoder(surfaces[:, 0]))
        if self.x_norm == "rms":
            # raw F is ~5-11x and Phi ~100-370x larger than the |.| = 1 velocity code
            x = unit_rms(x)
        if self.ablate == "motion":
            residual = torch.zeros_like(residual)    # velocity code = FPE(0) everywhere
        elif self.ablate == "appearance":
            x = torch.zeros_like(x)
        elif self.ablate is not None:
            raise ValueError(f"ablate must be one of {ABLATIONS} or None, got {self.ablate!r}")
        return flow, conf, residual, x

    def evidence_channels(self, surfaces: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Head mfunet: where the evidence is, and how much to trust the flow -> (B, 2, H, W).

        density    : log(1 + decayed event count) of the surface that ends at the label
                     time, per SENSOR pixel block (area downsampling averaged it)
        confidence : how concentrated the Eq.12 probability volume is (sum of P^2,
                     paper_flow.flow_from_cost): 0 for a flat cost volume, low along
                     edges. Only a weak predictor of flow error (measured), so the
                     head may learn to ignore it
        Removed with the appearance / motion input respectively by ``self.ablate``.
        """
        s = surfaces[:, self.label_t].abs().sum(1, keepdim=True) * self.res_ratio ** 2
        density = torch.log1p(s)
        if self.ablate == "appearance":
            density = torch.zeros_like(density)
        if self.ablate == "motion":
            conf = torch.zeros_like(conf)
        return torch.cat([density, conf], dim=1).float()

    def velocity_hv(self, residual: torch.Tensor) -> torch.Tensor:
        """Residual velocity (B, 2, H, W), working px -> Mv (B, d, H, W).

        Converted to SENSOR px first, so vel_unit_px means the same at every
        resolution ratio.
        """
        r = residual * self.res_ratio
        return encode_velocity(r[:, 0], r[:, 1], self.phi_vx, self.phi_vy,
                               vel_bw=self.vel_bw, axis_combine=self.axis_combine,
                               norm=self.vel_norm, unit=self.vel_unit_px)

    @torch.no_grad()
    def _flow_uncached(self, surfaces: torch.Tensor) -> torch.Tensor:
        """Paper Eq.10-14: multi-time fields -> multi-scale cost volume -> flow.

        Returns (B, 3, H, W) = [u_x, u_y, match confidence].
        """
        fields = self.encode_times(surfaces)
        cost = multiscale_cost_volume(fields, self.matcher.M, self.match_scales)
        flow, conf = flow_from_cost(cost, self.matcher.M, alpha=self.flow_alpha,
                                    vel_scale=self.vel_scale, smooth=self.flow_smooth,
                                    return_confidence=True)
        return torch.cat([flow, conf], dim=1)

    @torch.no_grad()
    def compute_motion(self, surfaces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, 2, H, W) -> flow (B, 2, H, W) and match confidence (B, 1, H, W),
        from the flow cache when attached.

        Only the samples that miss are sent through the cost volume. Cached and
        fresh flow both pass through the same fp16 round trip, so the features are
        bit-identical whether or not a sample was cached.
        """
        if self.flow_cache is None:
            out = self._flow_uncached(surfaces)
            return out[:, :2], out[:, 2:3]
        self.flow_cache.bind(self)
        flows: list[torch.Tensor | None] = [self.flow_cache.load(s) for s in surfaces]
        miss = [b for b, f in enumerate(flows) if f is None]
        if miss:
            fresh = self._flow_uncached(surfaces[miss])
            for j, b in enumerate(miss):
                self.flow_cache.save(surfaces[b], fresh[j])
                flows[b] = fresh[j].to(torch.float16)
        out = torch.stack([f.to(device=surfaces.device, dtype=torch.float32) for f in flows])
        return out[:, :2], out[:, 2:3]

    def compute_flow(self, surfaces: torch.Tensor) -> torch.Tensor:
        """(B, T, 2, H, W) -> flow (B, 2, H, W) (see ``compute_motion``)."""
        return self.compute_motion(surfaces)[0]

    def paper_features(self, surfaces: torch.Tensor):
        """Paper front-end features for a linear (Ridge) readout.

        (B, T, 2, H, W) -> (feats, flow):
          feats = event HV X (Phi or F, event_feature) fused with the residual-velocity
                  HV Mv (event_combine): (B, 2d, H, W), or (B, 4d, H, W) for concat
          flow  = decoded optical flow (B, 2, H, W)
        """
        flow, _conf, residual, x = self.motion_inputs(surfaces)
        # residual velocity -> hypervector (axis_combine), fused with X (event_combine)
        feats = combine_event_velocity(x, self.velocity_hv(residual), self.event_combine)
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

        # ---- motion-first heads: Mv + explicit motion channels + small appearance
        #      mfunet adds event density + flow confidence and a U-Net (wide view)
        if task == "segmentation" and self.head_type in MOTION_FIRST_HEADS:
            if not multitime:
                raise ValueError(f"head {self.head_type} needs the multi-time stack "
                                 "(dataset.time_frames)")
            flow, conf, residual, x = self.motion_inputs(surface)
            raw = torch.zeros_like(flow) if self.ablate == "motion" else flow
            scalars = motion_scalars(residual * self.res_ratio,       # sensor px
                                     event_pixel_mask(surface[:, :1]),
                                     raw * self.res_ratio, unit=self.vel_unit_px)
            if self.head_type == "mfunet":
                scalars = torch.cat([scalars, self.evidence_channels(surface, conf)], dim=1)
            outputs["flow"] = flow
            outputs["seg_logits"] = self.seg_head(self.velocity_hv(residual), x, scalars)
            return outputs

        # ---- paper two-time / multi-scale motion path -----------------------
        if task == "segmentation" and self.head_type == "motion" and multitime:
            flow, _conf, residual, phi = self.motion_inputs(surface)  # cached when enabled
            mag = residual.pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
            motion = torch.cat([residual, mag], dim=1)
            outputs["flow"] = flow
            outputs["seg_logits"] = self.seg_head(phi, motion=motion, surface=surface[:, 0])
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

        # ---- single-time paths (fall back to the label-time frame if a stack) -
        if multitime:
            surface = surface[:, self.label_t]
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
