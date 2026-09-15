"""End-to-end VSA motion segmentation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vsa_motionseg.config import resolve_device
from vsa_motionseg.data.evimo_adapter import EVIMO2Adapter
from vsa_motionseg.data.time_surface import TimeSurfaceBuilder, events_window
from vsa_motionseg.evaluation.benchmark import StageTimer
from vsa_motionseg.motion.ego_motion import compensate_ego
from vsa_motionseg.motion.flow_estimation import estimate_flow_vsa
from vsa_motionseg.motion.residual_flow import residual_flow
from vsa_motionseg.segmentation.clustering import dbscan_segments
from vsa_motionseg.segmentation.dynamic_classifier import DynamicClassifier
from vsa_motionseg.segmentation.graph_refinement import local_vsa_refinement
from vsa_motionseg.segmentation.region_growing import region_growing
from vsa_motionseg.segmentation.temporal_tracking import TemporalTracker, segment_summary
from vsa_motionseg.vsa.encoder import MultiScaleVSAEncoder
from vsa_motionseg.vsa.motion_codebook import MotionCodebook


class VSAMotionSegPipeline:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.get("runtime", {}).get("device", "cpu"))
        vsa = cfg["vsa"]
        self.encoder = MultiScaleVSAEncoder(
            d=int(vsa["dimension"]),
            patch_size=int(vsa["kernel_size"]),
            sigma_k=float(vsa["gaussian_sigma"]),
            rank=int(vsa["basis_rank"]),
            scales=list(vsa.get("scales", [1, 2, 4])),
            seed=int(vsa.get("seed", 42)),
            representation=str(vsa.get("representation", "real")),
            artifact_dir=vsa.get("artifact_dir"),
        )
        self.encoder.to(self.device)
        self.encoder.eval()
        self.codebook = MotionCodebook(d=int(vsa["dimension"]), seed=int(vsa.get("seed", 42)))
        self.codebook.to(self.device)
        self.classifier = DynamicClassifier()
        proto_path = cfg.get("classifier", {}).get("prototype_path")
        if proto_path and Path(proto_path).exists():
            from vsa_motionseg.vsa.prototypes import load_prototypes

            protos = load_prototypes(proto_path).to(self.device)
            self.classifier = DynamicClassifier(protos)
        t_cfg = cfg.get("temporal", {})
        self.tracker = TemporalTracker(
            max_track_gap=int(t_cfg.get("max_track_gap", 3)),
            min_iou=float(t_cfg.get("min_iou", 0.2)),
            flow_weight=float(t_cfg.get("flow_weight", 1.0)),
            hv_weight=float(t_cfg.get("hv_weight", 0.5)),
            centroid_weight=float(t_cfg.get("centroid_weight", 0.1)),
        )

    def process_window(
        self,
        events: torch.Tensor,
        height: int,
        width: int,
        t_end: float,
        window_s: float,
    ) -> dict[str, Any]:
        timer = StageTimer()
        ev_cfg = self.cfg["events"]
        window_ms = float(ev_cfg.get("window_ms", 50))
        window_s = window_ms / 1000.0
        t0 = t_end - window_s
        t_mid = t0 + 0.5 * window_s

        tsb = TimeSurfaceBuilder(
            height,
            width,
            tau_ms=ev_cfg.get("tau_ms"),
            legacy_decay=float(ev_cfg.get("legacy_decay", 0.8)),
            normalize=bool(ev_cfg.get("normalize_time_surface", True)),
        )

        def _ts(t_a: float, t_b: float):
            ev = events_window(events, t_a, t_b) if events.numel() else events
            return tsb.from_events(ev, t_end=t_b)

        pack0 = timer.run("Time-surface construction", lambda: _ts(t0, t_mid))
        pack1 = timer.run("Time-surface t1", lambda: _ts(t_mid, t_end))
        s0 = pack0["surface"].unsqueeze(0).to(self.device)
        s1 = pack1["surface"].unsqueeze(0).to(self.device)
        active = (pack1["active_mask"] | pack0["active_mask"]).to(self.device)

        F0 = timer.run("VSA encoding t0", lambda: self.encoder(s0))
        F1 = timer.run("VSA encoding t1", lambda: self.encoder(s1))
        if not isinstance(F0, torch.Tensor) or F0.is_complex():
            pass
        elif F0.dtype != torch.complex64:
            F0 = F0.to(torch.complex64)
            F1 = F1.to(torch.complex64)

        flow_cfg = {**self.cfg.get("flow", {}), "delta_t_ms": window_ms / 2}
        flow_out = timer.run(
            "Cost-volume flow",
            lambda: estimate_flow_vsa(F0, F1, flow_cfg, active_mask=active),
        )
        optical = flow_out["flow"][0]
        valid = flow_out["valid_mask"][0]

        ego_cfg = self.cfg.get("ego_motion", {})
        ransac = {
            "inlier_threshold": float(ego_cfg.get("inlier_threshold", 1.5)),
            "min_inliers": int(ego_cfg.get("min_inliers", 50)),
            "max_iterations": int(ego_cfg.get("max_iterations", 200)),
        }
        rigid, _ = compensate_ego(optical, str(ego_cfg.get("mode", "stationary_camera")), valid, ransac)
        res = timer.run("Ego/residual", lambda: residual_flow(optical, rigid))

        H, W = height, width
        d = int(self.cfg["vsa"]["dimension"])
        from vsa_motionseg.vsa.fpe import bind, fpe

        F_hw = F1[0].permute(1, 2, 0)
        mx = fpe(self.codebook.phases_mx, self.codebook.quantize(res[0]))
        my = fpe(self.codebook.phases_my, self.codebook.quantize(res[1]))
        Q = bind(bind(F_hw, mx), my).permute(2, 0, 1)

        dynamic = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        conf_map = torch.zeros(H, W, device=self.device)
        if self.classifier.prototypes is not None:
            flat = Q.reshape(d, -1).T
            pred, conf = self.classifier.predict(
                flat, threshold=float(self.cfg.get("classifier", {}).get("confidence_threshold", 0.0))
            )
            dynamic = (pred == 1).reshape(H, W)
            conf_map = conf.reshape(H, W)
        else:
            dynamic = active & (res.norm(dim=0) > 0.05)
            conf_map = flow_out["max_prob"][0]

        res_cpu = res.detach().cpu()
        F1_cpu = F1[0].detach().cpu()
        dynamic_cpu = dynamic.detach().cpu()
        Q_cpu = Q.detach().cpu()
        conf_cpu = conf_map.detach().cpu()

        cl_cfg = self.cfg.get("clustering", {})
        method = cl_cfg.get("method", "region_growing")

        def _cluster():
            if method == "dbscan":
                return dbscan_segments(dynamic_cpu, res_cpu, F1_cpu, cl_cfg)
            return region_growing(
                dynamic_cpu,
                res_cpu,
                F1_cpu,
                flow_threshold=float(cl_cfg.get("flow_threshold", 2.0)),
                hv_threshold=float(cl_cfg.get("hv_similarity_threshold", 0.3)),
                neighborhood=int(cl_cfg.get("neighborhood", 8)),
                min_cluster_size=int(cl_cfg.get("min_cluster_size", 20)),
            )

        labels = timer.run("Clustering", _cluster)

        ref_cfg = self.cfg.get("refinement", {})
        if ref_cfg.get("enabled", True):
            Q_cpu = timer.run(
                "Local VSA refinement",
                lambda: local_vsa_refinement(
                    Q_cpu,
                    res_cpu,
                    conf_cpu,
                    lambda_bundle=float(ref_cfg.get("lambda_bundle", 0.2)),
                    sigma_flow=float(ref_cfg.get("sigma_flow", 2.0)),
                    sigma_hv=float(ref_cfg.get("sigma_hv", 0.3)),
                    connectivity=int(ref_cfg.get("connectivity", 8)),
                ),
            )

        segs = segment_summary(labels, Q_cpu, res_cpu, conf_cpu)
        track_map = self.tracker.update(segs, 0) if self.cfg.get("temporal", {}).get("enabled", True) else {}

        return {
            "event_hypervector": F1_cpu,
            "motion_hypervector": Q_cpu,
            "residual_flow": res_cpu,
            "motion_confidence": conf_cpu,
            "event_count": pack1["event_count"],
            "active_mask": active.cpu(),
            "dynamic_mask": dynamic_cpu,
            "segment_labels": labels,
            "track_map": track_map,
            "optical_flow": optical.detach().cpu(),
            "benchmark": timer.report.format(),
            "device": str(self.device),
        }

    def run_sequence(self, adapter: EVIMO2Adapter, max_frames: int | None = None) -> list[dict]:
        out = []
        n = len(adapter) if max_frames is None else min(len(adapter), max_frames)
        window_ms = float(self.cfg["events"]["window_ms"])
        window_s = window_ms / 1000.0
        for i in range(n):
            fr = adapter.get_frame(i)
            ts = fr.timestamp
            ev = adapter.events_in_window(ts - window_s, ts)
            out.append(
                self.process_window(ev, fr.image_height, fr.image_width, ts, window_s)
            )
        return out
