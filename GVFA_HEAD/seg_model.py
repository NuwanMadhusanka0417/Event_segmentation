"""Frozen GVFA encoder + thin trainable Adapter/SegHead for per-event FG/BG."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fpe_codebook import bind_hv, bundle_weighted
from gvfa_encoder import (
    EventGraph,
    encode_edge_h_spatial,
    encode_edge_h_temporal,
)
from seg_config import (
    D as HV_DIM,
    FLOW_KEEP,
    FLOW_SMOOTH_ITERS,
    NUM_LAYERS,
    SEED,
    SPATIAL_MMAX,
    SPATIAL_R_T_MS,
    SPATIAL_R_XY_FRAC,
    TEMPORAL_MMAX,
    TEMPORAL_R_T_MS,
    TEMPORAL_R_XY_FRAC,
    W_NODE_MOTION,
    W_NODE_P,
    W_NODE_T,
    W_NODE_X,
    W_NODE_Y,
    build_multigraph,
    make_codebooks,
    node_flow,
    smooth_flow,
)

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graphcnnVSA_Binding_FULL_new import GraphCNN  # noqa: E402


@dataclass
class GraphCfg:
    width: int = 346
    height: int = 260
    num_layers: int = NUM_LAYERS
    hv_dim: int = HV_DIM
    adapter_dim: int = 128
    device: str = "cpu"
    spatial_r_xy_frac: float = SPATIAL_R_XY_FRAC
    spatial_r_t_ms: float = SPATIAL_R_T_MS
    spatial_mmax: int = SPATIAL_MMAX
    temporal_r_xy_frac: float = TEMPORAL_R_XY_FRAC
    temporal_r_t_ms: float = TEMPORAL_R_T_MS
    temporal_mmax: int = TEMPORAL_MMAX


def _encode_nodes_xytp_motion(x, y, t, p, vx, vy, node_cbs) -> torch.Tensor:
    """Node FPE from (x, y, t, p) + causal motion (|v|, angle). No raw (x,y) to head."""
    t_us = (t - t[0]) * 1e6
    terms = [
        (node_cbs["x"].encode(x), W_NODE_X),
        (node_cbs["y"].encode(y), W_NODE_Y),
        (node_cbs["t"].encode(t_us, interpolate=True), W_NODE_T),
        (node_cbs["p"].encode(p), W_NODE_P),
    ]

    vx = np.asarray(vx, dtype=np.float64).ravel()
    vy = np.asarray(vy, dtype=np.float64).ravel()
    motion_valid = np.isfinite(vx) & np.isfinite(vy) & (np.hypot(vx, vy) > 1e-12)
    speed = np.hypot(vx, vy)
    ang = np.arctan2(vy, vx)
    z_speed = node_cbs["speed"].encode(speed, interpolate=True)
    z_dir = node_cbs["dir"].encode(ang, interpolate=True)
    z_motion = bind_hv(z_speed, z_dir)
    if (~motion_valid).any():
        z_motion = z_motion.clone()
        z_motion[torch.from_numpy(~motion_valid)] = 0.0
    terms.append((z_motion, W_NODE_MOTION))
    return bundle_weighted(terms)


def build_causal_graph(events: dict[str, np.ndarray], cfg: GraphCfg) -> dict[str, Any]:
    """Past-only spatial+temporal ellipsoid graphs + causal normal flow."""
    t = np.asarray(events["t"], dtype=np.float64)
    x = np.asarray(events["x"], dtype=np.float64)
    y = np.asarray(events["y"], dtype=np.float64)
    p = np.asarray(events["p"], dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]

    edge_s, edge_t, attr_s, attr_t, _, _ = build_multigraph(
        t, x, y, p,
        sensor=(cfg.width, cfg.height),
        spatial_r_xy_frac=cfg.spatial_r_xy_frac,
        spatial_r_t_ms=cfg.spatial_r_t_ms,
        spatial_mmax=cfg.spatial_mmax,
        temporal_r_xy_frac=cfg.temporal_r_xy_frac,
        temporal_r_t_ms=cfg.temporal_r_t_ms,
        temporal_mmax=cfg.temporal_mmax,
    )

    # Causal normal flow on the temporal graph; smooth on spatial graph.
    # Input feature only — no ego-motion fitting / residual.
    vx, vy = node_flow(t, x, y, edge_t)
    vx, vy, _, _ = smooth_flow(
        vx, vy, edge_s, n_iters=FLOW_SMOOTH_ITERS, keep=FLOW_KEEP, verbose=False,
    )

    return {
        "t": t, "x": x, "y": y, "p": p,
        "vx": vx, "vy": vy,
        "edge_spatial": edge_s, "edge_temporal": edge_t,
        "attr_spatial": attr_s, "attr_temporal": attr_t,
        "order": order,
    }


def spatial_majority_smooth(
    pred: np.ndarray,
    edge_spatial: torch.Tensor,
    *,
    majority: float = 0.6,
) -> np.ndarray:
    """One-pass binary neighbour-majority flip on the spatial graph.

    Flip an event if a clear majority of its spatial neighbours disagree.
    """
    out = np.asarray(pred, dtype=bool).copy()
    n = len(out)
    if edge_spatial is None or edge_spatial.numel() == 0 or n == 0:
        return out

    rec = edge_spatial[0].cpu().numpy().astype(np.int64)
    src = edge_spatial[1].cpu().numpy().astype(np.int64)
    a = np.concatenate([rec, src])
    b = np.concatenate([src, rec])

    n_fg = np.zeros(n, dtype=np.float64)
    n_bg = np.zeros(n, dtype=np.float64)
    np.add.at(n_fg, a, out[b].astype(np.float64))
    np.add.at(n_bg, a, (~out[b]).astype(np.float64))
    deg = n_fg + n_bg
    has = deg > 0
    # Neighbours prefer FG
    flip_to_fg = has & (~out) & (n_fg / np.maximum(deg, 1) >= majority)
    flip_to_bg = has & out & (n_bg / np.maximum(deg, 1) >= majority)
    out[flip_to_fg] = True
    out[flip_to_bg] = False
    return out


class FrozenEncoder(nn.Module):
    """Loads GraphCNN from src/, always eval + no_grad. Returns [N, 2D]."""

    def __init__(self, cfg: GraphCfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        node_cbs, cb_s, cb_t, w_s, w_t = make_codebooks(
            sensor=(cfg.width, cfg.height), seed=SEED)
        self.node_cbs = node_cbs
        self.cb_spatial, self.cb_temporal = cb_s, cb_t
        self.w_spatial, self.w_temporal = w_s, w_t

        self.cnn_spatial = GraphCNN(
            input_dim=cfg.hv_dim, num_layers=cfg.num_layers, delta=1,
            graph_pooling_type="sum", neighbor_pooling_type="average",
            device=str(self.device), equation=10, edge_feat_dim=3,
            edge_projection_type="orthogonal", use_reservoir=False, rng_seed=SEED,
        )
        self.cnn_temporal = GraphCNN(
            input_dim=cfg.hv_dim, num_layers=cfg.num_layers, delta=1,
            graph_pooling_type="sum", neighbor_pooling_type="average",
            device=str(self.device), equation=10, edge_feat_dim=6,
            edge_projection_type="orthogonal", use_reservoir=False, rng_seed=SEED,
        )
        self.cnn_spatial.eval()
        self.cnn_temporal.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.buffers():
            p.requires_grad_(False)

    def encode_nodes(self, graph: dict[str, Any]) -> torch.Tensor:
        return _encode_nodes_xytp_motion(
            graph["x"], graph["y"], graph["t"], graph["p"],
            graph["vx"], graph["vy"], self.node_cbs,
        )

    @torch.no_grad()
    def forward(self, graph: dict[str, Any], node_hv: torch.Tensor | None = None):
        if node_hv is None:
            node_hv = self.encode_nodes(graph)
        node_hv = node_hv.to(self.device)
        eh_s = encode_edge_h_spatial(
            graph["attr_spatial"], self.cb_spatial, self.w_spatial).to(self.device)
        eh_t = encode_edge_h_temporal(
            graph["attr_temporal"], self.cb_temporal, self.w_temporal).to(self.device)
        g_s = EventGraph(node_hv, graph["edge_spatial"].to(self.device), edge_H=eh_s)
        g_t = EventGraph(node_hv, graph["edge_temporal"].to(self.device), edge_H=eh_t)
        H_s, _ = self.cnn_spatial([g_s], return_node_rep=True)
        H_t, _ = self.cnn_temporal([g_t], return_node_rep=True)
        return torch.cat([H_s, H_t], dim=1)


class Adapter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SegHead(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)


class SegModel(nn.Module):
    """Frozen encoder → mean-center → Adapter → SegHead. No raw (x,y) to head."""

    def __init__(self, cfg: GraphCfg | None = None):
        super().__init__()
        self.cfg = cfg or GraphCfg()
        self.encoder = FrozenEncoder(self.cfg)
        in_dim = 2 * self.cfg.hv_dim
        self.adapter = Adapter(in_dim, self.cfg.adapter_dim)
        self.head = SegHead(self.cfg.adapter_dim)

    def trainable_parameters(self):
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def assert_encoder_frozen(self) -> int:
        n_enc = 0
        for p in self.encoder.parameters():
            assert not p.requires_grad, "encoder parameter has requires_grad=True"
            n_enc += p.numel()
        for p in self.encoder.buffers():
            if getattr(p, "requires_grad", False):
                raise AssertionError("encoder buffer unexpectedly requires grad")
        n_train = sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)
        return n_train

    def forward(self, graph: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            H = self.encoder(graph)  # [N, 2D]
        # Position-dominance guard: mean-center per window before the head.
        H = H.detach()
        H = H - H.mean(dim=0, keepdim=True)
        return self.head(self.adapter(H))
