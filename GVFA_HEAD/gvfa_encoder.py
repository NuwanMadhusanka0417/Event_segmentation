"""Adapter: event graph -> edge-conditioned GVFA (graphcnnVSA_Binding_FULL_new)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from fpe_codebook import FPECodebook, bundle_weighted

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graphcnnVSA_Binding_FULL_new import GraphCNN  # noqa: E402


class EventGraph:
    """Single-event-graph container expected by GraphCNN.forward(batch_graph)."""

    def __init__(self, node_features, edge_index, edge_attr=None, edge_H=None):
        """
        edge_index : [2, E] long, segment convention [receiver, source] (later, earlier).
        edge_attr  : optional raw [E, F] (kept for debug; GVFA uses edge_H when set).
        edge_H     : optional precomputed [E, D] edge hypervectors.
        """
        n = node_features.shape[0]
        self.g = list(range(n))
        self.node_features = node_features
        self.edge_mat = edge_index.long()
        self.edge_index = edge_index[[1, 0], :].long()
        self.edge_attr = edge_attr.float() if edge_attr is not None else None
        self.edge_H = edge_H.float() if edge_H is not None else None


def encode_edge_h_spatial(
    edge_attr: np.ndarray,
    codebooks: dict[str, FPECodebook],
    weights: dict[str, float],
) -> torch.Tensor:
    """FPE-bundle spatial edge attrs (dx, dy, dt) -> [E, D] L2-normalized."""
    dx, dy, dt = edge_attr[:, 0], edge_attr[:, 1], edge_attr[:, 2]
    dt_us = dt * 1e6
    terms = [
        (codebooks["dx"].encode(dx), weights["dx"]),
        (codebooks["dy"].encode(dy), weights["dy"]),
        (codebooks["dt"].encode(dt_us, interpolate=True), weights["dt"]),
    ]
    return bundle_weighted(terms)


def encode_edge_h_temporal(
    edge_attr: np.ndarray,
    codebooks: dict[str, FPECodebook],
    weights: dict[str, float],
) -> torch.Tensor:
    """FPE-bundle temporal edge attrs (dx,dy,dt,vx,vy,dp) -> [E, D]."""
    dx, dy, dt = edge_attr[:, 0], edge_attr[:, 1], edge_attr[:, 2]
    vx, vy, dp = edge_attr[:, 3], edge_attr[:, 4], edge_attr[:, 5]
    dt_us = dt * 1e6
    terms = [
        (codebooks["dx"].encode(dx), weights["dx"]),
        (codebooks["dy"].encode(dy), weights["dy"]),
        (codebooks["dt"].encode(dt_us, interpolate=True), weights["dt"]),
        (codebooks["vx"].encode(vx, interpolate=True), weights["vx"]),
        (codebooks["vy"].encode(vy, interpolate=True), weights["vy"]),
        (codebooks["dp"].encode(dp), weights["dp"]),
    ]
    return bundle_weighted(terms)


def encode_graph(
    node_hv,
    edge_index,
    edge_attr=None,
    *,
    edge_H=None,
    graph_kind: str = "spatial",
    edge_codebooks: dict[str, FPECodebook] | None = None,
    edge_weights: dict[str, float] | None = None,
    num_layers: int = 3,
    edge_feat_dim: int = 3,
    device: str = "cpu",
    use_reservoir: bool = False,
    delta: int = 1,
    equation: int = 10,
    rng_seed: int = 0,
):
    """Run edge-conditioned GVFA; return contextual node hypervectors [N, D]."""
    if edge_H is None and edge_attr is not None and edge_codebooks is not None:
        if isinstance(edge_attr, np.ndarray):
            ea = edge_attr
        else:
            ea = edge_attr.detach().cpu().numpy()
        if graph_kind == "spatial":
            edge_H = encode_edge_h_spatial(ea, edge_codebooks, edge_weights)
        else:
            edge_H = encode_edge_h_temporal(ea, edge_codebooks, edge_weights)

    ea_t = None
    if edge_attr is not None and isinstance(edge_attr, np.ndarray):
        ea_t = torch.from_numpy(edge_attr.astype(np.float32))
    elif edge_attr is not None:
        ea_t = edge_attr.float()

    graph = EventGraph(node_hv, edge_index, edge_attr=ea_t, edge_H=edge_H)
    model = GraphCNN(
        input_dim=node_hv.shape[1],
        num_layers=num_layers,
        delta=delta,
        graph_pooling_type="sum",
        neighbor_pooling_type="average",
        device=device,
        equation=equation,
        edge_feat_dim=edge_feat_dim,
        edge_projection_type="orthogonal",
        use_reservoir=use_reservoir,
        rng_seed=rng_seed,
    )
    model.eval()
    with torch.no_grad():
        H, _ = model([graph], return_node_rep=True)
    return H
