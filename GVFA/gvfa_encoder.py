"""Adapter: event graph -> new edge-conditioned GVFA (graphcnnVSA_Binding_FULL_new)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graphcnnVSA_Binding_FULL_new import GraphCNN  # noqa: E402


class EventGraph:
    """Single-event-graph container expected by GraphCNN.forward(batch_graph)."""

    def __init__(self, node_features, edge_index, edge_attr):
        """
        edge_index : [2, E] long, segment convention [receiver, source] (later, earlier).
        edge_attr  : [E, F] float, Cartesian deltas source -> receiver.
        """
        n = node_features.shape[0]
        self.g = list(range(n))
        self.node_features = node_features
        self.edge_mat = edge_index.long()
        # Edge-conditioned MP expects [source, destination].
        self.edge_index = edge_index[[1, 0], :].long()
        self.edge_attr = edge_attr.float()


def encode_graph(
    node_hv,
    edge_index,
    edge_attr,
    *,
    num_layers=3,
    edge_feat_dim=3,
    device="cpu",
    use_reservoir=False,
    delta=1,
    equation=10,
    rng_seed=0,
):
    """Run edge-conditioned GVFA; return contextual node hypervectors [N, D]."""
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.from_numpy(edge_attr.astype(np.float32))
    graph = EventGraph(node_hv, edge_index, edge_attr)
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
