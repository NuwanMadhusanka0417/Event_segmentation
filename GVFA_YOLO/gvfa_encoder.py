"""Edge-conditioned GVFA encoder (graphcnnVSA_Binding_FULL_new from src/)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graphcnnVSA_Binding_FULL_new import GraphCNN  # noqa: E402

from graph import build_multigraph, fpe_encode, SEED, SENSOR as DEFAULT_SENSOR


class EventGraph:
    """Single-event-graph container expected by GraphCNN.forward(batch_graph)."""

    def __init__(self, node_features, edge_index, edge_attr):
        n = node_features.shape[0]
        self.g = list(range(n))
        self.node_features = node_features
        self.edge_mat = edge_index.long()
        self.edge_index = edge_index[[1, 0], :].long()
        self.edge_attr = edge_attr.float()


def _run_graph(model, node_hv, edge_index, edge_attr, device):
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.from_numpy(edge_attr.astype(np.float32))
    graph = EventGraph(node_hv, edge_index, edge_attr)
    model.eval()
    with torch.no_grad():
        H, _ = model([graph], return_node_rep=True)
    return H.to(device)


class GVFAEncoder(nn.Module):
    """Two-graph edge-conditioned GVFA -> single bundled hypervector per node [N, D].

    Pipeline:
      events [t_us, x, y, p]
        -> causal ellipsoid spatial + temporal graphs (GVFA/segment.py)
        -> FPE node encoding {x, y, t}
        -> frozen GraphCNN on each graph with edge features (Eq. 5 / Eq. 6)
        -> bundle H = normalize(H_spatial + H_temporal)
    """

    def __init__(self, dim=4000, num_layers=3, device="cpu", seed=SEED,
                 sensor=None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.seed = seed
        self.sensor = sensor or DEFAULT_SENSOR
        common = dict(
            input_dim=dim,
            num_layers=num_layers,
            delta=1,
            graph_pooling_type="sum",
            neighbor_pooling_type="average",
            device=device,
            equation=10,
            edge_projection_type="orthogonal",
            use_reservoir=False,
            rng_seed=seed,
        )
        self.gcn_spatial = GraphCNN(**common, edge_feat_dim=3)
        self.gcn_temporal = GraphCNN(**common, edge_feat_dim=6)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, events):
        ev = np.asarray(events, dtype=np.float64)
        t_us, x, y, p = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
        t_sec = t_us / 1e6

        (edge_spatial, edge_temporal,
         attr_spatial, attr_temporal, _, _) = build_multigraph(
            t_sec, x, y, p, sensor=self.sensor)

        x_hv = fpe_encode(x, y, t_sec, dim=self.dim, seed=self.seed,
                          sensor=self.sensor)

        H_spatial = _run_graph(
            self.gcn_spatial, x_hv, edge_spatial, attr_spatial, self.device)
        H_temporal = _run_graph(
            self.gcn_temporal, x_hv, edge_temporal, attr_temporal, self.device)

        H = F.normalize(H_spatial + H_temporal, p=2, dim=1)

        pos = torch.tensor(
            np.stack([x, y], axis=1), dtype=torch.float32, device=self.device)
        return H, pos
