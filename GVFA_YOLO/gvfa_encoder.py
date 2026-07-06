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

from graph import build_multigraph, SEED, SENSOR as DEFAULT_SENSOR
from fpe_codebook import FPECodebook
from pool import voxel_max_pool, POOL_CELL


class EventGraph:
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
    """Frozen two-graph GVFA + FPE codebook nodes + optional voxel pooling."""

    def __init__(self, dim=4000, num_layers=3, device="cpu", seed=SEED,
                 sensor=None, pool=True, pool_cell=POOL_CELL, verbose_init=False):
        super().__init__()
        self.dim = dim
        self.device = device
        self.seed = seed
        self.sensor = sensor or DEFAULT_SENSOR
        self.pool = pool
        self.pool_cell = pool_cell
        self.last_stats = {}

        if verbose_init:
            print(f"[init] building FPE codebook (D={dim})...")
        self.codebook = FPECodebook(dim=dim, sensor=self.sensor, seed=seed)

        if verbose_init:
            print("[init] loading frozen GraphCNN (spatial + temporal)...")
        common = dict(
            input_dim=dim, num_layers=num_layers, delta=1,
            graph_pooling_type="sum", neighbor_pooling_type="average",
            device=device, equation=10, edge_projection_type="orthogonal",
            use_reservoir=False, rng_seed=seed,
        )
        self.gcn_spatial = GraphCNN(**common, edge_feat_dim=3)
        self.gcn_temporal = GraphCNN(**common, edge_feat_dim=6)
        for p in self.parameters():
            p.requires_grad_(False)
        if verbose_init:
            print("[init] GVFAEncoder ready (frozen)")

    @torch.no_grad()
    def forward(self, events, pool=None, verbose=False):
        if pool is None:
            pool = self.pool
        ev = np.asarray(events, dtype=np.float64)
        if ev.size == 0 or ev.shape[0] == 0:
            if verbose:
                print("[GVFA] empty window — skip")
            self.last_stats = {"n_events": 0, "n_pooled": 0, "edge_temporal_dim": 0}
            z = torch.zeros((0, self.dim), device=self.device)
            return z, torch.zeros((0, 2), device=self.device, dtype=torch.float32)

        if verbose:
            print("[GVFA] --- pipeline start ---")

        t_us, x, y, p = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
        t_sec = t_us / 1e6
        t_win0 = float(t_us.min())
        if verbose:
            print(f"[GVFA] 1/7 events loaded: N={ev.shape[0]}")

        (edge_spatial, edge_temporal,
         attr_spatial, attr_temporal, _, _,
         vx_edge, vy_edge) = build_multigraph(t_sec, x, y, p, sensor=self.sensor)
        if verbose:
            print(f"[GVFA] 2/7 graphs built: spatial E={edge_spatial.shape[1]}, "
                  f"temporal E={edge_temporal.shape[1]}")

        x_hv = self.codebook.encode_nodes(x, y, t_us, p, t_window_start_us=t_win0)
        if verbose:
            print(f"[GVFA] 3/7 node codebook lookup done: {tuple(x_hv.shape)}")

        edge_hv_s = edge_hv_t = None
        if attr_spatial.shape[0]:
            edge_hv_s = self.codebook.encode_edges_spatial(
                attr_spatial[:, 0], attr_spatial[:, 1], attr_spatial[:, 2])
        if attr_temporal.shape[0]:
            edge_hv_t = self.codebook.encode_edges_temporal(
                attr_temporal[:, 0], attr_temporal[:, 1], attr_temporal[:, 2],
                vx_edge, vy_edge, attr_temporal[:, 5])
        if verbose:
            es = 0 if edge_hv_s is None else edge_hv_s.shape[0]
            et = 0 if edge_hv_t is None else edge_hv_t.shape[0]
            print(f"[GVFA] 4/7 edge codebook: spatial={es}, temporal={et}")

        if verbose:
            print("[GVFA] 5/7 applying spatial GraphCNN...")
        H_spatial = _run_graph(
            self.gcn_spatial, x_hv, edge_spatial, attr_spatial, self.device)
        if verbose:
            print("[GVFA] 6/7 applying temporal GraphCNN...")
        H_temporal = _run_graph(
            self.gcn_temporal, x_hv, edge_temporal, attr_temporal, self.device)

        H = F.normalize(H_spatial + H_temporal, p=2, dim=1)
        pos = torch.tensor(np.stack([x, y], 1), dtype=torch.float32, device=self.device)

        n_events = int(H.shape[0])
        if pool:
            if verbose:
                print(f"[GVFA] 7/7 voxel max-pool (cell={self.pool_cell}px)...")
            H, pos = voxel_max_pool(H, pos, cell=self.pool_cell)
        n_pooled = int(H.shape[0])

        if verbose:
            print(f"[GVFA] DONE: {n_events} events -> {n_pooled} pooled nodes  "
                  f"H={tuple(H.shape)}")

        self.last_stats = {
            "n_events": n_events,
            "n_pooled": n_pooled,
            "codebook_mb": self.codebook.size_mb,
            "edge_spatial_dim": 0 if edge_hv_s is None else edge_hv_s.shape[0],
            "edge_temporal_dim": 0 if edge_hv_t is None else edge_hv_t.shape[0],
        }
        return H, pos
