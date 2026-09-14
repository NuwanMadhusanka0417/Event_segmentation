"""Synthetic moving-square scene on CPU."""

import torch

from vsa_motionseg.config import load_config
from vsa_motionseg.evaluation.metrics import iou
from vsa_motionseg.pipeline import VSAMotionSegPipeline


def _square_events(x0, y0, size, t0, t1, vx, vy, p=1.0):
    rows = []
    t = t0
    while t <= t1:
        cx = x0 + vx * (t - t0)
        cy = y0 + vy * (t - t0)
        for dy in range(size):
            for dx in range(size):
                rows.append([t, cx + dx, cy + dy, p])
        t += 0.002
    return torch.tensor(rows, dtype=torch.float32)


def test_synthetic_two_motions():
    cfg = load_config(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    cfg["vsa"]["dimension"] = 128
    cfg["vsa"]["basis_rank"] = 8
    cfg["flow"]["search_radius"] = 2
    cfg["flow"]["confidence_threshold"] = 0.01
    cfg["vsa"]["artifact_dir"] = None
    cfg["clustering"]["min_cluster_size"] = 5
    cfg["clustering"]["flow_threshold"] = 20.0
    pipe = VSAMotionSegPipeline(cfg)

    static = _square_events(5, 5, 6, 0.0, 0.05, 0, 0)
    move_a = _square_events(40, 40, 8, 0.0, 0.05, 4.0, 0.0)
    move_b = _square_events(40, 80, 8, 0.0, 0.05, 0.0, 3.0)
    noise = torch.tensor([[0.01, 100.0, 100.0, 1.0]])
    events = torch.cat([static, move_a, move_b, noise], dim=0)

    r = pipe.process_window(events, 128, 128, t_end=0.05, window_s=0.05)
    assert r["motion_hypervector"].shape[0] == cfg["vsa"]["dimension"]
    assert r["active_mask"].any()
    dyn = r["dynamic_mask"]
    assert dyn.sum() > 0
    labels = r["segment_labels"]
    n_seg = len(set(labels.unique().tolist()) - {0})
    assert n_seg >= 1
    assert r["residual_flow"].shape == (2, 128, 128)


def test_metrics_iou():
    pred = torch.tensor([[0, 1], [1, 1]])
    tgt = torch.tensor([[0, 1], [0, 1]])
    v = torch.ones(2, 2, dtype=torch.bool)
    assert iou(pred, tgt, v) > 0.5
