"""Training-free VSA grouping on synthetic flow fields (exact ground truth, no events needed)."""

from types import SimpleNamespace

import numpy as np
import torch

from hdems.vsa.fpe import make_base_phases
from hdems.vsa_grouping import BackgroundPrior, VSAGroupingParams, encode_pixels, vsa_group_objects

H, W, D = 96, 128, 512
MODEL = SimpleNamespace(matcher=SimpleNamespace(phx=make_base_phases(D, seed=0),
                                                phy=make_base_phases(D, seed=1)),
                        phi_vx=make_base_phases(D, seed=7), phi_vy=make_base_phases(D, seed=8))
P = VSAGroupingParams(sigma_s=15.0, sigma_v=0.25, merge_vel=0.25, min_object_px=40,
                      min_motion=0.15, background="border", smooth_labels_px=3)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def _events(density=0.4, seed=0):
    return np.random.default_rng(seed).random((H, W)) < density


def _noisy(flow, seed=0, sd=0.05):
    return (flow + np.random.default_rng(seed).normal(0, sd, flow.shape)).astype(np.float32)


def _box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def _match(objects, truth, ev):
    """Best IoU of any predicted object with the true object mask (on event pixels)."""
    return max((_iou((objects == i) & ev, truth & ev) for i in np.unique(objects) if i > 0), default=0.0)


def test_encoding_kernel():
    d = 4096
    ph = [make_base_phases(d, seed=s) for s in (0, 1, 7, 8)]
    xs, ys = torch.tensor([10.0, 25.0]), torch.tensor([5.0, 13.0])
    vx, vy = torch.tensor([0.5, 0.8]), torch.tensor([-0.2, 0.1])
    Hc = encode_pixels(xs, ys, vx, vy, ph, sigma_s=15.0, sigma_v=0.5)
    sim = float(Hc[0] @ Hc[1]) / d
    dpos2 = (15.0 ** 2 + 8.0 ** 2) / 15.0 ** 2
    dvel2 = (0.3 ** 2 + 0.3 ** 2) / 0.5 ** 2
    assert abs(sim - np.exp(-dpos2 / 2) * np.exp(-dvel2 / 2)) < 0.05


def test_one_object_on_static_background():
    ev = _events()
    obj = _box(30, 60, 40, 80)
    flow = np.zeros((2, H, W), np.float32)
    flow[0][obj] = 2.0
    objects, info = vsa_group_objects(_noisy(flow), ev, MODEL, P)
    assert len(np.unique(objects[objects > 0])) == 1
    assert _match(objects, obj, ev) > 0.9


def test_two_objects_different_velocities():
    ev = _events(seed=1)
    a, b = _box(10, 40, 10, 45), _box(50, 85, 70, 115)
    flow = np.zeros((2, H, W), np.float32)
    flow[0][a], flow[1][b] = 2.0, -1.5
    objects, _ = vsa_group_objects(_noisy(flow, 1), ev, MODEL, P)
    assert len(np.unique(objects[objects > 0])) == 2
    assert _match(objects, a, ev) > 0.85 and _match(objects, b, ev) > 0.85


def test_rotating_disc_is_one_object():
    ev = _events(seed=2)
    cy, cx, r, w = 48.0, 64.0, 25.0, 0.05                     # 0.05 rad per interval
    disc = (YY - cy) ** 2 + (XX - cx) ** 2 < r ** 2
    flow = np.zeros((2, H, W), np.float32)
    flow[0][disc] = (-w * (YY - cy))[disc]
    flow[1][disc] = (w * (XX - cx))[disc]
    objects, _ = vsa_group_objects(_noisy(flow, 2), ev, MODEL, P)
    assert len(np.unique(objects[objects > 0])) == 1
    assert _match(objects, disc, ev) > 0.85


def test_moving_camera_background_stays_one_cluster():
    ev = _events(seed=3)
    flow = np.stack([1.0 + 0.01 * XX, -0.5 + 0.008 * YY]).astype(np.float32)   # camera: 1.0-2.3 px
    obj = _box(35, 65, 50, 90)
    flow[0][obj], flow[1][obj] = -1.0, 1.0
    objects, info = vsa_group_objects(_noisy(flow, 3), ev, MODEL, P)
    assert info["n_init"] > 3                                 # k-means over-segments ...
    assert len(np.unique(objects[objects > 0])) == 1          # ... the merge rejoins the background
    assert _match(objects, obj, ev) > 0.85


def test_temporal_prior_handles_a_large_mover():
    ev = _events(seed=4)
    board = _box(0, H, 0, int(0.6 * W))                      # 60% of the image, touches 3 borders
    prior = BackgroundPrior()
    still = np.zeros((2, H, W), np.float32)
    p_t = VSAGroupingParams(**{**P.__dict__, "background": "temporal"})
    vsa_group_objects(_noisy(still, 4), ev, MODEL, p_t, prior=prior, seq_id="s", frame_index=1)
    moving = still.copy()
    moving[0][board] = 1.5
    objects, info = vsa_group_objects(_noisy(moving, 5), ev, MODEL, p_t,
                                      prior=prior, seq_id="s", frame_index=2)
    assert info["bg_source"] == "temporal"
    assert _match(objects, board, ev) > 0.9                   # the board is the object
    p_e = VSAGroupingParams(**{**P.__dict__, "background": "extent"})
    wrong, _ = vsa_group_objects(_noisy(moving, 5), ev, MODEL, p_e)
    assert _match(wrong, board, ev) < 0.1                     # extent calls the board background


def test_prior_is_per_sequence_and_expires():
    prior = BackgroundPrior(max_gap=5)
    prior.update("a", 10, torch.ones(4))
    assert prior.get("a", 12) is not None
    assert prior.get("a", 16) is None and prior.get("b", 11) is None and prior.get("a", 10) is None


def test_mixed_cluster_is_split_but_a_smooth_field_is_not():
    from hdems.vsa_grouping import _split_mixed
    ev = _events(seed=8)
    lab = np.where(ev, 0, -1)                                  # ONE cluster over the image
    flow = _noisy(np.stack([1.0 + 0.01 * XX, -0.5 + 0.008 * YY]), 8)   # smooth camera field
    assert len(np.unique(_split_mixed(lab, flow, P)[ev])) == 1
    obj = _box(30, 60, 40, 80)
    flow[0][obj], flow[1][obj] = -1.0, 1.0                     # + an object: two velocity groups
    split = _split_mixed(lab, flow, P)
    parts = [split[ev & obj], split[ev & ~obj]]
    assert len(np.unique(split[ev])) >= 2
    assert len(np.unique(parts[0])) == 1 and parts[0][0] not in np.unique(parts[1])


def test_hull_iou_fills_sparse_objects():
    from hdems.instances import hull_iou
    gt = _box(20, 60, 30, 90).astype(np.int64)                # dense GT object
    sparse = (_box(20, 60, 30, 90) & _events(0.2, seed=7)).astype(np.int64)
    assert hull_iou(sparse, gt) > 0.9                          # hull of sparse events ~ the object
    assert hull_iou(np.zeros_like(gt), gt) == 0.0
    assert np.isnan(hull_iou(np.zeros_like(gt), np.zeros_like(gt)))


def test_deterministic():
    ev = _events(seed=6)
    flow = np.zeros((2, H, W), np.float32)
    flow[0][_box(20, 50, 20, 60)] = 1.2
    a, _ = vsa_group_objects(_noisy(flow, 6), ev, MODEL, P)
    b, _ = vsa_group_objects(_noisy(flow, 6), ev, MODEL, P)
    assert np.array_equal(a, b)
