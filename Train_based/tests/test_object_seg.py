"""Instance embedding + object-level decisions (hdems/object_seg.py, discriminative loss,
MotionUNetHead instance output) on synthetic maps."""

import numpy as np
import torch

from hdems.losses.seg import discriminative_loss
from hdems.models.segmentation import MotionUNetHead
from hdems.object_seg import (ObjectMemory, ObjectParams, cluster_embeddings, object_scores,
                              segment_objects, split_spatially)

H, W = 40, 60


def _instances():
    """Background 0, object 1 (left), object 2 (right) and object 3 = same look as 1, far away."""
    inst = torch.zeros(H, W, dtype=torch.long)
    inst[5:15, 5:15] = 1
    inst[20:35, 20:35] = 2
    inst[5:15, 45:55] = 3
    return inst


def _embedding(inst, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])   # 3 looks like 1
    return (centres[inst].permute(2, 0, 1) + noise * torch.randn(2, H, W, generator=g)).float()


def test_discriminative_loss_separated_vs_mixed():
    inst = _instances()
    valid = torch.ones(1, H, W, dtype=torch.bool)
    centres = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])      # all distinct
    good = (centres[inst].permute(2, 0, 1) + 0.05 * torch.randn(2, H, W)).float()[None]
    bad = torch.randn(1, 2, H, W, generator=torch.Generator().manual_seed(1)) * 0.1
    lg = discriminative_loss(good, inst[None], valid)
    lb = discriminative_loss(bad, inst[None], valid)
    assert lg < 0.05 and lb > 1.0


def test_discriminative_loss_trains_an_embedding():
    inst = _instances()[None]
    valid = torch.ones(1, H, W, dtype=torch.bool)
    feat = torch.stack([inst[0].float() == k for k in range(4)]).float()[None]   # one-hot input
    net = torch.nn.Conv2d(4, 3, 1)
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = discriminative_loss(net(feat), inst, valid)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.1 * first


def test_clustering_and_spatial_split_recover_objects():
    inst = _instances()
    emb = _embedding(inst)
    mask = torch.ones(H, W, dtype=torch.bool)
    p = ObjectParams(bandwidth=1.0, link_px=1)
    objects = split_spatially(cluster_embeddings(emb, mask, p), p.link_px)
    # every GT instance maps to exactly one predicted object, and 1 and 3 stay apart
    for k in range(4):
        assert len(np.unique(objects[inst.numpy() == k])) == 1
    assert objects[inst.numpy() == 1][0] != objects[inst.numpy() == 3][0]
    assert len(np.unique(objects)) == 4


def test_object_decision_labels_whole_object():
    """Half of object 2 has weak per-pixel evidence -- the object decision still takes all of it."""
    inst = _instances()
    emb = _embedding(inst)
    prob = torch.full((H, W), 0.05)
    o2 = inst == 2
    prob[o2] = 0.9
    prob[20:35, 20:28] = 0.3                      # the slow half
    mask = torch.ones(H, W, dtype=torch.bool)
    p = ObjectParams(threshold=0.5, link_px=1)
    moving, _objects, _s = segment_objects(prob, emb, mask, p)
    assert moving[o2.numpy()].all()
    assert not moving[~o2.numpy()].any()
    assert ((prob > 0.5).numpy() & o2.numpy()).sum() < o2.sum()   # per pixel would miss part


def test_memory_keeps_a_pausing_object_moving():
    objects = np.zeros((H, W), np.int64)
    objects[10:20, 10:20] = 1
    p = ObjectParams(memory=0.9, threshold=0.5)
    mem = ObjectMemory(p)
    s1 = mem("seq", objects, np.array([0.0, 0.95]))
    assert s1[1] == 0.95
    shifted = np.zeros_like(objects)
    shifted[11:21, 12:22] = 1                     # moved 1-2 px, pauses: own score 0.2
    s2 = mem("seq", shifted, np.array([0.0, 0.2]))
    assert s2[1] > 0.5
    s_other = mem("other_seq", shifted, np.array([0.0, 0.2]))     # memory is per sequence
    assert s_other[1] == 0.2
    assert ObjectMemory(ObjectParams(memory=0.0))("seq", shifted, np.array([0.0, 0.2]))[1] == 0.2


def test_object_scores_mean_probability():
    objects = np.array([[0, 1, 1], [2, 2, 0]])
    prob = np.array([[0.9, 0.2, 0.4], [1.0, 0.0, 0.7]])
    s = object_scores(prob, objects)
    assert np.allclose(s, [0.0, 0.3, 0.5])


def test_unet_head_instance_output_shapes():
    head = MotionUNetHead(16, 2, motion_dim=8, app_dim=4, scalar_ch=8, widths=(8, 16), mv_ch=16,
                          instance_dim=6)
    mv = torch.randn(1, 16, 24, 32)
    x = torch.randn(1, 16, 24, 32, dtype=torch.complex64)
    out = head(mv, x, torch.randn(1, 8, 24, 32))
    assert isinstance(out, tuple)
    logits, emb = out
    assert logits.shape == (1, 2, 24, 32) and emb.shape == (1, 6, 24, 32)
    plain = MotionUNetHead(16, 2, motion_dim=8, app_dim=4, scalar_ch=8, widths=(8, 16), mv_ch=16)
    assert plain(mv, x, torch.randn(1, 8, 24, 32)).shape == (1, 2, 24, 32)
