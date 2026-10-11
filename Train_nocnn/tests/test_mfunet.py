"""mfunet: U-Net head, evidence channels (event density, flow confidence), flip augmentation."""

import torch

from hdems.data.evimo import FLIPS, flip_sample
from hdems.models.hdems import HDEMS
from hdems.models.paper_flow import flow_from_cost
from hdems.models.segmentation import MotionUNetHead


def _cfield(d, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(1, d, h, w, generator=g), torch.randn(1, d, h, w, generator=g))


def test_confidence_is_zero_for_flat_matches_and_flow_is_unchanged():
    M, H, W = 5, 6, 7
    C = torch.zeros(1, M, M, H, W)
    C[0, 3, 1, 2, 2] = 1.0                                 # one clear match at one pixel
    flow, conf = flow_from_cost(C, M, alpha=0.85, return_confidence=True)
    assert torch.equal(flow, flow_from_cost(C, M, alpha=0.85))
    assert conf.shape == (1, 1, H, W)
    assert conf[0, 0, 2, 2] > 0.9 and conf[0, 0, 0, 0] == 0   # flat elsewhere -> 0


def test_unet_shapes_on_odd_sizes():
    head = MotionUNetHead(16, 2, motion_dim=8, app_dim=4, scalar_ch=8).eval()
    for h, w in ((15, 20), (16, 20), (33, 41)):
        out = head(_cfield(16, h, w, 1), _cfield(16, h, w, 2), torch.randn(1, 8, h, w))
        assert out.shape == (1, 2, h, w)


def test_unet_appearance_dropout_also_drops_event_density():
    torch.manual_seed(0)
    head = MotionUNetHead(16, 2, motion_dim=8, app_dim=4, app_dropout=1.0, scalar_ch=8,
                          app_scalars=(6,))
    mv, x = _cfield(16, 12, 12, 1), _cfield(16, 12, 12, 2)
    sc = torch.randn(1, 8, 12, 12)
    no_app = sc.clone()
    no_app[:, 6] = 0
    head.train()
    assert torch.allclose(head(mv, x, sc), head(mv, torch.zeros_like(x), no_app), atol=1e-6)


def test_unet_sees_far_beyond_5x5():
    head = MotionUNetHead(8, 2, motion_dim=8, app_dim=0, scalar_ch=8).eval()
    # measure the CONVOLUTIONS only: GroupNorm's whole-image statistics would make
    # every output depend on every input and pass this test trivially
    for m in head.modules():
        for name, child in m.named_children():
            if isinstance(child, torch.nn.GroupNorm):
                setattr(m, name, torch.nn.Identity())
    n = 257
    sc = torch.zeros(1, 8, n, n, requires_grad=True)
    out = head(_cfield(8, n, n, 1), _cfield(8, n, n, 2), sc)
    out[0, 1, n // 2, n // 2].backward()
    rows = sc.grad.abs().sum((0, 1, 3)).nonzero()
    assert int(rows.max() - rows.min()) + 1 >= 100        # receptive field >= 100 px


def test_flip_sample_flips_every_image_and_nothing_else():
    s = {"surface": torch.arange(4 * 2 * 3 * 5.0).view(4, 2, 3, 5),
         "mask": torch.arange(15).view(3, 5), "score_mask": torch.rand(3, 5) > 0.5,
         "moving_ids": torch.tensor([1, 2, 3])}
    for dims in FLIPS:
        f = flip_sample(s, dims)
        assert torch.equal(flip_sample(f, dims)["surface"], s["surface"])   # own inverse
        assert torch.equal(f["moving_ids"], s["moving_ids"])
        if dims:
            assert torch.equal(f["mask"], torch.flip(s["mask"], dims))
            assert torch.equal(f["score_mask"], torch.flip(s["score_mask"], dims))
    assert torch.equal(flip_sample(s, (-1,))["surface"][..., 0], s["surface"][..., -1])


def _tiny(head):
    return {
        "d": 32,
        "encoder": {"patch_size": 5, "sigma_k": 1.0, "kernel": "conv",
                    "polarity_binding": True, "scales": 1},
        "matching": {"M": 5, "scales": [0, 1, 2], "alpha": 0.3, "smooth": 3},
        "dataset": {"height": 16, "width": 20, "time_frames": [1.0, 0.75, 0.5, 0.0]},
        "segmentation": {"head": head, "num_classes": 2, "mf_motion_dim": 8,
                         "mf_widths": [8, 16, 16, 16]},
        "velocity": {"event_feature": "f"},
        "flow_cache": {"enabled": False},
    }


def test_mfunet_forward_and_ablations():
    model = HDEMS(_tiny("mfunet")).eval()
    s = (torch.rand(1, 4, 2, 16, 20, generator=torch.Generator().manual_seed(3)) > 0.7).float()
    with torch.no_grad():
        flow, conf = model.compute_motion(s)
        assert flow.shape == (1, 2, 16, 20) and conf.shape == (1, 1, 16, 20)
        assert torch.equal(model.compute_flow(s), flow)
        full = model(s, task="segmentation")["seg_logits"]
        assert full.shape == (1, 2, 16, 20)
        ev = model.evidence_channels(s, conf)
        model.ablate = "motion"
        assert model.evidence_channels(s, conf)[:, 1].abs().max() == 0      # no confidence
        assert torch.equal(model.evidence_channels(s, conf)[:, 0], ev[:, 0])
        model(s, task="segmentation")
        model.ablate = "appearance"
        assert model.evidence_channels(s, conf)[:, 0].abs().max() == 0      # no density
        model(s, task="segmentation")
        model.ablate = None
