import torch

from vsa_motionseg.motion.cost_volume import cost_to_flow, cost_volume_two_fields
from vsa_motionseg.motion.ego_motion import ransac_global_translation
from vsa_motionseg.vsa.prototypes import predict_prototypes, train_prototypes


def test_cost_volume_shape():
    B, d, H, W = 1, 32, 8, 8
    F0 = torch.randn(B, d, H, W, dtype=torch.complex64)
    F1 = torch.roll(F0, shifts=(0, 1), dims=(2, 3))
    C = cost_volume_two_fields(F0, F1, M=5)
    assert C.shape == (B, 5, 5, H, W)
    out = cost_to_flow(C, temperature=0.1, delta_t=0.05)
    assert out["flow"].shape == (B, 2, H, W)
    assert out["flow"][0, 0].abs().mean() > 0


def test_ransac_failure():
    flow = torch.zeros(2, 4, 4)
    valid = torch.zeros(4, 4, dtype=torch.bool)
    r = ransac_global_translation(flow, valid, min_inliers=50)
    assert r["success"] is False


def test_prototype_train_predict():
    d = 64
    Q = torch.randn(100, d, dtype=torch.complex64)
    y = torch.zeros(100, dtype=torch.long)
    y[50:] = 1
    P = train_prototypes(Q, y, bipolar=False)
    pred, conf = predict_prototypes(Q, P, threshold=-2.0)
    assert pred.shape == (100,)
    assert pred.max() <= 1 and pred.min() >= 0
