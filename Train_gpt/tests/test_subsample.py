import torch

from vsa_motionseg.training.subsample import subsample_labeled


def test_subsample_cap():
    Q = torch.randn(1000, 8, dtype=torch.complex64)
    y = torch.zeros(1000, dtype=torch.long)
    y[500:] = 1
    Q2, y2 = subsample_labeled(Q, y, 200, stratified=True, seed=0)
    assert Q2.shape[0] == 200
    assert (y2 == 0).sum() > 0
    assert (y2 == 1).sum() > 0


def test_subsample_noop():
    Q = torch.randn(10, 4)
    y = torch.zeros(10, dtype=torch.long)
    Q2, y2 = subsample_labeled(Q, y, None)
    assert Q2.shape[0] == 10
