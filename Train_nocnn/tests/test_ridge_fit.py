"""Ridge readout fit / head smoke tests."""

from __future__ import annotations

import torch

from hdems.ridge_fit import fit_ridge_streaming, save_ridge_weights, load_ridge_weights
from hdems.ridge_head import RidgeHead
from hdems.seg_features import prepare_seg_features


def test_streaming_ridge_fit_and_head_forward() -> None:
    torch.manual_seed(0)
    n, d, num_classes = 400, 16, 5
    x = torch.randn(n, d)
    y = torch.randint(0, num_classes, (n,))
    result = fit_ridge_streaming(
        [(x, y)],
        num_classes=num_classes,
        alpha=1.0,
        imbalance="none",
        feature_mean=torch.zeros(d),
        mean_center=False,
        motion_features=False,
    )
    assert result.weight.shape == (d, num_classes)
    assert result.bias.shape == (num_classes,)

    head = RidgeHead(num_classes, mean_center=False, motion_features=False)
    head.set_from_result(result)
    phi = torch.complex(
        torch.randn(1, d // 2, 4, 4),
        torch.randn(1, d // 2, 4, 4),
    )
    logits = head(phi, surface=None)
    assert logits.shape == (1, num_classes, 4, 4)


def test_ridge_checkpoint_roundtrip(tmp_path) -> None:
    d, num_classes = 8, 3
    x = torch.randn(50, d)
    y = torch.randint(0, num_classes, (50,))
    result = fit_ridge_streaming(
        [(x, y)],
        num_classes=num_classes,
        alpha=0.1,
        imbalance="balanced",
    )
    path = tmp_path / "ridge.pt"
    save_ridge_weights(str(path), result, extra={"val_miou": 0.42})
    ckpt = load_ridge_weights(str(path))
    assert ckpt["feature_dim"] == d
    assert ckpt["val_miou"] == 0.42

    head = RidgeHead(num_classes)
    head.load(path)
    assert head.is_loaded
    assert head.num_readout_params() == d * num_classes + num_classes


def test_prepare_seg_features_motion_requires_surface() -> None:
    phi = torch.complex(torch.ones(1, 4, 2, 2), torch.zeros(1, 4, 2, 2))
    surface = torch.randn(1, 2, 2, 2)
    out = prepare_seg_features(phi, surface, mean_center=False, motion_features=True)
    assert out.shape[1] == 8 + 2
