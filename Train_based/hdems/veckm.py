"""VecKM_flow (Yuan et al., ICCV 2025; github.com/dhyuan99/VecKM_flow, MIT) per-event normal
flow, and the per-frame event data shared by the flow-source experiments
(scripts/check_veckm_cue.py, scripts/eval_graphcut.py).

The repository is not vendored: pass the folder of the unzipped repository (it contains the
pretrained weights) as ``repo``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

INTERVAL_S = 0.0125          # one flow interval = the benchmark's event window


def undistort_radtan(x: np.ndarray, y: np.ndarray, mm: dict, iters: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Pixel -> undistorted NORMALISED coordinates (focal length 1), radial-tangential model
    (the fixed-point inversion cv2.undistortPoints uses)."""
    fx, fy, cx, cy = (float(mm[k]) for k in ("fx", "fy", "cx", "cy"))
    k1, k2, k3 = (float(mm.get(k, 0.0)) for k in ("k1", "k2", "k3"))
    p1, p2 = float(mm.get("p1", 0.0)), float(mm.get("p2", 0.0))
    xd, yd = (x - cx) / fx, (y - cy) / fy
    xu, yu = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = xu * xu + yu * yu
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        dx = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
        dy = p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
        xu, yu = (xd - dx) / rad, (yd - dy) / rad
    return xu, yu


def load_veckm(repo: Path, training_set: str = "UNION"):
    """Their NormalEstimator with the released weights, loaded from the file directly
    (their package reads it through importlib.resources, which needs an __init__.py
    the models folder does not have on Python < 3.10). Same weights_only load."""
    sys.path.insert(0, str(Path(repo).resolve()))
    from VecKM_flow import params as vk_params                      # noqa: E402
    from VecKM_flow.estimator import NormalEstimator                # noqa: E402
    P = getattr(vk_params, f"{training_set}Params")
    net = NormalEstimator(P.d, P.alpha)
    net.load_state_dict(torch.load(Path(repo) / "VecKM_flow" / "models" / f"{training_set}.pth",
                                   map_location="cpu", weights_only=True))
    net.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
    return SimpleNamespace(P=P, estimator=net, cuda_available=cuda, auto_scale_time=True)


def veckm_normal_flow(est, t: np.ndarray, xu: np.ndarray, yu: np.ndarray, ensemble: int = 3):
    """VecKM_flow on ONE slice of events -> (normal flow (n, 2) normalised units / s,
    uncertainty (n,)), following VecKMNormalFlowEstimator.inference for a single step."""
    P = est.P
    tt = torch.from_numpy((t - t.min()) / P.t_radius).float()
    xy = torch.from_numpy(np.stack([xu, yu], 1) / P.pxl_radius).float()
    scale = float(np.clip(len(t) / 50000.0, 0.33, 1.33)) if est.auto_scale_time else 1.0
    txy = torch.cat([tt[:, None] * scale, xy], 1)
    if est.cuda_available:
        txy = txy.cuda()
    with torch.no_grad():
        pred, unc = est.estimator.inference(txy, ensemble=ensemble)
    return (pred * scale).numpy(), unc.numpy()


def veckm_slice(stream, ts: float, slice_s: float, H: int, W: int):
    """Events of one VecKM step ending at the label time ``ts`` (time scale as in their
    code) -> (t (n,), xy (n, 2) float pixels), on-sensor events only."""
    i1 = int(np.searchsorted(stream.t, ts, side="right"))
    i0 = int(np.searchsorted(stream.t, ts - slice_s, side="left"))
    scale = float(np.clip((i1 - i0) / 50000.0, 0.33, 1.33))
    i0 = int(np.searchsorted(stream.t, ts - slice_s / scale, side="left"))
    t = np.asarray(stream.t[i0:i1], dtype=np.float64)
    xy = np.asarray(stream.xy[i0:i1]).astype(np.float64).reshape(-1, 2)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    return t[ok], xy[ok]
