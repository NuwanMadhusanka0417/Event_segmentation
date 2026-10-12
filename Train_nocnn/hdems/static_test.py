"""Training-free static-scene test: which event pixels does the camera's own motion
NOT explain?  (eval --moving-test; run_segmentation.pbs MODE=moving)

Why not cluster the raw flow (hdems.vsa_grouping)? With a moving camera the static
scene is not ONE motion: near surfaces move more than far ones and rotation adds a
position-dependent field, so the clustering split the background into 8-30 pieces and
called all but one of them objects (run vsagroup_motion_r2_20261012: FG IoU 0.05 on the
validation scene). Here the camera's motion is removed with geometry instead:

  rotation     measured by the IMU gyroscope (hdems.imu, sample["rot_flow"]). Its image
               motion depends only on the pixel, not on depth, so it is SUBTRACTED.
               Below ``min_rot`` (median over the image) it is left in: the front end
               rounds sub-pixel motion to zero (Eq.12 alpha), so subtracting a real
               0.3 px rotation from a flow that reads 0 would CREATE motion.
  translation  a gyro cannot measure it and its image motion depends on depth. What is
               left after derotation is tested against a background model:

    epipolar   depth-independent. For a translating camera a static point moves along
               the line through the epipole e (the focus of expansion), with ONE sign for
               the whole scene: d ∥ D(q) = t_z q - t_xy (normalized coordinates). t is
               estimated by RANSAC (2 pixels fix it) on the event pixels.
               score = distance from that line + the part pointing the WRONG way (px).
               Blind spots: a mover sliding exactly along its epipolar line in the
               scene's direction, and a mover that keeps still in the image.
    quadratic  robust quadratic fit (a plane seen by a translating camera has exactly
               this flow): RANSAC start + Tukey IRLS, so a large mover cannot capture it
    affine     the old global affine fit (hdems.models.motion.ego_residual)
    none       |derotated flow| -- only right for a camera that does not translate

  When the derotated background barely moves (median |d| < ``static_px``), there is no
  translation to estimate and every model scores |derotated flow|.

Measured (ratio 2, thresholds picked on validation, the 50 eval figure frames of run
vsagroup_motion_r2_20261012): FG IoU none 0.358, quadratic 0.336, epipolar 0.312, VSA
grouping 0.289 (raw flow 0.280). The background models only pay off when the camera
turns fast (15-45 deg/s: epipolar 0.221 vs 0.173); most eval frames have a slow camera
(30-77% of event pixels read exactly 0 flow), so the default model is ``none``.

All distances in WORKING pixels per flow interval (12.5 ms); the config gives them at
full resolution (params_from_config divides by the resolution ratio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hdems.config import resolution_ratio_of

MODELS = ("epipolar", "quadratic", "affine", "none")


@dataclass(frozen=True)
class StaticTestParams:
    """Working-resolution units (see params_from_config)."""

    model: str = "none"            # none | epipolar | quadratic | affine
    threshold: float = 0.375       # score above this = moving (px per interval)
    static_px: float = 1.0         # derotated background median below this: camera not translating
    tau: float = 0.3               # inlier distance (px): epipolar line / quadratic fit
    hypotheses: int = 400          # RANSAC hypotheses (epipolar, quadratic)
    min_rot: float = 0.25          # rotational flow (median) below this is not subtracted
    seed: int = 0


def params_from_config(cfg: dict[str, Any]) -> StaticTestParams:
    """``moving_test:`` section (full-resolution px) -> working units."""
    g = cfg.get("moving_test", {}) or {}
    r = resolution_ratio_of(cfg)
    model = str(g.get("model", "none"))
    if model not in MODELS:
        raise ValueError(f"moving_test.model must be one of {MODELS}, got {model!r}")
    return StaticTestParams(
        model=model,
        threshold=float(g.get("threshold_px", 0.75)) / r,
        static_px=float(g.get("static_px", 2.0)) / r,
        tau=float(g.get("tau_px", 0.6)) / r,
        hypotheses=int(g.get("hypotheses", 400)),
        min_rot=float(g.get("min_rot_px", 0.5)) / r,
        seed=int(g.get("seed", 0)),
    )


def derotate(flow: np.ndarray, rot_flow: np.ndarray | None, min_rot: float) -> tuple[np.ndarray, bool]:
    """flow - rot_flow, unless the rotation is too small for the front end to see.
    (2, H, W) working px -> (derotated flow, subtracted?)."""
    if rot_flow is None:
        return flow, False
    if float(np.median(np.hypot(rot_flow[0], rot_flow[1]))) < min_rot:
        return flow, False
    return flow - rot_flow, True


def quadratic_residual(d: np.ndarray, ev: np.ndarray, tol: float, seed: int = 0,
                       hypotheses: int = 200, iters: int = 8) -> np.ndarray:
    """|d - robust quadratic fit| at the event pixels (H, W).

    RANSAC start (6 pixels fix both axes' quadratics; the hypothesis with most pixels
    within ``tol`` wins), then IRLS with Tukey weights. A Cauchy IRLS from a
    least-squares or 'nothing moves' start settles BETWEEN the background and a large
    mover (measured on a 21% mover: AUC 0.91); the redescending weights ignore it.
    """
    H, W = ev.shape
    yy, xx = np.nonzero(ev)
    x, y = (xx - W / 2) / W, (yy - H / 2) / H
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
    D = np.stack([d[0][yy, xx], d[1][yy, xx]], 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(A), min(len(A), 4000), replace=False)   # scores hypotheses
    best, best_n = None, -1
    for _ in range(hypotheses):
        pick = rng.choice(len(A), 6, replace=False)
        try:
            coef = np.linalg.solve(A[pick], D[pick])
        except np.linalg.LinAlgError:
            continue
        n = int((np.hypot(*(D[probe] - A[probe] @ coef).T) < tol).sum())
        if n > best_n:
            best, best_n = coef, n
    if best is None:
        best = np.linalg.lstsq(A, D, rcond=None)[0]
    res = np.hypot(*(D - A @ best).T)
    for _ in range(iters):
        s = max(1.4826 * float(np.median(res[res < 3 * tol])) if (res < 3 * tol).any() else tol, 0.02)
        c = 4.685 * s
        w = np.where(res < c, (1.0 - (res / c) ** 2) ** 2, 0.0)
        if w.sum() < 6:
            break
        Aw = A * w[:, None]
        coef = np.linalg.solve(Aw.T @ A + 1e-9 * np.eye(6), Aw.T @ D)
        res = np.hypot(*(D - A @ coef).T)
    out = np.zeros((H, W))
    out[yy, xx] = res
    return out


def affine_residual(d: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """|d - robust global affine fit| (hdems.models.motion.ego_residual) (H, W)."""
    from hdems.models.motion import ego_residual
    res, _ = ego_residual(torch.from_numpy(d).float()[None], valid=torch.from_numpy(ev)[None])
    return np.hypot(*res[0].numpy()).astype(np.float64)


def epipolar_residual(d: np.ndarray, ev: np.ndarray, intrinsics, ratio: int,
                      p: StaticTestParams) -> tuple[np.ndarray, dict[str, Any]]:
    """Distance of each derotated displacement from its epipolar line, plus the
    component pointing against the scene's direction (H, W) working px; and info."""
    fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
    H, W = ev.shape
    yy, xx = np.nonzero(ev)
    u, v = d[0][yy, xx].astype(np.float64), d[1][yy, xx].astype(np.float64)
    out = np.zeros((H, W))
    cand = np.nonzero(np.hypot(u, v) > p.static_px)[0]       # pixels with a direction
    if cand.size < 20:
        out[yy, xx] = np.hypot(u, v)
        return out, {"mode": "static"}
    qx = ((xx + 0.5) * ratio - 0.5 - cx) / fx                 # working px -> normalized
    qy = ((yy + 0.5) * ratio - 0.5 - cy) / fy
    dx, dy = u * ratio / fx, v * ratio / fy
    C = np.stack([dy, -dx, dx * qy - dy * qx], 1)             # C @ t = d x D(t)
    f = 0.5 * (fx + fy) / ratio                               # normalized -> working px

    def perp_and_norm(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Dx = qx[:, None] * T[None, :, 2] - T[None, :, 0]
        Dy = qy[:, None] * T[None, :, 2] - T[None, :, 1]
        nD = np.maximum(np.hypot(Dx, Dy), 1e-3)
        return np.abs(C @ T.T) / nD * f, nD

    rng = np.random.default_rng(p.seed)
    pick = rng.choice(cand, size=(p.hypotheses, 2))
    T = np.cross(C[pick[:, 0]], C[pick[:, 1]])
    nT = np.linalg.norm(T, axis=1)
    T = T[nT > 1e-12] / nT[nT > 1e-12, None]
    if not len(T):
        out[yy, xx] = np.hypot(u, v)
        return out, {"mode": "static"}
    counts = np.concatenate([(perp_and_norm(T[i:i + 64])[0] < p.tau).sum(0)
                             for i in range(0, len(T), 64)])
    t = T[int(np.argmax(counts))]
    for _ in range(3):                                        # refine on the inliers
        perp, nD = perp_and_norm(t[None])
        inl = perp[:, 0] < p.tau
        if inl.sum() < 3:
            break
        t = np.linalg.svd(C[inl] / nD[inl], full_matrices=False)[2][-1]
    perp, nD = perp_and_norm(t[None])
    perp, nD = perp[:, 0], nD[:, 0]
    Dx, Dy = qx * t[2] - t[0], qy * t[2] - t[1]
    par = (dx * Dx + dy * Dy) / nD * f                        # signed, along the line
    inl = perp < p.tau
    sgn = float(np.sign(np.median(par[inl]))) if inl.any() else 1.0
    out[yy, xx] = perp + np.maximum(0.0, -(sgn or 1.0) * par)
    return out, {"mode": "epipolar", "t": t.tolist(), "inlier_frac": float(inl.mean())}


def moving_score(
    flow,
    events,
    p: StaticTestParams,
    *,
    rot_flow=None,
    intrinsics=None,
    ratio: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """flow (2,H,W) working px per interval, events (H,W) bool
    -> (score (H,W) px: high = not explained by the camera's motion,
        derotated flow (2,H,W), info).

    ``rot_flow`` (2,H,W): the IMU's rotational flow (None = no IMU);
    ``intrinsics``: (fx, fy, cx, cy) in sensor px (needed by the epipolar model).
    """
    as_np = lambda a: a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    flow = as_np(flow).astype(np.float64)
    ev = as_np(events).astype(bool)
    d, used = derotate(flow, None if rot_flow is None else as_np(rot_flow).astype(np.float64),
                       p.min_rot)
    info: dict[str, Any] = {"derotated": used, "model": p.model}
    mag = np.where(ev, np.hypot(d[0], d[1]), 0.0)
    if ev.sum() < 20 or p.model == "none" or float(np.median(mag[ev])) < p.static_px:
        info["mode"] = "static"
        return mag, d, info
    if p.model == "epipolar":
        if intrinsics is None:
            raise ValueError("moving_test.model=epipolar needs the camera intrinsics")
        score, extra = epipolar_residual(d, ev, as_np(intrinsics), ratio, p)
        info.update(extra)
    elif p.model == "quadratic":
        score, info["mode"] = quadratic_residual(d, ev, p.tau, p.seed, p.hypotheses), "quadratic"
    else:
        score, info["mode"] = affine_residual(d, ev), "affine"
    return np.where(ev, score, 0.0), d, info
