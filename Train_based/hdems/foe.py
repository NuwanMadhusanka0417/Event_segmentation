"""Depth-free test of "is this pixel static?" from the focus of expansion (FOE).

Why
---
The ego fit models the background flow with a low-order polynomial. Rotation is
depth-free and the IMU removes it, but TRANSLATION makes static points move by an
amount that depends on their depth (parallax): near objects on the table move more
than the wall behind them, and no quadratic captures that. The residual then flags
static near objects as "moving" (and fits movers away when they dominate).

Geometry
--------
Once the rotation is removed, a static point at image position x moves along the
line through the FOE e, the image of the translation direction T:

    d(x) = -(Δ T_z / Z(x)) (x - e)          (displacement toward EARLIER times)

Depth Z(x) only scales the displacement ALONG that line. So without any depth:

    r_perp(x) = |d × (x - e)| / |x - e|      the component across the FOE line: 0 if static
    sign      d · (x - e) has ONE sign for every static point (Z > 0)

A mover breaks the first unless it happens to move exactly along its FOE line, and
the second if it moves against the expansion. Both hold in pixel coordinates too
(pixels are an affine map of the normalised coordinates, which keeps lines and
their intersection), so no intrinsics are needed.

Sideways motion (T_z ≈ 0) puts e at infinity. In homogeneous form e = (e1, e2, e3)
and the direction from x toward the FOE is v = e3·x − (e1, e2), which stays defined
(for e3 = 0 every static flow is parallel to (e1, e2)).

Estimation: each flow vector is a line l = x̃ × (d, 0) (normalised: l·ẽ is the
distance of e from the line in px). RANSAC on pairs of lines (e = l_i × l_j), scored
by the number of flow vectors whose perpendicular residual is below ``inlier_px``
AND that agree on the sign, then a least-squares refit (SVD) on the inliers.

Small translation: if few pixels move at all after derotation, the FOE is not
defined (and a mover would capture it). Then a static point simply does not move:
the residual is |d| itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

CUES = ("perp", "foe", "derotated")


@dataclass(frozen=True)
class FoeParams:
    """Thresholds in SENSOR px per flow interval (estimate_foe takes the ratio)."""
    min_flow_px: float = 1.0        # flow vectors used for the FOE
    min_moving_frac: float = 0.3    # fewer valid pixels moving than this -> "no translation"
    inlier_px: float = 1.0          # perpendicular residual of an inlier
    hypotheses: int = 256           # RANSAC pairs
    max_points: int = 4000          # flow vectors scored per hypothesis (random subset)
    seed: int = 0


@dataclass
class Foe:
    """Homogeneous FOE (pixel coordinates of the working grid) and the sign a static
    point's d · v takes; ``None`` fields = translation too small to define one."""
    e: torch.Tensor | None          # (3,)
    sign: float
    inliers: float                  # fraction of the moving vectors that are inliers
    moving_frac: float              # fraction of the valid pixels that move at all


def _grid(H: int, W: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype), indexing="ij")
    return xx, yy


def _components(d: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(perpendicular, signed parallel) components of d w.r.t. the FOE direction.
    d: (2, ...) and x / y: (...) for one FOE e (3,); for K FOEs e (K, 3), the points
    are (1, S) and the result (K, S)."""
    if e.dim() == 1:
        e1, e2, e3 = e[0], e[1], e[2]
    else:
        e1, e2, e3 = e[:, 0:1], e[:, 1:2], e[:, 2:3]
    vx, vy = e3 * x - e1, e3 * y - e2
    n = torch.sqrt(vx * vx + vy * vy).clamp_min(1e-9)
    perp = (d[0] * vy - d[1] * vx).abs() / n
    par = (d[0] * vx + d[1] * vy) / n
    return perp, par


def estimate_foe(d: torch.Tensor, valid: torch.Tensor, params: FoeParams = FoeParams(),
                 ratio: float = 1.0) -> Foe:
    """FOE of one derotated flow field d (2, H, W), working px (= ``ratio`` sensor px),
    from the ``valid`` pixels."""
    params = FoeParams(**{**params.__dict__, "min_flow_px": params.min_flow_px / ratio,
                          "inlier_px": params.inlier_px / ratio})
    H, W = d.shape[-2:]
    d = d.float()
    xx, yy = _grid(H, W, d.device, d.dtype)
    mag = d.norm(dim=0)
    n_valid = int(valid.sum())
    use = valid & (mag > params.min_flow_px) & torch.isfinite(mag)
    n_use = int(use.sum())
    moving_frac = n_use / max(n_valid, 1)
    if n_use < 20 or moving_frac < params.min_moving_frac:
        return Foe(None, 1.0, 0.0, moving_frac)

    px, py = xx[use], yy[use]
    dx, dy = d[0][use], d[1][use]
    m = mag[use]
    lines = torch.stack([-dy, dx, px * dy - py * dx], dim=1) / m[:, None]      # (N, 3)
    g = torch.Generator(device="cpu").manual_seed(params.seed)
    N = lines.shape[0]
    K = params.hypotheses
    i = torch.randint(N, (K,), generator=g)
    j = torch.randint(N, (K,), generator=g)
    keep = i != j
    i, j = i[keep].to(d.device), j[keep].to(d.device)
    hyp = torch.cross(lines[i], lines[j], dim=1)                                # (K, 3)
    hyp = hyp / hyp.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sub = (torch.randperm(N, generator=g)[: params.max_points]).to(d.device)
    dd = torch.stack([dx[sub], dy[sub]])[:, None, :]                            # (2, 1, S)
    perp, par = _components(dd, px[sub][None], py[sub][None], hyp)              # (K, S)
    inl = perp < params.inlier_px
    score = torch.maximum((inl & (par > 0)).sum(1), (inl & (par < 0)).sum(1))
    best = hyp[int(score.argmax())]

    # least-squares refit on the inliers of the best hypothesis (all vectors)
    for _ in range(2):
        perp, par = _components(torch.stack([dx, dy]), px, py, best)
        s = 1.0 if int(((perp < params.inlier_px) & (par > 0)).sum()) >= \
            int(((perp < params.inlier_px) & (par < 0)).sum()) else -1.0
        inl = (perp < params.inlier_px) & (par * s > 0)
        if int(inl.sum()) < 10:
            break
        _, _, vh = torch.linalg.svd(lines[inl].double(), full_matrices=False)
        cand = vh[-1].to(best)
        # SVD returns ±e: keep the orientation, the sign is re-read below
        best = cand if float((cand * best).sum()) >= 0 else -cand
    perp, par = _components(torch.stack([dx, dy]), px, py, best)
    inl = perp < params.inlier_px
    s = 1.0 if int((inl & (par > 0)).sum()) >= int((inl & (par < 0)).sum()) else -1.0
    frac = float((inl & (par * s > 0)).float().mean())
    return Foe(best, s, frac, moving_frac)


def foe_cues(d: torch.Tensor, foe: Foe) -> dict[str, torch.Tensor]:
    """Per-pixel cues (H, W), working px, for one derotated flow field d (2, H, W):

    derotated  |d|                         (static points do not move without translation)
    perp       |d × v| / |v|               depth-free, 0 for static points
    foe        sqrt(perp² + wrong²)        + the parallel part that moves AGAINST the
                                           expansion (impossible for a static point)
    With no FOE (translation too small) perp = foe = |d|.
    """
    d = d.float()
    mag = d.norm(dim=0)
    if foe.e is None:
        return {"derotated": mag, "perp": mag, "foe": mag}
    xx, yy = _grid(*d.shape[-2:], d.device, d.dtype)
    perp, par = _components(d, xx, yy, foe.e.to(d))
    wrong = torch.relu(-foe.sign * par)
    return {"derotated": mag, "perp": perp, "foe": torch.sqrt(perp * perp + wrong * wrong)}


def foe_cue_maps(flow: torch.Tensor, rot: torch.Tensor | None, valid: torch.Tensor,
                 params: FoeParams = FoeParams(), ratio: float = 1.0
                 ) -> tuple[dict[str, torch.Tensor], list[Foe]]:
    """Batch version: flow (B, 2, H, W), rot (B, 2, H, W) or None, valid (B, H, W), all
    in working px (= ``ratio`` sensor px) -> ({cue: (B, 1, H, W)}, [Foe per sample]),
    cues in working px."""
    d = flow if rot is None else flow - rot
    maps: dict[str, list[torch.Tensor]] = {c: [] for c in CUES}
    foes = []
    for b in range(d.shape[0]):
        foe = estimate_foe(d[b], valid[b], params, ratio)
        foes.append(foe)
        for c, v in foe_cues(d[b], foe).items():
            maps[c].append(v)
    return {c: torch.stack(v)[:, None] for c, v in maps.items()}, foes
