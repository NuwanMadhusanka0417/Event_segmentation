"""Motion decoding for segmentation: read velocity out of the bundled field and
ego-compensate it.

This restores the displacement axis that a raw Phi -> classifier path drops:

  Phase 1 (decode_flow):   soft-argmax over the VSA cost volume  <F(x), F(x+v)>
                           -> per-pixel velocity (the "read the stamp out of the
                           mailbox" step; reuses field.cost_volume_from_field).
  Phase 3 (ego_residual):  robust global affine fit of that flow (IRLS) ->
                           residual velocity ~= 0 on background, large on
                           independently moving objects. Residual velocity is the
                           actual segmentation cue.

Both are zero-parameter. decode_flow is differentiable (not that it needs to be --
the encoder/matcher upstream are frozen); ego_residual runs under no_grad and is
consumed as a fixed input feature by the head.
"""

from __future__ import annotations

import torch

from hdems.vsa.field import cost_volume_from_field


def decode_flow(
    F: torch.Tensor,
    Phi: torch.Tensor,
    phx: torch.Tensor,
    phy: torch.Tensor,
    M: int = 7,
    beta: float = 1.0,
) -> torch.Tensor:
    """Soft-argmax velocity from the VSA cost volume.

    Parameters
    ----------
    F, Phi : (B, d, H, W) complex   descriptor field and its bundled field
    phx, phy : (d,)                 displacement codebook phases (from the matcher)
    M : matching window (odd)       displacement range is [-M//2, M//2]
    beta : soft-argmax temperature  (higher -> sharper / closer to hard argmax)

    Returns
    -------
    flow : (B, 2, H, W) float32     expected displacement per pixel
    """
    C = cost_volume_from_field(F, Phi, phx, phy, M)          # (B, M, M, H, W)
    B, _, _, H, W = C.shape
    m = M // 2
    prob = torch.softmax(C.reshape(B, M * M, H, W) * beta, dim=1)

    offs = torch.arange(-m, m + 1, device=C.device, dtype=torch.float32)
    g0 = offs.view(M, 1).expand(M, M).reshape(M * M)         # first cost-volume axis
    g1 = offs.view(1, M).expand(M, M).reshape(M * M)         # second cost-volume axis
    u = (prob * g0.view(1, M * M, 1, 1)).sum(1)
    v = (prob * g1.view(1, M * M, 1, 1)).sum(1)
    return torch.stack([u, v], dim=1)


EGO_MODELS = ("affine", "quadratic", "planar8")


def _basis_planar8(H: int, W: int, dev: torch.device) -> torch.Tensor:
    """Design matrix (2HW, 8) of the 8-parameter planar flow (Hua et al. 2025, Eq. 6):

        u_x = a1 x² + a2 xy + a3 x + a4 y + a5
        u_y = a1 xy + a2 y² + a6 y + a7 x + a8

    -- the flow of a plane under any camera motion, rotation included. It is the
    quadratic model with 4 constraints (no y² in u_x, no x² in u_y, the curvature
    terms shared by both axes), so it cannot bend toward non-physical patterns.
    The ties only hold with ONE scale for x and y (square pixels): both are divided
    by max(H, W) / 2, not normalised to [-1, 1] separately. The origin may sit
    anywhere (the constraints survive a shift), so no intrinsics are needed.
    Rows 0..HW-1 are u_x, rows HW..2HW-1 are u_y.
    """
    s = max(H, W) / 2.0
    ys = ((torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2.0) / s).view(H, 1).expand(H, W).reshape(-1)
    xs = ((torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2.0) / s).view(1, W).expand(H, W).reshape(-1)
    one, zero = torch.ones_like(xs), torch.zeros_like(xs)
    ax = torch.stack([xs * xs, xs * ys, xs, ys, one, zero, zero, zero], dim=1)
    ay = torch.stack([xs * ys, ys * ys, zero, zero, zero, ys, xs, one], dim=1)
    return torch.cat([ax, ay], dim=0)


def _basis(H: int, W: int, model: str, dev: torch.device) -> torch.Tensor:
    """Design matrix (HW, 3 | 6) over normalised pixel coordinates."""
    ys = torch.linspace(-1.0, 1.0, H, device=dev).view(H, 1).expand(H, W).reshape(-1)
    xs = torch.linspace(-1.0, 1.0, W, device=dev).view(1, W).expand(H, W).reshape(-1)
    cols = [xs, ys, torch.ones_like(xs)]
    if model == "quadratic":
        cols += [xs * xs, xs * ys, ys * ys]
    elif model != "affine":
        raise ValueError(f"ego model must be one of {EGO_MODELS}, got {model!r}")
    return torch.stack(cols, dim=1)


@torch.no_grad()
def ego_residual_joint(
    flow: torch.Tensor,
    iters: int = 4,
    eps: float = 1e-4,
    *,
    valid: torch.Tensor | None = None,
    model: str = "quadratic",
    prior_px: float | None = None,
    floor_px: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust background model with ONE weight per pixel for both flow axes.

    model     affine (3 coefficients per axis) | quadratic (6 per axis) | planar8
              (8 shared by both axes, ``_basis_planar8``). Camera rotation and a
              translating camera over a plane give flow that is quadratic in the image
              coordinates -- exactly the 8-parameter planar flow, which the quadratic
              model contains; the affine model cannot follow either.
    prior_px  where the fit STARTS. None: plain least squares, so the first pass is
              pulled by everything -- a mover covering half the events captures it.
              A value (working px per interval): the first weights come from the flow
              itself (Tukey, 0 beyond 4 x prior), i.e. the fit starts at "nothing
              moves". After IMU derotation the static scene has little flow left, so
              the fit starts on the background and a large mover cannot take it over.
              Falls back to the plain start when < 20% of the pixels would be used.
              Later passes use Tukey weights (exactly 0 beyond c = 4 x scale): Cauchy
              weights never reach 0, and the quadratic model is flexible enough to bend
              towards a large mover that keeps even a small weight. The scale is the
              median residual of the CURRENT inliers (weight >= 0.5), not of all pixels,
              for the same reason.

    Returns (residual (B, 2, H, W), background model field (B, 2, H, W)).
    """
    B, C, H, W = flow.shape
    dev = flow.device
    joint = model == "planar8"              # one coefficient vector for BOTH axes
    A = _basis_planar8(H, W, dev) if joint else _basis(H, W, model, dev)   # (2HW | HW, P)
    eye = torch.eye(A.shape[1], device=dev)

    def fit(w: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Weighted least squares -> the model's flow (2, HW)."""
        if joint:
            w2 = torch.cat([w, w])
            Aw = A * w2[:, None]
            coef = torch.linalg.solve(A.t() @ Aw + eps * eye, Aw.t() @ f.reshape(-1))   # (8,)
            return (A @ coef).view(2, -1)
        Aw = A * w[:, None]
        coef = torch.linalg.solve(A.t() @ Aw + eps * eye, Aw.t() @ f.t())               # (P, 2)
        return (A @ coef).t()

    residual = torch.empty_like(flow)
    for b in range(B):
        f = flow[b].reshape(C, -1)                                 # (2, HW)
        sel = (torch.ones(H * W, dtype=torch.bool, device=dev) if valid is None
               else valid[b].reshape(-1).to(dev))
        if int(sel.sum()) < 4 * A.shape[1]:                        # too few events to fit
            residual[b] = flow[b]
            continue
        sel_f = sel.to(flow.dtype)
        w = sel_f.clone()
        if prior_px:
            # Tukey weights around "no motion": a pixel moving > 4 x prior gets weight 0
            # from the first pass on. If that leaves too little (a fast-translating camera
            # moves the whole background), start from plain least squares instead.
            w0 = sel_f * (1.0 - (f.norm(dim=0) / (4.0 * float(prior_px))) ** 2).clamp_min(0.0) ** 2
            if float(w0.sum()) >= 0.2 * float(sel_f.sum()):
                w = w0
        r = f
        for _ in range(iters):
            r = f - fit(w, f)
            rn = r.norm(dim=0)
            inl = sel & (w >= 0.5 * w[sel].max())
            s = (rn[inl] if inl.any() else rn[sel]).median().clamp_min(floor_px)
            c = 4.0 * s
            w_new = sel_f * (1.0 - (rn / c) ** 2).clamp_min(0.0) ** 2          # Tukey biweight
            if float(w_new.sum()) < 4 * A.shape[1]:                            # degenerate: keep
                break
            w = w_new
        residual[b] = r.reshape(C, H, W)
    return residual, flow - residual


@torch.no_grad()
def ego_residual(
    flow: torch.Tensor,
    iters: int = 3,
    eps: float = 1e-4,
    *,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract a robust global affine motion model -> residual (IMO) velocity.

    Fits each flow channel as an affine function of normalized pixel coordinates
    with a few IRLS (Cauchy-weighted) passes so moving-object pixels don't drag
    the background model. Returns residual flow and its magnitude.
    (The original model; ``ego_residual_joint`` is the quadratic / IMU-anchored one.)

    Parameters
    ----------
    flow  : (B, 2, H, W)
    valid : (B, H, W) bool, optional — pixels the model is fitted ON, normally the
        EVENT pixels. Pixels without events carry no flow evidence (the cost volume
        is flat there, so the soft-argmax returns ~0). They are 70-90% of a frame,
        so including them drags the global fit toward zero and the "residual"
        collapses to the raw flow — i.e. no ego compensation happens at all.
        The residual is still returned for every pixel.

    Returns
    -------
    residual : (B, 2, H, W)   flow minus the fitted global model
    magnitude: (B, 1, H, W)   ||residual||
    """
    B, C, H, W = flow.shape
    dev = flow.device
    ys = torch.linspace(-1.0, 1.0, H, device=dev).view(H, 1).expand(H, W).reshape(-1)
    xs = torch.linspace(-1.0, 1.0, W, device=dev).view(1, W).expand(H, W).reshape(-1)
    A = torch.stack([xs, ys, torch.ones_like(xs)], dim=1)    # (HW, 3)
    eye = torch.eye(3, device=dev)

    residual = torch.empty_like(flow)
    for b in range(B):
        if valid is None:
            sel = torch.ones(H * W, dtype=torch.bool, device=dev)
        else:
            sel = valid[b].reshape(-1).to(dev)
            if int(sel.sum()) < 16:          # too few events to fit: keep raw flow
                residual[b] = flow[b]
                continue
        sel_f = sel.to(flow.dtype)
        for c in range(C):
            f = flow[b, c].reshape(-1)
            w = sel_f.clone()
            coef = torch.zeros(3, device=dev)
            for _ in range(iters):
                Aw = A * w[:, None]
                coef = torch.linalg.solve(A.t() @ Aw + eps * eye, Aw.t() @ f)
                r = f - A @ coef
                s = r[sel].abs().median() + 1e-6
                w = sel_f / (1.0 + (r / (2.0 * s)) ** 2)     # Cauchy weights, event px only
            residual[b, c] = (f - A @ coef).reshape(H, W)

    magnitude = residual.pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
    return residual, magnitude
