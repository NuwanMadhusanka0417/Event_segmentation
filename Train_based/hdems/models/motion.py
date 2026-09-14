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


@torch.no_grad()
def ego_residual(
    flow: torch.Tensor,
    iters: int = 3,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract a robust global affine motion model -> residual (IMO) velocity.

    Fits each flow channel as an affine function of normalized pixel coordinates
    with a few IRLS (Cauchy-weighted) passes so moving-object pixels don't drag
    the background model. Returns residual flow and its magnitude.

    Parameters
    ----------
    flow : (B, 2, H, W)

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
        for c in range(C):
            f = flow[b, c].reshape(-1)
            w = torch.ones_like(f)
            coef = torch.zeros(3, device=dev)
            for _ in range(iters):
                Aw = A * w[:, None]
                coef = torch.linalg.solve(A.t() @ Aw + eps * eye, Aw.t() @ f)
                r = f - A @ coef
                s = r.abs().median() + 1e-6
                w = 1.0 / (1.0 + (r / (2.0 * s)) ** 2)       # Cauchy robust weights
            residual[b, c] = (f - A @ coef).reshape(H, W)

    magnitude = residual.pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
    return residual, magnitude
