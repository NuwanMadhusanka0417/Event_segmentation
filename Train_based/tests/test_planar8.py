"""8-parameter planar ego model (Hua et al. 2025, Eq. 2-6) in ego_residual_joint."""

import torch

from hdems.models.motion import ego_residual_joint

H, W = 48, 64


def _plane_flow(t=(0.02, -0.01, 0.05), w=(0.01, -0.02, 0.015), plane=(0.3, -0.2, 1.0),
                f=60.0, cx=29.0, cy=27.0):
    """Pixel flow of a plane d/Z = a x + b y + c under camera translation t and rotation
    w (Eq. 2): u = (1/Z) A(x) t + B(x) w in normalised coordinates, times f. The
    principal point is off-centre on purpose (the model must not need it)."""
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    x, y = (xx - cx) / f, (yy - cy) / f
    inv_z = plane[0] * x + plane[1] * y + plane[2]           # 1/Z with d = 1
    tx, ty, tz = t
    wx, wy, wz = w
    ux = inv_z * (-tx + x * tz) + (x * y * wx - (1 + x * x) * wy + y * wz)
    uy = inv_z * (-ty + y * tz) + ((1 + y * y) * wx - x * y * wy - x * wz)
    return (f * torch.stack([ux, uy])).float()[None]          # (1, 2, H, W) px


def test_planar_flow_is_fitted_exactly():
    flow = _plane_flow()
    assert flow.abs().max() > 1.0                            # a real motion, not ~0
    for model in ("planar8", "quadratic"):                   # quadratic contains planar8
        res, bg = ego_residual_joint(flow, model=model)
        assert res.abs().max() < 1e-3, model
        assert torch.allclose(bg, flow, atol=1e-3)


def test_planar8_cannot_absorb_a_non_physical_pattern():
    # u_x = y^2 is quadratic but no camera motion over a plane makes it
    yy = torch.linspace(-1.0, 1.0, H).view(H, 1).expand(H, W)
    flow = torch.zeros(1, 2, H, W)
    flow[0, 0] = 3.0 * yy * yy
    res_q, _ = ego_residual_joint(flow, model="quadratic")
    res_p, _ = ego_residual_joint(flow, model="planar8")
    assert res_q.abs().max() < 1e-3
    assert res_p.abs().max() > 0.5


def test_planar8_keeps_a_mover_out_with_the_prior():
    flow = _plane_flow(t=(0.0, 0.0, 0.01), w=(0.0, 0.0, 0.0))     # small leftover motion
    mov = torch.zeros(H, W, dtype=torch.bool)
    mov[10:30, 10:34] = True                                        # 16% of the image
    flow[0, 0][mov] += 4.0
    res, _ = ego_residual_joint(flow, model="planar8", prior_px=0.5)
    mag = res[0].norm(dim=0)
    assert float(mag[~mov].max()) < 0.05
    assert float(mag[mov].min()) > 3.5
