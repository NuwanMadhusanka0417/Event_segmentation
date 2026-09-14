import torch

from vsa_motionseg.vsa.fpe import bind, bundle, fpe, make_base_phases, similarity, unbind
from vsa_motionseg.vsa.kernels import eigen_basis, local_similarity_comparison


def test_bind_unbind():
    d = 256
    p = make_base_phases(d, seed=0)
    hv = fpe(p, 1.5)
    role = fpe(p, 42.0)
    rec = unbind(bind(hv, role), role)
    assert similarity(hv, rec).item() > 0.99


def test_bundle_similarity():
    d = 128
    p = make_base_phases(d, seed=1)
    vs = torch.stack([fpe(p, float(i)) for i in range(5)])
    b = bundle(vs, dim=0)
    assert b.shape[-1] == d


def test_vfa_kernel_locality():
    assert local_similarity_comparison(21, 1.5, seed=0) > 0


def test_eigen_basis_repro():
    f1, e1 = eigen_basis(21, 1.5, 64)
    f2, e2 = eigen_basis(21, 1.5, 64)
    assert torch.allclose(f1, f2)
    assert e1 == e2
