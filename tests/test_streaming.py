"""Parallel-streaming (StreamingDispersion) and finite-k_y (BinormalDispersion) solvers, and the diagnostics.

References: the vendored solvers of the streaming study (extern/streaming_ref; circular model, ŝ = 0) and, in
real geometry, the no-streaming limit, which must reproduce DispersionRelation (λ form, eq. 32)."""
import os
import sys
import numpy as np
import pytest
from tsm_disp.geometry import AnalyticGeometry, CircularAnalytic, miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.streaming import StreamingDispersion, BinormalDispersion, VelocityGrid
from tsm_disp import diagnostics as dg

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extern", "streaming_ref")


def _ref(module):
    sys.path.insert(0, REF)
    try:
        return __import__(module)
    finally:
        sys.path.remove(REF)


def _circ(q, **kw):
    return AnalyticGeometry(vMx=np.sin, gradpar=1.0 / q, **kw)


@pytest.mark.fast
@pytest.mark.parametrize("omega", [0.9 + 0.5j, 0.5 + 0.2j])
def test_tridiagonal_path_equals_reference(omega):
    ref = _ref("tsm_core")
    p = ref.Params(k=0.3, q=6.0, AP=2.0, eta_par=3.0, eta_perp=3.0, tau=1.0, M=16, Mphi=6)
    Dr = ref.Dispersion(p, ref.Grids(n_vpar=96, n_u=12, n_y=32)).evaluate(omega)
    d = StreamingDispersion(_circ(6.0), Primary.constant(2.0), tau=1, eta_par=3, eta_prp=3, M=16, Mphi=6, n_y=32,
                            grid=VelocityGrid(96, 12), path="tridiagonal")
    assert abs(d.D(omega, 0.3) - Dr) < 1e-13


@pytest.mark.slow
def test_dense_path_equals_reference():
    ref = _ref("tsm_core")
    p = ref.Params(k=0.3, q=6.0, AP=2.0, eta_par=3.0, eta_perp=3.0, tau=1.0, M=12, Mphi=6)
    Dr = ref.Dispersion(p, ref.Grids(n_vpar=64, n_u=10, n_y=16)).evaluate(0.9 + 0.5j)
    d = StreamingDispersion(_circ(6.0), Primary.constant(2.0), tau=1, eta_par=3, eta_prp=3, M=12, Mphi=6, n_y=16,
                            grid=VelocityGrid(64, 10), path="dense", nthreads=8)
    assert abs(d.D(0.9 + 0.5j, 0.3) - Dr) < 1e-13


@pytest.mark.slow
def test_reference_roots_q20():
    """Roots stored by the streaming study (fig11_q20.npz; M = ⌈6 + 2.2 k q⌉, M_φ = 6)."""
    dat = np.load(os.path.join(REF, "fig11_q20.npz"))
    q, k, w = float(dat["q"]), dat["ks"][10], dat["omega"][10]
    d = StreamingDispersion(_circ(q), Primary.constant(float(dat["AP"])), tau=1, eta_par=3, eta_prp=3,
                            M=int(np.ceil(6 + 2.2 * k * q)), Mphi=6, n_y=40, grid=VelocityGrid(96, 12))
    r = d.find_root(w, k, tol=1e-10)
    assert r.converged and abs(r.omega - w) < 2e-5        # grids of the stored run not recorded; D itself: 1e-16


@pytest.mark.slow
def test_no_streaming_limit_equals_lambda_form():
    """Without v∥ b·∇ the Fourier solver is eq. (32): CBC-like circular geometry, Gaussian primary."""
    geo = CircularAnalytic(q=20.0)
    pri = Primary.gaussian(0.72314)
    ref = DispersionRelation(geo, pri, tau=1, eta_par=3, eta_prp=3, epsabs=1e-11, epsrel=1e-11, nthreads=8,
                             warn=False).D(0.58 + 0.29j, 0.5)
    d = StreamingDispersion(geo, pri, tau=1, eta_par=3, eta_prp=3, M=32, Mphi=16, n_y=48, grid=VelocityGrid(96, 16),
                            streaming=False, nthreads=8)
    assert abs(d.D(0.58 + 0.29j, 0.5) - ref) < 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("binormal_drift", [False, True])
def test_binormal_equals_reference(binormal_drift):
    ref = _ref("tsm_ky")
    pr = ref.ParamsKy(k=0.3, ky0=0.1, q=6.0, AP=2.0, eta_par=3.0, eta_perp=3.0, tau=1.0, M=10, N=2, Mphi=4,
                      binormal_drift=binormal_drift, profile="flat")
    Dr = ref.DispersionKy(pr, ref.VGrid(n_vpar=48, n_u=8)).evaluate(0.9 + 0.5j)
    geo = _circ(6.0, vMy=np.cos, gyy=1.0, gxy=0.0)
    d = BinormalDispersion(geo, Primary.constant(2.0), 0.1, tau=1, eta_par=3, eta_prp=3, N=2, M=10, Mphi=4,
                           grid=VelocityGrid(48, 8), binormal_drift=binormal_drift, nthreads=8)
    assert abs(d.D(0.9 + 0.5j, 0.3) - Dr) < 1e-13


@pytest.mark.slow
def test_binormal_reduces_to_y_parametric():
    """k_y^P → 0 without binormal drift: the sideband solver equals the y-parametric one (N large)."""
    geo = _circ(6.0, vMy=np.cos, gyy=1.0, gxy=0.0)
    kw = dict(tau=1, eta_par=3, eta_prp=3, Mphi=4, grid=VelocityGrid(48, 8))
    Dy = StreamingDispersion(geo, Primary.constant(2.0), M=10, n_y=48, **kw).D(0.9 + 0.5j, 0.3)
    Db = BinormalDispersion(geo, Primary.constant(2.0), 1e-6, N=12, M=10, binormal_drift=False, nthreads=8, **kw)
    assert abs(Db.D(0.9 + 0.5j, 0.3) - Dy) < 1e-7          # N = 8, 12, 16: 1.4e-6, 4.5e-9, 1.5e-11


@pytest.mark.fast
def test_binormal_requires_periodic_coefficients():
    with pytest.raises(ValueError):
        BinormalDispersion(miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=6.0, shat=0.8), Primary.constant(1.0), 0.1)
    BinormalDispersion(miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=6.0, shat=0.0), Primary.constant(1.0), 0.1)


@pytest.mark.slow
def test_diagnostics_on_lambda_form():
    gm = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, zed_equal_arc=True)
    d = DispersionRelation(gm, Primary.gaussian(0.72314), tau=1, eta_par=3, eta_prp=3, epsabs=1e-9, epsrel=1e-9,
                           nthreads=8, warn=False)
    w0 = d.find_root(0.58 + 0.29j, 0.5, tol=1e-11).omega
    res = dg.residue(d, w0, 0.5)
    f, h = (lambda w: d.D(w, 0.5)), 1e-4
    fd = (-f(w0 + 2 * h) + 8 * f(w0 + h) - 8 * f(w0 - h) + f(w0 - 2 * h)) / (12 * h)
    assert abs(res * fd - 1) < 1e-7
    # the ISM region reaches γ = 0.251 at this k_x: a box above it counts the TSM, one dipping into it has no count
    assert dg.winding_number(d, 0.5, (w0.real - 0.05, w0.real + 0.05), (w0.imag - 0.03, w0.imag + 0.05), n=12) == 1
    with pytest.raises(ValueError):
        dg.winding_number(d, 0.5, (w0.real - 0.05, w0.real + 0.05), (w0.imag - 0.08, w0.imag + 0.05), n=12)
