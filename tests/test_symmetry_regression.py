"""Symmetries, regression against the original code, and analyticity of D."""
import numpy as np
import pytest
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import CircularAnalytic
from tsm_disp.primary import Primary
from tsm_disp import _kernels as K

ANCHOR_OMEGA = 9.618900159238024106e-01 + 1j * 4.088425624787327406e-01
ANCHOR_D = 0.08295608398656862 - 0.11593358582463427j      # 2025-01-10_test_forbidden_region/scan_high_kr/test.py


def _cbc(Phi=4 * 0.3557, form="Z", **kw):
    geo = CircularAnalytic(eps=0.5 / 2.7778, vMx=lambda th: np.sin(th) / 2.7778)
    return DispersionRelation(geo, Primary.gaussian(Phi), eta_par=3, eta_prp=3, form=form, **kw)


@pytest.mark.fast
def test_anchor_Z_form():
    d = _cbc(epsabs=1e-12, epsrel=1e-12)
    assert abs(d.D(ANCHOR_OMEGA, 0.5) - ANCHOR_D) < 1e-11       # the stored value has the old 1e-8 tolerance


@pytest.mark.fast
def test_lambda_form_stored_value():
    """λ form at the old TSM root (A^P label 2, k_x = 0.5); value of this code (2026-10-08)."""
    d = _cbc(Phi=2 * 0.3557, form="lambda", epsabs=1e-10, epsrel=1e-10, nthreads=8)
    assert abs(d.D(0.578 + 0.290j, 0.5) - (0.006919734835895102 - 0.009835552726769045j)) < 1e-11


@pytest.mark.fast
@pytest.mark.parametrize("form", ["Z", "lambda"])
def test_reality_symmetry(form):
    """D(−ω*) = D(ω)* for an up-down symmetric problem."""
    d = _cbc(Phi=2 * 0.3557, form=form, epsabs=1e-11, epsrel=1e-11, nthreads=8)
    w = 0.61 + 0.31j            # outside the ISM region (near it the y integrals become expensive)
    assert abs(d.D(-np.conj(w), 0.5) - np.conj(d.D(w, 0.5))) < 1e-10


@pytest.mark.fast
@pytest.mark.parametrize("form", [0, 1])
def test_local_symmetries(form):
    w, A, wM, b = 0.5 + 0.2j, 0.3, 0.12, 0.2
    for y in [0.3, 1.2, -2.0]:
        D1 = K.D_ism_py(w, y, A, wM, b, 1.0, 3, 3, 0.0, 0.0, form)
        D2 = K.D_ism_py(w, -y, -A, wM, b, 1.0, 3, 3, 0.0, 0.0, form)       # (y, A) → (−y, −A): ω_E unchanged
        D3 = K.D_ism_py(w, y + np.pi, -A, wM, b, 1.0, 3, 3, 0.0, 0.0, form)
        D4 = K.D_ism_py(w, y, A, wM, b, 1.0, 3, 3, 2 * np.pi, -2 * np.pi, form)
        assert abs(D1 - D2) < 1e-13 and abs(D1 - D3) < 1e-12 and abs(D1 - D4) < 1e-12
    # without temperature perturbations the phase shifts drop out
    assert abs(K.D_ism_py(w, 0.4, A, wM, b, 1.0, 0, 0, 0.7, -0.2, form) - K.D_ism_py(w, 0.4, A, wM, b, 1.0, 0, 0, 0, 0, form)) < 1e-14


@pytest.mark.fast
def test_deformed_contour_equals_real_for_growing_modes():
    d0 = _cbc(Phi=2 * 0.3557, form="lambda", epsabs=1e-11, epsrel=1e-11, nthreads=8, y_contour="real")
    d1 = _cbc(Phi=2 * 0.3557, form="lambda", epsabs=1e-11, epsrel=1e-11, nthreads=8, y_contour="deformed", eta_fixed=0.3)
    w = 0.6 + 0.3j
    assert abs(d0.D(w, 0.5) - d1.D(w, 0.5)) < 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("form", ["Z", "lambda"])
def test_cauchy_integral_across_real_axis(form):
    """Weak drive, GAM-like frequency (outside the ISM region): D is analytic across Im ω = 0, so
    the mean over a circle equals the value at its centre."""
    geo = CircularAnalytic()
    d = DispersionRelation(geo, Primary.constant(0.15), eta_par=1, eta_prp=1, form=form, epsabs=1e-11,
                           epsrel=1e-11, nthreads=16)
    w0, r, n = 0.62 + 0.0j, 0.05, 16
    pts = w0 + r * np.exp(2j * np.pi * np.arange(n) / n)
    mean = np.mean([d.D(p, 0.3) for p in pts])
    assert abs(mean - d.D(w0 + 1e-13j, 0.3)) < 1e-8


@pytest.mark.fast
def test_pole_subtraction_is_exact():
    """Near-real zeros of D_ISM(y): subtracting their poles analytically leaves the y average unchanged."""
    from tsm_disp.geometry import miller
    gm = miller(rhoc=0.5, rmaj=2.77778, qinp=20.0, shat=0.8, zed_equal_arc=True)
    z = np.linspace(-1.6, -1.2, 21)
    out = []
    for sp in (False, True):
        d = DispersionRelation(gm, Primary.gaussian(0.36157), eta_par=3, eta_prp=3, form="lambda", epsabs=1e-9,
                               epsrel=1e-9, nthreads=8, subtract_poles=sp)
        v, f, n = d.y_average(0.0804 + 0.0296j, 0.1, z)
        out.append((v, f))
    assert np.all(out[1][1] & 64)                 # poles found at every node
    assert np.max(np.abs(out[0][0] - out[1][0])) < 1e-10
