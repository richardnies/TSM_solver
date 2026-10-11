"""y-quadrature rules, the exact ∂D/∂ω and Newton's method (2026-10-11 speed-ups)."""
import numpy as np
import pytest
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.roots import find_root

CBC = dict(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, zed_equal_arc=True)
# (η, ω, k_x): growing TSM; TSM next to the ISM region (poles subtracted); damped GAM-like root (deformed contour)
CASES = [(3.0, 0.57987 + 0.29557j, 0.5), (3.0, 1.00878 + 0.29236j, 0.8), (0.0, 0.6035 - 0.0078j, 0.2)]


def _d(eta, **kw):
    return DispersionRelation(miller(**CBC), Primary.gaussian(0.72314), tau=1, eta_par=eta, eta_prp=eta,
                              nthreads=8, warn=False, **kw)


@pytest.mark.slow
@pytest.mark.parametrize("eta, omega, kx", CASES)
def test_trapezoid_equals_adaptive(eta, omega, kx):
    Dt = _d(eta, epsabs=1e-10, epsrel=1e-10).D(omega, kx)
    Da = _d(eta, epsabs=1e-10, epsrel=1e-10, y_rule="adaptive", tolN=1e-12).D(omega, kx)
    assert abs(Dt - Da) < 1e-11


@pytest.mark.slow
@pytest.mark.parametrize("form", ["lambda", "Z"])
@pytest.mark.parametrize("eta, omega, kx", CASES)
def test_derivative_matches_finite_difference(eta, omega, kx, form):
    if form == "Z" and kx > 0.3:
        pytest.skip("Z form is a small-b expansion")
    d = _d(eta, epsabs=1e-11, epsrel=1e-11, form=form)
    D, dD = d.D(omega, kx, derivative=True)
    assert abs(D - d.D(omega, kx)) == 0.0
    h = 1e-5 * max(abs(omega), 0.1)
    fd = (d.D(omega + h, kx) - d.D(omega - h, kx)) / (2 * h)
    assert abs(dD - fd) < 1e-6 * abs(fd)


@pytest.mark.slow
def test_newton_equals_muller():
    d = _d(3.0, epsabs=1e-9, epsrel=1e-9)
    guess = (0.6451 + 0.3052j) * 1.01
    rn = d.find_root(guess, 0.55, tol=1e-10)
    rm = find_root(lambda w: d.D(w, 0.55), guess, tol=1e-10)
    assert rn.converged and rm.converged
    assert abs(rn.omega - rm.omega) < 1e-9 and rn.nfev < rm.nfev
