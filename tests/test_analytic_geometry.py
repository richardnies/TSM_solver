"""Analytic θ-geometry (AnalyticGeometry). The top-hat primary case is examples/top_hat_primary/."""
import numpy as np
import pytest
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import AnalyticGeometry, CircularAnalytic
from tsm_disp.primary import Primary


@pytest.mark.fast
def test_geometry_functions():
    g = AnalyticGeometry(vMx=np.sin, gradx=lambda t: 1 + 0.5 * np.cos(t), B=lambda t: 1 + 0.2 * np.cos(t))
    t = np.linspace(-np.pi, np.pi, 7)
    assert np.allclose(g.vMx(t), np.sin(t), atol=0, rtol=1e-15)
    assert np.allclose(g.b(t, 0.3), 0.5 * 0.09 * ((1 + 0.5 * np.cos(t)) / (1 + 0.2 * np.cos(t)))**2, rtol=1e-14)
    assert abs(g.average(lambda z: np.ones_like(z)) - 1) < 1e-12               # weight normalised
    assert np.allclose(g.weight(t), (1 / (1 + 0.2 * np.cos(t))) / (2 * np.pi / np.sqrt(1 - 0.04)), rtol=1e-12)
    u = AnalyticGeometry(vMx=np.sin)
    assert np.isscalar(u.vMx(0.3)) and u.kperp2_kx2(0.3) == 1.0


@pytest.mark.fast
def test_analytic_reproduces_circular():
    """The original code's model written as AnalyticGeometry gives the same D as CircularAnalytic."""
    eps, R = 0.18, 2.7778
    ga = AnalyticGeometry(vMx=lambda t: np.sin(t) / R, gradx=lambda t: 1 + eps * np.cos(t),
                          jacobian=lambda t: 1 + eps * np.cos(t))
    prim = Primary.gaussian(0.5)
    kw = dict(tau=1, eta_par=3, eta_prp=3, epsabs=1e-10, epsrel=1e-10, nthreads=8, warn=False)
    Da = DispersionRelation(ga, prim, **kw).D(0.3 + 0.2j, 0.4)
    Dc = DispersionRelation(CircularAnalytic(eps=eps, rmaj=R), prim, **kw).D(0.3 + 0.2j, 0.4)
    assert abs(Da - Dc) < 1e-12
