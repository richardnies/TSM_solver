"""Analytic limits of the paper."""
import numpy as np
import pytest
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import CircularAnalytic, Uniform
from tsm_disp.primary import Primary
from tsm_disp import limits as L
from tsm_disp.roots import find_root


@pytest.mark.fast
@pytest.mark.parametrize("kx", [0.05, 0.5, 1.2])
@pytest.mark.parametrize("omega_fac", [0.3 + 0.4j, -0.7 + 0.1j, 0.05 + 1.5j])
def test_RDK_exact_relation_B5(kx, omega_fac):
    """No drift, θ-independent primary: D = (B.5) for any b (exact FLR and η⊥ of the λ form)."""
    d = DispersionRelation(Uniform(), Primary.constant(1.3), tau=1.5, eta_par=2, eta_prp=3, epsabs=1e-12, epsrel=1e-12)
    om = omega_fac * kx
    assert abs(d.D(om, kx) - L.D_RDK(om, kx, 1.3, 1.5, 3)) < 1e-11


@pytest.mark.fast
def test_RDK_growth_rate_small_b():
    d = DispersionRelation(Uniform(), Primary.constant(1.0), tau=1.0, eta_prp=3, epsabs=1e-12, epsrel=1e-12)
    r = find_root(lambda w: d.D(w, 0.02), 0.03j)
    assert r.converged
    assert abs(r.omega / L.omega_RDK_smallb(0.02, 1.0, 1.0, 3) - 1) < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("form", ["Z", "lambda"])
def test_GAM_frequency_converges_to_eq50(form):
    """Φ_P = 0: the root tends to ω_GAM of (50) with a k_x² correction (Landau-continued D)."""
    geo = CircularAnalytic()
    err = []
    for kx in [0.1, 0.05, 0.025]:
        d = DispersionRelation(geo, Primary.constant(0.0), form=form, epsabs=1e-12, epsrel=1e-12)
        wg = L.omega_GAM(geo, kx)
        r = find_root(lambda w: d.D(w, kx), wg - 1e-4j, tol=1e-12)
        assert r.converged
        err.append(abs(r.omega.real / wg - 1))
    assert err[2] < 1e-3
    assert 3.5 < err[0] / err[1] < 4.5 and 3.5 < err[1] / err[2] < 4.5


@pytest.mark.fast
def test_constants():
    assert abs(L.C_TSM(1.0, 3.0, 3.0) - 1.3636363636363638) < 1e-15        # = −get_Cm(1, 3, 3)
    assert L.C_RDK(1.0, 3.0) == 2.5                                         # = get_alphaS(1, 3)


@pytest.mark.slow
def test_nonresonant_single_mode_relation_52():
    """θ-independent primary, strong drive: the full root approaches the NR relation (52) as k_x → 0."""
    geo = CircularAnalytic()
    tau, ep, eq = 1.0, 0.0, 0.0
    errs = []
    for kx in [0.04, 0.02]:
        wg = L.omega_GAM(geo, kx)
        Phi = 1.2 * wg / kx              # k v_E / ω_GAM = 1.2: TSM regime
        d = DispersionRelation(geo, Primary.constant(Phi), tau=tau, eta_par=ep, eta_prp=eq, form="lambda",
                               epsabs=1e-11, epsrel=1e-11, nthreads=8)
        rn = find_root(lambda w: L.D_NR_single_mode(w, kx, Phi, wg, tau, ep, eq), wg * (0.8 + 0.6j), tol=1e-13)
        r = find_root(lambda w: d.D(w, kx), rn.omega, tol=1e-10)
        assert rn.converged and r.converged
        errs.append(abs(r.omega / rn.omega - 1))
    assert errs[1] < 0.6 * errs[0]
    assert errs[1] < 0.05
