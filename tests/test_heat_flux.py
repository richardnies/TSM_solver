"""Primary heat flux in stella's normalisation (tsm_disp.primary.heat_flux_stella, Primary.with_heat_flux).

Reference values: stella's qflx of 3-step `secondary = .true.` runs (benchmarks/heat_flux, vperp_max = 4.5 so that
stella's v⊥ integral is converged); the formula reproduces them to the 5 digits stella prints.
"""
import numpy as np
import pytest
from tsm_disp.geometry import miller, AnalyticGeometry
from tsm_disp.primary import Primary, stella_phiP, legacy_gaussian_norm, heat_flux_stella

CBC = dict(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, zed_equal_arc=True)
SHAPED = dict(CBC, kappa=1.5, kapprim=0.3, tri=0.3, triprim=0.5, shift=-0.2)
PHIINIT = 571.3060437320008
# (geometry, den0_phase, tpar0, tperp0, stella qflx)
STELLA = [(CBC, 1.0, 0.0, 3.0, -5.2301e2), (CBC, -0.5, 3.0, 3.0, 4.4697e2), (SHAPED, -0.7, 3.0, 1.0, 3.9516e2)]


@pytest.mark.slow
@pytest.mark.parametrize("geo_kw, den0_phase, tpar0, tperp0, q_stella", STELLA)
def test_matches_stella(geo_kw, den0_phase, tpar0, tperp0, q_stella):
    geo = miller(**geo_kw)
    pri = Primary.gaussian(stella_phiP(PHIINIT, 1000.0, geo.exb_nonlin_fac, fac_norm=legacy_gaussian_norm(np.pi)))
    Q = heat_flux_stella(pri, geo, 1e-3, tpar0, tperp0, -den0_phase, -den0_phase)
    assert abs(Q / q_stella - 1) < 3e-4


@pytest.mark.fast
def test_formula_and_inverse():
    """Q ∝ Φ_P² ⟨Ftheta_P²⟩(½η∥ sinΔ∥ + η⊥ sinΔ⊥)/(4 f² k_y ⟨|∇ρ|⟩) and with_heat_flux inverts it."""
    g = AnalyticGeometry(vMx=np.sin, grho=lambda t: 1 + 0.3 * np.cos(t), exb_nonlin_fac=0.5, sign_torflux=-1)
    p = Primary.top_hat(0.0, np.pi, 2.0)
    Q = heat_flux_stella(p, g, 0.2, 3.0, 1.0, 0.4, -0.1)
    Ftheta_P2, gr = 0.5, 1.0                                           # ⟨Ftheta_P²⟩ and ⟨|∇ρ|⟩ for uniform weight
    assert abs(Q - (1.5 * np.sin(0.4) + np.sin(-0.1)) * 4.0 * Ftheta_P2 / (4 * 0.25 * 0.2 * gr)) < 1e-12
    p2 = Primary.top_hat(0.0, np.pi, 1.0).with_heat_flux(Q, g, 0.2, 3.0, 1.0, 0.4, -0.1)
    assert abs(p2.Phi_P - 2.0) < 1e-12 and p2(0.1) == 1.0
    assert heat_flux_stella(p, g, 0.2, 3.0, 3.0, 0.0, 0.0) == 0.0               # in-phase primary: no flux
    with pytest.raises(ValueError):
        p.with_heat_flux(Q, g, 0.2, 3.0, 1.0, -0.4, 0.1)                       # phases give Q of the other sign
    with pytest.raises(ValueError):
        heat_flux_stella(p, AnalyticGeometry(vMx=np.sin), 0.2, 3.0, 1.0, 0.4, 0.0)    # no exb_nonlin_fac / grho
