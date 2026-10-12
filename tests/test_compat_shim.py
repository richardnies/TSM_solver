"""The disp_rel drop-in replacement keeps the old API."""
import numpy as np
import pytest
import disp_rel as disp

FAC = 0.7733975858850816
Ftheta_P = lambda th: np.exp(-(th / np.pi)**2) / FAC
T_D = lambda th: np.sin(th) / 2.7778


@pytest.mark.fast
def test_anchor_through_shim_with_old_keywords():
    D = disp.evaluate_disp_quasineutrality(omega=9.618900159238024106e-01 + 1j * 4.088425624787327406e-01, kr=0.5,
                                           Phi_P=4 * 0.3557, T_P=Ftheta_P, T_D=T_D, alpha_par=3, alpha_prp=3, tau=1,
                                           epsrel_theta=1e-8, epsabs_theta=1e-8, epsrel_y=1e-8, epsabs_y=1e-8,
                                           disp_model=0.1, Nintegrate=100, a_int=30, tol_adaptive=1e-4)
    assert abs(D - (0.08295608398656862 - 0.11593358582463427j)) < 1e-8


@pytest.mark.fast
def test_old_envelope_keyword_T_P():
    """The old drivers pass the primary envelope as T_P=: same results as Ftheta_P=."""
    w, kr, Phi, kw = 0.6 + 0.3j, 0.5, 0.7114, dict(alpha_par=3, alpha_prp=3, disp_model=-0.1)
    assert disp.evaluate_disp_quasineutrality(omega=w, kr=kr, Phi_P=Phi, T_P=Ftheta_P, T_D=T_D, **kw) == \
        disp.evaluate_disp_quasineutrality(omega=w, kr=kr, Phi_P=Phi, Ftheta_P=Ftheta_P, T_D=T_D, **kw)
    for f in (disp.evaluate_integrand_ITG, disp.evaluate_integrand_QN):
        assert f(theta_val=0.4, y_val=0.2, omega=w, kr=kr, Phi_P=Phi, T_P=Ftheta_P, T_D=T_D, **kw) == \
            f(0.4, 0.2, w, kr, Phi, Ftheta_P, T_D, **kw)
    with pytest.raises(TypeError):
        disp.evaluate_disp_quasineutrality(omega=w, kr=kr, Phi_P=Phi, T_D=T_D)


@pytest.mark.fast
def test_omegaN_alias_and_scalar_callables():
    a = disp.evaluate_disp_quasineutrality(omegaN=0.5 + 0.3j, kr=0.3, Phi_P=0.7, Ftheta_P=lambda th: 1, T_D=T_D,
                                           alpha_par=0, alpha_prp=3, disp_model=-0.1)
    b = disp.evaluate_disp_quasineutrality(omega=0.5 + 0.3j, kr=0.3, Phi_P=0.7, Ftheta_P=lambda th: 1.0, T_D=T_D,
                                           alpha_par=0, alpha_prp=3, disp_model=-0.1)
    assert a == b


@pytest.mark.fast
def test_theta_val_and_local_functions():
    w, kr, Phi = 0.6 + 0.3j, 0.5, 0.7114
    Dth = disp.evaluate_disp_quasineutrality(omega=w, kr=kr, Phi_P=Phi, Ftheta_P=Ftheta_P, T_D=T_D, alpha_par=3, alpha_prp=3,
                                             theta_val=0.4, disp_model=-0.1)
    # −w(θ) ⟨(D−1)/D⟩_y /(2π) with w = 1 + ε cos θ, ⟨⟩_y = (1/2π)∫dy
    from scipy.integrate import quad
    f = lambda y: (lambda D: (D - 1) / D)(disp.evaluate_integrand_ITG(0.4, y, w, kr, Phi, Ftheta_P, T_D, 3, 3, disp_model=-0.1))
    re = quad(lambda y: f(y).real, -np.pi, np.pi, epsabs=1e-11)[0]
    im = quad(lambda y: f(y).imag, -np.pi, np.pi, epsabs=1e-11)[0]
    ref = -(re + 1j * im) / (2 * np.pi) * (1 + 0.5 / 2.7778 * np.cos(0.4)) / (2 * np.pi)
    assert abs(Dth - ref) < 1e-9
    q = disp.evaluate_integrand_QN(0.4, 0.2, w, kr, Phi, Ftheta_P, T_D, 3, 3, disp_model=-0.1)
    assert abs(q - (1 - 1 / disp.evaluate_integrand_ITG(0.4, 0.2, w, kr, Phi, Ftheta_P, T_D, 3, 3, disp_model=-0.1))) < 1e-14
    assert disp.get_Cm(1, 3, 3) == pytest.approx(-1.3636363636363638) and disp.get_alphaS(1, 3) == 2.5
    with pytest.raises(NotImplementedError):
        disp.evaluate_disp_quasineutrality(omega=w, kr=kr, Phi_P=Phi, Ftheta_P=Ftheta_P, T_D=T_D, theta_val=0.4 + 0.1j)
