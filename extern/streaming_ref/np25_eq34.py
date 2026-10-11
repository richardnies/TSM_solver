"""Independent implementation of NP25 Eq. (34): the lambda-representation of
the ion response function N, with the velocity integrals done ANALYTICALLY.

    N = -2i int_0^inf dlambda  exp(i L (w - kvE)) /
          [ (1 + i L kvM)^{1/2} (2 + i L kvM) ] *
        { (Gam0(bt) + bt Gam0'(bt)) (kvM - 2 tau kvE eta_perp)/(2 + i L kvM)
          + Gam0(bt) [ tau kvE (eta_par/2 + eta_perp - 1)
                       + (kvM - tau kvE eta_par)/(2 (1 + i L kvM)) ] }

    bt = 2 b_i / (2 + i L kvM),   b_i = k^2/2

Here kvM = k_x v_Mx = k sin(theta) and kvE = k_x v_Ex^P = k vE0 sin(Y) are the
velocity-INDEPENDENT quantities of NP25 Eqs. (10) and (17).  This shares no
code with the direct velocity quadrature of Eq. (30) in tsm_core, so
comparing the two isolates any error in our velocity integrals.
"""
import numpy as np
from scipy.special import iv, roots_legendre


def Gam0_and_deriv(b):
    e = np.exp(-b)
    G0 = iv(0, b) * e
    G0p = e * (iv(1, b) - iv(0, b))
    return G0, G0p


def N_eq34(omega, k, kvM, kvE, eta_par, eta_perp, tau=1.0,
           nlam=800, lam_max=None):
    """N from Eq. (34) at one (theta, y) point."""
    wi = omega.imag
    if wi <= 0:
        raise ValueError("Eq. (34) requires Im(omega) > 0")
    if lam_max is None:
        lam_max = min(400.0, 30.0 / wi)
    x, w = roots_legendre(nlam)
    L = 0.5 * lam_max * (x + 1.0)
    wL = 0.5 * lam_max * w

    A = 1.0 + 1j * L * kvM
    B = 2.0 + 1j * L * kvM
    bt = (k ** 2) / B                      # 2 b_i / (2 + i L kvM), b_i = k^2/2
    G0, G0p = Gam0_and_deriv(bt)

    t1 = (G0 + bt * G0p) * (kvM - 2.0 * tau * kvE * eta_perp) / B
    t2 = G0 * (tau * kvE * (eta_par / 2.0 + eta_perp - 1.0)
               + (kvM - tau * kvE * eta_par) / (2.0 * A))
    integ = np.exp(1j * L * (omega - kvE)) / (np.sqrt(A) * B) * (t1 + t2)
    return -2j * np.sum(wL * integ)


def disp_eq32(omega, k, AP, eta_par, eta_perp, tau=1.0,
              n_theta=128, n_y=64, **kw):
    """NP25 Eq. (32) with N from Eq. (34)."""
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    Y = 2 * np.pi * np.arange(n_y) / n_y
    vE0 = np.sqrt(2.0) * AP
    b = 0.5 * k ** 2
    from scipy.special import ive
    const = 1.0 - ive(0, b) + tau
    tot = 0.0
    for yy in Y:
        kvE = k * vE0 * np.sin(yy)
        for tt in th:
            kvM = k * np.sin(tt)
            N = N_eq34(omega, k, kvM, kvE, eta_par, eta_perp, tau, **kw)
            tot += tau / (const - N)
    return 1.0 - tot / (n_theta * n_y)
