"""Non-resonant limit of the NP25 secondary-mode dispersion relation.

Implements Eqs. (52), (53)-(54) and (60)-(61) of Nies & Parra (2025) for a
single-Fourier-mode, theta-independent primary drive.  Used as an independent
analytic check on the normalisations of the full numerical dispersion
relation (drift amplitude, GAM frequency, drive amplitude, tau).

Normalisations as in tsm_core (frequencies in v_Ti/R, k = k_x rho_i).
"""

import numpy as np
from tsm_core import muller


def coefficients(eta_par, eta_perp, tau=1.0):
    C_RDK = 0.5 * (1.0 + tau * (1.0 + eta_perp))                    # Eq (60)
    pref = tau / (1.75 + 1.0 / tau)
    C_TSM = pref * ((1.0 + 0.5 * (eta_par + eta_perp)) ** 2
                    - (1.75 + 2.0 * eta_par + 1.5 * eta_perp))      # Eq (61)
    return C_RDK, C_TSM


def omega_gam(tau=1.0):
    """Eq. (50) with v_Mx = sin(theta), b_i = k^2/2 (large aspect ratio)."""
    return np.sqrt((1.75 + 1.0 / tau))


def disp_nonresonant(omega, k, AP, eta_par, eta_perp, tau=1.0):
    """Eq. (52)."""
    C_RDK, C_TSM = coefficients(eta_par, eta_perp, tau)
    wg = omega_gam(tau)
    X = k * np.sqrt(2.0) * AP / omega          # k v_Ex^{P0} / omega
    root = np.sqrt(1.0 - X ** 2 + 0j)
    if root.real < 0:
        root = -root                           # branch: Re sqrt > 0
    return (1.0 + X ** 2 * C_RDK
            - (wg / omega) ** 2 * (1.0 - C_TSM * (1.0 - 1.0 / root)))


def solve_nonresonant(k, AP, eta_par, eta_perp, tau=1.0, guess=None):
    f = lambda w: disp_nonresonant(w, k, AP, eta_par, eta_perp, tau)
    if guess is None:
        guess = 1.0 + 1.0j
    w, ok = muller(f, guess * 0.97 - 0.03j, guess * 1.03, guess)
    return w, ok


if __name__ == "__main__":
    eta = 3.0
    print("C_RDK, C_TSM =", coefficients(eta, eta))
    print("omega_GAM    =", omega_gam())
    for AP in [2, 4, 8, 16, 32, 64]:
        w, ok = solve_nonresonant(0.3, AP, eta, eta,
                                  guess=0.3 * np.sqrt(2) * AP * (0.6 + 0.6j))
        print(f"  AP={AP:5.1f}  omega={w.real:+.4f}{w.imag:+.4f}j  ok={ok}")
