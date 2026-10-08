"""Analytic limits of the dispersion relation (Nies & Parra 2026), used as tests.

Equation numbers refer to the published paper (PPCF 68 (2026) 045028).
"""
import numpy as np
from scipy.special import ive


def C_RDK(tau, eta_prp):
    """Eq. (60): C_RDK = ½[1 + τ(1 + η⊥)]  (old code: get_alphaS)."""
    return 0.5 * (1 + tau * (1 + eta_prp))


def C_TSM(tau, eta_par, eta_prp):
    """Eq. (61): C_TSM = τ (7/4 + 1/τ)^{-1} [(1 + (η∥+η⊥)/2)² − (7/4 + 2η∥ + 3η⊥/2)]  (old get_Cm = −C_TSM)."""
    return tau / (7 / 4 + 1 / tau) * ((1 + (eta_par + eta_prp) / 2)**2 - (7 / 4 + 2 * eta_par + 1.5 * eta_prp))


def omega_GAM(geometry, kx, tau=1.0):
    """Eq. (50): ω_GAM² = (7/4 + 1/τ) ⟨(k_x v_Mx)²⟩ / ⟨b⟩ (fluid limit, k_x → 0)."""
    num = geometry.average(lambda z: (kx * geometry.vMx(z))**2)
    den = geometry.average(lambda z: geometry.b(z, kx))
    return np.sqrt((7 / 4 + 1 / tau) * num / den)


def D_RDK(omega, kx, Phi_P, tau=1.0, eta_prp=0.0, b=None):
    """Eq. (B.5): exact D for a θ-independent primary of amplitude Φ_P without magnetic drift
    (uniform geometry, any b; default b = k_x²/2):

        D = 1 − τ/f_E − τ(1/f_ω − 1/f_E)/√(1 − (k_x Φ_P f_E/(ω f_ω))²),
        f_ω = 1 − Γ0 + τ,  f_E = (1 − Γ0)(1 + τ) − τ η⊥ b Γ0'   (B.3), principal square root.
    """
    if b is None:
        b = 0.5 * kx**2
    g0 = ive(0, b)
    g0p = ive(1, b) - g0
    fw = 1 - g0 + tau
    fE = (1 - g0) * (1 + tau) - tau * eta_prp * b * g0p
    A = kx * Phi_P * fE / (omega * fw)
    return 1 - tau / fE - tau * (1 / fw - 1 / fE) / np.sqrt(1 - A * A + 0j)


def omega_RDK_smallb(kx, Phi_P, tau=1.0, eta_prp=0.0):
    """Purely growing RDK root for b → 0: ω = i k_x Φ_P √C_RDK (from (51)/(52) with ω_GAM = 0)."""
    return 1j * kx * Phi_P * np.sqrt(C_RDK(tau, eta_prp))


def D_NR_single_mode(omega, kx, Phi_P, wGAM, tau=1.0, eta_par=0.0, eta_prp=0.0):
    """Eq. (52): non-resonant D for a θ-independent primary (single Fourier mode):

        0 = 1 + C_RDK (k v/ω)² − (ω_GAM/ω)² [1 − C_TSM (1 − 1/√(1 − (k v/ω)²))].
    """
    X = kx * Phi_P / omega
    return (1 + C_RDK(tau, eta_prp) * X**2
            - (wGAM / omega)**2 * (1 - C_TSM(tau, eta_par, eta_prp) * (1 - 1 / np.sqrt(1 - X * X + 0j))))
