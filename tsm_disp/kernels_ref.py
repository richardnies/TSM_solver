"""Slow, independent reference implementations of the ion response N (tests only).

Notation (stella units; v normalised to v_th = sqrt(2T/m)):

    N = ∫ d³v F_M J0²(√(2b) v⊥) [w1 + 2 w2 v∥² + w3 v⊥²] / (wbar − wM (v∥² + v⊥²/2)),

    wbar = ω − ω_E,      w1 = τ(−ω_E + ½η∥ω_E∥ + η⊥ω_E⊥),
    w2 = ½(ω_M − τη∥ω_E∥),   w3 = ½ω_M − τη⊥ω_E⊥,

which is Nies & Parra (2026) eq. (30) with the bi-Maxwellian primary (22)-(25); ω_E∥,⊥ are ω_E
evaluated at the phase-shifted primary phase y + Δ∥,⊥. The local kinetic response is
D_ISM = (1 − Γ0(b) + τ − N)/τ (eq. 31).

Two independent evaluations are provided:

* `N_velocity`: direct 2D quadrature over (v∥, v⊥). Valid for Im wbar ≠ 0 (no continuation).
* `N_energy`: the same integral written as ∫_0^∞ dE √E H(E)/(wbar − wM E) with
  v∥ = √E cos φ, v⊥ = √(2E) sin φ (dv∥ dv⊥ = dE dφ/√2). H is entire in E, so the Landau
  continuation to Im wbar < 0 on the resonant side (Re wbar·wM > 0) is the principal integral
  plus the residue −sgn(wM)·2πi·√Ω H(Ω)/wM at Ω = wbar/wM.
"""
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import j0, iv


def gamma0(b):
    """Γ0(b) = e^{-b} I0(b) for complex b (scipy's complex ive is NOT e^{-z}I0(z))."""
    return np.exp(-b) * iv(0, b)


def gamma0_prime(b):
    """dΓ0/db = e^{-b}(I1(b) − I0(b))."""
    return np.exp(-b) * (iv(1, b) - iv(0, b))


def frequencies(omega, omegaE, omegaE_par, omegaE_prp, omegaM, tau, eta_par, eta_prp):
    wbar = omega - omegaE
    w1 = tau * (-omegaE + 0.5 * eta_par * omegaE_par + eta_prp * omegaE_prp)
    w2 = 0.5 * (omegaM - tau * eta_par * omegaE_par)
    w3 = 0.5 * omegaM - tau * eta_prp * omegaE_prp
    return wbar, w1, w2, w3


def _cquad(f, a, b, **kw):
    re = quad(lambda x: np.real(f(x)), a, b, **kw)[0]
    im = quad(lambda x: np.imag(f(x)), a, b, **kw)[0]
    return re + 1j * im


def N_velocity(omega, omegaE, omegaE_par, omegaE_prp, omegaM, b, tau=1.0, eta_par=0.0, eta_prp=0.0,
               vmax=8.0, eps=1e-12):
    """Direct (v∥, v⊥) quadrature; requires Im(ω − ω_E) ≠ 0 (principal value, no continuation)."""
    wbar, w1, w2, w3 = frequencies(omega, omegaE, omegaE_par, omegaE_prp, omegaM, tau, eta_par, eta_prp)

    def f(vp, vpar):
        return (np.exp(-vpar**2 - vp**2) / np.pi**1.5 * 2 * np.pi * vp * j0(np.sqrt(2 * b) * vp)**2
                * (w1 + 2 * w2 * vpar**2 + w3 * vp**2) / (wbar - omegaM * (vpar**2 + 0.5 * vp**2)))

    re = dblquad(lambda vp, vpar: np.real(f(vp, vpar)), -vmax, vmax, 0, vmax, epsabs=eps, epsrel=eps)[0]
    im = dblquad(lambda vp, vpar: np.imag(f(vp, vpar)), -vmax, vmax, 0, vmax, epsabs=eps, epsrel=eps)[0]
    return re + 1j * im


def _H(E, b, w1, w2, w3):
    """H(E) such that ∫d³v F J0² num g(E) = ∫_0^∞ dE √E H(E) g(E). E may be complex."""
    sqE = np.sqrt(E + 0j)

    def g(phi):
        s = np.sin(phi)
        c = np.cos(phi)
        vp = np.sqrt(2.0) * sqE * s
        num = w1 + 2 * w2 * E * c**2 + w3 * 2 * E * s**2
        return s * np.exp(-E * (1 + s**2)) * j0c(np.sqrt(2 * b + 0j) * vp)**2 * num

    return 2 / np.sqrt(np.pi) * _cquad(g, 0, np.pi, epsabs=1e-14, epsrel=1e-13, limit=200)


def j0c(z):
    """J0 for complex argument (via the entire series in z²; adequate for |z| ≲ 30)."""
    z = np.asarray(z, dtype=complex)
    if np.all(np.imag(z) == 0):
        return j0(np.real(z)) + 0j
    from scipy.special import jv
    return jv(0, z)


def N_energy(omega, omegaE, omegaE_par, omegaE_prp, omegaM, b, tau=1.0, eta_par=0.0, eta_prp=0.0,
             Emax=60.0, eps=1e-13):
    """Energy-shell representation with the physical (Landau) continuation to Im(ω − ω_E) < 0."""
    wbar, w1, w2, w3 = frequencies(omega, omegaE, omegaE_par, omegaE_prp, omegaM, tau, eta_par, eta_prp)
    if omegaM == 0:
        return (gamma0(b) * (w1 + w2 + w3) + b * gamma0_prime(b) * w3) / wbar
    Om = wbar / omegaM
    pts = [p for p in (np.real(Om),) if 0 < p < Emax]

    def f(E):
        return np.sqrt(E) * _H(E, b, w1, w2, w3) / (Om - E)

    I = _cquad(f, 0, Emax, epsabs=eps, epsrel=eps, limit=400, points=pts or None) / omegaM
    if np.imag(wbar) < 0 and np.real(wbar) * omegaM > 0:
        I += -np.sign(omegaM) * 2j * np.pi * np.sqrt(Om) * _H(Om, b, w1, w2, w3) / omegaM
    return I


def D_ism_from_N(N, b, tau):
    return (tau + 1 - gamma0(b) - N) / tau


def _weight(C, vpar2, vprp2):
    C = np.asarray(C)
    return sum(C[n, m] * vpar2**n * vprp2**m for n in range(3) for m in range(3) if C[n, m] != 0)


def moment_velocity(wbar, omegaM, b, C, jpow=2, vmax=8.0, eps=1e-12):
    """∫d³v F_M J0^jpow Σ C[n,m] v∥^{2n} v⊥^{2m} / (ω̄ − ω_M (v∥² + v⊥²/2)), direct (v∥, v⊥) quadrature."""
    def f(vp, vpar):
        return (np.exp(-vpar**2 - vp**2) / np.pi**1.5 * 2 * np.pi * vp * j0(np.sqrt(2 * b) * vp)**jpow
                * _weight(C, vpar**2, vp**2) / (wbar - omegaM * (vpar**2 + 0.5 * vp**2)))
    re = dblquad(lambda vp, vpar: np.real(f(vp, vpar)), -vmax, vmax, 0, vmax, epsabs=eps, epsrel=eps)[0]
    im = dblquad(lambda vp, vpar: np.imag(f(vp, vpar)), -vmax, vmax, 0, vmax, epsabs=eps, epsrel=eps)[0]
    return re + 1j * im


def moment_energy(wbar, omegaM, b, C, jpow=2, Emax=60.0, eps=1e-13):
    """Energy-shell form of moment_velocity with the Landau continuation for Im ω̄ < 0 (resonant side)."""
    def H(E):
        sqE = np.sqrt(E + 0j)

        def g(phi):
            s, c = np.sin(phi), np.cos(phi)
            vp = np.sqrt(2.0) * sqE * s
            return (s * np.exp(-E * (1 + s**2)) * j0c(np.sqrt(2 * b + 0j) * vp)**jpow
                    * _weight(C, E * c**2, 2 * E * s**2))
        return 2 / np.sqrt(np.pi) * _cquad(g, 0, np.pi, epsabs=1e-14, epsrel=1e-13, limit=200)

    Om = wbar / omegaM
    pts = [p for p in (np.real(Om),) if 0 < p < Emax]
    I = _cquad(lambda E: np.sqrt(E) * H(E) / (Om - E), 0, Emax, epsabs=eps, epsrel=eps, limit=400,
               points=pts or None) / omegaM
    if np.imag(wbar) < 0 and np.real(wbar) * omegaM > 0:
        I += -np.sign(omegaM) * 2j * np.pi * np.sqrt(Om) * H(Om) / omegaM
    return I
