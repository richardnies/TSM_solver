"""The frozen primary streamer: v_Ex(z, y) = −Φ_P T_P(z) sin y  (ρ_ref v_th/a; the sign is the
convention of the original code, irrelevant after the y average).

Amplitude conventions are documented in README.md §Amplitude; `stella_phiP` converts a stella
`secondary = .true.` deck to Φ_P and `A_P` evaluates the paper's eq. (21).
"""
import numpy as np

#: Factor used by the historical drivers: Φ_P = 0.3557 × (plot label "A^P"); see README §Amplitude.
PHI_P_PER_LABEL_LEGACY = 0.3557


def legacy_gaussian_norm(width=np.pi, center=0.0):
    """fac_norm of the old drivers: RMS over a uniform θ grid of exp(−((θ−θ0)/width)²) (trapz, 1000 pts)."""
    th = np.linspace(-np.pi, np.pi, 1000)
    f = np.exp(-((th - center) / width)**2)
    return np.sqrt(np.trapz(f**2) / np.trapz(np.ones_like(f)))


class Primary:
    """Primary envelope T_P(z) (vectorised callable, real) and amplitude Φ_P."""

    def __init__(self, T_P, Phi_P, name="callable", breakpoints=()):
        """T_P: vectorised callable of the field-line angle (any analytic θ-variation); breakpoints: angles
        where T_P is discontinuous or sharp, used as panel edges of the θ quadrature."""
        self.T_P = T_P
        self.Phi_P = float(Phi_P)
        self.name = name
        self.breakpoints = [float(p) for p in breakpoints]

    def __call__(self, z):
        v = self.T_P(z)
        return np.broadcast_to(v, np.shape(z)).astype(float) if np.ndim(z) else float(v)

    def amplitude(self, z):
        """Φ_P T_P(z): local amplitude of v_Ex."""
        return self.Phi_P * self(z)

    @classmethod
    def constant(cls, Phi_P):
        return cls(lambda z: 1.0, Phi_P, name="constant")

    @classmethod
    def top_hat(cls, theta_P, Delta_P, Phi_P):
        """T_P = 1 for θ ∈ [θ^P − Δ^P/2, θ^P + Δ^P/2] (taken modulo 2π), 0 otherwise; 0 < Δ^P ≤ 2π.
        The two edges are breakpoints of the θ quadrature."""
        if not 0 < Delta_P <= 2 * np.pi:
            raise ValueError("Delta_P must be in (0, 2π]")
        th0, half = float(theta_P), 0.5 * float(Delta_P)

        def T(z):
            d = np.mod(np.asarray(z, float) - th0 + np.pi, 2 * np.pi) - np.pi      # θ − θ^P wrapped to [−π, π)
            return (np.abs(d) <= half).astype(float) if Delta_P < 2 * np.pi else np.ones(np.shape(z))
        edges = [] if Delta_P >= 2 * np.pi else [np.mod(th0 + s * half + np.pi, 2 * np.pi) - np.pi for s in (-1, 1)]
        self = cls(T, Phi_P, name="top hat θP=%.3g ΔP=%.3g" % (th0, Delta_P), breakpoints=edges)
        self.theta_P, self.Delta_P = th0, float(Delta_P)
        return self

    @classmethod
    def gaussian(cls, Phi_P, width=np.pi, center=0.0, normalise="legacy"):
        """T_P = exp(−((z−center)/width)²)/norm with norm = 'legacy' (old drivers' fac_norm),
        'rms' (exact RMS over uniform z on [−π, π]) or 'peak' (1)."""
        if normalise == "legacy":
            norm = legacy_gaussian_norm(width, center)
        elif normalise == "rms":
            from scipy.integrate import quad
            norm = np.sqrt(quad(lambda t: np.exp(-2 * ((t - center) / width)**2), -np.pi, np.pi,
                                epsabs=1e-14, epsrel=1e-14)[0] / (2 * np.pi))
        elif normalise == "peak":
            norm = 1.0
        else:
            raise ValueError(normalise)
        self = cls(lambda z: np.exp(-((z - center) / width)**2) / norm, Phi_P,
                   name="gaussian w=%.3g c=%.3g" % (width, center))
        self.norm = norm
        self.width, self.center = width, center
        return self

    @classmethod
    def tabulated(cls, z, TP, Phi_P, periodic=True):
        from .geometry.base import _Fourier
        from scipy.interpolate import CubicSpline
        z = np.asarray(z, float)
        TP = np.asarray(TP, float)
        uniform = np.allclose(np.diff(z), (z[-1] - z[0]) / (len(z) - 1))
        if periodic and uniform:
            f = _Fourier(z[0], z[-1] - z[0], TP[:-1])
        else:
            f = CubicSpline(z, TP, bc_type="periodic" if periodic else "not-a-knot")
        return cls(f, Phi_P, name="tabulated")


def stella_phiP(phiinit, y0, exb_nonlin_fac, fac_norm=1.0, den0=1.0, tite=1.0, b_P=0.0):
    """Φ_P (amplitude of v_Ex in ρ_ref v_th/a, solver convention) of a stella `secondary = .true.` deck.

    stella (stella_COPY/init_g.f90) initialises the primary Fourier coefficient at k_y^P = 1/y0 as
    g = phiinit · φ0(z) · [den0 + tpar0(v∥² − ½) + tperp0(v⊥² − 1)] F_M with φ0 = exp(−(z/width0)²) and
    freezes it; quasineutrality with modified-adiabatic electrons gives the potential coefficient
    φ̂ = phiinit · φ0 · den0 Γ0(b_P)^{1/2}/(1 − Γ0(b_P) + 1/tite) (→ phiinit φ0 den0 for b_P → 0, tite = 1).
    The real-space potential is 2|φ̂| cos(k_y y), and stella's radial E×B velocity is
    exb_nonlin_fac · ∂_y φ (exb_nonlin_fac contains the ½ of v_th² = 2T/m). Hence

        Φ_P T_P(z) = 2 exb_nonlin_fac k_y^P |φ̂(z)|,   T_P = φ0/fac_norm.

    The historical drivers set Φ_P = k_y^P · phiinit · fac_norm, i.e. without 2·exb_nonlin_fac
    (= 1/√(1−ε²) = 1.0166 for the CBC surface at ε = 0.18).
    η∥ = tpar0/den0 and η⊥ = tperp0/den0.
    """
    from scipy.special import ive
    g0 = ive(0, b_P)
    qn = den0 * np.sqrt(g0) / (1 - g0 + 1.0 / tite) * (1.0 / tite)    # = den0 for b_P = 0
    return 2.0 * exb_nonlin_fac * (1.0 / y0) * phiinit * fac_norm * qn


def A_P(primary, geometry, rmaj=None):
    """Paper eq. (21): A^P = √⟨(v_Ex R/(v_T ρ))²⟩_ψ, with ⟨·⟩_ψ over the field line and the primary
    phase y (⟨sin² y⟩ = ½), R = rmaj (units of a) and ρ = ρ_ref (B = B_ref):

        A^P = (R/a) Φ_P √(⟨T_P²⟩/2).

    Note: the plot labels of the historical drivers are A_label = Φ_P/0.3557 with Φ_P the uncorrected
    stella amplitude; for the CBC Gaussian primary A_label ≈ 1.37 × A^P(21) (README §Amplitude).
    """
    R = rmaj if rmaj is not None else geometry.rmaj
    return R * primary.Phi_P * np.sqrt(geometry.average(lambda z: primary(z)**2) / 2.0)
