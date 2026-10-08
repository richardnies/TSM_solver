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

    def __init__(self, T_P, Phi_P, name="callable"):
        self.T_P = T_P
        self.Phi_P = float(Phi_P)
        self.name = name

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
