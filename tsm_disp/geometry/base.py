"""Geometry interface and the tabulated (interpolated) implementation."""
import numpy as np
from scipy.interpolate import CubicSpline


class Geometry:
    """Abstract field-line geometry; see the package docstring for the three functions.

    Attributes: zmin, zmax, periodic, and scalars (shat, q, eps, rmaj, exb_nonlin_fac) when known.
    """
    zmin = -np.pi
    zmax = np.pi
    periodic = True
    shat = None
    q = None
    eps = None
    rmaj = None
    exb_nonlin_fac = None
    name = "geometry"

    def vMx(self, z):
        raise NotImplementedError

    def kperp2_kx2(self, z):
        raise NotImplementedError

    def weight(self, z):
        raise NotImplementedError

    def b(self, z, kx):
        """FLR parameter b = k_x² |∇x|² ρ²/2 at k_y = 0."""
        return 0.5 * kx**2 * self.kperp2_kx2(z)

    def breakpoints(self):
        """Interior points where the θ quadrature should start a new panel."""
        return []

    def average(self, f, n=4097):
        """Field-line average ⟨f⟩ = ∫ weight f dz of a vectorised callable (composite Simpson)."""
        z = np.linspace(self.zmin, self.zmax, n)
        from scipy.integrate import simpson
        return simpson(self.weight(z) * f(z), x=z)

    def __repr__(self):
        return "<%s %s z∈[%.3f, %.3f]%s>" % (type(self).__name__, self.name, self.zmin, self.zmax,
                                            " periodic" if self.periodic else "")


class _Fourier:
    """Trigonometric interpolant of a periodic function sampled on a uniform grid (endpoint excluded)."""

    def __init__(self, z0, L, f):
        f = np.asarray(f, dtype=float)
        self.z0, self.L, self.n = z0, L, len(f)
        self.c = np.fft.fft(f) / self.n
        k = np.fft.fftfreq(self.n, d=1.0 / self.n)
        if self.n % 2 == 0:        # split the Nyquist mode symmetrically so the interpolant is real
            k[self.n // 2] = self.n // 2
            self.c = np.append(self.c, self.c[self.n // 2] * 0.5)
            self.c[self.n // 2] *= 0.5
            k = np.append(k, -self.n // 2)
        self.k = k

    def __call__(self, z):
        z = np.asarray(z)
        ph = 2j * np.pi * np.multiply.outer(z - self.z0, self.k) / self.L
        out = np.exp(ph) @ self.c
        return out.real if np.isrealobj(z) else out


class TabulatedGeometry(Geometry):
    """Geometry from stella-convention arrays on a field-line grid z.

    Parameters (stella names and normalisation; lengths in a, B in B_ref):
        bmag      B/B_ref
        gradpar   b·∇z (a times)
        gds22     ŝ² |∇x|²  (stella: kperp2 = k_x² gds22/ŝ² at k_y = 0)
        gbdrift0  stella's radial drift coefficient; stella's ω_d contains
                  (k_x/ŝ)(T/2Z) gbdrift0 (v∥² + v⊥²/2) when cvdrift0 = gbdrift0, so
                  vMx = drift_sign · gbdrift0/(2 ŝ)
        shat      magnetic shear ŝ
    drift_sign: sign convention between stella's drift and the solver's ω_M (see README §Geometry).
    periodic:   True for tokamak tubes (z ∈ [−π, π], endpoint duplicated); Fourier interpolation on a
                uniform grid, periodic cubic spline otherwise. False: not-a-knot cubic spline.
    """

    def __init__(self, z, bmag, gradpar, gds22, gbdrift0, shat, *, q=None, eps=None, rmaj=None,
                 exb_nonlin_fac=None, periodic=True, drift_sign=1.0, name="tabulated", extra=None):
        z = np.asarray(z, float)
        self.z_grid = z
        self.zmin, self.zmax = float(z[0]), float(z[-1])
        self.periodic = periodic
        self.shat, self.q, self.eps, self.rmaj = shat, q, eps, rmaj
        self.exb_nonlin_fac = exb_nonlin_fac
        self.name = name
        self.extra = extra or {}
        self.arrays = dict(bmag=np.asarray(bmag, float), gradpar=np.asarray(gradpar, float),
                           gds22=np.asarray(gds22, float), gbdrift0=np.asarray(gbdrift0, float))
        vMx = drift_sign * self.arrays["gbdrift0"] / (2.0 * shat)
        kp = self.arrays["gds22"] / (shat**2 * self.arrays["bmag"]**2)
        jac = 1.0 / (self.arrays["bmag"] * self.arrays["gradpar"])
        self._vMx = self._interp(z, vMx)
        self._kp = self._interp(z, kp)
        self._jac = self._interp(z, jac)
        # normalisation of the weight with the interpolant itself (spectrally accurate if periodic)
        zz = np.linspace(self.zmin, self.zmax, 8193)
        from scipy.integrate import simpson
        self._jnorm = simpson(self._jac(zz), x=zz)

    def _interp(self, z, f):
        uniform = np.allclose(np.diff(z), (z[-1] - z[0]) / (len(z) - 1), rtol=1e-8, atol=1e-12)
        if self.periodic:
            if not np.isclose(f[0], f[-1], rtol=1e-6, atol=1e-10 * max(1.0, np.abs(f).max())):
                raise ValueError("periodic geometry but f(zmin) != f(zmax): %g vs %g" % (f[0], f[-1]))
            if uniform:
                return _Fourier(z[0], z[-1] - z[0], f[:-1])
            ff = f.copy()
            ff[-1] = ff[0]
            return CubicSpline(z, ff, bc_type="periodic")
        return CubicSpline(z, f, bc_type="not-a-knot")

    def vMx(self, z):
        return self._vMx(z)

    def kperp2_kx2(self, z):
        return self._kp(z)

    def weight(self, z):
        return self._jac(z) / self._jnorm
