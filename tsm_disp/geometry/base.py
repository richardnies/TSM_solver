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
    sign_torflux = None          # stella's sign of the toroidal flux (−1 for Miller), for flux diagnostics
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

    def gradpar(self, z):
        """b·∇z in 1/a (the inverse connection length): needed only by the parallel-streaming solvers."""
        raise NotImplementedError("%s does not provide b·∇z (gradpar); the streaming solvers need it"
                                  % type(self).__name__)

    def grho(self, z):
        """|∇ρ| in 1/a (ρ = r/a for Miller, √(ψ_t/ψ_t,edge) for VMEC): stella's grho, used to normalise fluxes."""
        raise NotImplementedError("%s does not provide |∇ρ| (grho); pass grho= explicitly" % type(self).__name__)

    def breakpoints(self):
        """Interior points where the θ quadrature should start a new panel."""
        return []

    def average(self, f, n=4097, points=()):
        """Field-line average ⟨f⟩ = ∫ weight f dz of a vectorised callable: Gauss–Legendre panels (32 nodes,
        about n nodes in total) on each interval between the geometry's breakpoints and the extra `points`
        (e.g. the jumps of a primary's T_P). The nodes are interior, so f is never sampled at a jump."""
        x, w = np.polynomial.legendre.leggauss(32)
        cuts = sorted({float(p) for p in list(self.breakpoints()) + list(points) if self.zmin < p < self.zmax})
        edges = [self.zmin] + cuts + [self.zmax]
        L = self.zmax - self.zmin
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            k = max(1, int(np.ceil(n / 32 * (b - a) / L)))
            e = np.linspace(a, b, k + 1)
            c, h = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
            z = (c[:, None] + h[:, None] * x[None, :]).ravel()
            total += np.sum((self.weight(z) * f(z)).reshape(k, 32) * w[None, :] * h[:, None])
        return total

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
        out = (np.exp(ph) * self.c).sum(axis=-1)       # not `@`: a BLAS call leaves OpenBLAS threads spinning
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
    gxx, vmx:   optional ŝ-free |∇x|² (= gds22/ŝ²) and v_Mx (= gbdrift0/(2ŝ)) arrays, used instead of the
                ratios (needed at ŝ = 0).
    extra:      further arrays; the finite-k_y solver uses gds2 (|∇y|²), gxy (∇x·∇y = gds21/ŝ, else gds21/ŝ),
                gbdrift and cvdrift (binormal drift v_My = gbdrift/2 per unit v∥² + v⊥²/2).
    periodic:   True for tokamak tubes (z ∈ [−π, π], endpoint duplicated); Fourier interpolation on a
                uniform grid, periodic cubic spline otherwise. False: not-a-knot cubic spline.
    """

    def __init__(self, z, bmag, gradpar, gds22, gbdrift0, shat, *, q=None, eps=None, rmaj=None,
                 exb_nonlin_fac=None, periodic=True, drift_sign=1.0, name="tabulated", extra=None,
                 gxx=None, vmx=None):
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
        self.drift_sign = drift_sign
        vMx = drift_sign * (np.asarray(vmx, float) if vmx is not None else self.arrays["gbdrift0"] / (2.0 * shat))
        gx = np.asarray(gxx, float) if gxx is not None else self.arrays["gds22"] / shat**2
        kp = gx / self.arrays["bmag"]**2
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

    # -- finite-k_y quantities (tsm_disp.streaming.BinormalDispersion) ----------------------------------------------
    def _extra_interp(self, key, f):
        name = "_x_" + key
        if not hasattr(self, name):
            f = np.asarray(f, float)
            if self.periodic and np.isclose(f[0], f[-1], rtol=1e-6, atol=1e-10 * max(1.0, np.abs(f).max())):
                setattr(self, name, self._interp(self.z_grid, f))
            else:
                setattr(self, name, CubicSpline(self.z_grid, f, bc_type="not-a-knot"))
        return getattr(self, name)

    def _need(self, key):
        if key not in self.extra or self.extra[key] is None:
            raise NotImplementedError("%s has no '%s' (needed for finite k_y)" % (self.name, key))
        return self.extra[key]

    def vMy(self, z):
        """Binormal magnetic drift per unit (v∥² + v⊥²/2), ρ v_th/a: drift_sign · gbdrift/2 (cvdrift = gbdrift
        assumed, i.e. β' = 0; checked by the finite-k_y solver)."""
        return self._extra_interp("vmy", self.drift_sign * 0.5 * np.asarray(self._need("gbdrift"), float))(z)

    def gyy(self, z):
        """|∇y|² (stella gds2)."""
        return self._extra_interp("gyy", self._need("gds2"))(z)

    def gxy(self, z):
        """∇x·∇y (stella gds21/ŝ)."""
        f = self.extra["gxy"] if "gxy" in self.extra else np.asarray(self._need("gds21"), float) / self.shat
        return self._extra_interp("gxy", f)(z)

    def b_k(self, z, kx, ky):
        """FLR parameter ½ k⊥²/B² of the mode (k_x, k_y): stella kperp2 = k_y² gds2 + 2 k_x k_y gds21/ŝ + k_x² gds22/ŝ²."""
        if ky == 0:
            return self.b(z, kx)
        B2 = self.arrays["bmag"]
        Bz = self._extra_interp("bmag", B2)(z)
        return 0.5 * (kx**2 * self.kperp2_kx2(z) * Bz**2 + 2 * kx * ky * self.gxy(z) + ky**2 * self.gyy(z)) / Bz**2

    def binormal_periodic(self, tol=1e-6):
        """True if the k_y ≠ 0 coefficients (gds2, gds21, gbdrift) are periodic over the tube (e.g. ŝ = 0)."""
        for key in ("gds2", "gbdrift"):
            f = np.asarray(self._need(key), float)
            if abs(f[0] - f[-1]) > tol * max(1.0, np.abs(f).max()):
                return False
        f = np.asarray(self.extra["gxy"] if "gxy" in self.extra else self._need("gds21"), float)
        return abs(f[0] - f[-1]) <= tol * max(1.0, np.abs(f).max())

    def gradpar(self, z):
        if not hasattr(self, "_gradpar"):
            f = np.asarray(self.arrays["gradpar"], float).copy()
            if self.periodic:
                f[-1] = f[0]
            self._gradpar = self._interp(self.z_grid, f)
        return self._gradpar(z)

    def grho(self, z):
        if "grho" not in self.extra:
            return Geometry.grho(self, z)
        if not hasattr(self, "_grho"):
            f = np.asarray(self.extra["grho"], float).copy()
            if self.periodic:
                f[-1] = f[0]
            self._grho = self._interp(self.z_grid, f)
        return self._grho(z)
