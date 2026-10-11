"""Analytic geometries: user-supplied functions of θ, the large-aspect-ratio circular model of the original code,
and a uniform slab."""
import numpy as np
from .base import Geometry


class CircularAnalytic(Geometry):
    """The geometry hard-coded in code_eval_disp_NEW (large-aspect-ratio circular tokamak):

        b = ½ k_x² (1 + ε cos θ)²,   weight ∝ 1 + ε cos θ,   vMx = sin θ / rmaj  (or a callable T_D),
        b·∇θ = 1/(q rmaj)  (if q is given; parallel-streaming solvers only).

    eps defaults to 0.5/2.7778 = 0.18 and rmaj to 2.7778 (CBC at r = a/2, units of a).
    """

    def __init__(self, eps=0.5 / 2.7778, rmaj=2.7778, vMx=None, q=None, shat=None):
        self.eps, self.rmaj, self.q, self.shat = eps, rmaj, q, shat
        self._vMx = vMx
        self.name = "circular eps=%.4g" % eps

    def vMx(self, z):
        if self._vMx is None:
            return np.sin(z) / self.rmaj
        v = self._vMx(z)
        return np.broadcast_to(v, np.shape(z)).astype(float) if np.ndim(z) else float(v)

    def kperp2_kx2(self, z):
        return (1.0 + self.eps * np.cos(z))**2

    def weight(self, z):
        return (1.0 + self.eps * np.cos(z)) / (2 * np.pi)

    def gradpar(self, z):
        if self.q is None:
            return Geometry.gradpar(self, z)
        g = 1.0 / (self.q * self.rmaj)
        return np.full(np.shape(z), g) if np.ndim(z) else g


class Uniform(Geometry):
    """θ-independent geometry: b = ½ k_x², constant weight, constant drift vMx (default 0: RDK limit)."""

    def __init__(self, vMx=0.0):
        self.v = float(vMx)
        self.name = "uniform vMx=%g" % self.v

    def vMx(self, z):
        return np.full(np.shape(z), self.v) if np.ndim(z) else self.v

    def kperp2_kx2(self, z):
        return np.ones(np.shape(z)) if np.ndim(z) else 1.0

    def weight(self, z):
        return np.full(np.shape(z), 1 / (2 * np.pi)) if np.ndim(z) else 1 / (2 * np.pi)


def _as_function(f):
    """A vectorised callable of θ from a callable or a constant."""
    if callable(f):
        return lambda z: np.broadcast_to(f(z), np.shape(z)).astype(float) if np.ndim(z) else float(f(z))
    c = float(f)
    return lambda z: np.full(np.shape(z), c) if np.ndim(z) else c


class AnalyticGeometry(Geometry):
    """Geometry given analytically as functions of the field-line angle θ ∈ [zmin, zmax] (default [−π, π]):

        vMx(θ)       radial magnetic drift per unit (v̂∥² + v̂⊥²/2), in ρ_ref v_th/a
        gradx(θ)     |∇x| (ρ_ref units, so that b = ½ k_x² |∇x|²/B²); default 1
        B(θ)         B/B_ref; default 1
        jacobian(θ)  1/(B·∇θ) up to a constant (the field-line average weight); default 1/B(θ),
                     i.e. b·∇θ uniform. The weight is normalised to ∫ weight dθ = 1.
        grho(θ)      optional |∇ρ| (1/a), only for the flux normalisation of primary.heat_flux_stella
        gradpar(θ)   optional b·∇θ (1/a), only for the parallel-streaming solvers (circular: 1/(q R))
        vMy, gyy, gxy  optional binormal drift (per unit energy), |∇y|² and ∇x·∇y, only for the finite-k_y solver
                     (circular ŝ = 0 model: vMy = cos θ/R, gyy = 1, gxy = 0)

    Each argument is a vectorised callable or a constant. `breakpoints` lists interior angles where the
    θ quadrature should start a new panel (discontinuities or sharp features of the functions).

    Example (the model of the original code): AnalyticGeometry(vMx=lambda t: np.sin(t)/2.7778,
    gradx=lambda t: 1 + 0.18*np.cos(t), jacobian=lambda t: 1 + 0.18*np.cos(t)).
    """

    def __init__(self, vMx, gradx=1.0, B=1.0, jacobian=None, *, grho=None, gradpar=None, vMy=None, gyy=None,
                 gxy=None, zmin=-np.pi, zmax=np.pi, periodic=True,
                 breakpoints=(), shat=None, q=None, eps=None, rmaj=None, exb_nonlin_fac=None, sign_torflux=None,
                 name="analytic"):
        self._vMx, self._gradx, self._B = _as_function(vMx), _as_function(gradx), _as_function(B)
        self._jac = _as_function(jacobian) if jacobian is not None else (lambda z: 1.0 / self._B(z))
        self.zmin, self.zmax, self.periodic = float(zmin), float(zmax), periodic
        self._breakpoints = sorted(float(p) for p in breakpoints)
        self.shat, self.q, self.eps, self.rmaj = shat, q, eps, rmaj
        self.exb_nonlin_fac, self.sign_torflux = exb_nonlin_fac, sign_torflux
        self._grho_f = _as_function(grho) if grho is not None else None
        self._gradpar_f = _as_function(gradpar) if gradpar is not None else None
        self._vMy_f = _as_function(vMy) if vMy is not None else None
        self._gyy_f = _as_function(gyy) if gyy is not None else None
        self._gxy_f = _as_function(gxy) if gxy is not None else _as_function(0.0)
        self.name = name
        from scipy.integrate import quad
        pts = [p for p in self._breakpoints if self.zmin < p < self.zmax] or None
        self._jnorm = quad(lambda z: self._jac(z), self.zmin, self.zmax, points=pts, epsabs=1e-14, epsrel=1e-13,
                           limit=500)[0]

    def vMx(self, z):
        return self._vMx(z)

    def gradx(self, z):
        return self._gradx(z)

    def bmag(self, z):
        return self._B(z)

    def kperp2_kx2(self, z):
        return self._gradx(z)**2 / self._B(z)**2

    def weight(self, z):
        return self._jac(z) / self._jnorm

    def grho(self, z):
        if self._grho_f is None:
            return Geometry.grho(self, z)
        return self._grho_f(z)

    def gradpar(self, z):
        if self._gradpar_f is None:
            return Geometry.gradpar(self, z)
        return self._gradpar_f(z)

    def _binormal(self, f, name):
        if f is None:
            raise NotImplementedError("AnalyticGeometry needs %s= for finite k_y" % name)
        return f

    def vMy(self, z):
        return self._binormal(self._vMy_f, "vMy")(z)

    def gyy(self, z):
        return self._binormal(self._gyy_f, "gyy")(z)

    def gxy(self, z):
        return self._gxy_f(z)

    def b_k(self, z, kx, ky):
        """½ k⊥²/B², k⊥² = k_x²|∇x|² + 2 k_x k_y ∇x·∇y + k_y²|∇y|²."""
        if ky == 0:
            return self.b(z, kx)
        return 0.5 * (kx**2 * self._gradx(z)**2 + 2 * kx * ky * self.gxy(z) + ky**2 * self.gyy(z)) / self._B(z)**2

    def breakpoints(self):
        return list(self._breakpoints)
