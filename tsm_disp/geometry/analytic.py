"""Analytic geometries: the large-aspect-ratio circular model of the original code, and a uniform slab."""
import numpy as np
from .base import Geometry


class CircularAnalytic(Geometry):
    """The geometry hard-coded in code_eval_disp_NEW (large-aspect-ratio circular tokamak):

        b = ½ k_x² (1 + ε cos θ)²,   weight ∝ 1 + ε cos θ,   vMx = sin θ / rmaj  (or a callable T_D).

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
