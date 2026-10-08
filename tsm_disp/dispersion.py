"""Dispersion relation of secondary modes, Nies & Parra (2026) eqs. (31)-(32):

    D(ω, k_x) = 1 − ⟨1/D_ISM⟩ = ∫ dz w(z) (1/2π)∫ dy (D_ISM − 1)/D_ISM,

with w the normalised field-line weight 1/(B·∇z), D_ISM = (1 − Γ0(b) + τ − N)/τ and N the ion
response (λ-integral form, exact FLR; or the small-b closed form). The y integral runs over the
primary phase; for Im ω ≤ 0 it is taken on a deformed contour that analytically continues D.
"""
import warnings
import numpy as np
from . import _kernels

_XGK21 = np.array([0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
                   0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
                   0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
                   0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
                   0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0])
_WGK21 = np.array([0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
                   0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
                   0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
                   0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
                   0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
                   0.149445554002916905664936468389821])
_WG10 = np.array([0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
                  0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
                  0.295524224714752870173892994651338])
# nodes on [-1, 1] in increasing order, Kronrod and Gauss weights aligned with them
_X = np.concatenate([-_XGK21[:-1], [0.0], _XGK21[-2::-1]])
_WK = np.concatenate([_WGK21[:-1], [_WGK21[-1]], _WGK21[-2::-1]])
_WGfull = np.zeros(21)
_gidx = [1, 3, 5, 7, 9]               # Gauss nodes are XGK21[1,3,...,9]
for j, i in enumerate(_gidx):
    _WGfull[i] = _WG10[j]
    _WGfull[20 - i] = _WG10[j]

FORMS = {"lambda": 0, "Z": 1}
CONTOURS = {"real": 0, "real_points": 1, "deformed": 2}


class FlagWarning(RuntimeWarning):
    pass


class DispersionRelation:
    """D(ω, k_x) for a geometry and a primary.

    Parameters
    ----------
    geometry : tsm_disp.geometry.Geometry
    primary : tsm_disp.primary.Primary
    tau : T_i/(Z T_e)
    eta_par, eta_prp : primary temperature-to-density ratios η∥, η⊥ (old names alpha_par, alpha_prp)
    delta_par, delta_prp : phase shifts of the primary temperature perturbations
    form : 'lambda' (exact FLR, eq. 34) or 'Z' (small-b closed form, the old disp_model 0.1)
    y_contour : 'auto' (real for Im ω > 0, deformed otherwise), 'real', 'real_points' (old code's
        prescription: real axis split where Re ω̄ = 0), 'deformed'
    epsabs, epsrel : tolerances of the field-line integral; epsabs_y, epsrel_y of the y integral;
        tolN of the velocity (λ) integral
    nthreads : OpenMP threads for the y integrals of a batch of field-line points
    theta_panels : initial number of equal panels of the field-line integral
    """

    def __init__(self, geometry, primary, *, tau=1.0, eta_par=0.0, eta_prp=0.0, delta_par=0.0,
                 delta_prp=0.0, form="lambda", y_contour="auto", epsabs=1e-10, epsrel=1e-10,
                 epsabs_y=None, epsrel_y=None, tolN=1e-12, nthreads=1, theta_panels=8,
                 asymp_J11=False, eta_fixed=0.0, eta_max=2.0, limit=400, warn=True):
        self.geometry, self.primary = geometry, primary
        self.tau, self.eta_par, self.eta_prp = float(tau), float(eta_par), float(eta_prp)
        self.delta_par, self.delta_prp = float(delta_par), float(delta_prp)
        if form not in FORMS:
            raise ValueError("form must be 'lambda' or 'Z'")
        self.form = form
        self.y_contour = y_contour
        self.epsabs, self.epsrel = epsabs, epsrel
        self.epsabs_y = 0.1 * epsabs if epsabs_y is None else epsabs_y
        self.epsrel_y = 0.1 * epsrel if epsrel_y is None else epsrel_y
        self.tolN, self.nthreads = tolN, nthreads
        self.theta_panels, self.asymp_J11 = theta_panels, asymp_J11
        self.eta_fixed, self.eta_max, self.limit, self.warn = eta_fixed, eta_max, limit, warn
        self.last = {}

    # -- local quantities -------------------------------------------------------------------------
    def _contour(self, omega):
        if self.y_contour == "auto":
            return CONTOURS["real"] if np.imag(omega) > 0 else CONTOURS["deformed"]
        return CONTOURS[self.y_contour]

    def _local(self, z, kx):
        z = np.ascontiguousarray(z, dtype=float)
        A = np.ascontiguousarray(kx * self.primary.amplitude(z), dtype=float)
        wM = np.ascontiguousarray(kx * self.geometry.vMx(z), dtype=float)
        b = np.ascontiguousarray(self.geometry.b(z, kx), dtype=float)
        return A, wM, b

    def y_average(self, omega, kx, z):
        """(1/2π)∫dy (D_ISM − 1)/D_ISM at field-line points z (array). Returns (values, flags, nevals)."""
        A, wM, b = self._local(np.atleast_1d(z), kx)
        return _kernels.y_average(complex(omega), A, wM, b, self.tau, self.eta_par, self.eta_prp,
                                  self.delta_par, self.delta_prp, FORMS[self.form], self._contour(omega),
                                  self.eta_fixed, self.eta_max, self.epsabs_y, self.epsrel_y, self.tolN,
                                  bool(self.asymp_J11), self.limit, self.nthreads)

    def D_ism(self, omega, kx, z, y):
        """Local kinetic response D_ISM(z, y) (eq. 31)."""
        A, wM, b = self._local(np.atleast_1d(z), kx)
        return _kernels.D_ism_py(complex(omega), complex(y), A[0], wM[0], b[0], self.tau, self.eta_par,
                                 self.eta_prp, self.delta_par, self.delta_prp, FORMS[self.form], self.tolN,
                                 bool(self.asymp_J11))

    def D_theta(self, omega, kx, z):
        """Field-line integrand w(z) · (1/2π)∫dy (D_ISM − 1)/D_ISM."""
        v, f, n = self.y_average(omega, kx, z)
        return self.geometry.weight(np.atleast_1d(z)) * v

    # -- field-line integral ------------------------------------------------------------------------
    def _panels(self):
        g = self.geometry
        edges = np.linspace(g.zmin, g.zmax, self.theta_panels + 1)
        extra = [p for p in g.breakpoints() if g.zmin < p < g.zmax]
        c = getattr(self.primary, "center", None)
        if c is not None and g.zmin < c < g.zmax:
            extra.append(c)
        edges = np.unique(np.concatenate([edges, extra]))
        return list(zip(edges[:-1], edges[1:]))

    def _eval_panels(self, omega, kx, panels):
        a = np.array([p[0] for p in panels])
        bb = np.array([p[1] for p in panels])
        c = 0.5 * (a + bb)
        h = 0.5 * (bb - a)
        z = (c[:, None] + h[:, None] * _X[None, :]).ravel()
        v, flags, nev = self.y_average(omega, kx, z)
        f = (self.geometry.weight(z) * v).reshape(len(panels), 21)
        resk = (f * _WK).sum(axis=1) * h
        resg = (f * _WGfull).sum(axis=1) * h
        mean = resk / (2 * h)
        resasc = (np.abs(f - mean[:, None]) * _WK).sum(axis=1) * np.abs(h)
        err = np.abs(resk - resg)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(resasc > 0, 200 * err / resasc, 0.0)
            err = np.where((resasc > 0) & (err > 0), resasc * np.minimum(1.0, r**1.5), err)
        resabs = (np.abs(f) * _WK).sum(axis=1) * np.abs(h)
        err = np.maximum(err, 50 * np.finfo(float).eps * resabs)
        return resk, err, int(np.bitwise_or.reduce(flags)), int(nev.sum())

    def D(self, omega, kx):
        """Dispersion function D(ω, k_x) (eq. 32). Roots D = 0 are the secondary modes."""
        panels = self._panels()
        vals, errs, flag, nev = self._eval_panels(omega, kx, panels)
        vals, errs = list(vals), list(errs)
        npan_eval = len(panels)
        while True:
            total = sum(vals)
            errsum = sum(errs)
            tol = max(self.epsabs, self.epsrel * abs(total))
            if errsum <= tol:
                break
            if len(panels) >= self.limit:
                flag |= 16
                break
            # bisect every panel whose error exceeds its share, evaluating all new panels in one batch
            share = tol / len(panels)
            idx = [i for i, e in enumerate(errs) if e > share] or [int(np.argmax(errs))]
            new = []
            for i in idx:
                a, b = panels[i]
                m = 0.5 * (a + b)
                new += [(a, m), (m, b)]
            v2, e2, f2, n2 = self._eval_panels(omega, kx, new)
            flag |= f2
            nev += n2
            npan_eval += len(new)
            for k, i in enumerate(sorted(idx, reverse=True)):
                del panels[i], vals[i], errs[i]
            panels += new
            vals += list(v2)
            errs += list(e2)
        self.last = dict(err=errsum, flag=flag, nevals=nev, npanels=len(panels), npanels_evaluated=npan_eval)
        if flag and self.warn:
            warnings.warn("D(ω=%s, kx=%g): flags %d (%s)" % (omega, kx, flag, describe_flags(flag)), FlagWarning)
        return total


def describe_flags(flag):
    names = {1: "λ-quadrature limit", 2: "continuation cancellation > e^25", 4: "y-quadrature limit",
             8: "y branch points pinch the contour", 16: "field-line quadrature limit"}
    return ", ".join(v for k, v in names.items() if flag & k)
