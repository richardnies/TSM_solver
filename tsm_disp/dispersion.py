"""Dispersion relation of secondary modes, Nies & Parra (2026) eqs. (31)-(32):

    D(ω, k_x) = 1 − ⟨1/D_ISM⟩ = ∫ dz w(z) (1/2π)∫ dy (D_ISM − 1)/D_ISM,

with w the normalised field-line weight 1/(B·∇z), D_ISM = (1 − Γ0(b) + τ − N)/τ and N the ion
response (λ-integral form, exact FLR; or the small-b closed form). The y integral runs over the
primary phase; for Im ω ≤ 0 it is taken on a deformed contour that analytically continues D.
"""
import warnings
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz    # numpy ≥ 2: trapezoid; < 2: trapz
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
                 asymp_J11=False, eta_fixed=0.0, eta_max=2.0, limit=400, warn=True, subtract_poles=True):
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
        self.subtract_poles = subtract_poles
        self.last = {}
        self._geo_cache = {}

    # -- local quantities -------------------------------------------------------------------------
    def _contour(self, omega):
        if self.y_contour == "auto":
            return CONTOURS["real"] if np.imag(omega) > 0 else CONTOURS["deformed"]
        return CONTOURS[self.y_contour]

    def _geo_at(self, z):
        """(vMx, b at k_x = 1, weight) at the nodes z, cached: the θ nodes of the field-line integral are the
        same for every ω and k_x."""
        g = self.geometry
        key = (id(g), z.tobytes())
        c = self._geo_cache.get(key)
        if c is None:
            if len(self._geo_cache) > 4096:
                self._geo_cache.clear()
            c = (np.ascontiguousarray(g.vMx(z), dtype=float), np.ascontiguousarray(g.b(z, 1.0), dtype=float),
                 np.ascontiguousarray(g.weight(z), dtype=float))
            self._geo_cache[key] = c
        return c

    def _local(self, z, kx):
        z = np.ascontiguousarray(z, dtype=float)
        vMx, b1, _ = self._geo_at(z)
        A = np.ascontiguousarray(kx * self.primary.amplitude(z), dtype=float)
        return A, kx * vMx, kx**2 * b1

    def y_average(self, omega, kx, z):
        """(1/2π)∫dy (D_ISM − 1)/D_ISM at field-line points z (array). Returns (values, flags, nevals)."""
        A, wM, b = self._local(np.atleast_1d(z), kx)
        return _kernels.y_average(complex(omega), A, wM, b, self.tau, self.eta_par, self.eta_prp,
                                  self.delta_par, self.delta_prp, FORMS[self.form], self._contour(omega),
                                  self.eta_fixed, self.eta_max, self.epsabs_y, self.epsrel_y, self.tolN,
                                  bool(self.asymp_J11), 200, self.nthreads, bool(self.subtract_poles))

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
        extra += [p for p in getattr(self.primary, "breakpoints", []) if g.zmin < p < g.zmax]
        edges = np.unique(np.concatenate([edges, extra]))
        return list(zip(edges[:-1], edges[1:]))

    def _eval_panels(self, omega, kx, panels):
        a = np.array([p[0] for p in panels])
        bb = np.array([p[1] for p in panels])
        c = 0.5 * (a + bb)
        h = 0.5 * (bb - a)
        z = (c[:, None] + h[:, None] * _X[None, :]).ravel()
        v, flags, nev = self.y_average(omega, kx, z)
        f = (self._geo_at(np.ascontiguousarray(z, dtype=float))[2] * v).reshape(len(panels), 21)
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
        self._resabs = resabs
        return resk, err, int(np.bitwise_or.reduce(flags & ~64)), int(nev.sum())

    def D(self, omega, kx):
        """Dispersion function D(ω, k_x) (eq. 32). Roots D = 0 are the secondary modes."""
        panels = self._panels()
        vals, errs, flag, nev = self._eval_panels(omega, kx, panels)
        vals, errs, rabs = list(vals), list(errs), list(self._resabs)
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
            # bisect every panel whose error exceeds its share (and is not at its round-off floor),
            # evaluating all new panels in one batch
            share = tol / len(panels)
            floor = [60 * np.finfo(float).eps * r for r in rabs]
            idx = [i for i, e in enumerate(errs) if e > share and e > floor[i]]
            if not idx:
                flag |= 32
                break
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
                del panels[i], vals[i], errs[i], rabs[i]
            panels += new
            vals += list(v2)
            errs += list(e2)
            rabs += list(self._resabs)
        self.last = dict(err=errsum, flag=flag, nevals=nev, npanels=len(panels), npanels_evaluated=npan_eval)
        if flag and self.warn:
            warnings.warn("D(ω=%s, kx=%g): flags %d (%s)" % (omega, kx, flag, describe_flags(flag)), FlagWarning)
        return total


def describe_flags(flag):
    names = {1: "λ-quadrature limit", 2: "continuation cancellation > e^25", 4: "y-quadrature limit",
             8: "y branch points pinch the contour", 16: "field-line quadrature limit",
             32: "field-line quadrature at round-off", 64: "near-real D_ISM zeros subtracted"}
    return ", ".join(v for k, v in names.items() if flag & k)


QUANTITIES = ("potential", "pressure", "heat_flux")


class ModeStructure:
    """Structure of a secondary mode on the flux surface, f(z, y) e^{i k_x x − iωt}.

    Attributes: z (field-line coordinate), y (primary phase, y = k_y^P × binormal coordinate),
    field (complex, shape (nz, ny)), y_average ⟨f⟩_y(z), quantity, omega, kx, and
    phi_average = ⟨φ̂⟩_ψ (= 1 at an exact root of D, a check of the root).
    Normalisation: Z e ⟨φ⟩_ψ/T = 1. Units: φ̂ in T/(Ze); P̂ in n T; Q̂_NL in n T ρ_ref v_th/a.
    """

    def __init__(self, z, y, field, quantity, omega, kx, weight, phi_average):
        self.z, self.y, self.field, self.quantity = z, y, field, quantity
        self.omega, self.kx, self.phi_average = omega, kx, phi_average
        self.weight = weight
        self.y_average = field.mean(axis=1)            # uniform periodic y grid: trapezoid = mean

    def real_space(self, x=0.0, t=0.0):
        """Re[f(z, y) e^{i k_x x − iωt}] on the (z, y) grid."""
        return np.real(self.field * np.exp(1j * self.kx * x - 1j * self.omega * t))

    def flux_surface_average(self):
        """⟨f⟩_ψ = ∫ w(z) ⟨f⟩_y dz (trapezoid on the z grid)."""
        return _trapezoid(self.weight * self.y_average, self.z)


def _mode_structure(self, omega, kx, quantity="potential", nz=129, ny=128, z=None, y=None):
    """Mode structure of the secondary at (ω, k_x) on the flux surface; quantity is one of

    'potential' : φ̂(z, y) = ⟨φ⟩_ψ / D_ISM(z, y)                         (quasineutrality, eq. 29)
    'pressure'  : P̂ = T ∫d³v (v∥² + v⊥²/2)/v_th² ĝ                       (paper eq. 38)
    'heat_flux' : Q̂_NL = v_Ex^P P̂, the radial E×B flux of P by the primary (paper eq. 40);
                  its y average carries the up–down asymmetry that drives the TSM.

    ĝ = J0 F_M φ̂ (ω1 + 2ω2 v∥² + ω3 v⊥²)/(ω̄ − ω_M E) is the gyrocenter response (paper eq. 28).
    The moments are always evaluated in the λ form (exact FLR); φ̂ uses the form of this object.
    ω should be a root (D(ω, k_x) ≈ 0): then ⟨φ̂⟩_ψ = 1, reported as `phi_average`.
    """
    if quantity not in QUANTITIES:
        raise ValueError("quantity must be one of %s" % (QUANTITIES,))
    g = self.geometry
    z = np.linspace(g.zmin, g.zmax, nz) if z is None else np.asarray(z, float)
    y = np.linspace(-np.pi, np.pi, ny, endpoint=False) if y is None else np.asarray(y, float)
    A, wM, b = self._local(z, kx)
    phi, P, flags = _kernels.mode_fields(complex(omega), A, wM, b, np.ascontiguousarray(y), self.tau,
                                         self.eta_par, self.eta_prp, self.delta_par, self.delta_prp,
                                         FORMS[self.form], self.tolN, quantity != "potential", self.nthreads)
    w = g.weight(z)
    phi_avg = _trapezoid(w * phi.mean(axis=1), z)
    if quantity == "potential":
        f = phi
    elif quantity == "pressure":
        f = P
    else:
        vEx = -(A / kx)[:, None] * np.sin(y)[None, :]          # v_Ex^P = −Φ_P T_P(z) sin y
        f = vEx * P
    if np.any(flags & 3) and self.warn:
        warnings.warn("mode_structure: velocity-integral flags %d" % np.bitwise_or.reduce(flags.ravel()), FlagWarning)
    return ModeStructure(z, y, f, quantity, omega, kx, w, phi_avg)


DispersionRelation.mode_structure = _mode_structure
