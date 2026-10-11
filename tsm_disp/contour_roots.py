"""All roots of 𝒟(ω) in a rectangle of the complex plane, without an initial guess.

For f analytic on a closed rectangle R apart from poles, none of them or of its zeros on ∂R,

    N = Z − P = (1/2π) Δ_∂R arg f                                   (argument principle)
    s_k = (1/2πi) ∮_∂R ω^k f′/f dω = Σ_zeros ζ^k − Σ_poles p^k
        = ω₀^k N − (k/2πi) ∮_∂R ω^{k−1} log f dω                    (Delves–Lyness moments, by parts)

with log f continued along ∂R from ω₀ (it returns as log f(ω₀) + 2πiN). Both need samples of f on ∂R only:
no derivative and no initial guess.

find_all_roots(target, kx, wr_range, wi_range):

1. Sampling. ∂R lies on a dyadic lattice; every edge starts from a grid of spacing ≈ max(side)/n_init (at least 4
   segments) and its segments are bisected until consecutive samples differ in phase by less than θ_max = π/4 (and,
   with a pole-free F, in arg F too); N is the sum of the principal phase steps. A segment still unresolved at
   1e-9 × the box size holds a zero (or a discontinuity) on the edge; an edge needing more than 8 n_init samples
   (phase winding fast next to a pole forest) is flagged. Samples are cached on the lattice, so neighbouring cells
   and the children of a split cell share them; the evaluations of each refinement pass form one batch (thread
   pool with `workers`, or any `executor`).
2. Isolation. A cell with more than n_moments zeros is bisected across its longer side; a split line that meets a
   zero is shifted (3/8, 5/8, ...).
3. Location. In a cell with 1 ≤ Z ≤ n_moments the zeros are the roots of the polynomial with power sums s₁ … s_Z
   (Newton's identities; s_k by a not-a-knot cubic spline of log f in arc length along each edge, after refining
   every edge to ≥ moment_segments segments and where log f changes by more than 1 between samples; error estimate
   from the spline through every other sample). Each estimate is polished (DispersionRelation: Newton with the
   exact ∂𝒟/∂ω, first at the sampling tolerance, then at full accuracy; otherwise Muller), abandoned if it leaves
   the box (1.5 × its diagonal) or meets a flagged evaluation, and accepted if converged, inside the cell and
   |f| ≤ ftol. The multiplicities m_i are those for which Σ m_i ζ_i^k reproduces s_k for k ≤ max(Z, 3) within the
   quadrature error; otherwise (a zero–pole pair, a missed zero, a non-analytic f) the cell is bisected (at most
   max_fails times along a lineage). A cell without zeros must have s₁, s₂, s₃ ≈ 0 as well (check_empty).
4. Verification. At each simple root the Cauchy–Riemann defect |∂f/∂ω̄|/|∂f/∂ω| (four-point stencil, step
   min(cr_step, 1e-2 × the distance to the nearest other root or pole)) is O(h²) for an analytic f and O(1) where f
   is not analytic (and, with an exact derivative, |f′ − ∂_x f|/|f′| as well); the residue of 1/f is 1/f′(ω₀).

Poles. With a pole-free F (zeros of 𝒟 only): BinormalDispersion F = 𝒟 det D; StreamingDispersion
F = 𝒟 Π_g det D(y_g) over the distinct matrices of the y-rule (nodes y and π − y share D when Δ = 0), whose zeros
are the poles of the discrete y-rule. Then Z = (1/2π)Δ arg F and P = Z − N in every cell, the zeros are located from
the moments of F whatever the poles in their cell, and a cell with P = 1 also gives its pole (s₁(f) = Σζ − p,
polished as a zero of 1/f). Without F (poles possible), N < 0 marks poles, and every N = 0 cell and every located
cell must reproduce s₁ … s₃: zero–pole pairs (e.g. the symmetric zeros and poles of tan 4ω, s₁ = 0) are bisected
until separated (down to max_depth_poles); pairs closer than ~dipole_tol × the cell size are invisible. A zero of
det D(y_j) that does not couple to the flux-surface average would be a spurious zero of F (it is not located).
The zeros of a discrete y-rule's 𝒟 between its poles (the discretised continuum, γ ≈ ISM growth rates) are genuine
zeros of the discrete function: confirm=<the solver with more y nodes> re-polishes every root and rejects those
that move.

Non-analytic regions. 𝒟 of DispersionRelation is continuous but not analytic in the ISM region (D_ISM = 0 on the
surface for some real (z, y): the y-average jumps where a zero y_p of D_ISM crosses the real axis, and its
field-line integral is non-analytic; local_ism_roots maps the region). There the field-line quadrature cannot
converge: with the sampling limit of 32 panels (normally 8–14) it stops with flag 16, at 1–6 s per evaluation with
8 threads (60 s at the default limit and tolerance). Such samples are flagged ('suspect'): their segments are not
refined, cells touching them are bisected regardless of N (the winding of a non-analytic region is arbitrary: the
whole CBC box at k_x = 0.5 has N = 0 although it holds the TSM) down to max_depth_suspect, roots are located in them
when N allows, and the remaining cells are reported in suspect_regions; from each terminal suspect cell, rescue_polish
polishes start at its smallest unflagged |f| samples (roots just outside the region). A polish that meets a flagged
evaluation is abandoned, a root whose CR defect exceeds cr_tol is rejected, with poles excluded (λ form) a negative
winding marks a non-analytic cell, and so do non-vanishing moments in a cell without zeros (a non-analytic island
inside the cell that no sample touches, e.g. the tip of the ISM region next to the TSM at η = 1.1). probe=True also
tests the centre of every cell found empty (4 evaluations). For the discrete solvers the ISM region is a forest of
poles and zeros of the y-rule: flagged edges, zero–pole pairs, a 'cluster' (more than max_cluster zeros + poles in a
cell below cluster_size), and zeros rejected by `confirm`.

Domains. valid(ω) restricts the sampling: cells whose boundary meets invalid (or non-finite) samples are bisected
down to max_depth_invalid and the rest reported in `excluded`. StreamingDispersion and BinormalDispersion default to
their velocity contour's domain (default_validity). Only boundaries are sampled: a flagged or invalid region that no
cell boundary crosses is seen only through the moments (check_empty) or probe=True.
"""
import bisect
import copy
import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .roots import find_root, RootResult

_K = 40                                   # lattice bits per axis
_OK, _SUSPECT, _INVALID, _BAD = 0, 1, 2, 3
_SPLIT_FRACTIONS = (0.5, 0.375, 0.625, 0.4375, 0.5625)


def _wrap(x):
    """Principal value in [−π, π]."""
    return x - 2 * np.pi * np.round(x / (2 * np.pi))


def _recip(v):
    return 1.0 / v if v != 0 else complex(np.inf)


# ----------------------------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Root:
    """A verified zero ω₀ of f. residue = Res_{ω₀}(1/f) = 1/f′(ω₀) (simple roots); cr_defect = |∂f/∂ω̄|/|∂f/∂ω|."""
    omega: complex
    D: complex
    multiplicity: int = 1
    residue: complex = None
    cr_defect: float = float("nan")
    cell: tuple = None
    converged: bool = True
    nfev: int = 0
    note: str = ""
    shift: float = float("nan")        # |Δω| when re-polished with the `confirm` solver


@dataclass
class Pole:
    """A pole p of f: residue = Res_p f (from 1/(1/f)′)."""
    omega: complex
    residue: complex = None
    order: int = 1
    cell: tuple = None
    converged: bool = True
    note: str = ""


@dataclass
class Region:
    """A cell not resolved into roots: rect = (wr0, wr1, wi0, wi1); N = winding of f, Z = of the pole-free F."""
    rect: tuple
    reason: str
    N: int = None
    Z: int = None
    detail: str = ""


@dataclass
class RootSearch:
    """Result of find_all_roots. roots: verified zeros (sorted by Im ω); poles; rejected: polished candidates that
    failed verification (note says why); suspect_regions, excluded: unresolved / invalid cells; N, Z: winding of f
    and of F around the box; nfev: evaluations (sampling, polish + verification); cells: terminal-cell log."""
    roots: list
    poles: list
    rejected: list
    suspect_regions: list
    excluded: list
    N: int
    Z: int
    nfev: int
    nfev_sampling: int
    nfev_polish: int
    seconds: float
    cells: list
    samples: tuple
    complete: bool = True

    def summary(self):
        out = ["%d root(s), %d pole(s), %d rejected, %d suspect region(s), %d excluded; winding N = %s%s; "
               "%d evaluations (%d sampling, %d polish), %.1f s%s"
               % (len(self.roots), len(self.poles), len(self.rejected), len(self.suspect_regions), len(self.excluded),
                  self.N, "" if self.Z is None else ", Z = %d" % self.Z, self.nfev, self.nfev_sampling,
                  self.nfev_polish, self.seconds, "" if self.complete else "  [INCOMPLETE: evaluation budget]")]
        for r in self.roots:
            res = complex(r.residue) if r.residue is not None else complex(np.nan, np.nan)
            out.append("  root  ω = %.10f %+.10fi  |f| = %.1e  m = %d  Res(1/f) = %.4g%+.4gi  CR = %.1e %s"
                       % (r.omega.real, r.omega.imag, abs(r.D), r.multiplicity, res.real, res.imag, r.cr_defect,
                          r.note))
        for p in self.poles:
            out.append("  pole  ω = %.8f %+.8fi  Res = %s %s" % (p.omega.real, p.omega.imag, p.residue, p.note))
        for r in self.rejected:
            out.append("  rejected ω = %.8f %+.8fi  |f| = %.1e  CR = %.1e  (%s)"
                       % (r.omega.real, r.omega.imag, abs(r.D), r.cr_defect, r.note))
        for g in self.suspect_regions:
            out.append("  suspect [%.4g, %.4g] × [%.4g, %.4g]  N = %s Z = %s: %s %s"
                       % (*g.rect, g.N, g.Z, g.reason, g.detail))
        for g in self.excluded:
            out.append("  excluded [%.4g, %.4g] × [%.4g, %.4g]: %s" % (*g.rect, g.reason))
        return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# Targets: one interface for the dispersion objects and plain callables
# ----------------------------------------------------------------------------------------------------------------
class _Target:
    """sample(ω) → (f, log|F|, arg F, suspect) for the boundary (cheap settings; F the pole-free function or None);
    f(ω) and fd(ω) → (f, f′) at full accuracy for polishing. local() is a per-thread clone."""
    poles = True            # may f have poles?
    has_det = False         # is a pole-free F available?
    has_derivative = False
    moment_floor = 1e-6     # absolute tolerance of the moment tests (cell units): sampling accuracy
    has_coarse = False      # f_coarse / fd_coarse: the cheap sampling settings, for a first polishing stage

    def __init__(self):
        self._tl = threading.local()

    def _clone(self):
        return self

    def local(self):
        c = getattr(self._tl, "c", None)
        if c is None:
            c = self._tl.c = self._clone()
        return c

    def polish_suspect(self):
        """True if the last f/fd evaluation of this (local) target was flagged (non-analytic region)."""
        return False


class _CallableTarget(_Target):
    def __init__(self, f, derivative=False, pole_free=None, poles=True, suspect=None):
        super().__init__()
        self._f, self.has_derivative, self._F = f, bool(derivative), pole_free
        self.has_det, self.poles, self._suspect = pole_free is not None, bool(poles), suspect

    def sample(self, w):
        v = self._f(w)
        v = complex(v[0] if self.has_derivative else v)
        lF = phF = None
        if self._F is not None:
            F = complex(self._F(w))
            lF, phF = (math.log(abs(F)) if F != 0 else -np.inf), math.atan2(F.imag, F.real)
        return v, lF, phF, bool(self._suspect(w)) if self._suspect is not None else False

    def f(self, w):
        v = self._f(w)
        return complex(v[0] if self.has_derivative else v)

    def fd(self, w):
        v, d = self._f(w)
        return complex(v), complex(d)


def _clone_with(obj, opts):
    c = copy.copy(obj)
    for k, v in opts.items():
        if not hasattr(c, k):
            raise AttributeError("%s has no attribute %r" % (type(obj).__name__, k))
        setattr(c, k, v)
    if hasattr(c, "last"):
        c.last = {}
    return c


class _LambdaTarget(_Target):
    """DispersionRelation (λ or Z form): no poles; exact ∂𝒟/∂ω; field-line quadrature flags mark the ISM region."""
    poles = False
    has_derivative = True
    moment_floor = 1e-3     # sampled with tolerance 1e-6, polished at full accuracy
    SUSPECT_FLAGS = 4 | 8 | 16          # y-quadrature limit, y branch points pinch, field-line quadrature limit

    def __init__(self, disp, kx, sample_options=None, polish_options=None):
        super().__init__()
        self.kx = float(kx)
        s = dict(warn=False, epsabs=max(disp.epsabs, 1e-6), epsrel=max(disp.epsrel, 1e-6),
                 epsabs_y=max(disp.epsabs_y, 1e-7), epsrel_y=max(disp.epsrel_y, 1e-7), tolN=max(disp.tolN, 1e-8),
                 limit=min(disp.limit, 32))
        s.update(sample_options or {})
        p = dict(warn=False, limit=min(disp.limit, 64))
        p.update(polish_options or {})
        self._s0, self._p0 = _clone_with(disp, s), _clone_with(disp, p)
        self.s, self.p = self._s0, self._p0

    def _clone(self):
        c = copy.copy(self)
        c.s, c.p = _clone_with(self._s0, {}), _clone_with(self._p0, {})
        return c

    has_coarse = True

    def sample(self, w):
        v = self.s.D(w, self.kx)
        return complex(v), None, None, bool(self.s.last.get("flag", 0) & self.SUSPECT_FLAGS)

    def f(self, w):
        self._cur = self.p
        return complex(self.p.D(w, self.kx))

    def fd(self, w):
        self._cur = self.p
        v, d = self.p.D(w, self.kx, derivative=True)
        return complex(v), complex(d)

    def f_coarse(self, w):
        self._cur = self.s
        return complex(self.s.D(w, self.kx))

    def fd_coarse(self, w):
        self._cur = self.s
        v, d = self.s.D(w, self.kx, derivative=True)
        return complex(v), complex(d)

    def polish_suspect(self):
        return bool(getattr(self, "_cur", self.p).last.get("flag", 0) & self.SUSPECT_FLAGS)


class _StreamingTarget(_Target):
    """StreamingDispersion: 𝒟 = 1 − τ⟨cᵀD(y)⁻¹e₀⟩_y, poles at det D(y_j) = 0; F = 𝒟 Π_g det D(y_g) is pole-free,
    the product running over the distinct matrices: D(y) depends on y only through (sin y, sin(y + Δ∥), sin(y + Δ⊥)),
    so nodes y and π − y (Δ = 0) share their poles, and counting both determinants would give F a spurious zero at
    every such pole."""

    def __init__(self, disp, kx, use_det=True):
        super().__init__()
        self.disp, self.kx, self.has_det = disp, float(kx), bool(use_det)
        Y = disp.Y
        keys = [tuple(np.round([np.sin(y), np.sin(y + disp.delta_par), np.sin(y + disp.delta_prp)], 12)) for y in Y]
        self.first = np.array([keys.index(k) == j for j, k in enumerate(keys)])

    def _eval(self, w):
        d = self.disp
        s = d._setup(self.kx)
        e0 = np.zeros(s["nphi"])
        e0[s["i0"]] = 1.0

        def one(iy):
            Dm = d.Dmatrix(w, self.kx, iy)
            sign, logdet = np.linalg.slogdet(Dm)
            return s["c"] @ np.linalg.solve(Dm, e0), sign, logdet

        if d.nthreads > 1:
            with ThreadPoolExecutor(d.nthreads) as ex:
                out = list(ex.map(one, range(d.n_y)))
        else:
            out = [one(i) for i in range(d.n_y)]
        v = 1.0 - d.tau * np.mean([o[0] for o in out])
        ph = float(np.sum([np.angle(o[1]) for o, u in zip(out, self.first) if u]))
        la = float(np.sum([o[2] for o, u in zip(out, self.first) if u]))
        return complex(v), la, ph

    def sample(self, w):
        w = complex(w)
        if not self.has_det:
            return complex(self.disp.D(w, self.kx)), None, None, False
        v, la, ph = self._eval(w)
        return v, (math.log(abs(v)) if v != 0 else -np.inf) + la, _wrap(math.atan2(v.imag, v.real) + ph), False

    def f(self, w):
        return complex(self.disp.D(complex(w), self.kx))


class _BinormalTarget(_Target):
    """BinormalDispersion: F = 𝒟 det D from D(ω, kx, with_det=True)."""

    def __init__(self, disp, kx, use_det=True):
        super().__init__()
        self.disp, self.kx, self.has_det = disp, float(kx), bool(use_det)

    def sample(self, w):
        if not self.has_det:
            return complex(self.disp.D(complex(w), self.kx)), None, None, False
        v, sign, logdet = self.disp.D(complex(w), self.kx, with_det=True)
        v = complex(v)
        return (v, (math.log(abs(v)) if v != 0 else -np.inf) + float(logdet),
                _wrap(math.atan2(v.imag, v.real) + float(np.angle(sign))), False)

    def f(self, w):
        return complex(self.disp.D(complex(w), self.kx))


def default_validity(disp, kx):
    """Domain where the solver's 𝒟 is the analytic function being sought (None: everywhere).

    StreamingDispersion / BinormalDispersion: real velocity contour (cdepth = 0) → Im ω > 0 (discretisation poles
    approach the real axis: keep Im ω ≳ 0.05); tanh-deformed contour (cdepth > 0) → Re ω > k_x Φ_P max T_P."""
    grid = getattr(disp, "grid", None)
    if grid is None or not hasattr(grid, "params"):
        return None
    if grid.params.get("cdepth", 0.0) == 0.0:
        return lambda w: complex(w).imag > 0
    z = np.linspace(disp.geometry.zmin, disp.geometry.zmax, 513)
    wmin = float(kx) * disp.primary.Phi_P * float(np.max(np.abs(disp.primary(z))))
    return lambda w: complex(w).real > wmin


def _make_target(target, kx, *, derivative, pole_free, poles, use_det, sample_options, polish_options, suspect):
    if isinstance(target, _Target):
        return target
    name = type(target).__name__
    if name == "DispersionRelation" or (hasattr(target, "y_average") and hasattr(target, "D_ism")):
        return _LambdaTarget(target, kx, sample_options, polish_options)
    if name == "StreamingDispersion" or (hasattr(target, "Dmatrix") and hasattr(target, "n_y")):
        return _StreamingTarget(target, kx, bool(use_det))
    if name == "BinormalDispersion" or (hasattr(target, "Dmatrix") and hasattr(target, "ky")):
        return _BinormalTarget(target, kx, bool(use_det))
    if callable(target):
        return _CallableTarget(target, derivative=derivative, pole_free=pole_free,
                               poles=True if poles is None else poles, suspect=suspect)
    raise TypeError("target must be a dispersion object or a callable ω → complex")


# ----------------------------------------------------------------------------------------------------------------
# Batched evaluation with cache and budget
# ----------------------------------------------------------------------------------------------------------------
class _Budget(Exception):
    pass


class _Flagged(Exception):
    """A polish evaluation was flagged (non-analytic region): abandon this polish."""


class _Sample:
    __slots__ = ("w", "f", "ph", "lf", "lF", "phF", "ok", "suspect", "err")

    def __init__(self, w, f=None, lF=None, phF=None, ok=False, suspect=False, err=""):
        self.w, self.f, self.ok, self.suspect, self.err = w, f, ok, suspect, err
        self.lF, self.phF = lF, phF
        if ok:
            self.ph = math.atan2(f.imag, f.real)
            self.lf = math.log(abs(f)) if f != 0 else -np.inf


class _Evaluator:
    def __init__(self, target, valid=None, workers=1, executor=None, max_evals=None):
        self.target, self.valid, self.max_evals = target, valid, max_evals
        self.workers, self._own = workers, None
        self.executor = executor
        if executor is None and workers > 1:
            self.executor = self._own = ThreadPoolExecutor(workers)
        self.nfev = 0
        self.nfev_polish = 0
        self.seconds = 0.0

    def close(self):
        if self._own is not None:
            self._own.shutdown()

    def _one(self, w):
        if self.valid is not None and not self.valid(w):
            return _Sample(w, err="invalid")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f, lF, phF, sus = self.target.local().sample(w)
        except Exception as e:                                  # an evaluation failure is an invalid point
            return _Sample(w, err="%s: %s" % (type(e).__name__, e))
        ok = np.isfinite(f) and f != 0 and (phF is None or (np.isfinite(phF) and np.isfinite(lF)))
        return _Sample(w, f, lF, phF, ok=bool(ok), suspect=sus, err="" if ok else "non-finite or zero")

    def batch(self, ws):
        if not ws:
            return []
        if self.max_evals is not None and self.nfev + len(ws) > self.max_evals:
            raise _Budget()
        t0 = time.time()
        if self.executor is not None and len(ws) > 1:
            out = list(self.executor.map(self._one, ws))
        else:
            out = [self._one(w) for w in ws]
        self.nfev += sum(1 for s in out if s.err != "invalid")
        self.seconds += time.time() - t0
        return out

    def map(self, fn, items):
        """Run fn over items (polish tasks) on the pool."""
        if self.executor is not None and len(items) > 1:
            return list(self.executor.map(fn, items))
        return [fn(x) for x in items]


class _Counted:
    """f, fd of the per-thread target, counting evaluations and recording flagged (suspect) ones; with abort=True a
    flagged evaluation raises _Flagged (they cost 10–100× more than ordinary ones)."""

    def __init__(self, ev, abort=False, coarse=False, region=None):
        self.ev, self.n, self.flagged, self.abort, self.region = ev, 0, False, abort, region
        self.t = ev.target.local()
        self._f, self._fd = (self.t.f_coarse, self.t.fd_coarse) if coarse else (self.t.f, getattr(self.t, "fd", None))

    def _range(self, w):
        """A polish leaving the region (centre, radius) is abandoned: far from the box the integrands can overflow."""
        if self.region is not None and abs(w - self.region[0]) > self.region[1]:
            raise _Flagged()

    def _check(self):
        if self.t.polish_suspect():
            self.flagged = True
            if self.abort:
                raise _Flagged()

    def f(self, w):
        self._range(w)
        self.n += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = self._f(complex(w))
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            return complex(np.inf)
        self._check()
        return v

    def fd(self, w):
        self._range(w)
        self.n += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = self._fd(complex(w))
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            return complex(np.inf), complex(np.nan)
        self._check()
        return v


# ----------------------------------------------------------------------------------------------------------------
# Lattice of boundary samples
# ----------------------------------------------------------------------------------------------------------------
class _Grid:
    """Samples on the dyadic lattice ω = a + (b−a) i/2^K + i[c + (d−c) j/2^K] of the box (a, b) × (c, d).

    Lines: orientation 0 = horizontal (fixed j, positions i), 1 = vertical (fixed i, positions j)."""

    def __init__(self, ev, rect, n_init=16, theta=np.pi / 4, hmin_rel=1e-9, min_seg=4, max_edge=None):
        (a, b), (c, d) = rect
        self.ev, self.a, self.c = ev, float(a), float(c)
        self.W, self.H = float(b - a), float(d - c)
        self.sx, self.sy = self.W / 2 ** _K, self.H / 2 ** _K
        self.theta, self.min_seg = theta, min_seg
        self.h0 = max(self.W, self.H) / n_init
        self.hmin = hmin_rel * max(self.W, self.H)
        self.pts = {}
        self.lines = ({}, {})
        self.seg = {}
        self.has_det = ev.target.has_det
        self.max_edge = max_edge or 8 * n_init       # samples per edge before its unresolved segments are 'suspect'

    def w(self, i, j):
        return complex(self.a + self.sx * i, self.c + self.sy * j)

    @staticmethod
    def key(o, line, s):
        return (s, line) if o == 0 else (line, s)

    def unit(self, o):
        return self.sx if o == 0 else self.sy

    def _evaluate(self, keys):
        keys = [k for k in keys if k not in self.pts]
        if not keys:
            return
        out = self.ev.batch([self.w(*k) for k in keys])
        for k, s in zip(keys, out):
            self.pts[k] = s
            bisect.insort(self.lines[0].setdefault(k[1], []), k[0])
            bisect.insort(self.lines[1].setdefault(k[0], []), k[1])

    def _positions(self, o, line, s0, s1):
        L = self.lines[o].get(line, [])
        return L[bisect.bisect_left(L, s0):bisect.bisect_right(L, s1)]

    def _initial(self, o, line, s0, s1):
        n = s1 - s0
        step = 1 << max(0, int(math.floor(math.log2(max(self.h0 / self.unit(o), 1.0)))))
        while step > 1 and n < self.min_seg * step:
            step >>= 1
        first = -(-s0 // step) * step
        return {s0, s1, *range(first, s1 + 1, step)}

    def segment(self, o, line, p, q):
        k = (o, line, p, q)
        st = self.seg.get(k)
        if st is not None:
            return st
        sp, sq = self.pts[self.key(o, line, p)], self.pts[self.key(o, line, q)]
        if not (sp.ok and sq.ok):
            st = _INVALID
        elif sp.suspect or sq.suspect:
            st = _SUSPECT
        else:
            good = abs(_wrap(sq.ph - sp.ph)) <= self.theta and \
                (not self.has_det or abs(_wrap(sq.phF - sp.phF)) <= self.theta)
            if good:
                st = _OK
            elif q - p >= 2 and (q - p) * self.unit(o) > self.hmin:
                return None                                          # refine (not cached)
            else:
                st = _BAD
        self.seg[k] = st
        return st

    def sample_edges(self, edges):
        """Sample the edges (o, line, s0, s1) until every segment is resolved, suspect, invalid or unresolvable."""
        need = set()
        for o, line, s0, s1 in edges:
            need |= {self.key(o, line, s) for s in self._initial(o, line, s0, s1)}
        self._evaluate(need)
        active = list(edges)
        while active:
            need, still = set(), []
            for e in active:
                o, line, s0, s1 = e
                pos = self._positions(o, line, s0, s1)
                refine = False
                for p, q in zip(pos[:-1], pos[1:]):
                    if self.segment(o, line, p, q) is None:
                        if len(pos) >= self.max_edge:        # phase winding too fast (pole forest?): give up here
                            self.seg[(o, line, p, q)] = _SUSPECT
                            continue
                        need.add(self.key(o, line, (p + q) // 2))
                        refine = True
                if refine:
                    still.append(e)
            self._evaluate(need)
            active = still

    def densify(self, cells, nseg, dlog=1.0, rounds=4):
        """Refine the edges of the cells for the moments: at least nseg segments per edge, then up to `rounds`
        bisections of every segment over which log f changes by more than dlog (|Δ log|f|| and Δ arg f together:
        a zero or pole just outside the cell makes log f steep along its edge)."""
        while True:
            need = set()
            for cell in cells:
                for o, line, s0, s1 in cell.edges():
                    pos = self._positions(o, line, s0, s1)
                    if len(pos) - 1 >= nseg:
                        continue
                    for p, q in zip(pos[:-1], pos[1:]):
                        if (q - p) * nseg > (s1 - s0) and q - p >= 2:
                            need.add(self.key(o, line, (p + q) // 2))
            if not need:
                break
            self._evaluate(need)
        for _ in range(rounds):
            need = set()
            for cell in cells:
                for o, line, s0, s1 in cell.edges():
                    pos = self._positions(o, line, s0, s1)
                    for p, q in zip(pos[:-1], pos[1:]):
                        a, b = self.pts[self.key(o, line, p)], self.pts[self.key(o, line, q)]
                        if q - p >= 2 and a.ok and b.ok and not (a.suspect or b.suspect) and \
                                abs(complex(b.lf - a.lf, _wrap(b.ph - a.ph))) > dlog:
                            need.add(self.key(o, line, (p + q) // 2))
            if not need:
                break
            self._evaluate(need)
        self.sample_edges([e for c in cells for e in c.edges()])

    def edge_status(self, o, line, s0, s1):
        pos = self._positions(o, line, s0, s1)
        return [self.segment(o, line, p, q) for p, q in zip(pos[:-1], pos[1:])], pos


# ----------------------------------------------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------------------------------------------
class _Cell:
    def __init__(self, i0, i1, j0, j1, depth=0):
        self.i0, self.i1, self.j0, self.j1, self.depth = i0, i1, j0, j1, depth
        self.status = None
        self.N = self.Z = None
        self.suspect = False
        self.bad = []           # midpoints of unresolved segments
        self.fails = 0          # failed locations along this cell's ancestry

    def edges(self):
        return [(0, self.j0, self.i0, self.i1), (1, self.i1, self.j0, self.j1),
                (0, self.j1, self.i0, self.i1), (1, self.i0, self.j0, self.j1)]

    def rect(self, g):
        w0, w1 = g.w(self.i0, self.j0), g.w(self.i1, self.j1)
        return (w0.real, w1.real, w0.imag, w1.imag)

    def size(self, g):
        return max((self.i1 - self.i0) * g.sx, (self.j1 - self.j0) * g.sy)

    def center(self, g):
        return 0.5 * (g.w(self.i0, self.j0) + g.w(self.i1, self.j1))

    def contains(self, g, w, margin=0.0):
        r = self.rect(g)
        return r[0] - margin <= w.real <= r[1] + margin and r[2] - margin <= w.imag <= r[3] + margin


def _boundary(g, cell):
    """Counter-clockwise boundary of the cell: the four edges, each (samples, segment statuses)."""
    out = []
    for k, (o, line, s0, s1) in enumerate(cell.edges()):
        st, pos = g.edge_status(o, line, s0, s1)
        keys = [g.key(o, line, s) for s in pos]
        if k >= 2:                                                    # top: right → left, left: top → bottom
            keys, st = keys[::-1], st[::-1]
        out.append(([g.pts[x] for x in keys], st))
    return out


def _analyse(g, cell):
    """Status, windings N (of f) and Z (of F) of a sampled cell, and its boundary for the moments."""
    bnd = _boundary(g, cell)
    sts = [s for _, st in bnd for s in st]
    cell.suspect = _SUSPECT in sts or any(s.suspect for e, _ in bnd for s in e)
    cell.bad = []
    for e, st in bnd:
        for s, a, b in zip(st, e[:-1], e[1:]):
            if s == _BAD:
                cell.bad.append(0.5 * (a.w + b.w))
    if _INVALID in sts or any(not s.ok for e, _ in bnd for s in e):
        cell.status = "invalid"
        return None
    steps, stepsF = [], []
    for e, st in bnd:
        for a, b in zip(e[:-1], e[1:]):
            steps.append(_wrap(b.ph - a.ph))
            if g.has_det:
                stepsF.append(_wrap(b.phF - a.phF))
    if cell.bad:
        cell.status = "bad"
        return bnd
    amb = cell.suspect and (max(abs(x) for x in steps) > g.theta or (stepsF and max(abs(x) for x in stepsF) > g.theta))
    cell.N = None if amb else int(round(sum(steps) / (2 * np.pi)))
    cell.Z = None if (amb or not g.has_det) else int(round(sum(stepsF) / (2 * np.pi)))
    cell.status = "suspect" if cell.suspect else "ok"
    return bnd


def _moments(bnd, kmax, center, rho, use_F=False):
    """s_k, k = 0 … kmax, of u = (ω − center)/ρ from the boundary samples (cubic spline in arc length per edge),
    and an error estimate: the difference from the spline through every other sample /15 (/3 if an edge has fewer
    than 7 samples: its coarse rule is then the trapezoid)."""
    ws, Ls, idx = [], [], []
    acc = prev = None
    for e, _ in bnd:                         # each edge without its last sample (the first of the next edge)
        idx.append(len(ws))
        for s in e[:-1]:
            ph, la = (s.phF, s.lF) if use_F else (s.ph, s.lf)
            acc = ph if acc is None else acc + _wrap(ph - prev)
            prev = ph
            ws.append(s.w)
            Ls.append(complex(la, acc))
    s0 = bnd[0][0][0]                        # close: back at ω₀ with log f + 2πiN
    acc_end = acc + _wrap((s0.phF if use_F else s0.ph) - prev)
    N = int(round((acc_end - Ls[0].imag) / (2 * np.pi)))
    ws.append(s0.w)
    Ls.append(Ls[0] + 2j * np.pi * N)
    idx.append(len(ws) - 1)
    u = (np.array(ws) - center) / rho
    L = np.array(Ls)
    s = np.zeros(kmax + 1, complex)
    st = np.zeros(kmax + 1, complex)
    s[0] = st[0] = N
    order = 15.0
    for k in range(1, kmax + 1):
        g = u ** (k - 1) * L
        I = Ic = 0j
        for a, b in zip(idx[:-1], idx[1:]):
            if b > a:
                I += _edge_integral(u[a:b + 1], g[a:b + 1])
                sub = np.unique(np.r_[np.arange(a, b + 1, 2), b])           # every other sample
                Ic += _edge_integral(u[sub], g[sub])
                if len(sub) < 4:
                    order = 3.0                                             # coarse rule is the trapezoid
        s[k] = u[0] ** k * N - k / (2j * np.pi) * I
        st[k] = u[0] ** k * N - k / (2j * np.pi) * Ic
    return s, np.abs(s - st) / order


def _edge_integral(u, g):
    """∫ g du along a straight edge through the points u (not-a-knot cubic spline in arc length; trapezoid for
    fewer than 4 points)."""
    if len(u) < 4:
        return np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(u))
    t = np.abs(u - u[0])
    I = CubicSpline(t, g.real).integrate(0, t[-1]) + 1j * CubicSpline(t, g.imag).integrate(0, t[-1])
    return I * (u[-1] - u[0]) / t[-1]


def _compositions(n, k):
    """All k-tuples of positive integers summing to n."""
    if k == 1:
        return [(n,)]
    return [(i,) + rest for i in range(1, n - k + 2) for rest in _compositions(n - i, k - 1)]


def _roots_from_power_sums(p):
    """Roots of Π(u − ζ_i), i = 1 … n, from the power sums p_k = Σζ^k, k = 1 … n (Newton's identities)."""
    n = len(p) - 1
    e = np.zeros(n + 1, complex)
    e[0] = 1.0
    for k in range(1, n + 1):
        e[k] = sum((-1) ** (i - 1) * e[k - i] * p[i] for i in range(1, k + 1)) / k
    coef = [(-1) ** k * e[k] for k in range(n + 1)]
    return np.roots(coef) if n > 1 else np.array([e[1]])


# ----------------------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------------------
def count_zeros_poles(target, kx, wr_range, wi_range, *, n_init=16, theta=np.pi / 4, valid=None, use_det=False,
                      derivative=False, pole_free=None, poles=None, workers=1, executor=None, max_evals=None,
                      sample_options=None):
    """Argument principle on the rectangle wr_range × wi_range with adaptive boundary sampling.

    Returns a dict: N = Z − P (winding of f), Z (winding of the pole-free F if available, else None), P = Z − N,
    status ('ok', 'suspect' = flagged samples on the boundary, 'bad' = a zero/discontinuity on the boundary,
    'invalid'), nfev, samples (ω, f) of the boundary. Arguments as in find_all_roots."""
    T = _make_target(target, kx, derivative=derivative, pole_free=pole_free, poles=poles, use_det=use_det,
                     sample_options=sample_options, polish_options=None, suspect=None)
    if valid is None and not isinstance(target, _Target):
        valid = default_validity(target, kx)
    ev = _Evaluator(T, valid, workers, executor, max_evals)
    try:
        g = _Grid(ev, (wr_range, wi_range), n_init=n_init, theta=theta)
        cell = _Cell(0, 2 ** _K, 0, 2 ** _K)
        g.sample_edges(cell.edges())
        bnd = _analyse(g, cell)
    finally:
        ev.close()
    Z = cell.Z if T.has_det else (cell.N if not T.poles else None)
    P = None if Z is None or cell.N is None else Z - cell.N
    smp = [s for e, _ in bnd for s in e] if bnd else []
    return dict(N=cell.N, Z=Z, P=P, status=cell.status, nfev=ev.nfev, bad=cell.bad,
                samples=(np.array([s.w for s in smp]), np.array([s.f if s.ok else np.nan for s in smp])))


def find_all_roots(target, kx, wr_range, wi_range, *, tol=1e-10, ftol=None, n_init=16, theta=np.pi / 4,
                   n_moments=3, max_depth=14, max_depth_suspect=6, max_depth_invalid=6, max_cluster=8,
                   cluster_size=1 / 16, valid=None, use_det=False, derivative=False, pole_free=None, poles=None,
                   find_poles=True, cr_step=None, cr_tol=1e-3, workers=1, executor=None, max_evals=4000,
                   sample_options=None, polish_options=None, suspect=None, probe=False, dipole_tol=1e-2,
                   moment_segments=8, rescue_polish=2, max_rescue=12, check_empty=True, max_fails=4,
                   max_depth_poles=10, confirm=None, confirm_tol=1e-4, verbose=False):
    """All zeros of f(ω) = 𝒟(ω, k_x) in the rectangle wr_range × wi_range, without an initial guess.

    target : DispersionRelation, StreamingDispersion, BinormalDispersion, or a callable ω → complex (kx ignored;
        with derivative=True it returns (f, f′); pole_free: optional callable ω → F whose zeros are those of f and
        which has no poles; poles=False declares f analytic (no poles); suspect: optional predicate ω → bool
        flagging samples in a known non-analytic region).
    tol : polish tolerance |Δω| ≤ tol max(1, |ω|); ftol: |f| at an accepted root (default 1e-6 × the median |f|
        on the box boundary, at least 1e-8).
    n_init, theta : initial samples along the longer side of the box, maximum phase step between samples.
    n_moments : most zeros located from the moments of one cell (else the cell is bisected).
    max_depth, max_depth_suspect, max_depth_invalid : bisection depths (each halves the longer side) for ordinary,
        flagged and invalid cells. max_cluster, cluster_size : a cell smaller than cluster_size × the box holding more
        than max_cluster zeros + poles is reported as a cluster (discretised continuum) instead of resolved.
    valid : predicate ω → bool (domain of 𝒟); default: default_validity(target, kx).
    use_det : sample the pole-free F = 𝒟 Π det D as well (StreamingDispersion / BinormalDispersion): zeros and poles
        are then counted separately, but arg F (a sum over the y nodes' determinants) varies much faster than arg 𝒟:
        2× the samples on the circular model, far more next to a dense pole forest (CBC). Default False: N = Z − P,
        with the moment tests for zero–pole pairs.
    find_poles, max_depth_poles : locate the poles too (cells with P = 1; cells with P ≥ 2 and no zeros, or with
        N = 0 and a zero–pole pair, are bisected down to max_depth_poles). With a pole-free F the zeros are located
        regardless of the poles in their cell.
    cr_step, cr_tol : step of the Cauchy–Riemann test (default 1e-4 × the box diagonal) and the largest defect of an
        accepted root.
    workers / executor : thread pool size for batches of boundary samples and polishes (the λ-form solver is better
        parallelised by its own nthreads); or any object with .map (e.g. a ProcessPoolExecutor).
    max_evals : budget of boundary samples (the result is marked incomplete when it is exhausted).
    sample_options / polish_options : attribute overrides of the DispersionRelation clones used for sampling
        (default epsabs = epsrel = 1e-6, epsabs_y = epsrel_y = 1e-7, tolN = 1e-8, limit = 32 panels: the phase needs
        no more, and the limit flags the ISM region cheaply) and for polishing (default limit = 64; the user's object
        is not modified).
    probe : test the centre of every empty cell for analyticity (4 evaluations each).
    dipole_tol : (poles possible, no F) moment error, in units of the cell half-diagonal, below which a cell is
        trusted not to hide a zero–pole pair. moment_segments : segments per edge before the moments are taken.
    rescue_polish, max_rescue : polishes started from each terminal suspect cell (its smallest unflagged |f|
        samples), at most max_rescue in all (smallest |f| first, 30 iterations each).
    check_empty : require s₁ … s₃ ≈ 0 in cells without zeros, else bisect them: catches a zero whose winding is
        cancelled by a non-analytic region inside the cell (or by a pole), at ~16 evaluations per empty cell.
    max_fails : failed locations along a cell's ancestry before it is reported instead of bisected.
    confirm, confirm_tol : a second solver (e.g. more y nodes or a finer velocity grid; a dispersion object or a
        callable): every root is re-polished with it, its shift |Δω| stored in Root.shift, and roots moving by more
        than confirm_tol·max(1, |ω|) are rejected (zeros between the poles of a discrete y-rule move with n_y).

    Returns a RootSearch (roots, poles, rejected, suspect_regions, excluded, nfev, ...; .summary())."""
    t_start = time.time()
    T = _make_target(target, kx, derivative=derivative, pole_free=pole_free, poles=poles, use_det=use_det,
                     sample_options=sample_options, polish_options=polish_options, suspect=suspect)
    if valid is None and not isinstance(target, _Target):
        valid = default_validity(target, kx)
    ev = _Evaluator(T, valid, workers, executor, max_evals)
    (a, b), (c, d) = wr_range, wi_range
    diag = math.hypot(b - a, d - c)
    cr_step = 1e-4 * diag if cr_step is None else cr_step
    margin = 1e-9 * diag
    polish_region = (0.5 * complex(a + b, c + d), 1.5 * diag)
    g = _Grid(ev, (wr_range, wi_range), n_init=n_init, theta=theta)
    root = _Cell(0, 2 ** _K, 0, 2 ** _K)
    found, rejected, poles_found, regions, excluded, terminal, rescue, pole_est = [], [], [], [], [], [], [], []
    state = dict(complete=True, ftol=ftol)
    say = print if verbose else (lambda *x, **k: None)

    def log(cell, what, **kw):
        terminal.append(dict(rect=cell.rect(g), depth=cell.depth, status=cell.status, N=cell.N, Z=cell.Z,
                             outcome=what, **kw))

    def zeros_of(cell):
        if T.has_det:
            return cell.Z
        return cell.N if not T.poles else None

    def split(cells):
        """Bisect each cell across its longer side; the split line is shifted if it meets a zero."""
        out = []
        pend = [(cell, 0) for cell in cells]
        while pend:
            lines, plan = [], []
            for cell, k in pend:
                vertical = (cell.i1 - cell.i0) * g.sx >= (cell.j1 - cell.j0) * g.sy
                n = (cell.i1 - cell.i0) if vertical else (cell.j1 - cell.j0)
                fr = _SPLIT_FRACTIONS[k] if n % 16 == 0 else 0.5
                m = (cell.i0 if vertical else cell.j0) + int(round(fr * n))
                e = (1, m, cell.j0, cell.j1) if vertical else (0, m, cell.i0, cell.i1)
                lines.append(e)
                plan.append((cell, k, vertical, m, e))
            g.sample_edges(lines)
            pend = []
            for cell, k, vertical, m, e in plan:
                st, _ = g.edge_status(*e)
                if _BAD in st and k + 1 < len(_SPLIT_FRACTIONS):
                    pend.append((cell, k + 1))
                    continue
                if vertical:
                    kids = [_Cell(cell.i0, m, cell.j0, cell.j1, cell.depth + 1),
                            _Cell(m, cell.i1, cell.j0, cell.j1, cell.depth + 1)]
                else:
                    kids = [_Cell(cell.i0, cell.i1, cell.j0, m, cell.depth + 1),
                            _Cell(cell.i0, cell.i1, m, cell.j1, cell.depth + 1)]
                for kid in kids:
                    kid.fails = cell.fails
                out += kids
        g.sample_edges([e for kid in out for e in kid.edges()])
        return out

    def polish_task(args):
        """Polish the estimates of one cell. Returns list of (estimate, RootResult, nfev, flagged)."""
        cell, guesses, pole, scale = args[:4]
        maxit = args[4] if len(args) > 4 else 60
        cf = _Counted(ev, abort=True, region=polish_region)
        cc = _Counted(ev, abort=True, coarse=True, region=polish_region) if T.has_coarse else None
        res = []
        for w0 in guesses:
            cf.flagged = False
            n0 = cf.n + (cc.n if cc else 0)
            try:
                if pole:
                    r = find_root(lambda w: _recip(cf.f(w)), w0, tol=tol, maxiter=maxit, step=1e-2 * scale)
                elif T.has_derivative:
                    if cc is not None:          # stage 1 at the sampling settings: flagged evaluations are cheaper
                        cc.flagged = False
                        r1 = find_root(cc.fd, w0, tol=max(tol, 1e-6), maxiter=maxit, step=1e-2 * scale,
                                       derivative=True)
                        if r1.converged:
                            w0 = r1.omega
                    r = find_root(cf.fd, w0, tol=tol, maxiter=maxit, step=1e-2 * scale, derivative=True)
                else:
                    r = find_root(cf.f, w0, tol=tol, maxiter=maxit, step=1e-2 * scale)
            except _Flagged:
                r = RootResult(complex(w0), complex(np.nan), False, 0, "flagged evaluation")
            nf = cf.n + (cc.n if cc else 0) - n0
            res.append((w0, r, nf, cf.flagged or (cc.flagged if cc else False)))
        return res

    def moment_check(s, err, u, m):
        """max_k |s_k − Σ m_i u_i^k| / tol_k, k = 1 … kmax (u: positions of the located zeros in cell units)."""
        k = np.arange(1, len(s))
        tolk = 5 * err[1:] + T.moment_floor
        ps = (np.asarray(m, float)[:, None] * np.asarray(u)[:, None] ** k[None, :]).sum(axis=0) if len(u) else 0 * k
        return float(np.max(np.abs(ps - s[1:]) / tolk))

    def resolved(err):
        """Without a pole-free F, a zero–pole pair closer than dipole_tol (cell units) is invisible to the moments."""
        return T.has_det or not T.poles or float(np.max(err[1:])) <= dipole_tol

    def try_locate(batch):
        """batch: list of (cell, nzeros, boundary). Estimates from the moments, polish, accept; the multiplicities
        are those whose power sums Σ m_i u_i^k match s_k (k ≤ max(Z, 3): also tests for zero–pole pairs and
        non-analyticity). Returns the cells (with partial finds) to bisect."""
        tasks, meta = [], []
        for cell, nz, bnd in batch:
            ctr, rho = cell.center(g), 0.5 * math.hypot((cell.i1 - cell.i0) * g.sx, (cell.j1 - cell.j0) * g.sy)
            s, err = _moments(bnd, max(nz, 3), ctr, rho, use_F=T.has_det)
            try:
                est = ctr + rho * _roots_from_power_sums(s[:nz + 1])
            except Exception:                                         # noqa: BLE001
                est = np.array([ctr] * nz)
            est = [e if np.isfinite(e) else ctr for e in est]
            tasks.append((cell, est, False, cell.size(g)))
            meta.append((cell, nz, s, err, rho, ctr))
        results = ev.map(polish_task, tasks)
        again = []
        for (cell, nz, s, err, rho, ctr), res in zip(meta, results):
            u = []
            ev.nfev_polish += sum(x[2] for x in res)
            good, flagged = [], False
            for w0, r, nf, fl in res:
                flagged |= fl
                if fl:
                    continue
                if r.converged and np.isfinite(r.D) and abs(r.D) <= state["ftol"] and cell.contains(g, r.omega, margin):
                    if all(abs(r.omega - q.omega) > 1e3 * tol * max(1.0, abs(r.omega)) for q in good):
                        good.append(Root(complex(r.omega), complex(r.D), 1, cell=cell.rect(g), nfev=nf))
            if flagged and not good:
                cell.status = "suspect"
            consistent = False
            if good and len(good) <= nz and resolved(err):
                u = [(q.omega - ctr) / rho for q in good]
                dev, m = min((moment_check(s, err, u, m), m) for m in _compositions(nz, len(good)))
                if dev <= 1.0:
                    consistent = True
                    for q, mi in zip(good, m):
                        q.multiplicity = int(mi)
            if consistent:
                for q in good:
                    found.append(q)
                if T.has_det and find_poles and cell.Z is not None and cell.N is not None and cell.Z - cell.N == 1:
                    sf, _ = _moments(_analyse(g, cell), 1, ctr, rho)          # f: s₁ = Σ m ζ − p
                    pole_est.append((cell, ctr + rho * (sum(q.multiplicity * u_ for q, u_ in zip(good, u)) - sf[1])))
                if cell.status == "suspect":
                    regions.append(Region(cell.rect(g), "flagged samples; winding and moments matched by the located "
                                          "root(s)", cell.N, cell.Z))
                log(cell, "located %d" % nz)
                say("cell %s: %d root(s) located" % (np.round(cell.rect(g), 4), nz))
            elif flagged and cell.depth >= max_depth_suspect:
                regions.append(Region(cell.rect(g), "non-analytic: polish met flagged evaluations", cell.N, cell.Z))
                rescue.append(cell)
                log(cell, "suspect (polish flagged)")
            else:
                again.append((cell, good))
        return again

    def probe_defect(cell):
        """CR defect |∂f/∂ω̄|/|∂f/∂ω| at the cell centre (four evaluations)."""
        cf = _Counted(ev)
        w, h = cell.center(g), min(cr_step, 1e-2 * cell.size(g))
        fx = (cf.f(w + h) - cf.f(w - h)) / (2 * h)
        fy = (cf.f(w + 1j * h) - cf.f(w - 1j * h)) / (2 * h)
        ev.nfev_polish += cf.n
        return abs(0.5 * (fx + 1j * fy)) / max(abs(0.5 * (fx - 1j * fy)), 1e-300), cf.flagged

    def empty(cell, what="empty"):
        """A cell without zeros; with probe=True its centre is tested for analyticity first."""
        if probe:
            dfc, fl = probe_defect(cell)
            if fl or not dfc <= cr_tol:
                cell.status = "suspect"
                if cell.depth < max_depth_suspect:
                    return cell                                       # → bisect
                regions.append(Region(cell.rect(g), "non-analytic (CR defect %.2g at the centre)" % dfc, cell.N,
                                      cell.Z))
                log(cell, "non-analytic (probe)")
                return None
        log(cell, what)
        return None

    pending = []
    try:
        g.sample_edges(root.edges())
        active = [root]
        first = True
        while active:
            to_split, locate, pole_cells, dipole_cells = [], [], [], []
            for cell in active:
                bnd = _analyse(g, cell)
                if first:
                    first = False
                    root.N, root.Z = cell.N, cell.Z
                    if state["ftol"] is None:
                        mags = [abs(s.f) for e, _ in (bnd or []) for s in e if s.ok]
                        state["ftol"] = max(1e-6 * float(np.median(mags)) if mags else 1e-8, 1e-8)
                    say("box: status %s, N = %s, Z = %s, %d evaluations" % (cell.status, cell.N, cell.Z, ev.nfev))
                size_ok = cell.depth < max_depth
                if cell.status == "invalid":
                    if cell.depth < max_depth_invalid:
                        to_split.append(cell)
                    else:
                        excluded.append(Region(cell.rect(g), "invalid samples on the boundary", cell.N, cell.Z))
                        log(cell, "excluded")
                    continue
                if cell.status == "bad":
                    if size_ok and cell.depth < max_depth_suspect + 4:
                        to_split.append(cell)
                    else:
                        regions.append(Region(cell.rect(g), "zero or discontinuity on the boundary", None, None,
                                              "at ω ≈ %s" % ", ".join("%.6g%+.6gi" % (x.real, x.imag) for x in cell.bad[:3])))
                        pending.append((cell, list(cell.bad[:3])))
                        log(cell, "bad")
                    continue
                nz = zeros_of(cell)
                if cell.status == "suspect":                  # bisected regardless of N: the non-analytic part's
                    if cell.N is not None and nz is not None and 1 <= nz <= n_moments:     # winding is arbitrary
                        locate.append((cell, nz, bnd))               # try: a root near the region may be analytic
                        continue
                    if cell.depth < max_depth_suspect:
                        to_split.append(cell)
                    else:
                        regions.append(Region(cell.rect(g), "flagged samples (non-analytic?)", cell.N, cell.Z))
                        rescue.append(cell)
                        log(cell, "suspect")
                    continue
                # status ok: N (and Z) known
                if not T.poles and cell.N < 0:
                    regions.append(Region(cell.rect(g), "negative winding of a pole-free function: non-analytic",
                                          cell.N, cell.Z))
                    rescue.append(cell)
                    log(cell, "non-analytic (N < 0)")
                    continue
                P = (cell.Z - cell.N) if T.has_det else (0 if not T.poles else None)
                big = cell.size(g) > cluster_size * max(g.W, g.H)
                if nz is not None:
                    if (nz + (P or 0)) > max_cluster and not big:
                        regions.append(Region(cell.rect(g), "cluster of %d zeros and %s poles (discretised continuum?)"
                                              % (nz, P), cell.N, cell.Z))
                        log(cell, "cluster")
                        continue
                    if nz == 0:
                        if P and find_poles:
                            if P == 1:
                                pole_cells.append((cell, bnd))
                            elif cell.depth < max_depth_poles:
                                to_split.append(cell)
                            else:
                                log(cell, "%d poles, not isolated" % P)
                            continue
                        if P:
                            log(cell, "%d pole(s), not located" % P)
                        elif check_empty:
                            dipole_cells.append(cell)               # moments must vanish too
                        elif empty(cell) is not None:
                            to_split.append(cell)
                        continue
                    if nz <= n_moments:                     # (the moments of F ignore the poles)
                        locate.append((cell, nz, bnd))
                    elif size_ok:
                        to_split.append(cell)
                    else:
                        regions.append(Region(cell.rect(g), "%d zeros unresolved at max_depth" % nz, cell.N, cell.Z))
                        log(cell, "unresolved")
                    continue
                # poles possible, only N = Z − P known
                if cell.N == 0:
                    dipole_cells.append(cell)
                elif cell.N == -1 and find_poles:
                    pole_cells.append((cell, bnd))
                elif 1 <= cell.N <= n_moments:
                    locate.append((cell, cell.N, bnd))
                elif size_ok:
                    to_split.append(cell)
                else:
                    regions.append(Region(cell.rect(g), "N = %d unresolved at max_depth" % cell.N, cell.N, cell.Z))
                    log(cell, "unresolved")
            # the moments need ≥ moment_segments segments per edge
            if locate or pole_cells or dipole_cells:
                g.densify([x[0] for x in locate] + [x[0] for x in pole_cells] + dipole_cells, moment_segments)
                locate = [(cell, nz, _analyse(g, cell)) for cell, nz, _ in locate]
                pole_cells = [(cell, _analyse(g, cell)) for cell, _ in pole_cells]
            for cell in dipole_cells:          # Z = 0 (or N = 0, poles possible): s₁ … s₃ must vanish as well,
                bnd = _analyse(g, cell)        # else a zero–pole pair or a zero cancelled by a non-analytic region
                rho = 0.5 * math.hypot((cell.i1 - cell.i0) * g.sx, (cell.j1 - cell.j0) * g.sy)
                s, err = _moments(bnd, 3, cell.center(g), rho)
                if moment_check(s, err, [], []) > 1.0 or not resolved(err):
                    if not T.poles or T.has_det:          # no poles inside: f is not analytic in the cell
                        cell.status = "suspect"
                    else:                                 # a zero–pole pair, unless the centre is non-analytic
                        dfc, fl = probe_defect(cell)
                        if fl or not dfc <= cr_tol:
                            cell.status = "suspect"
                    lim = max_depth_suspect if cell.status == "suspect" else max_depth_poles
                    if cell.depth < lim:
                        to_split.append(cell)
                    else:
                        regions.append(Region(cell.rect(g), "N = 0 but moments s₁…s₃ ≠ 0 (%s)" % (
                            "non-analytic?" if cell.status == "suspect" else "zero–pole pair?"), cell.N, cell.Z))
                        rescue.append(cell)
                        log(cell, "moments ≠ 0")
                elif empty(cell) is not None:
                    to_split.append(cell)
            # locate zeros
            if locate:
                for cell, good in try_locate(locate):
                    cell.fails += 1
                    lim = max_depth_suspect if cell.status == "suspect" else max_depth
                    if cell.depth < lim and cell.fails < max_fails:
                        to_split.append(cell)
                    else:
                        for q in good:
                            q.note = "cell count not matched"
                            found.append(q)
                        regions.append(Region(cell.rect(g), "zeros not all located (%d found, %d failed attempts)"
                                              % (len(good), cell.fails), cell.N, cell.Z))
                        rescue.append(cell)
                        log(cell, "partially located")
            # poles: P = 1 cells with zeros (estimate from the moments of f) …
            if pole_est:
                tasks = [(cell, [w], True, cell.size(g)) for cell, w in pole_est]
                for (cell, w0), res in zip(pole_est, ev.map(polish_task, tasks)):
                    _, r, nf, fl = res[0]
                    ev.nfev_polish += nf
                    ok = r.converged and cell.contains(g, r.omega, margin)
                    poles_found.append(Pole(complex(r.omega if ok else w0), None, 1, cell.rect(g), ok,
                                            "" if ok else "moment estimate only"))
                pole_est.clear()
            # … and cells with one pole and no zeros
            if pole_cells:
                tasks, mom = [], []
                for cell, bnd in pole_cells:
                    ctr = cell.center(g)
                    rho = 0.5 * math.hypot((cell.i1 - cell.i0) * g.sx, (cell.j1 - cell.j0) * g.sy)
                    s, err = _moments(bnd, 3, ctr, rho)       # f: s_k = −p^k (one pole, no zeros)
                    tasks.append((cell, [ctr - rho * s[1] if np.isfinite(s[1]) else ctr], True, cell.size(g)))
                    mom.append((s, err, ctr, rho))
                for (cell, _), (s, err, ctr, rho), res in zip(pole_cells, mom, ev.map(polish_task, tasks)):
                    w0, r, nf, fl = res[0]
                    ev.nfev_polish += nf
                    ok = r.converged and cell.contains(g, r.omega, margin)
                    if ok and not T.has_det:                  # N = −1 could hide zero–pole pairs
                        ok = resolved(err) and moment_check(s, err, [(r.omega - ctr) / rho], [-1]) <= 1.0
                        if not ok and cell.depth < max_depth:
                            to_split.append(cell)
                            continue
                    if ok:
                        poles_found.append(Pole(complex(r.omega), None, 1, cell.rect(g)))
                        log(cell, "pole")
                    else:
                        poles_found.append(Pole(complex(w0), None, 1, cell.rect(g), False, "moment estimate only"))
                        log(cell, "pole (estimate)")
            active = split(to_split) if to_split else []
            if verbose and active:
                say("depth %d: %d cells, %d evaluations, %.1f s" % (max(c.depth for c in active), len(active), ev.nfev,
                                                                     time.time() - t_start))
    except _Budget:
        state["complete"] = False

    # roots next to non-analytic regions: polish from the smallest |f| among the unflagged boundary samples of
    # every terminal suspect cell (a polish that meets a flagged evaluation is abandoned)
    if rescue_polish and rescue:
        starts = []
        for cell in rescue:
            smp = sorted((x for e, _ in _boundary(g, cell) for x in e if x.ok and not x.suspect), key=lambda x: abs(x.f))
            starts += [(abs(x.f), id(cell), cell, x.w) for x in smp[:rescue_polish]]
        starts = sorted(starts, key=lambda t: t[0])[:max_rescue]           # smallest |f| first
        tasks = [(cell, [w], False, cell.size(g), 30) for _, _, cell, w in starts]
        for (cell, _, _, _, _), res in zip(tasks, ev.map(polish_task, tasks)):
            for w0, r, nf, fl in res:
                ev.nfev_polish += nf
                if r.converged and not fl and abs(r.D) <= state["ftol"] and \
                        a - margin <= r.omega.real <= b + margin and c - margin <= r.omega.imag <= d + margin:
                    found.append(Root(complex(r.omega), complex(r.D), 1, cell=cell.rect(g), nfev=nf,
                                      note="next to a suspect region (rescue polish)"))

    # zeros on cell boundaries that were never isolated
    for cell, pts in pending:
        res = polish_task((cell, pts, False, cell.size(g)))
        for w0, r, nf, fl in res:
            ev.nfev_polish += nf
            if r.converged and abs(r.D) <= (state["ftol"] or 1e-8) and not fl and \
                    a - margin <= r.omega.real <= b + margin and c - margin <= r.omega.imag <= d + margin:
                found.append(Root(complex(r.omega), complex(r.D), 1, cell=cell.rect(g), nfev=nf, note="on a cell edge"))

    # dedupe, verify
    roots = []
    for q in sorted(found, key=lambda q: -q.omega.imag):
        if all(abs(q.omega - p.omega) > 1e3 * tol * max(1.0, abs(q.omega)) for p in roots):
            roots.append(q)
    verified = []
    cf = _Counted(ev)
    for q in roots:
        n0 = cf.n
        others = [abs(q.omega - p.omega) for p in roots if p is not q] + [abs(q.omega - p.omega) for p in poles_found]
        w, h = q.omega, min([cr_step] + [1e-2 * x for x in others])      # far below the distance to other roots
        cf.flagged = False
        if q.multiplicity > 1:                       # f′ = 0: no CR test, no residue
            q.note = (q.note + "; " if q.note else "") + "multiple root (CR test skipped)"
            verified.append(q)
            continue
        try:
            fx = (cf.f(w + h) - cf.f(w - h)) / (2 * h)
            fy = (cf.f(w + 1j * h) - cf.f(w - 1j * h)) / (2 * h)
            dw, dwb = 0.5 * (fx - 1j * fy), 0.5 * (fx + 1j * fy)
            q.cr_defect = abs(dwb) / max(abs(dw), 1e-300)
            if T.has_derivative:
                v, dexact = cf.fd(w)
                q.cr_defect = max(q.cr_defect, abs(dexact - dw) / max(abs(dw), 1e-300))
                dw = dexact
            q.residue = 1.0 / dw if q.multiplicity == 1 else None
        except Exception as e:                                         # noqa: BLE001
            q.note = (q.note + "; " if q.note else "") + "verification failed: %s" % e
        q.nfev += cf.n - n0
        if cf.flagged:
            q.note = (q.note + "; " if q.note else "") + "flagged evaluation near the root (non-analytic region)"
            rejected.append(q)
        elif not (q.cr_defect <= cr_tol):
            q.note = (q.note + "; " if q.note else "") + "Cauchy–Riemann defect %.2g > %.0e (non-analytic)" % (
                q.cr_defect, cr_tol)
            rejected.append(q)
        else:
            verified.append(q)
    ev.nfev_polish += cf.n
    # re-polish with a second (e.g. finer) solver: discretisation artefacts move
    if confirm is not None and verified:
        T2 = _make_target(confirm, kx, derivative=derivative, pole_free=None, poles=poles, use_det=False,
                          sample_options=sample_options, polish_options=polish_options, suspect=None)
        ev2 = _Evaluator(T2)
        keep = []
        for q in verified:
            c2 = _Counted(ev2, region=polish_region)
            try:
                if T2.has_derivative:
                    r2 = find_root(c2.fd, q.omega, tol=tol, maxiter=30, derivative=True)
                else:
                    r2 = find_root(c2.f, q.omega, tol=tol, maxiter=30, step=min(cr_step, 1e-3 * diag))
                q.shift = abs(r2.omega - q.omega) if r2.converged else float("inf")
            except _Flagged:
                q.shift = float("inf")
            ev.nfev_polish += c2.n
            if q.shift <= confirm_tol * max(1.0, abs(q.omega)):
                keep.append(q)
            else:
                q.note = (q.note + "; " if q.note else "") + "moves by %.2g with the confirm solver" % q.shift
                rejected.append(q)
        verified = keep
    # pole residues Res f = 1/(1/f)′
    for p in poles_found:
        if p.converged:
            h = cr_step
            dg = (_recip(cf.f(p.omega + h)) - _recip(cf.f(p.omega - h))) / (2 * h)
            p.residue = 1.0 / dg if dg != 0 else None
            ev.nfev_polish += 2
    ev.close()
    smp = [s for s in g.pts.values()]
    return RootSearch(roots=verified, poles=poles_found, rejected=rejected, suspect_regions=regions,
                      excluded=excluded, N=root.N, Z=(root.Z if T.has_det else (root.N if not T.poles else None)),
                      nfev=ev.nfev + ev.nfev_polish, nfev_sampling=ev.nfev, nfev_polish=ev.nfev_polish,
                      seconds=time.time() - t_start, cells=terminal,
                      samples=(np.array([s.w for s in smp]), np.array([s.f if s.ok else np.nan for s in smp]),
                               np.array([s.suspect for s in smp])),
                      complete=state["complete"])


def local_ism_roots(disp, kx, *, nz=33, ny=64, starts=None, gamma_min=1e-6, tol=1e-9):
    """Local roots ω(z, y) of D_ISM(z, y, ω) = 0 (DispersionRelation) on an nz × ny grid of the surface: the ISM
    region, where 𝒟 is not analytic, is their union {ω : D_ISM(z, y, ω) = 0 for some real (z, y)}.

    Muller from `starts` (default a fan scaled by A = k_x Φ_P max T_P) at every 8th y, continued in y in between.
    Returns rows (z, y, Re ω, Im ω) with Im ω > gamma_min (≈ 5 s for 33 × 64 on one thread)."""
    d = _clone_with(disp, dict(warn=False))
    g = d.geometry
    zs = np.linspace(g.zmin, g.zmax, nz)
    A = float(kx) * d.primary.Phi_P * float(np.max(np.abs(d.primary(np.linspace(g.zmin, g.zmax, 513)))))
    if starts is None:
        starts = A * np.array([0.1 + 0.64j, 0.64 + 0.64j, -0.64 + 0.64j, 1.28 + 0.64j, 1.28j, 0.43 + 0.1j,
                               -0.43 + 0.1j, 0.96 + 0.32j, -0.96 + 0.32j])
    bound = 10 * max(A, 1e-3)

    def Dl(z, y):
        def f(w):
            if not abs(w) < bound:                      # keep Muller away from extreme arguments
                return complex(np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return complex(d.D_ism(w, kx, z, y))
        return f

    out = []
    for z in zs:
        prev = []
        for k, y in enumerate(np.linspace(-np.pi, np.pi, ny, endpoint=False)):
            found = []
            for w0 in list(prev) + (list(starts) if k % 8 == 0 else []):
                r = find_root(Dl(z, y), w0, tol=tol, maxiter=30)
                if r.converged and abs(r.D) < 1e-8 and r.omega.imag > gamma_min and \
                        all(abs(r.omega - q) > 1e-6 for q in found):
                    found.append(r.omega)
            prev = found
            out += [(z, y, w.real, w.imag) for w in found]
    return np.array(out).reshape(-1, 4)
