"""Diagnostics for roots near marginality and near the ISM region (any solver: DispersionRelation,
StreamingDispersion, BinormalDispersion).

Near the ISM region (local zeros of D_ISM on the flux surface) and near marginality, roots are hard to follow:
𝒟 varies rapidly, the TSM pole can lose its weight to the drift-resonance continuum, and root finders hop between
branches. The tools here follow the practice of the streaming study vendored in extern/streaming_ref (report §7):

* root_map             coarse |𝒟| map → local minima → Muller polish: roots in a rectangle (no completeness
                       guarantee; tsm_disp.contour_roots.find_all_roots has one, and flags the ISM region);
* residue              Res = 1/𝒟′(ω₀), the weight of the pole (Φ ∝ Res/(ω − ω₀) for a generic source); a pole that
                       dissolves into the continuum loses its residue (factor 16 in the streaming study) while
                       |𝒟| stays small;
* spectral_response    R(ω_r) = 1/|𝒟(ω_r + iε)|, the zonal response to a real-frequency source; needs no analytic
                       continuation, so it is well defined inside the ISM region; a pole at ω₀ gives a Lorentzian
                       of half-width ε + |γ₀|;
* winding_number       roots minus poles of 𝒟 inside a rectangle (argument principle, adaptive sampling; raises
                       if the boundary crosses the ISM region); for BinormalDispersion the pole-free F = 𝒟 det D
                       counts roots only;
* scan_amplitude       continuation in the primary amplitude Φ_P with jump rejection; tracking UPWARD from Φ_P = 0
                       is much better conditioned than downward (branches leave Φ_P = 0 quadratically: 𝒟 is even in
                       Φ_P). Keep the velocity contour FIXED along a track (StreamingDispersion grid).
"""
import numpy as np
from .roots import find_root


def _Dfun(disp, kx):
    return lambda w: disp.D(w, kx)


def root_map(disp, kx, wr_range, wi_range, *, nr=31, ni=16, tol=1e-9, accept=1e-6, dedupe=1e-4, maxiter=40):
    """All roots of 𝒟(·, k_x) found from the local minima of |𝒟| on an nr × ni grid over wr_range × wi_range.

    Returns (roots, (WR, WI, absD)); roots: list of (ω, |𝒟(ω)|) sorted by growth rate, |𝒟| ≤ accept."""
    D = _Dfun(disp, kx)
    WR = np.linspace(*wr_range, nr)
    WI = np.linspace(*wi_range, ni)
    A = np.array([[abs(D(complex(x, y))) for x in WR] for y in WI])
    roots = []
    for i in range(ni):
        for j in range(nr):
            nb = A[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            if A[i, j] > nb.min():
                continue
            r = find_root(D, complex(WR[j], WI[i]), tol=tol, maxiter=maxiter)
            if not r.converged or abs(r.D) > accept:
                continue
            if not (WR[0] <= r.omega.real <= WR[-1] and WI[0] <= r.omega.imag <= WI[-1]):
                continue                                  # polished out of the box
            if all(abs(r.omega - q[0]) > dedupe * max(1.0, abs(r.omega)) for q in roots):
                roots.append((r.omega, abs(r.D)))
    roots.sort(key=lambda q: -q[0].imag)
    return roots, (WR, WI, A)


def residue(disp, omega, kx, h=None):
    """Res = 1/𝒟′(ω) at a root: exact derivative where the solver provides it (DispersionRelation, λ form),
    otherwise a fourth-order central difference with step h (default 1e-3 max(|ω|, 0.1))."""
    try:
        _, d = disp.D(omega, kx, derivative=True)
        return 1.0 / d
    except TypeError:
        pass
    h = h or 1e-3 * max(abs(omega), 0.1)
    f = lambda w: disp.D(w, kx)
    d = (-f(omega + 2 * h) + 8 * f(omega + h) - 8 * f(omega - h) + f(omega - 2 * h)) / (12 * h)
    return 1.0 / d


def spectral_response(disp, kx, omega_r, eps):
    """R(ω_r) = 1/|𝒟(ω_r + iε)| on an array of real frequencies (ε > 0: no analytic continuation needed)."""
    return np.array([1.0 / abs(disp.D(complex(w, eps), kx)) for w in np.atleast_1d(omega_r)])


def winding_number(disp, kx, wr_range, wi_range, n=40, use_det=None):
    """(number of zeros − number of poles) of 𝒟 inside the rectangle, by the argument principle with adaptive boundary
    sampling (tsm_disp.contour_roots.count_zeros_poles; n sets the initial samples per side, 2·max(4, n/2) along the
    longer one). For BinormalDispersion (use_det default True) the winding of F = 𝒟 det D is returned instead: F has no
    poles where det D = 0 (local ISM zeros), so it counts the roots of 𝒟 only. Raises if a zero or discontinuity lies
    on the boundary; warns if boundary samples are flagged (𝒟 not analytic there, e.g. the ISM region of the λ form).
    For all roots in a box use tsm_disp.contour_roots.find_all_roots."""
    import warnings
    from .contour_roots import count_zeros_poles
    if use_det is None:
        use_det = hasattr(disp, "Dmatrix") and "with_det" in disp.D.__code__.co_varnames
    c = count_zeros_poles(disp, kx, wr_range, wi_range, n_init=max(4, n // 2), use_det=use_det)
    if c["status"] == "bad":
        raise ValueError("zero or discontinuity on the contour near %s" % c["bad"][:3])
    out = c["Z"] if use_det else c["N"]
    if out is None:
        raise ValueError("winding_number: the boundary crosses a region where 𝒟 is not analytic (status %s; the ISM "
                         "region?): no zero count" % c["status"])
    if c["status"] != "ok":
        warnings.warn("winding_number: boundary %s (flagged samples: 𝒟 not analytic there?)" % c["status"])
    return out


def scan_amplitude(make_disp, amplitudes, omega0, kx, *, tol=1e-9, max_jump=0.15, accept=1e-6, maxiter=40):
    """Follow a root of 𝒟(ω; Φ_P) through the sequence `amplitudes` (e.g. upward from 0) by continuation.

    make_disp(Φ_P) → a dispersion object (with D(ω, kx), and find_root(ω0, kx) if available). The guess is the
    quadratic extrapolation of the last three roots; a step is rejected (scan stopped) if it does not converge, if
    |𝒟| > accept or if the root jumps by more than max_jump·max(0.15, |guess|). Returns a list of
    (Φ_P, ω, residue)."""
    out = []
    pts = []
    for amp in amplitudes:
        d = make_disp(amp)
        if len(pts) >= 3:
            ph, ws = zip(*pts[-3:])
            g = complex(np.polyval(np.polyfit(ph, ws, 2), amp))
        elif pts:
            g = pts[-1][1]
        else:
            g = complex(omega0)
        r = d.find_root(g, kx, tol=tol, maxiter=maxiter) if hasattr(d, "find_root") else \
            find_root(lambda w: d.D(w, kx), g, tol=tol, maxiter=maxiter)
        if not r.converged or abs(r.D) > accept or abs(r.omega - g) > max_jump * max(0.15, abs(g)):
            break
        pts.append((amp, r.omega))
        out.append((amp, r.omega, residue(d, r.omega, kx)))
    return out
