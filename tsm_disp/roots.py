"""Root finding and continuation scans for D(ω, k_x) = 0."""
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class RootResult:
    omega: complex
    D: complex
    converged: bool
    nfev: int
    message: str = ""
    seconds: float = 0.0
    history: list = field(default_factory=list)


def find_root(D, omega0, *, omega1=None, omega2=None, tol=1e-9, ftol=None, maxiter=60, step=None):
    """Root of an analytic function D(ω) by Muller's method (complex, order ≈ 1.84).

    D : callable ω → complex (e.g. lambda w: disp.D(w, kx))
    omega0 : initial guess; omega1, omega2 optional further points (default ω0 ± step).
    Converged when |Δω| ≤ tol·max(1, |ω|) (and |D| ≤ ftol if given).
    """
    t0 = time.time()
    if step is None:
        step = 1e-3 * max(abs(omega0), 1e-2)
    x = [complex(omega0), complex(omega1) if omega1 is not None else omega0 + step,
         complex(omega2) if omega2 is not None else omega0 + 1j * step]
    f = [D(xi) for xi in x]
    nfev = 3
    hist = list(zip(x, f))
    for it in range(maxiter):
        x0, x1, x2 = x
        f0, f1, f2 = f
        h1, h2 = x1 - x0, x2 - x1
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2
        disc = np.sqrt(b * b - 4 * a * c + 0j)
        den = b + disc if abs(b + disc) >= abs(b - disc) else b - disc
        dx = -2 * c / den if den != 0 else 1e-3 * (1 + abs(x2))
        x3 = x2 + dx
        f3 = D(x3)
        nfev += 1
        hist.append((x3, f3))
        x = [x1, x2, x3]
        f = [f1, f2, f3]
        if abs(dx) <= tol * max(1.0, abs(x3)) and (ftol is None or abs(f3) <= ftol):
            return RootResult(x3, f3, True, nfev, "converged", time.time() - t0, hist)
        if not np.isfinite(f3):
            break
    return RootResult(x[-1], f[-1], False, nfev, "not converged", time.time() - t0, hist)


@dataclass
class ScanResult:
    kx: np.ndarray
    omega: np.ndarray
    converged: np.ndarray
    D: np.ndarray
    nfev: np.ndarray
    seconds: np.ndarray


def scan_kx(disp, kx_values, omega0, *, tol=1e-9, max_jump=None, extrapolate=True, verbose=False, **kw):
    """Follow a root of disp.D(ω, k_x) along kx_values by continuation.

    The initial guess at each k_x is the linear extrapolation of the two previous roots (or the
    previous root). The scan stops (remaining entries NaN) when a root fails to converge or jumps by
    more than max_jump.
    """
    kx_values = np.asarray(kx_values, float)
    n = len(kx_values)
    om = np.full(n, np.nan + 1j * np.nan)
    Dv = np.full(n, np.nan + 1j * np.nan)
    conv = np.zeros(n, bool)
    nfev = np.zeros(n, int)
    secs = np.zeros(n)
    guess = complex(omega0)
    for i, kx in enumerate(kx_values):
        if i >= 2 and extrapolate and conv[i - 1] and conv[i - 2]:
            guess = om[i - 1] + (om[i - 1] - om[i - 2]) * (kx - kx_values[i - 1]) / (kx_values[i - 1] - kx_values[i - 2])
        r = find_root(lambda w: disp.D(w, kx), guess, tol=tol, **kw)
        nfev[i], secs[i] = r.nfev, r.seconds
        if verbose:
            print("kx=%.4f omega=%s |D|=%.1e conv=%s nfev=%d %.1fs" % (kx, r.omega, abs(r.D), r.converged, r.nfev, r.seconds), flush=True)
        if not r.converged or (max_jump is not None and i > 0 and conv[i - 1] and abs(r.omega - om[i - 1]) > max_jump):
            break
        om[i], Dv[i], conv[i] = r.omega, r.D, True
        guess = r.omega
    return ScanResult(kx_values, om, conv, Dv, nfev, secs)
