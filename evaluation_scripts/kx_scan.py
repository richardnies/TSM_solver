"""Continuation of a root of D(ω, k_x) in k_x, the ISM-region distance min|D_ISM|, and the ω(k_x) data and figure."""
import time
import numpy as np
from .plotting import style, save_pdf


def follow_kx(disp, kx, kx_start, guess, *, streaming_solver=None, tol=1e-8, maxiter=30, verbose=True):
    """Follow a growing root over the grid kx: from the grid point nearest kx_start (initial guess `guess`) upward,
    then downward from kx_start. The guess at each point is the linear extrapolation of the last two roots (one
    root: ω ∝ k_x, roughly true of the TSM); a direction stops when no growing root is found.

    disp : dispersion object with find_root(ω0, kx, tol, maxiter) (DispersionRelation: Newton with the exact ∂D/∂ω).
    streaming_solver : optional second, slower dispersion object (e.g. StreamingDispersion, with ion parallel
        streaming). At each k_x, disp's root ω_λ is found first and the streaming solver's root is then found from
        ω_λ (Muller from a distant guess can hop branches); the streaming root is the result and is continued from.
    Returns ω on the grid (NaN where no root was found)."""
    kx = np.asarray(kx, float)
    omega = np.full(len(kx), np.nan + 1j * np.nan)
    i0 = int(np.argmin(np.abs(kx - kx_start)))
    t0 = time.time()
    for order in (range(i0, len(kx)), range(i0 - 1, -1, -1)):
        prev = [] if order[0] == i0 else [(kx[i0], omega[i0])]     # the downward pass continues from kx_start
        for i in order:
            if len(prev) >= 2:
                (k1, w1), (k2, w2) = prev[-2:]
                g = w2 + (w2 - w1) * (kx[i] - k2) / (k2 - k1)
            elif prev:
                g = prev[-1][1] * kx[i] / prev[-1][0]
            else:
                g = guess
            if not np.isfinite(g):
                break
            r = disp.find_root(g, kx[i], tol=tol, maxiter=maxiter)
            if streaming_solver is not None and r.converged:
                r = streaming_solver.find_root(r.omega, kx[i], tol=tol, maxiter=maxiter)
            if not r.converged or r.omega.imag <= 0:
                if verbose:
                    print("kx = %.3f: no growing root found near %s, stopping this direction" % (kx[i], g))
                break
            omega[i] = r.omega
            prev.append((kx[i], r.omega))
            if verbose:
                print("kx = %.3f   omega = %+.5f %+.5fi   (%d evaluations of D, %.1f s)"
                      % (kx[i], r.omega.real, r.omega.imag, r.nfev, time.time() - t0), flush=True)
    return omega


def min_D_ism(disp, omega, kx, n=48):
    """min over (θ, y) of |D_ISM| on an n × n grid: → 0 where the root enters the ISM region (local ISM zeros on the
    surface), where D is not analytic and the TSM is subdominant to the ISMs (Nies & Parra 2026, §3.1)."""
    g = disp.geometry
    th = np.linspace(g.zmin, g.zmax, n)
    return min(abs(disp.D_ism(omega, kx, t, y)) for t in th for y in np.linspace(-np.pi, np.pi, n, endpoint=False))


def follow_kx_adaptive(disp, kx0, w0, kmin, kmax, dk, direction, *, ism_stop=0.02, tol=1e-8, maxiter=25,
                       verbose=True):
    """Continuation in k_x from the root (kx0, w0) in the given direction (±1) with step dk, halved (at most 3 times)
    when no root is found and restored when roots come easily; the guess is the polynomial extrapolation of the last
    ≤ 3 roots. Stops at kmin/kmax, when no growing root is found, or once the root reaches the ISM region
    (min|D_ISM| < ism_stop; to decide whether a branch really ends there, see tsm_disp.contour_roots.find_all_roots).
    Returns a list of (k_x, ω) excluding the start."""
    pts, out, step, nhalf = [(kx0, w0)], [], dk, 0
    while True:
        k = round(pts[-1][0] + direction * step, 10)
        if not kmin - 1e-9 <= k <= kmax + 1e-9:
            return out
        ks, ws = zip(*pts[-3:])
        guess = np.polyval(np.polyfit(ks, ws, len(ks) - 1), k) if len(ks) > 1 else ws[0] * k / ks[0]
        r = disp.find_root(guess, k, tol=tol, maxiter=maxiter)
        if not r.converged or r.omega.imag <= 0 or abs(r.omega - guess) > 0.1 * abs(guess) + 0.02:
            if nhalf == 3:
                if verbose:
                    print("kx = %.4f: no growing root near %s, stopping this direction" % (k, np.round(guess, 4)))
                return out
            step, nhalf = step / 2, nhalf + 1
            continue
        pts.append((k, r.omega))
        out.append((k, r.omega))
        m = min_D_ism(disp, r.omega, k)
        if verbose:
            print("kx = %.4f   omega = %+.5f %+.5fi   (%d evaluations of D, min|D_ISM| = %.3f)"
                  % (k, r.omega.real, r.omega.imag, r.nfev, m), flush=True)
        if m < ism_stop:
            if verbose:
                print("kx = %.4f: the root has reached the ISM region, stopping this direction" % k)
            return out
        if nhalf and r.nfev <= 8:
            step, nhalf = min(dk, 2 * step), nhalf - 1


def save_kx_scan(path, kx, omega, header, R=None):
    """Text file: k_x, ω_r, γ (v_th/a), and ω_r R, γ R if R is given."""
    cols = [kx, omega.real, omega.imag] + ([omega.real * R, omega.imag * R] if R is not None else [])
    names = "kx*rho  omega_r*a/v_th  gamma*a/v_th" + ("  omega_r*R/v_th  gamma*R/v_th" if R is not None else "")
    np.savetxt(path, np.c_[tuple(cols)], header=names + "\n" + header)
    return path


def plot_kx_scan(path, script, kx, omega, *, scale=1.0, unit=r"R/v_{th}", label=None, reference=None,
                 titles=("", ""), title_fontsize=None, xlabel=r"$k_x \rho_i$", markers=False, xlim=None,
                 ylim_zero=False, profiles=None, profile_title=None):
    """ω_r and γ (times `scale`, axis unit `unit`) against k_x; reference = (kx, ω, label) drawn as crosses;
    profiles = [(θ, f(θ), linestyle, colour, label), …] adds a third panel of what is assumed along the field line."""
    plt = style()
    n = 3 if profiles else 2
    fig, axs = plt.subplots(1, n, figsize=(27 if n == 3 else 20, 8))
    for ax, part, lab in [(axs[0], np.real, r"$\omega_r %s$" % unit), (axs[1], np.imag, r"$\gamma %s$" % unit)]:
        ax.plot(kx, part(omega) * scale, "-o" if markers else "-", lw=3, ms=6, c="C0", label=label)
        if reference is not None:
            ax.plot(reference[0], part(np.asarray(reference[1])) * scale, "x", ms=14, mew=3, c="k", label=reference[2])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim_zero:
            ax.set_ylim(bottom=0)
    if label is not None or reference is not None:
        axs[0].legend(fontsize=20)
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=title_fontsize)
    if profiles:
        for th, f, ls, c, lab in profiles:
            axs[2].plot(th, f, ls, lw=3, c=c, label=lab)
        axs[2].axhline(0, c="grey", lw=0.8)
        axs[2].set_xlabel(r"$\theta$")
        axs[2].set_xlim(th[0], th[-1])
        axs[2].legend(fontsize=20, loc="lower right")
        axs[2].set_title(profile_title or r"assumed along $\theta$")
    fig.tight_layout()
    return save_pdf(fig, path, script)
