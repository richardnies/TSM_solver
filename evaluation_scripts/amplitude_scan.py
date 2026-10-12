"""Continuation of a root in the primary amplitude Φ_P: the RDK limit at large Φ_P, the approach to the ISM region at
small Φ_P, and the data and figure.

Large Φ_P (Nies & Parra 2026, eq. 51): once ω² ~ (k_x v_Ex)² ≫ (k_x v_Mx)²/b ~ ω_GAM², the radial magnetic drift drops out
and the TSM becomes the purely growing RDK secondary mode. Without drift D depends on ω and Φ_P only through
ω/(k_x Φ_P), so the RDK root is ω = s k_x Φ_P with s fixed by k_x (b) and the primary's shape; in the long-wavelength
limit (eq. 51, θ-dependent primary, v_dia/v_Ex = τ(1 + η⊥))

    ω_RDK² = −k_x² ⟨b v_Ex (v_Ex + v_dia)⟩/⟨b⟩   ⇒   s = i √(C_RDK ⟨b Ftheta_P²⟩/⟨b⟩),  C_RDK = ½[1 + τ(1 + η⊥)].

Small Φ_P: the TSM reaches the ISM region (local zeros of D_ISM on the surface, where D is not analytic) at γ > 0;
min|D_ISM| over (θ, y) at the root goes linearly to zero there.
"""
import numpy as np
from tsm_disp.geometry.base import Geometry
from tsm_disp.limits import C_RDK
from .kx_scan import min_D_ism
from .plotting import style, save_pdf, drive_label


class WithoutDrift(Geometry):
    """The geometry `base` with the radial magnetic drift switched off (v_Mx = 0): D in the RDK limit."""

    def __init__(self, base):
        self.base = base
        for a in ("zmin", "zmax", "periodic", "shat", "q", "eps", "rmaj", "exb_nonlin_fac", "sign_torflux"):
            setattr(self, a, getattr(base, a, None))
        self.name = "%s without drift" % getattr(base, "name", "geometry")

    def vMx(self, z):
        return np.zeros(np.shape(z)) if np.ndim(z) else 0.0

    def kperp2_kx2(self, z):
        return self.base.kperp2_kx2(z)

    def b(self, z, kx):
        return self.base.b(z, kx)

    def weight(self, z):
        return self.base.weight(z)

    def breakpoints(self):
        return self.base.breakpoints()


def _imaginary_axis_root(d, kx, gamma_guess, n=40, span=(0.4, 2.5)):
    """Largest purely growing root iγ of d with γ in span × gamma_guess: D(iγ) is real (D(−ω*) = D(ω)*), so its sign
    changes on n points bracket the roots, which are then found by bisection (brentq). None if there is none."""
    from scipy.optimize import brentq
    g = gamma_guess * np.linspace(span[0], span[1], n)
    f = lambda x: d.D(1j * x, kx).real
    v = np.array([f(x) for x in g])
    k = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if not len(k):
        return None
    j = k[-1]
    return 1j * brentq(f, g[j], g[j + 1], xtol=1e-12 * gamma_guess)


def follow_amplitude(make_disp, kx, amplitudes, omega0, *, ism=False, ism_stop=None, tol=1e-8, maxiter=30,
                     max_jump=0.15, verbose=True):
    """Follow a root of D(ω; Φ_P) at fixed k_x through the monotonic sequence `amplitudes`, from a guess omega0 at
    amplitudes[0]. The guess at each step extrapolates ω/Φ_P linearly in ln Φ_P from the last two roots (ω ∝ Φ_P in
    the RDK limit). make_disp(Φ_P) → dispersion object with find_root(ω0, kx, tol, maxiter).

    Merger on the imaginary axis: at large Φ_P the TSM's ω_r falls to zero, where it meets its mirror root −ω* and
    the pair becomes two purely growing roots, one of which tends to the RDK mode. Newton cannot follow a root through
    this double root, so when a step fails the largest purely growing root is searched on the imaginary axis
    (D(iγ) is real there); from then on the guesses are kept on the axis, where Newton stays, and the axis is
    searched again whenever Newton fails.

    Stops when no growing root is found or the root jumps (|ω − guess| > max_jump |guess|); with ism (or ism_stop)
    min|D_ISM| is computed at every root, and with ism_stop the scan stops after a root with min|D_ISM| < ism_stop.
    Returns (Φ_P, ω, min|D_ISM|) arrays of the roots found (min|D_ISM| = NaN where not computed); the amplitude
    interval of the merger, if any, is in follow_amplitude.merger."""
    out, on_axis = [], False
    follow_amplitude.merger = None
    for amp in amplitudes:
        if not out:
            g = complex(omega0)
        elif len(out) == 1:
            g = out[-1][1] * amp / out[-1][0]
        else:
            (a1, w1, _), (a2, w2, _) = out[-2:]
            r1, r2 = w1 / a1, w2 / a2
            g = amp * (r2 + (r2 - r1) * np.log(amp / a2) / np.log(a2 / a1))
        if on_axis:
            g = 1j * g.imag
        d = make_disp(amp)
        r = d.find_root(g, kx, tol=tol, maxiter=maxiter)
        ok = r.converged and r.omega.imag > 0 and not (out and abs(r.omega - g) > max_jump * abs(g))
        w = r.omega if ok else None
        if not ok and out:                       # merger, or a poor guess on the axis: search the axis
            w = _imaginary_axis_root(d, kx, abs(g))
            if w is not None and not on_axis:
                on_axis = True
                follow_amplitude.merger = (out[-1][0], amp)
                if verbose:
                    print("Phi_P = %.4g: the root has merged with its mirror root on the imaginary axis; following the "
                          "largest purely growing root" % amp)
        if w is None:
            if verbose:
                print("Phi_P = %.4g: no growing root near %s, stopping" % (amp, np.round(g, 5)))
            break
        m = min_D_ism(d, w, kx) if (ism or ism_stop) else np.nan
        out.append((amp, w, m))
        if verbose:
            print("Phi_P = %8.4f   omega = %+.5f %+.5fi   min|D_ISM| = %.3f" % (amp, w.real, w.imag, m), flush=True)
        if ism_stop and m < ism_stop:
            if verbose:
                print("Phi_P = %.4g: the root has reached the ISM region (min|D_ISM| < %g), stopping" % (amp, ism_stop))
            break
    if not out:
        return np.array([]), np.array([], complex), np.array([])
    a, w, m = zip(*out)
    return np.array(a), np.array(w), np.array(m)


def rdk_ratio(disp_without_drift, kx, guess):
    """s = ω_RDK/(k_x Φ_P) from the drift-free dispersion object (any amplitude; guess near its root)."""
    r = disp_without_drift.find_root(guess, kx, tol=1e-10, maxiter=40)
    if not r.converged:
        raise RuntimeError("no drift-free (RDK) root near %s" % guess)
    return r.omega / (kx * disp_without_drift.primary.Phi_P)


def rdk_ratio_long_wavelength(geometry, primary, kx, tau, eta_prp):
    """Eq. (51) for a θ-dependent primary: s = i √(C_RDK ⟨b Ftheta_P²⟩/⟨b⟩) (k_x → 0)."""
    num = geometry.average(lambda z: geometry.b(z, kx) * primary(z)**2, points=getattr(primary, "breakpoints", []))
    den = geometry.average(lambda z: geometry.b(z, kx))
    return 1j * np.sqrt(C_RDK(tau, eta_prp) * num / den)


def ism_amplitude(phi, ism, npts=3):
    """Φ_P where min|D_ISM| extrapolates linearly to zero through the last npts roots of a downward scan."""
    if len(phi) < 2 or not np.all(np.isfinite(ism[-npts:])):
        return np.nan
    c = np.polyfit(phi[-npts:], ism[-npts:], 1)
    return -c[1] / c[0] if c[0] != 0 else np.nan


def save_amplitude_scan(path, phi, omega, ism, header):
    np.savetxt(path, np.c_[phi, omega.real, omega.imag, ism],
               header="Phi_P  omega_r*a/v_th  gamma*a/v_th  min|D_ISM| (NaN: not computed)\n" + header)
    return path


def plot_amplitude_scan(path, script, phi, omega, ism, kx, s_exact, s_long, phi_gam, phi_ism, *, case_title="",
                        eta_par=3.0, eta_prp=3.0, tau=1.0):
    """Left: ω/(k_x Φ_P) against Φ_P (log scale) with the RDK limits (drift-free root, eq. 51) and the amplitude where
    ω_RDK = ω_GAM. Right: min|D_ISM| at the root against Φ_P, with its linear extrapolation to the ISM region."""
    plt = style()
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    o = np.argsort(phi)
    phi, omega, ism = phi[o], omega[o], ism[o]
    r = omega / (kx * phi)
    ax = axs[0]
    ax.semilogx(phi, r.imag, "-o", lw=3, ms=5, c="C0", label=r"$\gamma/(k_x\Phi_P)$")
    ax.semilogx(phi, r.real, "-o", lw=3, ms=5, c="C1", label=r"$\omega_r/(k_x\Phi_P)$")
    ax.axhline(s_exact.imag, c="k", ls="--", lw=2.5, label=r"RDK: root without drift")
    ax.axhline(s_long.imag, c="0.4", ls=":", lw=2.5, label=r"RDK, eq. (51) ($k_x \to 0$)")
    if np.isfinite(phi_gam):
        ax.axvline(phi_gam, c="C2", ls="-.", lw=2, label=r"$\omega_{\rm RDK} = \omega_{\rm GAM}$")
    if np.isfinite(phi_ism):
        ax.axvline(phi_ism, c="C3", ls="-.", lw=2, label=r"ISM region reached")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel(r"$\Phi_P$ [$\rho v_{th}/a$]")
    ax.set_ylabel(r"$\omega/(k_x\Phi_P)$")
    ax.set_title(r"large $\Phi_P$: TSM $\to$ RDK" + (", " + case_title if case_title else ""))
    ax.legend(fontsize=17)
    ax = axs[1]
    m = np.isfinite(ism)
    ax.plot(phi[m], ism[m], "-o", lw=3, ms=6, c="C3", label=r"$\min_{\theta,y}|D_{\rm ISM}(\omega_{\rm TSM})|$")
    if np.isfinite(phi_ism) and m.sum() >= 2:
        x = np.linspace(phi_ism, phi[m][:3].max(), 50)
        c = np.polyfit(phi[m][:3], ism[m][:3], 1)
        ax.plot(x, np.polyval(c, x), "--", lw=2, c="C3", label=r"linear extrapolation: $\Phi_P = %.3f$" % phi_ism)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel(r"$\Phi_P$ [$\rho v_{th}/a$]")
    ax.set_ylabel(r"$\min|D_{\rm ISM}|$ at the root")
    ax.set_title(r"small $\Phi_P$: TSM $\to$ ISM region, $k_x\rho_i = %g$, " % kx + drive_label(eta_par, eta_prp, tau))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=17)
    fig.tight_layout()
    return save_pdf(fig, path, script)
