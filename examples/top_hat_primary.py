#!/usr/bin/env python3
"""Example: TSM against k_x for an analytically prescribed geometry and a top-hat primary.

Geometry given as functions of θ (AnalyticGeometry): radial magnetic drift v_Mx = sin θ, |∇x| = 1, B = 1
(uniform weight along θ). Primary (Primary.top_hat): T_P = 1 for θ ∈ [θ^P − Δ^P/2, θ^P + Δ^P/2] and 0
otherwise, with E×B amplitude Φ_P; τ = 1 and η∥ = η⊥ = 3. The two edges of the top hat are panel edges of
the θ integral, so the discontinuities cost nothing.

Run (≈ 20 s on 32 cores, a few minutes on 1):

    python examples/top_hat_primary.py                                  # θ^P = 0, Δ^P = π, Φ_P = 3
    python examples/top_hat_primary.py --theta-P 0.6 --Delta-P 1.2 --Phi-P 6 --guess 1.311+0.925j
    python examples/top_hat_primary.py --help

Writes examples/output/top_hat_primary.{txt,png,pdf}. Units: k_x in 1/ρ, ω in v_th/a (lengths in a, so
v_Mx = sin θ means a drift scale R = a). The scan stops where the root reaches the ISM region (default case:
k_x ρ ≈ 0.6). For other parameters, give a guess of ω at --kx-start (a growing root; scan |D(ω)| or continue
from a nearby case if unsure).
"""
import argparse
import os
import shutil
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import tsm_disp  # noqa: F401  (installed with pip)
except ImportError:  # running from a source checkout after `python setup.py build_ext --inplace`
    sys.path.insert(0, os.path.dirname(HERE))
from tsm_disp.geometry import AnalyticGeometry
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.roots import find_root


def min_D_ism(disp, omega, kx, n=48):
    """min over (θ, y) of |D_ISM|: → 0 where the TSM enters the ISM region (local ISM zeros on the surface),
    where D is not analytic and the TSM is subdominant to the ISMs (Nies & Parra 2026, §3.1)."""
    th = np.linspace(-np.pi, np.pi, n)
    return min(abs(disp.D_ism(omega, kx, t, y)) for t in th for y in np.linspace(-np.pi, np.pi, n, endpoint=False))


def follow(disp, kx0, w0, kmin, kmax, dk, direction, ism_stop=0.02):
    """Continuation in k_x from (kx0, w0); the step is halved (at most 3 times) when a root is not found,
    and the scan stops once the root reaches the ISM region (min|D_ISM| < ism_stop)."""
    pts, out, step, nhalf = [(kx0, w0)], [], dk, 0
    while True:
        k = round(pts[-1][0] + direction * step, 10)
        if not kmin - 1e-9 <= k <= kmax + 1e-9:
            return out
        ks, ws = zip(*pts[-3:])
        guess = np.polyval(np.polyfit(ks, ws, len(ks) - 1), k) if len(ks) > 1 else ws[0] * k / ks[0]
        r = find_root(lambda w: disp.D(w, k), guess, tol=1e-8, maxiter=25)
        if not r.converged or r.omega.imag <= 0 or abs(r.omega - guess) > 0.1 * abs(guess) + 0.02:
            if nhalf == 3:
                print("kx = %.4f: no growing root near %s, stopping this direction" % (k, np.round(guess, 4)))
                return out
            step, nhalf = step / 2, nhalf + 1
            continue
        pts.append((k, r.omega))
        out.append((k, r.omega))
        m = min_D_ism(disp, r.omega, k)
        print("kx = %.4f   omega = %+.5f %+.5fi   (%d evaluations of D, min|D_ISM| = %.3f)"
              % (k, r.omega.real, r.omega.imag, r.nfev, m), flush=True)
        if m < ism_stop:
            print("kx = %.4f: the root has reached the ISM region, stopping this direction" % k)
            return out
        if nhalf and r.nfev <= 8:
            step, nhalf = min(dk, 2 * step), nhalf - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--theta-P", type=float, default=0.0, help="centre θ^P of the top hat")
    ap.add_argument("--Delta-P", type=float, default=np.pi, help="width Δ^P of the top hat (≤ 2π)")
    ap.add_argument("--Phi-P", type=float, default=3.0, help="primary E×B amplitude Φ_P (ρ v_th/a)")
    ap.add_argument("--kx-min", type=float, default=0.05)
    ap.add_argument("--kx-max", type=float, default=1.0)
    ap.add_argument("--dkx", type=float, default=0.05)
    ap.add_argument("--kx-start", type=float, default=0.3, help="k_x at which the scan starts")
    ap.add_argument("--guess", type=complex, default=1.054 + 0.662j, help="initial guess of ω at kx-start")
    ap.add_argument("--tol", type=float, default=1e-6, help="tolerance of the θ integral")
    ap.add_argument("--nthreads", type=int, default=min(os.cpu_count() or 1, 32))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    a = ap.parse_args()

    # 1. geometry from functions of θ: v_Mx = sin θ, |∇x| = 1 (B = 1, uniform weight by default)
    geo = AnalyticGeometry(vMx=np.sin, gradx=1.0)
    # 2. primary: v_Ex = −Φ_P T_P(θ) sin y with T_P the top hat
    pri = Primary.top_hat(a.theta_P, a.Delta_P, a.Phi_P)
    # 3. dispersion relation (λ form: exact finite-Larmor-radius effects)
    disp = DispersionRelation(geo, pri, tau=1.0, eta_par=3.0, eta_prp=3.0, form="lambda",
                              epsabs=a.tol, epsrel=a.tol, nthreads=a.nthreads, warn=False)

    # 4. root at kx-start, then continuation up and down in k_x
    t0 = time.time()
    r = find_root(lambda w: disp.D(w, a.kx_start), a.guess, tol=1e-8, maxiter=40)
    if not r.converged:
        sys.exit("no root found near %s at kx = %g; try another --guess" % (a.guess, a.kx_start))
    print("kx = %.4f   omega = %+.5f %+.5fi   (start)" % (a.kx_start, r.omega.real, r.omega.imag))
    rows = sorted(follow(disp, a.kx_start, r.omega, a.kx_min, a.kx_max, a.dkx, -1) + [(a.kx_start, r.omega)]
                  + follow(disp, a.kx_start, r.omega, a.kx_min, a.kx_max, a.dkx, +1), key=lambda q: q[0])
    kx = np.array([q[0] for q in rows])
    omega = np.array([q[1] for q in rows])

    # 5. save
    os.makedirs(a.out, exist_ok=True)
    np.savetxt(os.path.join(a.out, "top_hat_primary.txt"), np.c_[kx, omega.real, omega.imag],
               header="kx*rho  omega_r*a/v_th  gamma*a/v_th\nvMx = sin(theta), |grad x| = 1, top-hat primary "
                      "theta_P=%g Delta_P=%g Phi_P=%g, eta_par=eta_prp=3, tau=1" % (a.theta_P, a.Delta_P, a.Phi_P))
    plot(kx, omega, geo, pri, a)
    print("wrote %s/top_hat_primary.{txt,png,pdf} (%.0f s)" % (a.out, time.time() - t0))


def plot(kx, omega, geo, pri, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 24, "axes.titlepad": 15,
                         "text.usetex": shutil.which("latex") is not None})
    fig, axs = plt.subplots(1, 3, figsize=(27, 8))
    axs[0].plot(kx, omega.real, "-o", lw=3, ms=6, c="C0")
    axs[1].plot(kx, omega.imag, "-o", lw=3, ms=6, c="C0")
    for ax, lab in zip(axs[:2], (r"$\omega_r a/v_{th}$", r"$\gamma a/v_{th}$")):
        ax.set_xlabel(r"$k_x \rho$")
        ax.set_ylabel(lab)
        ax.set_xlim(0, a.kx_max)
        ax.set_ylim(bottom=0)
    th = np.linspace(-np.pi, np.pi, 2001)
    axs[2].plot(th, pri(th), "-", lw=3, c="C2", label=r"$T_P(\theta)$")
    axs[2].plot(th, geo.vMx(th), "--", lw=3, c="C1", label=r"$v_{Mx} = \sin\theta$")
    axs[2].plot(th, geo.gradx(th), ":", lw=3, c="k", label=r"$|\nabla x| = 1$")
    axs[2].axhline(0, c="grey", lw=0.8)
    axs[2].set_xlabel(r"$\theta$")
    axs[2].set_xlim(-np.pi, np.pi)
    axs[2].legend(fontsize=20, loc="lower right")
    axs[0].set_title(r"$v_{Mx} = \sin\theta$, $|\nabla x| = 1$, $\eta_\parallel = \eta_\perp = 3$")
    axs[1].set_title(r"top hat $\theta^P = %.3g$, $\Delta^P = %.3g$, $\Phi_P = %g$" % (a.theta_P, a.Delta_P, a.Phi_P))
    axs[2].set_title(r"assumed along $\theta$")
    fig.tight_layout()
    stamp = "%s  ·  made by examples/top_hat_primary.py %s" % (os.path.abspath(a.out), " ".join(sys.argv[1:]))
    if len(stamp) > 190:                        # keep the provenance line within the figure width
        stamp = "…" + stamp[-189:]
    fig.text(0.005, 1.0, stamp, usetex=False, family="monospace", fontsize=10, color="0.5", va="bottom", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "top_hat_primary." + ext), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
