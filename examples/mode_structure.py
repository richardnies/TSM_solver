#!/usr/bin/env python3
"""Example: structure of a TSM on the flux surface — potential, pressure or nonlinear heat flux.

Same case as tsm_kx_scan.py (CBC, q = 20, η∥ = η⊥ = 3, Gaussian primary, Φ_P = 0.723) at k_xρ_i = 0.5.
The root ω is found first; then, for each requested quantity, the figure shows
  left : Re[f(θ, y) e^{i k_x x}] at x = 0 on the flux surface (y: phase of the primary, θ: field line)
  right: the primary-phase average ⟨f⟩_y(θ) (real and imaginary parts)
with f = φ̂ (potential), P̂ = T∫d³v (v∥²+v⊥²/2)/v_th² ĝ (pressure, paper eq. 38) or
Q̂_NL = v_Ex^P P̂ (nonlinear heat flux, eq. 40), normalised to Z e ⟨φ⟩_ψ/T = 1.

    python examples/mode_structure.py                       # all three quantities
    python examples/mode_structure.py --quantity heat_flux --kx 0.3
"""
import argparse
import os
import shutil
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import tsm_disp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation, QUANTITIES

LABEL = {"potential": (r"\hat\phi", r"Z e \hat\phi / T"),
         "pressure": (r"\hat P", r"\hat P / n T"),
         "heat_flux": (r"\hat Q_{NL}", r"\hat Q_{NL} / n T \rho v_{th} a^{-1}")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quantity", default="all", choices=("all",) + QUANTITIES)
    ap.add_argument("--kx", type=float, default=0.5)
    ap.add_argument("--guess", type=complex, default=0.58 + 0.29j, help="initial guess of ω (v_th/a)")
    ap.add_argument("--Phi-P", type=float, default=0.72314)
    ap.add_argument("--q", type=float, default=20.0)
    ap.add_argument("--nthreads", type=int, default=min(os.cpu_count() or 1, 32))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    a = ap.parse_args()

    geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=a.q, shat=0.8, zed_equal_arc=True)
    disp = DispersionRelation(geo, Primary.gaussian(a.Phi_P, width=np.pi), tau=1, eta_par=3, eta_prp=3,
                              epsabs=1e-9, epsrel=1e-9, nthreads=a.nthreads)
    r = disp.find_root(a.guess, a.kx, tol=1e-10)
    if not r.converged:
        sys.exit("no root found near %s; try --guess" % a.guess)
    print("k_x = %g: omega = %.5f %+.5fi (v_th/a)" % (a.kx, r.omega.real, r.omega.imag))
    os.makedirs(a.out, exist_ok=True)
    for q in (QUANTITIES if a.quantity == "all" else (a.quantity,)):
        m = disp.mode_structure(r.omega, a.kx, q, nz=161, ny=128)
        print("  %-9s <phi>_psi = %.6f%+.6fi (1 at the root)" % (q, m.phi_average.real, m.phi_average.imag))
        np.savez(os.path.join(a.out, "mode_%s.npz" % q), z=m.z, y=m.y, field=m.field, omega=r.omega, kx=a.kx)
        plot(m, q, a)


def plot(m, q, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tex = shutil.which("latex") is not None
    plt.rcParams.update({"font.family": "serif", "font.size": 24, "axes.titlepad": 15, "text.usetex": tex})
    sym, unit = LABEL[q]
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    f = m.real_space()
    vmax = np.abs(f).max()
    Y, Z = np.meshgrid(m.y, m.z)
    im = axs[0].pcolormesh(Y, Z, f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=axs[0])
    axs[0].set_xlabel(r"primary phase $y$")
    axs[0].set_ylabel(r"$\theta$ (equal-arc $z$)")
    axs[0].set_title(r"${\rm Re}[%s\, e^{i k_x x}]$ at $x = 0$" % sym)
    axs[1].plot(m.z, m.y_average.real, lw=3, label=r"${\rm Re}\,\langle %s\rangle_y$" % sym)
    axs[1].plot(m.z, m.y_average.imag, lw=3, ls="--", label=r"${\rm Im}\,\langle %s\rangle_y$" % sym)
    axs[1].axhline(0, c="k", lw=1)
    axs[1].set_xlabel(r"$\theta$ (equal-arc $z$)")
    axs[1].set_ylabel(r"$\langle %s \rangle_y$  [$%s$]" % (sym, unit))
    axs[1].set_title(r"$k_x\rho_i = %g$, $\omega a/v_{th} = %.3f %+.3fi$" % (m.kx, m.omega.real, m.omega.imag))
    axs[1].legend(fontsize=20)
    fig.tight_layout()
    stamp = "%s  ·  made by examples/mode_structure.py %s" % (os.path.abspath(a.out), " ".join(sys.argv[1:]))
    stamp = stamp if len(stamp) <= 150 else "…" + stamp[-149:]
    fig.text(0.005, 1.0, stamp, usetex=False, family="monospace", fontsize=10, color="0.5", va="bottom", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "mode_%s.%s" % (q, ext)), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
