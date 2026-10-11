#!/usr/bin/env python3
"""Example: frequency and growth rate of the toroidal secondary mode (TSM) against k_x.

Typical case of Nies & Parra (2026), Fig. 3: Cyclone Base Case surface (circular, r/a = 0.5, R/a = 2.778,
ŝ = 0.8, q = 20), modified adiabatic electrons with T_i/T_e = 1, a frozen streamer primary with
bi-Maxwellian temperature-to-density ratios η∥ = η⊥ = 3, Gaussian along the field line (width π), and
E×B amplitude Φ_P = 0.723 ρ_i v_th/a (the label 𝒜 = 2 of the paper's figures).

Run (≈ 10–60 s depending on the number of cores):

    python examples/tsm_kx_scan.py                # writes examples/output/tsm_kx_scan.{txt,png,pdf}
    python examples/tsm_kx_scan.py --help         # options (amplitude, k_x range, threads, ...)
    python examples/tsm_kx_scan.py --Q 1159 --ky-P 0.001 --delta 0.5  # amplitude from the primary heat flux (Φ_P = 0.723)

Output units: k_x in 1/ρ_i, frequencies in v_th/a with v_th = √(2T/m) (stella units); the plot shows
ω R/v_th as in the paper. Markers: stella gyrokinetic simulations of the same case (Θ = 0.2886 runs).
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
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.roots import find_root

# stella secondary-mode simulations of this case (𝒜 = 2, Θ = 0.2886), ω in v_th/a
STELLA_KX = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
STELLA_OMEGA = np.array([0.1043 + 0.0675j, 0.2168 + 0.1374j, 0.3347 + 0.2031j, 0.4547 + 0.2575j,
                         0.5793 + 0.2922j, 0.7130 + 0.3047j, 0.8578 + 0.3003j, 1.0091 + 0.2865j])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--Phi-P", type=float, default=0.72314, help="primary E×B amplitude (ρ_i v_th/a)")
    ap.add_argument("--Q", type=float, default=None,
                    help="set the amplitude instead by the primary's ion heat flux in stella's gyro-Bohm units "
                         "(needs --delta ≠ 0 and --ky-P)")
    ap.add_argument("--ky-P", type=float, default=1e-3, help="primary binormal wavenumber (for --Q; stella 1/y0)")
    ap.add_argument("--delta", type=float, default=0.0,
                    help="phase Δ∥ = Δ⊥ of the primary temperature relative to φ (stella: −den0_phase)")
    ap.add_argument("--q", type=float, default=20.0, help="safety factor")
    ap.add_argument("--kx-min", type=float, default=0.1)
    ap.add_argument("--kx-max", type=float, default=0.8)
    ap.add_argument("--dkx", type=float, default=0.05)
    ap.add_argument("--kx-start", type=float, default=0.5, help="k_x at which the scan starts")
    ap.add_argument("--guess", type=complex, default=0.58 + 0.29j, help="initial guess of ω at kx-start")
    ap.add_argument("--tol", type=float, default=1e-6, help="tolerance of the field-line integral")
    ap.add_argument("--nthreads", type=int, default=min(os.cpu_count() or 1, 32))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    a = ap.parse_args()

    # 1. geometry: Miller parametrisation of the CBC surface on stella's equal-arc coordinate
    geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=a.q, shat=0.8, kappa=1.0, zed_equal_arc=True)
    # 2. primary: v_Ex = −Φ_P T_P(z) sin y with T_P a Gaussian of width π normalised to unit RMS
    pri = Primary.gaussian(a.Phi_P, width=np.pi)
    if a.Q is not None:          # amplitude from the primary heat flux (stella's qflx normalisation)
        pri = pri.with_heat_flux(a.Q, geo, a.ky_P, eta_par=3.0, eta_prp=3.0, delta_par=a.delta, delta_prp=a.delta)
        a.Phi_P = pri.Phi_P
        print("Q = %g, k_y^P = %g, Δ = %g  ->  Phi_P = %.5f" % (a.Q, a.ky_P, a.delta, a.Phi_P))
    # 3. dispersion relation (λ form: exact finite-Larmor-radius effects)
    disp = DispersionRelation(geo, pri, tau=1.0, eta_par=3.0, eta_prp=3.0, delta_par=a.delta, delta_prp=a.delta,
                              form="lambda",
                              epsabs=a.tol, epsrel=a.tol, nthreads=a.nthreads, warn=False)

    # 4. follow the root from kx-start up and down by continuation
    kx = np.round(np.arange(a.kx_min, a.kx_max + 1e-9, a.dkx), 10)
    omega = np.full(len(kx), np.nan + 0j)
    i0 = int(np.argmin(np.abs(kx - a.kx_start)))
    t0 = time.time()
    for order in (range(i0, len(kx)), range(i0 - 1, -1, -1)):
        prev = []
        for i in order:
            if len(prev) >= 2:
                guess = 2 * prev[-1] - prev[-2]          # linear extrapolation
            elif prev:
                guess = prev[-1]
            else:
                guess = a.guess if i == i0 else omega[i0]
            r = find_root(lambda w: disp.D(w, kx[i]), guess, tol=1e-8, maxiter=30)
            if not r.converged or r.omega.imag <= 0:
                print("kx = %.3f: no growing root found near %s, stopping this direction" % (kx[i], guess))
                break
            omega[i] = r.omega
            prev.append(r.omega)
            print("kx = %.3f   omega = %+.5f %+.5fi   (%d evaluations of D, %.1f s)"
                  % (kx[i], r.omega.real, r.omega.imag, r.nfev, time.time() - t0), flush=True)

    # 5. save
    os.makedirs(a.out, exist_ok=True)
    R = geo.rmaj
    np.savetxt(os.path.join(a.out, "tsm_kx_scan.txt"),
               np.c_[kx, omega.real, omega.imag, omega.real * R, omega.imag * R],
               header="kx*rho_i  omega_r*a/v_th  gamma*a/v_th  omega_r*R/v_th  gamma*R/v_th\n"
                      "CBC q=%g, eta_par=eta_prp=3, Gaussian primary Phi_P=%g" % (a.q, a.Phi_P))
    plot(kx, omega, R, a)
    print("wrote %s/tsm_kx_scan.{txt,png,pdf} (%.0f s)" % (a.out, time.time() - t0))


def plot(kx, omega, R, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 24, "axes.titlepad": 15,
                         "text.usetex": shutil.which("latex") is not None})
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for ax, part, lab in [(axs[0], np.real, r"$\omega_r R/v_{th}$"), (axs[1], np.imag, r"$\gamma R/v_{th}$")]:
        ax.plot(kx, part(omega) * R, "-", lw=3, c="C0", label="TSM\\_solver (theory)" if plt.rcParams["text.usetex"] else "TSM_solver (theory)")
        if abs(a.q - 20) < 1e-12 and abs(a.Phi_P - 0.72314) < 1e-4 and a.delta == 0:
            ax.plot(STELLA_KX, part(STELLA_OMEGA) * R, "x", ms=14, mew=3, c="k", label="stella (gyrokinetic)")
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=20)
    axs[0].set_title(r"CBC, $q = %g$, $\eta_\parallel = \eta_\perp = 3$" % a.q)
    axs[1].set_title(r"Gaussian primary, $\Phi_P = %.3f$" % a.Phi_P)
    fig.tight_layout()
    stamp = "%s  ·  made by examples/tsm_kx_scan.py %s" % (os.path.abspath(a.out), " ".join(sys.argv[1:]))
    if len(stamp) > 150:                        # keep the provenance line within the figure width
        stamp = "…" + stamp[-149:]
    fig.text(0.005, 1.0, stamp, usetex=False, family="monospace", fontsize=10, color="0.5", va="bottom", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "tsm_kx_scan." + ext), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
