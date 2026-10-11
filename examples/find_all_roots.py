#!/usr/bin/env python3
"""Example: all roots of the dispersion function in a box of the complex ω plane, without an initial guess.

tsm_disp.contour_roots.find_all_roots samples 𝒟 on the box boundary until the phase is resolved, counts the zeros
by the argument principle, bisects the box until each cell holds a few zeros, locates them from the contour moments
and polishes and verifies them. Where 𝒟 is not analytic (the ISM region, D_ISM = 0 on the surface) the field-line
quadrature cannot converge; those samples are flagged and the cells reported instead of resolved.

Case: CBC (q = 20), Gaussian primary Φ_P = 0.723, η∥ = η⊥ = η, λ form (no streaming), k_x ρ_i = 0.5.

    python examples/find_all_roots.py                 # η = 3: the TSM 0.5809 + 0.2927i just above the ISM region
    python examples/find_all_roots.py --eta 1.5       # weaker drive
    python examples/find_all_roots.py --help

Writes examples/output/find_all_roots_eta<η>_kx<k_x>.{txt,png,pdf}: the roots, the suspect regions, and a map of the
search (terminal cells, boundary samples, flagged samples) over the local ISM roots ω(z, y) of D_ISM = 0.
"""
import argparse
import os
import shutil
import sys
import time
import warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import tsm_disp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(HERE))
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.contour_roots import find_all_roots, local_ism_roots


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eta", type=float, default=3.0, help="η∥ = η⊥")
    ap.add_argument("--kx", type=float, default=0.5)
    ap.add_argument("--Phi-P", type=float, default=0.72314)
    ap.add_argument("--wr", type=float, nargs=2, default=(0.0, 1.2), help="ω_r range (v_th/a)")
    ap.add_argument("--wi", type=float, nargs=2, default=(0.02, 0.5), help="γ range (v_th/a)")
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--no-ism", action="store_true", help="skip the local ISM roots (≈ 5 s)")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    a = ap.parse_args()

    geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, kappa=1.0, zed_equal_arc=True)
    pri = Primary.gaussian(a.Phi_P, width=np.pi)
    disp = DispersionRelation(geo, pri, tau=1.0, eta_par=a.eta, eta_prp=a.eta, nthreads=a.nthreads, warn=False)
    t0 = time.time()
    res = find_all_roots(disp, a.kx, tuple(a.wr), tuple(a.wi), verbose=True)
    print(res.summary())
    ism = None
    if not a.no_ism:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ism = local_ism_roots(disp, a.kx)
        print("local ISM roots: %d, ω_r ∈ [%.3f, %.3f], γ ≤ %.3f" % (len(ism), ism[:, 2].min(), ism[:, 2].max(),
                                                                      ism[:, 3].max()))
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, "find_all_roots_eta%g_kx%g" % (a.eta, a.kx))
    with open(stem + ".txt", "w") as fh:
        fh.write("# CBC q=20, Gaussian primary Phi_P=%g, eta_par=eta_prp=%g, kx=%g, box %s x %s (v_th/a)\n"
                 % (a.Phi_P, a.eta, a.kx, tuple(a.wr), tuple(a.wi)))
        fh.write(res.summary() + "\n")
    plot(res, ism, a, stem)
    print("wrote %s.{txt,png,pdf} (%.0f s)" % (stem, time.time() - t0))


def plot(res, ism, a, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "serif", "font.size": 24, "axes.titlepad": 15,
                         "text.usetex": shutil.which("latex") is not None})
    fig, ax = plt.subplots(figsize=(12, 9))
    if ism is not None and len(ism):
        ax.plot(ism[:, 2], ism[:, 3], ".", ms=4, c="C1", alpha=0.6, label=r"local ISM roots $D_{\rm ISM}(z,y,\omega)=0$")
    for g in res.suspect_regions:
        r = g.rect
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], fc="C3", alpha=0.15, ec="C3", lw=1.5))
    for c in res.cells:
        r = c["rect"]
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], fill=False, ec="0.4", lw=1.0))
    w, f, sus = res.samples
    ax.plot(w.real[~sus], w.imag[~sus], ".", ms=5, c="C0", label="boundary samples")
    ax.plot(w.real[sus], w.imag[sus], "x", ms=8, mew=2, c="C3", label="flagged samples (quadrature limit)")
    for q in res.roots:
        ax.plot(q.omega.real, q.omega.imag, "*", ms=24, c="k", mfc="gold", mew=1.5)
    if res.roots:
        ax.plot([], [], "*", ms=20, c="k", mfc="gold", label="roots (verified)")
    for q in res.rejected:
        ax.plot(q.omega.real, q.omega.imag, "P", ms=16, c="C3")
    if res.rejected:
        ax.plot([], [], "P", ms=14, c="C3", label="rejected (non-analytic)")
    ax.add_patch(Rectangle((0, 0), 0, 0, fc="C3", alpha=0.15, ec="C3", label="suspect cells"))
    ax.set_xlim(a.wr[0] - 0.02, a.wr[1] + 0.02)
    ax.set_ylim(min(a.wi[0], 0) - 0.01, a.wi[1] + 0.01)
    ax.set_xlabel(r"$\omega_r\, a/v_{th}$")
    ax.set_ylabel(r"$\gamma\, a/v_{th}$")
    ax.set_title(r"$k_x\rho_i = %g$, $\eta_\parallel = \eta_\perp = %g$: %d root(s), %d evaluations"
                 % (a.kx, a.eta, len(res.roots), res.nfev))
    ax.legend(fontsize=16, loc="upper right")
    fig.tight_layout()
    stamp = "%s  ·  made by examples/find_all_roots.py %s" % (os.path.abspath(a.out), " ".join(sys.argv[1:]))
    if len(stamp) > 120:
        stamp = "…" + stamp[-119:]
    fig.text(0.005, 1.0, stamp, usetex=False, family="monospace", fontsize=10, color="0.5", va="bottom", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(stem + "." + ext, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
