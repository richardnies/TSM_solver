#!/usr/bin/env python3
"""Benchmark tsm_disp against stella secondary-mode simulations (CBC, circular Miller surface).

Data (stella outputs were deleted; the extracted growth rates remain):
  A  q = 20, Gaussian primary of width π in zed (equal-arc), η∥ = η⊥ = 3, labels A = 1, 2, 4
     (Φ_legacy = 0.3557 A), phase-shift-periodic (Θ = 0.2886) and periodic runs:
     2023-06-13_QN_scans/2024-03-04_compare_with_GK_secondary/PhiP-*_rerun/secondary_equivalent_q-20_*
  B  q = 2.8, θ-constant primary (width0 = 100), η⊥ = 3, η∥ = 0, phiinit = 35, y0 = 100:
     2026-03-02_data_turbulence_saturation_.../stella_secondary-simulations/1D_scan_Fig1
Solver variants:
  miller    : Miller geometry of the deck (zed_equal_arc), primary in zed, Φ_P from the deck (stella_phiP)
  legacy    : circular analytic model of the original code, Φ_P = 0.3557 A (old-code inputs, new kernel)
  old code  : stored roots of code_eval_disp_NEW (λ form with the complex-ive bug), case A only
Writes data_*.txt next to this script; plot_benchmark.py makes the figures.

Usage: python run_benchmark.py [--nthreads 48]
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import miller, CircularAnalytic
from tsm_disp.primary import Primary, stella_phiP, legacy_gaussian_norm, PHI_P_PER_LABEL_LEGACY, A_P
from tsm_disp.roots import find_root

BASE_A = ("/scratch/gpfs/rnies/2023-03-09_generalised_secondary/2023-03-09_toroidal_secondary/2023-06-13_QN_scans/"
          "2024-03-04_compare_with_GK_secondary")
BASE_B = ("/scratch/gpfs/rnies/2026-03-02_data_turbulence_saturation_by_propagating_ZFs_with_stellar_paths/"
          "simulation_data/stella_secondary-simulations/1D_scan_Fig1")


def load(d, names=("kr", "omega_r", "omega_i", "omega_r_std", "omega_i_std")):
    out = {}
    for n in names:
        f = os.path.join(d, "data_%s.dat" % n)
        if os.path.exists(f):
            out[n] = np.atleast_1d(np.loadtxt(f))
    return out


def old_code_curve(label):
    d = os.path.join(BASE_A, "PhiP-%.2f_alphaprp-3_alphapar-3_ALT_width0-3.1_rerun" % label)
    k, w = [], []
    for s in sorted(glob.glob(d + "/initial_guess_omegalin_*")):
        x = load(s, ("kr", "omega_r", "omega_i"))
        if "kr" not in x:
            continue
        m = np.isfinite(x["omega_i"]) & np.isfinite(x["omega_r"])
        k.append(x["kr"][m])
        w.append(x["omega_r"][m] + 1j * x["omega_i"][m])
    if not k:
        return None, None
    k, w = np.concatenate(k), np.concatenate(w)
    i = np.argsort(k)
    return k[i], w[i]


def ism_min(disp, omega, kx, n=96):
    """min |D_ISM| over a (z, y) grid at ω: ≲ 1e-2 signals the ISM region, where D is not analytic."""
    g = disp.geometry
    zs = np.linspace(g.zmin, g.zmax, n)
    ys = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return min(abs(disp.D_ism(omega, kx, z, y)) for z in zs for y in ys)


def solve_curve(disp, kx_grid, guesses, tol=1e-9):
    """Roots on kx_grid; guesses: callable kx -> initial ω (stella interpolation), refined by continuation."""
    om = np.full(len(kx_grid), np.nan + 1j * np.nan)
    D = np.full(len(kx_grid), np.nan + 1j * np.nan)
    ism = np.full(len(kx_grid), np.nan)
    t0 = time.time()
    prev = None
    nfail = 0
    for i, kx in enumerate(kx_grid):
        if nfail >= 2 and kx > 0.5:      # two failures in a row at large k_x: the ISM region, stop
            print("   kx=%.3f skipped (ISM region)" % kx, flush=True)
            continue
        cands = [guesses(kx)] + ([prev] if prev is not None else [])
        best = None
        for g in cands:
            if g is None or not np.isfinite(g):
                continue
            r = find_root(lambda w: disp.D(w, kx), g, tol=tol, maxiter=15)
            if r.converged and r.omega.imag > 0 and (best is None or abs(r.omega - g) < abs(best.omega - g)):
                best = r
            if best is not None:
                break
        if best is not None:
            om[i], D[i] = best.omega, best.D
            prev = best.omega
            ism[i] = ism_min(disp, best.omega, kx)
            nfail = 0
        else:
            nfail += 1
        print("   kx=%.3f -> %s  min|D_ISM|=%.2e (%.0fs)" % (kx, om[i], ism[i], time.time() - t0), flush=True)
    return om, D, ism


def case_A(nthreads, labels=(1, 2, 4)):
    res = {}
    fac = legacy_gaussian_norm(np.pi)
    deck = dict(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, kappa=1.0, nzed_local=256)
    gm = miller(**deck, zed_equal_arc=True)
    gc = CircularAnalytic(eps=0.5 / 2.7778, rmaj=2.7778, vMx=lambda th: np.sin(th) / 2.7778)
    for L in labels:
        d = os.path.join(BASE_A, "PhiP-%.2f_alphaprp-3_alphapar-3_ALT_width0-3.1_rerun" % L)
        st = {k: load(os.path.join(d, "secondary_equivalent_q-20" + s)) for k, s in
              [("ITG", "_ITG"), ("ITG_periodic", "_ITG_periodic"), ("periodic", "_periodic")]}
        vExP = PHI_P_PER_LABEL_LEGACY * L
        phiinit = vExP * 1000.0 / fac
        Phi_true = stella_phiP(phiinit, 1000.0, gm.exb_nonlin_fac, fac)
        ref = st["ITG"]
        kref, wref = ref["kr"], ref["omega_r"] + 1j * ref["omega_i"]
        guess = lambda kx: np.interp(kx, kref, wref.real) + 1j * np.interp(kx, kref, wref.imag)
        kx_grid = np.round(np.arange(0.05, 1.0001, 0.05), 4)
        out = dict(label=L, Phi_legacy=vExP, Phi_true=Phi_true, phiinit=phiinit, kx=kx_grid.tolist(),
                   A_P_eq21=A_P(Primary.gaussian(Phi_true, np.pi), gm, rmaj=2.77778),
                   stella={k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in st.items()})
        for name, geo, Phi in [("miller", gm, Phi_true), ("legacy", gc, vExP)]:
            print("A: label %g variant %s Φ_P = %.5f" % (L, name, Phi), flush=True)
            disp = DispersionRelation(geo, Primary.gaussian(Phi, np.pi), tau=1.0, eta_par=3, eta_prp=3,
                                      form="lambda", epsabs=1e-7, epsrel=1e-7, nthreads=nthreads, warn=False, limit=120)
            om, D, ism = solve_curve(disp, kx_grid, guess, tol=1e-7)
            out[name] = dict(omega_r=om.real.tolist(), omega_i=om.imag.tolist(), ism_min=ism.tolist())
        ko, wo = old_code_curve(L)
        if ko is not None:
            out["old_code"] = dict(kx=ko.tolist(), omega_r=np.abs(wo.real).tolist(), omega_i=wo.imag.tolist())
        res[L] = out
        json.dump(out, open(os.path.join(HERE, "data_caseA_label%g.json" % L), "w"), indent=1)
    return res


def case_B(nthreads):
    import f90nml
    st = load(BASE_B, ("kx", "omegar", "omegai"))
    nml = f90nml.read(os.path.join(BASE_B, "run_kxS-2.0000e-01", "CBC.in"))
    mg = nml["millergeo_parameters"]
    gm = miller(**mg, zed_equal_arc=bool(nml["zgrid_parameters"]["zed_equal_arc"]))
    ig = nml["init_g_knobs"]
    width0 = float(ig["width0"])
    Phi = stella_phiP(float(ig["phiinit"]), float(nml["kt_grids_box_parameters"]["y0"]), gm.exb_nonlin_fac)
    pri = Primary.gaussian(Phi, width0, normalise="peak")
    disp = DispersionRelation(gm, pri, tau=1.0, eta_par=float(ig.get("tpar0", 0)), eta_prp=float(ig.get("tperp0", 0)),
                              form="lambda", epsabs=1e-7, epsrel=1e-7, nthreads=nthreads, warn=False, limit=120)
    kx_grid = np.array(st["kx"])
    wst = st["omegar"] + 1j * st["omegai"]
    guess = lambda kx: np.interp(kx, kx_grid, wst.real) + 1j * np.interp(kx, kx_grid, wst.imag)
    print("B: Φ_P = %.5f, q = %g" % (Phi, mg["qinp"]), flush=True)
    om, D, ism = solve_curve(disp, kx_grid, guess, tol=1e-7)
    out = dict(Phi_true=Phi, q=mg["qinp"], kx=kx_grid.tolist(), stella=dict(omega_r=st["omegar"].tolist(),
               omega_i=st["omegai"].tolist()), miller=dict(omega_r=om.real.tolist(), omega_i=om.imag.tolist(), ism_min=ism.tolist()),
               A_P_eq21=A_P(pri, gm, rmaj=float(mg["rmaj"])))
    json.dump(out, open(os.path.join(HERE, "data_caseB.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nthreads", type=int, default=48)
    ap.add_argument("--case", default="AB")
    a = ap.parse_args()
    if "A" in a.case:
        case_A(a.nthreads)
    if "B" in a.case:
        case_B(a.nthreads)
