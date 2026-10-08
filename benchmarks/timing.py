#!/usr/bin/env python3
"""Wall-clock comparison of code_eval_disp_NEW (old) and tsm_disp (new) on the same problems.

Problems (circular CBC model of the old code, Gaussian primary of width π, η∥ = η⊥ = 3):
  Z   : D at the regression point (ω = 0.96189+0.40884i, k_x = 0.5, Φ_P = 4·0.3557), Z form
  λ   : D at the TSM point ω = 0.578+0.290i, k_x = 0.5, Φ_P = 2·0.3557, λ form
  root: TSM root at k_x = 0.5 from the stella guess, λ form (old: scipy root 'lm' as in the drivers)
Accuracy is measured against a tight new-code reference (the old λ form also carries the complex-ive bug).
Writes timing.json next to this script.
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OLD = "/scratch/gpfs/rnies/2023-03-09_generalised_secondary/2023-03-09_toroidal_secondary/code_eval_disp_NEW"
import importlib.util
spec = importlib.util.spec_from_file_location("old_disp_rel", os.path.join(OLD, "disp_rel.py"))
sys.path.insert(1, OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import CircularAnalytic
from tsm_disp.primary import Primary
from tsm_disp.roots import find_root

FAC = 0.7733975858850816
T_P = lambda th: np.exp(-(th / np.pi)**2) / FAC
T_D = lambda th: np.sin(th) / 2.7778
geo = CircularAnalytic(eps=0.5 / 2.7778, vMx=lambda th: np.sin(th) / 2.7778)


def tm(f, n=1):
    t0 = time.time()
    for _ in range(n):
        v = f()
    return v, (time.time() - t0) / n


def new(Phi, form, tol, nthreads=1):
    return DispersionRelation(geo, Primary.gaussian(Phi), eta_par=3, eta_prp=3, form=form, epsabs=tol, epsrel=tol,
                              nthreads=nthreads, warn=False)


rows = []
wZ = 9.618900159238024106e-01 + 1j * 4.088425624787327406e-01
ref = new(4 * 0.3557, "Z", 1e-13).D(wZ, 0.5)
for eps in (1e-6, 1e-8):
    v, t = tm(lambda: old.evaluate_disp_quasineutrality(omega=wZ, kr=0.5, Phi_P=4 * 0.3557, T_P=T_P, T_D=T_D, alpha_par=3,
                                                         alpha_prp=3, epsrel_theta=eps, epsabs_theta=eps, epsrel_y=eps,
                                                         epsabs_y=eps, disp_model=0.1))
    rows.append(dict(problem="Z", code="old", setting="eps %g" % eps, seconds=t, error=abs(v - ref)))
for tol in (1e-8, 1e-10):
    d = new(4 * 0.3557, "Z", tol)
    v, t = tm(lambda: d.D(wZ, 0.5), 3)
    rows.append(dict(problem="Z", code="new", setting="tol %g, 1 thread" % tol, seconds=t, error=abs(v - ref)))

wL = 0.578 + 0.290j
ref = new(2 * 0.3557, "lambda", 1e-12, 48).D(wL, 0.5)
for eps, ta in ((1e-3, 1e-4), (1e-6, 1e-4), (1e-6, 1e-8)):
    v, t = tm(lambda: old.evaluate_disp_quasineutrality(omega=wL, kr=0.5, Phi_P=2 * 0.3557, T_P=T_P, T_D=T_D, alpha_par=3,
                                                         alpha_prp=3, epsrel_theta=eps, epsabs_theta=eps, epsrel_y=eps,
                                                         epsabs_y=eps, disp_model=-0.1, Nintegrate=1000, a_int=10,
                                                         tol_adaptive=ta))
    rows.append(dict(problem="lambda", code="old", setting="eps %g, tol_adaptive %g" % (eps, ta), seconds=t, error=abs(v - ref)))
for tol, nt in ((1e-6, 1), (1e-8, 1), (1e-8, 48)):
    d = new(2 * 0.3557, "lambda", tol, nt)
    v, t = tm(lambda: d.D(wL, 0.5), 2)
    rows.append(dict(problem="lambda", code="new", setting="tol %g, %d threads" % (tol, nt), seconds=t, error=abs(v - ref)))

# root finding at kx = 0.5 from the stella value
dref = new(2 * 0.3557, "lambda", 1e-11, 48)
rref = find_root(lambda w: dref.D(w, 0.5), 0.579 + 0.292j, tol=1e-12).omega


def old_root():
    f = lambda x: (lambda D: [D.real, D.imag])(old.evaluate_disp_quasineutrality(
        omega=x[0] + 1j * x[1], kr=0.5, Phi_P=2 * 0.3557, T_P=T_P, T_D=T_D, alpha_par=3, alpha_prp=3,
        epsrel_theta=1e-6, epsabs_theta=1e-6, epsrel_y=1e-6, epsabs_y=1e-6, disp_model=-0.1, Nintegrate=1000,
        a_int=10, tol_adaptive=1e-4))
    r = root(f, [0.579, 0.292], method="lm", tol=1e-6, options={"maxiter": 250})
    return r.x[0] + 1j * r.x[1]


v, t = tm(old_root)
rows.append(dict(problem="root", code="old", setting="lm, eps 1e-6, tol_adaptive 1e-4", seconds=t, error=abs(v - rref), omega=str(v)))
for tol, nt in ((1e-8, 1), (1e-8, 48)):
    d = new(2 * 0.3557, "lambda", tol, nt)
    r, t = tm(lambda: find_root(lambda w: d.D(w, 0.5), 0.579 + 0.292j, tol=1e-9))
    rows.append(dict(problem="root", code="new", setting="Muller, tol %g, %d threads" % (tol, nt), seconds=t,
                     error=abs(r.omega - rref), omega=str(r.omega), nfev=r.nfev))
for r in rows:
    print("%-7s %-4s %-38s %9.3f s   error %.1e" % (r["problem"], r["code"], r["setting"], r["seconds"], r["error"]))
json.dump(dict(rows=rows, root_reference=str(rref)), open(os.path.join(HERE, "timing.json"), "w"), indent=1)
