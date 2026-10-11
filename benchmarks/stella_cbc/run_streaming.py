#!/usr/bin/env python3
"""Parallel-streaming solver against the stella CBC benchmark (q = 20, labels 𝒜 = 1, 2, 4).

For each stella k_x the root of tsm_disp.streaming.StreamingDispersion (CBC Miller surface, equal-arc z, Gaussian
primary of width π at the true stella amplitude, η∥ = η⊥ = 3, default truncation M_φ = 16, M ≥ 24, velocity grid
96 × 12, n_y = 48) is found from the λ-form root of data_caseA_label*.json (no streaming). Writes
data_caseA_streaming_label*.json. Usage: python run_streaming.py [--nthreads 24] [--labels 1,2,4]
"""
import argparse
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.streaming import StreamingDispersion, VelocityGrid

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nthreads", type=int, default=24)
    ap.add_argument("--labels", default="1,2,4")
    a = ap.parse_args()
    gm = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, kappa=1.0, nzed_local=256, zed_equal_arc=True)
    for L in [int(x) for x in a.labels.split(",")]:
        d = json.load(open(os.path.join(HERE, "data_caseA_label%d.json" % L)))
        st = d["stella"]["ITG"]
        k = np.array(d["kx"])
        lam = np.array(d["miller"]["omega_r"], float) + 1j * np.array(d["miller"]["omega_i"], float)
        sd = StreamingDispersion(gm, Primary.gaussian(d["Phi_true"], np.pi), tau=1, eta_par=3, eta_prp=3, n_y=48,
                                 grid=VelocityGrid(96, 12), nthreads=a.nthreads)
        out = dict(label=L, Phi_true=d["Phi_true"], kx=[], omega_r=[], omega_i=[], M=[], seconds=[])
        for kx in st["kr"]:
            i = int(np.argmin(abs(k - kx)))
            if abs(k[i] - kx) > 1e-6 or not np.isfinite(lam[i]):
                continue
            t0 = time.time()
            r = sd.find_root(lam[i], kx, tol=1e-8, maxiter=30)
            if not r.converged:
                print("label %d kx %.2f: no convergence" % (L, kx), flush=True)
                continue
            out["kx"].append(kx)
            out["omega_r"].append(r.omega.real)
            out["omega_i"].append(r.omega.imag)
            out["M"].append(sd.last["M"])
            out["seconds"].append(time.time() - t0)
            print("label %d kx %.2f: streaming %s  (λ-form %s, %d evals, %.0f s)" % (
                L, kx, np.round(r.omega, 5), np.round(lam[i], 5), r.nfev, time.time() - t0), flush=True)
            json.dump(out, open(os.path.join(HERE, "data_caseA_streaming_label%d.json" % L), "w"), indent=1)
