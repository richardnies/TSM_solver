"""q = 2.8 stella scan (case B, θ-constant primary, η∥ = 0, η⊥ = 3): purely growing root of the parallel-streaming
solver at small k_x, where the λ form (no streaming) has none. D(iγ) is real on the imaginary axis; roots by
bisection of its sign changes on a coarse γ grid. Real velocity contour (the tanh contour is not valid at ω_r = 0).
Writes data_caseB_streaming.json (plot with plot_benchmark.py). ≈ 35 min on 16 threads.
"""
import os, sys, time, json, warnings
import numpy as np
from scipy.optimize import brentq
R = "/scratch/gpfs/rnies/2023-03-09_generalised_secondary/2023-03-09_toroidal_secondary/TSM_solver"
sys.path.insert(0, R)
warnings.simplefilter("ignore")
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.streaming import StreamingDispersion, VelocityGrid
d = json.load(open(R + "/benchmarks/stella_cbc/data_caseB.json"))
gm = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=2.8, shat=0.8, kappa=1.0, nzed_local=64)
pri = Primary.gaussian(d["Phi_true"], 100.0, normalise="peak")
gams = np.array([0.004, 0.008, 0.012, 0.016, 0.02, 0.03, 0.045, 0.06])
out = dict(Phi_true=d["Phi_true"], gamma=gams.tolist(), runs={})
plan = [(nv, nu, kx) for kx in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12) for (nv, nu) in ((192, 24),)] + \
       [(384, 24, 0.04), (384, 24, 0.08), (96, 12, 0.02), (96, 12, 0.06), (96, 12, 0.10), (96, 12, 0.12), (96, 12, 0.04), (96, 12, 0.08)]
for nv, nu, kx in plan:
    sd = StreamingDispersion(gm, pri, tau=1, eta_par=0, eta_prp=3, n_y=32, grid=VelocityGrid(nv, nu), nthreads=16, Mphi=12)
    t0 = time.time()
    f = lambda g: sd.D(1j * g, kx).real
    Ds = np.array([f(g) for g in gams])
    k = np.where(np.sign(Ds[:-1]) != np.sign(Ds[1:]))[0]
    roots = [brentq(f, gams[j], gams[j + 1], xtol=1e-6) for j in k]
    out["runs"]["%d %d %g" % (nv, nu, kx)] = dict(nv=nv, nu=nu, kx=kx, D=Ds.tolist(), roots=roots, M=sd.last["M"])
    print("kx %.2f grid %dx%d: D(iγ) %s  roots γ = %s  (%.0f s, M %d)" % (kx, nv, nu, np.round(Ds, 4), np.round(roots, 5),
          time.time() - t0, sd.last["M"]), flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_caseB_streaming.json"), "w"))
