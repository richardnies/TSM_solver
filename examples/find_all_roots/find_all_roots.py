#!/usr/bin/env python3
"""Example: all roots of the dispersion function in a box of the complex ω plane, without an initial guess.

tsm_disp.contour_roots.find_all_roots samples 𝒟 on the box boundary until the phase is resolved, counts the zeros by
the argument principle, bisects the box until each cell holds a few zeros, locates them from the contour moments and
polishes and verifies them. Where 𝒟 is not analytic (the ISM region, D_ISM = 0 on the surface) the field-line
quadrature cannot converge; those samples are flagged and the cells reported instead of resolved.

Case: CBC (q = 20), Gaussian primary Φ_P = 0.723, τ = TAU, η∥ = ETA_PAR, η⊥ = ETA_PERP, λ form (no streaming),
k_x ρ_i = 0.5. With η∥ = η⊥ = 3 the box holds the TSM 0.5809 + 0.2927i just above the ISM region; try 1.5 (weaker
drive).

With TSM_solver installed (pip install ".[plot]" in the repository) this file runs from anywhere, also as a copy.
Edit the parameters below and run (≈ 2 min on 8 threads):

    python find_all_roots.py

Writes find_all_roots_eta<η∥>-<η⊥>_kx<k_x>.{txt,pdf} next to this script: the roots, the suspect regions, and a map of the
search (terminal cells, boundary samples, flagged samples) over the local ISM roots ω(z, y) of D_ISM = 0.
"""
import os
import time
import warnings
import numpy as np
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.contour_roots import find_all_roots, local_ism_roots
from evaluation_scripts.root_search import save_root_search, plot_root_search
from evaluation_scripts.plotting import drive_label

HERE = os.path.dirname(os.path.abspath(__file__))      # data and figures are written next to this script

# ---- parameters -------------------------------------------------------------------------------------------------
TAU = 1.0                   # T_i/(Z T_e) (modified adiabatic electrons)
ETA_PAR = 3.0               # primary's parallel temperature-to-density ratio η∥
ETA_PERP = 3.0              # primary's perpendicular temperature-to-density ratio η⊥
KX = 0.5                    # k_x ρ_i
PHI_P = 0.72314             # primary E×B amplitude (ρ_i v_th/a)
WR = (0.0, 1.2)             # ω_r range of the box (v_th/a)
WI = (0.02, 0.5)            # γ range of the box (v_th/a)
ISM_MAP = True              # also map the ISM region by the local roots of D_ISM (≈ 5 s)
NTHREADS = 8
# -----------------------------------------------------------------------------------------------------------------

geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, kappa=1.0, zed_equal_arc=True)
pri = Primary.gaussian(PHI_P, width=np.pi)
disp = DispersionRelation(geo, pri, tau=TAU, eta_par=ETA_PAR, eta_prp=ETA_PERP, nthreads=NTHREADS, warn=False)
t0 = time.time()
res = find_all_roots(disp, KX, WR, WI, verbose=True)
print(res.summary())
ism = None
if ISM_MAP:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ism = local_ism_roots(disp, KX)
    print("local ISM roots: %d, ω_r ∈ [%.3f, %.3f], γ ≤ %.3f" % (len(ism), ism[:, 2].min(), ism[:, 2].max(), ism[:, 3].max()))
stem = os.path.join(HERE, "find_all_roots_eta%g-%g_kx%g" % (ETA_PAR, ETA_PERP, KX))
save_root_search(stem + ".txt", res, "CBC q=20, Gaussian primary Phi_P=%g, tau=%g, eta_par=%g, eta_prp=%g, kx=%g, box %s x %s (v_th/a)"
                 % (PHI_P, TAU, ETA_PAR, ETA_PERP, KX, WR, WI))
plot_root_search(stem + ".pdf", __file__, res, ism, WR, WI,
                 r"$k_x\rho_i = %g$, " % KX + drive_label(ETA_PAR, ETA_PERP, TAU) + ": %d root(s), %d evaluations" % (len(res.roots), res.nfev))
print("wrote %s.{txt,pdf} (%.0f s)" % (stem, time.time() - t0))
