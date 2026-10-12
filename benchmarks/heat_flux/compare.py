#!/usr/bin/env python3
"""Compare stella's primary heat flux (runs/*/CBC.fluxes, last line) with tsm_disp.primary.heat_flux_stella.

The solver primary is built from the deck exactly as for the secondary benchmark: Miller geometry from
&millergeo_parameters (equal-arc z), Φ_P from stella_phiP(phiinit, y0, exb_nonlin_fac, legacy fac_norm),
Ftheta_P = Gaussian of width width0, η = tpar0, tperp0, Δ∥ = Δ⊥ = −den0_phase, k_y^P = 1/y0.
Usage: python compare.py  (after make_decks.py and run_decks.sh)
"""
import glob
import os
import sys
import numpy as np
import f90nml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary, stella_phiP, legacy_gaussian_norm, heat_flux_stella

print("%-26s %14s %14s %10s" % ("case", "stella qflx", "formula", "rel. diff"))
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    nml = f90nml.read(os.path.join(d, "CBC.in"))
    m, ini, box = nml["millergeo_parameters"], nml["init_g_knobs"], nml["kt_grids_box_parameters"]
    geo = miller(**{k: (int(v) if k == "nzed_local" else float(v)) for k, v in m.items()}, zed_equal_arc=True)
    width, y0 = float(ini["width0"]), float(box["y0"])
    Phi = stella_phiP(float(ini["phiinit"]), y0, geo.exb_nonlin_fac, fac_norm=legacy_gaussian_norm(width))
    pri = Primary.gaussian(Phi, width=width)
    ph = -float(ini.get("den0_phase", 0.0))
    Q = heat_flux_stella(pri, geo, 1.0 / y0, float(ini["tpar0"]), float(ini["tperp0"]), ph, ph)
    qs = float(open(os.path.join(d, "CBC.fluxes")).read().split()[-1])
    rel = (Q - qs) / abs(qs) if abs(qs) > 1e-8 else np.nan
    print("%-26s %14.6e %14.6e %10.2e" % (os.path.basename(d), qs, Q, rel))
