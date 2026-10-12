#!/usr/bin/env python3
"""Example: structure of a TSM on the flux surface — potential, pressure or nonlinear heat flux.

Same case as examples/tsm_kx_scan (CBC, q = 20, η∥ = η⊥ = 3, Gaussian primary, Φ_P = 0.723) at k_xρ_i = 0.5.
The root ω is found first; then, for each quantity, the figure shows
  left : Re[f(θ, y) e^{i k_x x}] at x = 0 on the flux surface (y: phase of the primary, θ: field line)
  right: the primary-phase average ⟨f⟩_y(θ) (real and imaginary parts)
with f = φ̂ (potential), P̂ = T∫d³v (v∥²+v⊥²/2)/v_th² ĝ (pressure, paper eq. 38) or Q̂_NL = v_Ex^P P̂ (nonlinear heat
flux, eq. 40), normalised to Z e ⟨φ⟩_ψ/T = 1.

With TSM_solver installed (pip install ".[plot]" in the repository) this file runs from anywhere, also as a copy.
Edit the parameters below and run:

    python mode_structure.py

Writes mode_<quantity>.{npz,pdf} next to this script.
"""
import os
import sys
import numpy as np
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from evaluation_scripts.mode_structure import save_mode_structure, plot_mode_structure

HERE = os.path.dirname(os.path.abspath(__file__))      # data and figures are written next to this script

# ---- parameters -------------------------------------------------------------------------------------------------
QUANTITIES = ("potential", "pressure", "heat_flux")      # any of these
KX = 0.5                    # k_x ρ_i
GUESS = 0.58 + 0.29j        # initial guess of ω (v_th/a)
PHI_P = 0.72314             # primary E×B amplitude (ρ_i v_th/a)
TAU = 1.0                   # T_i/(Z T_e) (modified adiabatic electrons)
ETA_PAR = 3.0               # primary's parallel temperature-to-density ratio η∥
ETA_PERP = 3.0              # primary's perpendicular temperature-to-density ratio η⊥
SAFETY_FACTOR = 20.0
NZ, NY = 161, 128           # resolution of the surface (field line × primary phase)
NTHREADS = min(os.cpu_count() or 1, 32)
# -----------------------------------------------------------------------------------------------------------------

geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=SAFETY_FACTOR, shat=0.8, zed_equal_arc=True)
disp = DispersionRelation(geo, Primary.gaussian(PHI_P, width=np.pi), tau=TAU, eta_par=ETA_PAR, eta_prp=ETA_PERP,
                          epsabs=1e-9, epsrel=1e-9, nthreads=NTHREADS)
r = disp.find_root(GUESS, KX, tol=1e-10)
if not r.converged:
    sys.exit("no root found near %s; change GUESS" % GUESS)
print("k_x = %g: omega = %.5f %+.5fi (v_th/a)" % (KX, r.omega.real, r.omega.imag))
for q in QUANTITIES:
    m = disp.mode_structure(r.omega, KX, q, nz=NZ, ny=NY)
    print("  %-9s <phi>_psi = %.6f%+.6fi (1 at the root)" % (q, m.phi_average.real, m.phi_average.imag))
    stem = os.path.join(HERE, "mode_%s" % q)
    save_mode_structure(stem + ".npz", m)
    plot_mode_structure(stem + ".pdf", __file__, m, q)
print("wrote %s/mode_*.{npz,pdf}" % HERE)
