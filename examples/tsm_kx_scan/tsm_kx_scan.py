#!/usr/bin/env python3
"""Example: frequency and growth rate of the toroidal secondary mode (TSM) against k_x.

Typical case of Nies & Parra (2026), Fig. 3: Cyclone Base Case surface (circular, r/a = 0.5, R/a = 2.778, ŝ = 0.8,
q = 20), modified adiabatic electrons with T_i/T_e = 1, a frozen streamer primary with bi-Maxwellian
temperature-to-density ratios η∥ = η⊥ = 3 (ETA_PAR, ETA_PERP), Gaussian along the field line (width π), and E×B amplitude
Φ_P = 0.723 ρ_i v_th/a (the label 𝒜 = 2 of the paper's figures).

With TSM_solver installed (pip install ".[plot]" in the repository) this file runs from anywhere, also as a copy.
Edit the parameters below and run (≈ 10–60 s depending on the number of cores; ≈ 6 min on 16 cores with
SOLVER = "streaming"):

    python tsm_kx_scan.py

Writes tsm_kx_scan.{txt,pdf} (tsm_kx_scan_streaming.* with SOLVER = "streaming") next to this script. Units: k_x in
1/ρ_i, frequencies in v_th/a with v_th = √(2T/m) (stella units); the figure shows ω R/v_th as in the paper, with
stella gyrokinetic simulations of the same case (Θ = 0.2886 runs) as crosses when the parameters are the defaults.
"""
import os
import numpy as np
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from evaluation_scripts.kx_scan import follow_kx, save_kx_scan, plot_kx_scan
from evaluation_scripts.plotting import text, drive_label

HERE = os.path.dirname(os.path.abspath(__file__))      # data and figures are written next to this script

# ---- parameters -------------------------------------------------------------------------------------------------
SAFETY_FACTOR = 20.0        # q of the CBC surface
PHI_P = 0.72314             # primary E×B amplitude (ρ_i v_th/a)
TAU = 1.0                   # T_i/(Z T_e) (modified adiabatic electrons)
ETA_PAR = 3.0               # primary's parallel temperature-to-density ratio η∥
ETA_PERP = 3.0              # primary's perpendicular temperature-to-density ratio η⊥
HEAT_FLUX = None            # set the amplitude instead by the primary's ion heat flux in stella's gyro-Bohm units
                            #   (needs DELTA ≠ 0; e.g. HEAT_FLUX = 1159, DELTA = 0.5 gives Φ_P = 0.723)
KY_P = 1e-3                 # primary binormal wavenumber (for HEAT_FLUX; stella 1/y0)
DELTA = 0.0                 # phase Δ∥ = Δ⊥ of the primary temperature relative to φ (stella: −den0_phase)
SOLVER = "lambda"           # "lambda": eq. (32), no parallel streaming (fast); "streaming": field-line Fourier solver
                            #   with ion parallel streaming (tsm_disp.streaming), each root seeded by the λ-form root
KX = np.round(np.arange(0.1, 0.8 + 1e-9, 0.05), 10)     # k_x ρ_i grid
KX_START = 0.5              # k_x at which the scan starts ...
GUESS = 0.58 + 0.29j        # ... and the initial guess of ω (v_th/a) there
TOL = 1e-6                  # tolerance of the field-line integral (λ form)
NTHREADS = min(os.cpu_count() or 1, 32)
# -----------------------------------------------------------------------------------------------------------------

# stella secondary-mode simulations of this case (𝒜 = 2, Θ = 0.2886), ω in v_th/a
STELLA_KX = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
STELLA_OMEGA = np.array([0.1043 + 0.0675j, 0.2168 + 0.1374j, 0.3347 + 0.2031j, 0.4547 + 0.2575j,
                         0.5793 + 0.2922j, 0.7130 + 0.3047j, 0.8578 + 0.3003j, 1.0091 + 0.2865j])

# 1. geometry: Miller parametrisation of the CBC surface on stella's equal-arc coordinate
geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=SAFETY_FACTOR, shat=0.8, kappa=1.0, zed_equal_arc=True)

# 2. primary: v_Ex = −Φ_P Ftheta_P(z) sin y with Ftheta_P a Gaussian of width π normalised to unit RMS
pri = Primary.gaussian(PHI_P, width=np.pi)
if HEAT_FLUX is not None:   # amplitude from the primary heat flux (stella's qflx normalisation)
    pri = pri.with_heat_flux(HEAT_FLUX, geo, KY_P, eta_par=ETA_PAR, eta_prp=ETA_PERP, delta_par=DELTA, delta_prp=DELTA)
    print("Q = %g, k_y^P = %g, Δ = %g  ->  Phi_P = %.5f" % (HEAT_FLUX, KY_P, DELTA, pri.Phi_P))

# 3. dispersion relation (λ form: exact finite-Larmor-radius effects; Newton with the exact ∂D/∂ω)
disp = DispersionRelation(geo, pri, tau=TAU, eta_par=ETA_PAR, eta_prp=ETA_PERP, delta_par=DELTA, delta_prp=DELTA,
                          form="lambda", epsabs=TOL, epsrel=TOL, nthreads=NTHREADS, warn=False)
streaming_solver = None
if SOLVER == "streaming":   # retains v∥ b·∇: matters at moderate q (2–3× closer to stella at q = 20)
    from tsm_disp.streaming import StreamingDispersion
    streaming_solver = StreamingDispersion(geo, pri, tau=TAU, eta_par=ETA_PAR, eta_prp=ETA_PERP, delta_par=DELTA,
                                           delta_prp=DELTA, nthreads=NTHREADS)

# 4. follow the root from KX_START up and down by continuation
omega = follow_kx(disp, KX, KX_START, GUESS, streaming_solver=streaming_solver)   # λ-form root, then streaming root

# 5. data and figure next to this script
stem = os.path.join(HERE, "tsm_kx_scan" + ("_streaming" if SOLVER == "streaming" else ""))
R = geo.rmaj
save_kx_scan(stem + ".txt", KX, omega, "CBC q=%g, tau=%g, eta_par=%g, eta_prp=%g, Gaussian primary Phi_P=%g" % (SAFETY_FACTOR, TAU, ETA_PAR, ETA_PERP, pri.Phi_P),
             R=R)
default = (abs(SAFETY_FACTOR - 20) < 1e-12 and abs(pri.Phi_P - 0.72314) < 1e-4 and DELTA == 0 and TAU == 1
           and ETA_PAR == ETA_PERP == 3)
plot_kx_scan(stem + ".pdf", __file__, KX, omega, scale=R, unit=r"R/v_{th}",
             label=text("TSM_solver") + (" (streaming)" if SOLVER == "streaming" else " (theory)"),
             reference=(STELLA_KX, STELLA_OMEGA, "stella (gyrokinetic)") if default else None,
             titles=(r"CBC, $q = %g$, " % SAFETY_FACTOR + drive_label(ETA_PAR, ETA_PERP, TAU),
                     r"Gaussian primary, $\Phi_P = %.3f$" % pri.Phi_P))
print("wrote %s.{txt,pdf}" % stem)
