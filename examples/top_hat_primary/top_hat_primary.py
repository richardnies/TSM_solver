#!/usr/bin/env python3
"""Example: TSM against k_x for an analytically prescribed geometry and a top-hat primary.

Geometry given as functions of θ (AnalyticGeometry): radial magnetic drift v_Mx = sin θ, |∇x| = 1, B = 1 (uniform
weight along θ). Primary (Primary.top_hat): Ftheta_P = 1 for θ ∈ [θ_TH − w_TH/2, θ_TH + w_TH/2] and 0 otherwise, with E×B
amplitude Φ_P; τ = 1 and η∥ = η⊥ = 3 by default (TAU, ETA_PAR, ETA_PERP). The two edges of the top hat are panel edges of the θ integral, so the
discontinuities cost nothing.

With TSM_solver installed (pip install ".[plot]" in the repository) this file runs from anywhere, also as a copy.
Edit the parameters below and run (≈ 20 s on 32 cores, a few minutes on 1):

    python top_hat_primary.py

Other cases: CENTER_TOPHAT, WIDTH_TOPHAT, PHI_P, GUESS = 0.6, 1.2, 6, 1.311 + 0.925j; or the amplitude from the heat flux,
HEAT_FLUX, KY_P, DELTA = 9708, 0.001, 0.5 (Φ_P = 3). Writes top_hat_primary.{txt,pdf} next to this script. Units: k_x
in 1/ρ, ω in v_th/a (lengths in a, so v_Mx = sin θ means a drift scale R = a). The scan stops where the root
reaches the ISM region (default case: k_x ρ ≈ 0.6). For other parameters, give a guess of ω at KX_START (a growing
root; scan |D(ω)|, use tsm_disp.contour_roots.find_all_roots, or continue from a nearby case).
"""
import os
import sys
import time
import numpy as np
from tsm_disp.geometry import AnalyticGeometry
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from evaluation_scripts.kx_scan import follow_kx_adaptive, save_kx_scan, plot_kx_scan
from evaluation_scripts.plotting import drive_label

HERE = os.path.dirname(os.path.abspath(__file__))      # data and figures are written next to this script

# ---- parameters -------------------------------------------------------------------------------------------------
CENTER_TOPHAT = 0.0         # centre θ_TH of the top hat
WIDTH_TOPHAT = np.pi        # width w_TH of the top hat (≤ 2π)
PHI_P = 3.0                 # primary E×B amplitude Φ_P (ρ v_th/a)
TAU = 1.0                   # T_i/(Z T_e) (modified adiabatic electrons)
ETA_PAR = 3.0               # primary's parallel temperature-to-density ratio η∥
ETA_PERP = 3.0              # primary's perpendicular temperature-to-density ratio η⊥
HEAT_FLUX = None            # set the amplitude instead by the primary's ion heat flux in stella's gyro-Bohm units
                            #   (needs DELTA ≠ 0; see tsm_disp.primary.heat_flux_stella)
KY_P = 1e-3                 # primary binormal wavenumber (for HEAT_FLUX; stella 1/y0)
DELTA = 0.0                 # phase Δ∥ = Δ⊥ of the primary temperature relative to φ (stella: −den0_phase)
GRHO = 1.0                  # |∇ρ| for the flux normalisation (for HEAT_FLUX)
EXB_NONLIN_FAC = 0.5        # stella's exb_nonlin_fac relating φ to v_Ex (for HEAT_FLUX; ½ for dx/dψ dy/dα = 1)
KX_MIN, KX_MAX, DKX = 0.05, 1.0, 0.05                    # range and (largest) step of the k_x continuation
KX_START = 0.3              # k_x at which the scan starts ...
GUESS = 1.054 + 0.662j      # ... and the initial guess of ω (v_th/a) there
TOL = 1e-6                  # tolerance of the θ integral
NTHREADS = min(os.cpu_count() or 1, 32)
# -----------------------------------------------------------------------------------------------------------------

# 1. geometry from functions of θ: v_Mx = sin θ, |∇x| = 1 (B = 1, uniform weight by default)
geo = AnalyticGeometry(vMx=np.sin, gradx=1.0, grho=GRHO, exb_nonlin_fac=EXB_NONLIN_FAC, sign_torflux=-1)

# 2. primary: v_Ex = −Φ_P Ftheta_P(θ) sin y with Ftheta_P the top hat
pri = Primary.top_hat(CENTER_TOPHAT, WIDTH_TOPHAT, PHI_P)
if HEAT_FLUX is not None:   # amplitude from the primary heat flux (stella's qflx normalisation)
    pri = pri.with_heat_flux(HEAT_FLUX, geo, KY_P, eta_par=ETA_PAR, eta_prp=ETA_PERP, delta_par=DELTA, delta_prp=DELTA)
    print("Q = %g, k_y^P = %g, Δ = %g  ->  Phi_P = %.5f" % (HEAT_FLUX, KY_P, DELTA, pri.Phi_P))

# 3. dispersion relation (λ form: exact finite-Larmor-radius effects)
disp = DispersionRelation(geo, pri, tau=TAU, eta_par=ETA_PAR, eta_prp=ETA_PERP, delta_par=DELTA, delta_prp=DELTA,
                          form="lambda", epsabs=TOL, epsrel=TOL, nthreads=NTHREADS, warn=False)

# 4. root at KX_START, then adaptive continuation down and up in k_x (stops at the ISM region)
t0 = time.time()
r = disp.find_root(GUESS, KX_START, tol=1e-8, maxiter=40)
if not r.converged:
    sys.exit("no root found near %s at kx = %g; change GUESS" % (GUESS, KX_START))
print("kx = %.4f   omega = %+.5f %+.5fi   (start)" % (KX_START, r.omega.real, r.omega.imag))
rows = sorted(follow_kx_adaptive(disp, KX_START, r.omega, KX_MIN, KX_MAX, DKX, -1) + [(KX_START, r.omega)]
              + follow_kx_adaptive(disp, KX_START, r.omega, KX_MIN, KX_MAX, DKX, +1), key=lambda q: q[0])
kx = np.array([q[0] for q in rows])
omega = np.array([q[1] for q in rows])

# 5. data and figure next to this script
stem = os.path.join(HERE, "top_hat_primary")
save_kx_scan(stem + ".txt", kx, omega, "vMx = sin(theta), |grad x| = 1, top-hat primary center_tophat=%g width_tophat=%g Phi_P=%g, "
             "eta_par=%g, eta_prp=%g, Delta=%g, tau=%g%s" % (CENTER_TOPHAT, WIDTH_TOPHAT, pri.Phi_P, ETA_PAR, ETA_PERP, DELTA, TAU,
                                                     "" if HEAT_FLUX is None else ", from Q=%g at ky_P=%g" % (HEAT_FLUX, KY_P)))
th = np.linspace(-np.pi, np.pi, 2001)
amp = (r"$\Phi_P = %.3g$" % pri.Phi_P if HEAT_FLUX is None
       else r"$Q = %g$ ($k_y^P = %g$, $\Phi_P = %.3g$)" % (HEAT_FLUX, KY_P, pri.Phi_P))
plot_kx_scan(stem + ".pdf", __file__, kx, omega, scale=1.0, unit=r"a/v_{th}", xlabel=r"$k_x \rho$", markers=True,
             xlim=(0, KX_MAX), ylim_zero=True,
             titles=(r"$v_{Mx} = \sin\theta$, $|\nabla x| = 1$, " + drive_label(ETA_PAR, ETA_PERP, TAU)
                     + (r", $\Delta = %g$" % DELTA if DELTA else ""),
                     r"top hat $\theta_{\rm TH} = %.3g$, $w_{\rm TH} = %.3g$, " % (CENTER_TOPHAT, WIDTH_TOPHAT) + amp),
             title_fontsize=20,
             profiles=[(th, pri(th), "-", "C2", r"$F_{\theta,P}(\theta)$"), (th, geo.vMx(th), "--", "C1", r"$v_{Mx} = \sin\theta$"),
                       (th, geo.gradx(th), ":", "k", r"$|\nabla x| = 1$")])
print("wrote %s.{txt,pdf} (%.0f s)" % (stem, time.time() - t0))
