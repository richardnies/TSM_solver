#!/usr/bin/env python3
"""Example: the TSM at fixed k_x as the primary amplitude Φ_P is scanned up and down.

Same case as examples/tsm_kx_scan (CBC, q = 20, Gaussian primary of width π, η∥ = η⊥ = 3, τ = 1), at k_xρ_i = 0.3.
From the example amplitude Φ_P = 0.723 the root is followed

  upward   to 100× larger Φ_P: once ω_RDK ≫ ω_GAM the radial magnetic drift no longer matters and the TSM becomes the
           purely growing RDK secondary mode (Nies & Parra 2026, eq. 51): ω/(k_x Φ_P) tends to the root of the same
           dispersion relation without drift (dashed), close to eq. (51)'s long-wavelength value (dotted);
  downward towards the ISM region (local zeros of D_ISM on the surface), where D is not analytic and the TSM is
           lost: min|D_ISM| at the root falls linearly to zero. The scan stops once min|D_ISM| < ISM_STOP, shortly
           before the region: there the field-line quadrature stops converging and each evaluation of D gets slow
           (a root search inside the region costs about a minute). The amplitude where the region is reached is the
           linear extrapolation of min|D_ISM| through the last three roots (≈ 0.23 for the defaults, against
           0.222–0.234 from a guess-free search).

With CONFIRM_END the end of the downward branch is also checked without a guess (tsm_disp.contour_roots.find_all_roots,
≈ 2 more minutes): one step below the last root, no root may remain near the branch.

With TSM_solver installed (pip install ".[plot]" in the repository) this file runs from anywhere, also as a copy.
Edit the parameters below and run (≈ 15 s on 16 cores):

    python amplitude_scan.py

Writes amplitude_scan.{txt,pdf} next to this script.
"""
import os
import numpy as np
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.limits import omega_GAM
from evaluation_scripts.amplitude_scan import (WithoutDrift, follow_amplitude, rdk_ratio, rdk_ratio_long_wavelength,
                                               ism_amplitude, save_amplitude_scan, plot_amplitude_scan)

HERE = os.path.dirname(os.path.abspath(__file__))      # data and figures are written next to this script

# ---- parameters -------------------------------------------------------------------------------------------------
SAFETY_FACTOR = 20.0        # q of the CBC surface
KX = 0.3                    # k_x ρ_i
PHI_P = 0.72314             # starting primary E×B amplitude (ρ_i v_th/a) ...
GUESS = 0.336 + 0.204j      # ... and the initial guess of ω (v_th/a) there
TAU = 1.0                   # T_i/(Z T_e) (modified adiabatic electrons)
ETA_PAR = 3.0               # primary's parallel temperature-to-density ratio η∥
ETA_PERP = 3.0              # primary's perpendicular temperature-to-density ratio η⊥
PHI_MAX = 100 * PHI_P       # largest amplitude of the upward scan
STEP_UP = 1.25              # Φ_P factor per step upward
STEP_DOWN = 0.95            # Φ_P factor per step downward
ISM_STOP = 0.03             # stop the downward scan after a root with min|D_ISM| below this (smaller: slower)
CONFIRM_END = False         # guess-free search one step below the last root of the downward scan (≈ 2 min)
TOL = 1e-8                  # tolerance of the field-line integral
NTHREADS = min(os.cpu_count() or 1, 32)
# -----------------------------------------------------------------------------------------------------------------

geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=SAFETY_FACTOR, shat=0.8, kappa=1.0, zed_equal_arc=True)
pri = Primary.gaussian(PHI_P, width=np.pi)


def make(Phi, geometry=geo):
    return DispersionRelation(geometry, pri.with_amplitude(Phi), tau=TAU, eta_par=ETA_PAR, eta_prp=ETA_PERP,
                              epsabs=TOL, epsrel=TOL, nthreads=NTHREADS, warn=False)


# 1. upward: Φ_P → 100 Φ_P (geometric steps)
n_up = int(np.ceil(np.log(PHI_MAX / PHI_P) / np.log(STEP_UP)))
up = PHI_P * STEP_UP ** np.arange(n_up + 1)
print("upward scan, k_x = %g" % KX)
phi_u, om_u, ism_u = follow_amplitude(make, KX, up, GUESS)
# 2. downward from Φ_P until the root reaches the ISM region (min|D_ISM| is computed at every root)
down = PHI_P * STEP_DOWN ** np.arange(0, 200)
print("downward scan")
phi_d, om_d, ism_d = follow_amplitude(make, KX, down, om_u[0], ism_stop=ISM_STOP)
phi_ism = ism_amplitude(phi_d, ism_d)
print("min|D_ISM| extrapolates to zero at Phi_P = %.4f" % phi_ism)
end_note = ""
if CONFIRM_END:
    from tsm_disp.contour_roots import find_all_roots
    a = phi_d[-1] * STEP_DOWN
    g = om_d[-1] * STEP_DOWN
    box = ((0.6 * g.real, 1.4 * g.real), (0.005, 1.5 * om_d[-1].imag))
    res = find_all_roots(make(a), KX, *box)
    near = [q for q in res.roots if abs(q.omega - g) < 0.2 * abs(g)]
    end_note = "guess-free search at Phi_P = %.4f in %s: %d root(s) near the branch, %d suspect (ISM) cell(s)" % (
        a, np.round(box, 4).tolist(), len(near), len(res.suspect_regions))
    print(end_note)
    print(res.summary())
# 3. RDK limit: the same dispersion relation without drift (root ∝ k_x Φ_P), and eq. (51)
s_exact = rdk_ratio(make(phi_u[-1], WithoutDrift(geo)), KX, om_u[-1])
s_long = rdk_ratio_long_wavelength(geo, pri, KX, TAU, ETA_PERP)
w_gam = omega_GAM(geo, KX, TAU)
phi_gam = w_gam / (KX * abs(s_exact))
print("RDK: omega/(k_x Phi_P) = %.5f%+.5fi without drift, %.5f%+.5fi from eq. (51); last root %.5f%+.5fi; "
      "omega_RDK = omega_GAM (%.4f) at Phi_P = %.3f" % (s_exact.real, s_exact.imag, s_long.real, s_long.imag,
                                                        (om_u[-1] / (KX * phi_u[-1])).real, (om_u[-1] / (KX * phi_u[-1])).imag,
                                                        w_gam, phi_gam))

# 4. data and figure next to this script
phi = np.concatenate([phi_d[::-1], phi_u[1:]])
omega = np.concatenate([om_d[::-1], om_u[1:]])
ism = np.concatenate([ism_d[::-1], ism_u[1:]])
stem = os.path.join(HERE, "amplitude_scan")
save_amplitude_scan(stem + ".txt", phi, omega, ism,
                    "CBC q=%g, Gaussian primary, kx=%g, tau=%g, eta_par=%g, eta_prp=%g\n"
                    "RDK omega/(kx Phi_P): %s (no drift), %s (eq. 51); omega_GAM = %.5f\n"
                    "min|D_ISM| -> 0 at Phi_P = %.5f; %s" % (SAFETY_FACTOR, KX, TAU, ETA_PAR, ETA_PERP, np.round(s_exact, 6),
                                                          np.round(s_long, 6), w_gam, phi_ism, end_note))
plot_amplitude_scan(stem + ".pdf", __file__, phi, omega, ism, KX, s_exact, s_long, phi_gam, phi_ism,
                    case_title=r"CBC, $q = %g$" % SAFETY_FACTOR, eta_par=ETA_PAR, eta_prp=ETA_PERP, tau=TAU)
print("wrote %s.{txt,pdf}" % stem)
