"""Drop-in replacement for code_eval_disp_NEW/disp_rel.py, backed by tsm_disp.

Old drivers keep working after changing their sys.path line to this directory:

    sys.path.append('/scratch/gpfs/rnies/2023-03-09_generalised_secondary/2023-03-09_toroidal_secondary/tsm_disp')
    import disp_rel as disp

Differences from the original (see README.md):
* the λ form (disp_model < 0) evaluates Γ0(b̃) correctly for complex b̃ (the original used scipy's
  complex `ive`, which is e^{−|Re z|} I0(z)): λ-form results change at the 1e-3 level;
* a_int, Nintegrate, tol_adaptive, sing_ITG are accepted and ignored (the velocity integral is
  adaptive with tolerance TOL_N);
* for Im ω ≤ 0 the primary-phase integral is taken on a deformed contour (Y_CONTOUR = 'auto'),
  the analytic continuation of D; set Y_CONTOUR = 'real_points' for the original prescription;
* tolerances below 1e-13 are raised to 1e-13; θ must be real.
"""
import os
import warnings
import numpy as np

from tsm_disp.dispersion import DispersionRelation
from tsm_disp.geometry import CircularAnalytic
from tsm_disp.primary import Primary
from tsm_disp import limits as _limits

Y_CONTOUR = "auto"
TOL_N = 1e-12
NTHREADS = int(os.environ.get("TSM_DISP_NTHREADS", "1"))
_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, DeprecationWarning, stacklevel=3)


def T_k(theta):
    return 1


def _form(disp_model):
    if disp_model < 0:
        return "lambda"
    if disp_model < 0.5:
        return "Z"
    raise ValueError("disp_model >= 0.5 selected reduced models that were removed from the original code")


def _vectorise(f):
    def g(z):
        z = np.asarray(z)
        if z.ndim == 0:
            return float(np.real(f(float(z))))
        out = np.empty(z.shape)
        for i, zi in np.ndenumerate(z):
            out[i] = np.real(f(float(zi)))
        return out
    return g


def _maybe_vector(f):
    """Use f directly if it is vectorised (accepts arrays), else wrap it."""
    try:
        v = f(np.array([0.1, 0.2]))
        if np.shape(v) == (2,):
            return lambda z: np.real(f(z))
        if np.ndim(v) == 0:
            return lambda z: np.broadcast_to(np.real(f(z)), np.shape(z)).astype(float) if np.ndim(z) else float(np.real(f(z)))
    except Exception:
        pass
    return _vectorise(f)


def _disp(Phi_P, T_P, T_D, alpha_par, alpha_prp, Delta_par, Delta_prp, tau, eps_AR, disp_model,
          epsabs, epsrel, epsabs_y, epsrel_y, asymp_J11, a_int, Nintegrate, tol_adaptive, sing_ITG):
    if sing_ITG:
        _warn_once("sing", "sing_ITG is ignored by tsm_disp")
    geo = CircularAnalytic(eps=eps_AR, vMx=_maybe_vector(T_D))
    pri = Primary(_maybe_vector(T_P), Phi_P)
    return DispersionRelation(geo, pri, tau=tau, eta_par=alpha_par, eta_prp=alpha_prp, delta_par=Delta_par,
                              delta_prp=Delta_prp, form=_form(disp_model), y_contour=Y_CONTOUR,
                              epsabs=max(epsabs, 1e-13), epsrel=max(epsrel, 1e-13),
                              epsabs_y=max(epsabs_y, 1e-13), epsrel_y=max(epsrel_y, 1e-13), tolN=TOL_N,
                              nthreads=NTHREADS, asymp_J11=(asymp_J11 > 0), warn=False)


def evaluate_disp_quasineutrality(omega=None, kr=None, Phi_P=None, T_P=None, T_D=None, alpha_par=0, alpha_prp=0,
                                  Delta_par=0, Delta_prp=0, tau=1, epsrel_theta=1e-14, epsabs_theta=1e-14,
                                  epsrel_y=1e-12, epsabs_y=1e-12, theta_val=None, eps_AR=0.5 / 2.7778,
                                  disp_model=0.1, a_int=100, Nintegrate=10000, tol_adaptive=1e-12,
                                  asymp_J11=-1, sing_ITG=False, omegaN=None):
    """D(ω) of the circular model (old API). With theta_val: −w(θ)·⟨(D−1)/D⟩_y/(2π) as before,
    where w = 1 + ε cos θ."""
    if omega is None:
        omega = omegaN
    d = _disp(Phi_P, T_P, T_D, alpha_par, alpha_prp, Delta_par, Delta_prp, tau, eps_AR, disp_model,
              epsabs_theta, epsrel_theta, epsabs_y, epsrel_y, asymp_J11, a_int, Nintegrate, tol_adaptive, sing_ITG)
    if theta_val is None:
        return d.D(complex(omega), kr)
    if np.imag(theta_val) != 0:
        raise NotImplementedError("complex theta_val is not supported by tsm_disp")
    th = float(np.real(theta_val))
    v, f, n = d.y_average(complex(omega), kr, np.array([th]))
    return -v[0] * (1 + eps_AR * np.cos(th)) / (2 * np.pi)


def evaluate_integrand_ITG(theta_val, y_val, omega, kr, Phi_P, T_P, T_D, alpha_par=0, alpha_prp=0, Delta_par=0,
                           Delta_prp=0, tau=1, disp_model=0.1, eps_AR=0.5 / 2.7778, a_int=100, Nintegrate=10000,
                           tol_adaptive=1e-12, asymp_J11=-1):
    """D_ISM(θ, y) (old name D_ITG)."""
    d = _disp(Phi_P, T_P, T_D, alpha_par, alpha_prp, Delta_par, Delta_prp, tau, eps_AR, disp_model,
              1e-10, 1e-10, 1e-10, 1e-10, asymp_J11, a_int, Nintegrate, tol_adaptive, False)
    if np.imag(theta_val) != 0:
        raise NotImplementedError("complex theta_val is not supported by tsm_disp")
    return d.D_ism(complex(omega), kr, float(np.real(theta_val)), complex(y_val))


def evaluate_integrand_QN(theta_val, y_val, omega, kr, Phi_P, T_P, T_D, alpha_par=0, alpha_prp=0, Delta_par=0,
                          Delta_prp=0, tau=1, disp_model=0.1, eps_AR=0.5 / 2.7778, a_int=0.5, Nintegrate=10000,
                          tol_adaptive=1e-4, asymp_J11=-1):
    return 1. - 1. / evaluate_integrand_ITG(theta_val, y_val, omega, kr, Phi_P, T_P, T_D, alpha_par, alpha_prp,
                                            Delta_par, Delta_prp, tau, disp_model, eps_AR, a_int, Nintegrate,
                                            tol_adaptive, asymp_J11)


def get_Cm(tau, alpha_par, alpha_prp):
    """−C_TSM of the paper's eq. (61)."""
    return tau / (7 / 4 + 1 / tau) * (7 / 4 + 2 * alpha_par + 3 / 2 * alpha_prp - (1 + (alpha_par + alpha_prp) / 2)**2)


def get_alphaS(tau, alpha_prp):
    """C_RDK of the paper's eq. (60)."""
    return 0.5 * (1 + tau * (1 + alpha_prp))
