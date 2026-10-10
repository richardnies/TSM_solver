"""Velocity moments and the mode-structure diagnostic."""
import numpy as np
import pytest
from tsm_disp import _kernels as K
from tsm_disp import kernels_ref as R
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.roots import find_root


def _cases(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        C = np.zeros((3, 3), complex)
        for (a, c) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
            C[a, c] = rng.normal() + 1j * rng.normal()
        wM = rng.choice([-1, 1]) * rng.uniform(0.1, 0.5)
        wbar = rng.uniform(-0.8, 0.8) + 1j * rng.choice([1, -1]) * rng.uniform(0.05, 0.6)
        out.append((wbar, wM, [0.02, 0.3, 1.0][i % 3], C, 1 + i % 2))
    return out


@pytest.mark.fast
def test_general_moment_reproduces_N():
    w, oE, wM, b = 0.7 + 0.3j, 0.2, 0.15, 0.4
    w1, w2, w3 = -0.2 + 0.5 * 3 * 0.2 + 3 * 0.2, 0.5 * (wM - 3 * 0.2), 0.5 * wM - 3 * 0.2
    C = np.zeros((3, 3), complex)
    C[0, 0], C[1, 0], C[0, 1] = w1, 2 * w2, w3
    assert abs(K.moment_py(w, oE, oE, oE, wM, b, C, 2)[0] - K.N_lambda_py(w, oE, oE, oE, wM, b, 1.0, 3, 3)[0]) < 1e-15


@pytest.mark.slow
@pytest.mark.parametrize("case", _cases(8, 5))
def test_general_moments_vs_energy_shell(case):
    """Weights up to v∥⁴, v∥²v⊥², v⊥⁴ with J0 and J0², both half-planes (Landau continuation)."""
    wbar, wM, b, C, jpow = case
    v = K.moment_py(wbar, 0, 0, 0, wM, b, C, jpow)[0]
    ref = R.moment_energy(wbar, wM, b, C, jpow)
    assert abs(v - ref) < 1e-12 * max(1, abs(ref))


@pytest.mark.fast
def test_mode_structure_normalisation_and_symmetry():
    geo = miller(rhoc=0.5, rmaj=2.77778, qinp=20, shat=0.8, zed_equal_arc=True)
    d = DispersionRelation(geo, Primary.gaussian(0.72314, np.pi), eta_par=3, eta_prp=3, nthreads=8,
                           epsabs=1e-10, epsrel=1e-10)
    r = find_root(lambda w: d.D(w, 0.5), 0.58 + 0.29j, tol=1e-11)
    m = d.mode_structure(r.omega, 0.5, "potential", nz=129, ny=64)
    assert abs(m.phi_average - 1) < 1e-5                   # ⟨φ̂⟩_ψ = 1 ⇔ D(ω) = 0
    assert abs(m.flux_surface_average() - 1) < 1e-5
    q = d.mode_structure(r.omega, 0.5, "heat_flux", nz=129, ny=64)
    i, j = np.argmin(abs(q.z - 1.5)), np.argmin(abs(q.z + 1.5))
    assert abs(q.y_average[i] - q.y_average[j]) > 0.1 * abs(q.y_average).max()   # up–down asymmetric
