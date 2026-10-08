"""The ion response N against independent quadratures (kernels_ref)."""
import numpy as np
import pytest
from tsm_disp import _kernels as K
from tsm_disp import kernels_ref as R
from conftest import ref_point


def _cases(n, seed, imag_sign=None, min_imag=0.02):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = [1e-3, 0.05, 0.17, 0.6, 1.0, 2.0][i % 6]
        wM = rng.choice([-1, 1]) * rng.uniform(0.05, 0.6)
        s = imag_sign if imag_sign is not None else rng.choice([1, -1])
        wbar = rng.uniform(-1, 1) + 1j * s * rng.uniform(min_imag, 0.8)
        oE = rng.uniform(-0.6, 0.6)
        out.append((wbar + oE, oE, oE * rng.uniform(0.5, 1.5), oE * rng.uniform(0.5, 1.5), wM, b, 1.0,
                    rng.choice([0., 3.]), rng.choice([0., 3.])))
    return out


@pytest.mark.fast
def test_reference_point_against_direct_quadrature():
    p = ref_point()
    N, flag, nev = K.N_lambda_py(p["omega"], p["oE"], p["oE"], p["oE"], p["oM"], p["b"], 1.0, 3, 3)
    # direct (v∥, v⊥) quadrature, computed once: 0.15853199165700044−0.06825640181282505j
    assert abs(N - (0.15853199165700044 - 0.06825640181282505j)) < 1e-14
    assert flag == 0


@pytest.mark.slow
@pytest.mark.parametrize("case", _cases(12, 11, imag_sign=1))
def test_lambda_form_vs_velocity_space(case):
    N, flag, nev = K.N_lambda_py(*case)
    Nv = R.N_velocity(*case, eps=1e-11)
    assert abs(N - Nv) < 1e-9 * max(1, abs(Nv))


@pytest.mark.slow
@pytest.mark.parametrize("case", _cases(36, 1))
def test_lambda_form_vs_energy_shell_both_half_planes(case):
    """Includes the Landau continuation (Im ω̄ < 0) on both sides of the resonance and both drift signs."""
    N, flag, nev = K.N_lambda_py(*case)
    Ne = R.N_energy(*case)
    assert abs(N - Ne) < 1e-12 * max(1, abs(Ne))


@pytest.mark.fast
@pytest.mark.parametrize("b", [0.01, 1.0])
@pytest.mark.parametrize("wM", [0.05, -0.05])
def test_continuity_across_real_axis(b, wM):
    """The continued N is analytic across the resonant real axis: N(Ω ± iε) differ by O(ε)."""
    jumps = []
    for eps in [1e-4, 1e-6, 1e-8]:
        Np = K.N_lambda_py(np.sign(wM) * 0.58 + 1j * eps, 0, 0, 0, wM, b, 1.0, 0, 0)[0]
        Nm = K.N_lambda_py(np.sign(wM) * 0.58 - 1j * eps, 0, 0, 0, wM, b, 1.0, 0, 0)[0]
        jumps.append(abs(Np - Nm) / eps)
    assert max(jumps) < 1.5 * min(jumps) + 1e-3      # jump ∝ ε (finite derivative), no O(1) discontinuity


@pytest.mark.fast
def test_zero_drift_closed_form():
    w, oE, b = 0.3 + 0.2j, 0.1, 0.4
    N = K.N_lambda_py(w, oE, oE, oE, 0.0, b, 1.5, 0.0, 2.0)[0]
    Ng = R.N_energy(w, oE, oE, oE, 0.0, b, 1.5, 0.0, 2.0)
    assert abs(N - Ng) < 1e-15


@pytest.mark.slow
def test_Z_form_is_order_b_expansion():
    """|N_Z − N| ∝ b² (the Z form keeps O(b) FLR)."""
    errs = []
    for kr in [0.02, 0.01]:
        b = 0.5 * kr**2
        wM = kr * 50.0
        oE = -30.0 * kr * np.sin(0.3)
        w = 0.6 + 0.15j
        Nz = K.N_Z_py(w, oE, oE, oE, wM, b, 1.0, 3, 3)
        Ne = R.N_energy(w, oE, oE, oE, wM, b, 1.0, 3, 3)
        errs.append(abs(Nz - Ne))
    assert 12 < errs[0] / errs[1] < 20          # b halves twice per kr halving: ratio 16


@pytest.mark.fast
def test_old_lambda_kernel_bug_documented(old_kernels):
    """The original λ form used scipy's complex ive (e^{−|Re z|}I0) for Γ0(b̃): D_ISM off by ~1e-3."""
    p = ref_point()
    th, eps, kr = 0.4, 0.18, 0.5
    PhiP = p["A"] / kr
    Dold = old_kernels.eval_integrand_DITG_lambdaint(p["y"], 1.0, 3., 3., 0., 0., PhiP, kr, np.sin(th) / 2.7778,
                                                     p["omega"], th, eps, -0.1, 30., 100000, 1e-12)
    Dnew = K.D_ism_py(p["omega"], p["y"], p["A"], p["oM"], p["b"], 1.0, 3., 3., 0., 0., 0)
    assert abs(Dnew - (0.9916061321920067 + 0.06825640181282505j)) < 1e-13
    assert 5e-4 < abs(Dold - Dnew) < 2e-3


@pytest.mark.fast
def test_Z_kernel_matches_old_code_pointwise(old_kernels):
    omega, kr, eps = 0.96189 + 0.40884j, 0.5, 0.5 / 2.7778
    worst = 0
    for th in np.linspace(-3, 3, 13):
        for y in np.linspace(-3, 3, 13):
            PhiP = 4 * 0.3557 * np.exp(-(th / np.pi)**2) / 0.7733975858850816
            TD = np.sin(th) / 2.7778
            Do = old_kernels.eval_integrand_DITG(y, 1.0, 3., 3., 0., 0., PhiP, kr, TD, omega, th, eps, 0.1, -1.0)
            b = 0.5 * kr**2 * (1 + eps * np.cos(th))**2
            Dn = K.D_ism_py(omega, y, kr * PhiP, kr * TD, b, 1.0, 3., 3., 0., 0., 1)
            worst = max(worst, abs(Do - Dn))
    assert worst < 1e-8          # the old closed forms lose up to ~1e-8 to cancellation near |ζ| = 100
