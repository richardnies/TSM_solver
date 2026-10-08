"""J functions of the Z form against 50-digit mpmath."""
import numpy as np
import pytest
from tsm_disp import _kernels as K

mp = pytest.importorskip("mpmath")
mp.mp.dps = 50


def J_exact(z):
    z = mp.mpc(z)
    Z = 1j * mp.sqrt(mp.pi) * mp.exp(-z**2) * mp.erfc(-1j * z)
    s, z2 = Z * z, z * z
    I = s * s
    return [(-1 + I), 2 * z2 * (1 + s + 0.25 * (1 + I) / z2), 1 + (-I * (1 + 2 * z2) - 2 * z2 * (1 + 2 * s)),
            -0.5 + (-I * (1.5 + z2) - z2 * (5 + 6 * s)), z2**2 * (4 * I + 4 + 8 * s) + z2 * (4 * I + 12 * s + 8) + 3 * I - 2]


@pytest.mark.fast
@pytest.mark.parametrize("r,tol", [(0.3, 1e-13), (1, 1e-13), (3, 1e-11), (5.99, 1e-10), (6.01, 1e-10), (9, 1e-12),
                                   (40, 1e-12), (100, 1e-12)])
def test_J_relative_accuracy(r, tol):
    worst = 0
    for ph in np.linspace(-np.pi, np.pi, 25)[1:-1]:
        z = r * np.exp(1j * ph)
        Je = J_exact(z)
        if max(abs(a) for a in Je) > 1e200:
            continue
        Jk, Z = K.J_py(z)
        # e^{−ζ²} has condition number 2|ζ|²: allow that on the lower-half-plane diagonal
        cond = 2 * r * r if np.imag(z) < 0 else 1.0
        worst = max(worst, max(abs(complex(a) - b) / abs(complex(a)) for a, b in zip(Je, Jk)) / cond)
    assert worst < tol


@pytest.mark.fast
def test_plasma_dispersion_function():
    for z in [0.5 + 0.5j, 3 - 2j, 50 + 1j, -7 + 0.1j]:
        Ze = complex(1j * mp.sqrt(mp.pi) * mp.exp(-mp.mpc(z)**2) * mp.erfc(-1j * mp.mpc(z)))
        assert abs(K.plasma_disp_py(z) - Ze) < 1e-13 * abs(Ze)
    for z in [20 + 3j, 15 - 2j, -30 + 1j]:
        Ze = complex(1j * mp.sqrt(mp.pi) * mp.exp(-mp.mpc(z)**2) * mp.erfc(-1j * mp.mpc(z)))
        assert abs(K.plasma_disp_py(z, True) - Ze) < 1e-13 * abs(Ze)


@pytest.mark.fast
def test_branch_of_zeta():
    """ζ = √Ω on the sheet continuous with Im ω̄ > 0 (old get_arg_disp generalised to complex y)."""
    for wM in [0.3, -0.3]:
        for wr in [0.5, -0.5]:
            zp = K.arg_disp_py(wr + 1e-9j, (wr + 1e-9j) / wM)
            zm = K.arg_disp_py(wr - 1e-9j, (wr - 1e-9j) / wM)
            assert abs(zp - zm) < 1e-6            # continuous across the real ω̄ axis
            assert np.imag(K.arg_disp_py(wr + 0.3j, (wr + 0.3j) / wM)) > 0
