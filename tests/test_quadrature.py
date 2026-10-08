"""Gauss–Kronrod rules and the Fourier interpolant."""
import numpy as np
import pytest
from tsm_disp.dispersion import _X, _WK, _WGfull
from tsm_disp.geometry.base import _Fourier


@pytest.mark.fast
def test_gk21_weights_exact_for_polynomials():
    for k in range(0, 31):          # Kronrod 21 is exact to degree 31
        exact = (1 - (-1)**(k + 1)) / (k + 1)
        assert abs(np.sum(_WK * _X**k) - exact) < 1e-14
    for k in range(0, 19):          # Gauss 10 exact to degree 19
        exact = (1 - (-1)**(k + 1)) / (k + 1)
        assert abs(np.sum(_WGfull * _X**k) - exact) < 1e-14


@pytest.mark.fast
def test_fourier_interpolant_reproduces_grid_and_trig():
    z = np.linspace(-np.pi, np.pi, 65)
    f = np.exp(np.cos(z)) + 0.3 * np.sin(2 * z)
    F = _Fourier(z[0], z[-1] - z[0], f[:-1])
    assert np.max(np.abs(F(z) - f)) < 1e-13
    zz = np.linspace(-3, 3, 101)
    assert np.max(np.abs(F(zz) - (np.exp(np.cos(zz)) + 0.3 * np.sin(2 * zz)))) < 1e-12


@pytest.mark.fast
def test_gamma0_complex_series_vs_scipy():
    from tsm_disp import _kernels as K
    from tsm_disp.kernels_ref import gamma0, gamma0_prime
    for z in [0.17 / (1 + 0.7j), 2.9j, -2.5 + 1j, 5 + 3j, 1e-8, 2.99 * np.exp(2.2j), 3.01 * np.exp(-1j)]:
        g0, g0p = K.py_gamma0(z)
        assert abs(g0 - gamma0(z)) < 1e-14 * max(1, abs(gamma0(z)))
        assert abs(g0p - gamma0_prime(z)) < 1e-14 * max(1, abs(gamma0_prime(z)))
