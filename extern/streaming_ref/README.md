# Reference implementation: TSM with parallel streaming and finite k_y (vendored, unmodified)

Copied on 2026-10-11 from `tsm_parallel_streaming.zip` (`tsm/src/`, `tsm/data/`), separate work generalising the
Nies & Parra (2026) TSM dispersion relation to retain ion parallel streaming (Fourier truncation along the field
line) and finite primary binormal wavenumber. These files are the regression reference for
`tsm_disp/streaming.py`; they are not imported by the package.

| file | sha1 | content |
|---|---|---|
| `tsm_core.py` | f8b1b721 | y-parametric streaming solver (`Dispersion`, `Grids`, Usmani/Thomas tridiagonal moments) |
| `tsm_ky.py` | a4c0b715 | binormal-truncated solver (`DispersionKy`, `VGrid`, `ParamsKy`) |
| `np25_eq34.py` | d6a0037e | independent NP25 eq. (34) |
| `nonresonant.py` | 9e06803c | NP25 eqs. (52)–(61) |
| `fig11_q20.npz`, `conv.npz`, `ky_hires.npz` | | reference roots (q = 20 k_x scan; M, M_φ convergence; k_y track) |

Model: circular large-aspect-ratio tokamak, ŝ = 0, |∇x| = 1, B constant, b·∇θ = 1/(qR), no trapped particles;
frequencies in v_Ti/R (v_Ti² = 2T/m), k = k_x ρ_i, v_E0 = A^P taken as the peak E×B amplitude.

Notes (checked 2026-10-11, see `tsm_disp/streaming.py`):
* `tsm_ky.DispersionKy` applies the source operator after the inverse, J0 (Δ A⁻¹ − I) J0, whereas the
  gyrokinetic equation for h gives J0 (A⁻¹ Δ − I) J0. The response matrices D differ (1–6 %), but in this
  circular model 𝒟 = 1 − τ[D⁻¹]₀₀ is identical to round-off (10⁻¹⁶, flat or Gaussian primary, with or without
  binormal drift), so its results are unaffected. With z-dependent geometry (e.g. Miller, ŝ = 0) the two orders
  differ slightly (2e-4 relative in 𝒟 in one test); TSM_solver uses A⁻¹ Δ.
* `tsm_core.py` module docstring still says v_E0 = √2 A^P (the code uses v_E0 = A^P), as do `np25_eq34.py`
  and `nonresonant.py`.
