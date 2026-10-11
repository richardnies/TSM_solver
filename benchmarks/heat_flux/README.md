# Primary heat flux: formula vs stella

`tsm_disp.primary.heat_flux_stella` against stella's `qflx` for the frozen primary of `secondary = .true.` runs
(3 time steps, zonal seed 1e-16, so the flux is that of the primary alone; it is constant in time).

* `make_decks.py` writes `runs/*/CBC.in` from `template.in` (shaped-Miller benchmark deck, q = 20, y0 = 1000,
  Gaussian φ0 of width π), varying the shaping (CBC or κ = 1.5, δ = 0.3), `den0_phase`, `tpar0`, `tperp0`,
  `width0`, `y0` and the velocity grid.
* `run_decks.sh` runs them (`mpirun -n 4 /home/rnies/stella_COPY/stella`, seconds each).
* `compare.py` prints stella's qflx against the formula (Δ∥ = Δ⊥ = −den0_phase, η = tpar0, tperp0,
  k_y^P = 1/y0, Φ_P from `stella_phiP`).

Result (2026-10-11):

| case | stella qflx | formula | rel. diff |
|---|---|---|---|
| CBC, den0_phase = +1.0, tpar0 = 0, vperp_max 4.5 | −523.01 | −523.06 | −1.0e-4 |
| CBC, den0_phase = −0.5, vperp_max 4.5 | 446.97 | 447.02 | 1.1e-4 |
| shaped, den0_phase = −0.7, tperp0 = 1, vperp_max 4.5 | 395.16 | 395.20 | 1.0e-4 |
| CBC, den0_phase = −0.8, tperp0 = 0, width0 = 1, y0 = 200 (vperp_max 3) | 405.79 | 405.92 | 3.2e-4 |
| same three cases with stella's default vperp_max = 3 | | | 0.8–2.0e-3 |
| CBC, den0_phase = 0 | 1.8e-13 | 0 | |

With the default `vperp_max = 3` stella's v⊥ integral is truncated: ∫(v⊥² − 1)F_M ≠ 0 on its grid, so the
temperature perturbation leaks into the density and φ (|φ/φ_expected| − 1 up to 9e-4, phase up to 8e-4 rad);
more μ points at the same cut-off make it worse, a larger cut-off removes it.
