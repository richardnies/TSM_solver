# TSM_solver

Solver for the dispersion relation of **toroidal secondary modes** (TSMs) and the other secondary
instabilities of zonal flows: Nies & Parra, *Theory of zonal flow growth and propagation in toroidal
geometry*, Plasma Phys. Control. Fusion **68** (2026) 045028. Given a frozen streamer primary mode
(amplitude, temperature perturbations, variation along the field line) and a flux-tube geometry
(Miller, VMEC, or the geometry of a stella run), it finds the complex frequency $`\omega`$ of the zonal
secondary mode at radial wavenumber $`k_x`$. The Python package is called `tsm_disp`.

## Installation

Requirements: Python ≥ 3.8 and a C compiler (gcc, clang or Intel); tested with Python 3.8 (numpy 1.24,
scipy 1.10) and 3.13 (numpy 2.5, scipy 1.18). Numpy, scipy and Cython (≥ 3) are installed automatically.

```bash
git clone https://github.com/richardnies/TSM_solver.git TSM_solver
cd TSM_solver
pip install ".[plot]"          # or: pip install -e ".[plot,test]" to develop and run the tests
```

This installs two packages: `tsm_disp`, the solver, and `evaluation_scripts`, the functions the examples call
(continuation, data files, figures). Optional extras: `stella` (f90nml, netCDF4: read stella runs and decks), `vmec`
(simsopt: VMEC equilibria), `plot` (matplotlib, for the examples), `test` (pytest, mpmath). The build uses OpenMP when
the compiler supports `-fopenmp` and prints `building with OpenMP`; otherwise it runs single-threaded
(the message `clang: error: unsupported option '-fopenmp'` from Apple clang is this check and harmless; for
OpenMP on macOS install `libomp` or build with a gcc, e.g. `CC=gcc-14 pip install ...`). On x86-64 the kernels are
compiled with AVX2 and FMA when the compiler and the building machine support them (`AVX2 + FMA` in the build
output, ≈ 14 % faster); the extension then needs an AVX2 CPU (Intel since 2013, AMD Zen), so set `TSM_NO_AVX2=1`
when building for older machines. Without installing,
`python setup.py build_ext --inplace` builds the extension inside the source tree; it needs Cython ≥ 3 in the
environment (`pip install "cython>=3"`), and the examples then need the repository on the path, e.g.
`PYTHONPATH=/path/to/TSM_solver python tsm_kx_scan.py` (an editable install is cleaner: it exposes only the two
packages, not the repository's other modules such as the compatibility shim `disp_rel.py`).

## Quickstart

```bash
python examples/tsm_kx_scan/tsm_kx_scan.py      # ~20 s on 32 cores, ~30 s on 8, a few minutes on 1
```

For the circular Cyclone Base Case surface ($`q = 20`$), $`\eta_\parallel = \eta_\perp = 3`$ and a Gaussian primary it follows
the most unstable TSM from $`k_x\rho_i = 0.5`$ up to 0.8 and down to 0.1 and writes, next to the script,
`tsm_kx_scan.txt` ($`k_x`$, $`\omega_r`$, $`\gamma`$) and `tsm_kx_scan.pdf` ($`\omega_r`$, $`\gamma`$ against $`k_x`$, with stella
gyrokinetic simulations of the same case). A copy of the script in any other directory runs the same way and writes
next to the copy.

**Examples.** Each example is a directory `examples/<name>/` holding one script, `<name>.py`, with its parameters
(amplitude, $`\tau`$ = `TAU`, $`\eta_\parallel`$ = `ETA_PAR`, $`\eta_\perp`$ = `ETA_PERP`, $`q`$, $`k_x`$ range, initial guess, solver,
threads, …) set at the top; edit them and run the script. The scripts need only the installed package
(`pip install ".[plot]"`, or `pip install -e ".[plot]"` to edit the package as well), so each can be copied anywhere,
e.g. one copy per case, and run from any directory; its figures (PDF) and data are written next to the script, i.e.
next to the copy. The scripts build the geometry, primary and dispersion relation with `tsm_disp` and call the
functions in `evaluation_scripts` (root continuation in $`k_x`$, data files, figures; installed with `tsm_disp`), which
can be reused in your own scripts:

| example | what it does |
|---|---|
| `tsm_kx_scan` | TSM ω(k_x) on the CBC; λ form or, with `SOLVER = "streaming"`, ion parallel streaming; amplitude from Φ_P or from the primary heat flux |
| `mode_structure` | potential, pressure and nonlinear heat flux of the TSM on the flux surface |
| `amplitude_scan` | TSM at fixed $`k_x`$ as $`\Phi_P`$ is scanned: up to the RDK limit (paper eq. 51; exact drift-free root), down into the ISM region |
| `top_hat_primary` | TSM ω(k_x) for an analytic geometry ($`v_{Mx} = \sin\theta`$) and a top-hat primary |
| `find_all_roots` | all roots in a box of the complex ω plane without an initial guess, with a map of the ISM region |

The core of the example in a few lines:

```python
import numpy as np
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.roots import find_root

geo = miller(rhoc=0.5, rmaj=2.77778, qinp=20, shat=0.8, zed_equal_arc=True)   # flux tube
pri = Primary.gaussian(Phi_P=0.723, width=np.pi)                                # frozen primary
disp = DispersionRelation(geo, pri, tau=1, eta_par=3, eta_prp=3, nthreads=8)  # D(ω, k_x)
root = disp.find_root(0.58 + 0.29j, 0.5)                                        # D = 0 near a guess (Newton)
print(root.omega)            # ≈ 0.5808 + 0.2927j  (units v_th/a, v_th = √(2T/m))
```

**Mode structure.** `disp.mode_structure(omega, kx, quantity)` returns the eigenmode on the flux surface
(field-line coordinate × primary phase y) for `quantity` = `"potential"` ($`\hat\phi = \langle\phi\rangle_\psi/D_{\rm ISM}`$),
`"pressure"` ($`\hat P = T\int d^3v\,(v_\parallel^2+v_\perp^2/2)/v_{th}^2\,\hat g`$, paper eq. 38) or `"heat_flux"`
($`\hat Q_{NL} = v_{Ex}^P \hat P`$, eq. 40, whose $`y`$ average carries the up–down asymmetry that drives the TSM),
normalised to $`Ze\langle\phi\rangle_\psi/T = 1`$; `phi_average` = 1 checks that ω is a root.
`examples/mode_structure/mode_structure.py` plots all three for the example case.

**Finding a first root.** Roots are followed by continuation, so a scan needs one good initial guess. Good
guesses: for the TSM, $`\omega_r \approx 2k_x\rho_i v_{th}/R`$ with $`\gamma`$ a fraction of it; for the GAM
($`\Phi_P \to 0`$), `tsm_disp.limits.omega_GAM(geo, kx)`; for the purely growing RDK mode (no drift),
`limits.omega_RDK_smallb`. `find_root` returns `converged = False` when it fails; try another guess.
Roots should not be trusted where `DispersionRelation.D_ism` vanishes on the surface (the paper's ISM
region, where $`D`$ is not analytic).

## Equations

The secondary (zonal, wavenumber $`k_x`$) of a frozen streamer primary $`v_{Ex}(z,y) = -\Phi_P F_{\theta,P}(z)\sin y`$
(amplitude $`\Phi_P`$ and envelope along the field line $`F_{\theta,P}`$, `Ftheta_P` in the code; called $`T_P`$ before
2026-10-12, renamed to avoid confusion with a temperature) obeys (paper eqs. 31–32)

```math
D(\omega,k_x) = 1-\Big\langle \frac{1}{D_{\rm ISM}}\Big\rangle = \int dz\, w(z)\,\frac{1}{2\pi}\int_{-\pi}^{\pi}dy\,\frac{D_{\rm ISM}-1}{D_{\rm ISM}} = 0,
\qquad D_{\rm ISM} = \frac{1-\Gamma_0(b)+\tau-N}{\tau},
```

with $`w = 1/(\mathbf B\cdot\nabla z)`$ normalised to $`\int w\,dz = 1`$, $`b = k_x^2|\nabla x|^2\rho^2/2`$ and the ion response

```math
N=\int d^3\hat v\,\hat F_M J_0^2(\sqrt{2b}\,\hat v_\perp)\,
\frac{\omega_1+2\omega_2\hat v_\parallel^2+\omega_3\hat v_\perp^2}{\bar\omega-\omega_{M}(\hat v_\parallel^2+\hat v_\perp^2/2)},\qquad
\begin{aligned}
&\bar\omega=\omega-\omega_E,\quad \omega_E = k_x v_{Ex},\quad \omega_M = k_x v_{Mx},\\
&\omega_1=\tau(-\omega_E+\tfrac12\eta_\parallel\omega_{E\parallel}+\eta_\perp\omega_{E\perp}),\quad
\omega_2=\tfrac12(\omega_M-\tau\eta_\parallel\omega_{E\parallel}),\quad \omega_3=\tfrac12\omega_M-\tau\eta_\perp\omega_{E\perp},
\end{aligned}
```

where $`\omega_{E\parallel,\perp} = \omega_E(y+\Delta_{\parallel,\perp})`$ carry optional phase shifts. Two evaluations of $`N`$:

* **λ form** (`form="lambda"`, old `disp_model < 0`): the Laplace representation (paper eq. 34), exact in $`b`$.
  The λ contour is chosen per point (straight ray $`\lambda = is/\bar\omega`$ with Gauss–Laguerre or adaptive
  Gauss–Kronrod; a real segment plus ray near the resonance; a detour around the singularities
  $`\lambda = i/\omega_M,\,2i/\omega_M`$ for the Landau continuation to $`{\rm Im}\,\bar\omega<0`$).
* **Z form** (`form="Z"`, old `disp_model = 0.1`): the $`O(b)`$ closed form in the plasma dispersion function,
  $`\zeta=\sqrt{\bar\omega/\omega_M}`$ on the sheet continuous with $`{\rm Im}\,\bar\omega>0`$. For $`|\zeta|\ge 6`$ the J
  functions use a 48-term cancellation-free asymptotic series (`scripts/derive_J_asymptotics.py`).

**Units** (stella): lengths in $`a`$, wavenumbers in $`1/\rho_{\rm ref}`$, velocities in $`v_{th}=\sqrt{2T/m}`$,
frequencies in $`v_{th}/a`$. Multiply frequencies by $`R/a`$ for the paper's $`v_{Ti}/R`$.

**Analytic continuation.** For $`{\rm Im}\,\omega\le 0`$ the $`y`$ integral runs on $`y = t + i\,{\rm sgn}(A)\,\eta\cos t`$,
which keeps the branch points ($`\bar\omega = 0`$) on the side they occupy for $`{\rm Im}\,\omega>0`$. This is the analytic
continuation of $`D`$ wherever $`D_{\rm ISM}`$ has no zeros near the contour. Inside the paper's ISM region
($`D_{\rm ISM}=0`$ somewhere on the surface) $`D`$ is not analytic, and no contour is attempted (paper §3.3).
Near-real zeros of $`D_{\rm ISM}(y)`$ are located by Newton's method and their poles subtracted analytically
(flag 64), which keeps the cost bounded near the ISM region.

## Geometry (`tsm_disp.geometry`)

Only $`k_y=0`$ quantities enter: along the field-line coordinate $`z`$,

```math
v_{Mx} = \frac{\mathrm{gbdrift0}}{2\hat s},\qquad
\frac{|\nabla x|^2}{B^2} = \frac{\mathrm{gds22}}{\hat s^2\,\mathrm{bmag}^2},\qquad
w \propto \frac{1}{\mathrm{bmag}\cdot\mathrm{gradpar}} ,
```

in stella's conventions: stella advances $`\partial_t g + i\omega_d g = \dots`$ with
$`\omega_d = (k_x/\hat s)(\mathrm{cvdrift0}\,v_\parallel^2+\mathrm{gbdrift0}\,v_\perp^2/2)/2`$ and
$`\phi\propto e^{-i\omega t}`$, and `cvdrift0` = `gbdrift0`. For the circular CBC this gives
$`v_{Mx} = -\sin\theta/R`$, the mirror
image of the old `T_D = sin θ/2.7778`; the two are equivalent for up-down-symmetric problems.

| constructor | source | verified against |
|---|---|---|
| `AnalyticGeometry(vMx, gradx=1, B=1, jacobian=1/B, breakpoints=())` | functions of θ (callables or constants) | `CircularAnalytic` (1e-12) |
| `CircularAnalytic(eps, rmaj, vMx)` | the model of the original code | itself (regression) |
| `Uniform(vMx=0)` | slab / RDK limit | paper eq. (B.5) |
| `miller(rhoc, rmaj, rgeo, shift, qinp, shat, kappa, kapprim, tri, triprim, …, zed_equal_arc)` | port of stella's `geometry_miller.f90` | stella's `geometry_miller.*.output` (≤ 5e-9, its print precision) and `.out.nc` of equal-arc runs (≤ 1e-12) |
| `vmec(wout, torflux, alpha0, zeta_center, nfield_periods, nzed)` | simsopt `vmec_fieldlines`, stella's grid | stella's precise-QH run (≤ 2e-6) |
| `from_stella_run(run_dir)` | `<run>.out.nc` (or `<run>.geometry`, 4 digits) + input | Miller port (1e-6) |

Tokamak tubes with `nperiod = 1` are periodic (Fourier interpolation); VMEC tubes are not (field-line average).

**Analytic θ-variation.** Geometry and primary can both be given as functions of θ:
`AnalyticGeometry(vMx, gradx, B, jacobian)` and `Primary(Ftheta_P, Phi_P, breakpoints=[...])`, with `Ftheta_P` a
vectorised callable of θ; list discontinuities or sharp features in `breakpoints` (they become panel edges of
the θ integral). `Primary.top_hat(center_tophat, width_tophat, Phi_P)` is 1 for $`|\theta-\theta_{\rm TH}| \le w_{\rm TH}/2`$
(modulo 2π) and 0 otherwise. `examples/top_hat_primary/top_hat_primary.py` (≈ 20 s on 32 cores) follows the TSM
in $`k_x`$ for $`v_{Mx} = \sin\theta`$, $`|\nabla x| = 1`$ and a top-hat primary ($`\theta_{\rm TH} = 0`$, $`w_{\rm TH} = \pi`$,
$`\Phi_P = 3`$ by default; parameters `CENTER_TOPHAT`, `WIDTH_TOPHAT`, `PHI_P`, `GUESS`, or `HEAT_FLUX`, `KY_P`, `DELTA` to set the
amplitude by the primary heat flux, e.g. 9708, 0.001, 0.5 for $`\Phi_P = 3`$) until the root reaches the ISM region.

```python
from tsm_disp.geometry import AnalyticGeometry
geo = AnalyticGeometry(vMx=np.sin, gradx=1.0)                    # B = 1, weight ∝ 1/B (b·∇θ uniform)
pri = Primary.top_hat(center_tophat=0.0, width_tophat=np.pi, Phi_P=3.0)
disp = DispersionRelation(geo, pri, tau=1, eta_par=3, eta_prp=3, nthreads=8)
disp.find_root(1.05 + 0.66j, 0.3).omega                         # 1.0535 + 0.6621i
```

## Amplitude conventions

stella's `secondary = .true.` initialises and freezes the primary coefficient at $`k_y^P = 1/y_0`$,

```math
g = \mathrm{phiinit}\;\phi_0(z)\left[\mathrm{den0} + \mathrm{tpar0}\,\big(v_\parallel^2-\tfrac12\big) + \mathrm{tperp0}\,\big(v_\perp^2-1\big)\right]F_M ,
```

whose potential has real-space amplitude $`2|\hat\phi|`$ and radial E×B velocity $`f_{\rm exb}\,\partial_y\phi`$, with
$`f_{\rm exb}`$ = `exb_nonlin_fac` (the ½ of $`v_{th}^2=2T/m`$ is inside it). Hence

```math
\Phi_P\,F_{\theta,P}(z) = 2 f_{\rm exb}\,k_y^P\,|\hat\phi(z)|,\qquad \eta_\parallel = \mathrm{tpar0},\quad \eta_\perp = \mathrm{tperp0}
```

(`primary.stella_phiP`). For the CBC surface $`2 f_{\rm exb} = 1/\sqrt{1-\epsilon^2} = 1.0166`$.
The historical drivers used $`\Phi_P = k_y^P \cdot \mathrm{phiinit} \cdot \mathrm{fac\_norm}`$ (`fac_norm` the RMS
normalisation of the Gaussian), so their solver curves had a 1.66 % weaker drive than the stella runs they were
compared with. Their plot label is $`\mathcal A = \Phi_P/0.3557`$; 0.3557 is the true amplitude of the reference
deck phiinit = 35, $`y_0 = 100`$ ($`2 \cdot 0.5083 \cdot 0.01 \cdot 35 = 0.3558`$).

The paper's eq. (21),

```math
A^P = \sqrt{\big\langle (v_{Ex} R / v_{Ti}\rho_i)^2 \big\rangle_\psi}\quad\text{(average over $z$ and $y$)},
```

is `primary.A_P`. For the Gaussian CBC primary $`A^P = (R/a)\,\Phi_P\sqrt{\langle F_{\theta,P}^2\rangle/2} = 2.024\,\Phi_P`$,
so a legacy label $`\mathcal A`$ corresponds to $`A^P = 0.732\,\mathcal A`$. The labels follow a different
convention from eq. (21), and the paper's quoted values should be checked against it.

### Amplitude from the primary heat flux

Instead of $`\Phi_P`$ or $`A^P`$, the primary can be set by its flux-surface-averaged ion heat flux $`Q`$ in
stella's normalisation (the `qflx` of the `.fluxes` file, gyro-Bohm units $`Q/(n T v_{th}\rho_*^2)`$).
A frozen primary carries heat flux only if its temperature perturbation is out of phase with $`\varphi`$
(phases $`\Delta_\parallel, \Delta_\perp`$; stella: $`\Delta = -`$`den0_phase`). stella's diagnostic
(`get_fluxes`: $`-s_\psi\,\tfrac12\,\mathrm{mode\_fac}\,k_y\,\mathrm{Im}[\int d^3v\,(v_\parallel^2+v_\perp^2)J_0 g\,\varphi^*]`$,
weighted with the Jacobian and divided by $`\langle|\nabla\rho|\rangle`$) gives, for $`k_y^P\rho \to 0`$ and $`T_e = T_i`$,

```math
Q = -s_\psi\,\Big(\tfrac12\eta_\parallel\sin\Delta_\parallel + \eta_\perp\sin\Delta_\perp\Big)\,
\frac{\Phi_P^2\,\langle F_{\theta,P}^2\rangle}{4 f_{\rm exb}^2\,k_y^P\,\langle|\nabla\rho|\rangle},
```

with $`s_\psi`$ = `sign_torflux` (−1 for Miller), $`f_{\rm exb}`$ = `exb_nonlin_fac` and $`\langle\cdot\rangle`$ the
field-line average. At fixed E×B amplitude $`Q \propto \Phi_P^2/k_y^P`$, so the primary's binormal wavenumber
must be given:

```python
pri = Primary.gaussian(1.0, width=np.pi).with_heat_flux(Q=1159, geometry=geo, ky_P=1e-3, eta_par=3, eta_prp=3,
                                                        delta_par=0.5, delta_prp=0.5)     # Φ_P = 0.7232
disp = DispersionRelation(geo, pri, eta_par=3, eta_prp=3, delta_par=0.5, delta_prp=0.5)  # same η and Δ
heat_flux_stella(pri, geo, 1e-3, 3, 3, 0.5, 0.5)                                         # → 1159
```

`examples/tsm_kx_scan/tsm_kx_scan.py` with `HEAT_FLUX = 1159`, `KY_P = 0.001`, `DELTA = 0.5` does the $`k_x`$ scan this way.
Checked against 3-step stella runs of the frozen primary (`benchmarks/heat_flux/`, CBC and shaped Miller,
$`\Delta = \pm`$, $`\eta_\parallel = 0`$ or $`\eta_\perp`$ = 0, $`k_y^P`$ = 0.001 and 0.005): agreement to
1.0–1.1e-4 (stella's 5 printed digits) with `vperp_max = 4.5`. With stella's default `vperp_max = 3` its
$`v_\perp`$ integral is truncated and its `qflx` is 0.1–0.3 % low (the temperature perturbation leaks into the
density and $`\varphi`$ at the 1e-3 level).

## Parallel streaming and finite primary $`k_y`$ (`tsm_disp.streaming`)

The solver above neglects parallel streaming ($`q\to\infty`$): $`D_{\rm ISM}`$ is local in θ. Two further solvers,
ported from a separate study of the TSM with ion parallel streaming (`extern/streaming_ref/`, circular $`\hat s = 0`$
model) and generalised here to the package's geometries, retain it by expanding along the field line,
$`h = \sum_m h_m e^{imz}`$. At each primary phase $`y`$ and velocity, the gyrocentre equation becomes a matrix equation,

```math
\mathsf A\,h = \Delta_g J_0\varphi,\qquad
\mathsf A = \bar\omega - \omega_M E - v_\parallel\,\hat{\boldsymbol b}\cdot\nabla z\,(-i\partial_z),\qquad
\Delta_g = \bar\omega - \omega_g ,
```

with every $`z`$-dependent coefficient ($`v_{Mx}`$, $`\hat{\boldsymbol b}\cdot\nabla z`$, $`J_0`$ and $`\Gamma_0`$ through $`b(z)`$,
$`F_{\theta,P}`$, the weight) a Toeplitz matrix in $`m`$. Quasineutrality truncated to $`|m|\le M`$ ($`h`$) and $`|m|\le M_\varphi`$ ($`\varphi`$) gives

```math
\mathcal D(\omega) = 1 - \tau\big\langle c^{\rm T} \mathsf D(y)^{-1} e_0 \big\rangle_y,\qquad
\mathsf D = (1+\tau) - \Gamma_0 - \sum_v W_v J_0 (\mathsf A^{-1}\Delta_g - 1) J_0,\qquad c_m = \int w\, e^{imz}\,dz ,
```

which reduces to eq. (32) without streaming (checked to $`3\times10^{-7}`$ in real geometry) and to the reference
solver in its circular model ($`10^{-16}`$).

| solver | physics | geometry needed | cost (CBC, 24 threads) |
|---|---|---|---|
| `DispersionRelation(form="lambda")` | eq. (32): no streaming, exact FLR | vMx, b, weight (any tube) | 0.01–0.05 s per D |
| `DispersionRelation(form="Z")` | eq. (32) to $`O(b)`$ | same | ~0.003 s |
| `StreamingDispersion` | + ion parallel streaming (no mirror force) | + `gradpar` (b·∇z); periodic tube | 1–3 s per D |
| `BinormalDispersion(ky_P)` | + primary $`k_y`$: sidebands, binormal drift, sideband FLR, primary gyroaverage | + vMy, \|∇y\|², ∇x·∇y, all periodic (ŝ = 0) | 5 s per D ($`N`$ = 6, $`M_\varphi`$ = 8, $`k_y^P\to0`$) to 70 s ($`M_\varphi`$ = 16, $`k_y^P`$ = 0.2) |

```python
from tsm_disp.streaming import StreamingDispersion, BinormalDispersion, VelocityGrid
sd = StreamingDispersion(geo, pri, tau=1, eta_par=3, eta_prp=3, nthreads=24)        # Mphi = 16, M ≥ 24 by default
root = sd.find_root(0.455 + 0.26j, 0.4)                                            # Muller
grid = VelocityGrid(96, 12, cdepth=1.0, alpha=0.65)        # contours continued below Im ω = 0 (keep FIXED on a track)
bd = BinormalDispersion(miller(..., shat=0), pri, ky_P=0.1, nthreads=24)   # N = 6 sidebands, Mphi = 8
D, sign, logdet = bd.D(omega, kx, with_det=True)           # det D for pole-free winding counts
```

**Truncation.** $`M`$ must exceed the transit criterion $`\omega_M/(v_\parallel\hat{\boldsymbol b}\cdot\nabla z)\sim k_x\rho\,q`$
(default $`\lceil 6 + 2.2\,k_x\max|v_{Mx}|/\langle\hat{\boldsymbol b}\cdot\nabla z\rangle\rceil`$). $`M_\varphi`$ = 5–6 is
enough for a flat primary on the circular model, but a localised secondary needs more: on the CBC with a Gaussian
primary $`M_\varphi`$ = 8, 12, 16 give relative errors $`2\times10^{-2}`$, $`8\times10^{-4}`$, $`7\times10^{-5}`$ (default 16).
Sidebands: at $`k_y^P = 0`$ the binormal solver must reproduce the $`y`$-parametric one. On the CBC $`\hat s = 0`$ tube
($`q`$ = 20, Gaussian primary, $`\Phi_P`$ = 0.723, $`k_x\rho_i`$ = 0.4), $`N`$ = 3, 6, 9 give relative errors in $`\omega_r`$ of
$`-6\times10^{-3}`$, $`-2\times10^{-5}`$, $`+2\times10^{-5}`$ and in $`\gamma`$ of $`+2\times10^{-3}`$, $`+4\times10^{-4}`$, $`<10^{-5}`$; the default is
$`N`$ = 6 (the reference study used 3). The error falls exponentially (circular model, $`\Phi_P`$ = 2: $`N`$ = 8, 12, 16 give
$`1.4\times10^{-6}`$, $`4.5\times10^{-9}`$, $`1.5\times10^{-11}`$).

**Finite primary $`k_y`$.** On the CBC $`\hat s = 0`$ tube ($`q`$ = 20, Gaussian primary, $`\Phi_P`$ = 0.723, $`k_x\rho_i`$ = 0.4, $`N`$ = 6,
$`M_\varphi`$ = 16) the TSM root moves from 0.4552 + 0.2577i ($`k_y^P\to0`$) to 0.4607 + 0.2528i at $`k_y^P\rho_i`$ = 0.1 ($`\omega_r`$ +1.2 %,
$`\gamma`$ −1.9 %) and 0.4743 + 0.2434i at 0.2 (+4.2 %, −5.5 %); $`N`$ = 9 with $`M_\varphi`$ = 8 gives both roots to within $`6\times10^{-5}`$ ($`\omega_r`$) and $`8\times10^{-4}`$ ($`\gamma`$).
stella's secondary runs use $`k_y^P`$ = $`10^{-3}`$, where the effect is negligible.

**Result (q = 20, `benchmarks/stella_cbc/run_streaming.py`, `fig_stella_q20_streaming.pdf`).** Streaming removes most
of the remaining deviation from stella of the λ form, most at small $`k_x`$ (small $`k_x\rho_i q`$):

| label | λ form: mean \|Δω_r\|, \|Δγ\| | streaming: mean \|Δω_r\|, \|Δγ\| | largest λ-form error → streaming |
|---|---|---|---|
| 𝒜 = 1 | 0.76 %, 1.38 % | 0.20 %, 0.45 % | γ +3.8 % → +0.03 % ($`k_x\rho_i`$ = 0.2) |
| 𝒜 = 2 | 0.68 %, 0.85 % | 0.13 %, 0.10 % | γ +4.4 % → −0.4 % (0.1) |
| 𝒜 = 4 | 0.81 %, 0.41 % | 0.34 %, 0.21 % | ω_r +2.3 % → +0.5 % (0.1) |

Not every point improves: for 𝒜 = 1 at $`k_x\rho_i`$ ≥ 0.6, $`\gamma`$ is +0.5 to +0.8 % above stella with or without streaming, and
$`\omega_r`$ rises to +0.5 % at 0.8.

**$`q`$ = 2.8 (paper Fig. 1 scan, $`\theta`$-constant primary, $`\Phi_P`$ = 0.355, $`\eta_\parallel = 0`$, $`\eta_\perp = 3`$;
`run_streaming_caseB.py`, `fig_stella_q2.8_streaming.pdf`).** Here stella's mode at $`k_x\rho_i`$ ≤ 0.12 is purely growing and
the λ form has no root at all ($`\mathcal D(i\gamma) > 0`$). With streaming, $`\mathcal D(i\gamma)`$ (real on the imaginary axis) changes
sign and a purely growing root appears, but with a third to a sixth of stella's growth rate and over a narrower range:

| $`k_x\rho_i`$ | 0.02 | 0.04 | 0.06 | 0.08 | 0.10 | 0.12 |
|---|---|---|---|---|---|---|
| stella $`\gamma a/v_{\rm th}`$ | 0.021 | 0.043 | 0.056 | 0.061 | 0.057 | 0.034 |
| streaming $`\gamma a/v_{\rm th}`$ | 0.0070 | 0.0131 | 0.0163 | 0.0162 | 0.0089 | none |

(velocity grid 192 × 24; 384 × 24 changes $`\gamma`$ by $`\le 0.5\,\%`$ at 0.04 and 0.08; 96 × 12 by up to 20 % at 0.02). At
$`k_x\rho_i`$ = 0.31–0.41 stella's mode is oscillatory (0.446 + 0.062i at 0.41) and neither solver has a root near it: a
$`|\mathcal D|`$ map over $`0.15\le\omega_r\le0.85`$, $`-0.04\le\gamma\le0.16`$ has no zero (minimum 0.035, at the box edge). What
is missing is open; this model has no mirror force (trapped ions at $`\epsilon`$ = 0.18) and a frozen primary.

Journal: 2026-10-11_ion-parallel-streaming-cuts-the-tsm-solver-s

**Approximations and limits.** No mirror force ($`v_\parallel`$ constant along the orbit: no trapped particles);
frozen primary; the finite-$`k_y`$ model needs periodic binormal coefficients, i.e. $`\hat s = 0`$ tubes
(`miller(..., shat=0)`; the $`\hat s\theta`$ secular terms of gds2/gds21/gbdrift would need twist-and-shift
coupling). The reference study applies the source operator after the inverse, $`J_0(\Delta_g\mathsf A^{-1}-1)J_0`$;
this changes $`\mathsf D`$ but not $`\mathcal D`$ in its circular model (checked to $`10^{-16}`$) and $`\mathcal D`$ by
~$`10^{-4}`$ relative in $`z`$-dependent geometry; the order here follows from the equation for $`h`$.

**Amplitude.** Amplitudes are always the physical E×B velocity $`\Phi_P F_{\theta,P}(z)`$ of `Primary`. The reference study
reads NP25 eq. (21) as a peak rather than an RMS and reports that the RMS reading overshoots NP25 Fig. 3(a) by
30–40 % in growth rate (its appendix H); this is its claim about the paper's figure labels, not tested here.

### Geometry for the new solvers

`gradpar(z)` (b·∇z) for streaming: tabulated geometries interpolate stella's array, `CircularAnalytic(q=…)` uses
$`1/(qR)`$, `AnalyticGeometry(gradpar=…)`. For finite $`k_y`$, `vMy(z)` = gbdrift/2 (per unit $`v_\parallel^2+v_\perp^2/2`$),
`gyy(z)` = gds2 = $`|\nabla y|^2`$, `gxy(z)` = gds21/ŝ = $`\nabla x\cdot\nabla y`$ and `b_k(z, kx, ky)` =
$`\tfrac12(k_x^2|\nabla x|^2 + 2k_xk_y\nabla x\cdot\nabla y + k_y^2|\nabla y|^2)/B^2`$ (stella's kperp2). The Miller port computes
gds2, gds21, gbdrift, cvdrift (agreeing with stella's `geometry_miller.*.output` to its $`5\times10^{-9}`$ print
precision) and $`\hat s`$-free $`|\nabla x|^2`$, $`v_{Mx}`$, $`\nabla x\cdot\nabla y`$ (so $`\hat s = 0`$ works); `from_stella_run` reads
them from `.out.nc` / `.geometry`; `AnalyticGeometry(vMy=…, gyy=…, gxy=…)`.

## Near marginality and the ISM region (`tsm_disp.diagnostics`)

Following the streaming study (report §7):

* `root_map(disp, kx, wr_range, wi_range)`: coarse $`|\mathcal D|`$ map → local minima → Muller: all roots in a box.
* `residue(disp, omega, kx)` = $`1/\mathcal D'(\omega_0)`$ (exact derivative for the λ form): the weight of the pole. A pole
  that dissolves into the drift-resonance continuum loses its residue while $`|\mathcal D|`$ stays small.
* `spectral_response(disp, kx, omega_r, eps)` = $`1/|\mathcal D(\omega_r + i\epsilon)|`$: needs no analytic continuation,
  so it is defined inside the ISM region.
* `winding_number(disp, kx, box)`: zeros − poles of $`\mathcal D`$ in a box; for `BinormalDispersion` the pole-free
  $`F = \mathcal D\det\mathsf D`$ counts roots only.
* `scan_amplitude(make_disp, amplitudes, omega0, kx)`: continuation in $`\Phi_P`$ with jump rejection; tracking **upward**
  from $`\Phi_P = 0`$ is much better conditioned (branches leave $`\Phi_P = 0`$ quadratically, $`\mathcal D`$ being even in $`\Phi_P`$).

With the streaming solvers keep the velocity contour (`VelocityGrid(cdepth, alpha)`) fixed along a track; switching it
on mid-track makes trackers hop branches. The deformed contour ($`v_\parallel = t - ic\tanh(t/w)`$: below the axis for
$`v_\parallel > 0`$, above it for $`v_\parallel < 0`$) is a valid continuation only for $`\omega_r > k_x\Phi_P\max F_{\theta,P}`$ (every
streaming resonance $`\omega - \omega_E = m v_\parallel\hat{\boldsymbol b}\cdot\nabla z`$ on the side where the contour dips). Below that it is
wrong even at $`\gamma > 0`$: at $`q`$ = 2.8, $`k_x\rho_i`$ = 0.08 ($`k_x\Phi_P\max F_{\theta,P}`$ = 0.028) it agrees with the real contour to
$`10^{-4}`$ at $`\omega`$ = 0.1 + 0.05i and 0.3 + 0.05i, but gives 0.055 + 0.015i against 0.048 at $`\omega`$ = 0.05i. Use the real
contour (`VelocityGrid()` default, $`\gamma > 0`$ only) for purely growing and slow modes.

## All roots in a box, without an initial guess (`tsm_disp.contour_roots`)

```python
from tsm_disp.contour_roots import find_all_roots
res = find_all_roots(disp, 0.5, (0.0, 1.2), (0.02, 0.5))   # k_x, ω_r range, γ range (v_th/a)
print(res.summary())   # roots: ω, |𝒟|, multiplicity, Res(1/𝒟), CR defect; poles; rejected; suspect regions; evaluations
```

For $`f(\omega) = \mathcal D(\omega, k_x)`$ analytic in a rectangle $`R`$ apart from poles, the argument principle and the
contour moments

```math
N = Z - P = \frac{1}{2\pi}\,\Delta_{\partial R}\arg f,\qquad
s_k = \frac{1}{2\pi i}\oint_{\partial R}\omega^k\,\frac{f'}{f}\,d\omega = \sum_{\rm zeros}\zeta^k - \sum_{\rm poles}p^k
    = \omega_0^k N - \frac{k}{2\pi i}\oint_{\partial R}\omega^{k-1}\log f\,d\omega
```

need only samples of $`f`$ on $`\partial R`$ ($`\log f`$ continued along $`\partial R`$ from $`\omega_0`$). The search:

1. **Boundary sampling.** Each edge is bisected until consecutive samples differ in phase by less than $`\pi/4`$; $`N`$
   is the sum of the phase steps. Samples sit on a dyadic lattice and are shared by neighbouring and child cells.
2. **Isolation.** A cell with more than 3 zeros is bisected across its longer side (a split line through a zero is
   shifted).
3. **Location.** In a cell with $`1 \le Z \le 3`$ zeros they are the roots of the polynomial with power sums
   $`s_1\dots s_Z`$ (Newton's identities), polished by Newton's method (exact $`\partial\mathcal D/\partial\omega`$) or
   Muller's, and accepted if they lie in the cell and reproduce $`s_1\dots s_{\max(Z,3)}`$ (this also gives the
   multiplicities). Cells without zeros must have $`s_{1,2,3} \approx 0`$ too, else they are bisected.
4. **Verification.** The Cauchy–Riemann defect $`|\partial_{\bar\omega} f|/|\partial_\omega f|`$ (four-point stencil) must be
   below $`10^{-3}`$; `residue` $`= 1/\mathcal D'(\omega_0)`$.

**The ISM region.** Where $`D_{\rm ISM}(z, y, \omega) = 0`$ for some real $`(z, y)`$, $`\mathcal D`$ of `DispersionRelation`
is continuous but not analytic, and its winding number is no zero count: the box below holds the TSM but has
$`N = 0`$. There the field-line quadrature cannot converge; the search samples with at most 32 panels (normally 8–14),
so such evaluations stop with flag 16 after 1–6 s (60 s at the default settings) and are flagged. Cells touching
flagged samples, or without zeros but with $`s_k \ne 0`$, are bisected down to `max_depth_suspect` = 6 and reported in
`res.suspect_regions`; polishes restart from their smallest unflagged $`|\mathcal D|`$ and are abandoned when they meet a
flagged evaluation. `local_ism_roots(disp, kx)` maps the region by the local roots $`\omega(z, y)`$ of $`D_{\rm ISM} = 0`$
(`examples/find_all_roots/find_all_roots.py` plots both; set `ETA_PAR = ETA_PERP = 1.5` there for the weaker drive).

**Poles.** `StreamingDispersion` has poles where $`\det\mathsf D(y_j) = 0`$ at a node of its $`y`$ rule, and zeros between
them (the discretised continuum). By default ($`N = Z - P`$) zero–pole pairs show up in $`s_{1,2,3}`$ and are bisected;
`use_det=True` samples the pole-free $`F = \mathcal D\prod_g\det\mathsf D(y_g)`$ (distinct matrices only;
`BinormalDispersion`: $`F = \mathcal D\det\mathsf D`$) to count zeros and poles separately, at 2× the samples or far more
next to the poles. `confirm=<the solver with more y nodes>` re-polishes every root and rejects the ones that move.
For the real velocity contour keep $`\gamma \gtrsim 0.05`$ (`default_validity` enforces $`\gamma > 0`$; for the tanh
contour $`\omega_r > k_x\Phi_P\max F_{\theta,P}`$).

Results (CBC $`q = 20`$, $`\Phi_P = 0.723`$, $`k_x\rho_i = 0.5`$, 8 threads; `examples/find_all_roots/`):

| solver, $`\eta`$ | box ($`\omega_r \times \gamma`$) | roots | suspect cells | evaluations | time |
|---|---|---|---|---|---|
| λ, 3 | $`[0, 1.2]\times[0.02, 0.5]`$ | TSM $`0.5808764 + 0.2927117i`$ | 11 (the ISM region) | 395 | 92 s |
| λ, 3 | $`[0.3, 1.2]\times[0.27, 0.5]`$ (above the ISM region) | TSM | 0 | 62 | 1.5 s |
| λ, 1.5 | $`[0, 1.2]\times[0.02, 0.5]`$ | TSM $`0.4973331 + 0.1263595i`$, 0.012 above the ISM tip | 6 | 414 | 95 s |
| λ, 1.1 | $`[0.3, 0.8]\times[0.02, 0.2]`$ | TSM $`0.4631526 + 0.0642792i`$, 0.002 from the ISM tip | 6 | 320 | 119 s |
| λ, 1.0 | $`[0.3, 0.8]\times[0.02, 0.2]`$ | none: the TSM has entered the ISM region ($`\gamma \le 0.065`$ flagged) | 6 | 350 | 114 s |
| streaming, 3 | $`[0.45, 0.75]\times[0.27, 0.4]`$ | $`0.5799333 + 0.2923443i`$ (= `benchmarks/stella_cbc`) | 0 | 47 | 203 s |

`diagnostics.root_map` on the first box (25 × 13 grid) finds the same TSM (and a damped root outside the box) with
343 evaluations in 102 s, but without a completeness guarantee or a flag on the ISM region.

**Limits.** Only cell boundaries are sampled: a non-analytic island that no boundary crosses is caught by the moment
test or by `probe=True` (Cauchy–Riemann test at every empty cell's centre). Without $`F`$ a zero–pole pair closer than
about `dipole_tol` × the cell size is invisible. A zero on the outer boundary is polished from the unresolved segment
(note "on a cell edge").

## More usage

```python
from tsm_disp.geometry import miller, vmec, from_stella_run, CircularAnalytic
from tsm_disp.roots import scan_kx

geo = miller(rhoc=0.5, rmaj=2.77778, qinp=20, shat=0.8, kappa=1.5, kapprim=0.3, tri=0.3, triprim=0.5,
             shift=-0.2, zed_equal_arc=True)                               # shaped Miller surface
geo = from_stella_run("/path/to/stella/run")                                 # <run>.out.nc (or .geometry) + input
geo = vmec("wout_precise_QH.nc", torflux=0.635, alpha0=-1.08, zeta_center=0.39,
           nfield_periods=1.45, nzed=1024)                                   # needs simsopt
geo = CircularAnalytic()                                                     # the model of the original code

disp = DispersionRelation(geo, pri, tau=1, eta_par=3, eta_prp=3, form="lambda",  # or form="Z" (small b, fast)
                          epsabs=1e-8, epsrel=1e-8, nthreads=32)
scan = scan_kx(disp, np.linspace(0.1, 1, 10), root.omega)                   # continuation in k_x
disp.D_ism(omega, kx, z, y)                                                  # local response, eq. (31)
from tsm_disp.primary import stella_phiP, A_P, heat_flux_stella              # amplitude conversions
```

**Drivers of the original code** (`code_eval_disp_NEW/disp_rel.py`): point their `sys.path.append(...)` to
this repository; `disp_rel.py` here accepts the old function names and keywords (also `omegaN=`;
`a_int`, `Nintegrate`, `tol_adaptive`, `sing_ITG` are ignored). Set `TSM_DISP_NTHREADS` for OpenMP threads.

## Build and test

```bash
python setup.py build_ext --inplace
python -m pytest -m fast        # ~1 min
python -m pytest -m slow        # brute-force quadratures, limits, VMEC (~3 min)
```

Tests: λ form against two independent velocity-space quadratures in both half-planes, including the
Landau continuation (1e-12); continuity of the continued $`N`$ across the real axis; the Z form as the $`O(b)`$
expansion (error ∝ $`b^2`$); J functions against 50-digit mpmath; the paper's exact RDK relation (B.5) to
1e-11; the RDK growth rate; the GAM frequency (50) with a $`k_x^2`$ approach; the non-resonant relation (52);
reality and local symmetries; Cauchy's integral of $`D`$ across $`{\rm Im}\,\omega=0`$; the regression value of the
original code (Z form, 1e-12); the geometry ports against stella output; the compatibility shim.

## stella benchmark (`benchmarks/stella_cbc/`)

Journal: 2026-10-08_secondary-mode-solver-rebuilt-old-lambda-form-had

CBC, $`q=20`$, $`\eta_\parallel=\eta_\perp=3`$, Gaussian primary (width π in zed), Θ = 0.2886 stella runs
(`fig_stella_q20.pdf`, `fig_stella_q20_dev.pdf`; data `data_caseA_label*.json`). Deviation from stella:

| variant | $`\omega_r`$ | $`\gamma`$ |
|---|---|---|
| Miller geometry, true stella amplitude, $`k_x\rho_i \ge 0.3`$ | +0.2 to +1.0 % | −0.4 to +2.2 % |
| same, $`k_x\rho_i = 0.1`$–0.2 | +0.8 to +2.8 % | +0.1 to +4.4 % |
| circular model, legacy amplitude (old inputs) | −2.3 to +1.1 % | −1.7 to +2.7 % |
| original code (stored roots) | −10 to +3.3 % | −9.4 to +8.8 % |

Roots are not followed into the ISM region ($`\min|D_{\rm ISM}|\to 0`$: $`k_x\rho_i \gtrsim 0.85`$ at $`\mathcal A=1`$,
$`\gtrsim 0.55`$ at $`\mathcal A = 4`$, where stella's frequency leaves the TSM branch).
At $`q = 2.8`$ (`fig_stella_q2.8.pdf`) the theory has no root near stella's modes: below $`k_x\rho_i \approx 1/2q`$
stella's RDK mode is not reproduced, as parallel streaming (neglected in the theory) matters there.

## Timing (`benchmarks/timing.py`, `timing.json`; error against a tight new reference)

| problem | old | new, 1 thread | new, 48 threads |
|---|---|---|---|
| Z-form D, regression point | 0.30 s (err 1e-12) | 0.03 s (err 1e-15) | |
| λ-form D, TSM point | 5.1 s (eps 1e-6; err 1.5e-2 from the bug) | 1.3 s (err 1e-16) | 0.07 s |
| TSM root, $`k_x = 0.5`$ | 62 s (lm; ω off by 9e-3) | 8.9 s (Muller) | 0.55 s |

Since 2026-10-11 (same cases, `benchmarks/timing.py` tolerances; measured back to back on a loaded node, so
compare the columns rather than the absolute numbers):

| problem | 2026-10-08 code, 1 / 16 threads | now, 1 / 16 threads |
|---|---|---|
| λ-form D, TSM point (tolerance 1e-8) | 1.34 s / 0.105 s | 0.21 s / 0.014 s |
| D together with ∂D/∂ω | — | 0.22 s / 0.015 s |
| TSM root from a guess 1 % off | 10.8 s, 7 evaluations (Muller) / 0.92 s | 1.35 s, 5 evaluations (Newton) / 0.10 s |

What changed, in order of gain: (i) the λ-integrand no longer calls libm's `cabs` (hypot) and the C99 complex
multiply/divide routines (inline arithmetic, `-fcx-fortran-rules` where the compiler accepts it), and the
geometry interpolation no longer wakes OpenBLAS threads (2.5×); (ii) the y-average uses the periodic trapezoid
rule with doubling (its error decays exponentially for the analytic periodic y-integrand; adaptive GK15
fallback; `y_rule="adaptive"` restores the old rule) and the λ tolerance follows the requested tolerance
(`tolN` = epsabs_y clamped to [1e-12, 1e-8]) (2.5×); (iii) `disp.find_root` uses Newton's method with
∂D/∂ω integrated on the same nodes as D (2 % extra), pole terms included (≈ 1.5× fewer evaluations). (iv) Since 2026-10-12 the
x86-64 build adds `-mavx2 -mfma`: λ-form D 0.218 → 0.185 s (1 thread) and 0.0142 → 0.0122 s (16 threads), results
unchanged to 1e-17 (the gain is from fused multiply-add and instruction selection; the adaptive, scalar
integrand does not vectorise, and AVX-512 (`-march=native`) gains nothing more).
Results agree with the previous code to ≤ 1e-11 (stella-benchmark roots: 4e-11 relative).

## Changes with respect to `code_eval_disp_NEW`

1. **Bug fixed (λ form):** $`\Gamma_0(\tilde b)=e^{-\tilde b}I_0(\tilde b)`$ for complex $`\tilde b`$ was computed with scipy's
   `ive`, which for complex argument is $`e^{-|{\rm Re}\,z|}I_0(z)`$. $`D_{\rm ISM}`$ was off by ~1e-3 at $`b\approx0.17`$,
   more at larger $`b`$ or weak growth.
2. **Z form:** large-$`|\zeta|`$ cancellation (up to 1e-8 noise in the J functions near $`|\zeta|=100`$) removed; the
   Stokes term below the real axis was missing a factor ζ and the quadratic term; ζ jumped sheet exactly on
   the real ω̄ axis.
3. **Old prescription for Im ω < 0** (real y axis split at $`{\rm Re}\,\bar\omega = 0`$, with the split points at the mirror
   image of the true crossings) is not analytic: D jumps by ~6e-2 across the real axis in the test case.
   Available as `y_contour="real_points"`.
4. **Speed:** one complex quadrature instead of real and imaginary parts separately at both levels, adaptive
   Gauss–Kronrod, Gauss–Laguerre rays, OpenMP over field-line nodes, periodic trapezoid rule in y, Newton root
   finding with the exact ∂D/∂ω (see Timing).
5. **Geometry:** general (Miller, VMEC, stella output); the circular model is `CircularAnalytic`.

## Layout

`tsm_disp/_kernels.pyx` (compiled kernels), `kernels_ref.py` (reference quadratures for tests),
`dispersion.py`, `streaming.py` (parallel streaming, finite k_y), `diagnostics.py`, `contour_roots.py` (all roots in a box), `roots.py`, `limits.py`,
`primary.py`, `geometry/`; `extern/streaming_ref/` (vendored reference of the streaming study); `tests/`; `benchmarks/stella_cbc/`
(stella comparison), `benchmarks/decks/` (shaped-Miller decks), `benchmarks/timing.py`; `examples/<name>/<name>.py` (parameter scripts) calling `evaluation_scripts/`; `scripts/`.
