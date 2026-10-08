# tsm_disp — dispersion relation of toroidal secondary modes

Solver for the generalised secondary-mode dispersion relation of Nies & Parra, *Theory of zonal flow
growth and propagation in toroidal geometry*, PPCF **68** (2026) 045028, in general flux-tube geometry
(Miller, VMEC, or the geometry of a stella run). It replaces `../code_eval_disp_NEW`, which stays
untouched as the reference implementation; `disp_rel.py` here is a drop-in replacement for old drivers.

## Equations

The secondary (zonal, wavenumber $k_x$) of a frozen streamer primary $v_{Ex}(z,y) = -\Phi_P T_P(z)\sin y$
obeys (paper eqs. 31–32)

```math
D(\omega,k_x) = 1-\Big\langle \frac{1}{D_{\rm ISM}}\Big\rangle = \int dz\, w(z)\,\frac{1}{2\pi}\int_{-\pi}^{\pi}dy\,\frac{D_{\rm ISM}-1}{D_{\rm ISM}} = 0,
\qquad D_{\rm ISM} = \frac{1-\Gamma_0(b)+\tau-N}{\tau},
```

with $w = 1/(\mathbf B\cdot\nabla z)$ normalised to $\int w\,dz = 1$, $b = k_x^2|\nabla x|^2\rho^2/2$ and the ion response

```math
N=\int d^3\hat v\,\hat F_M J_0^2(\sqrt{2b}\,\hat v_\perp)\,
\frac{\omega_1+2\omega_2\hat v_\parallel^2+\omega_3\hat v_\perp^2}{\bar\omega-\omega_{M}(\hat v_\parallel^2+\hat v_\perp^2/2)},\qquad
\begin{aligned}
&\bar\omega=\omega-\omega_E,\quad \omega_E = k_x v_{Ex},\quad \omega_M = k_x v_{Mx},\\
&\omega_1=\tau(-\omega_E+\tfrac12\eta_\parallel\omega_{E\parallel}+\eta_\perp\omega_{E\perp}),\quad
\omega_2=\tfrac12(\omega_M-\tau\eta_\parallel\omega_{E\parallel}),\quad \omega_3=\tfrac12\omega_M-\tau\eta_\perp\omega_{E\perp},
\end{aligned}
```

where $\omega_{E\parallel,\perp} = \omega_E(y+\Delta_{\parallel,\perp})$ carry optional phase shifts. Two evaluations of $N$:

* **λ form** (`form="lambda"`, old `disp_model < 0`): the Laplace representation (paper eq. 34), exact in $b$.
  The λ contour is chosen per point (straight ray $\lambda = is/\bar\omega$ with Gauss–Laguerre or adaptive
  Gauss–Kronrod; a real segment plus ray near the resonance; a detour around the singularities
  $\lambda = i/\omega_M,\,2i/\omega_M$ for the Landau continuation to ${\rm Im}\,\bar\omega<0$).
* **Z form** (`form="Z"`, old `disp_model = 0.1`): the $O(b)$ closed form in the plasma dispersion function,
  $\zeta=\sqrt{\bar\omega/\omega_M}$ on the sheet continuous with ${\rm Im}\,\bar\omega>0$. For $|\zeta|\ge 6$ the J
  functions use a 48-term cancellation-free asymptotic series (`scripts/derive_J_asymptotics.py`).

**Units** (stella): lengths in $a$, wavenumbers in $1/\rho_{\rm ref}$, velocities in $v_{th}=\sqrt{2T/m}$,
frequencies in $v_{th}/a$. Multiply frequencies by $R/a$ for the paper's $v_{Ti}/R$.

**Analytic continuation.** For ${\rm Im}\,\omega\le 0$ the $y$ integral runs on $y = t + i\,{\rm sgn}(A)\,\eta\cos t$,
which keeps the branch points ($\bar\omega = 0$) on the side they occupy for ${\rm Im}\,\omega>0$. This is the analytic
continuation of $D$ wherever $D_{\rm ISM}$ has no zeros near the contour. Inside the paper's ISM region
($D_{\rm ISM}=0$ somewhere on the surface) $D$ is not analytic, and no contour is attempted (paper §3.3).
Near-real zeros of $D_{\rm ISM}(y)$ are located by Newton's method and their poles subtracted analytically
(flag 64), which keeps the cost bounded near the ISM region.

## Geometry (`tsm_disp.geometry`)

Only $k_y=0$ quantities enter: along the field-line coordinate $z$,

```math
v_{Mx} = \frac{\texttt{gbdrift0}}{2\hat s},\qquad
\frac{|\nabla x|^2}{B^2} = \frac{\texttt{gds22}}{\hat s^2\,\texttt{bmag}^2},\qquad
w \propto \frac{1}{\texttt{bmag}\cdot\texttt{gradpar}} ,
```

in stella's conventions: stella advances $\partial_t g + i\omega_d g = \dots$ with
$\omega_d = (k_x/\hat s)(\texttt{cvdrift0}\,v_\parallel^2+\texttt{gbdrift0}\,v_\perp^2/2)/2$ and $\phi\propto e^{-i\omega t}$,
and $\texttt{cvdrift0}=\texttt{gbdrift0}$. For the circular CBC this gives $v_{Mx} = -\sin\theta/R$, the mirror
image of the old `T_D = sin θ/2.7778`; the two are equivalent for up-down-symmetric problems.

| constructor | source | verified against |
|---|---|---|
| `CircularAnalytic(eps, rmaj, vMx)` | the model of the original code | itself (regression) |
| `Uniform(vMx=0)` | slab / RDK limit | paper eq. (B.5) |
| `miller(rhoc, rmaj, rgeo, shift, qinp, shat, kappa, kapprim, tri, triprim, …, zed_equal_arc)` | port of stella's `geometry_miller.f90` | stella's `geometry_miller.*.output` (≤ 5e-9, its print precision) and `.out.nc` of equal-arc runs (≤ 1e-12) |
| `vmec(wout, torflux, alpha0, zeta_center, nfield_periods, nzed)` | simsopt `vmec_fieldlines`, stella's grid | stella's precise-QH run (≤ 2e-6) |
| `from_stella_run(run_dir)` | `<run>.out.nc` (or `<run>.geometry`, 4 digits) + input | Miller port (1e-6) |

Tokamak tubes with `nperiod = 1` are periodic (Fourier interpolation); VMEC tubes are not (field-line average).

## Amplitude conventions

stella's `secondary = .true.` initialises and freezes the primary coefficient
$g = \texttt{phiinit}\,\phi_0(z)[\texttt{den0}+\texttt{tpar0}(v_\parallel^2-\tfrac12)+\texttt{tperp0}(v_\perp^2-1)]F_M$ at $k_y^P = 1/y_0$,
whose potential has real-space amplitude $2|\hat\phi|$ and radial E×B velocity
$\texttt{exb\_nonlin\_fac}\,\partial_y\phi$ (the ½ of $v_{th}^2=2T/m$ is inside `exb_nonlin_fac`). Hence

```math
\Phi_P\,T_P(z) = 2\,\texttt{exb\_nonlin\_fac}\;k_y^P\,|\hat\phi(z)|,\qquad \eta_\parallel=\texttt{tpar0},\ \eta_\perp=\texttt{tperp0}
```

(`primary.stella_phiP`). For the CBC surface $2\,\texttt{exb\_nonlin\_fac} = 1/\sqrt{1-\epsilon^2} = 1.0166$.
The historical drivers used $\Phi_P = k_y^P\,\texttt{phiinit}\,\texttt{fac\_norm}$, so their solver curves had a 1.66 %
weaker drive than the stella runs they were compared with. Their plot label is $\mathcal A = \Phi_P/0.3557$;
0.3557 is the true amplitude of the reference deck phiinit = 35, $y_0 = 100$ ($2\cdot 0.5083\cdot 0.01\cdot 35 = 0.3558$).

The paper's eq. (21), $A^P = \sqrt{\langle (v_{Ex}R/v_{Ti}\rho_i)^2\rangle_\psi}$ with the average over $z$ and $y$,
is `primary.A_P`. For the Gaussian CBC primary, $A^P = (R/a)\,\Phi_P\sqrt{\langle T_P^2\rangle/2} = 2.024\,\Phi_P$,
so a legacy label $\mathcal A$ corresponds to $A^P = 0.732\,\mathcal A$. The labels follow a different convention
from eq. (21), and the paper's quoted values should be checked against it.

## Usage

```python
import numpy as np
from tsm_disp.geometry import miller
from tsm_disp.primary import Primary
from tsm_disp.dispersion import DispersionRelation
from tsm_disp.roots import find_root, scan_kx

geo = miller(rhoc=0.5, rmaj=2.77778, qinp=20, shat=0.8, kappa=1.5, tri=0.3, zed_equal_arc=True)
pri = Primary.gaussian(Phi_P=0.72, width=np.pi)
disp = DispersionRelation(geo, pri, tau=1, eta_par=3, eta_prp=3, form="lambda", epsabs=1e-8, epsrel=1e-8, nthreads=32)
root = find_root(lambda w: disp.D(w, 0.5), 0.58 + 0.29j)
scan = scan_kx(disp, np.linspace(0.1, 1, 10), root.omega)
```

Old drivers: replace their `sys.path.append(.../code_eval_disp_NEW)` by this directory. The shim accepts the
old keywords (also `omegaN=`); `a_int`, `Nintegrate`, `tol_adaptive`, `sing_ITG` are ignored. Set
`TSM_DISP_NTHREADS` to use OpenMP threads.

## Build and test

```bash
python setup.py build_ext --inplace
python -m pytest -m fast        # ~1 min
python -m pytest -m slow        # brute-force quadratures, limits, VMEC (~3 min)
```

Tests: λ form against two independent velocity-space quadratures in both half-planes, including the
Landau continuation (1e-12); continuity of the continued $N$ across the real axis; the Z form as the $O(b)$
expansion (error ∝ $b^2$); J functions against 50-digit mpmath; the paper's exact RDK relation (B.5) to
1e-11; the RDK growth rate; the GAM frequency (50) with a $k_x^2$ approach; the non-resonant relation (52);
reality and local symmetries; Cauchy's integral of $D$ across ${\rm Im}\,\omega=0$; the regression value of the
original code (Z form, 1e-12); the geometry ports against stella output; the compatibility shim.

## stella benchmark (`benchmarks/stella_cbc/`)

Journal: 2026-10-08_secondary-mode-solver-rebuilt-old-lambda-form-had

CBC, $q=20$, $\eta_\parallel=\eta_\perp=3$, Gaussian primary (width π in zed), Θ = 0.2886 stella runs
(`fig_stella_q20.pdf`, `fig_stella_q20_dev.pdf`; data `data_caseA_label*.json`). Deviation from stella:

| variant | $\omega_r$ | $\gamma$ |
|---|---|---|
| Miller geometry, true stella amplitude, $k_x\rho_i \ge 0.3$ | +0.2 to +1.0 % | −0.4 to +2.2 % |
| same, $k_x\rho_i = 0.1$–0.2 | +0.8 to +2.8 % | +0.1 to +4.4 % |
| circular model, legacy amplitude (old inputs) | −2.3 to +1.1 % | −1.7 to +2.7 % |
| original code (stored roots) | −10 to +3.3 % | −9.4 to +8.8 % |

Roots are not followed into the ISM region ($\min|D_{\rm ISM}|\to 0$: $k_x\rho_i \gtrsim 0.85$ at $\mathcal A=1$,
$\gtrsim 0.55$ at $\mathcal A = 4$, where stella's frequency leaves the TSM branch).
At $q = 2.8$ (`fig_stella_q2.8.pdf`) the theory has no root near stella's modes: below $k_x\rho_i \approx 1/2q$
stella's RDK mode is not reproduced, as parallel streaming (neglected in the theory) matters there.

## Timing (`benchmarks/timing.py`, `timing.json`; error against a tight new reference)

| problem | old | new, 1 thread | new, 48 threads |
|---|---|---|---|
| Z-form D, regression point | 0.30 s (err 1e-12) | 0.03 s (err 1e-15) | |
| λ-form D, TSM point | 5.1 s (eps 1e-6; err 1.5e-2 from the bug) | 1.3 s (err 1e-16) | 0.07 s |
| TSM root, $k_x = 0.5$ | 62 s (lm; ω off by 9e-3) | 8.9 s (Muller) | 0.55 s |

## Changes with respect to `code_eval_disp_NEW`

1. **Bug fixed (λ form):** $\Gamma_0(\tilde b)=e^{-\tilde b}I_0(\tilde b)$ for complex $\tilde b$ was computed with scipy's
   `ive`, which for complex argument is $e^{-|{\rm Re}\,z|}I_0(z)$. $D_{\rm ISM}$ was off by ~1e-3 at $b\approx0.17$,
   more at larger $b$ or weak growth.
2. **Z form:** large-$|\zeta|$ cancellation (up to 1e-8 noise in the J functions near $|\zeta|=100$) removed; the
   Stokes term below the real axis was missing a factor ζ and the quadratic term; ζ jumped sheet exactly on
   the real ω̄ axis.
3. **Old prescription for Im ω < 0** (real y axis split at ${\rm Re}\,\bar\omega = 0$, with the split points at the mirror
   image of the true crossings) is not analytic: D jumps by ~6e-2 across the real axis in the test case.
   Available as `y_contour="real_points"`.
4. **Speed:** one complex quadrature instead of real and imaginary parts separately at both levels, adaptive
   Gauss–Kronrod, Gauss–Laguerre rays, OpenMP over field-line nodes, Muller root finding (see
   `benchmarks/timing.py`).
5. **Geometry:** general (Miller, VMEC, stella output); the circular model is `CircularAnalytic`.

## Layout

`tsm_disp/_kernels.pyx` (compiled kernels), `kernels_ref.py` (reference quadratures for tests),
`dispersion.py`, `roots.py`, `limits.py`, `primary.py`, `geometry/`; `tests/`; `benchmarks/stella_cbc/`
(stella comparison), `benchmarks/decks/` (shaped-Miller decks), `benchmarks/timing.py`; `scripts/`.
