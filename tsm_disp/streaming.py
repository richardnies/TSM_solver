"""Secondary dispersion relation with ion parallel streaming (Fourier truncation along the field line).

The solver of dispersion.py neglects parallel streaming (q → ∞): the local response D_ISM(θ, y) is algebraic in θ.
Retaining v∥ b·∇ makes θ a coordinate: the gyrocentre distribution h of the secondary obeys, at each velocity and
primary phase y,

    A h = Δ_g J0 φ,    A = ω̄(z, y) − ω_M(z) E − v∥ b·∇z(z) (−i∂_z),    Δ_g = ω̄ − ω_g,

with ω̄ = ω − ω_E, ω_E = −k_x Φ_P T_P(z) sin y, E = v∥² + v⊥²/2, ω_M = k_x v_Mx(z), and the primary drive
ω_g = τ[ω_E + η∥ ω_E∥ (v∥² − ½) + η⊥ ω_E⊥ (v⊥² − 1)] (ω_E∥,⊥ at y + Δ∥,⊥). Writing g = Σ_m g_m e^{imz}, every
z-dependent coefficient is a Toeplitz matrix in m (T[f]_{mm'} = f̂_{m−m'}), b·∇z(−i∂_z) is T[b·∇z] diag(m), and
J0, Γ0 (through b(z)), T_P and the field-line weight w(z) likewise. Quasineutrality with modified adiabatic
electrons, truncated to |m| ≤ M for h and |m| ≤ M_φ for φ, gives at each y

    D(y) φ = τ Φ e_0,    D = (1 + τ) I − T[Γ0] − Σ_v W_v J0_v (A_v⁻¹ Δ_v − I) J0_v,

and with Φ = ⟨φ⟩_ψ = ⟨c·φ(y)⟩_y, c_m = ∫ w(z) e^{imz} dz, the dispersion relation

    𝒟(ω) = 1 − τ ⟨ cᵀ D(y, ω)⁻¹ e_0 ⟩_y = 0.

Without streaming (b·∇z → 0) every operator is a multiplication in z and 𝒟 → 1 − ⟨1/D_ISM⟩, eq. (32) of Nies &
Parra (2026), solved by dispersion.DispersionRelation. In the circular model (v_Mx ∝ sin z, b·∇z = 1/(qR), b, B,
w constant, flat primary) A is tridiagonal and this is the solver of the vendored reference
extern/streaming_ref/tsm_core.py (Usmani/log-ratio moments, kept here as the fast path).

Approximations: no mirror force (v∥ constant along the orbit: passing particles, no trapping); the secondary is
zonal (k_y = 0; see BinormalDispersion for finite primary k_y); frozen primary. Velocity space: Gauss–Legendre in v∥
on the Landau contour v∥ = t − i c tanh(t/w) and Gauss–Laguerre in u = v⊥² on the ray u = s e^{−iα}; c > 0 and
α > 0 continue 𝒟 below the real ω axis (transit and drift resonances). Keep the grid FIXED when following a root
through marginality. Units as in the rest of the package: lengths in a, velocities in v_th = √(2T/m),
frequencies in v_th/a.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.special import ive, jv, roots_legendre, roots_laguerre


class VelocityGrid:
    """Velocity quadrature ⟨f⟩_v = π^{−1/2}∫dv∥ e^{−v∥²}∫du e^{−u} f on deformable contours (fixed at construction).

    n_vpar Gauss–Legendre nodes on [−vpar_max, vpar_max] mapped to v∥ = t − i cdepth tanh(t/width) (Maxwellian
    carried explicitly), n_u Gauss–Laguerre nodes on u = s e^{−iα}. cdepth = alpha = 0: real axes (Im ω > 0 only).

    The tanh contour lies below the axis for v∥ > 0 and above it for v∥ < 0, so it continues the streaming
    resonances correctly only if Re(ω − ω_E) > 0 everywhere, i.e. ω_r > k_x Φ_P max T_P. For slower or purely
    growing modes it gives wrong values even at Im ω > 0 (q = 2.8 CBC, k_x = 0.08, k_x Φ_P max T_P = 0.028: equal to the
    real contour to 1e-4 at ω = 0.1 + 0.05i and 0.3 + 0.05i, but 0.055 + 0.015i against 0.048 at ω = 0.05i); use the real
    contour there.
    """

    def __init__(self, n_vpar=96, n_u=12, vpar_max=6.0, cdepth=0.0, width=0.7, alpha=0.0):
        t, wt = roots_legendre(n_vpar)
        t, wt = vpar_max * t, wt * vpar_max
        if cdepth == 0.0:
            vpar, jac = t.astype(complex), np.ones(n_vpar, dtype=complex)
        else:
            vpar = t - 1j * cdepth * np.tanh(t / width)
            jac = 1.0 - 1j * cdepth / width / np.cosh(t / width) ** 2
        s, ws = roots_laguerre(n_u)
        rot = np.exp(-1j * alpha)
        u = s * rot
        self.vpar, self.wv = vpar, wt * jac * np.exp(-vpar ** 2) / np.sqrt(np.pi)
        self.u, self.wu = u, ws * rot * np.exp(s - u)          # Gauss–Laguerre carries e^{−s}; correct to e^{−u}
        VP, UU = np.meshgrid(vpar, u, indexing="ij")
        self.VP, self.UU = VP.reshape(-1), UU.reshape(-1)
        self.W = (self.wv[:, None] * self.wu[None, :]).reshape(-1)
        self.iu = np.tile(np.arange(n_u), n_vpar)               # u-node index of each flat node
        self.N = self.VP.size
        self.params = dict(n_vpar=n_vpar, n_u=n_u, vpar_max=vpar_max, cdepth=cdepth, width=width, alpha=alpha)

    def __repr__(self):
        return "VelocityGrid(%s)" % ", ".join("%s=%g" % kv for kv in self.params.items())


# ----------------------------------------------------------------------------------------------------------------
# Fourier/Toeplitz helpers on a periodic tube z ∈ [−π, π)
# ----------------------------------------------------------------------------------------------------------------
def _zgrid(nz):
    return -np.pi + 2 * np.pi * np.arange(nz) / nz


def _harmonics(fvals, J):
    """f̂_j, j = −J … J, of f(z) = Σ_j f̂_j e^{ijz} sampled on _zgrid(nz) (last axis); returns (..., 2J+1)."""
    nz = fvals.shape[-1]
    if nz < 2 * J + 1:
        raise ValueError("too few samples for the requested harmonics")
    c = np.fft.fft(fvals, axis=-1) / nz
    j = np.arange(-J, J + 1)
    return c[..., j % nz] * np.where(j % 2, -1.0, 1.0)       # z_0 = −π: phase (−1)^j


def _toeplitz(fvals, nm):
    """T[f]_{ab} = f̂_{m_a − m_b} for m = −(nm−1)/2 … (nm−1)/2 (the matrix of multiplication by f)."""
    c = _harmonics(fvals, nm - 1)
    a = np.arange(nm)
    return c[..., (a[:, None] - a[None, :]) + nm - 1]


# ----------------------------------------------------------------------------------------------------------------
# Tridiagonal fast path (port of extern/streaming_ref/tsm_core.py, _tridiag_block_moments)
# ----------------------------------------------------------------------------------------------------------------
def tridiag_block_moments(sub, diag, sup, cols, weight):
    """Σ_n weight[n] (A_n⁻¹)[cols, cols] for tridiagonal A_n with constant sub/super-diagonals (per node).

    Usmani's closed form (A⁻¹)_ij = (−sup)^{j−i} θ_{i−1}φ_{j+1}/θ_n (i ≤ j) with the minors carried as ratios
    r_i = d_i − sub·sup/r_{i−1}, s_i = d_i − sub·sup/s_{i+1} and cumulative logs (bounded); the node sum is two
    small matrix products. sub, sup, weight: (N,); diag: (N, nm)."""
    N, nm = diag.shape
    P = sub * sup
    cols = np.asarray(cols)
    nc = cols.size
    r = np.empty((nm, N), dtype=complex)
    s = np.empty((nm, N), dtype=complex)
    r[0] = diag[:, 0]
    for i in range(1, nm):
        r[i] = diag[:, i] - P / r[i - 1]
    s[nm - 1] = diag[:, nm - 1]
    for i in range(nm - 2, -1, -1):
        s[i] = diag[:, i] - P / s[i + 1]
    Lr = np.cumsum(np.log(r)[::-1], axis=0)[::-1]
    Ls = np.zeros((nm + 1, N), dtype=complex)
    Ls[:nm] = np.cumsum(np.log(s)[::-1], axis=0)[::-1]
    Lsj = Ls[1:]
    ref = 0.5 * (Lsj[cols[nc // 2]] + Lr[cols[nc // 2]])
    e = np.arange(nc)
    msup = (-sup)[None, :] ** e[:, None]
    msub = (-sub)[None, :] ** e[:, None]
    alpha = np.exp(ref[None, :] - Lr[cols]) / msup
    beta = np.exp(Lsj[cols] - ref[None, :]) * msup
    alphaL = np.exp(ref[None, :] - Lr[cols]) / msub
    betaL = np.exp(Lsj[cols] - ref[None, :]) * msub
    upper = (alpha * weight[None, :]) @ beta.T
    lower = (betaL * weight[None, :]) @ alphaL.T
    iu = np.triu(np.ones((nc, nc)))
    return upper * iu + lower * (1.0 - iu)


class StreamingDispersion:
    """𝒟(ω, k_x) with ion parallel streaming on a periodic field-line tube (see the module docstring).

    geometry : periodic Geometry on z ∈ [−π, π] providing vMx, b, weight and gradpar (b·∇z): Miller /
        stella-output tubes with nperiod = 1, CircularAnalytic(q=...), AnalyticGeometry(gradpar=...).
    primary : tsm_disp.primary.Primary (T_P(z) and Φ_P)
    M, Mphi : harmonics |m| ≤ M for h (default max(⌈6 + 2.2 k_x max|v_Mx|/⟨b·∇z⟩⌉, Mphi + 8): the transit
        criterion M > ω_M/(v∥ b·∇z)) and |m| ≤ Mphi for φ. Mphi = 5–6 suffice for a flat primary on the circular
        model, but a localised secondary needs more: on CBC with a Gaussian primary Mphi = 8, 12, 16 give relative
        errors 2e-2, 8e-4, 7e-5 in 𝒟 (no-streaming limit against DispersionRelation).
    n_y : uniform quadrature in the primary phase y.
    grid : VelocityGrid (default 96 × 12, real contours: growing modes; pass cdepth, alpha > 0 for damped roots).
    path : 'auto' (tridiagonal fast path when the geometry is the circular model), 'tridiagonal', 'dense'.
    streaming : False drops v∥ b·∇ (then 𝒟 → the no-streaming 𝒟 of DispersionRelation as M → ∞).
    """

    def __init__(self, geometry, primary, *, tau=1.0, eta_par=0.0, eta_prp=0.0, delta_par=0.0, delta_prp=0.0,
                 M=None, Mphi=16, n_y=48, grid=None, nz=None, path="auto", streaming=True, nthreads=1,
                 chunk_bytes=6e7, Mmax=96):
        if not getattr(geometry, "periodic", False) or abs(geometry.zmin + np.pi) > 1e-9 \
                or abs(geometry.zmax - np.pi) > 1e-9:
            raise ValueError("the streaming solver needs a periodic tube z ∈ [−π, π] (one poloidal turn)")
        self.geometry, self.primary = geometry, primary
        self.tau, self.eta_par, self.eta_prp = float(tau), float(eta_par), float(eta_prp)
        self.delta_par, self.delta_prp = float(delta_par), float(delta_prp)
        self.M, self.Mphi, self.n_y = M, int(Mphi), int(n_y)
        self.grid = grid if grid is not None else VelocityGrid()
        self.nz, self.path, self.streaming = nz, path, bool(streaming)
        self.nthreads, self.chunk_bytes, self.Mmax = int(nthreads), float(chunk_bytes), int(Mmax)
        self.Y = 2 * np.pi * np.arange(self.n_y) / self.n_y
        self._cache = {}
        self.last = {}

    # -- k_x-dependent operators ----------------------------------------------------------------------------------
    def harmonics_needed(self, kx):
        z = _zgrid(512)
        vm = np.max(np.abs(self.geometry.vMx(z)))
        gp = np.mean(np.abs(self.geometry.gradpar(z))) if self.streaming else 0.0
        if gp == 0.0:
            return self.Mmax
        return int(min(self.Mmax, np.ceil(6 + 2.2 * kx * vm / gp)))

    def _setup(self, kx):
        key = float(kx)
        if key in self._cache:
            return self._cache[key]
        g, p, vg = self.geometry, self.primary, self.grid
        M = self.M if self.M is not None else max(self.harmonics_needed(kx), self.Mphi + 8)
        Mphi = min(self.Mphi, M)
        nm = 2 * M + 1
        ms = np.arange(-M, M + 1)
        win = np.arange(M - Mphi, M + Mphi + 1)
        nz = self.nz or max(256, 8 * nm)
        z = _zgrid(nz)
        vM = np.asarray(g.vMx(z), float) * np.ones(nz)
        bz = np.asarray(g.b(z, kx), float) * np.ones(nz)
        wz = np.asarray(g.weight(z), float) * np.ones(nz)
        gp = np.asarray(g.gradpar(z), float) * np.ones(nz) if self.streaming else np.zeros(nz)
        TP = np.asarray(p(z), float) * np.ones(nz)
        s = dict(kx=kx, M=M, Mphi=Mphi, nm=nm, ms=ms, win=win, nphi=win.size, i0=Mphi)
        # flux-surface-average vector c_m = ∫ w e^{imz} dz = 2π ŵ_{−m}
        wh = _harmonics(wz, Mphi)
        s["c"] = 2 * np.pi * wh[::-1]
        hv = _harmonics(vM, 8)                                   # j = −8 … 8
        scale = lambda f: np.max(np.abs(f)) + 1e-300
        flat = lambda f: np.ptp(f) < 1e-12 * scale(f)
        circular = (np.max(np.abs(hv[np.abs(np.arange(-8, 9)) >= 2])) < 1e-12 * scale(vM)
                    and flat(bz) and flat(wz) and flat(gp) and flat(TP))
        use_tri = (self.path == "tridiagonal") or (self.path == "auto" and circular)
        if self.path == "tridiagonal" and not circular:
            raise ValueError("the tridiagonal path needs the circular model (v_Mx ∝ e^{±iz}, constant b, w, b·∇z, T_P)")
        s["tri"] = use_tri
        if use_tri:
            s["kappa"] = kx * p.Phi_P * TP[0]
            s["vM0"], s["vMp"], s["vMm"] = kx * hv[8], kx * hv[9], kx * hv[7]     # f̂_0, f̂_{+1}, f̂_{−1}
            s["gp"] = gp[0]
            s["j0"] = jv(0, np.sqrt(2 * bz[0]) * np.sqrt(vg.UU))
            s["Gamma0"] = ive(0, bz[0])
        else:
            s["Tv"] = kx * _toeplitz(vM, nm)
            s["S0"] = _toeplitz(gp, nm) * ms[None, :]
            s["K"] = kx * p.Phi_P * _toeplitz(TP, nm)
            J0T = _toeplitz(jv(0, np.sqrt(2 * bz)[None, :] * np.sqrt(vg.u)[:, None]), nm)   # (n_u, nm, nm)
            s["J0w"] = np.ascontiguousarray(J0T[:, :, win])                                  # (n_u, nm, nphi)
            s["wJ0"] = np.ascontiguousarray(J0T[:, win, :])                                  # (n_u, nphi, nm)
            s["KJ0w"] = np.einsum("ab,ubc->uac", s["K"], s["J0w"])
            s["J0J0"] = np.einsum("uab,ubc->uac", s["wJ0"], s["J0w"])                       # (n_u, nphi, nphi)
            s["Gamma0"] = _toeplitz(ive(0, bz), nm)[np.ix_(win, win)]
        self._cache[key] = s
        return s

    # -- evaluation -------------------------------------------------------------------------------------------------
    def _drive_coefficient(self, Y):
        """s_v(y): Δ_v = ω + s_v K, with s_v = (1+τ) sin y + τη∥(v∥²−½) sin(y+Δ∥) + τη⊥(v⊥²−1) sin(y+Δ⊥)."""
        vg = self.grid
        return ((1 + self.tau) * np.sin(Y)[:, None]
                + self.tau * self.eta_par * (vg.VP ** 2 - 0.5)[None, :] * np.sin(Y + self.delta_par)[:, None]
                + self.tau * self.eta_prp * (vg.UU - 1.0)[None, :] * np.sin(Y + self.delta_prp)[:, None])

    def _Dmat_tri(self, s, omega, iy):
        vg = self.grid
        Y = self.Y[iy]
        sv = self._drive_coefficient(np.atleast_1d(Y))[0]
        E = vg.VP ** 2 + 0.5 * vg.UU
        diag = (omega + np.sin(Y) * s["kappa"] - E * s["vM0"])[:, None] - (s["gp"] * vg.VP)[:, None] * s["ms"][None, :]
        sub = -E * s["vMp"]                      # A[m, m−1] = −E k_x v̂_{+1}
        sup = -E * s["vMm"]                      # A[m, m+1] = −E k_x v̂_{−1}
        wJ2 = vg.W * s["j0"] ** 2
        Delta = omega + sv * s["kappa"]
        Nm = tridiag_block_moments(sub, diag, sup, s["win"], wJ2 * Delta) - np.sum(wJ2) * np.eye(s["nphi"])
        return (1.0 - s["Gamma0"] + self.tau) * np.eye(s["nphi"]) - Nm

    def _Dmat_dense(self, s, omega, iy):
        vg = self.grid
        Y = self.Y[iy]
        sv = self._drive_coefficient(np.atleast_1d(Y))[0]
        nm, nphi = s["nm"], s["nphi"]
        A0 = omega * np.eye(nm) + np.sin(Y) * s["K"]
        Msum = np.zeros((nphi, nphi), dtype=complex)
        Q = np.zeros((nphi, nphi), dtype=complex)
        E = vg.VP ** 2 + 0.5 * vg.UU
        nv = max(1, int(self.chunk_bytes / (16.0 * nm * nm)))
        for i0 in range(0, vg.N, nv):
            sl = slice(i0, min(i0 + nv, vg.N))
            iu = vg.iu[sl]
            A = A0[None] - E[sl, None, None] * s["Tv"][None] - vg.VP[sl, None, None] * s["S0"][None]
            R = omega * s["J0w"][iu] + sv[sl, None, None] * s["KJ0w"][iu]
            X = np.linalg.solve(A, R)                                    # A⁻¹ Δ J0 on the φ window
            Msum += np.einsum("v,vab,vbc->ac", vg.W[sl], s["wJ0"][iu], X)
            Q += np.einsum("v,vab->ab", vg.W[sl], s["J0J0"][iu])
        return (1.0 + self.tau) * np.eye(nphi) - s["Gamma0"] - (Msum - Q)

    def Dmatrix(self, omega, kx, iy):
        """D(y_iy, ω) on the φ window |m| ≤ M_φ."""
        s = self._setup(kx)
        return self._Dmat_tri(s, complex(omega), iy) if s["tri"] else self._Dmat_dense(s, complex(omega), iy)

    def D(self, omega, kx):
        """𝒟(ω, k_x) = 1 − τ ⟨cᵀ D(y)⁻¹ e_0⟩_y."""
        s = self._setup(kx)
        omega = complex(omega)
        e0 = np.zeros(s["nphi"])
        e0[s["i0"]] = 1.0

        def one(iy):
            Dm = self.Dmatrix(omega, kx, iy)
            return s["c"] @ np.linalg.solve(Dm, e0)

        if self.nthreads > 1:
            with ThreadPoolExecutor(self.nthreads) as ex:
                vals = list(ex.map(one, range(self.n_y)))
        else:
            vals = [one(i) for i in range(self.n_y)]
        self.last = dict(M=s["M"], Mphi=s["Mphi"], path="tridiagonal" if s["tri"] else "dense")
        return 1.0 - self.tau * np.mean(vals)

    def find_root(self, omega0, kx, **kw):
        """Root of 𝒟(·, k_x) near omega0 by Muller's method (tsm_disp.roots.find_root)."""
        from .roots import find_root
        return find_root(lambda w: self.D(w, kx), omega0, **kw)


class BinormalDispersion:
    """𝒟(ω, k_x) with parallel streaming AND the primary's finite binormal wavenumber k_y^P (no y-parameter).

    The secondary is expanded as g = Σ_{n,m} g_{nm} e^{i n y + i m z}, |n| ≤ N (sidebands with k_y = n k_y^P),
    |m| ≤ M. Relative to StreamingDispersion this adds, on each sideband n, the binormal magnetic drift
    n k_y^P v_My(z) E, FLR at b_n(z) = ½[k_x²|∇x|² + 2 k_x n k_y^P ∇x·∇y + (n k_y^P)²|∇y|²]/B², and the gyroaverage
    J0(√(2 b_P) v⊥), b_P = ½(k_y^P)²|∇y|²/B², of the primary acting on the secondary. The primary (∝ sin y) couples
    n → n ± 1, so A_v = ω − P_v − K_E,v is block-tridiagonal in n with Toeplitz blocks in m, and

        D = (1 + τ) I − Γ0 − Σ_v W_v J0_v (A_v⁻¹ Δ_v − I) J0_v,     𝒟 = 1 − τ cᵀ D⁻¹ e_{(0,0)},

    with Δ_v = ω − K_E,v − K_g,v (Δ applied BEFORE the inverse: the h equation is A h = Δ J0 φ). The vendored
    reference extern/streaming_ref/tsm_ky.py applies it after (Δ A⁻¹); the matrices D then differ, but 𝒟 agrees to
    round-off in its circular model (uniform weight and FLR along z), and differs slightly (~1e-4 relative) only
    for z-dependent geometry. log det D (with_det=True) gives the pole-free winding function F = 𝒟 det D.

    geometry : periodic Geometry on z ∈ [−π, π] providing, besides vMx, b, weight, gradpar, also vMy, gyy (|∇y|²),
        gxy (∇x·∇y) and b_k(z, kx, ky), all periodic: Miller with shat = 0, or AnalyticGeometry(vMy=…, gyy=…, gxy=…).
        For ŝ ≠ 0 the binormal coefficients are secular in z (twist-and-shift), which this Fourier model excludes.
    ky_P : primary binormal wavenumber (ρ units); M, Mphi, grid, nz as in StreamingDispersion.
    N : sidebands. Convergence (CBC ŝ = 0, q = 20, Gaussian primary Φ_P = 0.723, k_x = 0.4, k_y^P → 0, against the
        y-parametric StreamingDispersion): N = 3, 6, 9 give ω_r errors −6e-3, −2e-5, +2e-5 and γ errors +2e-3, +4e-4,
        0 (relative); the reference study's N = 3 is too coarse for 0.1 % accuracy. Cost grows ≈ N^2.
    """

    def __init__(self, geometry, primary, ky_P, *, tau=1.0, eta_par=0.0, eta_prp=0.0, delta_par=0.0,
                 delta_prp=0.0, N=6, M=None, Mphi=8, grid=None, nz=None, streaming=True, binormal_drift=True,
                 nthreads=1, chunk_bytes=8e7, Mmax=64, _order="physical"):
        if not getattr(geometry, "periodic", False) or abs(geometry.zmin + np.pi) > 1e-9 \
                or abs(geometry.zmax - np.pi) > 1e-9:
            raise ValueError("the binormal solver needs a periodic tube z ∈ [−π, π]")
        if hasattr(geometry, "binormal_periodic") and not geometry.binormal_periodic():
            raise ValueError("the k_y ≠ 0 coefficients (gds2, gds21, gbdrift) of this geometry are not periodic "
                             "over the tube (ŝ ≠ 0?): use a shear-free tube, e.g. miller(..., shat=0)")
        ex = getattr(geometry, "extra", {})
        if "cvdrift" in ex and "gbdrift" in ex and \
                np.max(np.abs(np.asarray(ex["cvdrift"]) - np.asarray(ex["gbdrift"]))) > 1e-10:
            raise ValueError("cvdrift ≠ gbdrift (β' ≠ 0) is not supported by the binormal solver")
        self.geometry, self.primary, self.ky = geometry, primary, float(ky_P)
        self.tau, self.eta_par, self.eta_prp = float(tau), float(eta_par), float(eta_prp)
        self.delta_par, self.delta_prp = float(delta_par), float(delta_prp)
        self.N, self.M, self.Mphi = int(N), M, int(Mphi)
        self.grid = grid if grid is not None else VelocityGrid()
        self.nz, self.streaming, self.binormal_drift = nz, bool(streaming), bool(binormal_drift)
        self.nthreads, self.chunk_bytes, self.Mmax = int(nthreads), float(chunk_bytes), int(Mmax)
        if _order not in ("physical", "reference"):
            raise ValueError(_order)
        self._order = _order
        self._cache = {}
        self.last = {}

    def _setup(self, kx):
        key = float(kx)
        if key in self._cache:
            return self._cache[key]
        g, p, vg = self.geometry, self.primary, self.grid
        z0 = _zgrid(512)
        wd = kx * np.max(np.abs(g.vMx(z0))) + self.N * self.ky * np.max(np.abs(g.vMy(z0)))   # largest drift frequency
        gp = np.mean(np.abs(g.gradpar(z0))) if self.streaming else 0.0
        Mt = self.Mmax if gp == 0 else int(min(self.Mmax, np.ceil(6 + 2.2 * wd / gp)))
        M = self.M if self.M is not None else max(Mt, self.Mphi + 8)
        Mphi = min(self.Mphi, M)
        nm, nn = 2 * M + 1, 2 * self.N + 1
        ms, ns = np.arange(-M, M + 1), np.arange(-self.N, self.N + 1)
        nz = self.nz or max(256, 8 * nm)
        z = _zgrid(nz)
        one = np.ones(nz)
        vMx, vMy = np.asarray(g.vMx(z), float) * one, np.asarray(g.vMy(z), float) * one
        if not self.binormal_drift:
            vMy = 0 * vMy
        gpz = np.asarray(g.gradpar(z), float) * one if self.streaming else np.zeros(nz)
        TP = np.asarray(p(z), float) * one
        wz = np.asarray(g.weight(z), float) * one
        bP = np.asarray(g.b_k(z, 0.0, self.ky), float) * one
        s = dict(kx=kx, M=M, Mphi=Mphi, nm=nm, nn=nn, ms=ms, ns=ns)
        s["S0"] = _toeplitz(gpz, nm) * ms[None, :]
        s["Tx"] = kx * _toeplitz(vMx, nm)
        s["Ty"] = self.ky * _toeplitz(vMy, nm)
        s["K0"] = kx * p.Phi_P * _toeplitz(TP, nm)                                       # drive (not gyroaveraged)
        s["KP"] = kx * p.Phi_P * _toeplitz(TP[None, :] * jv(0, np.sqrt(2 * bP)[None, :] * np.sqrt(vg.u)[:, None]), nm)
        J0n, G0n = [], []
        for n in ns:
            bn = np.asarray(g.b_k(z, kx, n * self.ky), float) * one
            J0n.append(_toeplitz(jv(0, np.sqrt(2 * bn)[None, :] * np.sqrt(vg.u)[:, None]), nm))   # (n_u, nm, nm)
            G0n.append(_toeplitz(ive(0, bn), nm))
        s["J0n"], s["G0n"] = np.array(J0n), np.array(G0n)                               # (nn, n_u, nm, nm)
        win_m = np.arange(M - Mphi, M + Mphi + 1)
        s["win"] = np.concatenate([jn * nm + win_m for jn in range(nn)])
        s["nw"] = s["win"].size
        s["i00"] = int(np.where(s["win"] == self.N * nm + M)[0][0])
        cvec = np.zeros(s["nw"], dtype=complex)
        nwm = win_m.size
        cvec[self.N * nwm:(self.N + 1) * nwm] = 2 * np.pi * _harmonics(wz, Mphi)[::-1]   # n = 0 block
        s["c"] = cvec
        self._cache[key] = s
        return s

    def _blocks(self, s, omega, sl):
        """A and Δ for the velocity nodes sl, full (nv, nt, nt) arrays."""
        vg = self.grid
        nm, nn, ns = s["nm"], s["nn"], s["ns"]
        nt = nm * nn
        nv = sl.stop - sl.start
        E = vg.VP[sl] ** 2 + 0.5 * vg.UU[sl]
        iu = vg.iu[sl]
        A = np.zeros((nv, nt, nt), dtype=complex)
        Dl = np.zeros((nv, nt, nt), dtype=complex)
        eye = np.eye(nm)
        KP = s["KP"][iu]                                                            # (nv, nm, nm)
        vpar2, uu = vg.VP[sl] ** 2 - 0.5, vg.UU[sl] - 1.0
        tp = self.tau * (1 + self.eta_par * vpar2 * np.exp(1j * self.delta_par) + self.eta_prp * uu * np.exp(1j * self.delta_prp))
        tm = self.tau * (1 + self.eta_par * vpar2 * np.exp(-1j * self.delta_par) + self.eta_prp * uu * np.exp(-1j * self.delta_prp))
        for jn, n in enumerate(ns):
            r = slice(jn * nm, (jn + 1) * nm)
            A[:, r, r] = omega * eye - E[:, None, None] * (s["Tx"] + n * s["Ty"])[None] - vg.VP[sl, None, None] * s["S0"][None]
            Dl[:, r, r] = omega * eye
            if jn > 0:                       # (sin y f)_n = (f_{n−1} − f_{n+1})/2i ; A ∋ +K_P sin y
                c = slice((jn - 1) * nm, jn * nm)
                A[:, r, c] = KP / 2j
                Dl[:, r, c] = (KP + tp[:, None, None] * s["K0"][None]) / 2j
            if jn < nn - 1:
                c = slice((jn + 1) * nm, (jn + 2) * nm)
                A[:, r, c] = -KP / 2j
                Dl[:, r, c] = -(KP + tm[:, None, None] * s["K0"][None]) / 2j
        return A, Dl

    def _response(self, s, omega, sl):
        """Σ_v W_v [J0 (A⁻¹Δ − I) J0]_{win,win} over the nodes sl."""
        vg = self.grid
        nm, nn = s["nm"], s["nn"]
        nt, win = nm * nn, s["win"]
        iu = vg.iu[sl]
        A, Dl = self._blocks(s, omega, sl)
        J0 = np.zeros((sl.stop - sl.start, nt, nt), dtype=complex)                  # block-diag over n
        for jn in range(nn):
            r = slice(jn * nm, (jn + 1) * nm)
            J0[:, r, r] = s["J0n"][jn][iu]
        J0w = J0[:, :, win]                                                          # (nv, nt, nw)
        if self._order == "physical":
            X = np.linalg.solve(A, Dl @ J0w)                                         # A⁻¹ Δ J0
        else:
            X = Dl @ np.linalg.solve(A, J0w)                                         # Δ A⁻¹ J0 (reference)
        Y = J0[:, win, :] @ (X - J0w)
        return np.einsum("v,vab->ab", vg.W[sl], Y)

    def Dmatrix(self, omega, kx):
        s = self._setup(kx)
        omega = complex(omega)
        vg = self.grid
        nt = s["nm"] * s["nn"]
        nv = max(1, int(self.chunk_bytes / (16.0 * nt * nt * 4)))
        sls = [slice(i, min(i + nv, vg.N)) for i in range(0, vg.N, nv)]
        if self.nthreads > 1:
            with ThreadPoolExecutor(self.nthreads) as ex:
                parts = list(ex.map(lambda sl: self._response(s, omega, sl), sls))
        else:
            parts = [self._response(s, omega, sl) for sl in sls]
        Nsum = np.sum(parts, axis=0)
        G0 = np.zeros((s["nw"], s["nw"]), dtype=complex)
        nwm = 2 * s["Mphi"] + 1
        win_m = np.arange(s["M"] - s["Mphi"], s["M"] + s["Mphi"] + 1)
        for jn in range(s["nn"]):
            r = slice(jn * nwm, (jn + 1) * nwm)
            G0[r, r] = s["G0n"][jn][np.ix_(win_m, win_m)]
        return (1.0 + self.tau) * np.eye(s["nw"]) - G0 - Nsum

    def D(self, omega, kx, with_det=False):
        """𝒟(ω, k_x) = 1 − τ cᵀ D⁻¹ e_{(0,0)}; with_det=True also returns (sign, log|det D|) of D."""
        s = self._setup(kx)
        Dm = self.Dmatrix(omega, kx)
        e = np.zeros(s["nw"])
        e[s["i00"]] = 1.0
        val = 1.0 - self.tau * (s["c"] @ np.linalg.solve(Dm, e))
        self.last = dict(M=s["M"], Mphi=s["Mphi"], N=self.N, nt=s["nm"] * s["nn"])
        if with_det:
            sign, logdet = np.linalg.slogdet(Dm)
            return val, sign, logdet
        return val

    def find_root(self, omega0, kx, **kw):
        from .roots import find_root
        return find_root(lambda w: self.D(w, kx), omega0, **kw)
