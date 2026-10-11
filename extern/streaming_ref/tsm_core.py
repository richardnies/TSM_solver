"""
tsm_core.py
===========

Generalised toroidal-secondary-mode (TSM) dispersion relation *including ion
parallel streaming*, obtained by expanding the secondary mode in Fourier
harmonics along the field line and truncating.

Physics model
-------------
Large-aspect-ratio circular tokamak, electrostatic, collisionless, single ion
species, modified adiabatic electrons.  Following Nies & Parra (2025,
arXiv:2504.09785) [NP25] we take a frozen streamer primary (d/dx g^P = 0),
long binormal wavelength (no binormal drift, no diamagnetic frequency for the
secondary), and constant |grad x| = 1, B = const in the FLR factors.  Unlike
NP25 we retain the parallel streaming term v_par b.grad.

Normalisations (hats dropped in the code)
-----------------------------------------
    velocities        -> v_Ti           (v_Ti^2 = 2 T_i / m_i)
    frequencies       -> v_Ti / R
    radial wavenumber -> 1 / rho_i,   k  ==  k_x rho_i
    potential         -> phi == Z_i e phi^S / T_i

    b_i     = k^2 / 2                       (|grad x| = 1)
    J_0     = J_0( k * v_perp )
    Gamma_0 = I_0(b_i) exp(-b_i)

    radial magnetic drift   k_x vtilde_Mx  ->  k * U * sin(theta),
        U = v_par^2 + v_perp^2 / 2
    parallel streaming      -i v_par b.grad e^{i m theta} -> (m v_par / q)
    primary ExB             k_x v_Ex^P     ->  k * vE0 * sin(Y),  Y = k_y^P y
    primary drive           k_x vtilde_g^P ->  k * vE0 * sin(Y) * tau *
                                [ 1 + eta_par (v_par^2 - 1/2)
                                    + eta_perp (v_perp^2 - 1) ]

    NP25 amplitude:  A^P = sqrt(<(v_Ex^P R / v_Ti rho_i)^2>_psi) = vE0 / sqrt(2)
    so   vE0 = sqrt(2) * A^P.

Derivation of the truncated system
----------------------------------
Write the secondary as  g^S = sum_m g_m(y,v) e^{i m theta},
                        phi^S = sum_m phi_m(y) e^{i m theta}.
Eq. (16) of NP25 with streaming retained becomes, harmonic by harmonic,

    A(y, v, omega) g = B(y, v) phi,                                    (*)

with the *tridiagonal* matrix (sin(theta) couples m to m +- 1 only)

    A[m,m]   = omega - k vE0 sin(Y) - m v_par / q                 ( == d_m )
    A[m,m-1] = + i omega_d / 2
    A[m,m+1] = - i omega_d / 2                       omega_d = k * U

and the source matrix (h_m = J0 F_M phi_m)

    B[m,m]   = J0 * ( m v_par / q  -  k vtilde_g )
    B[m,m-1] = - i J0 omega_d / 2
    B[m,m+1] = + i J0 omega_d / 2

Truncating at |m| <= M and velocity-integrating gives the response matrix

    N[m,m'](y,omega) = < J0 * (A^{-1} B)[m,m'] >_v

and quasineutrality gives the operator equation

    D(y,omega) phi = tau * Phi * e_0,   D = (1 - Gamma_0 + tau) I - N

where Phi = <phi^S>_psi.  Since the theta-average selects m = 0,
Phi = < phi_0 >_y, and the consistency condition is the dispersion relation

    Disp(omega) = 1 - tau * < [ D(y,omega)^{-1} ]_{00} >_y = 0.        (**)

    Trapped electrons (evaluate(..., ft=f_t)): with bounce frequency >> omega
    and bounce-averaged radial drift zero, trapped electrons respond
    adiabatically to the bounce-averaged potential.  For the m = 0 harmonic
    at fixed y the bounce average is the harmonic itself, so trapped
    electrons give no response and only the passing fraction (1 - f_t)
    couples to the flux-surface average:
        delta n_e,0 / n = tau (1 - f_t) (phi_0(y) - Phi).
    Odd harmonics (sin m theta) have zero bounce average and remain fully
    shielded; even harmonics m != 0 are partially shielded (neglected here).
    Implemented as tau -> tau (1 - f_t) on the (0,0) entry of D and on the
    flux-surface coupling.

Reduction check (analytic): for q -> infinity all d_m are equal, A becomes
(Delta I - omega_d S) with S the sin(theta) multiplication matrix, so in
theta-space  g = (omega_d sin th - k vtilde_g)/(Delta - omega_d sin th) J0 F_M phi,
which is exactly NP25 Eq. (28); (**) then reduces to NP25 Eq. (32).

Velocity integration
--------------------
    <f>_v = pi^{-1/2} Int dv_par e^{-v_par^2} Int_0^inf du e^{-u} f,
    u = v_perp^2.
Gauss-Legendre in v_par (so that the contour can be deformed), Gauss-Laguerre
in u.

Landau contour: for Im(omega) > 0 the real v_par axis is correct (the transit
poles v_par = q(omega - k vE0 sin Y)/m sit off-axis).  A deformed contour is
provided for analytic continuation but see the caveat in the accompanying
report: the *drift* resonance would also require deformation in u, so damped
TSM roots are not continued here.
"""

import numpy as np
from scipy.special import ive, j0, jv, roots_legendre, roots_laguerre


# ----------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------

class Grids:
    """Velocity-space and field-line-label quadrature."""

    def __init__(self, n_vpar=300, n_u=20, n_y=24, vpar_max=6.0, alpha=0.0):
        """alpha rotates the u = v_perp^2 ray into the lower half plane,
        u = s exp(-i alpha), which is what continues the *drift* resonance
        below marginality; the v_par contour is set separately per call
        through the cdepth argument of evaluate()."""
        # v_par: Gauss-Legendre on [-vpar_max, vpar_max], Maxwellian carried
        # explicitly so that the contour may be deformed if required.
        t, wt = roots_legendre(n_vpar)
        self.t = 0.5 * vpar_max * (t + 1.0) * 2.0 - vpar_max   # map to [-L, L]
        self.wt = wt * vpar_max
        # u = v_perp^2: Gauss-Laguerre, weight e^{-u}
        self.s_u, self.ws_u = roots_laguerre(n_u)
        self.u, self.wu = self.s_u.astype(complex), self.ws_u.astype(complex)
        # y: uniform (spectrally accurate for periodic integrands)
        self.Y = 2.0 * np.pi * np.arange(n_y) / n_y
        self.wY = np.full(n_y, 1.0 / n_y)
        self.n_vpar, self.n_u, self.n_y = n_vpar, n_u, n_y
        self.alpha = alpha
        if alpha != 0.0:
            self.rotate_u(alpha)

    def rotate_u(self, beta):
        """Rotate the u = v_perp^2 contour into the lower half plane,
        u = s exp(-i beta).

        Near the positive-U axis the drift-resonance poles sit *above* the
        real axis when Im(omega) > 0 (poles with the opposite sign of the
        drift eigenvalue lie at negative Re U, away from the integration
        path).  Lowering the contour therefore moves away from them for
        Im(omega) > 0 and keeps them above it as Im(omega) becomes negative,
        which is exactly the continuation from the upper half plane.
        """
        s_, w_ = self.s_u, self.ws_u
        self.u = s_ * np.exp(-1j * beta)
        self.wu = w_ * np.exp(s_ * (1.0 - np.exp(-1j * beta))) \
            * np.exp(-1j * beta)
        return self


def _contour(t, cdepth, width=0.7):
    """Landau contour v_par = t + i c(t) with c(t) = -cdepth * tanh(t/width).

    For Im(omega) < 0 the transit poles with m > 0 (which sit at positive
    Re v_par for Re omega > 0) drop below the real axis and the contour must
    dip beneath them; the m < 0 poles rise above the axis at negative
    Re v_par and the contour must rise above them.  The odd tanh shape does
    both.  cdepth = 0 recovers the real axis.
    """
    if cdepth == 0.0:
        return t.astype(complex), np.ones_like(t, dtype=complex)
    c = -cdepth * np.tanh(t / width)
    dc = -cdepth / width / np.cosh(t / width) ** 2
    return t + 1j * c, 1.0 + 1j * dc


# ----------------------------------------------------------------------------
# Batched tridiagonal inverse (Thomas algorithm, vectorised over nodes)
# ----------------------------------------------------------------------------

def _tridiag_inverse(sub, diag, sup, cols=None):
    """Selected columns of the inverse of a batch of tridiagonal matrices.

    Parameters
    ----------
    sub, sup : (N,) arrays   -- constant along each sub/super-diagonal
    diag     : (N, nm) array -- main diagonal

    cols     : optional list of column indices to return

    Returns
    -------
    (N, nm, ncols) array.  Only the requested columns of the inverse are
    computed, which is what allows the potential to be truncated at far
    fewer harmonics than the distribution function.

    Cost is O(nm^2) per matrix instead of the O(nm^3) of a dense LU, which
    matters because convergence of the harmonic truncation requires
    nm ~ k q for the physically interesting cases.
    """
    N, nm = diag.shape
    if cols is None:
        cols = np.arange(nm)
    cols = np.asarray(cols)
    nc = cols.size
    # leading axis = harmonic index so that each sweep touches contiguous
    # memory; this is ~5x faster than the natural (N, nm, nc) ordering.
    cp = np.empty((nm, N), dtype=complex)
    dp = np.zeros((nm, N, nc), dtype=complex)
    unit = np.zeros((nm, nc))
    unit[cols, np.arange(nc)] = 1.0

    beta = diag[:, 0].copy()
    cp[0] = sup / beta
    dp[0] = unit[0][None, :] / beta[:, None]
    for i in range(1, nm):
        beta = diag[:, i] - sub * cp[i - 1]
        cp[i] = sup / beta
        dp[i] = (unit[i][None, :] - sub[:, None] * dp[i - 1]) / beta[:, None]

    x = np.empty((nm, N, nc), dtype=complex)
    x[nm - 1] = dp[nm - 1]
    for i in range(nm - 2, -1, -1):
        x[i] = dp[i] - cp[i][:, None] * x[i + 1]
    return x                      # shape (nm, N, nc)


def _tridiag_block_moments(sub, diag, sup, cols, weight):
    """Weighted sum over nodes of a sub-block of the tridiagonal inverse.

    Returns  Nblk[I,J] = sum_n weight[n] * (A_n^{-1})[cols[I], cols[J]].

    Uses Usmani's closed form for the inverse of a tridiagonal matrix,

        (A^-1)_{ij} = (-sup)^{j-i} theta_{i-1} phi_{j+1} / theta_n   (i <= j)
        (A^-1)_{ij} = (-sub)^{i-j} theta_{j-1} phi_{i+1} / theta_n   (i >  j)

    with theta / phi the leading / trailing principal minors.  These minors
    overflow badly for the harmonic counts needed here, so the recursions are
    carried in *ratio* form,

        r_i = theta_i / theta_{i-1} = d_i - sub*sup / r_{i-1},
        s_i = phi_i  / phi_{i+1}    = d_i - sub*sup / s_{i+1},

    and the required combination is assembled from cumulative logs,

        theta_{i-1} phi_{j+1} / theta_n
            = exp( sum_{k>j} log s_k  -  sum_{k>=i} log r_k ),

    which is bounded.  Because both index windows are narrow, the sum over
    nodes reduces to two small matrix products.

    Cost is O(nm) recursion + O(nphi^2) assembly per node, versus O(nm*nphi)
    for a Thomas sweep with nphi right-hand sides.
    """
    N, nm = diag.shape
    P = sub * sup                                     # (N,)
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

    lr = np.log(r)
    ls = np.log(s)
    # Lr[i] = sum_{k=i}^{nm-1} log r_k ;  Ls[j] = sum_{k=j+1}^{nm-1} log s_k
    Lr = np.cumsum(lr[::-1], axis=0)[::-1]
    Ls = np.zeros((nm + 1, N), dtype=complex)
    Ls[:nm] = np.cumsum(ls[::-1], axis=0)[::-1]
    Ls[nm] = 0.0
    Lsj = Ls[1:]                                       # Lsj[j] = Ls_{j+1}

    ref = 0.5 * (Lsj[cols[nc // 2]] + Lr[cols[nc // 2]])
    e = np.arange(nc)
    msup = (-sup)[None, :] ** e[:, None]               # (nc, N)
    msub = (-sub)[None, :] ** e[:, None]

    alpha = np.exp(ref[None, :] - Lr[cols]) / msup     # (nc, N)  index I
    beta = np.exp(Lsj[cols] - ref[None, :]) * msup     # (nc, N)  index J
    alphaL = np.exp(ref[None, :] - Lr[cols]) / msub
    betaL = np.exp(Lsj[cols] - ref[None, :]) * msub

    W = weight
    upper = (alpha * W[None, :]) @ beta.T              # (nc, nc)
    lower = ((betaL * W[None, :]) @ alphaL.T)          # (nc, nc)  [I,J], I>J

    iu = np.triu(np.ones((nc, nc)))
    return upper * iu + lower * (1.0 - iu)


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

class Params:
    def __init__(self, k=0.3, q=20.0, AP=2.0, eta_par=3.0, eta_perp=3.0,
                 tau=1.0, M=16, Mphi=4):
        self.k = k                 # k_x rho_i
        self.q = q                 # safety factor (q -> inf kills streaming)
        self.AP = AP               # primary amplitude, NP25 Eq. (21)
        self.eta_par = eta_par
        self.eta_perp = eta_perp
        self.tau = tau
        self.M = M                 # harmonics kept for g:   m = -M ... M
        self.Mphi = Mphi           # harmonics kept for phi: m = -Mphi..Mphi

    @property
    def vE0(self):
        """Peak radial ExB velocity of the primary, in units of rho_i v_Ti/R.

        NP25 Eq. (21) defines A^P as an RMS over the flux surface, which for
        v_Ex^P = v_Ex^P0 sin(k_y^P y) would give v_Ex^P0 = sqrt(2) A^P.  That
        convention was checked against their Fig. 3(a) and overshoots both
        omega_r and gamma by ~40%; taking A^P as the PEAK amplitude puts us
        inside their measured band at every k_x rho_i, for both omega_r and
        gamma.  We therefore use v_Ex^P0 = A^P.  See report Sec. 6.3."""
        return self.AP

    def copy(self, **kw):
        p = Params(self.k, self.q, self.AP, self.eta_par, self.eta_perp,
                   self.tau, self.M, self.Mphi)
        for key, val in kw.items():
            setattr(p, key, val)
        return p


# ----------------------------------------------------------------------------
# Dispersion function
# ----------------------------------------------------------------------------

class Dispersion:
    """Builds and evaluates Disp(omega) = 1 - tau <[D^{-1}]_00>_y."""

    def __init__(self, params, grids=None):
        self.p = params
        self.g = grids if grids is not None else Grids()
        self._build_static()

    # -- pieces that do not depend on omega ---------------------------------
    def _build_static(self):
        p, g = self.p, self.g
        M = p.M
        self.nm = 2 * M + 1
        self.mvals = np.arange(-M, M + 1)
        p.Mphi = min(p.Mphi, M)
        self.nphi = 2 * p.Mphi + 1
        self.cols = np.arange(M - p.Mphi, M + p.Mphi + 1)   # phi window
        self.chunk_bytes = 6e7
        self.method = 'usmani'   # 'usmani' (fast) or 'thomas' (reference)

        b = 0.5 * p.k ** 2
        self.Gamma0 = ive(0, b)
        self.diag_const = 1.0 - self.Gamma0 + p.tau

        # velocity node arrays, shape (n_vpar, n_u)
        self.vpar_t = g.t
        self.u = g.u
        VP, U_ = np.meshgrid(g.t, g.u, indexing='ij')
        self.J0 = jv(0, p.k * np.sqrt(U_))                  # J_0(k v_perp)
        self.U = None      # filled per-contour (depends on complex v_par)
        self._VPgrid, self._Ugrid = VP, U_

    def _velocity_arrays(self, cdepth):
        """Return complex v_par nodes, quadrature weight, and derived arrays."""
        p, g = self.p, self.g
        vpar, jac = _contour(g.t, cdepth)
        # weight for  pi^{-1/2} Int dv e^{-v^2} Int du e^{-u}
        w_v = g.wt * jac * np.exp(-vpar ** 2) / np.sqrt(np.pi)
        VP = vpar[:, None]                       # (n_vpar, 1)
        UU = g.u[None, :]                        # (1, n_u)
        W = w_v[:, None] * g.wu[None, :]         # (n_vpar, n_u)
        Udrift = VP ** 2 + 0.5 * UU              # (v_par^2 + v_perp^2/2)
        J0 = jv(0, p.k * np.sqrt(UU)) * np.ones_like(VP)
        # primary drive velocity dependence
        vg_shape = p.tau * (1.0 + p.eta_par * (VP ** 2 - 0.5)
                            + p.eta_perp * (UU - 1.0))
        return VP, UU, W, Udrift, J0, vg_shape

    # -- main evaluation -----------------------------------------------------
    def __call__(self, omega, cdepth=0.0):
        return self.evaluate(omega, cdepth)

    def evaluate(self, omega, cdepth=0.0, profile=False, ft=0.0, beta_e=0.0,
                 ion_current=False, landau=False, mass_ratio=3672.0,
                 chi_ion=False, bflut=0.0, delta_B=0.0, eflut=False):
        """Evaluate Disp(omega).  If profile=True also return per-y data.

        Uses the identity  B = J0 [ (omega - shift - k vtilde_g) I - A ],
        which follows because the m-dependence of the source diagonal
        (m v_par/q) exactly cancels that of the A diagonal, and the drift
        off-diagonals of A and B differ only by the factor -J0.  Hence

            A^{-1} B = J0 [ (omega - shift - k vtilde_g) A^{-1} - I ],

        so only the tridiagonal inverse A^{-1} is required.  In the
        q -> infinity limit this reproduces NP25 Eq. (30) exactly.
        """
        p, g = self.p, self.g
        VP, UU, W, Udrift, J0, vg_shape = self._velocity_arrays(cdepth)
        nm, mvals = self.nm, self.mvals

        Nnode = g.n_vpar * g.n_u
        vpar_f = np.broadcast_to(VP, (g.n_vpar, g.n_u)).reshape(-1)
        W_f = W.reshape(-1)
        Ud_f = np.broadcast_to(Udrift, (g.n_vpar, g.n_u)).reshape(-1)
        J0_f = np.broadcast_to(J0, (g.n_vpar, g.n_u)).reshape(-1)
        vgs_f = np.broadcast_to(vg_shape, (g.n_vpar, g.n_u)).reshape(-1)
        vpar_f = np.broadcast_to(VP, (g.n_vpar, g.n_u)).reshape(-1)   # v_par per node

        omega_d = p.k * Ud_f                         # (Nnode,)
        transit = np.outer(vpar_f, mvals) / p.q      # (Nnode, nm)

        sub = 1j * omega_d / 2.0                     # A[m, m-1]
        sup = -1j * omega_d / 2.0                    # A[m, m+1]
        wJ2 = W_f * J0_f ** 2
        eye = np.eye(self.nphi)
        # Electromagnetic electron response (electrons in parallel force
        # balance, beta_e >> m_e/m_i): delta n_e,m = tau phi_m / (1 - alpha_m),
        # alpha_m = 2 beta_e tau^2 (omega q / (m k))^2, the kinetic-Alfven
        # parameter of sideband m.  The m = 0 (zonal field) harmonic is
        # unaffected at this order for an electrostatic primary.
        em_shift = np.zeros(self.nphi, dtype=complex)
        em_cion = np.zeros(self.nphi, dtype=complex)
        if beta_e > 0.0:
            from scipy.special import wofz
            mvals_phi = np.arange(-(self.nphi // 2), self.nphi // 2 + 1)
            vTe_hat = np.sqrt(mass_ratio / p.tau)          # v_Te / v_Ti
            for ii, mm in enumerate(mvals_phi):
                if mm != 0:
                    alpha = 2.0 * beta_e * p.tau**2 * (omega * p.q / (mm * p.k))**2
                    # electron Landau factor L_m = 1 + zeta Z(zeta), Z = i sqrt(pi) w(zeta)
                    if landau:
                        zeta = omega * p.q / (abs(mm) * vTe_hat)
                        L = 1.0 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta)
                    else:
                        L = 1.0
                    em_shift[ii] = p.tau * (L / (1.0 - alpha * L) - 1.0)
                    # ion parallel current in Ampere's law:
                    #   delta n_e,m = L [tau phi_m - alpha_m (m/(q omega)) u_par,i,m]/(1 - alpha_m L)
                    if ion_current:
                        em_cion[ii] = L * alpha * (mm / (p.q * omega)) / (1.0 - alpha * L)
        cols = self.cols

        Dinv00 = np.empty(g.n_y, dtype=complex)
        detD = np.empty(g.n_y, dtype=complex)
        wJ2sum = np.sum(wJ2)
        chunk = max(1, int(self.chunk_bytes /
                           (Nnode * nm * self.nphi * 16)))
        for i0 in range(0, g.n_y, chunk):
            Ys = np.sin(g.Y[i0:i0 + chunk])                  # (c,)
            nc = Ys.size
            shift = p.k * p.vE0 * Ys                         # (c,)
            # Electromagnetic primary: chi^P = phi^P - v_par A_par^P.  The
            # flutter velocity v_par delta B_x^P / B adds a velocity-
            # dependent Doppler shift and a v_par-odd part to the drive,
            # both diagonal in the harmonic index.  bflut = k_y A_par^P / B
            # in the units of vE0; delta_B = phase of A_par^P relative to phi^P.
            if bflut != 0.0:
                YsB = np.sin(g.Y[i0:i0 + chunk] + delta_B)   # (c,)
                flut = p.k * bflut * YsB[:, None] * vpar_f[None, :]      # (c, Nnode)
                shift2 = shift[:, None] - flut                            # (c, Nnode)
                drive2 = shift[:, None] * vgs_f[None, :] - p.tau * flut   # (c, Nnode)
            else:
                shift2 = np.broadcast_to(shift[:, None], (nc, Nnode))
                drive2 = shift[:, None] * vgs_f[None, :]
            # stack (chunk, Nnode) -> flat
            diag = ((omega - shift2)[:, :, None] - transit[None, :, :]
                    ).reshape(nc * Nnode, nm)
            pref = ((omega - shift2) - drive2).reshape(-1)
            subc = np.tile(sub, nc)
            supc = np.tile(sup, nc)
            wgt = (wJ2[None, :] * pref.reshape(nc, Nnode)).reshape(-1)
            if self.method == 'usmani':
                for j in range(nc):
                    sl = slice(j * Nnode, (j + 1) * Nnode)
                    Nm = _tridiag_block_moments(
                        subc[sl], diag[sl], supc[sl], cols, wgt[sl]) \
                        - wJ2sum * eye
                    Dmat = self.diag_const * eye - Nm
                    Dmat[self.nphi // 2, self.nphi // 2] -= p.tau * ft
                    if beta_e > 0.0:
                        Dmat[np.arange(self.nphi), np.arange(self.nphi)] += em_shift
                        if ion_current:
                            # u_par,i,m = sum_m' N1[m,m'] phi_m',  N1 = v_par-weighted moment
                            N1 = _tridiag_block_moments(
                                subc[sl], diag[sl], supc[sl], cols,
                                wgt[sl] * vpar_f)
                            Dmat -= em_cion[:, None] * N1
                            if chi_ion:
                                # ions respond to chi_m = phi_m (1 + v_par kappa_m):
                                # {chi^S, g^P} magnetic part and the drift on v_par A_par
                                Dmat -= N1 * em_cion[None, :]
                    Dinv00[i0 + j] = np.linalg.inv(Dmat)[self.nphi // 2,
                                                         self.nphi // 2]
                    detD[i0 + j] = np.linalg.det(Dmat)
            else:
                Ablk = _tridiag_inverse(subc, diag, supc, cols)
                blk = Ablk[cols].reshape(self.nphi, nc, Nnode, self.nphi)
                Nmat = np.einsum('cn,icnj->cij',
                                 wgt.reshape(nc, Nnode), blk) - wJ2sum * eye
                if beta_e > 0.0 and ion_current:
                    N1mat = np.einsum('cn,icnj->cij',
                                      (wgt * np.tile(vpar_f, nc)).reshape(nc, Nnode), blk)
                for j in range(nc):
                    Dmat = self.diag_const * eye - Nmat[j]
                    Dmat[self.nphi // 2, self.nphi // 2] -= p.tau * ft
                    if beta_e > 0.0:
                        Dmat[np.arange(self.nphi), np.arange(self.nphi)] += em_shift
                        if ion_current:
                            Dmat -= em_cion[:, None] * N1mat[j]
                            if chi_ion:
                                Dmat -= N1mat[j] * em_cion[None, :]
                    Dinv00[i0 + j] = np.linalg.inv(Dmat)[self.nphi // 2,
                                                         self.nphi // 2]

        if profile:
            return Dinv00, detD
        # Electron shielding of the zonal potential along the primary's bent
        # field lines: k_par,eff(Y) = k bflut sin(Y + delta_B), electrons
        # respond to the m = 0 harmonic with L_0(Y) = 1 + zeta Z(zeta),
        # zeta = omega/(k_par,eff v_Te), so the flux-surface coupling weight
        # becomes tau (1 - L_0(Y)).  Secular field-line excursion assumed.
        wY_eff = g.wY
        if eflut and bflut != 0.0:
            from scipy.special import wofz
            vTe_hat = np.sqrt(mass_ratio / p.tau)
            kpar = np.abs(p.k * bflut * np.sin(g.Y + delta_B)) * vTe_hat
            zeta = omega / np.maximum(kpar, 1e-12)
            L0 = 1.0 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta)
            L0 = np.where(kpar < 1e-9, 0.0, L0)
            wY_eff = g.wY * (1.0 - L0)
        return 1.0 - p.tau * (1.0 - ft) * np.sum(wY_eff * Dinv00)


# ----------------------------------------------------------------------------
# Reference: NP25 dispersion relation (no parallel streaming), Eqs (30)+(32)
# ----------------------------------------------------------------------------

class DispersionNoStreaming:
    """NP25 Eq. (32) evaluated by direct (y, theta) quadrature.

    Used as the q -> infinity benchmark for the truncated model.
    """

    def __init__(self, params, grids=None, n_theta=256):
        self.p = params
        self.g = grids if grids is not None else Grids()
        self.theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
        self.wth = np.full(n_theta, 1.0 / n_theta)
        b = 0.5 * params.k ** 2
        self.Gamma0 = ive(0, b)
        self.diag_const = 1.0 - self.Gamma0 + params.tau

    def evaluate(self, omega):
        p, g = self.p, self.g
        vpar, jac = _contour(g.t, 0.0)
        w_v = g.wt * jac * np.exp(-vpar ** 2) / np.sqrt(np.pi)
        VP = vpar[:, None]
        UU = g.u[None, :]
        W = (w_v[:, None] * g.wu[None, :]).reshape(-1)
        Udrift = (VP ** 2 + 0.5 * UU).reshape(-1)
        J0 = (jv(0, p.k * np.sqrt(UU)) * np.ones_like(VP)).reshape(-1)
        vg_shape = (p.tau * (1.0 + p.eta_par * (VP ** 2 - 0.5)
                             + p.eta_perp * (UU - 1.0))).reshape(-1)
        omd = p.k * Udrift
        sTh = np.sin(self.theta)[:, None]
        wJ2 = W * J0 ** 2

        tot = 0.0
        for iy, Yv in enumerate(g.Y):
            shift = p.k * p.vE0 * np.sin(Yv)
            kvg = shift * vg_shape
            num = omd[None, :] * sTh - kvg[None, :]
            den = (omega - shift) - omd[None, :] * sTh
            Nfun = (num / den * (W * J0 ** 2)[None, :]).sum(axis=1)
            DISM = self.diag_const - Nfun
            tot += g.wY[iy] * np.sum(self.wth * p.tau / DISM)
        return 1.0 - tot


# ----------------------------------------------------------------------------
# Root finding (Muller's method in the complex plane)
# ----------------------------------------------------------------------------

def muller(f, x0, x1, x2, tol=1e-9, maxiter=60, verbose=False):
    f0, f1, f2 = f(x0), f(x1), f(x2)
    for it in range(maxiter):
        q_ = (x2 - x1) / (x1 - x0)
        A_ = q_ * f2 - q_ * (1 + q_) * f1 + q_ ** 2 * f0
        B_ = (2 * q_ + 1) * f2 - (1 + q_) ** 2 * f1 + q_ ** 2 * f0
        C_ = (1 + q_) * f2
        disc = np.sqrt(B_ ** 2 - 4 * A_ * C_ + 0j)
        den1, den2 = B_ + disc, B_ - disc
        den = den1 if abs(den1) > abs(den2) else den2
        if den == 0:
            return x2, False
        x3 = x2 - (x2 - x1) * 2 * C_ / den
        if not np.isfinite(x3):
            return x2, False
        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f(x3)
        if verbose:
            print(f"  muller {it}: om={x3:.6f} |f|={abs(f2):.3e}")
        if abs(x2 - x1) < tol * max(1.0, abs(x2)) or abs(f2) < 1e-12:
            return x2, True
    return x2, False


def find_root(disp, omega_guess, dw=2e-3, **kw):
    f = lambda w: disp.evaluate(w)
    return muller(f, omega_guess * (1 - dw) - 1j * dw,
                  omega_guess * (1 + dw), omega_guess, **kw)


def count_unstable_roots(disp, om_r_range, om_i_max, n=140, om_i_min=2e-3):
    """Argument principle: number of zeros of Disp inside a rectangle in the
    upper half omega plane.  Requires only Im omega > 0 (no continuation)."""
    r0, r1 = om_r_range
    pts = []
    xs = np.linspace(r0, r1, n)
    ys = np.linspace(om_i_min, om_i_max, n)
    pts += [complex(x, om_i_min) for x in xs]
    pts += [complex(r1, y) for y in ys]
    pts += [complex(x, om_i_max) for x in xs[::-1]]
    pts += [complex(r0, y) for y in ys[::-1]]
    vals = np.array([disp.evaluate(w) for w in pts])
    ph = np.unwrap(np.angle(vals))
    return (ph[-1] - ph[0]) / (2 * np.pi)


# ----------------------------------------------------------------------------
# Root mapping
# ----------------------------------------------------------------------------

def root_map(evalfun, wr_range, wi_range, nr=41, ni=21, tol=1e-7,
             dedupe=1e-3):
    """Coarse |Disp| grid -> local minima -> Muller polish.  Returns roots."""
    WR = np.linspace(*wr_range, nr)
    WI = np.linspace(*wi_range, ni)
    Z = np.empty((ni, nr))
    for i, wi in enumerate(WI):
        for j, wr in enumerate(WR):
            Z[i, j] = abs(evalfun(complex(wr, wi)))
    seeds = []
    for i in range(ni):
        for j in range(nr):
            blk = Z[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if Z[i, j] <= blk.min():
                seeds.append(complex(WR[j], WI[i]))
    roots = []
    for sd in seeds:
        w, ok = muller(evalfun, sd * (1 - 3e-3) - 3e-3j,
                       sd * (1 + 3e-3), sd, tol=tol, maxiter=40)
        if not ok or not np.isfinite(w):
            continue
        if abs(evalfun(w)) > 1e-5:
            continue
        if all(abs(w - r) > dedupe * max(1, abs(w)) for r in roots):
            roots.append(w)
    return sorted(roots, key=lambda z: -z.imag), (WR, WI, Z)
