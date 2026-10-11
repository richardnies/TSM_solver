"""
tsm_ky.py
=========

Secondary-mode dispersion relation with *both* field-line (theta) and
binormal (y) Fourier truncation, retaining ion parallel streaming exactly.

Relative to tsm_core.py, treating y spectrally rather than parametrically
admits three effects that the k_y^P -> 0 theory of Nies & Parra [NP25]
excludes:

  * the binormal magnetic drift  k_y vtilde_My  on each sideband,
  * FLR through  J_0, Gamma_0  evaluated at  k_perp^2 = k_x^2 + (n k_y^P)^2,
  * the gyroaverage  J_0(k_y^P v_perp)  of the primary potential.

Representation
--------------
    g^S = sum_{n,m} g_{nm} exp(i n k_y^P y + i m theta),   |n|<=N, |m|<=M

The primary drive varies as sin(k_y^P y), so it couples n -> n +- 1; the
radial drift (sin theta) and binormal drift (cos theta) couple m -> m +- 1.
The matrix is therefore block-tridiagonal in n with tridiagonal blocks in m.

Crucially, y is now a coordinate rather than a parameter, so the
flux-surface average is simply the (n,m) = (0,0) component and the
dispersion relation collapses to

    Disp(omega) = 1 - tau * [ D(omega)^{-1} ]_{(00),(00)} = 0,

with  D = diag(1 - Gamma_0^{(n)} + tau) - N  and  N the velocity-integrated
response.  No y-quadrature appears at all.

Source identity
---------------
With  A = omega I - P - K_E  and  B = (P - K_g) diag(J0S), where P collects
streaming and both drift components, K_E the primary ExB advection and K_g
the primary drive, eliminating P gives

    A^{-1} B = [ (omega I - K_E - K_g) A^{-1} - I ] diag(J0S),

so only A need be inverted.  This is the operator-valued generalisation of
the scalar identity used in tsm_core.

Normalisation is identical to tsm_core (frequencies in v_Ti/R, k = k_x rho_i,
ky0 = k_y^P rho_i).  As in NP25 the relation between the primary amplitude
A^P and the primary distribution is taken in its long-binormal-wavelength
form, so the drive normalisation acquires O((k_y^P rho_i)^2) corrections that
are not included here.  Magnetic shear is zero and trapped particles are
absent, as in tsm_core.
"""

import numpy as np
from scipy.special import ive, jv, roots_legendre, roots_laguerre


class VGrid:
    """Velocity-space quadrature on a deformable (Landau) contour.

    Two deformations are available, needed to continue the dispersion
    function to Im(omega) < 0:

    * Parallel velocity:  v_par = t - i*cdepth*tanh(t/width).  The transit
      poles sit at v_par = q*omega/m; for m > 0 they lie at positive
      Re v_par and drop below the real axis as Im(omega) goes negative, while
      the m < 0 poles lie at negative Re v_par and rise above it.  The odd
      tanh shape passes beneath the former and above the latter.

    * Perpendicular energy:  u = s*exp(-i*alpha), a rotated ray.  The drift
      resonances lie at U = (omega - m v_par/q)/(lambda_j k), i.e. at
      u = 2[(omega - m v_par/q)/(lambda_j k) - v_par^2].  Those with
      lambda_j > 0 sit above the positive-u axis when Im(omega) > 0 and
      descend through it as omega is continued downwards, so the ray must be
      rotated below them.  Rotation (rather than a shift) keeps the endpoint
      at u = 0.

    The Maxwellian weights are carried explicitly rather than folded into the
    quadrature weights, so both contours can be moved at fixed nodes.
    """

    def __init__(self, n_vpar=96, n_u=12, vpar_max=6.0,
                 cdepth=0.0, width=0.7, alpha=0.0):
        t, wt = roots_legendre(n_vpar)
        t = vpar_max * t
        wt = wt * vpar_max
        if cdepth == 0.0:
            vpar = t.astype(complex)
            jac = np.ones_like(vpar)
        else:
            vpar = t - 1j * cdepth * np.tanh(t / width)
            jac = 1.0 - 1j * cdepth / width / np.cosh(t / width) ** 2
        wv = wt * jac * np.exp(-vpar ** 2) / np.sqrt(np.pi)

        s, ws = roots_laguerre(n_u)
        rot = np.exp(-1j * alpha)
        u = s * rot
        # Gauss-Laguerre carries weight e^{-s}; correct to e^{-u} du
        wu = ws * rot * np.exp(s - u)

        VP, UU = np.meshgrid(vpar, u, indexing='ij')
        self.VP = VP.reshape(-1)
        self.UU = UU.reshape(-1)
        self.W = (wv[:, None] * wu[None, :]).reshape(-1)
        self.N = self.VP.size
        self.params = dict(cdepth=cdepth, width=width, alpha=alpha,
                           n_vpar=n_vpar, n_u=n_u)


class ParamsKy:
    def __init__(self, k=0.3, ky0=0.05, q=6.0, AP=2.0, eta_par=3.0,
                 eta_perp=3.0, tau=1.0, M=8, N=3, Mphi=4,
                 binormal_drift=True, profile='flat', Jth=8):
        self.k, self.ky0, self.q, self.AP = k, ky0, q, AP
        self.eta_par, self.eta_perp, self.tau = eta_par, eta_perp, tau
        self.M, self.N, self.Mphi = M, N, Mphi
        self.binormal_drift = binormal_drift
        self.profile = profile      # 'flat' or 'gaussian'  (NP25 use the
        self.Jth = Jth              # latter: g^P ~ exp(-(theta/pi)^2))

    def theta_harmonics(self, nt=1024):
        """Fourier coefficients p_j of the primary's field-line profile, and
        <P^2>_theta.  P = 1 for 'flat'; P = exp(-(theta/pi)^2) for
        'gaussian', periodically extended over one poloidal turn."""
        th = -np.pi + 2.0 * np.pi * np.arange(nt) / nt
        P = np.ones(nt) if self.profile == 'flat' \
            else np.exp(-(th / np.pi) ** 2)
        js = np.arange(-self.Jth, self.Jth + 1)
        pj = np.array([np.mean(P * np.exp(-1j * j * th)) for j in js])
        return js, pj, float(np.mean(P ** 2))

    @property
    def vE0(self):
        """Peak radial ExB velocity; A^P is the peak amplitude, not the RMS
        (see tsm_core.Params.vE0 and report Sec. 6.3).  The field-line
        profile is normalised to unit peak, not unit RMS."""
        return self.AP


class DispersionKy:

    def __init__(self, p, vg=None, chunk=64):
        self.p = p
        self.vg = vg if vg is not None else VGrid()
        self.chunk = chunk
        self._setup()

    def _setup(self):
        p, vg = self.p, self.vg
        self.nm = 2 * p.M + 1
        self.nn = 2 * p.N + 1
        self.nt = self.nm * self.nn
        self.ms = np.arange(-p.M, p.M + 1)
        self.ns = np.arange(-p.N, p.N + 1)

        # per-sideband perpendicular wavenumber and FLR factors
        kperp = np.sqrt(p.k ** 2 + (self.ns * p.ky0) ** 2)      # (nn,)
        self.b_n = 0.5 * kperp ** 2
        self.Gam_n = ive(0, self.b_n)
        self.J0S = jv(0, kperp[:, None] * np.sqrt(vg.UU)[None, :])
        self.J0P = jv(0, p.ky0 * np.sqrt(vg.UU))

        # field-line profile of the primary as a banded matrix in m
        js, pj, _ = p.theta_harmonics()
        Pm = np.zeros((self.nm, self.nm), dtype=complex)
        for j, c in zip(js, pj):
            for jm in range(self.nm):
                mm = jm - j
                if 0 <= mm < self.nm:
                    Pm[jm, mm] = c
        self.Pm = Pm

        self.U = vg.VP ** 2 + 0.5 * vg.UU                        # drift weight
        self.G = p.tau * (1.0 + p.eta_par * (vg.VP ** 2 - 0.5)
                          + p.eta_perp * (vg.UU - 1.0))

        # index of the (n,m) = (0,0) component and of the phi window
        self.iz = p.N * self.nm + p.M
        win = []
        for jn in range(self.nn):
            for jm in range(p.M - p.Mphi, p.M + p.Mphi + 1):
                win.append(jn * self.nm + jm)
        self.win = np.array(win)
        self.izw = int(np.where(self.win == self.iz)[0][0])

    # -- assemble A for a chunk of velocity nodes ---------------------------
    def _matrices(self, omega, sl):
        p, vg = self.p, self.vg
        nm, nn, nt = self.nm, self.nn, self.nt
        nv = sl.stop - sl.start
        U = self.U[sl]
        vpar = vg.VP[sl]
        J0P = self.J0P[sl]
        G = self.G[sl]

        A = np.zeros((nv, nt, nt), dtype=complex)

        # diagonal: omega - streaming
        for jn in range(nn):
            for jm, m in enumerate(self.ms):
                i = jn * nm + jm
                A[:, i, i] = omega - m * vpar / p.q

        # -- drift terms (couple m -> m +- 1, diagonal in n) --------------
        # radial:   k U sin(theta)      ->  (f_{m-1} - f_{m+1}) / 2i
        # binormal: n ky0 U cos(theta)  ->  (f_{m-1} + f_{m+1}) / 2
        for jn, n in enumerate(self.ns):
            rad = p.k * U
            bin_ = (n * p.ky0 * U) if p.binormal_drift else 0.0 * U
            low = rad / 2j + bin_ / 2.0        # coefficient of f_{m-1}
            up = -rad / 2j + bin_ / 2.0        # coefficient of f_{m+1}
            for jm in range(nm):
                i = jn * nm + jm
                if jm > 0:
                    A[:, i, i - 1] -= low
                if jm < nm - 1:
                    A[:, i, i + 1] -= up

        # -- primary ExB advection K_E (n -> n +- 1, and m -> m - j
        #    through the field-line profile of the primary) -----------------
        kE = p.k * p.vE0 * J0P
        Pm = self.Pm
        for jn in range(nn):
            r = slice(jn * nm, (jn + 1) * nm)
            if jn > 0:
                c = slice((jn - 1) * nm, jn * nm)
                A[:, r, c] -= (kE / 2j)[:, None, None] * Pm[None, :, :]
            if jn < nn - 1:
                c = slice((jn + 1) * nm, (jn + 2) * nm)
                A[:, r, c] -= (-kE / 2j)[:, None, None] * Pm[None, :, :]
        return A, kE, p.k * p.vE0 * G

    def _apply_M2(self, omega, X, kE, kG):
        """(omega I - K_E - K_g) X, with K_E, K_g coupling n -> n +- 1."""
        nm, nn = self.nm, self.nn
        Y = omega * X
        c = (kE + kG) / 2j                     # (nv,)
        Pm = self.Pm
        for jn in range(nn):
            s = slice(jn * nm, (jn + 1) * nm)
            if jn > 0:
                blk = X[:, s.start - nm:s.stop - nm, :]
                Y[:, s, :] -= c[:, None, None] * np.einsum('ij,vjw->viw',
                                                           Pm, blk)
            if jn < nn - 1:
                blk = X[:, s.start + nm:s.stop + nm, :]
                Y[:, s, :] += c[:, None, None] * np.einsum('ij,vjw->viw',
                                                           Pm, blk)
        return Y

    # -- dispersion function -------------------------------------------------
    def evaluate(self, omega, with_det=False):
        p, vg = self.p, self.vg
        nt, nw = self.nt, self.win.size
        Nmat = np.zeros((nw, nw), dtype=complex)

        E = np.zeros((nt, nw))
        E[self.win, np.arange(nw)] = 1.0

        for i0 in range(0, vg.N, self.chunk):
            sl = slice(i0, min(i0 + self.chunk, vg.N))
            nv = sl.stop - sl.start
            A, kE, kG = self._matrices(omega, sl)
            X = np.linalg.solve(A, np.broadcast_to(E, (nv, nt, nw)))
            Y = self._apply_M2(omega, X, kE, kG) - E[None, :, :]
            Yw = Y[:, self.win, :]                       # (nv, nw, nw)

            # J0S on rows (index I) and columns (index J)
            jrow = np.empty((nv, nw), dtype=complex)
            for a, idx in enumerate(self.win):
                jrow[:, a] = self.J0S[idx // self.nm, sl]
            w = vg.W[sl]
            Nmat += np.einsum('v,va,vab,vb->ab', w, jrow, Yw, jrow)

        diag = np.array([1.0 - self.Gam_n[idx // self.nm] + self.p.tau
                         for idx in self.win])
        D = np.diag(diag) - Nmat
        disp = 1.0 - self.p.tau * np.linalg.inv(D)[self.izw, self.izw]
        if with_det:
            sign, logdet = np.linalg.slogdet(D)
            return disp, sign, logdet
        return disp


def muller(f, x0, x1, x2, tol=1e-9, maxiter=45):
    f0, f1, f2 = f(x0), f(x1), f(x2)
    for _ in range(maxiter):
        q_ = (x2 - x1) / (x1 - x0)
        A_ = q_ * f2 - q_ * (1 + q_) * f1 + q_ ** 2 * f0
        B_ = (2 * q_ + 1) * f2 - (1 + q_) ** 2 * f1 + q_ ** 2 * f0
        C_ = (1 + q_) * f2
        disc = np.sqrt(B_ ** 2 - 4 * A_ * C_ + 0j)
        d1, d2 = B_ + disc, B_ - disc
        den = d1 if abs(d1) > abs(d2) else d2
        if den == 0 or not np.isfinite(den):
            return x2, False
        x3 = x2 - (x2 - x1) * 2 * C_ / den
        if not np.isfinite(x3):
            return x2, False
        x0, x1, x2, f0, f1, f2 = x1, x2, x3, f1, f2, f(x3)
        if abs(x2 - x1) < tol * max(1.0, abs(x2)) or abs(f2) < 1e-12:
            return x2, True
    return x2, False


def find_root(d, guess):
    w, ok = muller(d.evaluate, guess * (1 - 3e-3) - 3e-3j,
                   guess * (1 + 3e-3), guess)
    if not ok or not np.isfinite(w) or abs(d.evaluate(w)) > 1e-6:
        return np.nan + 1j * np.nan
    return -np.conj(w) if w.real < 0 else w
