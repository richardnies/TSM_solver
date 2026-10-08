# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled kernels of the secondary-mode dispersion relation (Nies & Parra 2026).

Contents
--------
* complex adaptive Gauss–Kronrod (15-point) quadrature on a real interval, nogil;
* the ion response N in the λ-integral form (eq. 34; exact FLR), with the contours that
  continue it analytically to Im(ω − ω_E) < 0;
* the small-b closed form of N through the plasma dispersion function (the old "disp_model 0.1");
* the primary-phase (y) average of (D_ISM − 1)/D_ISM for a batch of field-line points, optionally
  on a deformed y-contour, parallelised with OpenMP.

All frequencies are in the units of the caller (stella: v_th/a). Variable names follow
`kernels_ref.py`: wbar = ω − ω_E, w1, w2, w3 the numerator coefficients, wM = k_x v_Mx.
"""
cimport cython
from cython.parallel cimport prange, parallel
from libc.math cimport sqrt, exp, log, fabs, cos, sin, M_PI
from scipy.special.cython_special cimport wofz, iv as sp_iv, ive as sp_ive
import numpy as np

cdef extern from "complex.h" nogil:
    double complex cexp(double complex)
    double complex csqrt(double complex)
    double complex csin(double complex)
    double complex ccos(double complex)
    double complex casin(double complex)
    double complex conj(double complex)
    double creal(double complex)
    double cimag(double complex)
    double cabs(double complex)
    double carg(double complex)

ctypedef double complex (*cfun)(double, void*) nogil

DEF MAXINT = 256
cdef double EPMACH = 2.220446049250313e-16
cdef double UFLOW = 2.2250738585072014e-308
cdef double SQRTPI = 1.7724538509055159

# ----------------------------------------------------------------------------------------------
# Gauss–Kronrod 15 (QUADPACK qk15 nodes and weights)
# ----------------------------------------------------------------------------------------------
cdef double XGK[8]
cdef double WGK[8]
cdef double WG[4]
XGK[:] = [0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
          0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
          0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
          0.207784955007898467600689403773245, 0.0]
WGK[:] = [0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
          0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
          0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
          0.204432940075298892414161999234649, 0.209482141084727828012999174891714]
WG[:] = [0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
         0.381830050505118944950369775488975, 0.417959183673469387755102040816327]

# Gauss–Laguerre nodes/weights (weight e^{-s}) for rays whose singularities are far from [0, ∞)
cdef double GLX16[16]
cdef double GLW16[16]
cdef double GLX48[48]
cdef double GLW48[48]
_x, _w = np.polynomial.laguerre.laggauss(16)
for _i in range(16):
    GLX16[_i] = _x[_i]
    GLW16[_i] = _w[_i]
_x, _w = np.polynomial.laguerre.laggauss(48)
for _i in range(48):
    GLX48[_i] = _x[_i]
    GLW48[_i] = _w[_i]
# |Im √s_k| thresholds (s_k: singularities in the ray variable) above which GL16 / GL48 reach 1e-12;
# measured on 2964 random cases (scripts/gl_threshold_study.py)
cdef double Q_GL16 = 2.5
cdef double Q_GL48 = 1.5


cdef void gk15(cfun f, void* p, double a, double b, double complex* res, double* err, double* rabs) nogil:
    cdef double centr = 0.5 * (a + b)
    cdef double hl = 0.5 * (b - a)
    cdef double complex fv1[7]
    cdef double complex fv2[7]
    cdef double complex fc = f(centr, p)
    cdef double complex resg = fc * WG[3]
    cdef double complex resk = fc * WGK[7]
    cdef double resabs = WGK[7] * cabs(fc)
    cdef double complex reskh
    cdef double resasc, abserr, absc, r
    cdef int j, jtw, jtwm1
    for j in range(3):
        jtw = 2 * j + 1
        absc = hl * XGK[jtw]
        fv1[jtw] = f(centr - absc, p)
        fv2[jtw] = f(centr + absc, p)
        resg = resg + WG[j] * (fv1[jtw] + fv2[jtw])
        resk = resk + WGK[jtw] * (fv1[jtw] + fv2[jtw])
        resabs = resabs + WGK[jtw] * (cabs(fv1[jtw]) + cabs(fv2[jtw]))
    for j in range(4):
        jtwm1 = 2 * j
        absc = hl * XGK[jtwm1]
        fv1[jtwm1] = f(centr - absc, p)
        fv2[jtwm1] = f(centr + absc, p)
        resk = resk + WGK[jtwm1] * (fv1[jtwm1] + fv2[jtwm1])
        resabs = resabs + WGK[jtwm1] * (cabs(fv1[jtwm1]) + cabs(fv2[jtwm1]))
    reskh = resk * 0.5
    resasc = WGK[7] * cabs(fc - reskh)
    for j in range(7):
        resasc = resasc + WGK[j] * (cabs(fv1[j] - reskh) + cabs(fv2[j] - reskh))
    res[0] = resk * hl
    resabs = resabs * fabs(hl)
    resasc = resasc * fabs(hl)
    abserr = cabs((resk - resg) * hl)
    if resasc != 0.0 and abserr != 0.0:
        r = 200.0 * abserr / resasc
        abserr = resasc * (1.0 if r >= 1.0 else r ** 1.5)
    if resabs > UFLOW / (50.0 * EPMACH):
        if abserr < 50.0 * EPMACH * resabs:
            abserr = 50.0 * EPMACH * resabs
    err[0] = abserr
    rabs[0] = resabs


cdef double complex adapt(cfun f, void* p, double* pts, int npts, double epsabs, double epsrel,
                          int limit, double* abserr, int* ier, int* neval) nogil:
    """Globally adaptive GK15 over [pts[0], pts[npts-1]] with initial breakpoints pts (sorted)."""
    cdef double A[MAXINT]
    cdef double B[MAXINT]
    cdef double E[MAXINT]
    cdef double RA[MAXINT]
    cdef double complex V[MAXINT]
    cdef int n = 0, i, imax
    cdef double complex total, vl, vr
    cdef double errsum, tol, el, er, emax, mid, ral, rar
    if limit > MAXINT:
        limit = MAXINT
    ier[0] = 0
    for i in range(npts - 1):
        if pts[i + 1] > pts[i] and n < MAXINT:
            A[n] = pts[i]
            B[n] = pts[i + 1]
            gk15(f, p, A[n], B[n], &V[n], &E[n], &RA[n])
            neval[0] += 15
            n += 1
    while True:
        total = 0.0
        errsum = 0.0
        emax = -1.0
        imax = 0
        for i in range(n):
            total = total + V[i]
            errsum = errsum + E[i]
            if E[i] > emax:
                emax = E[i]
                imax = i
        tol = epsabs if epsabs > epsrel * cabs(total) else epsrel * cabs(total)
        if errsum <= tol or emax <= 0.0:
            break
        if n >= limit:
            ier[0] = 1
            break
        if emax <= 60.0 * EPMACH * RA[imax]:
            ier[0] = 3              # the largest error is round-off: refining cannot reduce it
            break
        mid = 0.5 * (A[imax] + B[imax])
        if mid <= A[imax] or mid >= B[imax] or (B[imax] - A[imax]) < 1e-13 * (pts[npts - 1] - pts[0]):
            E[imax] = 0.0           # cannot refine further: accept and flag
            ier[0] = 2
            continue
        gk15(f, p, A[imax], mid, &vl, &el, &ral)
        gk15(f, p, mid, B[imax], &vr, &er, &rar)
        neval[0] += 30
        A[n] = mid
        B[n] = B[imax]
        V[n] = vr
        E[n] = er
        RA[n] = rar
        B[imax] = mid
        V[imax] = vl
        E[imax] = el
        RA[imax] = ral
        n += 1
    abserr[0] = errsum
    return total


# ----------------------------------------------------------------------------------------------
# Γ0(z) = e^{-z} I0(z) and Γ0'(z) for complex z (scipy's complex ive is e^{-|Re z|} I0(z): do not use)
# ----------------------------------------------------------------------------------------------
cdef inline void gamma0_both(double complex z, double complex* g0, double complex* g0p) nogil:
    cdef double complex q, t0, t1, s0, s1, ez
    cdef int k
    if cabs(z) <= 3.0:
        q = 0.25 * z * z
        t0 = 1.0
        t1 = 0.5 * z
        s0 = t0
        s1 = t1
        for k in range(1, 40):
            t0 = t0 * q / (k * k)
            t1 = t1 * q / (k * (k + 1.0))
            s0 = s0 + t0
            s1 = s1 + t1
            if cabs(t0) < 1e-17 * cabs(s0) and cabs(t1) < 1e-17 * (cabs(s1) + 1e-300):
                break
    else:
        s0 = sp_iv(0.0, z)
        s1 = sp_iv(1.0, z)
    ez = cexp(-z)
    g0[0] = ez * s0
    g0p[0] = ez * (s1 - s0)


def py_gamma0(double complex z):
    cdef double complex g0, g0p
    gamma0_both(z, &g0, &g0p)
    return g0, g0p


# ----------------------------------------------------------------------------------------------
# λ-form of N
# ----------------------------------------------------------------------------------------------
cdef struct LamP:
    double complex wbar
    double complex w1
    double complex w2
    double complex w3
    double wM
    double b
    double complex lam0
    double complex dirn
    double complex pref      # multiplies the integrand on this leg (e^{iλ0 ω̄} when factored out)
    double complex rot       # e^{-iβ}: rotates the sqrt cut to the direction e^{i(β+π)}
    double complex rothalf   # e^{iβ/2}


cdef double complex lam_f(double x, void* vp) nogil:
    cdef LamP* p = <LamP*> vp
    cdef double complex lam = p.lam0 + p.dirn * x
    cdef double complex il = 1j * lam * p.wM
    cdef double complex d1 = 1.0 / (1.0 + 0.5 * il)
    cdef double complex d2 = 1.0 / (1.0 + il)
    cdef double complex sq = 1.0 / (p.rothalf * csqrt((1.0 + il) * p.rot))
    cdef double complex bh = p.b * d1
    cdef double complex g0, g0p
    gamma0_both(bh, &g0, &g0p)
    return (-1j * p.pref * cexp(1j * (lam - p.lam0) * p.wbar) * sq * d1
            * (g0 * (p.w1 + p.w2 * d2 + p.w3 * d1) + bh * g0p * p.w3 * d1) * p.dirn)


cdef double complex leg(LamP* p, double x0, double x1, double* brk, int nbrk,
                        double tol, int* ier, int* neval) nogil:
    cdef double pts[12]
    cdef int n = 0, i, j
    cdef double tmp, err
    pts[n] = x0
    n += 1
    for i in range(nbrk):
        if brk[i] > x0 and brk[i] < x1 and n < 11:
            pts[n] = brk[i]
            n += 1
    pts[n] = x1
    n += 1
    for i in range(1, n):                     # insertion sort
        tmp = pts[i]
        j = i - 1
        while j >= 0 and pts[j] > tmp:
            pts[j + 1] = pts[j]
            j -= 1
        pts[j + 1] = tmp
    cdef int ie = 0
    cdef double complex r = adapt(lam_f, p, pts, n, tol, tol, 200, &err, &ie, neval)
    if ie > ier[0]:
        ier[0] = ie
    return r


cdef double complex lam_g(double s, void* vp) nogil:
    """Ray integrand divided by e^{-s} (for Gauss–Laguerre); the ray is λ = lam0 + i s/ω̄."""
    cdef LamP* p = <LamP*> vp
    cdef double complex lam = p.lam0 + p.dirn * s
    cdef double complex il = 1j * lam * p.wM
    cdef double complex d1 = 1.0 / (1.0 + 0.5 * il)
    cdef double complex d2 = 1.0 / (1.0 + il)
    cdef double complex sq = 1.0 / (p.rothalf * csqrt((1.0 + il) * p.rot))
    cdef double complex bh = p.b * d1
    cdef double complex g0, g0p
    gamma0_both(bh, &g0, &g0p)
    return (-1j * p.pref * sq * d1 * (g0 * (p.w1 + p.w2 * d2 + p.w3 * d1) + bh * g0p * p.w3 * d1) * p.dirn)


cdef double complex ray(LamP* p, double complex lam0, double tol, int* ier, int* neval, bint principal=False) nogil:
    """∫ along λ = lam0 + i s/ω̄, s ∈ [0, ∞): Gauss–Laguerre when the singularities λ = ik/ωM
    (k = 1, 2) are far from the ray in the sense |Im √s_k| ≥ Q, adaptive GK15 otherwise."""
    cdef double complex sk
    cdef double q = 1e300, qk, smax
    cdef double brk[4]
    cdef int k, nb = 2
    cdef double complex acc = 0.0
    cdef double complex rot_save = p.rot, rh_save = p.rothalf
    cdef double complex out
    if principal:                      # sqrt continuous along this ray: principal branch
        p.rot = 1.0
        p.rothalf = 1.0
    p.lam0 = lam0
    p.dirn = 1j / p.wbar
    p.pref = cexp(1j * lam0 * p.wbar)
    for k in range(1, 3):
        sk = -1j * p.wbar * (1j * k / p.wM - lam0)
        qk = fabs(cimag(csqrt(sk)))
        if qk < q:
            q = qk
    if q >= Q_GL16:
        for k in range(16):
            acc = acc + GLW16[k] * lam_g(GLX16[k], p)
        neval[0] += 16
        out = acc
    elif q >= Q_GL48:
        for k in range(48):
            acc = acc + GLW48[k] * lam_g(GLX48[k], p)
        neval[0] += 48
        out = acc
    else:
        brk[0] = 1.0
        brk[1] = 4.0
        for k in range(1, 3):
            sk = -1j * p.wbar * (1j * k / p.wM - lam0)
            if creal(sk) > 0 and creal(sk) < 40.0:
                brk[nb] = creal(sk)
                nb += 1
        smax = 40.0 + (log(cabs(p.pref)) if cabs(p.pref) > 1.0 else 0.0)
        out = leg(p, 0.0, smax, brk, nb, tol, ier, neval)
    p.rot = rot_save
    p.rothalf = rh_save
    return out


cdef double ray_growth(double complex Om, double b) nogil:
    """max over the integrated ray s ∈ [0, 40] of −s − 2 Re b̃(s), b̃ = b/(1 − s/2Ω): the log of
    the largest integrand magnitude (|Γ0(b̃)| ≲ e^{−2 Re b̃}). Re b̃ is smallest near
    s* = 2 Re Ω + 2|Im Ω|(1 + √(1+ρ²))/ρ, ρ = Re Ω/|Im Ω|."""
    cdef double ORe = creal(Om), OIm = fabs(cimag(Om)), rho, ss, g, gmax = -1e300
    cdef double cand[3]
    cdef int i
    if ORe <= 0:
        return 0.0
    if OIm < 1e-300:
        OIm = 1e-300
    rho = ORe / OIm
    ss = 2 * ORe + 2 * OIm * (1 + sqrt(1 + rho * rho)) / rho
    cand[0] = 0.0
    cand[1] = 40.0
    cand[2] = ss if ss < 40.0 else 40.0
    for i in range(3):
        g = -cand[i] - 2 * creal(b / (1.0 - cand[i] / (2 * Om)))
        if g > gmax:
            gmax = g
    return gmax


cdef double complex N_lambda(double complex wbar, double complex w1, double complex w2, double complex w3,
                             double wM, double b, double tol, int* flag, int* neval) nogil:
    """N for any complex wbar (physical sheet Im wbar > 0; Landau continuation below).

    Contours in λ (singularities at λ = i/ωM: square root; λ = 2i/ωM: essential, from Γ0(b̃)):
      Im ω̄ > 0, non-resonant (Re ω̄·ωM ≤ 0):  ray λ = i s/ω̄ from 0;
      Im ω̄ > 0, resonant: ray from 0 if it is far from the singularities, else the real segment
                [0, a] followed by the ray from a (a = c/|ωM|, which keeps |b̃| ≤ 1);
      Im ω̄ ≤ 0, non-resonant: ray from 0 (no singularity is crossed while continuing);
      Im ω̄ ≤ 0, resonant: detour right of the singularities ([0, a], then up to a + i sgn(ωM) H),
                then the ray, which passes above them.
    flag bits: 1 = quadrature limit reached, 2 = continuation cancellation above e^25.
    """
    cdef LamP p
    cdef double complex Om, g0, g0p, sk
    cdef double sg, c, a, H, ca, ch, q
    cdef double brk[4]
    cdef int ier = 0
    cdef double complex res = 0.0
    if wM == 0.0:
        gamma0_both(b, &g0, &g0p)
        return (g0 * (w1 + w2 + w3) + b * g0p * w3) / wbar
    sg = 1.0 if wM > 0 else -1.0
    p.wbar = wbar
    p.w1 = w1
    p.w2 = w2
    p.w3 = w3
    p.wM = wM
    p.b = b
    p.rot = -1j * sg                               # e^{-iβ}, β = sg π/2: sqrt cut along −i sg
    p.rothalf = (1.0 + 1j * sg) / sqrt(2.0)        # e^{iβ/2}
    Om = wbar / wM
    cdef bint resonant = creal(wbar) * wM > 0
    if cimag(wbar) > 0:
        q = fabs(cimag(csqrt(Om)))
        # straight ray if the singularities are far from it (Gauss–Laguerre) or lie beyond the integrated
        # range s ≤ 40 without making Γ0(b̃) large on it; otherwise the two-leg contour avoids them
        if (not resonant) or q >= Q_GL48 or (creal(Om) > 40.0 and ray_growth(Om, b) < 4.0):
            res = ray(&p, 0.0, tol, &ier, neval)
        else:
            c = 2.0 * b if 2.0 * b > 1.0 else 1.0
            a = c / fabs(wM)
            p.lam0 = 0.0
            p.dirn = 1.0
            p.pref = 1.0
            res = leg(&p, 0.0, a, brk, 0, tol, &ier, neval)
            res = res + ray(&p, a, tol, &ier, neval)
    else:
        if not resonant:
            res = ray(&p, 0.0, tol, &ier, neval)
        elif creal(Om) > 40.0:
            # the singularities lie at |e^{iλω̄}| = e^{−Re Ω} < e^{−40}: the straight ray (with the
            # square root continuous along it) differs from the detour by a negligible loop integral
            res = ray(&p, 0.0, tol, &ier, neval, True)
        else:
            # leg offset a minimises the cancellation e^{a|Im ω̄|} times the growth of Γ0(b̃) ~ e^{4b/c}
            # (a floor keeps the second leg away from the square-root singularity while the
            # cancellation ca·|Im Ω| stays below 5)
            ca = 2.0 * sqrt(b / fabs(cimag(Om)))
            ch = 4.0 * b if 4.0 * b > 0.5 else 0.5
            c = 5.0 / fabs(cimag(Om))
            if c > 0.5:
                c = 0.5
            if ca < c:
                ca = c
            if ca > ch:
                ca = ch
            if ca < 0.05:
                ca = 0.05
            a = ca / fabs(wM)
            H = (2.0 + ch) / fabs(wM)
            if a * fabs(cimag(wbar)) + 4.0 * b / ca > 25.0:
                flag[0] |= 2
            p.lam0 = 0.0
            p.dirn = 1.0
            p.pref = 1.0
            res = leg(&p, 0.0, a, brk, 0, tol, &ier, neval)
            p.lam0 = a
            p.dirn = 1j * sg
            p.pref = cexp(1j * a * wbar)
            brk[0] = 1.0 / fabs(wM); brk[1] = 2.0 / fabs(wM)
            res = res + leg(&p, 0.0, H, brk, 2, tol, &ier, neval)
            res = res + ray(&p, a + 1j * sg * H, tol, &ier, neval)
    if ier == 1:
        flag[0] |= 1
    return res


# ----------------------------------------------------------------------------------------------
# Small-b closed form (port of code_eval_disp_NEW/disp_rel_utils.pyx, disp_model 0.1)
# ----------------------------------------------------------------------------------------------
include "_jasym.pxi"

#: |ζ| above which the J functions use their asymptotic series (the closed forms lose ~4 log10|ζ|
#: digits to cancellation); the old code switched at 100 with a two-term series.
cdef double ZETA_ASYM = 6.0


cdef inline double complex plasma_disp(double complex zeta) nogil:
    """Z(ζ) = i√π w(ζ), entire (Faddeeva function; accurate for all ζ)."""
    return 1j * SQRTPI * wofz(zeta)


cdef inline double complex plasma_disp_asymp(double complex zeta) nogil:
    """Asymptotic Z(ζ): −(1/ζ) Σ (2n−1)!!/(2ζ²)^n, plus 2i√π e^{−ζ²} for Im ζ < 0 (exact relation)."""
    cdef double complex u = 1.0 / (zeta * zeta)
    cdef double complex s = 0.0
    cdef int n
    for n in range(NJASYM, -1, -1):
        s = s * u + SASYM[n]
    s = -s / zeta
    if cimag(zeta) < 0:
        s = s + 2j * SQRTPI * cexp(-zeta * zeta)
    return s


cdef inline double complex arg_disp(double complex wbar, double complex Om) nogil:
    """ζ = √Ω on the sheet continuous with Im ω̄ > 0 (cut of the continuation along Re ω̄ = 0, Im ω̄ < 0).
    On the real ω̄ axis the physical limit Im ω̄ → 0⁺ is taken (the original code jumped sheets there)."""
    cdef double angle
    if cimag(wbar) == 0.0:
        Om = Om * (1.0 + 1e-300j * creal(wbar) / creal(Om)) if creal(Om) != 0.0 else Om
        wbar = wbar + 1e-300j
    angle = carg(Om)
    if angle < 0:
        angle = angle + 2 * M_PI
    cdef double complex z = -sqrt(cabs(Om)) * cexp(0.5j * angle)
    if cimag(wbar) > 0 or creal(Om) < 0:
        z = -z
    return z


cdef void Jfuncs(double complex z, double complex Z, bint asym, double complex* J) nogil:
    """J11, ∂aJ11, ∂bJ11, ∂²abJ11, ∂²b²J11 as functions of ζ (closed forms of the old code).

    Each J is a quadratic α + β s + γ s² in s = ζ Z(ζ). For |ζ| ≥ ZETA_ASYM (or asym) the
    s-series part is replaced by its expansion in u = 1/ζ² (exact cancellation, scripts/
    derive_J_asymptotics.py), and the exponential part ε = 2i√π ζ e^{−ζ²} (Im ζ < 0 only) is added
    exactly: J(s_ser + ε) − J(s_ser) = ε(β + γ(2 s_ser + ε)).
    """
    cdef double complex z2 = z * z
    cdef double complex s, I11, u, acc, sser, eps, un, t
    cdef double tprev
    cdef double complex al[5]
    cdef double complex be[5]
    cdef double complex ga[5]
    cdef int k, m
    be[0] = 0.0
    ga[0] = 1.0
    be[1] = 2 * z2
    ga[1] = 0.5
    be[2] = -4 * z2
    ga[2] = -(1 + 2 * z2)
    be[3] = -6 * z2
    ga[3] = -(1.5 + z2)
    be[4] = 8 * z2 * z2 + 12 * z2
    ga[4] = 4 * z2 * z2 + 4 * z2 + 3
    if asym or cabs(z) >= ZETA_ASYM:
        u = 1.0 / z2
        for k in range(5):
            # forward summation, truncated at the smallest term (the series is asymptotic)
            acc = 0.0
            un = u
            tprev = 1e300
            for m in range(NJASYM):
                t = JASYM[k][m] * un
                if m > 2 and cabs(t) > tprev:
                    break
                acc = acc + t
                if cabs(t) > 0:
                    tprev = cabs(t)
                if m > 2 and cabs(t) < 1e-18 * cabs(acc):
                    break
                un = un * u
            J[k] = acc
        if cimag(z) < 0:
            sser = 0.0
            for m in range(NJASYM, -1, -1):
                sser = sser * u + SASYM[m]
            sser = -sser
            eps = 2j * SQRTPI * z * cexp(-z2)
            for k in range(5):
                J[k] = J[k] + eps * (be[k] + ga[k] * (2 * sser + eps))
        return
    al[0] = -1.0
    al[1] = 2 * z2 + 0.5
    al[2] = 1 - 2 * z2
    al[3] = -0.5 - 5 * z2
    al[4] = 4 * z2 * z2 + 8 * z2 - 2
    s = Z * z
    I11 = s * s
    for k in range(5):
        J[k] = al[k] + be[k] * s + ga[k] * I11


cdef double complex N_Z(double complex omega, double complex oE, double complex oEpar, double complex oEprp,
                        double wM, double b, double tau, double ap, double aq, bint asym) nogil:
    """O(b) closed form of N (the quantity K of the old code): D_ISM = (τ + 1 − Γ0(b) − K)/τ."""
    cdef double complex wbar = omega - oE
    cdef double complex Om, z, Z
    cdef double complex J[5]
    cdef int k
    if wM == 0.0:                      # no drift: all J vanish (|ζ| → ∞)
        for k in range(5):
            J[k] = 0.0
    else:
        Om = wbar / wM
        z = arg_disp(wbar, Om)
        Z = plasma_disp(z)
        Jfuncs(z, Z, asym, J)
    cdef double complex c = 1.0 / wbar
    cdef double complex Nd = (1.0 / tau - c * (oE - 0.5 * ap * oEpar - aq * oEprp)) * J[0] \
        + c * (ap * oEpar * J[1] + aq * oEprp * J[2])
    cdef double complex Pd = -tau * ((1.0 / tau - c * (oE - 0.5 * ap * oEpar - aq * oEprp)) * J[2]
                                     + c * (ap * oEpar * J[3] + aq * oEprp * J[4]))
    return -tau * oE * c + b * tau * (oE + aq * oEprp) * c + tau * Nd - b * Pd


# ----------------------------------------------------------------------------------------------
# Python access to the local response
# ----------------------------------------------------------------------------------------------
def N_lambda_py(double complex omega, double complex oE, double complex oEpar, double complex oEprp,
                double wM, double b, double tau=1.0, double eta_par=0.0, double eta_prp=0.0, double tol=1e-12):
    """N in the λ form. Returns (N, flag, number of integrand evaluations)."""
    cdef double complex wbar = omega - oE
    cdef double complex w1 = tau * (-oE + 0.5 * eta_par * oEpar + eta_prp * oEprp)
    cdef double complex w2 = 0.5 * (wM - tau * eta_par * oEpar)
    cdef double complex w3 = 0.5 * wM - tau * eta_prp * oEprp
    cdef int flag = 0, neval = 0
    cdef double complex N = N_lambda(wbar, w1, w2, w3, wM, b, tol, &flag, &neval)
    return N, flag, neval


def N_Z_py(double complex omega, double complex oE, double complex oEpar, double complex oEprp,
           double wM, double b, double tau=1.0, double eta_par=0.0, double eta_prp=0.0, bint asym=False):
    return N_Z(omega, oE, oEpar, oEprp, wM, b, tau, eta_par, eta_prp, asym)


def J_py(double complex zeta, bint asym=False):
    """(J11, ∂aJ11, ∂bJ11, ∂²abJ11, ∂²b²J11) at ζ, and Z(ζ)."""
    cdef double complex J[5]
    cdef double complex Z = plasma_disp(zeta)
    Jfuncs(zeta, Z, asym, J)
    return (J[0], J[1], J[2], J[3], J[4]), Z


def plasma_disp_py(double complex zeta, bint asym=False):
    if asym:
        return plasma_disp_asymp(zeta)
    return plasma_disp(zeta)


def arg_disp_py(double complex wbar, double complex Om):
    return arg_disp(wbar, Om)


# ----------------------------------------------------------------------------------------------
# y average
# ----------------------------------------------------------------------------------------------
cdef struct YP:
    double complex omega
    double A            # k_x Φ_P T_P(z): ω_E = −A sin y
    double wM
    double b
    double g0b          # Γ0(b)
    double tau
    double ap
    double aq
    double dpar
    double dprp
    double eta          # contour deformation: y = t + i sgnA η cos t
    double sgnA
    int form            # 0: λ form, 1: Z form
    int asym
    double tolN
    int flag
    int neval
    int npole           # near-real zeros of D_ISM(y) whose poles are subtracted analytically
    double complex yp[2]
    double complex Rp[2]


cdef double complex D_ism_yp(YP* p, double complex y) nogil:
    cdef double complex oE = -p.A * csin(y)
    cdef double complex oEpar = -p.A * csin(y + p.dpar)
    cdef double complex oEprp = -p.A * csin(y + p.dprp)
    cdef double complex N, wbar, w1, w2, w3
    if p.form == 1:
        N = N_Z(p.omega, oE, oEpar, oEprp, p.wM, p.b, p.tau, p.ap, p.aq, p.asym)
    else:
        wbar = p.omega - oE
        w1 = p.tau * (-oE + 0.5 * p.ap * oEpar + p.aq * oEprp)
        w2 = 0.5 * (p.wM - p.tau * p.ap * oEpar)
        w3 = 0.5 * p.wM - p.tau * p.aq * oEprp
        N = N_lambda(wbar, w1, w2, w3, p.wM, p.b, p.tolN, &p.flag, &p.neval)
    return (p.tau + 1.0 - p.g0b - N) / p.tau


cdef double complex y_f(double t, void* vp) nogil:
    cdef YP* p = <YP*> vp
    cdef double complex y = t + 1j * p.sgnA * p.eta * cos(t)
    cdef double complex dy = 1.0 - 1j * p.sgnA * p.eta * sin(t)
    cdef double complex D = D_ism_yp(p, y)
    cdef double complex f = (D - 1.0) / D
    cdef double complex h
    cdef int k
    for k in range(p.npole):            # minus the periodic pole R ½cot((y − y_p)/2)
        h = 0.5 * (y - p.yp[k])
        f = f - p.Rp[k] * 0.5 * ccos(h) / csin(h)
    return f * dy


cdef double complex dD_dy(YP* p, double complex y) nogil:
    """dD_ISM/dy by the fourth-order central difference (D_ISM is analytic in y)."""
    cdef double h = 1e-3
    return (-D_ism_yp(p, y + 2 * h) + 8 * D_ism_yp(p, y + h) - 8 * D_ism_yp(p, y - h)
            + D_ism_yp(p, y - 2 * h)) / (12 * h)


cdef void find_poles(YP* p, double dmax) nogil:
    """Locate up to two zeros y_p of D_ISM(y) with |Im y_p| < 0.5 (Newton from the smallest sampled
    |D_ISM| on a 24-point grid) and store the residues R = −1/D'(y_p) of (D − 1)/D."""
    cdef double complex Dk[24]
    cdef double ak[24]
    cdef double complex y, D0, d1, step
    cdef int k, j, it, m, kbest
    cdef double best
    cdef bint ok
    p.npole = 0
    for k in range(24):
        Dk[k] = D_ism_yp(p, -M_PI + 2 * M_PI * k / 24.0)
        ak[k] = cabs(Dk[k])
    for m in range(2):
        best = dmax
        kbest = -1
        for k in range(24):            # smallest local minimum not yet used
            if ak[k] < best and ak[k] <= ak[(k + 1) % 24] and ak[k] <= ak[(k + 23) % 24]:
                best = ak[k]
                kbest = k
        if kbest < 0:
            return
        ak[kbest] = 1e300
        y = -M_PI + 2 * M_PI * kbest / 24.0
        ok = False
        for it in range(20):
            D0 = D_ism_yp(p, y)
            d1 = dD_dy(p, y)
            if d1 == 0:
                break
            step = D0 / d1
            y = y - step
            if cabs(y) > 10.0:
                break
            if cabs(step) < 1e-13:
                ok = True
                break
        if not ok or fabs(cimag(y)) > 0.5:
            continue
        while creal(y) > M_PI:
            y = y - 2 * M_PI
        while creal(y) <= -M_PI:
            y = y + 2 * M_PI
        for j in range(p.npole):
            if cabs(y - p.yp[j]) < 1e-6:
                ok = False
        if not ok or cimag(y) == 0.0:
            continue
        p.yp[p.npole] = y
        p.Rp[p.npole] = -1.0 / dD_dy(p, y)
        p.npole += 1


cdef double deformation(double complex omega, double A, double eta_max, int* pinch) nogil:
    """Smallest safe η keeping the branch points y* (ω̄ = 0) on their Im ω > 0 sides."""
    if cimag(omega) > 0 or A == 0.0:
        return 0.0
    cdef double complex S = -omega / A
    if fabs(creal(S)) >= 1.0:
        return 0.0
    cdef double complex y1 = casin(S)
    cdef double sgnA = 1.0 if A > 0 else -1.0
    cdef double c = cos(creal(y1))
    cdef double need = sgnA * cimag(y1)
    if need <= 0.0:
        return 0.0
    if c < 1e-3:
        c = 1e-3
    need = 2.0 * need / c
    if need > eta_max:
        pinch[0] = 1
        return eta_max
    return need


cdef extern from "math.h" nogil:
    double _asin "asin"(double)


cdef double complex y_avg_one(double complex omega, double A, double wM, double b, double tau,
                              double eta_par, double eta_prp, double delta_par, double delta_prp,
                              int form, int contour, double eta_fixed, double eta_max,
                              double epsabs, double epsrel, double tolN, bint asym, int limit,
                              bint subtract_poles, int* flag_out, long* nev_out) nogil:
    cdef YP p
    cdef double pts[8]
    cdef int npts, ier = 0, pinch = 0, j, k
    cdef double err, s, ycr, tmp
    cdef double complex res
    p.omega = omega
    p.A = A
    p.wM = wM
    p.b = b
    p.g0b = sp_ive(0.0, b)
    p.tau = tau
    p.ap = eta_par
    p.aq = eta_prp
    p.dpar = delta_par
    p.dprp = delta_prp
    p.sgnA = 1.0 if A >= 0 else -1.0
    p.form = form
    p.asym = asym
    p.tolN = tolN
    p.flag = 0
    p.neval = 0
    p.eta = 0.0
    if contour == 2:
        if eta_fixed > 0:
            p.eta = eta_fixed
        else:
            p.eta = deformation(omega, A, eta_max, &pinch)
    pts[0] = -M_PI; pts[1] = -0.5 * M_PI; pts[2] = 0.0; pts[3] = 0.5 * M_PI; pts[4] = M_PI
    npts = 5
    if contour == 1 and A != 0.0:
        s = -creal(omega) / A
        if s > -1.0 and s < 1.0:
            ycr = _asin(s)
            pts[npts] = ycr
            npts = npts + 1
            tmp = M_PI - ycr
            if tmp > M_PI:
                tmp = tmp - 2 * M_PI
            pts[npts] = tmp
            npts = npts + 1
        for j in range(1, npts):
            tmp = pts[j]
            k = j - 1
            while k >= 0 and pts[k] > tmp:
                pts[k + 1] = pts[k]
                k = k - 1
            pts[k + 1] = tmp
    p.npole = 0
    if p.eta == 0.0 and subtract_poles:
        find_poles(&p, 0.3)
    res = adapt(y_f, &p, pts, npts, epsabs, epsrel, limit, &err, &ier, &p.neval)
    for k in range(p.npole):            # ∫ over a period of ½cot((y − y_p)/2) = iπ sgn(Im y_p)
        res = res + p.Rp[k] * 1j * M_PI * (1.0 if cimag(p.yp[k]) > 0 else -1.0)
    res = res / (2 * M_PI)
    flag_out[0] = p.flag | (4 if ier == 1 else 0) | (8 if pinch else 0) | (64 if p.npole > 0 else 0)
    nev_out[0] = p.neval
    return res


def y_average(double complex omega, double[::1] A, double[::1] wM, double[::1] b,
              double tau=1.0, double eta_par=0.0, double eta_prp=0.0, double delta_par=0.0, double delta_prp=0.0,
              int form=0, int contour=0, double eta_fixed=0.0, double eta_max=2.0,
              double epsabs=1e-10, double epsrel=1e-10, double tolN=1e-12, bint asym=False,
              int limit=200, int nthreads=1, bint subtract_poles=True):
    """(1/2π) ∫ dy (D_ISM − 1)/D_ISM at each field-line point i (ω_E = −A_i sin y).

    contour: 0 real axis; 1 real axis with breakpoints where Re ω̄ = 0 (old-code behaviour);
             2 deformed y = t + i sgn(A) η cos t (η automatic for Im ω ≤ 0, or eta_fixed if > 0).
    Returns (values, flags, nevals); flag bits: 1 λ-quadrature limit, 2 continuation cancellation
    above e^25, 4 y-quadrature limit, 8 branch points pinch the y contour, 64 near-real zeros of D_ISM
    were found and their poles subtracted analytically (real contour only; subtract_poles).
    """
    cdef Py_ssize_t n = A.shape[0], i
    out = np.empty(n, dtype=np.complex128)
    flags = np.zeros(n, dtype=np.int32)
    nev = np.zeros(n, dtype=np.int64)
    cdef double complex[::1] out_v = out
    cdef int[::1] flags_v = flags
    cdef long[::1] nev_v = nev
    if nthreads <= 1:
        for i in range(n):
            out_v[i] = y_avg_one(omega, A[i], wM[i], b[i], tau, eta_par, eta_prp, delta_par, delta_prp,
                                 form, contour, eta_fixed, eta_max, epsabs, epsrel, tolN, asym, limit,
                                 subtract_poles, &flags_v[i], &nev_v[i])
    else:
        with nogil:
            for i in prange(n, schedule='dynamic', num_threads=nthreads):
                out_v[i] = y_avg_one(omega, A[i], wM[i], b[i], tau, eta_par, eta_prp, delta_par, delta_prp,
                                     form, contour, eta_fixed, eta_max, epsabs, epsrel, tolN, asym, limit,
                                     subtract_poles, &flags_v[i], &nev_v[i])
    return out, flags, nev


def D_ism_py(double complex omega, double complex y, double A, double wM, double b, double tau=1.0,
             double eta_par=0.0, double eta_prp=0.0, double delta_par=0.0, double delta_prp=0.0,
             int form=0, double tolN=1e-12, bint asym=False):
    """D_ISM at one (field-line point, primary phase); ω_E = −A sin y."""
    cdef YP p
    p.omega = omega
    p.A = A
    p.wM = wM
    p.b = b
    p.g0b = sp_ive(0.0, b)
    p.tau = tau
    p.ap = eta_par
    p.aq = eta_prp
    p.dpar = delta_par
    p.dprp = delta_prp
    p.form = form
    p.asym = asym
    p.tolN = tolN
    p.flag = 0
    p.neval = 0
    return D_ism_yp(&p, y)
