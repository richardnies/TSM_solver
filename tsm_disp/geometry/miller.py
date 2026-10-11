"""Miller local equilibrium, ported from stella (geometry/geometry_miller.f90, get_local_geo).

Computes bmag, gradpar = b·∇θ, gds22, gbdrift0 = cvdrift0, grho, dψ_N/dρ (k_y = 0 quantities, used by every
solver) and gds2, gds21, gbdrift, cvdrift (k_y ≠ 0: binormal drift and FLR, used by the finite-k_y solver), with
stella's discretisation: R, Z on three radii ρ ± dr with dr = 10⁻³ ρ/R0, centred differences in ρ and θ (periodic
ends), trapezoid θ integrals (definite and indefinite). betaprim enters only dI/dρ, gbdrift and cvdrift; the k_y = 0
quantities do not depend on betaprim, betadbprim, d2qdr2 or d2psidr2.

Units: lengths in a (minor radius), B in B_ref = toroidal field at R = rgeo; θ is the geometric
poloidal angle of the Miller parametrisation, which is the field-line coordinate z.
"""
import numpy as np
from .base import TabulatedGeometry


def miller_arrays(rhoc=0.5, rmaj=2.77778, rgeo=None, shift=0.0, qinp=1.4, shat=0.796, kappa=1.0,
                  kapprim=0.0, tri=0.0, triprim=0.0, nzed_local=256, nperiod=1, betaprim=0.0):
    """Return a dict of stella's Miller arrays on θ_j = j (2 nperiod − 1) π/nz, j = −nz … nz."""
    if rgeo is None:
        rgeo = rmaj
    nz2pi = nzed_local // 2
    nz = nz2pi + nzed_local * (nperiod - 1)
    j = np.arange(-nz, nz + 1)
    theta = j * (2 * nperiod - 1) * np.pi / nz
    dr = 1e-3 * rhoc / rmaj
    radii = rhoc + np.array([-dr, 0.0, dr])

    def Rpos(r, th):
        g = np.cos(th + tri * np.sin(th))
        gp = -np.sin(th + tri * np.sin(th)) * triprim * np.sin(th)
        d = r - rhoc
        return rmaj + shift * d + g * rhoc + (g + rhoc * gp) * d

    def Zpos(r, th):
        d = r - rhoc
        return kappa * np.sin(th) * rhoc + (rhoc * kapprim + kappa) * np.sin(th) * d

    R = np.array([Rpos(r, theta) for r in radii])
    Z = np.array([Zpos(r, theta) for r in radii])
    delthet = np.diff(theta)

    def d_dtheta(f):                       # stella get_dthet: centred, periodic ends
        df = np.empty_like(f)
        df[1:-1] = (f[2:] - f[:-2]) / (delthet[:-1] + delthet[1:])
        df[0] = (f[1] - f[-2]) / (delthet[0] + delthet[-1])
        df[-1] = df[0]
        return df

    def theta_integrate(f):                # stella theta_integrate over the central 2π
        i0, i1 = nz - nz2pi, nz + nz2pi
        return 0.5 * np.sum(delthet[i0:i1] * (f[i0:i1] + f[i0 + 1:i1 + 1]))

    def theta_integrate_indef(f):          # stella theta_integrate_indef: ∫_0^θ, trapezoid
        out = np.zeros_like(f)
        for i in range(nz + 1, 2 * nz + 1):
            out[i] = out[i - 1] + 0.5 * delthet[i - 1] * (f[i - 1] + f[i])
        for i in range(nz - 1, -1, -1):
            out[i] = out[i + 1] - 0.5 * delthet[i] * (f[i + 1] + f[i])
        return out

    dRdrho = 0.5 * (R[2] - R[0]) / dr
    dZdrho = 0.5 * (Z[2] - Z[0]) / dr
    R2 = R[1]
    dRdth = d_dtheta(R2)
    dZdth = d_dtheta(Z[1])
    jacrho = R2 * (dRdrho * dZdth - dRdth * dZdrho)
    bi = rgeo
    dpsipdrho = theta_integrate(jacrho / R2**2) / (2 * np.pi) * bi / qinp
    grho = R2 * np.sqrt(dRdth**2 + dZdth**2) / jacrho
    gpsi = grho * dpsipdrho
    bmag = np.sqrt(bi**2 + gpsi**2) / R2
    dBdth = d_dtheta(bmag)
    gradpar = dpsipdrho / (bmag * jacrho)
    dqdr = shat * qinp / rhoc
    gds22 = (grho * dpsipdrho * dqdr)**2
    cvdrift0 = -2.0 * bi * dqdr / bmag**2 * gradpar * dBdth

    # ---- k_y ≠ 0 quantities (stella get_dIdrho, get_djacdrho, get_dgr2dr, get_dBdrho, get_varthet,
    #      get_dvarthdr, get_graddotgrad, get_gds, and the gbdrift/cvdrift lines of get_local_geo)
    d2Rdrdth, d2Zdrdth = d_dtheta(dRdrho), d_dtheta(dZdrho)
    drz = (dRdrho * dRdth + dZdrho * dZdth) / jacrho
    drzdth = d_dtheta(drz)
    denom = theta_integrate(jacrho * (1.0 + (bi / gpsi)**2) / R2**2)
    num1 = theta_integrate(jacrho * (2.0 * dRdrho / R2 + dqdr / qinp) / R2**2)
    num2 = theta_integrate((-2.0 * (dRdth * d2Rdrdth + dZdth * d2Zdrdth) / jacrho + drzdth
                            + betaprim * jacrho / dpsipdrho**2) / grho**2)
    dIdrho = bi * (num1 + num2) / denom
    djacdrho = (R2 / grho)**2 * (2.0 * (dRdth * d2Rdrdth + dZdth * d2Zdrdth) / jacrho - drzdth
                                 + jacrho * (bi * dIdrho / R2**2 - betaprim) / dpsipdrho**2)
    dgpsi2dr = 2.0 * (gpsi**2 * (dRdrho / R2 - djacdrho / jacrho)
                      + (R2 / jacrho)**2 * (dRdth * d2Rdrdth + d2Zdrdth * dZdth) * dpsipdrho**2)
    dBdrho = (bi * dIdrho + 0.5 * dgpsi2dr) / (bmag * R2**2) - bmag * dRdrho / R2
    varthet = bi * theta_integrate_indef(jacrho / R2**2) / (dpsipdrho * qinp)
    dvarthdr = theta_integrate_indef(bi * jacrho * (dIdrho / bi - dqdr / qinp + djacdrho / jacrho
                                                    - 2.0 * dRdrho / R2) / R2**2) / (dpsipdrho * qinp)
    gradthet2 = (R2 / jacrho)**2 * (dRdrho**2 + dZdrho**2)
    gradrho_gradthet = -(R2 / jacrho)**2 * (dRdrho * dRdth + dZdrho * dZdth)
    sec = varthet * dqdr + qinp * dvarthdr
    loc = bi * jacrho / (dpsipdrho * R2**2)
    gradalph_gradthet = -sec * gradrho_gradthet - loc * gradthet2
    gradrho_gradalph = -sec * grho**2 - loc * gradrho_gradthet
    gradalph2 = 1.0 / R2**2 + (sec * grho)**2 + 2.0 * loc * sec * gradrho_gradthet + loc**2 * gradthet2
    gds2 = gradalph2 * dpsipdrho**2
    gds21 = gradrho_gradalph * dqdr * dpsipdrho**2
    cross = dpsipdrho * (gradrho_gradalph * gradalph_gradthet - gradalph2 * gradrho_gradthet)
    gbdrift = 2.0 * (-dBdrho + cross * dBdth * dpsipdrho / bmag**2) / bmag
    cvdrift = gbdrift + 2.0 * betaprim / bmag**2
    # ŝ-free forms of the quantities the solvers divide by ŝ (valid also at ŝ = 0): |∇x|² = gds22/ŝ²,
    # v_Mx = gbdrift0/(2ŝ), ∇x·∇y = gds21/ŝ   (dq/dr = ŝ q/ρ)
    qr = qinp / rhoc
    gxx = (grho * dpsipdrho * qr)**2
    vmx = -bi * qr / bmag**2 * gradpar * dBdth
    gxy = gradrho_gradalph * qr * dpsipdrho**2
    return dict(theta=theta, R=R2, Z=Z[1], dRdrho=dRdrho, dZdrho=dZdrho, dRdth=dRdth, dZdth=dZdth,
                jacrho=jacrho, grho=grho, bmag=bmag, dBdth=dBdth, gradpar=gradpar, gds22=gds22,
                gbdrift0=cvdrift0.copy(), cvdrift0=cvdrift0, dpsipdrho=dpsipdrho, dqdr=dqdr, bi=bi,
                gds2=gds2, gds21=gds21, gbdrift=gbdrift, cvdrift=cvdrift, varthet=varthet, dvarthdr=dvarthdr,
                dIdrho=dIdrho, dBdrho=dBdrho, cross=cross, gxx=gxx, vmx=vmx, gxy=gxy)


def equal_arc(a, nperiod=1):
    """Re-parametrise Miller arrays by stella's equal-arc coordinate (zed_equal_arc = .true.).

    stella: gradpararc = (θ_max − θ_min)/((2 np − 1) ∫_{−π}^{π} dθ/gradpar), arc(θ) = ∫_0^θ gradpararc/gradpar,
    and the quantities are evaluated at θ(zed) on a uniform zed grid; b·∇zed = gradpararc.
    """
    from scipy.interpolate import CubicSpline
    from .base import _Fourier
    th, gp = a["theta"], a["gradpar"]
    nz = (len(th) - 1) // 2
    dth = np.diff(th)
    nz2pi = nz if nperiod == 1 else None
    i0, i1 = 0, len(th) - 1
    if nperiod != 1:
        raise NotImplementedError("equal-arc re-parametrisation implemented for nperiod = 1")
    inv = 1.0 / gp
    integral = 0.5 * np.sum(dth[i0:i1] * (inv[i0:i1] + inv[i0 + 1:i1 + 1]))
    gpar_arc = (th[-1] - th[0]) / ((2 * nperiod - 1) * integral)
    f = gpar_arc / gp
    arc = np.zeros_like(th)
    for k in range(nz + 1, len(th)):
        arc[k] = arc[k - 1] + 0.5 * dth[k - 1] * (f[k - 1] + f[k])
    for k in range(nz - 1, -1, -1):
        arc[k] = arc[k + 1] - 0.5 * dth[k] * (f[k + 1] + f[k])
    zed = th.copy()                                   # uniform grid on [−π, π]
    th_of_z = CubicSpline(arc, th)(zed)
    out = dict(a)
    for key in ("bmag", "gds22", "gbdrift0", "cvdrift0", "grho", "R", "Z", "gxx"):
        out[key] = _Fourier(th[0], th[-1] - th[0], a[key][:-1])(th_of_z)
    # v_Mx ∝ gradpar(θ): in the equal-arc coordinate b·∇zed = gpar_arc, so v_Mx(zed) = v_Mx(θ) gpar_arc/gradpar(θ)
    out["vmx"] = _Fourier(th[0], th[-1] - th[0], (a["vmx"] / a["gradpar"])[:-1])(th_of_z) * gpar_arc
    for key in ("gds2", "gds21", "gbdrift", "cvdrift", "gxy"):  # secular in θ for ŝ ≠ 0: not periodic
        if key in a:
            out[key] = CubicSpline(th, a[key])(th_of_z)
    out["gradpar"] = np.full_like(zed, gpar_arc)
    out["theta"] = zed
    out["theta_of_zed"] = th_of_z
    return out


def miller(rhoc=0.5, rmaj=2.77778, rgeo=None, shift=0.0, qinp=1.4, shat=0.796, kappa=1.0, kapprim=0.0,
           tri=0.0, triprim=0.0, nzed_local=256, nperiod=1, zed_equal_arc=False, betaprim=0.0, **ignored):
    """TabulatedGeometry for a Miller surface (stella's &millergeo_parameters; extra keys ignored).

    The field-line coordinate is θ on [−(2 nperiod − 1)π, (2 nperiod − 1)π], or stella's equal-arc
    coordinate if zed_equal_arc (needed when a stella primary is defined in zed). For nperiod = 1 the
    tube is one poloidal turn and the geometry is periodic (Fourier interpolation).
    """
    a = miller_arrays(rhoc, rmaj, rgeo, shift, qinp, shat, kappa, kapprim, tri, triprim, int(nzed_local), nperiod,
                      betaprim)
    if zed_equal_arc:
        a = equal_arc(a, nperiod)
    exb = 0.5 * (qinp / rhoc) * a["dpsipdrho"]          # −½/C · dx/dψ · dy/dα with C = −1
    g = TabulatedGeometry(a["theta"], a["bmag"], a["gradpar"], a["gds22"], a["gbdrift0"], shat,
                          q=qinp, eps=rhoc / rmaj, rmaj=rmaj, exb_nonlin_fac=exb, periodic=(nperiod == 1),
                          gxx=a["gxx"], vmx=a["vmx"],
                          name="miller rhoc=%g rmaj=%g q=%g shat=%g kappa=%g tri=%g" % (rhoc, rmaj, qinp, shat, kappa, tri),
                          extra=a)
    g.sign_torflux = -1                                  # stella's Miller convention
    g.params = dict(rhoc=rhoc, rmaj=rmaj, rgeo=rgeo if rgeo is not None else rmaj, shift=shift, qinp=qinp,
                    shat=shat, kappa=kappa, kapprim=kapprim, tri=tri, triprim=triprim, nzed_local=nzed_local,
                    nperiod=nperiod, zed_equal_arc=zed_equal_arc)
    return g
