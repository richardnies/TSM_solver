"""Miller local equilibrium, ported from stella (geometry/geometry_miller.f90, get_local_geo).

Only the k_y = 0 quantities are computed (bmag, gradpar = b·∇θ, gds22, gbdrift0 = cvdrift0,
grho, dψ_N/dρ), with stella's discretisation: R, Z on three radii ρ ± dr with dr = 10⁻³ ρ/R0,
centred differences in ρ and θ (periodic ends), trapezoid θ integrals. The drifts do not depend on
betaprim, betadbprim, d2qdr2 or d2psidr2, which therefore are not arguments.

Units: lengths in a (minor radius), B in B_ref = toroidal field at R = rgeo; θ is the geometric
poloidal angle of the Miller parametrisation, which is the field-line coordinate z.
"""
import numpy as np
from .base import TabulatedGeometry


def miller_arrays(rhoc=0.5, rmaj=2.77778, rgeo=None, shift=0.0, qinp=1.4, shat=0.796, kappa=1.0,
                  kapprim=0.0, tri=0.0, triprim=0.0, nzed_local=256, nperiod=1):
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
    return dict(theta=theta, R=R2, Z=Z[1], dRdrho=dRdrho, dZdrho=dZdrho, dRdth=dRdth, dZdth=dZdth,
                jacrho=jacrho, grho=grho, bmag=bmag, dBdth=dBdth, gradpar=gradpar, gds22=gds22,
                gbdrift0=cvdrift0.copy(), cvdrift0=cvdrift0, dpsipdrho=dpsipdrho, dqdr=dqdr, bi=bi)


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
    for key in ("bmag", "gds22", "gbdrift0", "cvdrift0", "grho", "R", "Z"):
        out[key] = _Fourier(th[0], th[-1] - th[0], a[key][:-1])(th_of_z)
    out["gradpar"] = np.full_like(zed, gpar_arc)
    out["theta"] = zed
    out["theta_of_zed"] = th_of_z
    return out


def miller(rhoc=0.5, rmaj=2.77778, rgeo=None, shift=0.0, qinp=1.4, shat=0.796, kappa=1.0, kapprim=0.0,
           tri=0.0, triprim=0.0, nzed_local=256, nperiod=1, zed_equal_arc=False, **ignored):
    """TabulatedGeometry for a Miller surface (stella's &millergeo_parameters; extra keys ignored).

    The field-line coordinate is θ on [−(2 nperiod − 1)π, (2 nperiod − 1)π], or stella's equal-arc
    coordinate if zed_equal_arc (needed when a stella primary is defined in zed). For nperiod = 1 the
    tube is one poloidal turn and the geometry is periodic (Fourier interpolation).
    """
    a = miller_arrays(rhoc, rmaj, rgeo, shift, qinp, shat, kappa, kapprim, tri, triprim, nzed_local, nperiod)
    if zed_equal_arc:
        a = equal_arc(a, nperiod)
    exb = 0.5 * (qinp / rhoc) * a["dpsipdrho"]          # −½/C · dx/dψ · dy/dα with C = −1
    g = TabulatedGeometry(a["theta"], a["bmag"], a["gradpar"], a["gds22"], a["gbdrift0"], shat,
                          q=qinp, eps=rhoc / rmaj, rmaj=rmaj, exb_nonlin_fac=exb, periodic=(nperiod == 1),
                          name="miller rhoc=%g rmaj=%g q=%g shat=%g kappa=%g tri=%g" % (rhoc, rmaj, qinp, shat, kappa, tri),
                          extra=a)
    g.params = dict(rhoc=rhoc, rmaj=rmaj, rgeo=rgeo if rgeo is not None else rmaj, shift=shift, qinp=qinp,
                    shat=shat, kappa=kappa, kapprim=kapprim, tri=tri, triprim=triprim, nzed_local=nzed_local,
                    nperiod=nperiod, zed_equal_arc=zed_equal_arc)
    return g
