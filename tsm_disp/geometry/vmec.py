"""Field-line geometry from a VMEC wout file, through simsopt's vmec_fieldlines, on stella's grid.

stella's field line: α = θ_pest − ι ζ (ζ the VMEC toroidal angle), ζ_j = zeta_center +
π j nfield_periods/(nfp nzgrid), j = −nzgrid … nzgrid; zed = (ζ − zeta_center) nfp/nfield_periods or,
with zed_equal_arc, the normalised arc length on [−π, π]. Only k_y = 0 quantities are used
(bmag, b·∇z, gds22, gbdrift0), which do not depend on the magnetic-shear shift of α.
The tube is generally not periodic (field-line average, not a flux-surface average).
"""
import numpy as np
from .base import TabulatedGeometry


def vmec(wout, *, torflux=None, s=None, alpha0=0.0, zeta_center=0.0, nfield_periods=1.0, nzed=512,
         zed_equal_arc=True, refine=8):
    """TabulatedGeometry for a VMEC flux tube (stella's &vmec_parameters names; s = torflux)."""
    from simsopt.mhd.vmec import Vmec
    from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
    if s is None:
        s = 0.6354167 if torflux is None else torflux
    vs = Vmec(wout)
    nfp = vs.wout.nfp
    nzg = (nzed // 2) * (refine if zed_equal_arc else 1)
    j = np.arange(-nzg, nzg + 1)
    zeta = zeta_center + np.pi * j * nfield_periods / (nfp * nzg)
    fl = vmec_fieldlines(vs, s, alpha0, phi1d=zeta, phi_center=0.0)
    bmag = fl.bmag[0, 0]
    gpar_phi = fl.gradpar_phi[0, 0]
    gds22 = fl.gds22[0, 0]
    gb0 = fl.gbdrift0[0, 0]
    gb, cv = fl.gbdrift[0, 0], fl.cvdrift[0, 0]
    gpsi2 = fl.grad_psi_dot_grad_psi[0, 0]
    shat = float(fl.shat[0])
    q = 1.0 / float(fl.iota[0])
    dz_dzeta = nfp / nfield_periods
    if zed_equal_arc:
        # arc length ∝ ∫ dζ/(b·∇ζ); normalised to [−π, π] (stella get_arc_length_grid)
        f = 1.0 / gpar_phi
        arc = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(zeta))])
        L = arc[-1]
        zarc = -np.pi + 2 * np.pi * arc / L
        zgrid = np.linspace(-np.pi, np.pi, 2 * (nzed // 2) + 1)
        from scipy.interpolate import CubicSpline
        zeta_of_z = CubicSpline(zarc, zeta)(zgrid)
        fl2 = vmec_fieldlines(vs, s, alpha0, phi1d=zeta_of_z, phi_center=0.0)
        bmag, gds22, gb0 = fl2.bmag[0, 0], fl2.gds22[0, 0], fl2.gbdrift0[0, 0]
        gb, cv = fl2.gbdrift[0, 0], fl2.cvdrift[0, 0]
        gpsi2 = fl2.grad_psi_dot_grad_psi[0, 0]
        gradpar = np.full_like(zgrid, 2 * np.pi / L)
        z = zgrid
        zeta = zeta_of_z
    else:
        z = (zeta - zeta_center) * dz_dzeta
        gradpar = gpar_phi / dz_dzeta
        f = 1.0 / gpar_phi
        arc = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(zeta))])
        L = arc[-1]
    geo = TabulatedGeometry(z, bmag, gradpar, gds22, gb0, shat, q=q, eps=None, rmaj=None,
                            exb_nonlin_fac=0.5, periodic=False,
                            name="vmec %s s=%g alpha0=%g zeta_c=%g nfp_tube=%g" % (str(wout).split('/')[-1], s, alpha0,
                                                                                  zeta_center, nfield_periods))
    # k_y ≠ 0 drifts (not used by the solver) and the field-line length, for diagnostics:
    # stella's ω_d,y contains ½ k_y (cvdrift v∥² + gbdrift v⊥²/2), so v_My = gbdrift/2 when cvdrift = gbdrift
    # |∇ρ| with ρ = √s = √(ψ/ψ_edge), in 1/L_ref (stella's grho)
    grho = np.sqrt(gpsi2) * fl.L_reference / (2 * np.sqrt(s) * abs(fl.edge_toroidal_flux_over_2pi))
    geo.extra = dict(gbdrift=np.asarray(gb), cvdrift=np.asarray(cv), zeta=np.asarray(zeta), length=float(L),
                     grho=np.asarray(grho))
    geo.sign_torflux = int(np.sign(fl.edge_toroidal_flux_over_2pi))
    geo.params = dict(wout=str(wout), s=s, alpha0=alpha0, zeta_center=zeta_center, nfield_periods=nfield_periods,
                      nfp=nfp, iota=1 / q)
    return geo
