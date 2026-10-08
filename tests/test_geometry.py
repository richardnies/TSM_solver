"""Geometry constructors against stella's own output."""
import glob
import numpy as np
import pytest
from conftest import MILLER_RUN, MILLER_ARC_RUNS, VMEC_RUN, WOUT_QH, needs
from tsm_disp.geometry import CircularAnalytic, miller, from_stella_run
from tsm_disp.geometry.miller import miller_arrays


@needs(MILLER_RUN)
@pytest.mark.fast
def test_miller_port_vs_stella_miller_output():
    import f90nml
    nml = f90nml.read(MILLER_RUN + "/CBC.in")["millergeo_parameters"]
    a = miller_arrays(**{k: nml[k] for k in ("rhoc", "rmaj", "rgeo", "qinp", "shat", "kappa", "nzed_local")})
    cols = open(MILLER_RUN + "/geometry_miller.CBC.output").readlines()[1].split()
    name = {c.split(".", 1)[1]: i for i, c in enumerate(cols)}
    dat = np.loadtxt(MILLER_RUN + "/geometry_miller.CBC.output", skiprows=2)
    for k, sk in [("bmag", "bmag"), ("gradpar", "gradpar"), ("gds22", "gds22"), ("gbdrift0", "gbdrift0"),
                  ("jacrho", "jacr"), ("R", "R")]:
        assert np.max(np.abs(a[k] - dat[:, name[sk]])) < 1e-8        # e18.9 output
    hdr = open(MILLER_RUN + "/geometry_miller.CBC.output").readline()
    assert abs(a["dpsipdrho"] - float(hdr.split("dpsi/dr:")[1])) < 1e-9


@needs(MILLER_RUN)
@pytest.mark.fast
def test_from_stella_run_nc_and_text():
    import f90nml
    gs = from_stella_run(MILLER_RUN)
    gt = from_stella_run(MILLER_RUN, source="geometry")
    gm = miller(**f90nml.read(MILLER_RUN + "/CBC.in")["millergeo_parameters"])
    z = gs.z_grid
    for nm in ("vMx", "kperp2_kx2", "weight"):
        assert np.max(np.abs(getattr(gs, nm)(z) - getattr(gm, nm)(z))) < 3e-6      # stella's spline to nzed=24
        assert np.max(np.abs(getattr(gt, nm)(z) - getattr(gs, nm)(z))) < 2e-3      # 4 significant digits
    assert abs(gs.exb_nonlin_fac - 0.5083) < 1e-4 and abs(gs.exb_nonlin_fac - gm.exb_nonlin_fac) < 1e-9


@pytest.mark.fast
@pytest.mark.parametrize("run", MILLER_ARC_RUNS)
def test_miller_equal_arc_vs_stella(run):
    import f90nml
    if not glob.glob(run + "/*.out.nc"):
        pytest.skip("missing run")
    nml = f90nml.read(glob.glob(run + "/*.in")[0])
    gs = from_stella_run(run)
    gm = miller(**nml["millergeo_parameters"], zed_equal_arc=True)
    z = gs.z_grid
    for nm in ("vMx", "kperp2_kx2", "weight"):
        assert np.max(np.abs(getattr(gs, nm)(z) - getattr(gm, nm)(z))) < 1e-10


@pytest.mark.fast
def test_circular_limit_of_miller():
    """Miller CBC vs the analytic circular model: same |v_Mx|, b and weight up to O(ε²)."""
    gm = miller(rhoc=0.5, rmaj=2.7778, qinp=20, shat=0.8)
    gc = CircularAnalytic(eps=0.5 / 2.7778, rmaj=2.7778)
    z = np.linspace(-3, 3, 31)
    eps2 = (0.5 / 2.7778)**2
    assert np.max(np.abs(-gm.vMx(z) - gc.vMx(z))) < 2 * eps2 / 2.7778          # stella's sign: v_Mx = −sin θ/R
    assert np.max(np.abs(gm.kperp2_kx2(z) / gc.kperp2_kx2(z) - 1)) < 2 * eps2
    assert np.max(np.abs(gm.weight(z) / gc.weight(z) - 1)) < 2 * eps2
    assert abs(gm.exb_nonlin_fac - 0.5 / np.sqrt(1 - (0.5 / 2.7778)**2)) < 1e-3


@needs(VMEC_RUN)
@needs(WOUT_QH)
@pytest.mark.slow
def test_vmec_vs_stella_vmec_run():
    pytest.importorskip("simsopt")
    from tsm_disp.geometry import vmec
    gs = from_stella_run(VMEC_RUN)
    gv = vmec(WOUT_QH, torflux=0.635, alpha0=-1.082318298664033, zeta_center=0.3941387995434229,
              nfield_periods=1.4504941148886465, nzed=1024)
    z = np.linspace(-3.1, 3.1, 25)
    for nm in ("vMx", "kperp2_kx2", "weight"):
        a, b = getattr(gs, nm)(z), getattr(gv, nm)(z)
        assert np.max(np.abs(a - b)) < 1e-5 * np.max(np.abs(a))
    assert abs(gs.shat - gv.shat) < 1e-5 * abs(gs.shat) * 1e3 and abs(gs.q - gv.q) < 1e-6
