import os
import sys
import warnings
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kernels_importable():
    try:
        import tsm_disp._kernels  # noqa: F401
        return True
    except ImportError:
        for m in [m for m in sys.modules if m == "tsm_disp" or m.startswith("tsm_disp.")]:
            del sys.modules[m]
        return False


# Use the source tree if its extension is built in place (python setup.py build_ext --inplace, or
# pip install -e .); otherwise make sure an uncompiled checkout does not shadow the installed package.
sys.path.insert(0, ROOT)
if not _kernels_importable():
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != ROOT]
    if not _kernels_importable():
        raise ImportError("tsm_disp._kernels not found: run `pip install .` or `python setup.py build_ext --inplace`")
    sys.path.append(ROOT)            # for the disp_rel compatibility shim at the repository root
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tsm_disp")

OLD_CODE = "/scratch/gpfs/rnies/2023-03-09_generalised_secondary/2023-03-09_toroidal_secondary/code_eval_disp_NEW"
MILLER_RUN = ("/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2025-10-13_gammaE_effect_on_turbulence/"
              "2026-10-02_scan_gammaE_phiZ_frozen_eps_q/run_g-RHphiNL_eps-0.180_qinp-1.400_fprim-2.200_tprim-6.90_"
              "nx-96_ny-48_nzed-24_nvgrid-16_nmu-8_wall-02h/run_g_exb-0.2160")
MILLER_ARC_RUNS = ["/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2024-08-28_compare_QS_tok/2025-03-28_run_stella_rhoc/run_tprim-1.5_rhoc-0.5",
                   "/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2024-08-28_compare_QS_tok/2025-02-01_geometry_files/tok_like_precise_QA"]
VMEC_RUN = "/scratch/gpfs/rnies/vmec_configurations/evaluate_true_flux_surface_averages/precise_QH/runs/run_tube_pos-0.2500_nturns_tube-1.0000"
WOUT_QH = "/scratch/gpfs/rnies/vmec_configurations/wout_precise_QH.nc"

#: legacy fac_norm of the Gaussian primary (width π) used by all drivers
FAC_NORM = 0.7733975858850816


def needs(path):
    return pytest.mark.skipif(not os.path.exists(path), reason="missing %s" % path)


@pytest.fixture(scope="session")
def old_kernels():
    if OLD_CODE not in sys.path:
        sys.path.append(OLD_CODE)
    try:
        import disp_rel_utils
    except ImportError:
        pytest.skip("old compiled kernels not importable")
    return disp_rel_utils


def ref_point():
    """The reference local point used throughout (CBC-like TSM parameters)."""
    tau, kr, th, y, eps = 1.0, 0.5, 0.4, 0.3, 0.18
    omega = 0.96189 + 0.40884j
    A = 4 * 0.3557 * np.exp(-(th / np.pi)**2) / FAC_NORM * kr
    oE = -A * np.sin(y)
    oM = kr * np.sin(th) / 2.7778
    b = 0.5 * kr**2 * (1 + eps * np.cos(th))**2
    return dict(omega=omega, oE=oE, oM=oM, b=b, tau=tau, eta=(3.0, 3.0), A=A, y=y)
