#!/usr/bin/env python3
"""Prepare stella secondary-mode decks in shaped Miller geometry (not submitted).

Template: the q = 20 CBC secondary deck of 2025-01-10_secondary_ITG (Θ = 0.2886, η∥ = η⊥ = 3, Gaussian
primary of width π in zed, k_y^P = 10⁻³). Changes: Miller shaping (κ, κ', δ, δ', Shafranov shift) and
phiinit chosen so that the true E×B amplitude Φ_P = 2·exb_nonlin_fac·k_y^P·phiinit·fac_norm equals the
target (exb_nonlin_fac of the shaped surface from the Miller port). Run with /home/rnies/stella_COPY/stella.

Usage: python make_shaped_decks.py [--out DIR]
"""
import argparse
import json
import os
import shutil
import sys
import numpy as np
import f90nml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tsm_disp.geometry import miller
from tsm_disp.primary import legacy_gaussian_norm

TEMPLATE = ("/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-09-28_secondary/2023-06-17_toroidal_neoclassical_secondary/"
            "2025-01-10_secondary_ITG/tpar-3.00_tprp-3.00_phase-0.00_fprim-0.00_tprim-0.00_qinp-20.00_vExP-0.71_kyP-0.0010_"
            "width0-3.14_phaseshift-0.2886/run_kxS-5.0000e-01/CBC.in")
OUT = ("/scratch/gpfs/rnies/2022-03-28_gyrokinetic_sims_stella/2022-09-28_secondary/2023-06-17_toroidal_neoclassical_secondary/"
       "2026-10-08_shaped_miller_benchmark")
SHAPE = dict(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, kappa=1.5, kapprim=0.3, tri=0.3, triprim=0.5,
             shift=-0.2, nzed_local=256)
PHI_TARGETS = [0.7232, 1.4464]          # = 2·0.5083·0.3557·{2, 4}: the CBC labels A = 2, 4 with stella's true amplitude
KX = np.round(np.arange(0.1, 1.0001, 0.1), 4)
Y0 = 1000.0

BATCH = """#!/bin/sh
#SBATCH -A pppl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00-06:00
#SBATCH --qos=pppl-short
#SBATCH --job-name=shaped_secondary

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for dirname in run_*/ ; do
	cd $dirname
	if ! test -f "CBC.out.nc"; then
		srun /home/rnies/stella_COPY/stella CBC.in
		rm -f *.error *.fluxes *.final_fields *.species.input millerlocal*
		rm -rf restart
	fi
	cd ..
done
"""


def main(out):
    geo = miller(**SHAPE, zed_equal_arc=True)
    fac = legacy_gaussian_norm(np.pi)
    meta = dict(shape=SHAPE, exb_nonlin_fac=geo.exb_nonlin_fac, fac_norm=fac, y0=Y0, eta_par=3, eta_prp=3,
                template=TEMPLATE, scans={})
    os.makedirs(out, exist_ok=True)
    for Phi in PHI_TARGETS:
        phiinit = Phi * Y0 / (2 * geo.exb_nonlin_fac * fac)
        vExP = phiinit / Y0 * fac            # the old drivers' "vExP"
        sdir = os.path.join(out, "PhiP-%.4f_kappa-%.2f_tri-%.2f_qinp-%.0f" % (Phi, SHAPE["kappa"], SHAPE["tri"], SHAPE["qinp"]))
        os.makedirs(sdir, exist_ok=True)
        open(os.path.join(sdir, "batch_launch_scan"), "w").write(BATCH)
        for kx in KX:
            rdir = os.path.join(sdir, "run_kxS-%.4e" % kx)
            os.makedirs(rdir, exist_ok=True)
            nml = f90nml.read(TEMPLATE)
            for k, v in SHAPE.items():
                nml["millergeo_parameters"][k] = v
            nml["kt_grids_box_parameters"]["x0"] = 1.0 / kx
            nml["kt_grids_box_parameters"]["y0"] = Y0
            nml["init_g_knobs"]["phiinit"] = float(phiinit)
            nml["knobs"]["delt"] = float(min(0.1, 0.2 / (kx * min(1.0, vExP))))
            nml["knobs"]["tend"] = float(500.0 * 0.7114 / vExP)
            nml.write(os.path.join(rdir, "CBC.in"), force=True)
        meta["scans"][os.path.basename(sdir)] = dict(Phi_P=Phi, phiinit=phiinit, vExP_legacy=vExP, kx=KX.tolist())
    json.dump(meta, open(os.path.join(out, "decks.json"), "w"), indent=1)
    shutil.copy(os.path.abspath(__file__), os.path.join(out, "make_shaped_decks.py"))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    main(ap.parse_args().out)
