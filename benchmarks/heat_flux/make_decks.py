#!/usr/bin/env python3
"""stella decks to check the primary heat-flux formula (tsm_disp.primary.heat_flux_stella) against stella's qflx.

Each deck is a 3-step `secondary = .true.` run (frozen primary, tiny zonal seed) of the template
(template.in: the shaped-Miller benchmark deck), varied in shaping, den0_phase, tpar0, tperp0, width0, y0.
Run with run_decks.sh; compare with compare.py. Usage: python make_decks.py
"""
import os
import f90nml

HERE = os.path.dirname(os.path.abspath(__file__))
CBC = dict(kappa=1.0, kapprim=0.0, tri=0.0, triprim=0.0, shift=0.0)
CASES = {
    "cbc_ph-0.5": dict(geo=CBC, init=dict(den0_phase=-0.5, tpar0=3.0, tperp0=3.0)),
    "cbc_ph+1.0_tpar0": dict(geo=CBC, init=dict(den0_phase=1.0, tpar0=0.0, tperp0=3.0)),
    "cbc_ph-0.8_narrow_y0-200": dict(geo=CBC, init=dict(den0_phase=-0.8, tpar0=3.0, tperp0=0.0, width0=1.0),
                                     box=dict(y0=200.0)),
    "shaped_ph-0.7": dict(geo={}, init=dict(den0_phase=-0.7, tpar0=3.0, tperp0=1.0)),
    "cbc_ph0": dict(geo=CBC, init=dict(den0_phase=0.0, tpar0=3.0, tperp0=3.0)),
    # velocity-resolution checks: more points only (nvgrid 72, nmu 32, vpa_max 4, vperp_max = 3 default), and
    # a larger v⊥ cut-off (vperp_max 4.5): the deviation from the formula is the truncation of the v⊥ integral
    "cbc_ph+1.0_tpar0_hires": dict(geo=CBC, init=dict(den0_phase=1.0, tpar0=0.0, tperp0=3.0),
                                   vgrid=dict(nvgrid=72, nmu=32, vpa_max=4.0)),
    "cbc_ph-0.5_hires": dict(geo=CBC, init=dict(den0_phase=-0.5, tpar0=3.0, tperp0=3.0),
                             vgrid=dict(nvgrid=72, nmu=32, vpa_max=4.0)),
    "cbc_ph+1.0_tpar0_vperp4.5": dict(geo=CBC, init=dict(den0_phase=1.0, tpar0=0.0, tperp0=3.0),
                                      vgrid=dict(nvgrid=72, nmu=32, vpa_max=4.5, vperp_max=4.5)),
    "cbc_ph-0.5_vperp4.5": dict(geo=CBC, init=dict(den0_phase=-0.5, tpar0=3.0, tperp0=3.0),
                                vgrid=dict(nvgrid=72, nmu=32, vpa_max=4.5, vperp_max=4.5)),
    "shaped_ph-0.7_vperp4.5": dict(geo={}, init=dict(den0_phase=-0.7, tpar0=3.0, tperp0=1.0),
                                   vgrid=dict(nvgrid=72, nmu=32, vpa_max=4.5, vperp_max=4.5)),
}

if __name__ == "__main__":
    for name, c in CASES.items():
        nml = f90nml.read(os.path.join(HERE, "template.in"))
        nml["millergeo_parameters"].update(c["geo"])
        nml["init_g_knobs"].update(c["init"])
        nml["kt_grids_box_parameters"].update(c.get("box", {}))
        nml["vpamu_grids_parameters"].update(c.get("vgrid", {}))
        nml["knobs"]["tend"] = 0.3
        nml["stella_diagnostics_knobs"].update(save_for_restart=False, write_phi_vs_time=False, write_kspectra=False)
        d = os.path.join(HERE, "runs", name)
        os.makedirs(d, exist_ok=True)
        nml.write(os.path.join(d, "CBC.in"), force=True)
        print("wrote", d)
