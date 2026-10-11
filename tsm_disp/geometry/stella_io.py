"""Read the geometry of a stella run: <prefix>.out.nc (full precision, preferred) or <prefix>.geometry.

The `.geometry` text file has 4 significant digits and its header changed between stella versions
(old: dxdXcoord, exb_nonlin_p; new: dxdpsi, flux_fac); columns are therefore matched by name.
The input file (<prefix>.in, f90nml) supplies geo_option, nperiod and the Miller parameters.
"""
import glob
import os
import numpy as np
from .base import TabulatedGeometry


def _find(run_dir, prefix, ext):
    if prefix is not None:
        p = os.path.join(run_dir, prefix + ext)
        return p if os.path.exists(p) else None
    c = sorted(glob.glob(os.path.join(run_dir, "*" + ext)))
    return c[0] if c else None


def read_input(path):
    import f90nml
    return f90nml.read(path)


def read_geometry_txt(path):
    """Parse a stella .geometry file: returns (header dict, dict of columns for the first α)."""
    lines = open(path).read().splitlines()
    names = lines[0].lstrip("#").split()
    vals = lines[1].lstrip("#").split()
    hdr = {}
    for n, v in zip(names, vals):
        try:
            hdr[n] = float(v)
        except ValueError:
            hdr[n] = float(v.replace("-", "E-").replace("+", "E+").lstrip("E")) if "E" not in v else np.nan
    i = next(k for k, l in enumerate(lines) if "alpha" in l and "zed" in l)
    cols = lines[i].replace("#", " ").split()
    rows = []
    for l in lines[i + 1:]:
        if not l.strip():
            if rows:
                break
            continue
        rows.append([_fortran_float(x) for x in l.split()])
    a = np.array(rows)
    return hdr, {c: a[:, k] for k, c in enumerate(cols)}


def _fortran_float(s):
    try:
        return float(s)
    except ValueError:          # e.g. '-0.4492-115' (3-digit exponent without E)
        for k in range(len(s) - 1, 0, -1):
            if s[k] in "+-" and s[k - 1] not in "eE":
                return float(s[:k] + "E" + s[k:])
        raise


def from_stella_run(run_dir, prefix=None, source="auto", periodic=None):
    """TabulatedGeometry (z = stella's zed) of a stella run directory.

    source: 'auto' (netCDF if present, else .geometry), 'nc' or 'geometry'.
    periodic: default True for Miller tubes with nperiod = 1, False for VMEC (from the input file).
    """
    nc_path = _find(run_dir, prefix, ".out.nc")
    geo_path = _find(run_dir, prefix, ".geometry")
    in_path = _find(run_dir, prefix, ".in")
    nml = read_input(in_path) if in_path else {}
    geo_option = str(nml.get("geo_knobs", {}).get("geo_option", "miller")).lower()
    zg = nml.get("zgrid_parameters", {})
    nperiod = int(zg.get("nperiod", 1))
    mil = nml.get("millergeo_parameters", {})
    if periodic is None:
        periodic = (geo_option != "vmec") and nperiod == 1
    use_nc = (source == "nc") or (source == "auto" and nc_path is not None)
    hdr = {}
    if geo_path:
        try:
            hdr, _ = read_geometry_txt(geo_path)
        except Exception:
            hdr = {}
    if use_nc:
        import netCDF4
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        with netCDF4.Dataset(nc_path) as d:
            def g(v):
                x = np.asarray(d.variables[v][:], float)
                return x[:, 0] if x.ndim == 2 else x
            z = g("zed")
            bmag, gds22, gb0 = g("bmag"), g("gds22"), g("gbdrift0")
            grho = g("grho") if "grho" in d.variables else None
            ky_extra = {v: g(v) for v in ("gds2", "gds21", "gbdrift", "cvdrift") if v in d.variables}
            gradpar = g("b_dot_grad_z") if "b_dot_grad_z" in d.variables else g("gradpar")
            shat, q = float(d.variables["shat"][:]), float(d.variables["q"][:])
            drhodpsi = float(d.variables["drhodpsi"][:]) if "drhodpsi" in d.variables else None
        src = nc_path
    else:
        if geo_path is None:
            raise FileNotFoundError("no .out.nc or .geometry in %s" % run_dir)
        hdr, c = read_geometry_txt(geo_path)
        z, bmag, gds22, gb0 = c["zed"], c["bmag"], c["gds22"], c["gbdrift0"]
        grho = c.get("grho")
        ky_extra = {v: c[v] for v in ("gds2", "gds21", "gbdrift", "cvdrift") if v in c}
        gradpar = c["bdot_grad_z"]
        shat, q = hdr["shat"], hdr["qinp"]
        drhodpsi = None
        src = geo_path
    # E×B factor: Miller −½/C dx/dψ dy/dα = ½ (q/ρ)(dψ/dρ); else the (4-digit) header value
    exb = None
    if geo_option != "vmec" and drhodpsi and "rhoc" in mil:
        exb = 0.5 * q / float(mil["rhoc"]) / drhodpsi
    elif "exb_nonlin" in hdr:
        exb = abs(hdr["exb_nonlin"])
    if periodic:                 # enforce exact periodicity of the duplicated end point
        for arr in (bmag, gds22, gb0, gradpar):
            arr[-1] = arr[0]
    rmaj = float(mil["rmaj"]) if "rmaj" in mil else None
    eps = float(mil["rhoc"]) / rmaj if rmaj else None
    geo = TabulatedGeometry(z, bmag, gradpar, gds22, gb0, shat, q=q, eps=eps, rmaj=rmaj, exb_nonlin_fac=exb,
                            periodic=periodic, name="stella %s (%s)" % (os.path.basename(os.path.normpath(run_dir)),
                                                                       os.path.basename(src)),
                            extra=dict(ky_extra, **({"grho": grho} if grho is not None else {})))
    geo.sign_torflux = -1 if geo_option != "vmec" else None
    geo.source, geo.input, geo.header = src, nml, hdr
    return geo
