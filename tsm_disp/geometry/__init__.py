"""Field-line geometry for the secondary dispersion relation.

Only k_y = 0 quantities enter (the secondary is zonal), so a geometry supplies, along the field-line
coordinate z on [zmin, zmax]:

    vMx(z)          radial magnetic drift per unit energy, in ρ_ref v_th/a: ω_M = k_x vMx (v∥² + v⊥²/2)/v_th²
    kperp2_kx2(z)   |∇x|²/B², so that b = ½ k_x² kperp2_kx2 (local ρ ∝ 1/B)
    weight(z)       1/(B·∇z), normalised to ∫ weight dz = 1 (field-line / flux-surface average)

Constructors: CircularAnalytic, Uniform (analytic); TabulatedGeometry (arrays); miller(...);
vmec(...); from_stella_run(...).
"""
from .base import Geometry, TabulatedGeometry
from .analytic import CircularAnalytic, Uniform
# imported eagerly so that importing a submodule cannot shadow the constructor of the same name
from .miller import miller
from .vmec import vmec                  # simsopt is imported inside vmec()
from .stella_io import from_stella_run

__all__ = ["Geometry", "TabulatedGeometry", "CircularAnalytic", "Uniform", "miller", "vmec", "from_stella_run"]
