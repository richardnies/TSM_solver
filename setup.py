"""Build and install TSM_solver: the Python package `tsm_disp` (the solver) and `evaluation_scripts` (the helpers of
the examples; needs matplotlib, extra `plot`), so that the example scripts run from any directory.

    pip install ".[plot]"         # or: pip install -e ".[plot]"   (editable, for development)
    python setup.py build_ext --inplace   # build the extension in place without installing

OpenMP is used when the C compiler supports -fopenmp (gcc, Intel); otherwise the extension is built
without it and runs single-threaded (e.g. Apple clang). Set TSM_NO_OPENMP=1 to force a serial build.
"""
import os
import shutil
import sys
import tempfile
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


def compiler_accepts(flags, src="int main(void){return 0;}\n", link=True):
    """True if the C compiler compiles (and links) a test program with the extra flags."""
    try:                                   # Python >= 3.12 has no distutils; setuptools vendors it
        from setuptools._distutils.ccompiler import new_compiler
        from setuptools._distutils.sysconfig import customize_compiler
    except ImportError:
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler
    cc = new_compiler()
    customize_compiler(cc)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.c")
    with open(path, "w") as f:
        f.write(src)
    try:
        obj = cc.compile([path], output_dir=d, extra_postargs=flags)
        if link:
            cc.link_executable(obj, os.path.join(d, "t"), extra_postargs=flags)
        return True
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def has_openmp():
    if os.environ.get("TSM_NO_OPENMP"):
        return False
    return compiler_accepts(["-fopenmp"], "#include <omp.h>\nint main(void){return omp_get_max_threads() > 0 ? 0 : 1;}\n")


omp = has_openmp()
print("TSM_solver: building %s OpenMP" % ("with" if omp else "WITHOUT"), file=sys.stderr)
flags = ["-O3"] + (["-fopenmp"] if omp else [])
# inline complex multiplication/division (no C99 Annex G NaN/inf recovery calls __muldc3/__divdc3,
# ~20 % of the run time); gcc and recent clang accept it
if compiler_accepts(["-Werror", "-fcx-fortran-rules"], link=False):
    flags.append("-fcx-fortran-rules")
    print("TSM_solver: inline complex arithmetic (-fcx-fortran-rules)", file=sys.stderr)

ext = Extension("tsm_disp._kernels", sources=["tsm_disp/_kernels.pyx"],
                extra_compile_args=flags, extra_link_args=(["-fopenmp"] if omp else []))

here = os.path.dirname(os.path.abspath(__file__))
setup(
    name="TSM_solver",
    version="0.1.0",
    description="Dispersion relation of toroidal secondary modes of zonal flows (Nies & Parra, PPCF 68 (2026) 045028)",
    long_description=open(os.path.join(here, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Richard Nies",
    packages=find_packages(include=["tsm_disp", "tsm_disp.*", "evaluation_scripts", "evaluation_scripts.*"]),
    package_data={"tsm_disp": ["_kernels.pyx", "_jasym.pxi"]},
    ext_modules=cythonize([ext], language_level=3),
    python_requires=">=3.8",
    install_requires=["numpy>=1.19", "scipy>=1.6"],
    extras_require={
        "stella": ["f90nml", "netCDF4"],            # read stella runs, decks
        "vmec": ["simsopt"],                         # VMEC geometry
        "plot": ["matplotlib"],
        "test": ["pytest", "mpmath", "f90nml", "netCDF4"],
    },
    zip_safe=False,
)
