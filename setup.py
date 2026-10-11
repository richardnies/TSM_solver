"""Build and install TSM_solver (Python package `tsm_disp`).

    pip install .                 # or: pip install -e .   (editable, for development)
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


def has_openmp():
    if os.environ.get("TSM_NO_OPENMP"):
        return False
    try:                                   # Python >= 3.12 has no distutils; setuptools vendors it
        from setuptools._distutils.ccompiler import new_compiler
        from setuptools._distutils.sysconfig import customize_compiler
        from setuptools._distutils.errors import CompileError, LinkError
    except ImportError:
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler
        from distutils.errors import CompileError, LinkError
    cc = new_compiler()
    customize_compiler(cc)
    d = tempfile.mkdtemp()
    src = os.path.join(d, "t.c")
    with open(src, "w") as f:
        f.write("#include <omp.h>\nint main(void){return omp_get_max_threads() > 0 ? 0 : 1;}\n")
    try:
        obj = cc.compile([src], output_dir=d, extra_postargs=["-fopenmp"])
        cc.link_executable(obj, os.path.join(d, "t"), extra_postargs=["-fopenmp"])
        return True
    except (CompileError, LinkError, Exception):
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


omp = has_openmp()
print("TSM_solver: building %s OpenMP" % ("with" if omp else "WITHOUT"), file=sys.stderr)
flags = ["-O3"] + (["-fopenmp"] if omp else [])

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
    packages=find_packages(include=["tsm_disp", "tsm_disp.*"]),
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
