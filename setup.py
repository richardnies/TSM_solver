"""Build: python setup.py build_ext --inplace"""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "tsm_disp._kernels",
    sources=["tsm_disp/_kernels.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="tsm_disp",
    version="0.1.0",
    description="Dispersion relation of toroidal secondary modes (Nies & Parra, PPCF 68 (2026) 045028)",
    packages=find_packages(),
    ext_modules=cythonize([ext], language_level=3, annotate=False),
    python_requires=">=3.8",
    install_requires=["numpy", "scipy"],
)
