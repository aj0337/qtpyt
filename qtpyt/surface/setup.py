"""Build script for the compiled decimation extension ``_recursive``, for use from this directory."""
from setuptools import setup, Extension
from Cython.Build import cythonize


ext_modules = [
    Extension(
        "_recursive",
        ["_recursive.pyx"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    )
)
