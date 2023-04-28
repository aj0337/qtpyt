#!/usr/bin/env python3
""" qtpyt is the a python package for quantum transport simulations."""
DOCLINES = (__doc__ or "").split("\n")

import os
import sys

import pkg_resources

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

from distutils.command.build_ext import build_ext

from setuptools import Extension, find_packages, setup

macros = [("NPY_NO_DEPRECATED_API", "0")]

extensions = []

try:
    import Cython

    USE_CYTHON = True
except:
    USE_CYTHON = False

suffix = ".pyx" if USE_CYTHON else ".c"

if USE_CYTHON:
    from Cython.Build import cythonize

ext_cython = {
    "qtpyt.surface._recursive": {},
}

extra_compile_args = ["-fopenmp", "-march=native", "-O3"]  # , "-ffast-math"]
extra_link_args = ["-fopenmp"]

np_incl = pkg_resources.resource_filename("numpy", "core/include")

for ext_name, ext_dict in ext_cython.items():

    source = ext_name.replace(".", os.path.sep) + suffix
    ext = Extension(
        ext_name,
        sources=[source] + ext_dict.get("sources", []),
        include_dirs=[np_incl] + ext_dict.get("include", []),
        define_macros=macros + ext_dict.get("macros", []),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    if USE_CYTHON:
        extensions += cythonize(ext, quiet=False)

    else:
        extensions.append(ext)

NAME = "qtpyt"
VERSION = "0.5.3"
URL = "https://gitlab.ethz.ch/ggandus/qtpyt.git"

AUTHOR = "Guido Gandus"
EMAIL = "gandusgui@gmail.com"

LICENSE = "MIT License"


def readme():
    if not os.path.exists("README.md"):
        return ""
    return open("README.md", "r").read()


requires = {
    "python_requires": ">= " + "3.7",
    "install_requires": [
        "setuptools",
        "numpy >= " + "1.21" + ",<1.22",
        "scipy",
        "numba >= " + "0.55",
        "ase",
    ],
    "setup_requires": ["numpy >= " + "1.21",],
}

metadata = dict(
    name="qtpyt",
    description=DOCLINES[0],
    long_description=readme(),
    url="https://gitlab.ethz.ch/ggandus/qtpyt.git",
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    platforms="any",
    test_suite="pytest",
    version=VERSION,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=find_packages(include=["qtpyt", "qtpyt.*", "qttools", "qttools.*"]),
    entry_points={"console_scripts": ["qt-cp2k = qttools.cp2k.prepare_kpts:main"]},
    ext_modules=extensions,
    **requires
)


if __name__ == "__main__":
    setup(**metadata)
