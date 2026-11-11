"""Sphinx configuration for the qtpyt documentation."""
import os
import sys

# Document the source tree directly; the package is not pip-installed on Read the Docs.
sys.path.insert(0, os.path.abspath(".."))

project = "qtpyt"
author = "Guido Gandus and contributors"
copyright = "qtpyt developers"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Pages may be written in Markdown (.md) or reStructuredText (.rst).
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# NumPy-style docstrings, see .claude/skills/physics-docstrings.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

autosummary_generate = True
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": True}

# Not available on the build machine: MPI, GPAW and the compiled decimation kernel.
# cupy and mkl_fft must NOT be mocked: they are optional imports inside try/except,
# and a mock would be picked up as the array or FFT backend.
autodoc_mock_imports = ["mpi4py", "gpaw", "qtpyt.surface._recursive"]

# Modules left out of the API pages: GPU-only (need CuPy), a build script, and
# tk_gpaw, which imports modules missing from the repository.
autosummary_context = {
    "api_skip": [
        "qtpyt.kernels.hs_gpu",
        "qtpyt.kernels.linalg_gpu",
        "qtpyt.surface.setup",
        "qtpyt.lo.tk_gpaw",
        # The package re-exports numpy's `fft` under this name; the module itself
        # imports `use_pyfftw`, which does not exist.
        "qtpyt.screening.fft",
    ]
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_title = "qtpyt"
html_theme_options = {
    "github_url": "https://github.com/aj0337/qtpyt",
    "navbar_align": "left",
    "show_toc_level": 2,
}
