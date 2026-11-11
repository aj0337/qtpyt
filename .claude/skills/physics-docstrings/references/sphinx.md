# Sphinx settings the docstrings assume

The docstrings written by this skill render correctly with the following `docs/conf.py` settings. Propose these when the user sets up the docs; do not create `docs/` unasked.

```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",     # NumPy-style sections
    "sphinx.ext.mathjax",      # :math: and .. math::
    "sphinx.ext.intersphinx",  # links to numpy/scipy/ase docs
    "sphinx.ext.viewcode",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False         # return type shown inline with the description
napoleon_preprocess_types = True   # "array_like, shape (n, n)" etc. rendered cleanly

autodoc_typehints = "signature"    # hints in the signature, physics in the description
autodoc_member_order = "bysource"
autosummary_generate = True

# Not importable on ReadTheDocs builders: optional GPU backend and the compiled extension.
autodoc_mock_imports = ["cupy", "qtpyt.surface._recursive"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase", None),
}
```

Why `autodoc_typehints = "signature"`: the type hint states the Python type once in the signature, and the docstring's type field adds what the hint cannot (shape, physical meaning). With `"description"` Sphinx would print the type a second time next to the docstring's own.

## Common rendering problems

- Backslashes vanish or turn into odd characters in math: the docstring is not a raw string. Use `r"""`.
- A section heading renders as plain text: the dashed underline must be at least as long as the heading, and there must be a blank line before the heading.
- A cross-reference renders as plain monospace: the target path is wrong, or the object is not included in any autodoc directive.
- `.. math::` block shows as text: needs a blank line before and after, and the equation indented under the directive.
