# Installation

qtpyt needs Python ≥ 3.7 and

- numpy ≥ 1.21, < 1.22
- scipy
- numba ≥ 0.55
- ase
- mpi4py, for the parallel energy grid
- Cython and a C compiler with OpenMP, for the compiled surface Green's function

Optional: CuPy, which makes `qtpyt.xp` a GPU backend, and GPAW, for
{mod}`qtpyt.lo.tk_gpaw`.

From the repository root:

```bash
pip install -e .
```
