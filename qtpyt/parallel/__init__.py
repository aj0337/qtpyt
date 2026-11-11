"""MPI and thread setup for calculations distributed over an energy grid.

Importing this package creates ``comm`` (``MPI.COMM_WORLD``) and sets the
number of Numba, MKL and OpenBLAS threads to ``OMP_NUM_THREADS``, or to the
number of CPUs when it is unset. Requires ``mpi4py``.
"""
import os
import sys
from warnings import warn

from mpi4py import MPI
from numba import set_num_threads

comm = MPI.COMM_WORLD

num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
# if num_threads is not None and comm.size > 1:

# set to omp value
set_num_threads(num_threads)

os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

try:
    os.environ.get("NUMBA_NUM_THREADS", None)
except:
    warn(
        """NUMBA_NUM_THREADS not set!
        
        Please, set it to the desired OMP_NUM_THREADS."""
    )
