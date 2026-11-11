"""Reductions over MPI ranks."""
from __future__ import annotations

import numpy as np
from qtpyt.parallel import MPI, comm


def comm_sum(a : np.ndarray) -> np.ndarray:
    """Sum of an array over all MPI ranks, after summing its first axis if it has more than two.

    Use it to add up contributions of the energies held by each rank, for
    example to an energy integral.

    Parameters
    ----------
    a : ndarray
        Local contribution.

    Returns
    -------
    total : ndarray
        The same on every rank.
    """
    if a.ndim > 2:
        a = a.sum(0)
    if comm.size == 1:
        return a
    b = np.empty_like(a)
    mpitype = MPI._typedict[a.dtype.char]
    comm.Allreduce([a, a.size, mpitype], [b, mpitype], MPI.SUM)
    return b
