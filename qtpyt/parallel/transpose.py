"""In-place transposition of matrices, without a second copy in memory."""
from __future__ import annotations

from numpy.typing import NDArray

import numpy as np
from numba import boolean, njit

# https://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix


@njit(["f8[:,::1](f8[:,::1])", "c16[:,::1](c16[:,::1])"], cache=True)
def transpose(a: NDArray) -> NDArray:
    """Transpose a C-ordered matrix in its own memory, by following permutation cycles.

    Parameters
    ----------
    a : ndarray of float64 or complex128, shape (n, m), C-ordered
        Overwritten: its memory then holds the transpose in C order.

    Returns
    -------
    aT : ndarray, shape (m, n)
        View of the same memory with the transposed shape. ``a`` itself
        keeps its old shape, so read the result from the return value.
    """
    n, m = a.shape
    a = a.reshape(-1)
    nm1 = n * m - 1
    visited = np.zeros(a.size, boolean)
    cycle = -1
    while cycle != a.size - 1:
        cycle += 1
        if visited[cycle]:
            continue
        w = cycle
        while True:
            w = nm1 if w == nm1 else (n * w) % nm1
            # swap
            tmp = a[w]
            a[w] = a[cycle]
            a[cycle] = tmp
            visited[w] = True
            if w == cycle:
                break
    return a.reshape(m, n)
