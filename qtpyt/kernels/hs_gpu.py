r"""Element-wise kernels that build :math:`zS - H` and :math:`\Gamma` in column-major layout (GPU, CuPy and Numba CUDA).

The inverse Green's function is assembled from blocks :math:`zS - H` and
the broadenings :math:`\Gamma = i(\Sigma - \Sigma^\dagger)`. These kernels
write them directly into Fortran-ordered arrays, the layout the LAPACK
solvers of :mod:`qtpyt.kernels` read, avoiding a separate transpose.
"""
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray

import numpy as np
import cupy as cp
from numba import cuda
from math import ceil

threadsperblock = (16, 16)


def get_blockspergrid(a: NDArray, threadsperblock: Tuple[int, int]) -> Tuple[int, int]:
    """CUDA grid size that covers every element of the 2D array ``a``.

    Parameters
    ----------
    a : ndarray, shape (n, m)
    threadsperblock : tuple of two int
    """
    blockspergrid_x = ceil(a.shape[0] / threadsperblock[0])
    blockspergrid_y = ceil(a.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return blockspergrid


@cuda.jit
def _geamT(z, h, s, m):
    x, y = cuda.grid(2)
    if x < h.shape[0] and y < h.shape[1]:
        m[y, x] = z * s[x, y] - h[x, y]


@cuda.jit
def _geamC(z, h, s, m):
    x, y = cuda.grid(2)
    if x < h.shape[0] and y < h.shape[1]:
        m[x, y] = z * s[x, y].conjugate() - h[x, y].conjugate()


@cuda.jit
def _get_lambda(sigma, lambd):
    x, y = cuda.grid(2)
    if x < sigma.shape[0] and y < sigma.shape[1]:
        lambd[y, x] = 1.0j * (sigma[x, y] - sigma[y, x].conjugate())


def geamT(z: complex, h: NDArray, s: NDArray, m: Optional[NDArray] = None) -> NDArray[np.complex128]:
    r"""Transpose of :math:`zS - H`, as a Fortran-ordered array.

    Parameters
    ----------
    z : complex
        Complex energy :math:`E + i\eta`.
    h, s : ndarray, shape (n, m), C-ordered
        Hamiltonian and overlap blocks.
    m : ndarray, shape (m, n), Fortran-ordered, optional
        Array for the result.

    Returns
    -------
    m : ndarray
        :math:`(zS - H)^T`, whose memory holds :math:`zS - H` in row-major order.
    """
    assert h.flags.c_contiguous & s.flags.c_contiguous
    if m is not None:
        assert m.flags.f_contiguous
    else:
        m = cp.empty(h.shape, complex, order="F")
    blockspergrid = get_blockspergrid(h, threadsperblock)
    _geamT[blockspergrid, threadsperblock](z, h, s, m)
    return m


def geamC(z: complex, h: NDArray, s: NDArray, m: Optional[NDArray] = None) -> NDArray[np.complex128]:
    r"""Complex conjugate :math:`zS^* - H^*`, as a Fortran-ordered array.

    With real :math:`z` this is the conjugate of :math:`zS - H`; for complex
    :math:`z` only :math:`S` and :math:`H` are conjugated.

    Parameters
    ----------
    z : complex
        Complex energy :math:`E + i\eta`.
    h, s : ndarray, shape (n, m), C-ordered
        Hamiltonian and overlap blocks.
    m : ndarray, shape (n, m), Fortran-ordered, optional
        Array for the result.

    Returns
    -------
    m : ndarray
    """
    assert h.flags.c_contiguous & s.flags.c_contiguous
    if m is not None:
        assert m.flags.f_contiguous
    else:
        m = cp.empty(h.shape, complex, order="F")
    blockspergrid = get_blockspergrid(h, threadsperblock)
    _geamC[blockspergrid, threadsperblock](z, h, s, m)
    return m


def get_lambda(sigma: NDArray, lambd: Optional[NDArray] = None) -> NDArray[np.complex128]:
    r"""Broadening :math:`\Gamma = i(\Sigma - \Sigma^\dagger)`, transposed, as a Fortran-ordered array.

    Parameters
    ----------
    sigma : ndarray, shape (n, n), C-ordered
        Retarded self-energy.
    lambd : ndarray, shape (n, n), Fortran-ordered, optional
        Array for the result.

    Returns
    -------
    lambd : ndarray
        :math:`\Gamma^T`, whose memory holds :math:`\Gamma` in row-major order.
    """
    assert sigma.flags.c_contiguous
    if lambd is not None:
        assert lambd.flags.f_contiguous
    else:
        lambd = cp.empty(sigma.shape, complex, order="F")
    blockspergrid = get_blockspergrid(sigma, threadsperblock)
    _get_lambda[blockspergrid, threadsperblock](sigma, lambd)
    return lambd


if __name__ == "__main__":
    z = complex(2.0)

    h = cp.random.random((10, 10)).astype(complex)
    s = cp.random.random((10, 10)).astype(complex)
    m = cp.empty((10, 10), complex, order="F")

    blockspergrid = get_blockspergrid(h, threadsperblock)

    geamT(z, h, s, m)
    expected = z * s.T - h.T
    assert cp.allclose(expected, m)

    geamC(z, h, s, m)
    expected = z * s.conj() - h.conj()
    assert cp.allclose(expected, m)

    sigma = cp.random.random((10, 10)).astype(complex)
    lambd = cp.empty((10, 10), complex, order="F")

    get_lambda(sigma, lambd)
    expected = 1.0j * (sigma - sigma.T.conj()).T
    assert cp.allclose(expected, lambd)
