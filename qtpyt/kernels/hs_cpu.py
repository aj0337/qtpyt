r"""Element-wise kernels that build :math:`zS - H` and :math:`\Gamma` in column-major layout (CPU).

The inverse Green's function is assembled from blocks :math:`zS - H` and
the broadenings :math:`\Gamma = i(\Sigma - \Sigma^\dagger)`. These kernels
write them directly into Fortran-ordered arrays, the layout the LAPACK
solvers of :mod:`qtpyt.kernels` read, avoiding a separate transpose.
"""
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray

import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True, fastmath=True)
def _geamT(z, h, s, m):
    for i in prange(h.shape[0]):
        for j in range(h.shape[1]):
            m[j, i] = z * s[i, j] - h[i, j]


@numba.njit(parallel=True, fastmath=True)
def _geamC(z, h, s, m):
    for i in prange(h.shape[0]):
        for j in range(h.shape[1]):
            m[i, j] = z * s[i, j].conjugate() - h[i, j].conjugate()


@numba.njit(parallel=False, fastmath=True)
def _get_lambda(sigma, lambd):
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            lambd[j, i] = 1.0j * (sigma[i, j] - sigma[j, i].conjugate())


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
        m = np.empty(h.shape, complex, order="F")
    _geamT(z, h, s, m)
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
        m = np.empty(h.shape, complex, order="F")
    _geamC(z, h, s, m)
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
        lambd = np.empty(sigma.shape, complex, order="F")
    _get_lambda(sigma, lambd)
    return lambd


if __name__ == "__main__":
    z = complex(2.0)

    h = np.random.random((900, 900)).astype(complex)
    s = np.random.random((900, 900)).astype(complex)
    m = np.empty((900, 900), complex, order="F")

    geamT(z, h, s, m)
    expected = z * s.T - h.T
    assert np.allclose(expected, m)

    geamC(z, h, s, m)
    expected = z * s.conj() - h.conj()
    assert np.allclose(expected, m)

    sigma = np.random.random((900, 900)).astype(complex)
    lambd = np.empty((900, 900), complex, order="F")

    get_lambda(sigma, lambd)
    expected = 1.0j * (sigma - sigma.T.conj()).T
    assert np.allclose(expected, lambd)
