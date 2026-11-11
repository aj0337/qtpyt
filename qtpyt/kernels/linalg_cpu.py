r"""Linear solves for Green's-function products (CPU, Intel MKL).

Observables need products such as :math:`G\Gamma G^\dagger` with
:math:`G = M^{-1}`. These routines factorize :math:`M = zS - H - \Sigma`
once and apply :math:`M^{-1}` and :math:`(M^\dagger)^{-1}` by triangular
solves, without forming :math:`G`. The input arrays are overwritten. Requires ``libmkl_rt.so`` at import.
"""
from __future__ import annotations

from typing import Tuple

from numpy.typing import NDArray

import numpy as np
import ctypes

mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")

i4 = ctypes.c_int
f8 = ctypes.c_double
c16 = ctypes.c_longdouble


def dot(a: NDArray, b: NDArray, c: NDArray, alpha: float = 1.0, beta: float = 0.0) -> NDArray:
    r"""Real matrix product :math:`C \leftarrow \alpha AB + \beta C` (MKL ``dgemm``).

    Parameters
    ----------
    a : ndarray of float64, shape (m, k), C-ordered
    b : ndarray of float64, shape (k, n), C-ordered
    c : ndarray of float64, shape (m, n), C-ordered
        Overwritten with the result.
    alpha, beta : float
        Scaling factors.
    """

    Order = i4(101)
    TransA = i4(111)
    TransB = i4(111)

    a_ptr = np.ctypeslib.as_ctypes(a)
    b_ptr = np.ctypeslib.as_ctypes(b)
    c_ptr = np.ctypeslib.as_ctypes(c)

    m = i4(a.shape[0])
    n = i4(b.shape[1])
    k = i4(a.shape[1])

    lda = k
    ldb = n
    ldc = n

    mkl.cblas_dgemm(
        Order,
        TransA,
        TransB,
        m,
        n,
        k,
        f8(alpha),
        a_ptr,
        lda,
        b_ptr,
        ldb,
        f8(beta),
        c_ptr,
        ldc,
    )

    return c


def solve(a: NDArray, b: NDArray, overwrite_b: bool = False) -> NDArray:
    r"""Solution :math:`X = A^{-1}B` of :math:`AX = B`.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Matrix :math:`A`, for example :math:`G^{-1}`; overwritten by its LU factors.
    b : ndarray, shape (n, k)
        Right-hand side.
    overwrite_b : bool, default False
        Write the solution into ``b``.

    Returns
    -------
    x : ndarray, shape (n, k)
    """
    Order = i4(102) if a.flags.f_contiguous else i4(101)

    _b = b if overwrite_b else b.copy(order="A")

    a_ptr = a.ctypes.data_as(ctypes.POINTER(c16))
    b_ptr = _b.ctypes.data_as(ctypes.POINTER(c16))

    n = i4(a.shape[0])
    nrhs = i4(b.shape[1])
    ipiv = np.empty(a.shape[0], np.int32)
    ipiv_ptr = ipiv.ctypes.data_as(ctypes.POINTER(i4))

    lda = n
    ldb = n if b.flags.f_contiguous else nrhs

    mkl.lapacke_zgesv(Order, n, nrhs, a_ptr, lda, ipiv_ptr, b_ptr, ldb)

    return _b


def lrsolve(a: NDArray, b: NDArray) -> NDArray:
    r"""Product :math:`A^{-1} B (A^\dagger)^{-1}`, for example :math:`G\Gamma G^\dagger` from :math:`G^{-1}` and :math:`\Gamma`.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Matrix :math:`A`; overwritten.
    b : ndarray, shape (n, n)
        Matrix :math:`B`; overwritten.

    Returns
    -------
    c : ndarray, shape (n, n)
    """
    Order = i4(102) if a.flags.f_contiguous else i4(101)

    a_ptr = a.ctypes.data_as(ctypes.POINTER(c16))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(c16))

    m = i4(a.shape[0])
    n = i4(a.shape[1])
    lda = m if a.flags.f_contiguous else n

    ipiv = np.empty(a.shape[0], np.int32)
    ipiv_ptr = ipiv.ctypes.data_as(ctypes.POINTER(c16))

    mkl.lapacke_zgetrf(Order, m, n, a_ptr, lda, ipiv_ptr)

    trans = ctypes.c_char(b"N")
    nrhs = i4(b.shape[1])
    ldb = m if b.flags.f_contiguous else nrhs

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, b_ptr, ldb)

    np.conj(a, out=a)

    c = b.copy(order="C") if a.flags.f_contiguous else b.copy(order="F")
    c_ptr = c.ctypes.data_as(ctypes.POINTER(c16))

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, c_ptr, ldb)

    return c


def tsolve(a: NDArray, b: NDArray, c: NDArray) -> Tuple[NDArray, NDArray]:
    r"""Two solves with one factorization: :math:`A^{-1}B` and :math:`(A^\dagger)^{-1}C`.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Matrix :math:`A`; overwritten by its LU factors.
    b : ndarray, shape (n, k)
        Overwritten by :math:`A^{-1}B`.
    c : ndarray, shape (n, l)
        Overwritten by :math:`(A^\dagger)^{-1}C`.

    Returns
    -------
    x, y : ndarray
        :math:`A^{-1}B` and :math:`(A^\dagger)^{-1}C`.
    """
    Order = i4(102) if a.flags.f_contiguous else i4(101)

    a_ptr = a.ctypes.data_as(ctypes.POINTER(c16))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(c16))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(c16))

    m = i4(a.shape[0])
    n = i4(a.shape[1])
    lda = m if a.flags.f_contiguous else n

    ipiv = np.empty(a.shape[0], np.int32)
    ipiv_ptr = ipiv.ctypes.data_as(ctypes.POINTER(c16))

    mkl.lapacke_zgetrf(Order, m, n, a_ptr, lda, ipiv_ptr)

    trans = ctypes.c_char(b"N")
    nrhs = i4(b.shape[1])
    ldb = m if b.flags.f_contiguous else nrhs

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, b_ptr, ldb)

    trans = ctypes.c_char(b"C")
    nrhs = i4(c.shape[1])
    ldc = m if c.flags.f_contiguous else nrhs

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, c_ptr, ldc)

    return b, c
