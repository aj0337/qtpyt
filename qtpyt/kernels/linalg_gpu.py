r"""Linear solves for Green's-function products (GPU, cuSOLVER).

Observables need products such as :math:`G\Gamma G^\dagger` with
:math:`G = M^{-1}`. These routines factorize :math:`M = zS - H - \Sigma`
once and apply :math:`M^{-1}` and :math:`(M^\dagger)^{-1}` by triangular
solves, without forming :math:`G`. The input arrays are overwritten. All arrays must be Fortran-ordered.
"""
from __future__ import annotations

from typing import Tuple

from numpy.typing import NDArray

import numpy as np
import cupy as cp

device = cp.cuda.device

cublas = device.cublas
cusolver = device.cusolver

cublas_handle = device.get_cublas_handle()
cusolver_handle = device.get_cusolver_handle()


def _assert_farray(*arrays):
    for array in arrays:
        assert array.flags.f_contiguous


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
        With ``True`` the solution is written into a copy of ``b``.

    Returns
    -------
    x : ndarray, shape (n, k)
    """
    _assert_farray(a, b)
    if overwrite_b:
        b = b.copy("A")

    getrf = cusolver.zgetrf
    getrf_buffer = cusolver.zgetrf_bufferSize
    getrs = cusolver.zgetrs
    trans = cublas.CUBLAS_OP_N

    m, n = a.shape
    nrhs = b.shape[1]
    lda = m
    ldb = m

    dev_info = cp.empty(1, dtype=np.int32)
    ipiv = cp.empty(m, np.int32)

    buffersize = getrf_buffer(cusolver_handle, m, n, a.data.ptr, lda)
    workspace = cp.empty(buffersize, dtype=a.dtype)

    getrf(
        cusolver_handle,
        m,
        n,
        a.data.ptr,
        lda,
        workspace.data.ptr,
        ipiv.data.ptr,
        dev_info.data.ptr,
    )
    del workspace

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        b.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    return b


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
    _assert_farray(a, b)

    getrf = cusolver.zgetrf
    getrf_buffer = cusolver.zgetrf_bufferSize
    getrs = cusolver.zgetrs
    trans = cublas.CUBLAS_OP_N

    m, n = a.shape
    nrhs = b.shape[1]
    lda = m
    ldb = m

    dev_info = cp.empty(1, dtype=np.int32)
    ipiv = cp.empty(m, np.int32)

    buffersize = getrf_buffer(cusolver_handle, m, n, a.data.ptr, lda)
    workspace = cp.empty(buffersize, dtype=a.dtype)

    getrf(
        cusolver_handle,
        m,
        n,
        a.data.ptr,
        lda,
        workspace.data.ptr,
        ipiv.data.ptr,
        dev_info.data.ptr,
    )
    del workspace

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        b.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    cp.conj(a, out=a)

    c = b.copy(order="C")

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        c.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

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
    _assert_farray(a, b, c)

    getrf = cusolver.zgetrf
    getrf_buffer = cusolver.zgetrf_bufferSize
    getrs = cusolver.zgetrs
    trans = cublas.CUBLAS_OP_N

    m, n = a.shape
    nrhs = b.shape[1]
    lda = m
    ldb = m

    dev_info = cp.empty(1, dtype=np.int32)
    ipiv = cp.empty(m, np.int32)

    buffersize = getrf_buffer(cusolver_handle, m, n, a.data.ptr, lda)
    workspace = cp.empty(buffersize, dtype=a.dtype)

    getrf(
        cusolver_handle,
        m,
        n,
        a.data.ptr,
        lda,
        workspace.data.ptr,
        ipiv.data.ptr,
        dev_info.data.ptr,
    )
    del workspace

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        b.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    trans = cublas.CUBLAS_OP_C

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        c.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    return b, c
