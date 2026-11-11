r"""Matrices of a finite stack of identical principal layers.

A perfect crystal along the transport direction is a repetition of one
principal layer. Its Hamiltonian is block tridiagonal and its Green's
function is block Toeplitz (each block depends only on the distance
between layers), so both follow from a few blocks.
"""
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray

import numpy as np
from qtpyt.surface.unfold import bloch_unfold


def tritoeplz_expand(N: int, A: NDArray, B: NDArray, C: Optional[NDArray] = None, out: Optional[NDArray] = None) -> NDArray:
    r"""Matrix of ``N`` identical principal layers coupled to their neighbours.

    Parameters
    ----------
    N : int
        Number of layers.
    A : ndarray, shape (n, n)
        Block of one layer, for example :math:`H_{ii}`.
    B : ndarray, shape (n, n)
        Coupling to the next layer, :math:`H_{i,i+1}`.
    C : ndarray, shape (n, n), optional
        Coupling to the previous layer. Defaults to :math:`B^\dagger`.
    out : ndarray, shape (N n, N n), optional
        Array for the result.

    Returns
    -------
    M : ndarray, shape (N n, N n)
        Block tridiagonal matrix with ``A`` on the diagonal, ``B`` above and
        ``C`` below.
    """
    # C = B.T.conj()
    #
    #   A B
    #   C A B
    #   0  C A B
    #        .  .  .
    n = A.shape[0]

    if C is None:
        C = B.T.conj()

    if out is None:
        out = np.zeros((N * n, N * n), dtype=A.dtype)

    out = out.reshape(N, n, N, n)
    for i in range(N - 1):
        out[i, :, i, :] = A
        out[i, :, i + 1, :] = B
        out[i + 1, :, i, :] = C
    out[N - 1, :, N - 1, :] = A

    return out.reshape(n * N, n * N)


def tritoeplz_expand_bounds(N: int, B: NDArray, id: str = "left", out: Optional[NDArray] = None) -> NDArray:
    """Coupling across the ends of a stack of ``N`` layers, as a full-size matrix.

    The matrix is zero except for one corner block ``B``: block
    ``(N-1, 0)`` for ``id='left'``, block ``(0, N-1)`` for ``id='right'``.

    Parameters
    ----------
    N : int
        Number of layers.
    B : ndarray, shape (n, n)
        Coupling block.
    id : {'left', 'right'}, default 'left'
        Which corner holds ``B``.
    out : ndarray, shape (N n, N n), optional
        Array for the result, overwritten.

    Returns
    -------
    M : ndarray, shape (N n, N n)
    """
    n = B.shape[0]

    if out is None:
        out = np.zeros((n * N, n * N), dtype=B.dtype)

    out[:] = 0.0
    out = out.reshape(N, n, N, n)
    if id == "left":
        out[-1, :, 0, :] = B
    else:
        out[0, :, -1, :] = B
    return out.reshape(n * N, n * N)


def seminf_expand(N: int, hs_ii: Tuple[NDArray, NDArray], hs_ij: Tuple[NDArray, NDArray], sigmaL: NDArray, sigmaR: NDArray, z: complex, hs_ji: Optional[Tuple[NDArray, NDArray]] = None, out: Optional[NDArray] = None) -> NDArray[np.complex128]:
    r"""Green's function of ``N`` consecutive layers of a crystal that is infinite along transport.

    The layers are attached to semi-infinite crystal on both sides through
    ``sigmaL`` and ``sigmaR``. The diagonal block is
    :math:`G_{00} = (zS - H - \Sigma_L - \Sigma_R)^{-1}`, the first row and
    column follow from
    :math:`G_{i0} = -X\,G_{i-1,0}`, :math:`G_{i,N-1} = -Y\,G_{i+1,N-1}` with
    :math:`X = (zS - H - \Sigma_R)^{-1}\tau_{ji}` and
    :math:`Y = (zS - H - \Sigma_L)^{-1}\tau_{ij}`, and the rest from
    translation invariance, :math:`G_{ij} = G_{i-j}`.

    Parameters
    ----------
    N : int
        Number of layers.
    hs_ii : tuple of ndarray, shape (n, n)
        ``(H_ii, S_ii)`` of one layer.
    hs_ij : tuple of ndarray, shape (n, n)
        ``(H, S)`` coupling to the next layer.
    sigmaL, sigmaR : ndarray, shape (n, n)
        Self-energies of the semi-infinite crystal on the left and on the
        right of one layer.
    z : complex
        Complex energy :math:`E + i\eta`.
    hs_ji : tuple of ndarray, optional
        Coupling to the previous layer. Defaults to the conjugate transpose
        of ``hs_ij``.
    out : ndarray, shape (N n, N n), optional
        Array for the result.

    Returns
    -------
    G : ndarray, shape (N n, N n)
    """
    #
    #   G00  G01  G02  G03 .
    #   G10  G00  G01  G02 .
    #   G20  G10  G00  G01 .
    #   G30  G20  G10  G00 .
    #    .    .    .    .  .
    #
    h_ii, s_ii = hs_ii
    h_ij, s_ij = hs_ij
    if hs_ji is None:
        hs_ji = h_ij.T.conj(), s_ij.T.conj()
    h_ji, s_ji = hs_ji

    n = h_ii.shape[0]

    if out is None:
        out = np.zeros((N * n, N * n), dtype=complex)

    out = out.reshape(N, n, N, n)

    m_ii = z * s_ii - h_ii
    m_ij = z * s_ij - h_ij
    m_ji = z * s_ji - h_ji

    G = np.linalg.inv(m_ii - sigmaL - sigmaR)
    Y = np.linalg.solve(m_ii - sigmaL, m_ij)
    X = np.linalg.solve(m_ii - sigmaR, m_ji)

    out[0, :, 0, :] = out[N - 1, :, N - 1, :] = G

    for i in range(1, N):
        out[i, :, 0, :] = -X.dot(out[i - 1, :, 0, :])

    for i in range(N - 2, -1, -1):
        out[i, :, N - 1, :] = -Y.dot(out[i + 1, :, N - 1, :])

    out = out.reshape(n * N, n * N)

    for i in range(1, N):
        out[i * n : (i + 1) * n, n : (i + 1) * n] = out[(i - 1) * n : i * n, : i * n]

    for i in range(N - 3, -1, -1):
        out[i * n : (i + 1) * n, (i + 1) * n : (N - 1) * n] = out[
            (i + 1) * n : (i + 2) * n, (i + 2) * n : N * n
        ]

    return out

