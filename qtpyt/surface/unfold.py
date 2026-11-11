r"""Real-space supercell matrices from matrices at transverse k-points."""
from __future__ import annotations

from typing import Optional, Sequence

from numpy.typing import NDArray

import numpy as np

from ._unfold_numba import _bloch_unfold


def bloch_unfold(A: NDArray[np.complex128], kpts: NDArray[np.float64], N: Sequence[int], out: Optional[NDArray[np.complex128]] = None) -> NDArray[np.complex128]:
    r"""Real-space matrix of a supercell of ``N`` small cells, from the small cell's k-space matrices.

    A periodic supercell built from :math:`N_1 N_2` copies of a small cell
    has the same physics as the small cell sampled at :math:`N_1 N_2`
    k-points. This inverts the Bloch transform: block :math:`(I, J)` of the
    supercell matrix is

    .. math::

        A_{IJ} = \frac{1}{N_k}\sum_{\mathbf k} A(\mathbf k)\,
        e^{2\pi i\,\mathbf k\cdot(\mathbf R_J - \mathbf R_I)}.

    Parameters
    ----------
    A : ndarray of complex128, shape (n_k, n, m)
        Matrices of the small cell at each k-point.
    kpts : ndarray, shape (n_k, 3)
        k-points, in units of the reciprocal lattice vectors of the small
        cell.
    N : sequence of three int
        Repetitions of the small cell. At most two may exceed 1; the last
        of these runs fastest in the supercell orbital order.
    out : ndarray of complex128, optional
        Array for the result.

    Returns
    -------
    A_sc : ndarray, shape (N_1 N_2 n, N_1 N_2 m)

    Raises
    ------
    NotImplementedError
        If all three entries of ``N`` exceed 1.
    """

    assert A.dtype == complex, "Input arrays must be complex128!"

    i, j, k = N
    nr, nc = A.shape[1:]

    # NOTE that for 1D expansion only 2nd column of kpts are used
    if i == 1:
        if k == 1:  # (1D) y-expand
            m, n = 1, j
            kpts = kpts[:, [0, 1]]
        else:  # (1D) z- or (2D) yz-expand
            m, n = j, k
            kpts = kpts[:, [1, 2]]
    else:  # i>1
        if j == 1:
            if k == 1:  # (1D) x-expand
                m, n = 1, i
                kpts = kpts[:, [0, 0]]  # Hack to swap x-coord to last position.
            else:  # (2D) xz-expand
                m, n = i, k
                kpts = kpts[:, [0, 2]]
        else:
            if k == 1:  # xy-expand
                m, n = i, j
                kpts = kpts[:, [0, 1]]
            else:
                raise NotImplementedError("Cannot expand 3D supercells.")

    mxn = m * n
    if out is None:
        out = np.zeros((mxn, nr, mxn, nc), complex)
    else:
        assert out.dtype == complex, "Output matrix must be complex128!"
        if out.ndim != 4:
            out.shape = (mxn, nr, mxn, nc)

    _bloch_unfold(A, kpts, m, n, out)

    out.shape = (mxn * nr, mxn * nc)
    return out
