r"""Inverse of a 2×2 block matrix by the Schur complement."""
from __future__ import annotations

from typing import List, Sequence

from numpy.typing import NDArray

import numpy as np
from numpy.core.defchararray import array
from numpy.core.shape_base import block


def block_inv(M: Sequence[Sequence[NDArray]]) -> List[List[NDArray]]:
    r"""Blocks of the inverse of :math:`\begin{pmatrix} A & B \\ C & D \end{pmatrix}`.

    With the Schur complement :math:`\mathcal S = D - C A^{-1} B`,

    .. math::

        M^{-1} = \begin{pmatrix}
        A^{-1} + A^{-1} B \mathcal S^{-1} C A^{-1} & -A^{-1} B \mathcal S^{-1} \\
        -\mathcal S^{-1} C A^{-1} & \mathcal S^{-1}
        \end{pmatrix}.

    Applied to :math:`G^{-1}`, the lower-right block is the Green's function
    of the second region dressed by the first.

    Parameters
    ----------
    M : 2×2 nested sequence of ndarray
        Blocks ``[[A, B], [C, D]]``, with ``A`` and ``D`` square.

    Returns
    -------
    Minv : 2×2 nested list of ndarray
    """
    Minv = [[None, None], [None, None]]
    A = M[0][0]
    B = M[0][1]
    C = M[1][0]
    D = M[1][1]
    Ainv = np.linalg.inv(A)
    S = D - C.dot(Ainv).dot(B)
    Sinv = np.linalg.inv(S)
    Minv[1][1] = Sinv
    Ainv_dot_B = Ainv.dot(B)
    C_dot_Ainv = C.dot(Ainv)
    Minv[0][1] = -Ainv_dot_B.dot(Sinv)
    Minv[1][0] = -Sinv.dot(C_dot_Ainv)
    Minv[0][0] = Ainv
    Minv[0][0] += Ainv_dot_B.dot(Sinv).dot(C_dot_Ainv)
    return Minv


def test_block_inv() -> None:
    """Check :func:`block_inv` against a dense inverse of a random matrix."""
    mat = np.random.random((4, 4))
    M = [[mat[:2, :2], mat[:2, 2:]], [mat[2:, :2], mat[2:, 2:]]]
    computed = np.block(block_inv(M))
    expected = np.linalg.inv(mat)
    np.testing.assert_allclose(expected, computed)


if __name__ == "__main__":
    test_block_inv()
