r"""Block tridiagonal matrices.

A device built from principal layers that couple only to their nearest
neighbours has a Hamiltonian with non-zero blocks only on the diagonal and
next to it,

.. math::

    H = \begin{pmatrix}
    H_{00} & H_{01} &        &        \\
    H_{10} & H_{11} & H_{12} &        \\
           & H_{21} & H_{22} & \ddots \\
           &        & \ddots & \ddots
    \end{pmatrix}.

Storing only these blocks keeps memory linear in the device length
instead of quadratic, and lets the Green's function be computed block by
block (:mod:`qtpyt.block_tridiag.recursive`).
"""
from __future__ import annotations

from itertools import accumulate, product
from typing import List, Optional, Sequence, Tuple

from numpy.typing import NDArray

import numpy as np
from qtpyt import xp
from qtpyt.base._kernels import dotdiag, dottrace


def _boudscheck(i, j, N):
    if abs(i) > N:
        raise IndexError(f"Index {i} is out of bound for axis 0 with size {N}")
    if abs(j) > N:
        raise IndexError(f"Index {j} is out of bound for axis 1 with size {N}")
    if abs(j - i) > 1:
        raise IndexError(f"Index {i,j} is out of tridiagonal")


def _wraparound(i, j, N):
    return i % N, j % N


class _BTBuffer:
    def __init__(self, N):
        self.N = N
        self.m_qii = [None for _ in range(N)]
        self.m_qij = [None for _ in range(N - 1)]
        self.m_qji = [None for _ in range(N - 1)]

    def __getitem__(self, pos):
        i, j = pos
        _boudscheck(i, j, self.N)
        i, j = _wraparound(i, j, self.N)
        if i == j:
            return self.m_qii[i]
        if i > j:
            return self.m_qji[j]
        elif j > i:
            return self.m_qij[i]

    def __setitem__(self, pos, val):
        i, j = pos
        _boudscheck(i, j, self.N)
        i, j = _wraparound(i, j, self.N)
        if i == j:
            self.m_qii[i] = val
        if i > j:
            self.m_qji[j] = val
        elif j > i:
            self.m_qij[i] = val

    def __iter__(self):
        yield self.m_qii[0]
        for q in range(1, self.N):
            yield self.m_qij[q - 1]
            yield self.m_qji[q - 1]
            yield self.m_qii[q]

    def __len__(self):
        return self.N


def empty_buffer(N: int) -> _BTBuffer:
    """Container for the blocks of an ``N``-block tridiagonal matrix, all unset.

    Parameters
    ----------
    N : int
        Number of diagonal blocks.
    """
    return _BTBuffer(N)


class BTMatrix(_BTBuffer):
    """Block tridiagonal matrix stored as its diagonal and first off-diagonal blocks.

    Blocks are read and written as ``M[q, p]`` with ``|q - p| <= 1``.
    Element-wise ``+``, ``-`` and ``*`` with another :class:`BTMatrix` or a
    scalar act block by block.

    Parameters
    ----------
    m_qii : list of ndarray, length N
        Diagonal blocks :math:`M_{qq}`, square, of size :math:`n_q`.
    m_qij : list of ndarray, length N - 1
        Upper blocks :math:`M_{q,q+1}`, shape :math:`(n_q, n_{q+1})`.
    m_qji : list of ndarray, length N - 1
        Lower blocks :math:`M_{q+1,q}`, shape :math:`(n_{q+1}, n_q)`.

    Attributes
    ----------
    N : int
        Number of diagonal blocks.
    shape : tuple of int
        Shape ``(n_orb, n_orb)`` of the full matrix.
    """

    def __init__(self, m_qii: List[NDArray], m_qij: List[NDArray], m_qji: List[NDArray]) -> None:
        super().__init__(len(m_qii))
        self.m_qii = m_qii
        self.m_qij = m_qij
        self.m_qji = m_qji

        shapes = [m.shape[0] for m in self.m_qii]
        self.m = sum(shapes)
        self.shapes = np.fromiter(product(shapes, shapes), "i8,i8").reshape(
            self.N, self.N
        )
        self.shape = 2 * (sum(shapes),)
        self._nodes = np.fromiter(accumulate([0] + shapes), "i8")

    def todense(self) -> NDArray:
        """The full ``(n_orb, n_orb)`` matrix, zeros outside the tridiagonal blocks."""
        order = "C" if self.m_qii[0].flags.c_contiguous else "F"
        M = xp.zeros(2 * (self.m,), dtype=self.m_qii[0].dtype, order=order)
        M[self.ix_(0, 0)] = self.m_qii[0]
        for q in range(1, self.N):
            M[self.ix_(q - 1, q)] = self.m_qij[q - 1]
            M[self.ix_(q, q - 1)] = self.m_qji[q - 1]
            M[self.ix_(q, q)] = self.m_qii[q]
        return M

    def ix_(self, i: int, j: Optional[int] = None) -> Tuple[slice, slice]:
        """Slices that select block ``(i, j)`` in the full matrix.

        Parameters
        ----------
        i : int
            Row block.
        j : int, optional
            Column block, ``i`` if omitted.
        """
        if j is None:
            j = i
        il = self._nodes[i]
        jl = self._nodes[j]
        iu = il + self.shapes[i, j][0]
        ju = jl + self.shapes[i, j][1]
        return (slice(il, iu), slice(jl, ju))

    def __eq__(self, other, rtol=1e-05, atol=1e-08):
        allclose = lambda x, y: xp.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
        for a, b in zip(self, other):
            allclose(a, b)

    def diagonal(self) -> NDArray:
        """Diagonal of the full matrix, of length ``n_orb``."""
        return xp.concatenate([self.m_qii[q].diagonal() for q in range(self.N)])

    def trace(self) -> complex:
        """Trace of the full matrix."""
        return xp.sum([self.m_qii[q].trace() for q in range(self.N)])

    def dot(self, other: "BTMatrix") -> "BTMatrix":
        """Diagonal blocks of the product with another block tridiagonal matrix.

        Only the blocks :math:`(AB)_{qq}` are computed; the off-diagonal
        blocks of the result are ``None``. This is what traces and local
        densities need.

        Parameters
        ----------
        other : BTMatrix
            Right factor, with the same block sizes.
        """
        out = empty_buffer(self.N)
        out.m_qii[0] = xp.dot(self.m_qii[0], other.m_qii[0])
        for q in range(self.N - 1):
            out.m_qii[q + 1] = xp.dot(self.m_qii[q + 1], other.m_qii[q + 1])
            out.m_qii[q] += xp.dot(self.m_qij[q], other.m_qji[q])
            out.m_qii[q + 1] += xp.dot(self.m_qji[q], other.m_qij[q])
        return BTMatrix(out.m_qii, out.m_qij, out.m_qji)

    def copy(self) -> "BTMatrix":
        """Copy of the matrix with copied blocks."""
        return BTMatrix(
            [m_qii.copy() for m_qii in self.m_qii],
            [m_qij.copy() for m_qij in self.m_qij],
            [m_qji.copy() for m_qji in self.m_qji],
        )

    def rotate(self, rot: Sequence[Optional[NDArray]]) -> "BTMatrix":
        r"""Change of basis :math:`U^\dagger M U` with a block diagonal :math:`U`.

        Use it to rotate selected layers into another basis (for example
        localized orbitals) while keeping the block tridiagonal structure.

        Parameters
        ----------
        rot : sequence of ndarray or None, length N
            :math:`U_{qq}` for each block; ``None`` leaves the block
            unchanged.

        Returns
        -------
        M_rot : BTMatrix
        """
        # [UT_00*H_00*U_00  UT_00*H_01*U_11        0               0         ]
        # [                                                                  ]
        # [UT_11*H_10*U_00  UT_11*H_11*U_11  UT_11*H_12*U_22        0        ]
        # [                                                                  ]
        # [      0          UT_22*H_21*U_11  UT_22*H_22*U_22  UT_22*H_23*U_33]
        # [                                                                  ]
        # [      0               0           UT_33*H_32*U_22  UT_33*H_33*U_33]
        out = empty_buffer(self.N)

        def dots(a=None, b=None, c=None):
            # Helper function
            if a is None:
                if c is None:
                    return b.copy()
                return b.dot(c)
            if c is None:
                return a.dot(b)
            return a.dot(b).dot(c)

        q = 0
        uT1 = rot[q].T.conj() if rot[q] is not None else None
        for q in range(self.N - 1):
            uT2 = rot[q + 1].T.conj() if rot[q + 1] is not None else None
            out.m_qii[q] = dots(uT1, self.m_qii[q], rot[q])
            out.m_qij[q] = dots(uT1, self.m_qij[q], rot[q + 1])
            out.m_qji[q] = dots(uT2, self.m_qji[q], rot[q])
            uT1 = uT2
        q = self.N - 1
        out.m_qii[q] = dots(uT1, self.m_qii[q], rot[q])
        return BTMatrix(out.m_qii, out.m_qij, out.m_qji)

    def dotdiag(self, other: "BTMatrix") -> NDArray:
        """Diagonal of the product with another block tridiagonal matrix.

        Exact, because a diagonal element of the product involves only
        tridiagonal blocks. With ``other`` the overlap :math:`S` this gives
        the orbital-resolved :math:`(GS)_{ii}`.

        Parameters
        ----------
        other : BTMatrix
        """
        out = [
            xp.zeros(self.m_qii[q].shape[0], self.m_qii[q].dtype) for q in range(self.N)
        ]
        dotdiag(self.m_qii[0], other.m_qii[0], out[0])
        for q in range(self.N - 1):
            dotdiag(self.m_qii[q + 1], other.m_qii[q + 1], out[q + 1])
            dotdiag(self.m_qij[q], other.m_qji[q], out[q])
            dotdiag(self.m_qji[q], other.m_qij[q], out[q + 1])
        return xp.concatenate(out)

    def dottrace(self, other: "BTMatrix") -> complex:
        r"""Trace of the product :math:`\operatorname{Tr}[AB]` with another block tridiagonal matrix.

        Parameters
        ----------
        other : BTMatrix
        """
        out = complex(0.0)
        out = dottrace(self.m_qii[0], other.m_qii[0], out)
        for q in range(self.N - 1):
            out = dottrace(self.m_qii[q + 1], other.m_qii[q + 1], out)
            out = dottrace(self.m_qij[q], other.m_qji[q], out)
            out = dottrace(self.m_qji[q], other.m_qij[q], out)
        return out

    _specials = [
        "__add__",
        "__sub__",
        "__iadd__",
        "__isub__",
        "__mul__",
        "__imul__",
        "__radd__",
        "__rsub__",
        "__rmul__",
    ]

    @classmethod
    def _create(cls):
        """
        TODO :: Add option to add diagonal array or constant.
        """
        ##https://code.activestate.com/recipes/496741-object-proxying/

        def make_method(name):
            def method(self, other, **kwargs):
                op = getattr(xp.ndarray, name)
                out = empty_buffer(self.N)
                if isinstance(other, BTMatrix):
                    out.m_qii[0] = op(self.m_qii[0], other.m_qii[0])
                    for q in range(1, self.N):
                        out.m_qii[q] = op(self.m_qii[q], other.m_qii[q])
                        out.m_qij[q - 1] = op(self.m_qij[q - 1], other.m_qij[q - 1])
                        out.m_qji[q - 1] = op(self.m_qji[q - 1], other.m_qji[q - 1])
                else:
                    out.m_qii[0] = op(self.m_qii[0], other)
                    for q in range(1, self.N):
                        out.m_qii[q] = op(self.m_qii[q], other)
                        out.m_qij[q - 1] = op(self.m_qij[q - 1], other)
                        out.m_qji[q - 1] = op(self.m_qji[q - 1], other)
                return BTMatrix(out.m_qii, out.m_qij, out.m_qji)

            return method

        namespace = {}
        for name in cls._specials:
            namespace[name] = make_method(name)
        return type("%s(%s)" % (cls.__name__, xp.ndarray.__name__), (cls,), namespace)

    def __new__(cls, *args, **kwargs):
        theclass = cls._create()
        ins = object.__new__(theclass)
        theclass.__init__(ins, *args, **kwargs)
        return ins
