r"""Restricting Green's functions to a subspace of orbitals, and embedding back.

Many analyses focus on a few orbitals: the frontier orbitals of a
molecule, the :math:`d` shell of an impurity, a set of localized orbitals.
In a non-orthogonal basis, picking the corresponding rows and columns of
:math:`G` is not a projection, because the subspace orbitals overlap with
the rest. The functions here project with the overlap included, assuming
the subspace orbitals are orthogonal to all other orbitals (as the
localized orbitals of :mod:`qtpyt.lo` are constructed to be). Both dense
matrices and block tridiagonal matrices
(:class:`~qtpyt.block_tridiag.btmatrix.BTMatrix`) are handled.
"""
from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Sequence, Union

from numpy.typing import NDArray

import numpy as np

from qtpyt import xp
from qtpyt.base.greenfunction import BaseGreenFunction, GreenFunction
from qtpyt.block_tridiag.btmatrix import BTMatrix, _BTBuffer


class ProjectedGreenFunction(BaseGreenFunction):
    r"""Green's function of the full system projected onto a subspace of orbitals.

    The subspace keeps all the effects of the rest of the system (leads,
    the remaining orbitals of the device) through the parent Green's
    function. Pair it with :class:`~qtpyt.hybridization.Hybridization`
    to obtain the hybridization function of the subspace.

    Parameters
    ----------
    gf : GreenFunction
        Green's function of the full system, dense or block tridiagonal.
    indices : sequence of int, or block specification
        Orbitals of the subspace, see :class:`Projector`.

    Notes
    -----
    :attr:`H` and :attr:`S` are the subspace blocks of the parent's
    Hamiltonian and overlap. :meth:`retarded` returns :func:`project`
    applied to the parent :math:`G^r(E)`. Setting ``eta``, ``kt`` or
    ``mu`` on this object also sets it on the parent.
    """

    def __init__(self, gf: GreenFunction, indices: Any) -> None:
        self.parent = gf
        self.projector = Projector(gf.S, indices)
        H = self.projector.cut(gf.H)
        super().__init__(H, self.projector.S, gf.eta, gf.mu, gf.kt)
        self.energy = None

    def __setattr__(self, name: str, value: Any) -> Any:
        if name in ["eta", "kt", "mu"]:
            setattr(self.parent, name, value)
        super().__setattr__(name, value)

    def retarded(self, energy: float) -> NDArray:
        r"""Projected retarded Green's function :math:`G^r_{qq}(E)` of the subspace.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        if self.energy != energy:
            self.energy = energy
            Gr = self.parent.retarded(energy)
            self.Gr = self.projector.project(Gr)
        return self.Gr


class CutGreenFunction(ProjectedGreenFunction):
    r"""Subspace block of the Green's function, without overlap projection.

    :meth:`retarded` returns the rows and columns of the parent
    :math:`G^r(E)` that belong to ``indices``. This equals
    :class:`ProjectedGreenFunction` in an orthonormal basis.

    Parameters
    ----------
    gf : GreenFunction
        Green's function of the full system.
    indices : sequence of int, or block specification
        Orbitals of the subspace, see :class:`Projector`.
    """

    def __init__(self, gf: GreenFunction, indices: Any) -> None:
        super().__init__(gf, indices)
        self.projector.project = self.projector.cut


class Projector:
    """Projector onto a subspace orthogonal to the rest; returns a dense or block tridiagonal projector.

    Parameters
    ----------
    S : ndarray or BTMatrix
        Overlap matrix of the full system. A dense array gives a
        :class:`BaseProjector`, a block tridiagonal one a
        :class:`BTProjector`.
    indices : sequence of int, or int, or tuple
        Orbitals of the subspace. For a dense ``S``, indices into the full
        matrix. For a block tridiagonal ``S``, one of: a block index ``q``
        (the whole block); a pair ``(q, idx)`` with ``idx`` relative to the
        first orbital of block ``q``; or indices into the full matrix, all
        within one block.
    """

    def __new__(self, S: Any, indices: Any) -> Union["BaseProjector", "BTProjector"]:
        if isinstance(S, xp.ndarray):
            return BaseProjector(S, indices)

        else:
            # Preprocess input
            if isinstance(indices, int):
                q = indices
                indx = None
            elif len(indices) == 2 and hasattr(indices[1], "__len__"):
                q = indices[0]
                indx = indices[1]
            else:
                nodes = S._nodes
                # basis function at node belongs to right block
                q = np.searchsorted(nodes[1:], indices[0], side="right")
                indx = indices - nodes[q]
            return BTProjector(S, q, indx)


class BaseProjector:
    """Projector onto a subspace of a system described by dense matrices.

    Parameters
    ----------
    S : ndarray, shape (n_orb, n_orb)
        Overlap matrix of the full system.
    indices : sequence of int
        Orbitals of the subspace.

    Attributes
    ----------
    S : ndarray
        Overlap block :math:`S_{qq}` of the subspace.
    """

    def __init__(self, S: NDArray, indices: Sequence[int]) -> None:
        self._S_pp = S
        self.indices = indices
        self.S = self.cut(self._S_pp)

    def project(self, X: NDArray) -> NDArray:
        """Project a full-space matrix onto the subspace, see :func:`project`.

        Parameters
        ----------
        X : ndarray, shape (n_orb, n_orb)
        """
        return project(X, self._S_pp, self.indices)

    def expand(self, X: NDArray) -> NDArray:
        """Embed a subspace matrix into the full space.

        Parameters
        ----------
        X : ndarray, shape (n_q, n_q)
        """
        return expand(X, self._S_pp, self.indices)

    def cut(self, X: NDArray) -> NDArray:
        """Rows and columns of ``X`` that belong to the subspace.

        Parameters
        ----------
        X : ndarray, shape (n_orb, n_orb)
        """
        return X[np.ix_(self.indices, self.indices)]


class BTProjector:
    """Projector onto a subspace inside one block of a block tridiagonal system.

    Parameters
    ----------
    S : BTMatrix
        Overlap matrix of the full system.
    q : int
        Block that contains the subspace.
    indices : sequence of int, optional
        Orbitals of the subspace, relative to the first orbital of block
        ``q``. The whole block if omitted.
    """

    def __init__(self, S: Any, q: int, indices: Optional[Sequence[int]] = None) -> None:
        self._S_pp = S
        self.q = q
        self.indices = indices
        self.S = self.cut(self._S_pp)

    def project(self, X: Any) -> NDArray:
        """Project a block tridiagonal matrix onto the subspace, see :func:`project`.

        Parameters
        ----------
        X : BTMatrix
        """
        return project(X, self._S_pp, self.q, self.indices)

    def expand(self, X: NDArray) -> None:
        """Not implemented for block tridiagonal systems; returns ``None``.

        Parameters
        ----------
        X : ndarray
        """
        pass
        # return BTMatrix(...)

    def cut(self, X: Any) -> NDArray:
        """Subspace block of a block tridiagonal matrix.

        Parameters
        ----------
        X : BTMatrix
        """
        X = X[self.q, self.q]
        if self.indices is not None:
            X = X[np.ix_(self.indices, self.indices)]
        return X


def rotate_matrix(M: NDArray, U: NDArray) -> NDArray:
    r"""Basis change :math:`U^\dagger M U`.

    Parameters
    ----------
    M : ndarray, shape (n, n)
    U : ndarray, shape (n, m)
    """
    return U.T.conj().dot(M.dot(U))


@singledispatch
def project(A: Any, S: Any, q: Any) -> NDArray:
    r"""Project a full-space matrix such as :math:`G` onto a subspace of orbitals.

    Parameters
    ----------
    A : ndarray or BTMatrix
        Matrix in the full basis, for example :math:`G^r(E)`.
    S : ndarray or BTMatrix
        Overlap matrix of the full basis, of the same kind as ``A``.
    q : sequence of int, or int
        For dense matrices, orbitals of the subspace. For block tridiagonal
        matrices, the block index; an optional fourth argument ``indices``
        then selects orbitals within that block.

    Returns
    -------
    A_qq : ndarray, shape (n_q, n_q)

    Notes
    -----
    .. math::

        A_{qq} = S_{qq}^{-1}\,(S A S)_{qq}\,S_{qq}^{-1}.

    :math:`(SAS)_{qq}` holds the matrix elements
    :math:`\langle\phi_a|\hat A|\phi_b\rangle` of the operator between
    subspace orbitals, and the inverse overlaps convert them back to the
    form :math:`G` has in its own basis. For a block tridiagonal matrix the
    product :math:`SAS` needs only block :math:`q` and its two neighbours.
    """
    return rotate_matrix(rotate_matrix(A, S[:, q]), xp.linalg.inv(S[xp.ix_(q, q)]))


@project.register(BTMatrix)
def _(A, S, q, indices=None):
    """Project A onto subspace q orthogonal to the rest.
    Args : 
        A : (BTMatrix)
        S : (BTMatrix) overlap matrix
        q : index of subspace's block
        indices : (optional) subspace indices within block q.
    """
    # [(D_21*H_11 + D_22*H_21)*S_12
    # + (D_22*H_23 + D_23*H_33)*S_32
    # + (D_21*H_12 + D_22*H_22 + D_23*H_32)*S_22]
    D = S[q, q].T.dot(A[q, q])
    B = xp.zeros_like(A[q, q])
    if (q - 1) >= 0:
        D += S[q, q - 1].dot(A[q - 1, q])
        T = S[q, q].dot(A[q, q - 1])
        T += S[q, q - 1].dot(A[q - 1, q - 1])
        B += T.dot(S[q - 1, q])
    if (q + 1) < A.N:
        D += S[q, q + 1].dot(A[q + 1, q])
        T = S[q, q].dot(A[q, q + 1])
        T += S[q, q + 1].dot(A[q + 1, q + 1])
        B += T.dot(S[q + 1, q])
    D.dot(S[q, q], out=D)
    B += D
    B = rotate_matrix(B, xp.linalg.inv(S[q, q]))
    if indices is not None:
        return project(B, S[q, q], indices)
    return B


class ExpandSelfEnergy:
    r"""Self-energy defined on a subspace, embedded into the full orbital space.

    Use it to add a self-energy computed for a few orbitals (for example
    from an impurity solver) to the Green's function of the full device.

    Parameters
    ----------
    S : ndarray, shape (n_orb, n_orb)
        Overlap matrix of the full system.
    selfenergy : object
        Self-energy on the subspace, providing ``retarded(energy)``.
    indices : sequence of int
        Orbitals of the subspace.
    """

    def __init__(self, S: NDArray, selfenergy: Any, indices: Sequence[int]) -> None:
        self.S = S
        self.selfenergy = selfenergy
        self.indices = indices

    def expand(self, X: NDArray) -> NDArray:
        """Embed a subspace matrix into the full space, see :func:`expand`.

        Parameters
        ----------
        X : ndarray, shape (n_q, n_q)
        """
        return expand(self.S, X, self.indices)

    def retarded(self, energy: float) -> NDArray:
        r"""Retarded self-energy :math:`\Sigma(E)` on the full orbital space.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        return self.expand(self.selfenergy.retarded(energy))


@singledispatch
def expand(S: Any, A: NDArray, q: Any) -> Any:
    r"""Embed a subspace matrix into the full orbital space.

    Parameters
    ----------
    S : ndarray or BTMatrix
        Overlap matrix of the full basis.
    A : ndarray, shape (n_q, n_q)
        Matrix on the subspace, for example a self-energy.
    q : sequence of int, or int
        For dense matrices, orbitals of the subspace. For block tridiagonal
        matrices, the block index; an optional fourth argument ``indices``
        selects orbitals within that block.

    Returns
    -------
    A_full : ndarray or block tridiagonal buffer
        Matrix on the full space. For block tridiagonal ``S`` only block
        :math:`q` and its neighbours are non-zero.

    Notes
    -----
    .. math::

        A_\mathrm{full} = S_{:q}\,S_{qq}\,A\,S_{qq}\,S_{q:},

    which for a subspace with :math:`S_{qq} = 1` gives the matrix elements
    :math:`\langle\phi_\mu|\hat A|\phi_\nu\rangle` in the full basis.
    """
    return rotate_matrix(rotate_matrix(A, S[xp.ix_(q, q)]), S[q, :])


@expand.register(BTMatrix)
def _(S, A, q, indices=None):
    # [0,              0,              0,              0]
    # [0, G_22*S_12*S_21, G_22*S_12*S_22, G_22*S_12*S_23]
    # [0, G_22*S_21*S_22,   G_22*S_22**2, G_22*S_22*S_23]
    # [0, G_22*S_21*S_32, G_22*S_22*S_32, G_22*S_23*S_32]
    N = S.N
    B = _BTBuffer(N)
    if indices is not None:
        A = expand(A, S[q, q], indices)
    B[q, q] = rotate_matrix(A, S[q, q])
    if q > 0:
        B[q - 1, q] = S[q - 1, q].dot(A).dot(S[q, q])
        B[q, q - 1] = S[q, q].dot(A).dot(S[q, q - 1])
        B[q - 1, q - 1] = S[q - 1, q].dot(A).dot(S[q, q - 1])
    if (q + 1) < N:
        B[q + 1, q] = S[q + 1, q].dot(A).dot(S[q, q])
        B[q, q + 1] = S[q, q].dot(A).dot(S[q, q + 1])
        B[q + 1, q + 1] = S[q + 1, q].dot(A).dot(S[q, q + 1])
    return B  # BTMatrix(B.m_qii, B.m_qij, B.m_qji)
