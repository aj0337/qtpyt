r"""Local orbitals: basis changes that isolate a subspace of the device.

Interacting or chemically interesting orbitals (the :math:`d` shell of a
transition-metal atom, the frontier orbitals of a molecule) are rarely
basis functions of the DFT calculation. The tools here build them: by
diagonalizing the Hamiltonian on a subset of orbitals (local or molecular
orbitals), by Löwdin orthogonalization, and by making the rest of the basis
orthogonal to a chosen active subspace :math:`m`, so that the subspace can
be treated separately (for example as an impurity) and projected with
:mod:`qtpyt.projector`.

All rotations act as :math:`H \to U^\dagger H U`, :math:`S \to U^\dagger S U`.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from numpy.typing import NDArray

import numpy as np
from math import sqrt
from scipy import linalg as la
from scipy.linalg import block_diag


dagger = lambda mat: np.conj(mat.T)


def rotate_matrix(h: NDArray, u: NDArray) -> NDArray:
    r"""Basis change :math:`U^\dagger H U`.

    Parameters
    ----------
    h : ndarray, shape (n, n)
    u : ndarray, shape (n, m)
    """
    return np.dot(u.T.conj(), np.dot(h, u))


def get_subspace(matrix: NDArray, index: Sequence[int]) -> NDArray:
    """Block of a square matrix on the orbitals listed in ``index``.

    Parameters
    ----------
    matrix : ndarray, shape (n, n)
    index : sequence of int
    """
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    return matrix.take(index, 0).take(index, 1)


def normalize(matrix: NDArray, s: Optional[NDArray] = None) -> None:
    r"""Normalize each column :math:`c` so that :math:`c^\dagger S c = 1`, in place.

    Parameters
    ----------
    matrix : ndarray, shape (n, k)
        Orbital coefficients, one orbital per column.
    s : ndarray, shape (n, n), optional
        Overlap matrix; the Euclidean norm is used if omitted.
    """
    for col in matrix.T:
        if s is None:
            col /= np.linalg.norm(col)
        else:
            col /= np.sqrt(np.dot(col.conj(), np.dot(s, col)))


from scipy.linalg import eigh


def get_orthonormal_subspace(h: NDArray, s: NDArray, index_j: Optional[Sequence[int]] = None) -> Tuple[NDArray, NDArray[np.float64]]:
    r"""Eigenstates of the Hamiltonian restricted to a subset of orbitals.

    Solves :math:`H_{jj} v = \varepsilon S_{jj} v`. The eigenvectors are
    :math:`S_{jj}`-orthonormal.

    Parameters
    ----------
    h, s : ndarray, shape (n, n)
        Hamiltonian and overlap.
    index_j : sequence of int, optional
        Orbitals of the subset; all if omitted.

    Returns
    -------
    v : ndarray, shape (n_j, n_j)
        Eigenvectors as columns.
    e : ndarray, shape (n_j,)
        Eigenenergies, increasing.
    """
    if index_j is not None:
        h_jj = get_subspace(h, index_j)
        s_jj = get_subspace(s, index_j)
    else:
        h_jj = h
        s_jj = s
    # from IPython import embed; embed()
    w, v = eigh(h_jj, s_jj)
    return v, w
    # e_j, v_jj = np.linalg.eig(np.linalg.solve(s_jj, h_jj))
    # normalize(v_jj, s_jj)  # normalize: <v_j|s|v_j> = 1
    # permute_list = np.argsort(e_j.real)
    # e_j = np.take(e_j, permute_list)
    # v_jj = np.take(v_jj, permute_list, axis=1)
    # return v_jj, e_j


def subdiagonalize(h: NDArray, s: NDArray, index_j: Sequence[int]) -> Tuple[NDArray, NDArray[np.float64]]:
    """Rotation that diagonalizes the Hamiltonian within a subset of orbitals.

    Orbitals outside the subset are left unchanged. Applied to the orbitals
    of a molecule, the rotated basis holds its molecular orbitals, still
    coupled to the rest of the device.

    Parameters
    ----------
    h, s : ndarray, shape (n, n)
        Hamiltonian and overlap.
    index_j : sequence of int
        Orbitals of the subset.

    Returns
    -------
    U : ndarray, shape (n, n)
        Rotation, the identity outside the subset.
    e : ndarray
        Energies of the new orbitals, see :func:`get_orthonormal_subspace`.
    """

    v_jj, e_j = get_orthonormal_subspace(h, s, index_j)
    U = np.identity(h.shape[0], h.dtype)
    U[np.ix_(index_j, index_j)] = v_jj[:]
    return U, e_j


def subdiagonalize_atoms(basis: Any, h: NDArray, s: NDArray, a: Optional[Sequence[int]] = None) -> Tuple[NDArray, List[NDArray[np.float64]]]:
    """Rotation that diagonalizes the Hamiltonian on each selected atom separately.

    The new orbitals of an atom are the eigenstates of its on-site block,
    local orbitals with well-defined on-site energies (for example a
    crystal-field-split :math:`d` shell).

    Parameters
    ----------
    basis : Basis
        Orbital basis.
    h, s : ndarray, shape (n, n)
        Hamiltonian and overlap.
    a : int or sequence of int, optional
        Atoms to treat; all if omitted. Other atoms are left unchanged.

    Returns
    -------
    U : ndarray, shape (n, n)
        Block diagonal rotation.
    e_aj : list of ndarray
        On-site energies of the new orbitals of each treated atom.
    """

    if a is None:
        a = range(len(basis.atoms))
    if isinstance(a, int):
        a = [a]
    U_mm = []
    e_aj = []
    na_tot = len(basis.atoms)
    for a0 in range(na_tot):
        bfs = basis.get_indices(a0)
        if a0 in a:
            v_jj, e_j = get_orthonormal_subspace(h, s, bfs)
            U_mm.append(v_jj)
            e_aj.append(e_j)
        else:
            U_mm.append(np.eye(len(bfs)))
    U_mm = block_diag(*U_mm)
    return U_mm, e_aj


def cutcoupling(h: NDArray, s: NDArray, index_n: Sequence[int]) -> None:
    """Decouple orbitals from the rest of the system, keeping their on-site energies.

    All Hamiltonian and overlap elements of the orbitals in ``index_n`` are
    set to zero except :math:`H_{ii}` and :math:`S_{ii} = 1`. Modified in place.

    Parameters
    ----------
    h, s : ndarray, shape (n, n)
    index_n : sequence of int
    """
    for i in index_n:
        s[:, i] = 0.0
        s[i, :] = 0.0
        s[i, i] = 1.0
        Ei = h[i, i]
        h[:, i] = 0.0
        h[i, :] = 0.0
        h[i, i] = Ei


def lowdin_rotation(h: NDArray, s: NDArray, index_j: Optional[Sequence[int]] = None) -> NDArray:
    r"""Symmetric (Löwdin) orthogonalization :math:`U = S_{jj}^{-1/2}` of a subset of orbitals.

    Of all orthonormal sets, the Löwdin orbitals are the closest to the
    original orbitals, so they keep their atomic character.

    Parameters
    ----------
    h : ndarray, shape (n, n)
        Hamiltonian, used only for its dtype.
    s : ndarray, shape (n, n)
        Overlap.
    index_j : sequence of int, optional
        Orbitals to orthogonalize; all if omitted.

    Returns
    -------
    U : ndarray, shape (n, n)
        Rotation, the identity outside the subset.
    """
    s_jj = get_subspace(s, index_j) if index_j is not None else s
    eig, U_jj = np.linalg.eigh(s_jj)
    eig = np.abs(eig)
    U_jj = np.dot(U_jj / np.sqrt(eig), dagger(U_jj))
    if index_j is not None:
        U = np.identity(s.shape[0], h.dtype)
        U[np.ix_(index_j, index_j)] = U_jj[:]
        U_jj = U
    return U_jj


def get_U1(bfs_m: Any, bfs_i: Any, c_MM: Optional[NDArray] = None) -> Tuple[NDArray, Any, Any]:
    """Reordering that puts the active orbitals ``bfs_m`` last.

    Parameters
    ----------
    bfs_m : ndarray of int
        Active orbitals. Overwritten with their new positions.
    bfs_i : ndarray of int
        Remaining orbitals. Overwritten with their new positions.
    c_MM : ndarray, optional
        A rotation whose columns are reordered instead of the identity.

    Returns
    -------
    U1 : ndarray
    bfs_m, bfs_i : ndarray of int
        New positions: ``bfs_i`` first, then ``bfs_m``.
    """
    nbf = len(bfs_m) + len(bfs_i)
    if c_MM is None:
        c_MM = np.eye(nbf)
    U1 = np.take(c_MM, np.concatenate([bfs_i, bfs_m]), axis=1)
    """If apply is True permute inplace bfs_m bfs_i"""
    bfs_m[:] = range(nbf - len(bfs_m), nbf)
    bfs_i[:] = range(nbf - len(bfs_m))
    return U1, bfs_m, bfs_i


def get_U2(s1: NDArray, bfs_m: Sequence[int], bfs_i: Sequence[int]) -> NDArray:
    r"""Rotation that makes the remaining orbitals orthogonal to the active ones.

    Each remaining orbital loses its component along the active subspace,
    :math:`|i'\rangle = |i\rangle - \sum_{m,m'} |m\rangle (S_{mm}^{-1})_{mm'} S_{m'i}`,
    so that :math:`S_{mi'} = 0` while the active orbitals are unchanged.

    Parameters
    ----------
    s1 : ndarray, shape (n, n)
        Overlap matrix.
    bfs_m : sequence of int
        Active orbitals.
    bfs_i : sequence of int
        Remaining orbitals.
    """
    nbf = len(bfs_m) + len(bfs_i)
    s_mm = get_subspace(s1, bfs_m)
    s_mi = s1.take(bfs_m, axis=0).take(bfs_i, axis=1)
    U_mi = -np.linalg.inv(s_mm).dot(s_mi)
    U2 = np.eye(s1.shape[0])
    U2[np.ix_(bfs_m, bfs_i)] = U_mi
    return U2


def get_U2T(s1_MM: NDArray, bfs_m: Sequence[int], bfs_i: Sequence[int]) -> NDArray:
    """Rotation that makes the active orbitals orthogonal to the remaining ones.

    Mirror of :func:`get_U2`: the remaining orbitals are unchanged.

    Parameters
    ----------
    s1_MM : ndarray, shape (n, n)
        Overlap matrix.
    bfs_m : sequence of int
        Active orbitals.
    bfs_i : sequence of int
        Remaining orbitals.
    """
    nbf = len(bfs_m) + len(bfs_i)
    s_ii = get_subspace(s1_MM, bfs_i)
    s_im = s1_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    U_im = -np.linalg.inv(s_ii).dot(s_im)
    U2T = np.eye(s1_MM.shape[0])
    U2T[np.ix_(bfs_i, bfs_m)] = U_im
    return U2T


def extract_orthogonal_subspaces(h: NDArray, s: NDArray, bfs_m: Any, U: Optional[NDArray] = None, T: bool = False) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], NDArray]:
    """Split the system into an active subspace and the rest, with zero overlap between them.

    Parameters
    ----------
    h, s : ndarray, shape (n, n)
        Hamiltonian and overlap.
    bfs_m : ndarray of int
        Active orbitals (overwritten with their new positions).
    U : ndarray, optional
        Rotation applied first, for example from
        :func:`subdiagonalize_atoms`.
    T : bool, default False
        Orthogonalize the active orbitals to the rest
        (:func:`get_U2T`) instead of the rest to the active ones
        (:func:`get_U2`).

    Returns
    -------
    hs_mm : tuple of ndarray
        ``(H, S)`` of the active subspace.
    hs_ii : tuple of ndarray
        ``(H, S)`` of the rest.
    hs_im : tuple of ndarray
        ``(H, S)`` coupling from the rest to the active subspace;
        :math:`S_{im} = 0`.
    U : ndarray
        Total rotation.
    """
    nbf = h.shape[0]
    bfs_i = np.setdiff1d(range(nbf), bfs_m)
    U1, bfs_m, bfs_i = get_U1(bfs_m, bfs_i, U)
    h1 = rotate_matrix(h, U1)
    s1 = rotate_matrix(s, U1)
    U2 = get_U2(s1, bfs_m, bfs_i) if not T else get_U2T(s1, bfs_m, bfs_i)
    h2 = rotate_matrix(h1, U2)
    s2 = rotate_matrix(s1, U2)
    hs_mm, hs_ii, hs_im = split_subspaces(h2, s2, bfs_m, bfs_i)
    U = U1.dot(U2)
    return hs_mm, hs_ii, hs_im, U


def split_subspaces(h_MM: NDArray, s_MM: NDArray, bfs_m: Sequence[int], bfs_i: Sequence[int]) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Blocks of :math:`H` and :math:`S` on two sets of orbitals and between them.

    Parameters
    ----------
    h_MM, s_MM : ndarray, shape (n, n)
    bfs_m, bfs_i : sequence of int

    Returns
    -------
    hs_mm, hs_ii, hs_im : tuple of ndarray
        ``(H, S)`` on ``m``, on ``i``, and from ``i`` to ``m``.
    """
    nbf = h_MM.shape[0]
    h_ii = get_subspace(h_MM, bfs_i)
    s_ii = get_subspace(s_MM, bfs_i)
    h_mm = get_subspace(h_MM, bfs_m)
    s_mm = get_subspace(s_MM, bfs_m)
    h_im = h_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    s_im = s_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    return (h_mm, s_mm), (h_ii, s_ii), (h_im, s_im)


flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]


def get_orthogonal_subspaces(basis: Any, h_MM: NDArray, s_MM: NDArray, a: Any = None, cutoff: float = np.inf, condition: Optional[Callable[[float], bool]] = None) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], NDArray]:
    r"""Active subspace of local orbitals on chosen atoms, and the rest orthogonal to it.

    The on-site Hamiltonian of each atom in ``a`` is diagonalized
    (:func:`subdiagonalize_atoms`); the local orbitals whose energy passes
    ``condition`` form the active subspace, for example the :math:`d`
    orbitals near the Fermi level of a magnetic impurity. The rest of the
    basis is then made orthogonal to it (:func:`extract_orthogonal_subspaces`).

    Parameters
    ----------
    basis : Basis
        Orbital basis.
    h_MM, s_MM : ndarray, shape (n, n)
        Hamiltonian and overlap.
    a : int or sequence of int
        Atoms that host the active orbitals.
    cutoff : float, default inf
        Keep local orbitals with :math:`|\varepsilon| < \mathrm{cutoff}`, measured from
        the energy zero of ``h_MM``.
    condition : callable, optional
        ``condition(energy) -> bool``, replacing the ``cutoff`` test.

    Returns
    -------
    hs_mm, hs_ii, hs_im, U
        See :func:`extract_orthogonal_subspaces`.
    """
    c_MM, e_aj = subdiagonalize_atoms(basis, h_MM, s_MM, a)
    bfs_imp = basis.get_indices(a)
    e_j = (e for e_j in e_aj for e in e_j)
    if condition is None:
        condition = lambda e: abs(e) < cutoff
    # bfs_m = list(bfs_imp[j] for j, e in enumerate(e_j) if abs(e)<cutoff)
    bfs_m = list(bfs_imp[j] for j, e in enumerate(e_j) if condition(e))
    return extract_orthogonal_subspaces(h_MM, s_MM, bfs_m, c_MM)


def order_transform(basis: Any, groups: Optional[List[List[int]]] = None, nuc: Optional[int] = None) -> Tuple[NDArray, List[List[List[int]]]]:
    """Reordering that groups orbitals by atom group, element and orbital type.

    Within each group of atoms, atoms of the same element are collected,
    and their orbitals are ordered orbital type first: the first orbital of
    every such atom, then the second, and so on. Atoms of one element must
    carry the same number of orbitals.

    Parameters
    ----------
    basis : Basis
        Orbital basis.
    groups : list of list of int, optional
        Atoms of each group.
    nuc : int, optional
        Number of consecutive unit cells of equal size, each taken as a
        group. Ignored if ``groups`` is given. With neither, all atoms form
        one group.

    Returns
    -------
    U : ndarray, shape (n_orb, n_orb)
        Permutation matrix.
    indices : list
        For each group, the atoms of each element.

    Raises
    ------
    ValueError
        If ``groups`` is not a list of lists, or the number of atoms is not
        a multiple of ``nuc``.
    """
    atoms = basis.atoms
    symbols = atoms.symbols
    natoms = len(atoms)

    if groups is not None:
        try:
            groups[0][0]
        except IndexError:
            raise ValueError("`groups` must be a list of lists of atoms in each group.")
        nuc = len(groups)
        range_uc = lambda uc: groups[uc]
    else:
        if nuc is not None:
            if not natoms % nuc == 0.0:
                raise ValueError("Number of unit cells are not a dividend of atoms.")
            nuc_atoms = natoms // nuc
        else:
            nuc = 1
            nuc_atoms = natoms
        range_uc = lambda uc: range(uc * nuc_atoms, (uc + 1) * nuc_atoms)

    indices = []
    for uc in range(nuc):
        elements = np.unique(atoms[range_uc(uc)].symbols)
        indices.append(
            list(
                list(filter(lambda a: symbols[a] == elem, range_uc(uc)))
                for elem in elements
            )
        )

    perm_indices = []
    for uc in indices:
        for elem in uc:
            perm_indices.extend(
                basis.get_indices(elem).reshape(len(elem), -1).T.flatten()
            )

    U = np.eye(basis.nao).take(perm_indices, axis=1)
    return U, indices


def get_ll_mm_rr(H: NDArray, S: NDArray, nbf_l: int, nbf_m: int, nbf_r: int) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Central block and its couplings for orbitals ordered left, middle, right.

    Parameters
    ----------
    H, S : ndarray, shape (n, n)
    nbf_l, nbf_m, nbf_r : int
        Number of orbitals in the left, middle and right parts.

    Returns
    -------
    hs_lm, hs_mm, hs_rm : tuple of ndarray
        ``(H, S)`` from left to middle, on the middle, and from right to
        middle.
    """
    rng_l = list(range(nbf_l))
    rng_m = list(range(nbf_l, nbf_l + nbf_m))
    rng_r = list(range(nbf_l + nbf_m, nbf_l + nbf_m + nbf_r))
    s_lm = S.take(rng_l, 0).take(rng_m, 1)
    h_lm = H.take(rng_l, 0).take(rng_m, 1)
    s_rm = S.take(rng_r, 0).take(rng_m, 1)
    h_rm = H.take(rng_r, 0).take(rng_m, 1)
    h_mm = get_subspace(H, rng_m)
    s_mm = get_subspace(S, rng_m)
    return (h_lm, s_lm), (h_mm, s_mm), (h_rm, s_rm)


def get_U3(S: NDArray, bfs_l: Sequence[int], bfs_m: Sequence[int], bfs_r: Sequence[int]) -> NDArray:
    """Rotation that makes the left and right orbitals orthogonal to the middle ones.

    The middle orbitals are unchanged. Exact only when the left and right
    orbitals do not overlap with each other.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Overlap matrix.
    bfs_l, bfs_m, bfs_r : sequence of int
        Left, middle and right orbitals.
    """
    mm = np.ix_(bfs_m, bfs_m)
    ml = np.ix_(bfs_m, bfs_l)
    mr = np.ix_(bfs_m, bfs_r)
    U = np.eye(len(bfs_l) + len(bfs_m) + len(bfs_r))
    U[ml] = -np.linalg.inv(S[mm]).dot(S[ml])
    U[mr] = -np.linalg.inv(S[mm]).dot(S[mr])
    return U
