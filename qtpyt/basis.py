r"""Bookkeeping of which atomic orbital belongs to which atom.

In a localized basis every matrix index is an orbital centred on an atom.
:class:`Basis` keeps that map, so that rows and columns of :math:`H`,
:math:`S` or :math:`G` can be selected by atom, by element or by
position.
"""
from __future__ import annotations

import numbers
from typing import Dict, List, Optional, Sequence, Union

import ase.neighborlist
import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike, NDArray
#### NeighborList
from ase.data import covalent_radii
from ase.neighborlist import NeighborList


def get_neighbors(atoms: Atoms) -> List[NDArray[np.int_]]:
    """Bonded neighbours of every atom, from covalent radii.

    Two atoms are neighbours when their distance is below the sum of their
    covalent radii (:data:`ase.data.covalent_radii`). Periodic images are
    included according to ``atoms.pbc``.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure.

    Returns
    -------
    neighbors : list of ndarray
        ``neighbors[a]`` holds the indices of the atoms bonded to atom ``a``.
    """
    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nl.update(atoms)

    nlists = []
    for a in atoms:
        nlists.append(nl.get_neighbors(a.index)[0])

    return nlists


def argsort(positions: NDArray[np.float64], order: str = "xyz") -> NDArray[np.int_]:
    """Indices that sort positions by coordinates, first character of ``order`` first.

    Parameters
    ----------
    positions : ndarray, shape (n, 3)
        Cartesian positions.
    order : str, default 'xyz'
        Permutation of ``'x'``, ``'y'``, ``'z'``. Positions are sorted by the
        first coordinate, ties broken by the second, then by the third
        (the reverse of the key order of :func:`numpy.lexsort`).

    Returns
    -------
    idx : ndarray of int, shape (n,)
    """
    i, j, k = ["xyz".index(i) for i in order]
    idx = np.lexsort((positions[:, k], positions[:, j], positions[:, i]))
    return idx


class Basis:
    """Map between atoms and the atomic orbitals that index the matrices.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure.
    nao_a : array_like of int, shape (n_atoms,)
        Number of atomic orbitals on each atom.
    M_a : array_like of int, shape (n_atoms,), optional
        Index of the first orbital of each atom in the full matrices.
        Defaults to orbitals stored atom by atom, starting from zero.

    Attributes
    ----------
    nao : int
        Total number of orbitals.

    Examples
    --------
    >>> from ase import Atoms
    >>> from qtpyt.basis import Basis
    >>> atoms = Atoms('OH', positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[1, 1, 1])
    >>> basis = Basis.from_dictionary(atoms, {'O': 2, 'H': 1})
    >>> int(basis.nao)
    3
    >>> basis.get_indices([1])
    array([2])
    """

    def __init__(self, atoms: Atoms, nao_a: ArrayLike, M_a: Optional[ArrayLike] = None) -> None:
        self.atoms = atoms
        self.nao_a = np.array(nao_a, ndmin=1)
        if M_a is None:
            M_a = np.cumsum(np.insert(self.nao_a[:-1], 0, 0))
        self.M_a = np.array(M_a, ndmin=1)
        self.nao = np.sum(self.nao_a)  # Total # of atomic orbitals

    @classmethod
    def from_dictionary(cls, atoms: Atoms, basis: Dict[str, int]) -> "Basis":
        """Basis with the same number of orbitals on every atom of an element.

        Parameters
        ----------
        atoms : ase.Atoms
            The structure.
        basis : dict
            ``{symbol: number of orbitals}``, for example ``{'Au': 9, 'C': 4}``.
        """
        nao_a = [basis[symbol] for symbol in atoms.symbols]
        return cls(atoms, nao_a)

    @property
    def centers(self) -> NDArray[np.float64]:
        """Position of every orbital, the position of the atom it sits on.

        An array of shape ``(nao, 3)``.
        """
        return self._expand(self.atoms.positions)

    def repeat(self, N: Sequence[int]) -> "Basis":
        """Basis of the structure repeated ``N`` times, as :meth:`ase.Atoms.repeat`.

        Parameters
        ----------
        N : sequence of three int
            Repetitions along the three cell vectors.
        """
        nao_a = np.tile(self.nao_a, np.prod(N))
        atoms = self.atoms.repeat(N)
        return Basis(atoms, nao_a)

    def argsort(self, order: str = "xyz") -> NDArray[np.int_]:
        """Atom indices sorted by position, see :func:`argsort`.

        Parameters
        ----------
        order : str, default 'xyz'
            Sort key order, first character first.
        """
        return argsort(self.atoms.positions, order)

    def get_indices(self, indices: Optional[Union[int, Sequence[int], NDArray[np.int_]]] = None) -> NDArray[np.int_]:
        """Orbital indices of a set of atoms, in the order the atoms are given.

        Use the result to cut the block of :math:`H`, :math:`S` or
        :math:`G` that belongs to those atoms.

        Parameters
        ----------
        indices : int or sequence of int, optional
            Atom indices. All atoms if omitted.

        Returns
        -------
        idx : ndarray of int
            Indices into the full matrices.
        """
        if indices is None:
            M_a, nao_a, nao = self.M_a, self.nao_a, self.nao
        elif isinstance(indices, numbers.Integral):
            return np.arange(self.nao_a[indices]) + self.M_a[indices]
        else:
            M_a = self.M_a[indices]
            nao_a = self.nao_a[indices]
            nao = np.sum(nao_a)

        idx = self._expand(M_a, nao_a)
        i0 = M_a[0]
        for i in range(1, nao):
            if idx[i] == i0:
                idx[i] = idx[i - 1] + 1
            else:
                i0 = idx[i]
        return idx

    def take(self, basis_dict: Dict[str, Union[int, Sequence[int]]]) -> NDArray[np.int_]:
        r"""Indices of chosen orbitals on every atom of chosen elements.

        For example the :math:`p_z` orbital of every carbon atom, to build a
        tight-binding model of the :math:`\pi` system.

        Parameters
        ----------
        basis_dict : dict
            ``{symbol: orbital index or indices within the atom}``, for
            example ``{'C': 3, 'Au': [0, 1]}``.

        Returns
        -------
        idx : ndarray of int
            Sorted indices into the full matrices.
        """
        idxlist = []
        for kind, idx in basis_dict.items():
            if isinstance(idx, numbers.Integral):
                idx = [idx]
            for a in self.atoms.symbols.search(kind):
                for i in idx:
                    idxlist.append(self.M_a[a] + i)
        idxlist = np.asarray(idxlist)
        idxlist.sort()
        return idxlist

    def extract(self) -> "Basis":
        """Copy of this basis with orbital indices renumbered from zero.

        Use it after selecting atoms with ``basis[i]`` when the selected
        block is treated as a matrix of its own.
        """
        return self.__class__(self.atoms, self.nao_a)

    def __getitem__(self, i):
        """Subset of atoms that keeps their orbital indices in the full matrices."""
        return self.__class__(self.atoms[i], self.nao_a[i], self.M_a[i])

    def __len__(self):
        return self.nao

    def _expand(self, array, repeats=None):
        """Helper function. Expand atoms array to basis functions."""
        if repeats is None:
            repeats = self.nao_a
        return np.repeat(array, repeats, axis=0)
