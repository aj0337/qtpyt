r"""Basis changes of a device Hamiltonian that keep its coupling to the leads consistent."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from numpy.typing import NDArray

import numpy as np
from functools import reduce

from .tools import (
    lowdin_rotation,
    subdiagonalize,
    rotate_matrix,
    get_subspace,
    subdiagonalize_atoms,
    get_U1,
    get_U2,
)


class CoupledHamiltonian:
    r"""Device Hamiltonian and overlap together with the lead couplings, rotated as one.

    Each rotation :math:`U` of the device basis transforms
    :math:`H \to U^\dagger H U`, :math:`S \to U^\dagger S U`, and the lead
    couplings :math:`H_{im} \to H_{im} U`, so that the transport problem is
    unchanged. Use it to move to molecular orbitals, Löwdin orbitals or an
    orthogonal active subspace before computing the Green's function.

    Parameters
    ----------
    H, S : ndarray, shape (n_orb, n_orb)
        Device Hamiltonian and overlap, rotated in place.
    selfenergies : list of (index, self-energy) pairs, default []
        Lead self-energies with coupling attributes ``h_im`` and ``s_im``.

    Attributes
    ----------
    U : list of ndarray
        Rotations applied so far, in order.
    """

    def __init__(self, H: NDArray, S: NDArray, selfenergies: List[Any] = []) -> None:
        self.H = H
        self.S = S
        self.selfenergies = selfenergies
        self.U = []

    def align_bf(self, align_bf: Optional[int]) -> float:
        r"""Shift the device levels so that one orbital matches the left lead.

        :math:`H \to H - \Delta\,S` with
        :math:`\Delta = (H_{bb} - \operatorname{Re} h^{L}_{bb}) / S_{bb}`, where
        :math:`h^L` is the left lead's principal-layer Hamiltonian.

        Parameters
        ----------
        align_bf : int
            Reference orbital :math:`b`.

        Returns
        -------
        diff : float
            The shift :math:`\Delta`.
        """
        h_mm = self.H
        s_mm = self.S
        h1_ii = self.selfenergies[0].h_ii
        if align_bf is not None:
            diff = (
                h_mm[align_bf, align_bf] - np.real(h1_ii[align_bf, align_bf])
            ) / s_mm[align_bf, align_bf]
            # print('# Aligning scat. H to left lead H. diff=', diff)
            h_mm -= diff * s_mm
        return diff

    def apply_rotation(self, U: NDArray) -> None:
        r"""Change the device basis by :math:`U` and update the lead couplings.

        Parameters
        ----------
        U : ndarray, shape (n_orb, n_orb)
        """
        h_mm = self.H
        s_mm = self.S
        h_mm[:] = rotate_matrix(h_mm, U)
        s_mm[:] = rotate_matrix(s_mm, U)
        # Rotate coupling between lead and central region
        for indices, selfenergy in self.selfenergies:
            selfenergy.h_im[:] = np.dot(selfenergy.h_im, U[indices])
            selfenergy.s_im[:] = np.dot(selfenergy.s_im, U[indices])
        self.U.append(U)

    def diagonalize(self) -> None:
        """Rotate to the eigenstates of the isolated device (molecular orbitals)."""
        nbf = len(self.H)
        self.subdiagonalize(range(nbf))

    def subdiagonalize(self, bfs: Sequence[int]) -> None:
        """Diagonalize the device Hamiltonian within a subset of orbitals, see :func:`~qtpyt.lo.tools.subdiagonalize`.

        Parameters
        ----------
        bfs : sequence of int
        """
        U, eig = subdiagonalize(self.H, self.S, bfs)
        self.apply_rotation(U)

    def take(self, bfs: Sequence[int], apply: bool = False) -> None:
        """Keep only a subset of device orbitals, dropping the others.

        Parameters
        ----------
        bfs : sequence of int
            Orbitals to keep.
        apply : bool, default False
            Ignored.
        """
        nbf = len(self.H)
        U = np.eye(nbf).take(bfs, 1)
        self.H = rotate_matrix(self.H, U)
        self.S = rotate_matrix(self.S, U)
        for indices, selfenergy in self.selfenergies:
            selfenergy.h_im = np.dot(selfenergy.h_im, U[indices])
            selfenergy.s_im = np.dot(selfenergy.s_im, U[indices])
        self.U.append(U)

    def lowdin_rotation(self, bfs: Optional[Sequence[int]] = None) -> None:
        """Löwdin-orthogonalize a subset of orbitals, see :func:`~qtpyt.lo.tools.lowdin_rotation`.

        Parameters
        ----------
        bfs : sequence of int, optional
            All orbitals if omitted.
        """
        U = lowdin_rotation(self.H, self.S, bfs)
        self.apply_rotation(U)

    def subdiagonalize_atoms(self, basis: Any, a: Optional[Sequence[int]] = None) -> None:
        """Diagonalize the Hamiltonian on each atom separately, see :func:`~qtpyt.lo.tools.subdiagonalize_atoms`.

        Parameters
        ----------
        basis : Basis
        a : sequence of int, optional
            Atoms to treat; all if omitted.
        """
        U, eig = subdiagonalize_atoms(basis, self.H, self.S, a)
        self.apply_rotation(U)

    def partition_orthogonal(self, bfs_m: Sequence[int], bfs_i: Sequence[int]) -> None:
        """Move orbitals ``bfs_m`` last and make all other orbitals orthogonal to them.

        Parameters
        ----------
        bfs_m : sequence of int
            Active subspace, unchanged.
        bfs_i : sequence of int
            Remaining orbitals.
        """
        U1, bfs_m, bfs_i = get_U1(bfs_m, bfs_i)
        self.apply_rotation(U1)
        U2 = get_U2(self.S, bfs_m, bfs_i)
        self.apply_rotation(U2)

    def get_rotation(self) -> NDArray:
        """Total rotation, the product of all rotations applied."""
        return reduce(lambda a, b: np.dot(a, b), self.U)
