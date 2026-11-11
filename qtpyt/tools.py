r"""Preparing Hamiltonians and lead couplings before a transport calculation.

The functions here adjust matrices taken from a periodic DFT calculation
so that they describe an open system: removing couplings across the
periodic boundary, and resizing or reordering the coupling between a lead
and the device.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from qtpyt import xp
from qtpyt.base.selfenergy import BaseSelfEnergy


def remove_pbc(basis: "Basis", A_kMM: NDArray, direction: str = "x", eps: float = -1e-3) -> None:
    r"""Remove couplings that wrap around the periodic boundary along the transport direction.

    A Hamiltonian from a periodic calculation couples orbitals near one
    face of the cell to orbitals near the opposite face through the
    boundary. In an open transport geometry these couplings are unphysical.
    Matrix elements between orbitals whose centres are farther apart than
    half the cell length along ``direction`` are set to zero.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the cell, with atoms and their positions.
    A_kMM : ndarray, shape (n_k, n_orb, n_orb)
        Matrices (Hamiltonian or overlap at each k-point), modified in place.
    direction : {'x', 'y', 'z'}, default 'x'
        Transport direction. The cell vector along it must be parallel to
        that Cartesian axis.
    eps : float, default -1e-3
        Tolerance in Å. Elements are kept when the distance is below
        :math:`L/2 - \mathrm{eps}`, so the default keeps pairs exactly at
        half the cell length :math:`L`.
    """

    # Transport direction
    p_dir = "xyz".index(direction)

    L = basis.atoms.cell[p_dir, p_dir]

    cutoff = L / 2 - eps
    # Coordinates of central unit cell i (along transport)
    centers_p_i = basis.centers[:, p_dir]
    # Coordinates of neighbooring unit cell j
    # centers_p_j = centers_p_i + L
    # Distance between j atoms and i atoms
    dist_p_ji = np.abs(centers_p_i[:, None] - centers_p_i[None, :])
    # Mask j atoms farther than L
    mask_ji = (dist_p_ji < cutoff).astype(A_kMM.dtype)

    A_kMM *= mask_ji[None, :]


def expand_coupling(selfenergy: BaseSelfEnergy, nbf_m: int, id: str = "left") -> None:
    r"""Embed a lead coupling into the full orbital space of the device.

    The lead couples only to the device orbitals at its own edge. This
    pads the coupling with zeros so that the self-energy acts on all
    :math:`n_m` device orbitals, with the lead attached to the first
    :math:`n_i` orbitals (left lead) or the last :math:`n_i` (right lead).

    Parameters
    ----------
    selfenergy : BaseSelfEnergy
        Lead self-energy with square couplings ``h_ij``, ``s_ij`` of size
        :math:`n_i`. Modified in place, on CPU or GPU.
    nbf_m : int
        Number of orbitals :math:`n_m` in the device.
    id : {'left', 'right'}, default 'left'
        Side of the device the lead is on.

    Raises
    ------
    RuntimeError
        If the self-energy matrices are neither NumPy nor CuPy arrays.
    """
    if type(selfenergy.H).__module__ == "numpy":
        # Force numpy array for principallayer
        # which is only implemented for CPU
        _xp = np
    elif type(selfenergy.H).__module__ == "cupy":
        _xp = xp
    else:
        raise RuntimeError("Array module not recognised.")
    nbf_i = selfenergy.h_ij.shape[0]
    h_ij = _xp.zeros((nbf_i, nbf_m), complex)
    s_ij = _xp.zeros_like(h_ij)
    if id == "left":
        h_ij[:, :nbf_i] = selfenergy.h_ij
        s_ij[:, :nbf_i] = selfenergy.s_ij
    else:
        h_ij[:, -nbf_i:] = selfenergy.h_ij
        s_ij[:, -nbf_i:] = selfenergy.s_ij
    selfenergy.h_ij = h_ij
    selfenergy.s_ij = s_ij
    selfenergy.Sigma = _xp.zeros((nbf_m, nbf_m), complex)


def rotate_couplings(
    basis: "Basis",
    selfenergy: BaseSelfEnergy,
    N: Sequence[int] = (1, 1, 1),
    order: str = "xyz",
    positions: Optional[NDArray[np.float64]] = None,
) -> None:
    r"""Reorder the device side of a lead coupling to match the device's atom order.

    The lead and the device are often built separately, so the atoms at
    their interface can be listed in different orders. This permutes the
    columns of the coupling so that they follow the orbital order of the
    neighbouring region.

    Parameters
    ----------
    basis : Basis
        Orbitals of the lead region the coupling refers to. Can describe
        one principal layer that is repeated ``N`` times.
    selfenergy : BaseSelfEnergy
        Self-energy whose ``h_ij`` and ``s_ij`` are permuted in place.
    N : sequence of int, default (1, 1, 1)
        Repetitions of ``basis`` along x, y, z.
    order : str, default 'xyz'
        Permutation of ``'x'``, ``'y'``, ``'z'``: the coordinate by which
        the interface atoms of the neighbouring region are sorted first,
        second and third.
    positions : ndarray, shape (n_atoms, 3), optional
        Positions of the interface atoms of the neighbouring region.
    """
    from qtpyt.surface.tools import order_indices

    permutation = order_indices(basis, N, order, positions)

    h_ij = selfenergy.h_ij
    s_ij = selfenergy.s_ij
    h_ij.take(permutation, axis=1, out=h_ij)
    s_ij.take(permutation, axis=1, out=s_ij)


def tri2full(H_nn: NDArray, UL: str = "L", map: Callable = np.conj) -> None:
    """Complete a Hermitian (or symmetric) matrix from one of its triangles.

    Codes that store only half of the Hamiltonian give the other half as
    zeros. This copies one triangle onto the other, in place.

    Parameters
    ----------
    H_nn : ndarray, shape (n, n)
        Square matrix, modified in place.
    UL : {'L', 'U'}, default 'L'
        ``'L'`` copies the lower triangle into the upper one, ``'U'`` the
        upper into the lower.
    map : callable, default numpy.conj
        ``map(src, dst)`` applied to each row segment. The default gives a
        Hermitian matrix. For an anti-Hermitian one::

            def antihermitian(src, dst):
                np.conj(-src, dst)

            tri2full(H_nn, map=antihermitian)
    """
    N, tmp = H_nn.shape
    assert N == tmp, "Matrix must be square"
    if UL != "L":
        H_nn = H_nn.T

    for n in range(N - 1):
        map(H_nn[n + 1 :, n], H_nn[n, n + 1 :])
