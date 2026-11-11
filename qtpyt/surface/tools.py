r"""Preparing lead and device matrices from periodic DFT calculations.

The typical route: compute :math:`H(\mathbf k)` and :math:`S(\mathbf k)` of the
bulk electrode, turn them into principal-layer matrices at each transverse
k-point with :func:`prepare_leads_matrices`, and remove the couplings across
the periodic boundary of the device with :func:`prepare_central_matrices`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from numpy.typing import NDArray

import numpy as np
from qtpyt.basis import argsort
from scipy.spatial.distance import cdist

from .kpts import (
    apply_inversion_symm,
    build_partial_kpts,
    fourier_sum,
    monkhorst_pack,
    partial_bloch_sum,
)


def remove_pbc(basis: "Basis", A_kMM: NDArray, direction: str = "x", eps: float = -1e-3) -> None:
    r"""Keep only couplings between a cell and its next neighbour that point forward.

    An element between orbital :math:`a` and orbital :math:`b` is kept when
    :math:`|x_a + L - x_b| > L - \mathrm{eps}` along ``direction``, with
    :math:`L` the cell length, and set to zero otherwise. ``A_kMM`` is
    modified in place. For removing couplings across the periodic boundary
    of a device, see :func:`qtpyt.tools.remove_pbc`.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the cell.
    A_kMM : ndarray, shape (n_k, n_orb, n_orb)
        Matrices, modified in place.
    direction : {'x', 'y', 'z'}, default 'x'
        Transport direction.
    eps : float, default -1e-3
        Distance tolerance in Å.
    """

    # Transport direction
    p_dir = "xyz".index(direction)

    L = basis.atoms.cell[p_dir, p_dir]

    cutoff = L - eps
    # Coordinates of central unit cell i (along transport)
    centers_p_i = basis.centers[:, p_dir]
    # Coordinates of neighbooring unit cell j
    centers_p_j = centers_p_i + L
    # Distance between j atoms and i atoms
    dist_p_ji = np.abs(centers_p_j[:, None] - centers_p_i[None, :])
    # Mask j atoms farther than L
    mask_ji = (dist_p_ji > cutoff).astype(A_kMM.dtype)

    A_kMM *= mask_ji[None, :]


def map_B2A(pos_A: NDArray[np.float64], pos_B: NDArray[np.float64]) -> NDArray[np.int_]:
    """For each position in ``pos_A``, the index of the closest position in ``pos_B``.

    When ``A`` and ``B`` hold the same atoms in different orders,
    ``pos_B[map_B2A(pos_A, pos_B)]`` equals ``pos_A``. Distances are
    measured as the sum of absolute coordinate differences.

    Parameters
    ----------
    pos_A : ndarray, shape (n_A, 3)
    pos_B : ndarray, shape (n_B, 3)

    Returns
    -------
    B2A : ndarray of int, shape (n_A,)
    """
    B2A = cdist(pos_A, pos_B, metric="cityblock").argmin(1)
    return B2A


def map_supercell(bsc: "Basis", bcc: "Basis") -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Match the atoms of a device region to those of a lead supercell.

    Parameters
    ----------
    bsc : Basis
        Basis of the lead supercell.
    bcc : Basis
        Basis of the device region.

    Returns
    -------
    orbitals : ndarray of int
        Orbital indices of the supercell atoms that coincide with device
        atoms.
    sc2cc : ndarray of int
        The corresponding supercell atom indices.
    """
    cc2sc = map_B2A(bsc.atoms.positions, bcc.atoms.positions)
    cc2sc.sort()
    sc2cc = map_B2A(bcc[cc2sc].atoms.positions, bsc.atoms.positions)
    return bsc.get_indices(sc2cc), sc2cc


def order_indices(basis: "Basis", N: Sequence[int], order: str = "xyz", positions: Optional[NDArray[np.float64]] = None) -> NDArray[np.int_]:
    """Orbital order of a repeated cell that matches a target atom order.

    Parameters
    ----------
    basis : Basis
        Basis of the small cell.
    N : sequence of three int
        Repetitions of the cell.
    order : str, default 'xyz'
        Sort atoms by these coordinates when ``positions`` is not given,
        see :func:`qtpyt.basis.argsort`.
    positions : ndarray, shape (n_atoms, 3), optional
        Target atom positions; the repeated atoms are matched to them by
        distance.

    Returns
    -------
    idx : ndarray of int
        Orbital indices of the repeated cell in the target order.
    """
    repeated = basis.repeat(N)
    # Get map from repeated to target order.
    if positions is not None:
        r2o = cdist(repeated.atoms.positions, positions).argmin(0)
    else:
        positions = repeated.atoms.positions.round(3)
        r2o = argsort(positions, order)
    return repeated.get_indices(r2o)


def prepare_leads_matrices(
    H_kMM: NDArray, S_kMM: NDArray, size: Sequence[int], offset: Sequence[float] = (0.0, 0.0, 0.0), direction: str = "x", align: Optional[Tuple[int, Union[float, complex]]] = None
) -> Tuple[NDArray[np.float64], NDArray, NDArray, NDArray, NDArray]:
    r"""Principal-layer matrices of a bulk electrode at each transverse k-point.

    Fourier transforms the bulk :math:`H(\mathbf k)`, :math:`S(\mathbf k)` along
    the transport direction to obtain the on-layer blocks
    (:math:`R_\parallel = 0`) and the coupling to the next layer
    (:math:`R_\parallel = 1`), keeping the transverse momentum. The result is
    the input of :class:`~qtpyt.surface.principallayer.PrincipalLayer` and
    :class:`~qtpyt.surface.principallayer.PrincipalSelfEnergy`.

    Parameters
    ----------
    H_kMM, S_kMM : ndarray, shape (n_k, n, n)
        Bulk Hamiltonian and overlap on the Monkhorst–Pack grid, or on its
        irreducible part under time reversal.
    size, offset
        The Monkhorst–Pack grid, see
        :func:`~qtpyt.surface.kpts.monkhorst_pack`.
    direction : {'x', 'y', 'z'}, default 'x'
        Transport direction.
    align : tuple (int, float), optional
        ``(orbital, energy)``: shift the electrode levels so that this
        orbital has this on-site energy, see :func:`align_orbitals`.

    Returns
    -------
    kpts_t : ndarray
        Transverse k-points.
    H_kii, S_kii : ndarray, shape (n_kt, n, n)
        On-layer blocks.
    H_kij, S_kij : ndarray, shape (n_kt, n, n)
        Coupling to the next layer.
    """

    kpts = monkhorst_pack(size, offset)
    if kpts.shape[0] > H_kMM.shape[0]:
        # Switch to irreducible k-points
        kpts = apply_inversion_symm(kpts)[0]

    kpts_p, kpts_t = build_partial_kpts(size, offset, direction)

    p_dir = "xyz".index(direction)

    R = [0, 0, 0]
    H_kii = partial_bloch_sum(H_kMM, kpts, R, kpts_p, kpts_t)
    S_kii = partial_bloch_sum(S_kMM, kpts, R, kpts_p, kpts_t)

    R[p_dir] = 1
    H_kij = partial_bloch_sum(H_kMM, kpts, R, kpts_p, kpts_t)
    S_kij = partial_bloch_sum(S_kMM, kpts, R, kpts_p, kpts_t)

    if align is not None:
        align_orbitals(kpts_t, H_kii, S_kii, H_kij, S_kij, align)
    # remove_pbc(basis, H_kij, direction)
    # remove_pbc(basis, S_kij, direction)

    return kpts_t, H_kii, S_kii, H_kij, S_kij


def prepare_central_matrices(
    basis: "Basis", H_kMM: NDArray, S_kMM: NDArray, size: Sequence[int], offset: Sequence[float] = (0.0, 0.0, 0.0), direction: str = "x"
) -> Tuple[NDArray[np.float64], NDArray, NDArray]:
    """Device matrices with the couplings across the periodic boundary removed.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the device cell.
    H_kMM, S_kMM : ndarray, shape (n_k, n_orb, n_orb)
        Device Hamiltonian and overlap at each k-point, modified in place
        by :func:`qtpyt.tools.remove_pbc`.
    size, offset
        The Monkhorst–Pack grid.
    direction : {'x', 'y', 'z'}, default 'x'
        Transport direction.

    Returns
    -------
    kpts : ndarray
        k-points of the matrices (irreducible if fewer matrices than grid
        points were given).
    H_kMM, S_kMM : ndarray
    """
    from qtpyt.tools import remove_pbc

    kpts = monkhorst_pack(size, offset)
    if kpts.shape[0] > H_kMM.shape[0]:
        # Switch to irreducible k-points
        kpts = apply_inversion_symm(kpts)[0]

    remove_pbc(basis, H_kMM, direction)
    remove_pbc(basis, S_kMM, direction)

    return kpts, H_kMM, S_kMM


def align_orbitals(
    kpts: np.ndarray,
    H_kii: np.ndarray,
    S_kii: np.ndarray,
    H_kij: np.ndarray,
    S_kij: np.ndarray,
    align: Tuple[int, Union[float, complex]],
) -> None:
    r"""Shift the electrode levels to line up with the device.

    Separate DFT calculations of electrode and device have different
    energy zeros. The electrode is shifted rigidly,
    :math:`H \to H - \Delta\, S`, so that the on-site energy of a reference
    orbital matches its value in the device. Matrices are modified in
    place.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Transverse k-points of the electrode matrices.
    H_kii, S_kii, H_kij, S_kij : ndarray, shape (n_k, n, n) or (n, n)
        Principal-layer matrices from :func:`prepare_leads_matrices`.
    align : tuple (int, float)
        ``(b, value)``: reference orbital :math:`b` and its on-site energy in
        the device. The shift is
        :math:`\Delta = \operatorname{Re}[(H_{bb} - \mathrm{value})/S_{bb}]`, with
        :math:`H_{bb}` the real-space on-site element.
    """
    bf, value = align
    if H_kii.ndim == 2:
        H_kii = H_kii[None, :]
        S_kii = S_kii[None, :]
    if len(H_kii) == 1:
        H_ii = H_kii[0]
        S_ii = S_kii[0]
    else:
        R = np.zeros(3)
        H_ii = fourier_sum(H_kii, kpts, R)
        S_ii = fourier_sum(S_kii, kpts, R)

    diff = np.real((H_ii[bf, bf] - value) / S_ii[bf, bf])
    H_kii -= diff * S_kii
    H_kij -= diff * S_kij
