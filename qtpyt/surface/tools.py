from typing import Tuple, Union

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


def remove_pbc(basis, A_kMM, direction="x", eps=-1e-3):
    """Mask matrix elements above diagonal connecting neighbors cells
    in parallel direction.

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


def map_B2A(pos_A, pos_B):
    """Find indices on `B` in `A`.

    Example:
          ----            ----
         |   d|          |   d|
    A=   |c   | ,   B=   |b   |
         |   b|          |   c|
         |a   |          |a   |
          ----            ----

    A = B[map_B2A(A, B)]
    """
    B2A = cdist(pos_A, pos_B, metric="cityblock").argmin(1)
    return B2A


def map_supercell(bsc, bcc):
    cc2sc = map_B2A(bsc.atoms.positions, bcc.atoms.positions)
    cc2sc.sort()
    sc2cc = map_B2A(bcc[cc2sc].atoms.positions, bsc.atoms.positions)
    return bsc.get_indices(sc2cc), sc2cc


def order_indices(basis, N, order="xyz", positions=None):
    repeated = basis.repeat(N)
    # Get map from repeated to target order.
    if positions is not None:
        r2o = cdist(repeated.atoms.positions, positions).argmin(0)
    else:
        positions = repeated.atoms.positions.round(3)
        r2o = argsort(positions, order)
    return repeated.get_indices(r2o)


def prepare_leads_matrices(
    H_kMM, S_kMM, size, offset=(0.0, 0.0, 0.0), direction="x", align=None
):
    """Prepare Hamiltonian and overlap matrices for the leads in a transport calculation.

    This function prepares the Hamiltonian (H) and overlap (S) matrices in the required format for calculating
    self-energies for the lead regions in quantum transport models. It calculates both the lead and coupling
    matrices for each k-point using Bloch summation methods.

    Args:
        H_kMM (np.ndarray): Hamiltonian matrices in k-space.
        S_kMM (np.ndarray): Overlap matrices in k-space.
        size (tuple of int): Monkhorst-Pack grid size for k-point sampling.
        offset (tuple of float, optional): Monkhorst-Pack grid offset. Defaults to (0.0, 0.0, 0.0).
        direction (str, optional): Transport direction ('x', 'y', or 'z'). Defaults to 'x'.
        align (tuple, optional): Basis index and value to align surface energies, used to match central region.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - kpts_t: k-points transverse to transport direction.
            - H_kii: Lead Hamiltonian matrix.
            - S_kii: Lead overlap matrix.
            - H_kij: Coupling Hamiltonian matrix.
            - S_kij: Coupling overlap matrix.

    Logic:
        - The function first generates the k-points for the Monkhorst-Pack grid (`kpts`).
        - If the grid exceeds the provided matrix size, it applies inversion symmetry reduction.
        - Partial Bloch summations for the Hamiltonian and overlap matrices are then calculated.
        - If the `align` parameter is given, the function adjusts (aligns) the matrices to the central scattering
          region.

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
    basis, H_kMM, S_kMM, size, offset=(0.0, 0.0, 0.0), direction="x"
):
    """Prepare input matrices for PrincipalLayer.

    Args:
        basis : basis function descriptor.
        H_kMM : Hamiltonian matrices in k-space.
        S_kMM : Overlap matrices in k-space.
        size, offset : Monkhorst-Pack used to sample Brillouin Zone.
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
):
    """Align surface energies with a central scattering region.

    Args:
        kpts : np.ndarray, ndim = 2
            K-points of surface matrices.
        H(S)_ii(ij) : np.ndarray, ndim = 2 or 3
            Matrices as given by `prepare_leads_matrices` output.
        H_MM : np.ndarray, ndim = 2
            Hamiltonian of scattering region.
        align : tuple
            basis index to align and corresponding value.
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
