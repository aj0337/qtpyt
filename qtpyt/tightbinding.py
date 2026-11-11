r"""Real-space Hamiltonians from k-point calculations, and band structures.

A periodic LCAO calculation (for example GPAW) gives :math:`H(\mathbf k)`
and :math:`S(\mathbf k)` on a Monkhorst–Pack grid. Their Fourier
transform gives the couplings :math:`H(\mathbf R)` between a unit cell
and its neighbour at lattice vector :math:`\mathbf R`, which is how leads
and principal layers are built.

The convention throughout is

.. math::

    A(\mathbf R) = \frac{1}{N_k}\sum_{\mathbf k} A(\mathbf k)\,
    e^{2\pi i\,\mathbf k\cdot\mathbf R},
    \qquad
    A(\mathbf k) = \sum_{\mathbf R} A(\mathbf R)\,
    e^{-2\pi i\,\mathbf k\cdot\mathbf R},

with :math:`\mathbf k` in units of the reciprocal lattice vectors and
:math:`\mathbf R` in units of the lattice vectors.
"""
from __future__ import annotations

from math import pi
from typing import Optional, Sequence, Tuple, Union

from numpy.typing import NDArray

import numpy as np
import scipy.linalg as sla

from qtpyt.surface.kpts import (
    apply_inversion_symm,
    expand_inversion_symm,
    fourier_sum,
    get_monkhorst_pack_size_and_offset,
)


def bloch_to_realspace(A: NDArray, kpts: NDArray[np.float64], R: NDArray, out: Optional[NDArray] = None) -> NDArray:
    r"""Real-space matrices :math:`A(\mathbf R)` from k-space matrices, also from the irreducible zone.

    Same as :func:`qtpyt.surface.kpts.fourier_sum`, except that ``A`` may
    be given only on the k-points left after time-reversal symmetry
    :math:`A(-\mathbf k) = A(\mathbf k)^*` is used. The missing half of
    the zone is restored from that symmetry.

    Parameters
    ----------
    A : ndarray, shape (n_k, n_orb, n_orb)
        Matrices at each k-point, on the full grid or on its irreducible
        part.
    kpts : ndarray, shape (n_kfull, 3)
        The full Γ-centred Monkhorst–Pack grid, in units of the reciprocal
        lattice vectors. If it has more points than ``A`` has matrices,
        ``A`` is taken to be on the irreducible part.
    R : ndarray, shape (3,) or (n_R, 3)
        Lattice vectors, in units of the lattice vectors.
    out : ndarray, optional
        Array to write the result into.

    Returns
    -------
    A_R : ndarray, shape (n_R, n_orb, n_orb) or (n_orb, n_orb)
        :math:`A(\mathbf R)`.

    Raises
    ------
    NotImplementedError
        If the k-point grid is not Γ-centred.
    """
    size, offset = get_monkhorst_pack_size_and_offset(kpts)
    if not np.allclose(offset, (0, 0, 0)):
        # TODO :: In this case one cannot simply out.conj()
        raise NotImplementedError

    kpts = np.atleast_2d(kpts)
    R = np.atleast_2d(R)
    # Irreducible brillouin zone
    ibz = False
    if kpts.shape[0] > A.shape[0]:
        # A's are at IBZ
        Nk = [len(set(kpts[:, i])) for i in range(3)]
        kpts = apply_inversion_symm(kpts)[0]
        ibz = True
    out = fourier_sum(A, kpts, R, out)
    if ibz:
        out *= kpts.shape[0]
        out += out.conj()
        try:
            gamma = np.where(np.sum(kpts, axis=1) < 1e-9)[0][0]
        except IndexError:
            pass
        else:
            #       ik          -ik
            # A(k) e    + A(-k) e
            #                                 T    *
            # NOTE: A(k) = A(-k)  & A(k) = ( A(k) )
            #
            #         ik   *     *   -ik       T    * -ik          -ik
            # ( A(k) e    )  =  A(k) e   = ( A(-k) )  e    = A(-k) e
            out -= A[gamma]
        out /= np.product(Nk)
    return out


class TightBinding:
    r"""Real-space tight-binding model of a crystal, built from k-space LCAO matrices.

    Stores :math:`H(\mathbf R)` and :math:`S(\mathbf R)` for the cells
    around the origin, and evaluates bands at any k-point from them. Use it
    to check that the real-space model reproduces the DFT bands before
    building leads from it.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Full Monkhorst–Pack grid, in units of the reciprocal lattice
        vectors.
    H_kMM : ndarray, shape (n_k or n_kirr, n_orb, n_orb)
        Hamiltonian at each k-point, on the full grid or on the irreducible
        part (then expanded by time-reversal symmetry). Any energy unit;
        bands come out in the same unit.
    S_kMM : ndarray, same shape as ``H_kMM``
        Overlap at each k-point.

    Attributes
    ----------
    H_NMM, S_NMM : ndarray, shape (n_R, n_orb, n_orb)
        :math:`H(\mathbf R)` and :math:`S(\mathbf R)` for the lattice
        vectors in ``R_cN``.
    R_cN : ndarray of int, shape (3, n_R)
        Lattice vectors of the cells kept, see :meth:`set_num_cells`.
    """

    def __init__(self, kpts: NDArray[np.float64], H_kMM: NDArray, S_kMM: NDArray) -> None:
        if H_kMM.shape[0] < kpts.shape[0]:
            ks, H_kMM = expand_inversion_symm(kpts, H_kMM)
            assert np.allclose(ks, kpts), "Invalid k-points"
            S_kMM = expand_inversion_symm(kpts, S_kMM)[1]

        self.kpts = kpts
        # here we have set kpts to the reducible representation,
        # so Nk is correctly computed when called from set_num_cells.
        self.set_num_cells()
        self.H_NMM = fourier_sum(H_kMM, self.kpts, self.R_cN.T)
        self.S_NMM = fourier_sum(S_kMM, self.kpts, self.R_cN.T)

    @property
    def Nk(self) -> NDArray[np.int_]:
        """Number of distinct k-point coordinates along each reciprocal lattice vector."""
        return np.array([len(set(self.kpts[:, i])) for i in range(3)])

    def set_num_cells(self, N_c: Optional[Sequence[int]] = None) -> None:
        r"""Choose how many neighbouring cells the real-space model keeps.

        The cells kept are centred on the origin, with integer coordinates
        from :math:`-\lfloor N/2 \rfloor` to :math:`N - 1 - \lfloor N/2 \rfloor`
        along each lattice vector. More cells than k-points
        along a direction cannot be resolved; a warning is printed in that
        case.

        Parameters
        ----------
        N_c : sequence of three int, optional
            Number of cells along each lattice vector. Defaults to the
            number of k-points along each direction.
        """

        if N_c is None:
            self.N_c = tuple(self.Nk)
        else:
            self.N_c = tuple(N_c)
        try:
            if np.any(np.asarray(self.Nk) < np.asarray(self.N_c)):
                print("WARNING: insufficient k-point sampling.")
        except AttributeError:
            # Nk is not defined
            pass
        # Lattice vectors
        R_cN = np.indices(self.N_c).reshape(3, -1)
        N_c = np.array(self.N_c)[:, np.newaxis]
        R_cN += N_c // 2
        R_cN %= N_c
        R_cN -= N_c // 2
        self.R_cN = R_cN

    def band_structure(self, path_kc: NDArray[np.float64], blochstates: bool = False) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.complex128]]]:
        r"""Band energies along a path in the Brillouin zone.

        Solves :math:`H(\mathbf k)\,c = \varepsilon\,S(\mathbf k)\,c` at each
        k-point, with :math:`H(\mathbf k)` rebuilt from the real-space model.

        Parameters
        ----------
        path_kc : ndarray, shape (n_path, 3)
            k-points along the path, in units of the reciprocal lattice
            vectors.
        blochstates : bool, default False
            Also return the LCAO coefficients of the Bloch states.

        Returns
        -------
        eps_kn : ndarray, shape (n_path, n_orb)
            Band energies at each k-point, increasing, in the unit of the
            input Hamiltonian.
        psi_kn : ndarray, shape (n_path, n_orb, n_orb)
            Only if ``blochstates``. ``psi_kn[k, n]`` holds the LCAO
            coefficients of band ``n``.
        """
        R_cN = self.R_cN
        # Lists for eigenvalues and eigenvectors along path
        eps_kn = []
        psi_kn = []

        for k_c in path_kc:
            # Evaluate fourier sum
            phase_N = np.exp(-2.0j * pi * np.dot(k_c, R_cN))
            H_MM = np.sum(phase_N[:, np.newaxis, np.newaxis] * self.H_NMM, axis=0)
            S_MM = np.sum(phase_N[:, np.newaxis, np.newaxis] * self.S_NMM, axis=0)

            if blochstates:
                eps_n, c_Mn = sla.eigh(H_MM, S_MM)
                # Sort eigenmodes according to increasing eigenvalues
                c_nM = c_Mn[:, eps_n.argsort()].transpose()
                psi_kn.append(c_nM)
            else:
                eps_n = sla.eigvalsh(H_MM, S_MM)

            # Sort eigenvalues in increasing order
            eps_n.sort()
            eps_kn.append(eps_n)

        # Convert to eV
        eps_kn = np.array(eps_kn)  # * units.Hartree

        if blochstates:
            return eps_kn, np.array(psi_kn)

        return eps_kn
