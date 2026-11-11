r"""Leads and surfaces that are periodic in the directions transverse to transport.

A wide electrode is a supercell of many transverse repetitions of a small
cell. Its Hamiltonian is block diagonal in the transverse Bloch momentum
:math:`\mathbf k_\perp`, so the surface Green's function is computed for the
small cell at each :math:`\mathbf k_\perp` and then unfolded into the
real-space supercell (:func:`~qtpyt.surface.unfold.bloch_unfold`). This is
much cheaper than decimating the supercell directly, and gives a
self-energy for a device whose transverse size matches the supercell.
"""
from __future__ import annotations

from itertools import chain
from typing import Any, Optional, Sequence, Tuple

from numpy.typing import NDArray

import numpy as np
from numba import njit, prange
from qtpyt import parallel
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.base.leads import LeadSelfEnergy, SanchoRubio
from qtpyt.base.selfenergy import BaseSelfEnergy
from qtpyt.basis import Basis
from qtpyt.surface.expand import seminf_expand, tritoeplz_expand

# from ._recursive import get_G
from .unfold import bloch_unfold


class PrincipalLayer(BaseGreenFunction):
    r"""Surface Green's function of a semi-infinite lead that is periodic transversally.

    For each transverse k-point the surface Green's function of the small
    principal layer is found by Sancho–Rubio decimation
    (:class:`~qtpyt.base.leads.SanchoRubio`), and the results are unfolded
    into the real-space supercell of ``Nr`` transverse repetitions.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Transverse k-points, in units of the reciprocal lattice vectors.
    hs_kii : tuple of ndarray, shape (n_k, n, n)
        ``(H_k, S_k)`` of the principal layer at each k-point.
    hs_kij : tuple of ndarray, shape (n_k, n, n)
        ``(H_k, S_k)`` coupling from one principal layer to the next, at
        each k-point.
    Nr : tuple of three int, optional
        Transverse repetitions of the small cell in the supercell. Defaults
        to the number of distinct k-point coordinates along each axis. A
        k-mesh denser than ``Nr`` samples the transverse zone more finely
        than the supercell size alone would.
    eta : float, default 1e-5
        Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`.

    Notes
    -----
    Unfolding works for supercells repeated along at most two directions.
    """

    def __init__(self, kpts: NDArray[np.float64], hs_kii: Tuple[NDArray, NDArray], hs_kij: Tuple[NDArray, NDArray], Nr: Optional[Sequence[int]] = None, eta: float = 1e-5) -> None:

        self.kpts = kpts
        self.Nr = Nr or self.N_k

        assert (
            self.kpts.shape[1] == 3
        ), "Invalid k-points. Must have (x,y,z) components."
        assert len(self.Nr) == 3, "Invalid # of transverse PLs. Must be (Nx, Ny, Nz)."

        H, S = (self.bloch_unfold(m) for m in hs_kii)

        self.PL = [
            SanchoRubio((hs_kii[0][k], hs_kii[1][k]), (hs_kij[0][k], hs_kij[1][k]), eta)
            for k in range(self.Nk)
        ]

        super().__init__(H, S, eta)

        self.G_kii = np.empty((kpts.shape[0],) + self.PL[0].shape, complex)
        self.G = np.empty(self.shape, complex)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for PL in self.PL:
                setattr(PL, name, value)
        super().__setattr__(name, value)

    @property
    def N_k(self) -> NDArray[np.int_]:
        """Number of distinct k-point coordinates along each axis."""
        return np.array([len(set(self.kpts[:, i])) for i in range(3)])

    @property
    def Nk(self) -> int:
        """Number of transverse k-points."""
        return self.kpts.shape[0]

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Surface Green's function :math:`g_s(E)` of the lead in the real-space supercell.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        if energy != self.energy:
            self.energy = energy
            for k in range(self.Nk):
                self.G_kii[k] = self.PL[k].retarded(energy)
            self.bloch_unfold(self.G_kii, self.G)
        return self.G

    def bloch_unfold(self, A: NDArray[np.complex128], out: Optional[NDArray[np.complex128]] = None) -> NDArray[np.complex128]:
        """Real-space supercell matrix from matrices at the transverse k-points.

        Parameters
        ----------
        A : ndarray, shape (n_k, n, n)
            Matrices at each k-point.
        out : ndarray, optional
            Array for the result.

        See Also
        --------
        qtpyt.surface.unfold.bloch_unfold
        """
        return bloch_unfold(A, self.kpts, self.Nr, out)


class PrincipalSelfEnergy(BaseSelfEnergy):
    r"""Self-energy of a transversally periodic lead on the device.

    Like :class:`~qtpyt.base.leads.LeadSelfEnergy`, with the lead's surface
    Green's function from :class:`PrincipalLayer`. The coupling to the
    device is the principal-layer coupling, unfolded to the supercell, so
    the device edge must be a copy of a supercell principal layer with
    atoms in the same order.

    Parameters
    ----------
    kpts, hs_kii, Nr, eta
        See :class:`PrincipalLayer`.
    hs_kij : tuple of ndarray, shape (n_k, n, n)
        Coupling from a principal layer to the next one on its right, for
        both leads; transposed internally for ``id="right"``.
    id : {'left', 'right'}, default 'left'
        Which side of the device the lead is on.
    """

    ids = ["left", "right"]

    def __init__(self, kpts: NDArray[np.float64], hs_kii: Tuple[NDArray, NDArray], hs_kij: Tuple[NDArray, NDArray], Nr: Optional[Sequence[int]] = None, id: str = "left", eta: float = 1e-5) -> None:

        assert id in self.ids, f"Invalid id. Choose between {self.ids}"

        if id == "right":
            h_kij = hs_kij[0].swapaxes(1, 2).conj()
            s_kij = hs_kij[1].swapaxes(1, 2).conj()
            hs_kij = (h_kij, s_kij)

        self.gf = PrincipalLayer(kpts, hs_kii, hs_kij, Nr, eta)

        hs_ij = tuple(bloch_unfold(m, self.gf.kpts, self.gf.Nr) for m in hs_kij)

        super().__init__(self.gf, hs_ij)


class SurfaceGreenFunction(BaseGreenFunction):
    r"""Green's function of a slab of ``Nd`` principal layers inside an infinite crystal.

    The crystal continues on both sides along the transport direction and
    is periodic transversally. The slab Green's function equals that of the
    perfect crystal restricted to the slab, so a defect or adsorbate placed
    in the slab can be embedded through :class:`SurfaceSelfEnergy`.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Transverse k-points.
    hs_kii : tuple of ndarray, shape (n_k, n, n)
        ``(H_k, S_k)`` of the principal layer.
    hs_kij : tuple of ndarray, shape (n_k, n, n)
        ``(H_k, S_k)`` coupling to the next principal layer.
    Nr : tuple of three int
        Repetitions of the principal layer: along the transport direction
        this is the slab thickness ``Nd``, along the others the transverse
        supercell.
    direction : {'x', 'y', 'z'}, default 'x'
        Transport direction.
    eta : float, default 1e-5
        Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`.

    Notes
    -----
    At each k-point the slab is attached to a left and a right
    semi-infinite lead of the same crystal. The first diagonal block is
    :math:`G_{00} = (zS - H - \Sigma_L - \Sigma_R)^{-1}`, and the rest of the
    slab follows from translation invariance, :math:`G_{ij}` depending only
    on :math:`i - j`.
    """

    def __init__(self, kpts: NDArray[np.float64], hs_kii: Tuple[NDArray, NDArray], hs_kij: Tuple[NDArray, NDArray], Nr: Sequence[int], direction: str = "x", eta: float = 1e-5) -> None:

        self.Nr = Nr
        self.kpts = kpts

        self.d = "xyz".index(direction)
        self.Nd = Nr[self.d]  # num. PLs along transport
        self.Nr_t = tuple(
            n if i != self.d else 1 for i, n in enumerate(Nr)
        )  # num. PLs along transverse

        hs_kji = tuple(m.swapaxes(1, 2).conj() for m in hs_kij)

        sel = [
            LeadSelfEnergy((hs_kii[0][k], hs_kii[1][k]), (hs_kij[0][k], hs_kij[1][k]), eta=eta)
            for k in range(self.Nk)
        ]
        
        ser = [
            LeadSelfEnergy((hs_kii[0][k], hs_kii[1][k]), (hs_kji[0][k], hs_kji[1][k]), eta=eta)
            for k in range(self.Nk)
        ]

        self.selfenergies = [sel, ser]
        
        self.work_kMM = np.zeros(
            (self.Nk,) + 2 * (hs_kii[0].shape[1] * self.Nd,), complex
        )  # Internal work array

        H, S = (
            self.unfold_periodic(A, B, C) for A, B, C in zip(hs_kii, hs_kij, hs_kji)
        )

        super().__init__(H, S, eta)

        self.G = np.zeros(self.shape, complex)
        # Store reference for quick leads extraction
        self.hs_kii = hs_kii
        self.hs_kij = hs_kij
        self.hs_kji = hs_kji

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for PL in chain(*self.selfenergies):
                setattr(PL, name, value)
        super().__setattr__(name, value)

    @property
    def Nk(self) -> int:
        """Number of transverse k-points."""
        return self.kpts.shape[0]

    def bloch_unfold(self, A: NDArray[np.complex128], out: Optional[NDArray[np.complex128]] = None) -> NDArray[np.complex128]:
        """Real-space transverse supercell matrix from matrices at the k-points.

        Parameters
        ----------
        A : ndarray, shape (n_k, m, m)
            Matrices at each k-point.
        out : ndarray, optional
            Array for the result.
        """
        return bloch_unfold(A, self.kpts, self.Nr_t, out)

    def unfold_periodic(self, A: NDArray, B: NDArray, C: NDArray, out: Optional[NDArray] = None) -> NDArray[np.complex128]:
        """Matrix of the slab from the principal-layer blocks, in the real-space supercell.

        Parameters
        ----------
        A, B, C : ndarray, shape (n_k, n, n)
            Diagonal, upper and lower principal-layer blocks at each k-point.
        out : ndarray, optional
            Array for the result.

        See Also
        --------
        qtpyt.surface.expand.tritoeplz_expand
        """
        for k in range(self.Nk):
            tritoeplz_expand(self.Nd, A[k], B[k], C[k], self.work_kMM[k])
        return self.bloch_unfold(self.work_kMM, out)

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Green's function :math:`G^r(E)` of the slab inside the infinite crystal.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        if energy != self.energy:
            self.energy = energy
            z = self.z(energy)
            for k in range(self.Nk):
                seminf_expand(
                    self.Nd,
                    (self.hs_kii[0][k], self.hs_kii[1][k]),
                    (self.hs_kij[0][k], self.hs_kij[1][k]),
                    self.selfenergies[0][k].retarded(energy),
                    self.selfenergies[1][k].retarded(energy),
                    z,
                    (self.hs_kji[0][k], self.hs_kji[1][k]),
                    self.work_kMM[k],
                )
            self.bloch_unfold(self.work_kMM, self.G)
        return self.G


@njit(
    ["(c16,c16[:,::1],c16[:,::1],c16[:,::1],c16[:,::1],i8[::1])"], parallel=True,
)
def _fill_sigma_subset(z, H, S, Ginv, Sigma, indices):
    n = len(indices)
    for i in prange(n):
        for j in range(n):
            Sigma[i, j] = (
                z * S[indices[i], indices[j]]
                - H[indices[i], indices[j]]
                - Ginv[indices[i], indices[j]]
            )


@njit(["(c16,c16[:,::1],c16[:,::1],c16[:,::1],c16[:,::1])"], parallel=True)
def _fill_sigma_full(z, H, S, Ginv, Sigma):
    n, m = H.shape
    for i in prange(n):
        for j in prange(m):
            Sigma[i, j] = z * S[i, j] - H[i, j] - Ginv[i, j]


def fill_sigma(z: complex, H: NDArray, S: NDArray, Ginv: NDArray, Sigma: Optional[NDArray] = None, indices: Optional[NDArray[np.int64]] = None) -> NDArray[np.complex128]:
    r"""Self-energy from the Dyson equation, :math:`\Sigma = zS - H - G^{-1}`.

    Given the Green's function of a region embedded in its surroundings,
    this recovers the self-energy that the surroundings exert on it.

    Parameters
    ----------
    z : complex
        Complex energy :math:`E + i\eta`.
    H, S : ndarray, shape (n, n)
        Hamiltonian and overlap of the region, complex.
    Ginv : ndarray, shape (n, n)
        Inverse of the embedded Green's function.
    Sigma : ndarray, optional
        Array for the result.
    indices : ndarray of int64, optional
        Restrict the result to these orbitals.

    Returns
    -------
    Sigma : ndarray
    """
    if Sigma is None:
        Sigma = np.empty_like(Ginv)
    if indices is None:
        _fill_sigma_full(z, H, S, Ginv, Sigma)
    else:
        _fill_sigma_subset(z, H, S, Ginv, Sigma, indices)
    return Sigma


class SurfaceSelfEnergy(BaseSelfEnergy):
    r"""Self-energy of an infinite crystal on a slab of its own principal layers.

    Embeds a region of ``Nd`` principal layers into the surrounding perfect
    crystal: :math:`\Sigma = zS - H - G^{-1}` with :math:`G` from
    :class:`SurfaceGreenFunction`. Replace the slab Hamiltonian by that of
    a defective slab and add this self-energy to study a defect in an
    otherwise perfect crystal.

    Parameters
    ----------
    kpts, hs_kii, hs_kij, Nr, direction, eta
        See :class:`SurfaceGreenFunction`.
    indices : sequence of int, optional
        Keep only these orbitals of the slab, see :attr:`indices`.
    """

    def __init__(
        self, kpts: NDArray[np.float64], hs_kii: Tuple[NDArray, NDArray], hs_kij: Tuple[NDArray, NDArray], Nr: Sequence[int], indices: Optional[Sequence[int]] = None, direction: str = "x", eta: float = 1e-5
    ) -> None:

        self.gf = SurfaceGreenFunction(kpts, hs_kii, hs_kij, Nr, direction, eta)
        super().__init__(self.gf, (np.empty((0, 0), complex), None))

        self.indices = indices

    @property
    def indices(self) -> Optional[Sequence[int]]:
        """Orbitals of the slab the self-energy is restricted to, or ``None`` for all.

        Use it when the embedded structure lacks some atoms of the perfect
        slab, for example at a vacancy.
        """
        return self._indices

    @indices.setter
    def indices(self, indices):
        if indices is None:
            shape = self.gf.shape
        else:
            shape = 2 * (len(indices),)
        self.Sigma = np.empty(shape, complex)
        self._indices = indices

    def take(self, indices: Optional[Sequence[int]]) -> None:
        """Restrict the self-energy to a subset of orbitals, see :attr:`indices`.

        Parameters
        ----------
        indices : sequence of int or None
        """
        self.indices = indices

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded self-energy :math:`\Sigma(E)` on the slab orbitals in :attr:`indices`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        if energy != self.energy:
            self.energy = energy
            Ginv = np.linalg.inv(self.gf.retarded(energy))
            fill_sigma(self.z(energy), self.H, self.S, Ginv, self.Sigma, self.indices)
        return self.Sigma


def extract_leads_from_surface(sf: SurfaceSelfEnergy, bpl: Basis, bcc: Basis) -> Tuple[PrincipalSelfEnergy, PrincipalSelfEnergy]:
    """Left and right lead self-energies of a surface self-energy, coupled to a device.

    Reuses the decimated leads of ``sf`` and reorders the couplings so that
    they attach to the matching atoms of the device.

    Parameters
    ----------
    sf : SurfaceSelfEnergy
        Surface self-energy whose leads are extracted.
    bpl : Basis
        Basis of one principal layer (one transverse cell).
    bcc : Basis
        Basis of the device. Its atoms at the left and right edges must
        coincide in position with the first and last principal layers of
        the slab.

    Returns
    -------
    SL, SR : PrincipalSelfEnergy
        Left and right lead self-energies on the full device orbital space.
    """
    from scipy.spatial.distance import cdist

    gf = sf.gf
    SL = PrincipalSelfEnergy(gf.kpts, gf.hs_kii, gf.hs_kij, gf.Nr_t, eta=gf.eta)
    SR = PrincipalSelfEnergy(gf.kpts, gf.hs_kii, gf.hs_kji, gf.Nr_t, eta=gf.eta)

    # Override principal layers.
    SL.gf.PL = [se.gf for se in gf.selfenergies[0]]
    SR.gf.PL = [se.gf for se in gf.selfenergies[1]]

    # Find position of left/right atoms in surface.
    bsc = bpl.repeat(gf.Nr_t)
    sc_pos = bsc.atoms.positions
    l_in_c = cdist(sc_pos, bcc.atoms.positions).argmin(1)
    # Shift to the right.
    sc_pos[:, gf.d] += (gf.Nd - 1) * bpl.atoms.cell.diagonal()[gf.d]
    r_in_c = cdist(sc_pos, bcc.atoms.positions).argmin(1)
    # Rearrange couplings.
    for in_c, selfenergy in zip([l_in_c, r_in_c], [SL, SR]):
        perm = bcc[in_c].get_indices()
        h_ij = np.zeros((bsc.nao, bcc.nao), complex)
        s_ij = np.zeros((bsc.nao, bcc.nao), complex)
        h_ij[:, perm] = selfenergy.h_ij
        s_ij[:, perm] = selfenergy.s_ij
        selfenergy.h_ij = h_ij
        selfenergy.s_ij = s_ij
        selfenergy.Sigma = np.zeros(2 * (bcc.nao,), complex)
    return SL, SR

