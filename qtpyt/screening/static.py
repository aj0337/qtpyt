r"""Static (energy-independent) self-energies: Hartree and Fock (exchange)."""
from __future__ import annotations

from typing import Any, Optional

from numpy.typing import NDArray

from qtpyt import xp
from qtpyt.base.selfenergy import ConstSelfEnergy
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.products import exchange_product, hartree_product


class Fock(ConstSelfEnergy):
    r"""Exchange (Fock) self-energy :math:`\Sigma^x_{ij} = -\sum_{kl} D_{kl} V_{ik,jl}`.

    The bare exchange interaction of each electron with the others of the
    same spin, computed from the density matrix of the region.

    Parameters
    ----------
    gf : DistGreenFunction
        Green's function whose density matrix enters.
    V_qq : ndarray, shape (n_q, n_q)
        Bare Coulomb matrix in the pair-orbital basis.
    U_pq : ndarray, shape (n_orb**2, n_q), optional
        Rotation from all orbital pairs to optimized pair orbitals.
    """

    def __init__(self, gf: DistGreenFunction, V_qq: NDArray, U_pq: Optional[NDArray] = None) -> None:
        self.gf = gf
        self.V_qq = V_qq
        self.U_pq = U_pq if U_pq is None else xp.asarray(U_pq)
        super().__init__(shape=2 * (gf.no,))

    def get_work_array(self) -> Optional[NDArray]:
        """Scratch array for the pair-orbital rotation, or ``None`` without one."""
        if self.U_pq is None:
            return None
        return xp.empty_like(self.U_pq)

    def update(self, D: Optional[NDArray] = None) -> None:
        """Recompute the exchange self-energy from a density matrix.

        Parameters
        ----------
        D : ndarray, shape (n_orb, n_orb), optional
            Density matrix for one spin; that of ``gf`` if omitted.
        """
        if D is None:
            assert self.gf.domain == "e"
            D = self.gf.get_density_matrix()

        pre = -1.0  # -1. from definition

        work1 = self.get_work_array()
        work2 = self.get_work_array()

        # for g, s in zip(Gl, self.Sigma):
        exchange_product(
            D,
            self.V_qq,
            pre=pre,
            out=self.Sigma,
            rot=self.U_pq,
            # rotd=self.U_qp,
            work1=work1,
            work2=work2,
        )


class Hartree(ConstSelfEnergy):
    r"""Change of the Hartree potential relative to a reference density.

    The Hartree self-energy :math:`\Sigma^H_{ij} = 2\sum_{kl} D_{kl} V_{ij,kl}`
    (factor 2 for spin) is the electrostatic potential of the electron
    density. The reference value from :meth:`initialize` is subtracted, so
    only the change since then acts, because the starting Hamiltonian (for
    example from DFT) already contains the Hartree potential of the
    reference density.

    Parameters
    ----------
    gf : DistGreenFunction
        Green's function whose density matrix enters.
    V_qq : ndarray, shape (n_q, n_q)
        Bare Coulomb matrix in the pair-orbital basis.
    U_pq : ndarray, shape (n_orb**2, n_q), optional
        Rotation from all orbital pairs to optimized pair orbitals.
    """

    def __init__(self, gf: DistGreenFunction, V_qq: NDArray, U_pq: Optional[NDArray] = None) -> None:
        self.gf = gf
        self.V_qq = V_qq
        self.U_pq = U_pq if U_pq is None else xp.asarray(U_pq)
        self.Sigma0 = xp.zeros(2 * (gf.no,), complex)
        super().__init__(shape=2 * (gf.no,))

    def get_work_array(self) -> Optional[NDArray]:
        """Scratch array for the pair-orbital rotation, or ``None`` without one."""
        if self.U_pq is None:
            return None
        return xp.empty(self.U_pq.shape[1], complex)

    def initialize(self, D: Optional[NDArray] = None) -> None:
        """Store the Hartree self-energy of the reference density.

        Parameters
        ----------
        D : ndarray, optional
            Reference density matrix for one spin; that of ``gf`` if omitted.
        """
        self.hartree_product(D, self.Sigma0)

    def hartree_product(self, D: Optional[NDArray] = None, Sigma: Optional[NDArray] = None) -> None:
        r"""Full Hartree self-energy :math:`2\sum_{kl} D_{kl} V_{ij,kl}` of a density matrix.

        Parameters
        ----------
        D : ndarray, optional
            Density matrix for one spin; that of ``gf`` if omitted.
        Sigma : ndarray, optional
            Array for the result; the stored self-energy if omitted.
        """
        if D is None:
            assert self.gf.domain == "e"
            D = self.gf.get_density_matrix()

        if Sigma is None:
            Sigma = self.Sigma

        pre = 2.0  # 2. from spin

        work1 = self.get_work_array()
        work2 = self.get_work_array()

        # for g, s in zip(Gl, self.Sigma if Sigma is None else Sigma):
        hartree_product(
            D,
            self.V_qq,
            pre=pre,
            out=Sigma,
            rot=self.U_pq,
            # rotd=self.U_qp,
            work1=work1,
            work2=work2,
        )

    def update(self, D: NDArray) -> None:
        """Set the self-energy to the Hartree change from the reference density to ``D``.

        Parameters
        ----------
        D : ndarray, shape (n_orb, n_orb)
            Density matrix for one spin.
        """
        self.hartree_product(D, self.Sigma)
        self.Sigma -= self.Sigma0
