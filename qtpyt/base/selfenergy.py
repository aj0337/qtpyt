r"""Self-energies: the effect of an eliminated region folded onto the region of interest.

A self-energy :math:`\Sigma(E)` replaces an attached system (a lead, a
neighbouring block of the device, an interaction) by an energy-dependent
correction to the Hamiltonian of the region it couples to. Its Hermitian
part shifts levels, its anti-Hermitian part broadens them. The objects in
this module are passed to :class:`~qtpyt.base.greenfunction.GreenFunction`
as ``(index, self-energy)`` pairs.
"""
from __future__ import annotations

from types import MethodType
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qtpyt import xp
from qtpyt.base._kernels import dagger
from qtpyt.base.greenfunction import BaseGreenFunction, GreenFunction


class BaseSelfEnergy:
    r"""Self-energy of a region that is coupled to the region of interest.

    An electron in the region of interest can hop into the attached region
    :math:`i`, propagate there, and hop back. Summing all such excursions
    gives the self-energy on the region of interest :math:`j`.

    Parameters
    ----------
    gf : BaseGreenFunction
        Green's function :math:`g_i` of the attached region, for a lead its
        surface Green's function.
    hs_ij : tuple of ndarray
        ``(H_ij, S_ij)``, Hamiltonian and overlap coupling between the
        attached region (rows, :math:`n_i`) and the region of interest
        (columns, :math:`n_j`).

    Notes
    -----
    With :math:`\tau_{ij} = z S_{ij} - H_{ij}` and
    :math:`\tau_{ji} = z S_{ij}^\dagger - H_{ij}^\dagger`,

    .. math::

        \Sigma^r(E) = \tau_{ji}\, g_i^r(E)\, \tau_{ij},

    a matrix of shape :math:`(n_j, n_j)`. The overlap enters because in a
    non-orthogonal basis the coupling is :math:`zS - H`, not :math:`H`
    alone. The result is kept for the last energy. The broadening
    :math:`\eta` is that of ``gf``.
    """

    def __init__(self, gf: BaseGreenFunction, hs_ij: Tuple[NDArray, NDArray]) -> None:
        self.gf = gf
        self.h_ij, self.s_ij = hs_ij
        self.Sigma = np.empty(self.shape, complex)
        self.energy = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape ``(n_j, n_j)`` of the self-energy matrix."""
        return 2 * (self.h_ij.shape[1],)

    @property
    def H(self) -> NDArray:
        """Hamiltonian of the attached region."""
        return self.gf.H

    @property
    def S(self) -> NDArray:
        """Overlap of the attached region."""
        return self.gf.S

    @property
    def eta(self) -> float:
        """Broadening of the attached region's Green's function."""
        return self.gf.eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.gf.eta = eta

    def z(self, energy: float) -> complex:
        """Complex energy of the attached region, see :meth:`BaseGreenFunction.z`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        return self.gf.z(energy)

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded self-energy :math:`\Sigma^r(E)` on the region of interest.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        Sigma : ndarray, shape (n_j, n_j)
            :math:`\Sigma^r(E)`.
        """
        if energy != self.energy:
            z = self.z(energy)
            tau_ij = z * self.s_ij - self.h_ij
            G = self.gf.retarded(energy)
            tau_ji = z * self.s_ij.T.conj() - self.h_ij.T.conj()
            tau_ji.dot(G).dot(tau_ij, out=self.Sigma)

        return self.Sigma

    def advanced(self, energy: float) -> NDArray[np.complex128]:
        r"""Advanced self-energy :math:`\Sigma^a(E) = [\Sigma^r(E)]^\dagger`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        return dagger(self.retarded(energy))

    def get_lambda(self, energy: float) -> NDArray[np.complex128]:
        r"""Broadening matrix :math:`\Gamma(E) = i[\Sigma^r(E) - \Sigma^a(E)]`.

        :math:`\Gamma` measures how strongly the region of interest leaks
        into the attached region: a level with broadening :math:`\Gamma`
        has width :math:`\Gamma` and an electron stays in it for a time of
        order :math:`\hbar/\Gamma`. It is Hermitian and positive
        semidefinite. The code calls it lambda; the literature writes
        :math:`\Gamma`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        Gamma : ndarray, shape (n_j, n_j)
            :math:`\Gamma(E)`.
        """
        Sigma = self.retarded(energy)
        return 1.0j * (Sigma - Sigma.T.conj())


class SelfEnergy(BaseSelfEnergy):
    r"""Self-energy of a finite block that may itself be coupled to other environments.

    Builds the :class:`~qtpyt.base.greenfunction.GreenFunction` of the
    attached block, including its own self-energies, and folds it onto the
    region of interest. Chaining these objects eliminates a device block by
    block.

    Parameters
    ----------
    hs_ii : tuple of array_like
        ``(H_ii, S_ii)`` of the attached block.
    hs_ij : tuple of array_like
        ``(H, S)`` coupling from the block on the left to the block on the
        right. For ``id="left"`` the attached block is on the left; for
        ``id="right"`` it is on the right and the coupling is transposed
        internally.
    selfenergies : sequence of (index, self-energy) pairs, default []
        Self-energies attached to the block, as in
        :class:`~qtpyt.base.greenfunction.GreenFunction`.
    idxleads : sequence of int, default [0, 1]
        Which entries of ``selfenergies`` are leads.
    id : {'left', 'right'}, default 'left'
        Side of the region of interest on which the attached block sits.
    eta : float, default 1e-5
        Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`.
    mu : float, default 0.0
        Chemical potential of the attached block.
    kt : float, default 0.0
        Thermal energy of the attached block.
    """

    ids = ["left", "right"]

    def __init__(
        self,
        hs_ii: Tuple[ArrayLike, ArrayLike],
        hs_ij: Tuple[ArrayLike, ArrayLike],
        selfenergies: List[Tuple[Any, Any]] = [],
        idxleads: List[int] = [0, 1],
        id: str = "left",
        eta: float = 1e-5,
        mu: float = 0.0,
        kt: float = 0.0,
    ) -> None:

        assert id in self.ids, f"Invalid id. Choose between {self.ids}"

        if id == "right":
            h_ij = hs_ij[0].swapaxes(0, 1).conj()
            s_ij = hs_ij[1].swapaxes(0, 1).conj()
            hs_ij = (h_ij, s_ij)

        gf = GreenFunction(hs_ii[0], hs_ii[1], selfenergies, idxleads, eta, mu, kt)

        super().__init__(gf, (xp.asarray(m) for m in hs_ij))


class ConstSelfEnergy:
    r"""Energy-independent self-energy :math:`\Sigma(E) = \Sigma_0`.

    Use it for a static correction to the Hamiltonian (for example a
    mean-field or double-counting term) or for a fixed broadening.

    Parameters
    ----------
    Sigma : ndarray, optional
        The constant matrix :math:`\Sigma_0`. If omitted, a zero matrix of
        shape ``shape`` is used.
    shape : tuple of int, optional
        Shape of the zero matrix when ``Sigma`` is omitted.
    """

    def __init__(self, Sigma: Optional[NDArray] = None, shape: Optional[Tuple[int, ...]] = None) -> None:
        if Sigma is None:
            Sigma = xp.zeros(shape, complex)
        self.Sigma = Sigma

    def retarded(self, energy: float) -> NDArray:
        r"""The constant :math:`\Sigma_0`, for any energy.

        Parameters
        ----------
        energy : float
            Ignored.
        """
        return self.Sigma


class DataSelfEnergy(ConstSelfEnergy):
    r"""Self-energy tabulated on an energy grid, for example from a GW or DMFT run.

    Parameters
    ----------
    energies : ndarray, shape (n_E,)
        Increasing energy grid on which :math:`\Sigma` is given, same unit
        as the Hamiltonian.
    Sigma : ndarray, shape (n_E, n, n)
        :math:`\Sigma(E_k)` at each grid energy.

    Notes
    -----
    No interpolation is done: :meth:`retarded` returns the tabulated value
    at the first grid energy :math:`E_k \ge E`. Evaluate on the same grid
    the data was produced on. Energies above the last grid point raise an
    ``IndexError``.
    """

    def __init__(self, energies: NDArray[np.float64], Sigma: NDArray) -> None:
        self.energies = energies
        self.energy = None
        super().__init__(Sigma=Sigma)

    def retarded(self, energy: float) -> NDArray:
        r"""Tabulated :math:`\Sigma(E_k)` at the first grid energy :math:`E_k \ge E`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        return self.Sigma[np.searchsorted(self.energies, energy)]


class StackSelfEnergy(ConstSelfEnergy):
    r"""Sum of several self-energies acting on the same block of orbitals.

    For example a lead and an interaction self-energy on the same
    orbitals, which :class:`~qtpyt.base.greenfunction.GreenFunction`
    would otherwise need as two entries with the same index.

    Parameters
    ----------
    selfenergies : list
        Self-energy objects, or plain arrays that are wrapped as
        :class:`ConstSelfEnergy` (the list is modified in place). All must
        have the same matrix shape, which is read from an object with a
        ``Sigma`` attribute.

    Raises
    ------
    RuntimeError
        If no self-energy in the list exposes a matrix shape.
    """

    def __init__(self, selfenergies: list) -> None:
        for selfenergy in selfenergies:
            if hasattr(selfenergy, "Sigma"):
                shape = selfenergy.Sigma.shape
            if shape is None:
                raise RuntimeError("Cannot retreive shape of selfenergy matrix.")
        if len(shape) > 2:  # DataSelfEnergy
            shape = shape[1:]
        super().__init__(shape=shape)
        for e, selfenergy in enumerate(list(selfenergies)):
            if isinstance(selfenergy, np.ndarray):
                # substitute with ConstSelfenergy
                selfenergies.insert(e, ConstSelfEnergy(Sigma=selfenergies.pop(e)))
        self.selfenergies = selfenergies

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Total retarded self-energy :math:`\sum_k \Sigma^r_k(E)`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        self.Sigma[:] = 0.0
        for selfenergy in self.selfenergies:
            self.Sigma += selfenergy.retarded(energy)
        return self.Sigma

    def lesser(self, energy: float) -> NDArray[np.complex128]:
        r"""Total lesser self-energy :math:`\sum_k \Sigma^<_k(E)`.

        Only members that define a lesser component contribute.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        self.Sigma[:] = 0.0
        for selfenergy in self.selfenergies:
            if hasattr(selfenergy, "lesser"):
                self.Sigma += selfenergy.lesser(energy)
        return self.Sigma


def ZeroFilter(selfenergy: Union[BaseSelfEnergy, ConstSelfEnergy], func: Callable) -> Union[BaseSelfEnergy, ConstSelfEnergy]:
    r"""Switch a self-energy off outside a chosen set of energies.

    Use it to restrict an interaction self-energy to an energy window, for
    example to test which part of the spectrum a correction acts on.

    Parameters
    ----------
    selfenergy : BaseSelfEnergy or ConstSelfEnergy
        The self-energy to filter. It is modified in place.
    func : callable
        ``func(energy) -> bool``. Where it returns ``False`` the retarded
        self-energy is zero.

    Returns
    -------
    selfenergy : BaseSelfEnergy or ConstSelfEnergy
        The same object. Its :meth:`retarded` gives :math:`\Sigma^r(E)`
        where ``func(E)`` is true and zero elsewhere.
        Its other components are unchanged.
    """

    Sigma = np.zeros_like(selfenergy.Sigma)

    def filter(self, energy):
        if func(energy):
            out = self._retarded(energy)
            return out
        else:
            return Sigma

    selfenergy._retarded = getattr(selfenergy, "retarded")
    selfenergy.retarded = MethodType(filter, selfenergy)
    return selfenergy
