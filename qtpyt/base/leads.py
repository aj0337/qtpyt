r"""Semi-infinite leads: surface Green's functions and lead self-energies.

A lead is a periodic electrode that extends to infinity. It is described
by one principal layer (a slab thick enough that only neighbouring layers
couple) and the coupling between consecutive layers. Its effect on the
device is the self-energy :math:`\Sigma_\alpha(E)` of
:class:`LeadSelfEnergy`.
"""
from __future__ import annotations

import functools
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lu_factor, lu_solve

from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.base.selfenergy import BaseSelfEnergy
from qtpyt.screening.tools import fermidistribution


class SanchoRubio(BaseGreenFunction):
    r"""Surface Green's function of a semi-infinite lead by Sancho–Rubio decimation.

    The surface Green's function :math:`g_s(E)` describes an electron on
    the outermost principal layer of a lead that continues forever behind
    it. It is obtained by decimation [1]_: each step eliminates every second
    layer and renormalizes the couplings, so after :math:`n` steps the
    surface feels the effect of :math:`2^n` layers. Inside a band the
    lead's continuum of states appears as a finite imaginary part of
    :math:`g_s`.

    Parameters
    ----------
    hs_ii : tuple of ndarray
        ``(H_ii, S_ii)`` of one principal layer, shape ``(n_i, n_i)``.
    hs_ij : tuple of ndarray
        ``(H_ij, S_ij)`` coupling from a principal layer to the next one.
        The surface obtained is the one whose missing neighbour is on the
        ``j`` side.
    eta : float, default 1e-5
        Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`.
        Decimation converges more slowly as :math:`\eta \to 0`.
    mu : float, default 0.0
        Chemical potential of the lead.
    kt : float, default 0.0
        Thermal energy of the lead.
    bias : float, default 0.0
        Rigid shift :math:`V` of the lead, same energy unit as the
        Hamiltonian. The bands move with :math:`z = E - V + i\eta`, and the
        occupation is :math:`f(E - \mu + V)`.

    Attributes
    ----------
    conv : float
        Convergence threshold, 1e-8. Decimation stops when the largest
        element of the renormalized inter-layer coupling falls below it.

    References
    ----------
    .. [1] M. P. Lopez Sancho, J. M. Lopez Sancho and J. Rubio,
       J. Phys. F: Met. Phys. 15, 851 (1985).
    """

    conv = 1e-8  # Convergence criteria for surface Green function

    def __init__(
        self,
        hs_ii: Tuple[NDArray, NDArray],
        hs_ij: Tuple[NDArray, NDArray],
        eta: float = 1e-5,
        mu: float = 0.0,
        kt: float = 0.0,
        bias: float = 0.0,
    ) -> None:
        self.h_ij, self.s_ij = hs_ij  # Coupling to next cell.
        super().__init__(hs_ii[0], hs_ii[1], eta=eta, mu=mu, kt=kt)
        self.bias = bias

    def z(self, energy: float) -> complex:
        r"""Complex energy :math:`z = E - V + i\eta`, shifted by the lead bias.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        return energy - self.bias + 1.0j * self.eta

    def fermi(self, energy: float) -> float:
        r"""Occupation :math:`f(E - \mu + V)` of the lead.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        return fermidistribution(energy - self.mu + self.bias, self.kt)

    def get_Ginv(self, energy: float) -> NDArray[np.complex128]:
        r"""Inverse surface Green's function :math:`g_s^{-1}(E)` after decimation.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        Ginv : ndarray, shape (n_i, n_i)
            Renormalized :math:`zS_{ii} - H_{ii}` of the surface layer.
        """
        z = self.z(energy)

        v_00 = z * self.S.T.conj() - self.H.T.conj()
        v_11 = v_00.copy()
        v_10 = z * self.s_ij - self.h_ij
        v_01 = z * self.s_ij.T.conj() - self.h_ij.T.conj()

        delta = self.conv + 1
        while delta > self.conv:
            lu, piv = lu_factor(v_11, check_finite=False)
            a = lu_solve((lu, piv), v_01, check_finite=False)
            b = lu_solve((lu, piv), v_10, check_finite=False)
            # a = np.linalg.solve(v_11, v_01)
            # b = np.linalg.solve(v_11, v_10)
            v_01_dot_b = np.dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= np.dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -np.dot(v_01, a)
            v_10 = -np.dot(v_10, b)
            delta = abs(v_01).max()
        return v_00

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Surface Green's function :math:`g_s(E)` of the semi-infinite lead.

        The result is kept for the last energy.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        g_s : ndarray, shape (n_i, n_i)
            :math:`g_s(E)` on the surface principal layer.
        """
        if energy != self.energy:
            self.energy = energy
            self.Gr = np.linalg.inv(self.get_Ginv(energy))
        return self.Gr


class LeadSelfEnergy(BaseSelfEnergy):
    r"""Self-energy of a semi-infinite lead on the scattering region.

    Electrons in the device can escape into the lead and never return, so
    the lead acts as a sink and source of electrons at its own chemical
    potential and temperature. :math:`\Gamma` from :meth:`get_lambda` sets
    how fast this happens.

    Parameters
    ----------
    hs_ii : tuple of ndarray
        ``(H_ii, S_ii)`` of one principal layer of the lead.
    hs_ij : tuple of ndarray
        ``(H_ij, S_ij)`` coupling from a principal layer to the next one on
        its right. Pass the same orientation for both leads; for
        ``id="right"`` it is transposed internally.
    hs_im : tuple of ndarray, optional
        ``(H, S)`` coupling between the lead's surface layer and the device,
        from the block on the left to the block on the right. Defaults to
        ``hs_ij``, which is correct when the device edge is itself a copy of
        a principal layer.
    nbf_m : int, optional
        Number of orbitals :math:`n_m` in the device. If given, the
        coupling ``hs_im`` must be square ``(n_i, n_i)`` and is placed on the
        first :math:`n_i` device orbitals for the left lead and on the last
        :math:`n_i` for the right lead, so that :math:`\Sigma` has shape
        ``(n_m, n_m)``.
    id : {'left', 'right'}, default 'left'
        Which side of the device the lead is on.
    eta : float, default 1e-5
        Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`.
    mu : float, default 0.0
        Chemical potential of the lead.
    kt : float, default 0.0
        Thermal energy of the lead.
    bias : float, default 0.0
        Bias of the lead, see :class:`SanchoRubio`.

    Notes
    -----
    .. math::

        \Sigma_\alpha(E) = \tau_{m i}\, g_s(E)\, \tau_{i m},
        \qquad \tau_{im} = z S_{im} - H_{im},

    with :math:`g_s` the surface Green's function from :class:`SanchoRubio`.
    The lead is always in its own equilibrium, so its lesser and greater
    self-energies follow from its Fermi function.
    """

    ids = ["left", "right"]

    def __init__(
        self,
        hs_ii: Tuple[NDArray, NDArray],
        hs_ij: Tuple[NDArray, NDArray],
        hs_im: Optional[Tuple[NDArray, NDArray]] = None,
        nbf_m: Optional[int] = None,
        id: str = "left",
        eta: float = 1e-5,
        mu: float = 0.0,
        kt: float = 0.0,
        bias: float = 0.0,
    ) -> None:

        assert id in self.ids, f"Invalid id. Choose between {self.ids}"

        if id == "right":
            h_ij = hs_ij[0].swapaxes(0, 1).conj()
            s_ij = hs_ij[1].swapaxes(0, 1).conj()
            hs_ij = (h_ij, s_ij)
            if hs_im is not None:
                h_im = hs_im[0].swapaxes(0, 1).conj()
                s_im = hs_im[1].swapaxes(0, 1).conj()
                hs_im = (h_im, s_im)
        if hs_im is None:
            hs_im = hs_ij

        gf = SanchoRubio(hs_ii, hs_ij, eta=eta, mu=mu, kt=kt, bias=bias)

        if nbf_m is not None:
            nbf_i = gf.shape[0]
            h_im = np.zeros((nbf_i, nbf_m), complex)
            s_im = np.zeros((nbf_i, nbf_m), complex)
            if id == "left":
                h_im[:, :nbf_i] = hs_im[0]
                s_im[:, :nbf_i] = hs_im[1]
            else:
                h_im[:, -nbf_i:] = hs_im[0]
                s_im[:, -nbf_i:] = hs_im[1]
            hs_im = (h_im, s_im)

        super().__init__(gf, hs_im)

    @property
    def bias(self) -> float:
        """Bias :math:`V` of the lead, see :class:`SanchoRubio`."""
        return self.gf.bias

    def lesser(self, energy: float) -> NDArray[np.complex128]:
        r"""Lesser self-energy :math:`\Sigma^<_\alpha = -f_\alpha(\Sigma^r - \Sigma^a) = i f_\alpha \Gamma_\alpha`.

        Describes electrons injected from the lead, filled up to its own
        Fermi level.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        # Always at equilibrium!
        return -self.gf.fermi(energy) * (self.retarded(energy) - self.advanced(energy))

    def greater(self, energy: float) -> NDArray[np.complex128]:
        r"""Greater self-energy :math:`\Sigma^>_\alpha = (1 - f_\alpha)(\Sigma^r - \Sigma^a)`.

        Describes the empty lead states into which device electrons can
        escape.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        return (1.0 - self.gf.fermi(energy)) * (
            self.retarded(energy) - self.advanced(energy)
        )
