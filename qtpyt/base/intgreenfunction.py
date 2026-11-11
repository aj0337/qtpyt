r"""Green's function of a block dressed by a second, coupled block.

Used when the device splits into a region :math:`A` and a region
:math:`D` that couples to it, for example a molecule and the interacting
subspace it hosts. The Green's function on :math:`A` is written as the
bare part plus the correction from excursions into :math:`D`, so both
contributions to the transmission can be inspected separately.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from qtpyt import xp
from qtpyt.base._kernels import get_lambda, dottrace, dotdiag, dagger


class GreenFunction:
    r"""Green's function of block :math:`A` including the effect of a coupled block :math:`D`.

    For a Hamiltonian partitioned as

    .. math::

        \begin{pmatrix} A & B \\ C & D \end{pmatrix},

    the exact upper-left block of the inverse is

    .. math::

        G_{AA} = g + g\,\Sigma\,g, \qquad g = A^{-1},

    where :math:`\Sigma = B\,\mathcal{S}^{-1} C` and
    :math:`\mathcal{S} = D - C A^{-1} B` is the Green's function inverse
    of :math:`D` dressed by :math:`A`. The first term propagates within
    :math:`A`; the second goes into :math:`D` and back.

    Parameters
    ----------
    greenfunction : GreenFunction
        Green's function :math:`g` of block :math:`A`, with the leads
        attached (:class:`qtpyt.base.greenfunction.GreenFunction`). It
        provides the lead broadenings ``gammas``.
    selfenergy : tuple
        ``(index, self-energy)`` pair: the self-energy :math:`\Sigma` from
        block :math:`D`, built from a self-energy object whose own Green's
        function already contains the coupling back to :math:`A`, and the
        orbitals of :math:`A` it acts on.

    Notes
    -----
    The self-energy is built in two steps with
    :class:`~qtpyt.base.selfenergy.SelfEnergy`: first fold :math:`A` onto
    :math:`D`, then give that self-energy to the Green's function of
    :math:`D` and fold the dressed :math:`D` back onto :math:`A`.
    """

    def __init__(self, greenfunction: Any, selfenergy: Tuple[Any, Any]) -> None:
        self.greenfunction = greenfunction
        self.selfenergy = selfenergy

        self.g = None
        self.Sigma = xp.zeros_like(self.greenfunction.Ginv)
        self.energy = None

    def get_partials(self, energy: float) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""The two contributions :math:`g` and :math:`g\Sigma g` to :math:`G_{AA}`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.

        Returns
        -------
        g : ndarray
            Green's function of :math:`A` without the coupled block.
        dg : ndarray
            Correction :math:`g\,\Sigma(E)\,g` from excursions into :math:`D`.
        """
        if self.energy != energy:
            self.g = self.greenfunction.retarded(energy)
            indices, selfenergy = self.selfenergy
            self.Sigma[indices] = selfenergy.retarded(energy)
        return self.g, self.g.dot(self.Sigma).dot(self.g)

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded Green's function :math:`G_{AA}(E) = g + g\Sigma g`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        g, dg = self.get_partials(energy)
        return g + dg

    def get_dos(self, energy: float) -> float:
        r"""Total density of states :math:`-\frac{1}{\pi}\operatorname{Im}\operatorname{Tr}[G_{AA} S]` of block :math:`A`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        S = self.greenfunction.S
        GS = dottrace(self.retarded(energy), S)
        return -GS.imag * 1 / xp.pi

    def get_pdos(self, energy: float) -> NDArray[np.float64]:
        r"""Orbital-resolved density of states :math:`-\frac{1}{\pi}\operatorname{Im}(G_{AA} S)_{ii}`.

        This is the Mulliken-type projection, whose sum over orbitals
        equals :meth:`get_dos`. It differs from
        :meth:`qtpyt.base.greenfunction.BaseGreenFunction.get_pdos`, which
        projects onto each normalized orbital.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.

        Returns
        -------
        pdos : ndarray, shape (n_orb,)
        """

        S = self.greenfunction.S
        GS = dotdiag(self.retarded(energy), S)
        return -GS.imag * 1 / xp.pi

    def get_transmission(self, energy: float) -> float:
        r"""Elastic transmission :math:`\operatorname{Tr}[G_{AA}\Gamma_L G_{AA}^\dagger \Gamma_R]`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.
        """
        # also update reference green's function selfenergies
        G = self.retarded(energy)
        gamma_L = self.greenfunction.gammas[0]
        gamma_R = self.greenfunction.gammas[1]
        T_e = dottrace(G.dot(gamma_L), dagger(G).dot(gamma_R)).real
        return T_e

    def get_partial_transmissions(self, energy: float) -> Tuple[float, float, float]:
        r"""Transmission split into direct, via-:math:`D`, and interference parts.

        Writing :math:`G_{AA} = g + \delta g` with :math:`\delta g = g\Sigma g`,
        the transmission separates into a path that stays in :math:`A`, a
        path through :math:`D`, and their interference. The three add up to
        :meth:`get_transmission`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.

        Returns
        -------
        T_b : float
            :math:`\operatorname{Tr}[g\Gamma_L g^\dagger\Gamma_R]`, through :math:`A` alone.
        T_a : float
            :math:`\operatorname{Tr}[\delta g\,\Gamma_L\,\delta g^\dagger\Gamma_R]`, through :math:`D`.
        T_i : float
            Interference term
            :math:`\operatorname{Tr}[g\Gamma_L\delta g^\dagger\Gamma_R] + \operatorname{Tr}[\delta g\,\Gamma_L g^\dagger\Gamma_R]`.
        """
        # also update reference green's function selfenergies
        g, dg = self.get_partials(energy)
        gamma_L = self.greenfunction.gammas[0]
        gamma_R = self.greenfunction.gammas[1]
        T_b = dottrace(g.dot(gamma_L), dagger(g).dot(gamma_R)).real
        T_a = dottrace(dg.dot(gamma_L), dagger(dg).dot(gamma_R)).real
        T_i = (
            dottrace(g.dot(gamma_L), dagger(dg).dot(gamma_R)).real
            + dottrace(dg.dot(gamma_L), dagger(g).dot(gamma_R)).real
        )
        return T_b, T_a, T_i
