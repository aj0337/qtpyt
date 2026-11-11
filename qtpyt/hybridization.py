r"""Hybridization function of a subspace embedded in a larger system.

The hybridization function :math:`\Delta(E)` is the self-energy that the
rest of the system exerts on a chosen set of orbitals, for example the
:math:`d` orbitals of a magnetic impurity. It is the input to impurity
solvers (Anderson model, DMFT): its imaginary part tells how strongly the
subspace exchanges electrons with its surroundings at each energy, and its
real part how much the surroundings shift the subspace levels.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qtpyt import xp
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.base.selfenergy import BaseSelfEnergy
from qtpyt.surface.principallayer import fill_sigma


class Hybridization(BaseSelfEnergy):
    r"""Hybridization function :math:`\Delta(E)` of a subspace with its environment.

    Parameters
    ----------
    gf : BaseGreenFunction
        Green's function projected on the subspace. Its ``H`` and ``S`` are
        the Hamiltonian and overlap of the subspace, and its ``retarded``
        gives the full Green's function of the embedded system restricted
        to the subspace orbitals. The subspace is assumed orthogonal to the
        rest of the system.

    Notes
    -----
    The hybridization is what is missing from the bare subspace to
    reproduce the embedded Green's function,

    .. math::

        \Delta(E) = g_0^{-1}(z) - G^{-1}(E), \qquad g_0^{-1}(z) = zS - H,

    with :math:`z = E + i\eta`. An effective static Hamiltonian of the
    subspace at energy :math:`E` is then :math:`H + \operatorname{Re}\Delta(E)`.
    The result is kept for the last energy.
    """

    def __init__(self, gf : BaseGreenFunction) -> None:
        super().__init__(gf, (gf.H, None))

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded hybridization function :math:`\Delta(E)`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        Delta : ndarray, shape (n_sub, n_sub)
            :math:`\Delta(E)` on the orbitals of the subspace.
        """
        #          -1        -1
        # /\(z) = g (z)  -  G (z)
        #          0
        if energy != self.energy:
            self.energy = energy
            Ginv = np.linalg.inv(self.gf.retarded(energy))
            fill_sigma(self.z(energy), self.H, self.S, Ginv, self.Sigma)

        return self.Sigma
