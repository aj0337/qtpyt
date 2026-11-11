r"""Green's function of a long device stored as a block tridiagonal matrix.

For devices with many orbitals the dense inverse of
:class:`~qtpyt.base.greenfunction.GreenFunction` becomes too costly. When
the device splits into layers that couple only to their neighbours (see
:mod:`~qtpyt.block_tridiag.graph_partition`), only the blocks of
:math:`G^r` that observables need are computed, recursively, layer by
layer. Cost and memory grow linearly with the number of layers.
"""
from __future__ import annotations

from math import pi
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from qtpyt import xp
from qtpyt.base._kernels import get_lambda
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.block_tridiag import solvers
from qtpyt.block_tridiag.btmatrix import BTMatrix, empty_buffer


class GreenFunction(BaseGreenFunction):
    r"""Retarded Green's function of a layered device attached to leads.

    Same physics as :class:`~qtpyt.base.greenfunction.GreenFunction`, with
    :math:`H`, :math:`S` and :math:`G` stored as
    :class:`~qtpyt.block_tridiag.btmatrix.BTMatrix`. The left lead couples
    to the first layer and the right lead to the last.

    Parameters
    ----------
    hs_list_ii : list of ndarray, shape (2, n_q, n_q), length N
        ``(H_qq, S_qq)`` of each layer, stacked.
    hs_list_ij : list of ndarray, shape (2, n_q, n_{q+1}), length N - 1
        ``(H_{q,q+1}, S_{q,q+1})`` coupling of each layer to the next one.
        The lower blocks are taken as the transpose of these.
    selfenergies : list of (int, self-energy) pairs, default []
        ``(q, selfenergy)``: the self-energy is added to the whole diagonal
        block of layer ``q``. Leads go on the first (``q=0``) and last
        (``q=N-1``) layers.
    idxleads : list of int, default [0, 1]
        Positions of the left and right leads in ``selfenergies``. Their
        broadenings are stored at the same positions in :attr:`gammas`, so
        the leads must be the first two entries.
    eta : float, default 1e-5
        Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`.
    mu : float, default 0.0
        Chemical potential.
    kt : float, default 0.0
        Thermal energy.
    solver : {'spectral', 'coupling', 'dyson'}, default 'spectral'
        Recursion used to compute the parts of :math:`G` that are needed,
        see :mod:`~qtpyt.block_tridiag.solvers`. Only ``'dyson'`` supports
        :meth:`get_transmission` with its ``ferretti`` and ``normalize``
        options and returns :math:`G^r` from :meth:`retarded`.

    Raises
    ------
    RuntimeError
        If ``solver`` is not one of the available solvers.

    Notes
    -----
    Setting :attr:`eta` also sets it on every attached self-energy.
    """

    def __init__(
        self,
        hs_list_ii: List[NDArray],
        hs_list_ij: List[NDArray],
        selfenergies: List[Tuple[int, Any]] = [],
        idxleads: List[int] = [0, 1],
        eta: float = 1e-5,
        mu: float = 0.0,
        kt: float = 0.0,
        solver: str = "spectral",
    ) -> None:
        Solver = solvers.__dict__.get(solver.capitalize(), None)
        if Solver is None:
            raise RuntimeError(f"Invalid solver {solver}")
        self.solver = Solver(self)

        self.hs_list_ii = hs_list_ii
        self.hs_list_ij = hs_list_ij

        # Here the Hamiltonian and Overlap may be on GPU.
        def _fill(buffer, i):
            buffer.m_qii[0] = xp.asarray(self.hs_list_ii[0][i])
            for q in range(1, buffer.N):
                buffer.m_qii[q] = xp.asarray(self.hs_list_ii[q][i])
                buffer.m_qij[q - 1] = xp.asarray(self.hs_list_ij[q - 1][i])
                buffer.m_qji[q - 1] = buffer.m_qij[q - 1].T  # reference

        N = len(hs_list_ii)
        S = empty_buffer(N)
        H = empty_buffer(N)

        _fill(S, 1)
        _fill(H, 0)

        S = BTMatrix(S.m_qii, S.m_qij, S.m_qji)
        H = BTMatrix(H.m_qii, H.m_qij, H.m_qji)

        self.Ginv = empty_buffer(N)

        self.selfenergies = selfenergies
        self.idxleads = idxleads
        self.gammas = [None] * len(idxleads)

        super().__init__(H, S, eta, mu, kt)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for _, selfenergy in self.selfenergies:
                selfenergy.eta = value
        super().__setattr__(name, value)

    def get_Ginv(self, energy: float) -> Any:
        r"""Inverse Green's function :math:`zS - H - \sum_q \Sigma_q(E)` as blocks.

        Also stores the lead broadenings in :attr:`gammas`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        z = self.z(energy)

        for q, (h_ii, s_ii) in enumerate(zip(self.H.m_qii, self.S.m_qii)):
            self.Ginv.m_qii[q] = z * s_ii - h_ii

        # Upper and lower diagonals.
        for q, (h_ij, s_ij) in enumerate(zip(self.H.m_qij, self.S.m_qij)):
            self.Ginv.m_qij[q] = z * s_ij - h_ij
            self.Ginv.m_qji[q] = z * s_ij.T.conj() - h_ij.T.conj()

        # Add selfenergies
        for i, (block, selfenergy) in enumerate(self.selfenergies):
            # Possibly offload to GPU.
            sigma = xp.asarray(selfenergy.retarded(energy))
            self.Ginv.m_qii[block] -= sigma
            if i in self.idxleads:
                self.gammas[i] = get_lambda(sigma)

        return self.Ginv

    def get_transmission(self, energy: float, ferretti: bool = False, normalize: bool = False) -> float:
        r"""Transmission between the two leads, computed by the solver.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        ferretti : bool, default False
            Include the vertex correction of
            :meth:`~qtpyt.block_tridiag.solvers.Dyson.get_transmission`.
        normalize : bool, default False
            Divide by the number of open channels, see
            :meth:`~qtpyt.block_tridiag.solvers.Dyson.get_transmission`.

        Returns
        -------
        T : float
        """
        return self.solver.get_transmission(energy, ferretti, normalize)

    def get_spectrals(self, energy: float) -> Tuple[Any, Any]:
        r"""Partial spectral functions :math:`A_L = G^r\Gamma_L G^a` and :math:`A_R = G^r\Gamma_R G^a`.

        Requires the ``'spectral'`` solver.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        A_L, A_R : BTMatrix
        """
        return self.solver.get_spectrals(energy)

    def retarded(self, energy: float, inverse: bool = False) -> Any:
        r"""Block tridiagonal part of the Green's function given by the solver.

        With the ``'dyson'`` solver this is :math:`G^r(E)`. With
        ``'spectral'`` it is :math:`A_L + A_R`, see
        :meth:`~qtpyt.block_tridiag.solvers.Spectral.get_retarded`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        inverse : bool, default False
            Ignored.

        Returns
        -------
        G : BTMatrix
        """
        return self.solver.get_retarded(energy)

    def get_dos(self, energy: float) -> float:
        r"""Total density of states :math:`-\frac{1}{\pi}\operatorname{Im}\operatorname{Tr}[G S]`.

        :math:`G` is what :meth:`retarded` returns, so this is the density of
        states only with the ``'dyson'`` solver.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        G = self.solver.get_retarded(energy)
        return -1 / pi * G.dottrace(self.S).imag

    def get_pdos(self, energy: float) -> NDArray[np.float64]:
        r"""Orbital-resolved density of states :math:`-\frac{1}{\pi}\operatorname{Im}(G S)_{ii}`.

        Mulliken-type projection, whose sum is :meth:`get_dos`. As for
        :meth:`get_dos`, meaningful with the ``'dyson'`` solver.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        pdos : ndarray, shape (n_orb,)
        """
        G = self.solver.get_retarded(energy)
        return -1 / pi * G.dotdiag(self.S).imag

    # def add_screening(self, V):
    #     if not hasattr(self, 'V'):
    #         self.V = np.zeros(self.nbf)
    #     assert V.size == self.nbf
    #     #Add screening and remove (if exists) current.
    #     h_qii = self.hs_list_ii[0]
    #     if sum(self.V) != 0:
    #         add_diagonal(h_qii, - self.V)
    #     self.V[:] = V
    #     add_diagonal(h_qii, self.V)

    # def remove_screening(self):
    #     h_qii = self.hs_list_ii[0]
    #     add_diagonal(h_qii, -self.V)
    #     self.V[:] = 0.
