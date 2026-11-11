r"""Solvers that extract observables from a block tridiagonal Green's function.

Each solver picks the recursion of :mod:`~qtpyt.block_tridiag.recursive`
suited to a set of observables, so that only the needed blocks of
:math:`G^r` are computed:

``Spectral``
    Partial spectral functions :math:`A_L`, :math:`A_R` and the elastic
    transmission. Use it for local currents and non-equilibrium densities.
``Coupling``
    Only the corner block :math:`G_{N1}`, the cheapest route to the elastic
    transmission.
``Dyson``
    The block tridiagonal part of :math:`G^r` and the corner block, for the
    density of states, occupations and transmission with corrections.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

from numpy.typing import NDArray

from qtpyt import xp
from qtpyt.base._kernels import dagger, get_lambda
from qtpyt.block_tridiag.btmatrix import BTMatrix

# from qtpyt.block_tridiag.greenfunction import GreenFunction
from qtpyt.block_tridiag.recursive import (
    coupling_method_N1,
    dyson_method,
    spectral_method,
)
import scipy
inv = xp.linalg.inv  # la.inv
dot = xp.dot


class Solver:
    """Common interface of the block tridiagonal solvers.

    Parameters
    ----------
    gf : GreenFunction
        The block tridiagonal Green's function
        (:class:`qtpyt.block_tridiag.greenfunction.GreenFunction`) that
        provides :math:`G^{-1}(E)` and the lead broadenings.
    method : callable
        Recursion from :mod:`~qtpyt.block_tridiag.recursive`, applied to the
        blocks of :math:`G^{-1}`.

    Notes
    -----
    The energy of the last evaluation is shared with ``gf``, and results
    are kept for that energy.
    """

    def __init__(self, gf: Any, method: Callable) -> None:
        self.gf = gf
        self.method = method

    @property
    def energy(self) -> float:
        """Energy of the last evaluation, shared with the Green's function."""
        return self.gf.energy

    @energy.setter
    def energy(self, energy):
        self.gf.energy = energy

    def inv(self, Ginv: Any, *args: Any, **kwargs: Any) -> Any:
        r"""Apply the solver's recursion to the blocks of :math:`G^{-1}`.

        Parameters
        ----------
        Ginv : BTMatrix
            Inverse Green's function. Its blocks are overwritten.
        *args, **kwargs
            Passed to the recursion.
        """
        return self.method(Ginv.m_qii, Ginv.m_qij, Ginv.m_qji, *args, **kwargs)

    def get_transmission(self, energy: float, *args: Any, **kwargs: Any) -> float:
        """Elastic transmission; raises ``NotImplementedError`` unless a solver defines it.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        *args, **kwargs
            Solver-specific options.
        """
        raise NotImplementedError(
            "{self.__class__.__name__} does not implement transmission."
        )

    def get_retarded(self, energy: float) -> Any:
        """Retarded Green's function; raises ``NotImplementedError`` unless a solver defines it.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        raise NotImplementedError(
            "{self.__class__.__name__} does not implement retarded."
        )

    def get_spectrals(self, energy: float) -> Any:
        """Partial spectral functions; raises ``NotImplementedError`` unless a solver defines them.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        raise NotImplementedError(
            "{self.__class__.__name__} does not implement spectrals."
        )


class Spectral(Solver):
    r"""Partial spectral functions of the two leads, and the transmission from them.

    Parameters
    ----------
    gf : GreenFunction
        Block tridiagonal Green's function.
    """

    def __init__(self, gf: Any) -> None:
        super().__init__(gf, spectral_method)
        self.A1 = None
        self.A2 = None

    def get_spectrals(self, energy: float) -> Tuple[Any, Any]:
        r"""Block tridiagonal parts of :math:`A_L = G^r\Gamma_L G^a` and :math:`A_R = G^r\Gamma_R G^a`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.

        Returns
        -------
        A_L, A_R : BTMatrix
        """
        if self.energy != energy:
            self.energy = energy
            self.A1, self.A2 = map(
                lambda A: BTMatrix(*A),
                self.inv(
                    self.gf.get_Ginv(energy), self.gf.gammas[0], self.gf.gammas[1]
                ),
            )
        return self.A1, self.A2

    def get_transmission(self, energy: float) -> float:
        r"""Elastic transmission :math:`T = \operatorname{Tr}[\Gamma_L (A_R)_{11}]`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        A2 = self.get_spectrals(energy)[1]
        gamma_L = self.gf.gammas[0]
        T_e = gamma_L.dot(A2[0, 0]).real.trace()
        return T_e

    def get_retarded(self, energy: float) -> Any:
        r"""Sum of the partial spectral functions :math:`A_L + A_R`.

        This is the spectral function :math:`A = i(G^r - G^a)` of the device
        when the leads are the only self-energies and :math:`\eta \to 0`.
        It is not the retarded Green's function.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.

        Returns
        -------
        A : BTMatrix
        """
        A1, A2 = self.get_spectrals(energy)
        return A1 + A2


class Coupling(Solver):
    r"""Elastic transmission from the corner block :math:`G_{N1}` alone.

    Parameters
    ----------
    gf : GreenFunction
        Block tridiagonal Green's function.
    """

    def __init__(self, gf: Any) -> None:
        super().__init__(gf, coupling_method_N1)

    def get_transmission(self, energy: float) -> float:
        r"""Elastic transmission :math:`T = \operatorname{Tr}[\Gamma_R G_{N1} \Gamma_L G_{N1}^\dagger]`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        """
        if self.energy != energy:
            self.energy = energy
            Ginv = self.gf.get_Ginv(energy)
            g_N1 = self.inv(Ginv)
            gamma_L = self.gf.gammas[0]
            gamma_R = self.gf.gammas[1]
            T_e = xp.einsum(
                "ij,jk,kl,lm->im", gamma_R, g_N1, gamma_L, dagger(g_N1), optimize=True
            ).real.trace()
        return T_e


class Dyson(Solver):
    r"""Block tridiagonal part of :math:`G^r` and the transmission, with optional corrections.

    Parameters
    ----------
    gf : GreenFunction
        Block tridiagonal Green's function.
    trans : bool, default True
        Also compute the corner block :math:`G_{1N}`, needed by
        :meth:`get_transmission`.
    """

    def __init__(self, gf: Any, trans: bool = True) -> None:
        super().__init__(gf, dyson_method)
        self.G = None
        self.g_1N = None
        self.trans = trans

    def get_retarded(self, energy: float) -> Any:
        r"""Block tridiagonal part of :math:`G^r(E)`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.

        Returns
        -------
        G : BTMatrix
        """
        if self.energy != energy:
            self.energy = energy
            Ginv = self.gf.get_Ginv(energy)
            self.G = None
            if self.trans:
                g_1N, G = self.inv(Ginv, trans=True)
                self.g_1N = g_1N
                self.G = BTMatrix(*G)
            else:
                G = self.inv(Ginv, trans=False)
                self.G = BTMatrix(*G)
        return self.G

    def get_projected_vertex_correction(self, energy: float, v_L: NDArray, v_R: NDArray, v_tip_L: NDArray, v_tip_R: NDArray) -> NDArray:
        r"""Interaction broadening carried from the central layer to the outer layers.

        The interaction self-energy lives on the central layer, while the
        lead broadenings live on the first and last layers. This maps
        :math:`1 + \Gamma_c` onto the first and last layers through two
        hopping steps on each side and adds the two results. It assumes five
        layers (lead side, tip, centre, tip, lead side) with first and last
        layers of equal size.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        v_L : ndarray
            Hopping from layer 0 to layer 1.
        v_R : ndarray
            Hopping from the last layer to the one before it.
        v_tip_L : ndarray
            Hopping from layer 1 to the central layer.
        v_tip_R : ndarray
            Hopping from the second-to-last layer to the central layer.

        Returns
        -------
        delta : ndarray, shape (n_1, n_1)
            .. math::

                v_L v_{tip,L} (1 + \Gamma_c) v_{tip,L}^\dagger v_L^\dagger
                + v_R v_{tip,R} (1 + \Gamma_c) v_{tip,R}^\dagger v_R^\dagger,

            with :math:`\Gamma_c = i(\Sigma_c - \Sigma_c^\dagger)` of the last
            non-lead entry of ``gf.selfenergies``.
        """
        for i, (indices, selfenergy) in enumerate(self.gf.selfenergies):
            if i not in self.gf.idxleads:
                sigma = selfenergy.retarded(energy)
                delta = xp.zeros_like(sigma, dtype=complex)
                delta += get_lambda(sigma)
        delta.flat[:: len(delta) + 1] += 1.0
        # Step 1: Expand delta to tip subspace (306, 306)
        delta_tip_L = v_tip_L @ delta @ v_tip_L.conj().T  # (306, 306)
        delta_tip_R = v_tip_R @ delta @ v_tip_R.conj().T  # (306, 306)

        # Step 2: Expand to lead subspace (810, 810)
        projected_delta = xp.zeros(
            (v_L.shape[0], v_L.shape[0]), dtype=complex
        )  # (810, 810)
        projected_delta += v_L @ delta_tip_L @ v_L.conj().T
        projected_delta += v_R @ delta_tip_R @ v_R.conj().T

        return projected_delta

        # delta = v_L @ delta @ v_L.conj().T + v_R @ delta @ v_R.conj().T
        # return delta


    def get_transmission(self, energy: float, ferretti: bool, normalize: bool = False, tol: float = 1e-10) -> float:
        r"""Transmission between the two leads, optionally with a vertex correction or per open channel.

        Parameters
        ----------
        energy : float
            Energy :math:`E`.
        ferretti : bool
            Include the projected vertex correction described in the Notes.
            Has no effect when only leads are attached.
        normalize : bool, default False
            Without the correction only: divide by the number of open
            channels, the eigenvalues of :math:`t^\dagger t` above ``tol``.
            The result is the mean transmission per open channel. Prints the
            energy and the channel count.
        tol : float, default 1e-10
            Threshold for an open channel.

        Returns
        -------
        T : float

        Notes
        -----
        Without the correction,

        .. math::

            T = \operatorname{Tr}[\Gamma_L G_{1N} \Gamma_R G_{1N}^\dagger],
            \qquad t = \Gamma_R^{1/2} G_{1N} \Gamma_L^{1/2}.

        With ``ferretti``, the matrix :math:`\Delta` of
        :meth:`get_projected_vertex_correction` gives

        .. math::

            T = \operatorname{Tr}[\Gamma_L G_{1N} \Gamma_R (1 + \Lambda) G_{1N}^\dagger],
            \qquad
            \Lambda = (\Gamma_L + \Gamma_R + 2\eta S_{11})^{-1}\Delta,

        a block-tridiagonal form of the correction in
        :meth:`qtpyt.base.greenfunction.GreenFunction.get_transmission`.
        """
        _ = self.get_retarded(energy)
        gamma_L = self.gf.gammas[0]
        gamma_R = self.gf.gammas[1]

        if (len(self.gf.idxleads) == len(self.gf.selfenergies)) or (not ferretti):
            T_e = xp.einsum(
                "ij,jk,kl,lm->im",
                gamma_L,
                self.g_1N,
                gamma_R,
                dagger(self.g_1N),
                optimize=True,
            ).real.trace()

            if normalize:
                sqrt_gamma_L = scipy.linalg.sqrtm(gamma_L)
                sqrt_gamma_R = scipy.linalg.sqrtm(gamma_R)
                t_matrix = sqrt_gamma_R @ self.g_1N @ sqrt_gamma_L
                tau = xp.linalg.eigvalsh(t_matrix.conj().T @ t_matrix)
                tau = xp.flip(xp.sort(tau.real))
                N_open = xp.sum(tau > tol)
                print("E, N", energy, N_open)
                return T_e / N_open if N_open > 0 else 0.0

            else:
                return T_e

        v_L = self.gf.H.m_qij[0]
        v_tip_L = self.gf.H.m_qij[1]
        v_R = self.gf.H.m_qji[-1]
        v_tip_R = self.gf.H.m_qji[-2]
        delta = self.get_projected_vertex_correction(energy, v_L, v_R, v_tip_L, v_tip_R)
        delta[:] = xp.linalg.solve(
            gamma_L + gamma_R + 2 * self.gf.eta * self.gf.S.m_qii[0], delta
        )
        delta.flat[:: len(delta) + 1] += 1.0

        # Compute transmission using delta correction
        T_e = xp.einsum(
            "ij,jk,kl,lm,mn->in",
            gamma_L,
            self.g_1N,
            gamma_R,
            delta,
            dagger(self.g_1N),
            optimize=True,
        ).real.trace()

        return T_e
