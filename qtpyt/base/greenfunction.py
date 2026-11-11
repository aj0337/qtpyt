r"""Green's functions of a finite region coupled to leads and other environments.

This module defines the central object of a transport calculation: the
retarded Green's function of the scattering region,

.. math::

    G^r(E) = \left[(E + i\eta) S - H - \Sigma(E)\right]^{-1},

together with the quantities derived from it: lesser and greater functions,
density of states, transmission and energy-resolved current. :math:`H` and
:math:`S` are the Hamiltonian and overlap in a localized orbital basis, and
:math:`\Sigma` collects the self-energies of everything that was eliminated
from the region (the leads, and optionally interactions).

All matrices live on the array backend ``qtpyt.xp``, which is CuPy when
CuPy is installed and NumPy otherwise.
"""
from __future__ import annotations

from math import pi
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qtpyt import xp
from qtpyt.base._kernels import dagger, dotdiag, dots, dottrace, get_lambda
from qtpyt.screening.tools import fermidistribution, rotate


class BaseGreenFunction:
    r"""Single-particle Green's function of a region with Hamiltonian and overlap.

    The Green's function answers the question: if an electron of energy
    :math:`E` is injected into orbital :math:`\nu`, what amplitude appears
    in orbital :math:`\mu`? Its poles sit at the energy levels of the
    region and its imaginary part gives the density of states. Subclasses
    decide how the inverse :math:`G^{-1}(E)` is built, for example by
    adding lead self-energies or by surface decimation.

    Parameters
    ----------
    H : ndarray, shape (n_orb, n_orb)
        Hamiltonian in a localized orbital basis, in any energy unit (eV
        when built from ASE/GPAW). Every energy passed to this object uses
        the same unit.
    S : ndarray, shape (n_orb, n_orb)
        Overlap matrix :math:`S_{\mu\nu} = \langle\phi_\mu|\phi_\nu\rangle`.
        Pass the identity for an orthogonal basis.
    eta : float, default 1e-5
        Imaginary shift :math:`\eta` in :math:`z = E + i\eta`. Each level is
        broadened into a Lorentzian of half-width :math:`\eta`. Choose it
        smaller than the features you want to resolve and larger than the
        spacing of your energy grid; otherwise the density of states turns
        into isolated spikes.
    mu : float, default 0.0
        Chemical potential :math:`\mu` of the reservoir that sets the
        occupation of the region.
    kt : float, default 0.0
        Thermal energy :math:`k_B T` of that reservoir. Zero gives a sharp
        Fermi step.

    Attributes
    ----------
    energy : float or None
        Energy of the most recent evaluation of :math:`G^r`, used by
        subclasses to reuse the result at the same energy.

    Notes
    -----
    The advanced function is :math:`G^a = (G^r)^\dagger`. In thermal
    equilibrium the lesser and greater functions follow from
    :math:`G^r` and the Fermi function :math:`f(E)`,

    .. math::

        G^<(E) = -f(E)\,[G^r(E) - G^a(E)], \qquad
        G^>(E) = [1 - f(E)]\,[G^r(E) - G^a(E)].
    """

    def __init__(
        self, H: NDArray, S: NDArray, eta: float = 1e-5, mu: float = 0.0, kt: float = 0.0
    ) -> None:
        self.H = H  # Hamiltonian
        self.S = S  # Overlap
        self.eta = eta  # immaginary shift
        self.mu = mu  # chemical potential
        self.kt = kt  # boltzmann temperature
        self.energy = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Matrix shape ``(n_orb, n_orb)`` of the region."""
        return self.H.shape

    def z(self, energy: float) -> complex:
        r"""Complex energy :math:`z = E + i\eta` just above the real axis.

        Evaluating at :math:`z` instead of :math:`E` selects the retarded
        (causal) Green's function and gives every level a width
        :math:`\eta`.

        Parameters
        ----------
        energy : float
            Real energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        z : complex
            :math:`E + i\eta`.
        """
        return energy + 1.0j * self.eta

    def fermi(self, energy: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        r"""Fermi–Dirac occupation of the reservoir.

        Parameters
        ----------
        energy : float or ndarray
            Energy :math:`E`, or an array of energies, same unit as :attr:`H`.

        Returns
        -------
        f : float or ndarray
            :math:`f(E) = 1/\{1 + \exp[(E - \mu)/k_B T]\}`. At ``kt=0`` this
            is a step that equals 1 for :math:`E \le \mu`, including
            :math:`E = \mu` itself.
        """
        return fermidistribution(energy - self.mu, self.kt)

    def get_Ginv(self, energy: float) -> NDArray[np.complex128]:
        r"""Inverse retarded Green's function :math:`[G^r(E)]^{-1}`.

        Defined by each subclass. Here it returns ``None``.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        """
        return None

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded Green's function :math:`G^r(E)` of the region.

        Defined by each subclass. Here it returns ``None``.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_r : ndarray, shape (n_orb, n_orb)
            :math:`G^r(E)` in the orbital basis of the region.
        """
        return None

    def advanced(self, energy: float) -> NDArray[np.complex128]:
        r"""Advanced Green's function :math:`G^a(E) = [G^r(E)]^\dagger`.

        Describes propagation backward in time; it carries the same
        information as :math:`G^r`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_a : ndarray, shape (n_orb, n_orb)
            :math:`G^a(E)` in the orbital basis of the region.
        """
        return dagger(self.retarded(energy))

    def lesser(self, energy: float) -> NDArray[np.complex128]:
        r"""Lesser Green's function :math:`G^<(E)` in thermal equilibrium.

        :math:`-iG^<` gives the occupied part of the spectrum: integrated
        over energy it yields the electron density matrix.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_lesser : ndarray, shape (n_orb, n_orb)
            :math:`G^<(E) = -f(E)\,[G^r(E) - G^a(E)]`, with the occupation
            :math:`f` of :meth:`fermi`.
        """
        return -self.fermi(energy) * (self.retarded(energy) - self.advanced(energy))

    def greater(self, energy: float) -> NDArray[np.complex128]:
        r"""Greater Green's function :math:`G^>(E)` in thermal equilibrium.

        :math:`iG^>` gives the empty part of the spectrum, the states
        available to an added electron.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_greater : ndarray, shape (n_orb, n_orb)
            :math:`G^>(E) = [1 - f(E)]\,[G^r(E) - G^a(E)]`.
        """
        return (1.0 - self.fermi(energy)) * (
            self.retarded(energy) - self.advanced(energy)
        )

    def get_dos(self, energy: float) -> float:
        r"""Total density of states of the region.

        Counts the states per unit energy at :math:`E`, each broadened by
        :math:`\eta` and by the coupling to the leads.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        dos : float
            :math:`D(E) = -\frac{1}{\pi}\operatorname{Im}\operatorname{Tr}[G^r(E)\,S]`,
            in states per energy unit.
        """
        return -dottrace(self.retarded(energy), self.S).imag / pi

    def get_pdos(self, energy: float) -> NDArray[np.float64]:
        r"""Density of states projected onto each basis orbital.

        Gives, for every orbital :math:`\phi_i`, how many states per unit
        energy at :math:`E` have weight on that orbital. Plot it to see
        which atoms or orbital characters a transmission peak comes from.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        pdos : ndarray, shape (n_orb,)
            :math:`D_i(E)` in states per energy unit, one value per orbital.

        See Also
        --------
        get_dos : Total density of states.

        Notes
        -----
        The projection uses the normalized orbital itself,

        .. math::

            D_i(E) = -\frac{1}{\pi}\,
            \frac{\operatorname{Im}\,(S\,G^r(E)\,S)_{ii}}{S_{ii}}.

        In a non-orthogonal basis the orbitals overlap, so
        :math:`\sum_i D_i(E)` differs from :meth:`get_dos`: the sum counts
        twice the weight that lives in the overlap between orbitals. For an
        orthogonal basis (:math:`S = 1`) the two agree.
        """
        return (
            -(
                dotdiag(self.S, self.retarded(energy).dot(self.S)) / self.S.diagonal()
            ).imag
            / pi
        )


class GreenFunction(BaseGreenFunction):
    r"""Retarded Green's function of a scattering region attached to leads.

    The region is finite, but electrons can leave it into the leads and,
    optionally, scatter on other degrees of freedom (phonons, electron
    correlation). Each of these environments enters as a self-energy
    :math:`\Sigma_\alpha(E)` added to the orbitals it couples to. The
    imaginary part of the self-energy turns the sharp levels of the
    isolated region into resonances whose width is the escape rate.

    Parameters
    ----------
    H : array_like, shape (n_orb, n_orb)
        Hamiltonian of the scattering region. See :class:`BaseGreenFunction`
        for units.
    S : array_like, shape (n_orb, n_orb)
        Overlap matrix of the scattering region.
    selfenergies : sequence of (index, self-energy) pairs, default []
        Environments coupled to the region. ``index`` selects the block of
        orbitals the environment couples to, as a NumPy index into an
        ``(n_orb, n_orb)`` matrix (for example ``numpy.ix_(i, i)`` or a
        tuple of slices). The self-energy object provides
        ``retarded(energy)`` returning :math:`\Sigma_\alpha(E)` on that
        block; leads also provide ``lesser``, ``greater`` and ``bias``.
    idxleads : sequence of int, default [0, 1]
        Positions in ``selfenergies`` of the two leads between which
        transmission and current are computed, the first one being the
        left lead L. All other entries are treated as interaction
        self-energies. The list is sorted on input.
    eta : float, default 1e-5
        Broadening, see :class:`BaseGreenFunction`.
    mu : float, default 0.0
        Chemical potential, see :class:`BaseGreenFunction`.
    kt : float, default 0.0
        Thermal energy, see :class:`BaseGreenFunction`.

    Attributes
    ----------
    gammas : list of ndarray
        Broadening matrices :math:`\Gamma_\alpha` of the leads in
        ``idxleads``, embedded in the full ``(n_orb, n_orb)`` space,
        filled at each new energy.
    delta : ndarray or None
        Vertex-correction matrix :math:`\Lambda` of the last
        :meth:`get_transmission` call with a correction switched on.

    Notes
    -----
    .. math::

        G^r(E) = \Big[(E + i\eta) S - H - \sum_\alpha \Sigma_\alpha(E)\Big]^{-1}.

    The result is kept for the last energy, so repeated calls at the same
    :math:`E` cost nothing. Changing :attr:`H` or a self-energy does not
    trigger a recomputation at that energy; set ``energy = None`` first.

    Setting :attr:`eta` on this object also sets it on every attached
    self-energy.
    """

    def __init__(
        self,
        H: ArrayLike,
        S: ArrayLike,
        selfenergies: Sequence[Tuple[Any, Any]] = [],
        idxleads: Sequence[int] = [0, 1],
        eta: float = 1e-5,
        mu: float = 0.0,
        kt: float = 0.0,
    ) -> None:
        # Offload to GPU
        H = xp.asarray(H)
        S = xp.asarray(S)

        self.selfenergies = selfenergies
        self.idxleads = sorted(idxleads)
        self.gammas = None
        self.delta = None
        super().__init__(H, S, eta, mu, kt)

    @property
    def equilibrium(self) -> bool:
        """Whether all leads in ``idxleads`` have the same bias (within 1e-7).

        In equilibrium the lesser and greater functions follow from the
        Fermi function alone; otherwise they are built from the lead
        self-energies.
        """
        biases = []
        for i in self.idxleads:
            selfenergy = self.selfenergies[i][1]
            biases.append(selfenergy.bias)
        return xp.all([abs(bias - biases[0]) < 1e-7 for bias in biases])

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for _, selfenergy in self.selfenergies:
                selfenergy.eta = value
        super().__setattr__(name, value)

    def get_Ginv(self, energy: float) -> NDArray[np.complex128]:
        r"""Inverse Green's function :math:`(E + i\eta)S - H - \sum_\alpha \Sigma_\alpha(E)`.

        Also stores the broadening :math:`\Gamma_\alpha` of each lead in
        :attr:`gammas`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        Ginv : ndarray, shape (n_orb, n_orb)
            :math:`[G^r(E)]^{-1}`.
        """
        Ginv = self.z(energy) * self.S - self.H
        self.gammas = []
        # Add selfenergies
        for i, (indices, selfenergy) in enumerate(self.selfenergies):
            sigma = xp.asarray(selfenergy.retarded(energy))
            Ginv[indices] -= sigma
            if i in self.idxleads:
                self.gammas.append(get_lambda(sigma))

        return Ginv

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded Green's function :math:`G^r(E)` including all self-energies.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_r : ndarray, shape (n_orb, n_orb)
            :math:`G^r(E)` in the orbital basis of the region.
        """
        if self.energy != energy:
            self.energy = energy
            Ginv = self.get_Ginv(energy)
            self.Gr = xp.linalg.inv(Ginv)
        return self.Gr

    def lesser(self, energy: float) -> NDArray[np.complex128]:
        r"""Lesser Green's function :math:`G^<(E)`, in or out of equilibrium.

        With all leads at the same bias this is the equilibrium expression
        of :meth:`BaseGreenFunction.lesser`. Under bias the occupation is not
        a single Fermi function, and the region is filled by each
        lead according to its own occupation,

        .. math::

            G^<(E) = G^r(E)\,\Sigma^<(E)\,G^a(E), \qquad
            \Sigma^< = \sum_\alpha \Sigma^<_\alpha,

        where the sum runs over every self-energy that provides a lesser
        component.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_lesser : ndarray, shape (n_orb, n_orb)
            :math:`G^<(E)`.
        """

        # -f(e-mu) (Gr - Ga)
        if self.equilibrium:
            return super().lesser(energy)

        # Gr S< Ga
        self.Sl = xp.zeros(self.shape, complex)
        for i, (indices, selfenergy) in enumerate(self.selfenergies):
            if hasattr(selfenergy, "lesser"):
                self.Sl[indices] += selfenergy.lesser(energy)
        return self.retarded(energy).dot(self.Sl).dot(self.advanced(energy))

    def greater(self, energy: float) -> NDArray[np.complex128]:
        r"""Greater Green's function :math:`G^>(E)`, in or out of equilibrium.

        In equilibrium this is :meth:`BaseGreenFunction.greater`. Under bias
        it follows from the identity :math:`G^> - G^< = G^r - G^a`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        G_greater : ndarray, shape (n_orb, n_orb)
            :math:`G^>(E)`.
        """
        # (1-f(e-mu)) (Gr - Ga)
        if self.equilibrium:
            return super().greater(energy)

        # < = > - r + a
        # > = < + r - a
        return self.lesser(energy) + self.retarded(energy) - self.advanced(energy)

    def get_transmission(
        self, energy: float, ferretti: bool = True, brazilian: bool = False
    ) -> Union[float, Tuple[float, float, float]]:
        r"""Transmission probability between the two leads at energy :math:`E`.

        Gives the number of electron channels at :math:`E` that carry an
        electron from lead L into lead R, the quantity that enters the
        Landauer conductance. When interaction self-energies are attached
        (any entry of ``selfenergies`` not in ``idxleads``), the
        interactions take flux out of the coherent channel. ``ferretti``
        and ``brazilian`` select two ways of adding that flux back as an
        inelastic term. With ``ferretti=True`` (the default) the method
        returns three numbers even when no interaction is attached.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.
        ferretti : bool, default True
            Add the vertex correction of Ferretti et al. [1]_. Takes
            precedence over ``brazilian``.
        brazilian : bool, default False
            Add the alternative correction described in the Notes.

        Returns
        -------
        T : float or tuple of float
            If only leads are attached and both corrections are off, the
            elastic transmission :math:`T_\mathrm{el}(E)` alone. Otherwise
            the tuple ``(T_total, T_elastic, T_inelastic)`` with
            ``T_total = T_elastic + T_inelastic``. The inelastic term is zero
            unless ``ferretti`` or ``brazilian`` is set.

        Notes
        -----
        With :math:`\Gamma_{L,R}` the broadening of the two leads,

        .. math::

            T_\mathrm{el}(E) = \operatorname{Tr}\left[\Gamma_L G^r \Gamma_R G^a\right].

        Let :math:`\Gamma_c = \sum_c i(\Sigma_c - \Sigma_c^\dagger)` be the
        broadening from all interaction self-energies. The Ferretti
        correction is

        .. math::

            T_\mathrm{inel} = \operatorname{Tr}\left[\Gamma_L G^r \Gamma_R \Lambda G^a\right],
            \qquad
            \Lambda = (\Gamma_L + \Gamma_R + 2\eta S)^{-1}\,\Gamma_c,

        where the :math:`2\eta S` term is the broadening that :math:`\eta`
        itself adds to the region. The ``brazilian`` correction treats the
        interaction as an additional lead that absorbs electrons coming
        from L,

        .. math::

            T_\mathrm{inel} = \operatorname{Tr}\left[\Gamma_L G^r \Gamma_c G^a\right].

        The matrix :math:`\Lambda` (or :math:`\Gamma_c` for ``brazilian``)
        is stored in :attr:`delta`.

        References
        ----------
        .. [1] A. Ferretti et al., Phys. Rev. B 72, 125114 (2005).
           https://journals.aps.org/prb/pdf/10.1103/PhysRevB.72.125114
        """
        if (len(self.idxleads) == len(self.selfenergies)) and not (
            brazilian or ferretti
        ):
            a_mm = self.retarded(energy).dot(self.gammas[0])  # updates gammas
            b_mm = self.advanced(energy).dot(self.gammas[1])
            return dottrace(a_mm, b_mm).real

        # Ferretti : https://journals.aps.org/prb/pdf/10.1103/PhysRevB.72.125114
        delta = xp.zeros(self.shape, complex)
        G_r = self.retarded(energy)  # required for populating gammas
        G_a = self.advanced(energy)
        for i, (indices, selfenergy) in enumerate(self.selfenergies):
            if i not in self.idxleads:
                sigma = selfenergy.retarded(energy)
                delta[indices] += get_lambda(sigma)  # 1.j * (sigma - sigma.T.conj())

        # Elastic part (delta = identity)
        T_elastic = (
            dots(
                self.gammas[0],
                G_r,
                self.gammas[1],
                G_a,
            )
            .trace()
            .real
        )

        if ferretti:
            delta[:] = xp.linalg.solve(
                self.gammas[0] + self.gammas[1] + 2 * self.eta * self.S, delta
            )
            self.delta = delta
            T_inelastic = (
                dots(
                    self.gammas[0],
                    G_r,
                    self.gammas[1],
                    delta,
                    G_a,
                )
                .trace()
                .real
            )
        elif brazilian:
            self.delta = delta
            T_inelastic = (
                dots(
                    self.gammas[0],
                    G_r,
                    delta,
                    G_a,
                )
                .trace()
                .real
            )

        else:
            T_inelastic = 0.0

        T_total = T_elastic + T_inelastic

        return T_total, T_elastic, T_inelastic

    def get_current(self, energy: float) -> float:
        r"""Energy-resolved current flowing out of the left lead.

        Integrate over the bias window to obtain the current. The left lead
        is the first entry of ``idxleads``, and its occupation is its own
        Fermi function, including its bias.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        i : float
            Spectral current density, without the prefactor :math:`e/h`.
            Negative values are returned as zero.

        Notes
        -----
        With :math:`\Sigma^<_L = i f_L \Gamma_L` and
        :math:`\Sigma^>_L = -i(1 - f_L)\Gamma_L`,

        .. math::

            i(E) = \max\left(0,\;
            -\operatorname{Tr}\left[\Sigma^<_L G^> - \Sigma^>_L G^<\right]\right).
        """
        selfenergy = self.selfenergies[self.idxleads[0]][1]
        fermi = selfenergy.gf.fermi(energy)
        Gl = self.lesser(energy)  # updates gammas
        Gg = self.greater(energy)
        Sl = 1.0j * self.gammas[0] * fermi
        Sg = 1.0j * self.gammas[0] * (fermi - 1)
        # return sum(dotdiag(Sl, Gg) - dotdiag(Sg, Gl)).real
        return max((-dottrace(Sl, Gg) + dottrace(Sg, Gl)).real, 0.0)
        # return (-dottrace(Sl, Gg) + dottrace(Sg, Gl)).real

    def get_spectrals(self, energy: float) -> List[NDArray[np.complex128]]:
        r"""Partial spectral functions, one for each lead.

        :math:`A_\alpha` contains the states of the region that are filled
        from lead :math:`\alpha`. Under bias, comparing :math:`A_L` and
        :math:`A_R` shows which lead populates which states.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        spectrals : list of ndarray, shape (n_orb, n_orb)
            :math:`A_\alpha(E) = G^r(E)\,\Gamma_\alpha(E)\,G^a(E)` for each lead
            in ``idxleads``, in order. When only leads are attached their sum
            approaches :math:`i(G^r - G^a)` as :math:`\eta \to 0`.
        """
        self.get_Ginv(energy)
        spectrals = []
        for gamma in self.gammas:
            spectrals.append(
                self.retarded(energy).dot(gamma).dot(self.advanced(energy))
            )

        return spectrals


class TransmissionCalculator:
    r"""Elastic transmission between two leads from a given Green's function.

    Parameters
    ----------
    gf : GreenFunction
        Green's function of the scattering region. It must already contain
        the self-energies of both leads; this class does not add them.
    leads : list of LeadSelfEnergy
        The two leads, left first. Exactly two are required.

    Raises
    ------
    NotImplementedError
        If ``leads`` does not contain exactly two leads.
    """

    def __init__(self, gf: GreenFunction, leads: List["LeadSelfEnergy"]) -> None:
        if len(leads) != 2:
            raise NotImplementedError("Only 2 leads implemented.")
        self.gf = gf
        self.leads = leads

    def get_transmission(self, energy: float) -> float:
        r"""Elastic transmission :math:`T(E) = \operatorname{Tr}[G^r \Gamma_L G^a \Gamma_R]`.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as the Hamiltonian.

        Returns
        -------
        T : float
            Transmission probability summed over channels.
        """
        # TODO :: improve with single factorization compatible with GPU
        gamma_L = self.leads[0].get_lambda(energy)
        gamma_R = self.leads[1].get_lambda(energy)
        a_mm = self.gf.retarded(energy).dot(gamma_L)
        b_mm = self.gf.advanced(energy).dot(gamma_R)
        return dottrace(a_mm, b_mm).real
