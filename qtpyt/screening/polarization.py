r"""Polarization, RPA screened interaction, GW self-energy and constrained RPA.

The chain of a GW calculation on the real-energy grid:

1. :class:`Chi`, the polarization :math:`P` (the density response of
   non-interacting electrons), from the bubble :math:`G \times G`;
2. :class:`WRPA`, the screened interaction :math:`W = [1 - VP]^{-1} V` in
   the random-phase approximation;
3. :class:`GW`, the correlation self-energy :math:`\Sigma = iGW`.

:class:`cRPA` gives instead the partially screened interaction of a
correlated subspace (the effective Hubbard :math:`U` for impurity models),
with the screening from inside that subspace removed.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from numpy.typing import NDArray

from math import pi, sqrt

# Also import numpy for array matrices
import numpy as np

from qtpyt import xp
from qtpyt.base.selfenergy import DataSelfEnergy
from qtpyt.parallel.egrid import GridDesc
from qtpyt.parallel.tools import comm_sum
from qtpyt.projector import BaseProjector, ProjectedGreenFunction
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.iterate_product import iterate_product
from qtpyt.screening.langreth import LangrethPair, assert_domain, change_domain
from qtpyt.screening.products import exchange_product, polarization_product
from qtpyt.screening.tools import lesser_from_retarded, roll, rotate


class Chi(LangrethPair):
    r"""Polarization :math:`P(E)` of non-interacting electrons, the RPA bubble.

    :math:`P` measures how the electron density rearranges in response to a
    potential: an electron is promoted from an occupied to an empty state,
    leaving a hole. In time it is the product of an electron and a hole
    propagator,

    .. math::

        P^\lessgtr_{ij,kl}(t) = -2i\,G^\lessgtr_{ik}(t)\,G^\gtrless_{lj}(-t)
        = 2i\,G^\lessgtr_{ik}(t)\,\big[G^\gtrless_{jl}(t)\big]^*,

    with the factor 2 for spin. The retarded component follows from
    :math:`P^r(t) = \theta(t)[P^>(t) - P^<(t)]`. After :meth:`update_polarization`
    the arrays hold :math:`P^r(E)` and :math:`P^>(E)`.

    Parameters
    ----------
    gf : DistGreenFunction
        Green's function of the region.
    U_pq : ndarray, shape (n_orb**2, n_q), optional
        Rotation from all orbital pairs to optimized pair orbitals; then
        :math:`P` is stored as :math:`U^\dagger P U`.
    oversample : int, default 10
        Zero-padding factor of the time–energy transforms.

    Notes
    -----
    Here and in the derived classes, a ``solve_*`` method does only its own
    step from the current inputs, and the matching ``update_*`` method also
    brings the inputs up to date first.
    """

    def __init__(self, gf: DistGreenFunction, U_pq: Optional[NDArray] = None, oversample: int = 10) -> None:
        self.gf = gf
        # Decide if polarization is for full or optimized pair orbitals.
        nq = gf.no ** 2 if U_pq is None else U_pq.shape[1]
        super().__init__(gf.global_energies, nq, dtype=complex, oversample=oversample)

        self.U_pq = U_pq if U_pq is None else xp.asarray(U_pq)
        # self.U_qp = None if U_pq is None else self.U_pq.T.copy()

    def get_work_array(self) -> Optional[NDArray]:
        """Scratch array for the pair-orbital rotation, or ``None`` without one."""
        if self.U_pq is None:
            return None
        return xp.empty_like(self.U_pq)

    def solve_polarization(self) -> None:
        r"""Compute :math:`P^r(E)` and :math:`P^>(E)` from :math:`G^<(t)` and :math:`G^>(t)`.

        The Green's function must already be in the time domain.
        """
        #
        #         </>        +     </>      >/<
        #       P   =  2 i U   ( G    x   G    ) U

        assert self.gf.domain == "t"
        Gl = self.gf.arrays["l"]
        Gg = self.gf.arrays["g"]
        Pl = self.arrays["l"]
        Pg = self.arrays["g"]

        # Prefactor: 2 for spin, j. from definition, and de / 2 pi for fft
        pre = 2.0j * self.de / (2.0 * pi)

        # Temporary work array
        work1 = self.get_work_array()
        work2 = self.get_work_array()
        # P^</>_ij,kl(t)      = -i G^</>_ik(t) G^>/<_lj(-t)
        #
        # G_lj(-t) = -G_jl(t)*
        #
        for t in range(Gl.shape[0]):
            for G, Gd, P in ([Gl[t], Gg[t], Pl[t]], [Gg[t], Gl[t], Pg[t]]):
                polarization_product(G, Gd.conj(), pre, P, self.U_pq, work1, work2)

        # # chi< = pre * Ud (G< otimes G>.conj ) U
        # iterate_product(
        #     polarization_product,
        #     Gl,
        #     Gg,
        #     self.arrays["l"],  # use retarded array
        #     pre=pre,
        #     rot=self.U_pq,
        #     # rotd=self.U_qp,
        #     work1=work1,
        #     work2=work2,
        # )

        # # chi> = pre * Ud (G> otimes G<.conj ) U
        # iterate_product(
        #     polarization_product,
        #     Gg,
        #     Gl,
        #     self.arrays["g"],
        #     pre=pre,
        #     rot=self.U_pq,
        #     # rotd=self.U_qp,
        #     work1=work1,
        #     work2=work2,
        # )  # P<(t), P>(t)

        self.domain = "t"
        self.convert_less_and_great_to_ret(-self.zero_index)  #  Pr(e), P>(e)

    def update_polarization(self) -> None:
        r"""Bring the Green's function up to date and compute :math:`P^r(E)`, :math:`P^>(E)`."""
        #
        #         r               <    >
        #       P   =  theta(t) (P  - P )
        #
        if self.gf.domain == None:
            self.gf.update()  # Gr(e), G<(e)
        if not "g" in self.gf.arrays.keys():
            self.gf.convert_retarded()  # G<(e), G>(e)
        if self.gf.domain == "e":
            self.gf.convert_domain()  # G<(t), G>(t)
        self.solve_polarization()  # Pr(e), P>(e)


class WRPA(Chi):
    r"""Screened Coulomb interaction :math:`W(E)` in the random-phase approximation.

    The bare repulsion :math:`V` between two electrons is reduced because
    the other electrons rearrange to screen it. Summing the chain of
    polarization bubbles gives

    .. math::

        W^r = [1 - V P^r]^{-1} V, \qquad W^> = W^r P^> W^a.

    After :meth:`update_screening` the arrays hold :math:`W^r(E)` and
    :math:`W^>(E)`, in the memory that held :math:`P^r` and :math:`P^>`.

    Parameters
    ----------
    gf : DistGreenFunction
        Green's function of the region.
    V_qq : ndarray, shape (n_q, n_q)
        Bare Coulomb matrix in the pair-orbital basis (full or optimized).
    U_pq : ndarray, shape (n_orb**2, n_q), optional
        Rotation to optimized pair orbitals, see :class:`Chi`.
    oversample : int, default 10
        Zero-padding factor.
    """

    def __init__(self, gf: DistGreenFunction, V_qq: NDArray, U_pq: Optional[NDArray] = None, oversample: int = 10) -> None:
        super().__init__(gf, U_pq, oversample=oversample)
        self.V_qq = V_qq

    @assert_domain("e")
    def solve_screening(self) -> None:
        r"""Turn :math:`P^r`, :math:`P^>` into :math:`W^r`, :math:`W^>` at every local energy."""
        I = np.eye(self.no)
        Wr = Pr = self.arrays["r"]
        Wg = Pg = self.arrays["g"]

        work = np.empty_like(Wr[0])
        for e in range(self.energies.size):
            # NOTE that Wr is Pr but it's okey since we read-write same energy.
            Wr[e] = np.linalg.solve(I - self.V_qq.dot(Pr[e]), self.V_qq)
            rotate(Pg[e], Wr[e], out=Wg[e], work=work)
            # Wg[e] = Wr[e].dot(Pg[e]).dot(Wr[e].T.conj())

    def update_screening(self) -> None:
        r"""Compute the polarization, then :math:`W^r(E)` and :math:`W^>(E)` in its place."""
        #       r             r
        #      W    = [I - V P ] V
        #
        #       >     r   <   a
        #      W   = W  P   W
        #
        self.update_polarization()  # Pr(e), P>(e)
        self.solve_screening()  # Wr(e), W>(e)


class GW(LangrethPair):
    r"""Correlation part of the GW self-energy with an RPA screened interaction.

    An electron propagating through the region polarizes its surroundings
    and is dressed by the screened interaction it creates. To lowest order
    in :math:`W`,

    .. math::

        \Sigma^\lessgtr_{ij}(t) = i \sum_{kl} G^\lessgtr_{kl}(t)\, W^\lessgtr_{ik,jl}(t),

    and :math:`\Sigma^r(t) = \theta(t)[\Sigma^> - \Sigma^<](t)`. :math:`W^\lessgtr`
    are the lesser and greater components of the screened interaction. After
    :meth:`update_correlation` the arrays hold :math:`\Sigma^r(E)` and
    :math:`\Sigma^<(E)`, and the object can be attached to a
    :class:`~qtpyt.base.greenfunction.GreenFunction` as a self-energy.

    Parameters
    ----------
    wrpa : WRPA
        Screened interaction.
    gf : DistGreenFunction, optional
        Green's function in :math:`\Sigma = iGW`; that of ``wrpa`` if omitted.
    U_pq : ndarray, optional
        Rotation to optimized pair orbitals; that of ``wrpa`` if omitted.
    oversample : int, default 10
        Zero-padding factor.
    """

    def __init__(
        self, wrpa: WRPA, gf: Optional[DistGreenFunction] = None, U_pq: Optional[NDArray] = None, oversample: int = 10
    ) -> None:
        self.wrpa = wrpa
        if gf is None:
            gf = self.wrpa.gf
        if U_pq is None and hasattr(self.wrpa, "U_pq"):  # W is in pair-orbital basis.
            U_pq = self.wrpa.U_pq
        self.gf = gf
        self.U_pq = U_pq
        super().__init__(
            gf.global_energies, gf.no, dtype=complex, oversample=oversample
        )

    def get_work_array(self) -> Optional[NDArray]:
        """Scratch array for the pair-orbital rotation, or ``None`` without one."""
        if self.U_pq is None:
            return None
        return xp.empty_like(self.U_pq)

    def solve_correlation(self) -> None:
        r"""Compute :math:`\Sigma^r(E)`, :math:`\Sigma^<(E)` from the current :math:`G` and :math:`W`."""

        if self.gf.domain == "e":
            # At this point gf has been updated.
            # If gf is the same as wrpa.gf, then domain is already t
            # Else, it needs to be converted.
            self.gf.convert_domain()
        Gl = self.gf.arrays["l"]
        Gg = self.gf.arrays["g"]
        Wl = self.wrpa.arrays["l"]
        Wg = self.wrpa.arrays["g"]

        self.arrays["g"] = (
            self.arrays["g"] if "g" in self.arrays.keys() else self.arrays.pop("r")
        )
        GWl = self.arrays["l"]
        GWg = self.arrays["g"]

        # Prefactor: j. from definition, and de / 2 pi for fft
        pre = 1.0j * self.de / (2 * np.pi)

        work1 = self.get_work_array()
        work2 = self.get_work_array()

        for t in range(Gl.shape[0]):
            for G, W, GW in [[Gl[t], Wl[t], GWl[t]], [Gg[t], Wg[t], GWg[t]]]:
                exchange_product(
                    G, W, pre=pre, out=GW, rot=self.U_pq, work1=work1, work2=work2,
                )

        self.domain = "t"
        self.convert_less_and_great_to_ret(
            self.zero_index, override="g"
        )  # domain << energy

    def update_correlation(self) -> None:
        r"""Recompute :math:`W` from the current :math:`G`, then the self-energy."""
        # update lesser greater green's

        self.wrpa.update_screening()  # Wr(e), W>(e)
        self.wrpa.convert_retarded()  # W<(e), W>(e)
        self.wrpa.convert_domain()  # W<(t), W>(t)
        self.solve_correlation()  # GWr(e), GW<(e)

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded self-energy :math:`\Sigma^r` at the first grid energy not below ``energy``.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, on the grid; no interpolation is done.
        """
        return self.arrays["r"][np.searchsorted(self.energies, energy)]

    def lesser(self, energy: float) -> NDArray[np.complex128]:
        r"""Lesser self-energy :math:`\Sigma^<` at the first grid energy not below ``energy``.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, on the grid; no interpolation is done.
        """
        return self.arrays["l"][np.searchsorted(self.energies, energy)]


# Constrained methods


def cut_rotation(U_pq: NDArray, indices: Sequence[int]) -> NDArray:
    """Rows of a pair-orbital rotation that belong to pairs within a subset of orbitals.

    Parameters
    ----------
    U_pq : ndarray, shape (n_orb**2, n_q)
        Rotation from all orbital pairs to optimized pair orbitals.
    indices : sequence of int
        Orbitals of the subset.

    Returns
    -------
    U_cq : ndarray, shape (len(indices)**2, n_q)
    """
    no2, nq = U_pq.shape
    no = int(sqrt(no2))
    assert no ** 2 == no2, "# p is not square of orbitals."
    U_cq = np.ascontiguousarray(
        U_pq.reshape(no, no, nq)[np.ix_(indices, indices)].reshape(
            len(indices) ** 2, nq
        )
    )
    return U_cq


class cRPA(LangrethPair):
    r"""Partially screened interaction :math:`U(E)` of a correlated subspace, by constrained RPA.

    When the correlated orbitals (for example a :math:`d` shell) are treated
    by an impurity solver, their own screening is already part of that
    solution and must not be counted twice. The polarization from
    transitions within the subspace is therefore set to zero before
    screening, and the resulting interaction is restricted to the subspace:

    .. math::

        U^r = \big[(1 - V P^r_\mathrm{rest})^{-1} V\big]_{cc}.

    After :meth:`update_unscreening` the array ``arrays['r']`` holds
    :math:`U^r(E)`.

    Parameters
    ----------
    wrpa : WRPA
        Screened-interaction calculation of the whole region.
    ic : ndarray of int
        Orbitals of the correlated subspace.
    """

    def __init__(self, wrpa: WRPA, ic: NDArray[np.int_]) -> None:
        self.U_qq = None  # sets to 0. polarization of `c` subspace
        self.U_cq = None
        if wrpa.U_pq is not None:
            U_pq = wrpa.U_pq
            U_qp = U_pq.T
            U_pr = np.eye(U_pq.shape[0], dtype=U_pq.dtype)
            #
            self.U_qq = U_qp.dot(U_pr).dot(U_pq)
            self.U_cq = cut_rotation(U_pq, ic)
        super().__init__(wrpa.global_energies, len(ic)**2, dtype=complex)
        self.wrpa = wrpa
        self.ic = ic
        self.ic4 = np.ix_(self.ic, self.ic, self.ic, self.ic)
        self.arrays["r"] = self.arrays.pop("l")
        self.arrays.pop("g")  # not required
        # self.domain = "e"

    def _screen_polarization(self, P, work=None):
        if self.U_qq is None:
            no = int(sqrt(self.no))
            P = P.reshape(4 * (no,))
            P[self.ic4] = 0.0
        else:
            work = P.dot(self.U_qq, out=work)
            return self.U_qq.T.dot(work, out=P)
            # return rotate(P, self.U_qq, work=work, overwrite_a=True)

    # @assert_domain("e")
    def screen_polarization(self) -> None:
        """Remove the polarization of transitions within the correlated subspace."""
        Pr = self.wrpa.arrays["r"]
        work = (
            None
            if self.U_qq is None
            else np.empty(self.U_qq.shape, dtype=self.U_qq.dtype)
        )
        for e in range(Pr.shape[0]):
            self._screen_polarization(Pr[e], work=work)

    def _cut_screening(self, W, work=None, out=None):
        if self.U_cq is None:
            no = int(sqrt(self.no))
            W = W.reshape(4 * (no,))
            return W[self.ic4]
        else:
            work = W.dot(self.U_cq.T, out=work)
            return self.U_cq.dot(work, out=out)
            # return rotate(W, self.U_cq, work=work, out=out)

    # @assert_domain("e")
    def solve_unscreening(self) -> None:
        """Restrict the screened interaction to the correlated subspace."""
        Wr = self.wrpa.arrays["r"]
        Ur = self.arrays["r"]
        work = (
            None
            if self.U_cq is None
            else np.empty(self.U_cq.T.shape, dtype=self.U_cq.dtype)
        )
        for e in range(Wr.shape[0]):
            self._cut_screening(Wr[e], work=work, out=Ur[e])

    def update_unscreening(self) -> None:
        r"""Compute :math:`P`, remove its subspace part, screen, and restrict to the subspace."""
        # update lesser greater green's
        self.wrpa.update_polarization()  # Pr(e), P>(e)
        self.screen_polarization()  # Pr'(e) :: ' means `c` subspace is 0.
        self.wrpa.solve_screening()  # Wr(e)
        self.solve_unscreening()

    # def __init__(self, gf: DistGreenFunction, V_qq, U_pq=None, oversample=10) -> None:
    #     # super().__init__(DistGreenFunction(gf, energies), V_qq, U_pq)
    #     Constrained.__init__(self, gf, U_pq)
    #     WRPA.__init__(self, gf.parent, V_qq, U_pq, oversample=oversample)
    #     self.chi = Chi(self.gfc, oversample=oversample)
    #     super().__init__(energies, no, dtype, oversample)

    #     self.rotate = self._cut if U_pq is None else self._rot

    # def _cut(self, W, out):
    #     out[:] = W.reshape(4 * (self.gf.no,))[self.ix4].reshape(2 * (len(self.chi.no),))
    #     return out

    # def _rot(self, W, out):
    #     return self.U_cq.dot(W, out=self.work).dot(self.U_cq.T, out=out)

    # @property
    # def W(self):
    #     raise NotImplementedError("Not available with this method.")

    # @property
    # def U(self):
    #     return self.chi.arrays["r"]

    # @assert_domain("e")
    # def screen_polarization(self):
    #     """Set to zero the polarization of the constrained region."""
    #     Pr = self.arrays["r"]
    #     no = self.gf.no
    #     rotate = lambda X, U: X
    #     if self.U_pq is not None:
    #         rotate = lambda X, U: U.dot(X).dot(U.T)
    #     for e in range(self.energies.size):
    #         P_pp = rotate(Pr[e], self.U_pq)
    #         P_pp = P_pp.reshape((no,) * 4)
    #         P_pp[self.ix4] = 0.0
    #         P_pp = P_pp.reshape((no ** 2,) * 2)
    #         Pr[e, ...] = rotate(P_pp, self.U_pq.T)

    # @assert_domain("e")
    # def solve_unscreening(self):
    #     W = self.arrays["r"]
    #     self.chi.arrays["r"] = self.chi.arrays.pop("l")
    #     U = self.chi.arrays["r"]
    #     for e in range(self.energies.size):
    #         self.rotate(W[e], out=U[e])

    # def update_unscreening2(self):
    #     self.update_polarization()  # Pr(e), P>(e)
    #     self.screen_polarization()
    #     self.solve_screening()  # Wr(e), W>(e)
    #     self.solve_unscreening()  # Ur(e)


# class Constrained:
#     """Base class for constrained calculations.

#     Args:
#         gf : qtpyt:DistGreenFunction
#             distributed green's function object.
#         indices : 1D array
#             indices of constrained subspace.
#         U_pq : 2D array (optional)
#             see Chi
#     """

#     def __init__(self, gf: DistGreenFunction, U_pq=None) -> None:
#         assert isinstance(
#             gf.gf0.projector, BaseProjector
#         ), "Invalid parent green's function for constrained region."
#         self.gfc = gf
#         idx = self.indices
#         self.ix = np.ix_(idx, idx)
#         self.ix4 = np.ix_(idx, idx, idx, idx)

#         if U_pq is None:
#             U_cq = None

#         else:
#             U_cq = self.cut_rotation(U_pq, idx)

#         self.U_cq = U_cq

#     @property
#     def indices(self):
#         return self.gfc.gf0.projector.indices

#     @staticmethod
#     def cut_rotation(U_pq, indices):
#         """Cut pairorbital rotation matrix."""
#         no2, nq = U_pq.shape
#         no = int(sqrt(no2))
#         assert no ** 2 == no2, "# p is not square of orbitals."
#         U_cq = np.ascontiguousarray(
#             U_pq.reshape(no, no, nq)[np.ix_(indices, indices)].reshape(
#                 len(indices) ** 2, nq
#             )
#         )
#         return U_cq

#     def get_cwork_array(self):
#         if self.U_cq is None:
#             return None
#         else:
#             return xp.empty_like(self.U_cq)


# class cGW(Constrained, GW):
#     """Constrained GW calculation.

#     The GW selfenergy is calculated only for a subspace of
#     the WRPA region where the screening is computed.
#     """

#     def __init__(self, gf: DistGreenFunction, V_qq, U_pq, oversample=10) -> None:
#         # super().__init__(DistGreenFunction(gf, energies), V_qq, U_pq)
#         Constrained.__init__(self, gf, U_pq)
#         GW.__init__(self, gf.parent, V_qq, U_pq, oversample=oversample)

#     def __getattr__(self, attr):
#         """When """
#         if attr in ['gf','U_pq','get_work_array']:
#             return getattr(self.wrpa, attr)
#         raise AttributeError

#     @assert_domain("t")
#     def solve_correlation(self):

#         if self.gf.domain == "e":
#             self.gf.convert_domain()
#         self.gfc.update_from_parent(self.gf)
#         Gl = self.gfc.arrays["l"]
#         Gg = self.gfc.arrays["g"]
#         Wl = self.arrays["l"]
#         Wg = self.arrays["g"]

#         # Prefactor: j. from definition, and de / 2 pi for fft
#         pre = 1.0j * self.de / (2 * np.pi)

#         work1 = self.get_cwork_array()
#         work2 = self.get_cwork_array()

#         for t in range(Gl.shape[0]):
#             for G, W in [[Gl[t], Wl[t]], [Gg[t], Wg[t]]]:
#                 exchange_product(
#                     G,
#                     W,
#                     pre=pre,
#                     out=G,  # OK to overwrite
#                     rot=self.U_cq,
#                     # rotd=self.U_qp,
#                     work1=work1,
#                     work2=work2,
#                 )

#         self.gfc.convert_less_and_great_to_ret(self.zero_index)  # domain << energy
#         self.gfc.domain = None

#     def retarded(self, energy):
#         """The retarded GW self-energy."""
#         return self.gfc.arrays["r"][np.searchsorted(self.energies, energy)]

#     def greater(self, energy):
#         return self.gfc.arrays["g"][np.searchsorted(self.energies, energy)]

#     def lesser(self, energy):
#         Gr = self.retarded(energy)
#         Gg = self.retarded(energy)
#         return lesser_from_retarded(self.retarded(energy), self.gfc.arr

# class cRPA(Constrained, WRPA):
#     """Constrained RPA."""

#     def __init__(self, gf: DistGreenFunction, V_qq, U_pq=None, oversample=10) -> None:
#         # super().__init__(DistGreenFunction(gf, energies), V_qq, U_pq)
#         Constrained.__init__(self, gf, U_pq)
#         WRPA.__init__(self, gf.parent, V_qq, U_pq, oversample=oversample)
#         self.chi = Chi(self.gfc, oversample=oversample)

#         self.work = self.get_cwork_array()
#         self.rotate = self._cut if self.work is None else self._rot

#     def _cut(self, W, out):
#         out[:] = W.reshape(4 * (self.gf.no,))[self.ix4].reshape(2 * (len(self.chi.no),))
#         return out

#     def _rot(self, W, out):
#         return self.U_cq.dot(W, out=self.work).dot(self.U_cq.T, out=out)

#     @property
#     def W(self):
#         return self.chi.arrays["g"]

#     @property
#     def U(self):
#         return self.chi.arrays["r"]

#     @assert_domain("e")
#     def solve_unscreening(self):
#         #       r      r       r  r  ^-1
#         #      U    = W [ I + P  W ]
#         Pr = self.chi.arrays["r"]
#         Pg = self.chi.arrays["g"]
#         Wr = self.arrays["r"]

#         I = np.eye(self.chi.no)

#         for e in range(self.energies.size):
#             cWr = self.rotate(Wr[e], out=Pg[e])
#             cWr.dot(np.linalg.inv(I + Pr[e].dot(cWr)), out=Pr[e])

#     def update_unscreening(self):
#         self.update_screening()  # Wr(e), W>(e)
#         self.gfc.update_from_parent(self.gf)  # g<(t), g>(t)
#         self.chi.update_polarization()  # Pr(e), P>(e)
#         self.solve_unscreening()

