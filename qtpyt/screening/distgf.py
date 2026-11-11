r"""Green's function tabulated on an energy grid distributed over MPI ranks."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from numpy.typing import NDArray

from functools import lru_cache
from math import pi
from typing import TypeVar, Union

import numpy as np

from qtpyt import xp
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.screening.langreth import LangrethPair, assert_domain, change_domain
from qtpyt.screening.tools import greater_from_retarded, smooth

T = TypeVar("T", bound="DistGreenFunction")


class DistGreenFunction(LangrethPair):
    r"""Green's function of a region evaluated on a whole energy grid.

    The input to the many-body self-energies of :mod:`qtpyt.screening`:
    :math:`G^r(E)` and :math:`G^<(E)` (or :math:`G^<` and :math:`G^>`) at every
    grid energy, each MPI rank holding a slice of the energies.

    Parameters
    ----------
    gf : GreenFunction or ProjectedGreenFunction
        Green's function that is evaluated at each energy.
    energies : ndarray, shape (n_E,)
        Uniform energy grid, same unit as the Hamiltonian.
    parent : DistGreenFunction, optional
        Larger Green's function this one is a projection of; then
        :meth:`update` updates the parent and projects, see
        :meth:`take_subspace`.

    Attributes
    ----------
    gf0 : GreenFunction
        The Green's function passed as ``gf``.
    """

    def __init__(
        self,
        gf: Union[ProjectedGreenFunction, GreenFunction],
        energies: NDArray[np.float64],
        parent: T = None,
    ) -> None:

        self.gf0 = gf
        self.parent = parent
        super().__init__(energies, gf.H.shape[0], complex)

    def update_from_parent(self, other: "DistGreenFunction") -> None:
        """Fill the arrays by projecting those of a larger Green's function.

        Array keys and domain are copied from ``other``.

        Parameters
        ----------
        other : DistGreenFunction
            Green's function of a region that contains this one.
        """
        project = self.gf0.projector.project
        old_k = tuple(self.arrays.keys())
        new_k = tuple(other.arrays.keys())
        for ok, nk in zip(old_k, new_k):
            for e in range(self.energies.size):
                self.arrays[ok][e] = project(other.arrays[nk][e])
        self.arrays = {new_k[0]: self.arrays[old_k[0]], new_k[1]: self.arrays[old_k[1]]}
        self.domain = other.domain

    @change_domain("e")
    def update(self, *, callback: Optional[Callable[[int, float], None]] = None, lg: bool = True) -> None:
        r"""Evaluate :math:`G^r(E)` and :math:`G^<(E)` at every local grid energy.

        Parameters
        ----------
        callback : callable, optional
            Called as ``callback(e, energy)`` before each energy, for
            progress reports.
        lg : bool, default True
            Afterwards replace :math:`G^r` by :math:`G^>`, so the arrays hold
            :math:`G^<` and :math:`G^>`.
        """
        # Simply project from parent
        if self.parent is not None:
            self.parent.update(callback=callback, lg=lg)
            self.update_from_parent(self.parent)
        # Actually compute
        else:
            asnumpy = lambda a: a  # Identity
            if xp.__name__ == "cupy":
                asnumpy = xp.asnumpy
            if callback is None:
                callback = lambda e, energy: None  # Identity

            self.arrays["r"] = (
                self.arrays["r"] if "r" in self.arrays.keys() else self.arrays.pop("g")
            )
            Gr = self.arrays["r"]
            Gl = self.arrays["l"]

            for e, energy in enumerate(self.energies):
                callback(e, energy)
                Gr[e] = asnumpy(self.gf0.retarded(energy))
                Gl[e] = asnumpy(self.gf0.lesser(energy))
            if lg:
                self.domain = "e"
                self.convert_retarded()

    @assert_domain("e")
    def convert_retarded(self) -> None:
        r"""Replace the retarded component by the greater one, :math:`G^> = G^< + G^r - G^a`."""
        greater_from_retarded(self.arrays["r"], self.arrays["l"])
        self.arrays["g"] = self.arrays.pop("r")

    @assert_domain("e")
    def get_density_matrix(self) -> NDArray[np.complex128]:
        r"""Single-particle density matrix, for one spin, from the lesser Green's function.

        .. math::

            D = -\frac{i}{2\pi} \int G^<(E)\,dE,

        integrated with the trapezoidal rule over the energy grid. The grid
        must cover the occupied spectrum.

        Returns
        -------
        D : ndarray, shape (n_orb, n_orb)
        """
        gl = self.arrays["l"]
        # Gl = self.collect_energies(gl)
        # Gl = smooth(self.global_energies, Gl, 1.0, 6.0)
        # self.collect_orbitals(Gl, gl)
        return self.sum_energies(gl, pre=-1.0j / (2 * np.pi))

    @assert_domain("e")
    def get_dos(self, out: Optional[NDArray[np.float64]] = None, root: int = 0) -> NDArray[np.float64]:
        r"""Density of states on the energy grid, :math:`\operatorname{Tr}[A(E) S]/2\pi` with :math:`A = i(G^> - G^<)`.

        Parameters
        ----------
        out : ndarray, optional
            Array for the local energies.
        root : int, default 0
            MPI rank that receives the full grid.

        Returns
        -------
        dos : ndarray, shape (n_E,)
            On rank ``root``; ``None`` on other ranks when running in parallel.
        """
        # g> - g< = gr - ga
        # A = Re[1.j * (gr - ga)]
        # DOS = Tr[A S] / (2 pi)
        if out is None:
            out = np.empty(self.energies.size)
        for e in range(self.energies.size):
            out[e] = (
                1.0j
                * (self.arrays["g"][e] - self.arrays["l"][e]).dot(self.gf0.S).trace()
            ).real / (2 * np.pi)
        return self.gather_energies(out, root)

    def get_num_electrons(self) -> float:
        r"""Number of electrons :math:`2\operatorname{Tr}[DS]`, both spins."""
        D = self.get_density_matrix()
        S = self.gf0.S
        # x2. degenerate spin
        return 2.0 * D.dot(S).real.trace()

    def take_subspace(self, indices: Sequence[int]) -> "DistGreenFunction":
        """Distributed Green's function of a subspace of orbitals, projected from this one.

        This object becomes the parent: updating the subspace updates this
        Green's function first.

        Parameters
        ----------
        indices : sequence of int
            Orbitals of the subspace, see :class:`~qtpyt.projector.ProjectedGreenFunction`.
        """
        return DistGreenFunction(
            ProjectedGreenFunction(self.gf0, indices), self.global_energies, parent=self
        )
