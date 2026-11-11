r"""Arrays of matrices on an energy grid, distributed over MPI ranks in two layouts.

An array :math:`A_{ij}(E_k)` can be split over ranks in two ways. Split by
energy ("orbital aligned"), each rank holds full matrices at a slice of
energies, which suits matrix operations at fixed energy. Split by matrix
element ("energy aligned"), each rank holds the whole energy dependence of a
slice of elements, which suits Fourier transforms and convolutions in energy.
:class:`DistGrid` moves arrays between the two layouts.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from numpy.typing import NDArray

import itertools
from functools import wraps
from warnings import warn

import numpy as np
from qtpyt.parallel import MPI, comm
from qtpyt.parallel.pencil import Pencil, Subcomm, _blockdist
# from qtpyt.parallel.tools import comm_sum
from qtpyt.parallel.transpose import transpose


def check_contiguous(dist_func: Callable) -> Callable:
    """Decorator: make input and output arrays C-ordered, transposing Fortran-ordered ones in place.

    Parameters
    ----------
    dist_func : callable
    """

    def c_contiguous(a: np.ndarray):
        if a.flags.f_contiguous:
            return transpose(a.T)
        return a

    @wraps(dist_func)
    def wrapper(self, a, b=None):
        """Transpose input and output inplace if necessary."""
        a = c_contiguous(a)
        b = dist_func(self, a, b)
        return c_contiguous(b)

    return wrapper


def check_serial(dist_func: Callable) -> Callable:
    """Decorator: with a single MPI rank, return the input (or copy it into ``b``) without communication.

    Parameters
    ----------
    dist_func : callable
    """
    if comm.size != 1:
        return dist_func

    @wraps(dist_func)
    def collect_serial(self, a, b=None):
        if b is not None and a is not b:
            b[:] = a
            return b
        return a

    return collect_serial


def reshape(arg: str) -> Callable:
    """Decorator factory: reshape between ``(n_E, n, n)`` and ``(n_E, n*n)`` around a redistribution.

    Parameters
    ----------
    arg : {'input', 'output'}
        Reshape the input array, or the output array (and back afterwards).
    """
    def reshape_array(a, N):
        if a is not None:
            a = a.reshape(N)
        return a

    if arg == "input":

        def reshape_input(dist_func):
            """Reshape input array from energy-orbital matrix
            to energy-flatten orbital index."""

            @wraps(dist_func)
            def wrapper(self, a, b=None):
                a = reshape_array(a, self.N2D)
                return dist_func(self, a, b)

            return wrapper

        return reshape_input

    elif arg == "output":

        def reshape_output(dist_func):
            """Same as `reshape_input` for output array but also
            to the inverse reshaping at the end."""

            @wraps(dist_func)
            def wrapper(self, a, b=None):
                b = reshape_array(b, self.N2D)
                b = dist_func(self, a, b)
                return reshape_array(b, self.N3D)

            return wrapper

        return reshape_output


class DistGrid:
    """Layouts and redistribution of an ``(n_E, n, n)`` array over MPI ranks.

    Parameters
    ----------
    ne : int
        Number of energies.
    no : int
        Matrix size.
    dtype : data-type, default complex

    Attributes
    ----------
    N3D : tuple of int
        Local shape ``(n_E_local, n, n)`` in the orbital-aligned layout.
    """

    def __init__(self, ne: int, no: int, dtype: Any = complex) -> None:
        subcomms = Subcomm(comm, [0, 1])
        # Array aligned in orbital (1-axis).
        self._pencil_aligned_orbs = Pencil(subcomms, (ne, no ** 2), 1)
        # Array aligned in energy (0-axis).
        # try:
        self._pencil_aligned_eners = self._pencil_aligned_orbs.pencil(0)
        self.transfer = self._pencil_aligned_orbs.transfer(
            self._pencil_aligned_eners, dtype
        )
        # except:
        #     warn("Tranfer not implemented.")
        self.dtype = dtype
        # Array shapes
        self.N2D = self._pencil_aligned_orbs.subshape
        self.N3D = (self._pencil_aligned_orbs.subshape[0], no, no)
        # Global shapes
        self.ne = ne
        self.no = no

    def shape(self) -> tuple:
        """Global ``(n_E, n)``."""
        return (self.ne, self.no)

    def _assert_dtypes(self, *arrays: np.ndarray):
        for a in arrays:
            assert a.dtype == self.dtype

    @reshape("input")
    @check_contiguous
    @check_serial
    def collect_energies(self, a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
        """Redistribute from the orbital-aligned to the energy-aligned layout.

        Parameters
        ----------
        a : ndarray, shape (n_E_local, n, n)
        b : ndarray, optional
            Array for the result.

        Returns
        -------
        b : ndarray, shape (n_E, n_elements_local)
        """
        if b is None:
            b = self.empty_aligned_eners()
        self._assert_dtypes(a, b)
        self.transfer.forward(a, b)
        return b

    @reshape("output")
    @check_contiguous
    @check_serial
    def collect_orbitals(self, a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
        """Redistribute from the energy-aligned to the orbital-aligned layout.

        Parameters
        ----------
        a : ndarray, shape (n_E, n_elements_local)
        b : ndarray, optional
            Array for the result.

        Returns
        -------
        b : ndarray, shape (n_E_local, n, n)
        """
        if b is None:
            b = self.empty_aligned_orbs().reshape(self.N2D)
        self._assert_dtypes(a, b)
        self.transfer.backward(a, b)
        return b

    def empty_aligned_eners(self) -> np.ndarray:
        """Uninitialized local array in the energy-aligned layout."""
        return np.empty(self._pencil_aligned_eners.subshape, self.dtype)

    def empty_aligned_orbs(self) -> np.ndarray:
        """Uninitialized local array ``(n_E_local, n, n)`` in the orbital-aligned layout."""
        return np.empty(self.N3D, self.dtype)


class GridDesc(DistGrid):
    """Distributed layout together with the energy grid it refers to.

    Parameters
    ----------
    energies : ndarray, shape (n_E,)
        Uniform energy grid.
    no : int
        Matrix size.
    dtype : data-type, default complex

    Attributes
    ----------
    global_energies : ndarray
        The full grid.
    zero_index : int
        Index of the grid energy closest to zero.
    de : float
        Grid spacing.
    """

    def __init__(self, energies: NDArray[np.float64], no: int, dtype: Any = complex) -> None:
        self.global_energies = energies
        super().__init__(energies.size, no, dtype)
        # Helpers for subclasses
        self.zero_index = abs(self.global_energies).argmin()
        self.de = abs(self.global_energies[1] - self.global_energies[0])

    @property
    def oslice(self) -> slice:
        """Matrix elements held by this rank in the energy-aligned layout."""
        substart = self._pencil_aligned_eners.substart[1]
        subshape = self._pencil_aligned_eners.subshape[1]
        return slice(substart, substart + subshape)

    @property
    def eslice(self) -> slice:
        """Energies held by this rank in the orbital-aligned layout."""
        substart = self._pencil_aligned_orbs.substart[0]
        subshape = self._pencil_aligned_orbs.subshape[0]
        return slice(substart, substart + subshape)

    @property
    def energies(self) -> NDArray[np.float64]:
        """Energies held by this rank."""
        return self.global_energies[self.eslice]

    def get_eintegral_weights(self) -> NDArray:
        """Trapezoidal-rule weights of the local energies."""
        if not hasattr(self, "weights"):
            # Trapezoidal rule
            weights = np.empty(self.ne, self.dtype)
            weights[1:-1] = self.global_energies[2:] - self.global_energies[:-2]
            weights[0] = self.global_energies[1] - self.global_energies[0]
            weights[-1] = self.global_energies[-1] - self.global_energies[-2]
            self.weights = weights[self.eslice]
            self.weights /= 2.0
        return self.weights

    def sum_energies(self, a: np.ndarray, pre: complex = 1.0) -> np.ndarray:
        r"""Energy integral :math:`c\int A(E)\,dE` over the whole grid, trapezoidal rule.

        Parameters
        ----------
        a : ndarray, shape (n_E_local, n, n)
            Orbital-aligned array.
        pre : complex, default 1.0
            Prefactor :math:`c`.

        Returns
        -------
        integral : ndarray, shape (n, n)
            The same on every rank.
        """
        assert a.shape == self.N3D, "Can only sum orbital aligned arrays."
        self.weights = self.get_eintegral_weights()
        a = np.tensordot(pre * self.weights, a, axes=([0], [0]))
        if comm.size == 1:
            return a
        b = np.empty_like(a)
        mpitype = MPI._typedict[a.dtype.char]
        comm.Allreduce([a, a.size, mpitype], [b, mpitype], MPI.SUM)
        return b

    def gather_energies(self, a: np.ndarray, root: int = 0) -> Optional[np.ndarray]:
        """Collect a quantity known at the local energies (such as a DOS) on one rank.

        Parameters
        ----------
        a : ndarray, shape (n_E_local,)
        root : int, default 0
            Receiving rank.

        Returns
        -------
        full : ndarray, shape (n_E,) or None
            On ``root``; ``None`` elsewhere when running in parallel.
        """
        assert a.ndim == 1, "Can only gather 1D arrays."
        if comm.size == 1:
            return a
        if comm.rank == root:
            b = np.empty(self.ne, a.dtype)
        else:
            b = None
        mpitype = MPI._typedict[a.dtype.char]
        counts = np.empty(comm.size, int)
        displs = np.empty(comm.size, int)
        for r in range(comm.size):
            counts[r], displs[r] = _blockdist(self.ne, comm.size, r)
        comm.Gatherv([a, a.size, mpitype], [b, counts, displs, mpitype], root)
        return b

    def write(self, a: np.ndarray, filename: str, condition: Optional[NDArray[np.bool_]] = None) -> None:
        """Write an orbital-aligned array to one binary file, all ranks at once (MPI-IO).

        Parameters
        ----------
        a : ndarray, shape (n_E_local, n, n)
        filename : str
        condition : ndarray of bool, shape (n_E,), optional
            Energies to write, for example
            ``(energies >= 0.0) & (energies <= 6.0)``. All if omitted.
        """
        assert (
            np.prod(a.shape[1:]) == self._pencil_aligned_orbs.subshape[1]
        ), "Must be orbital-aligned array."
        # indx = np.flatnonzero(condition)[self.eslice]
        if condition is None:
            condition = self.global_energies < np.inf
        indx = np.asarray(condition, bool)[self.eslice]
        indx = np.flatnonzero(indx)
        # comm = self.pencil_aligned_orbs.subcomm[0]
        send_buff = np.array([len(indx)], np.int32)
        recv_buff = np.empty(comm.size, np.int32)
        comm.Allgather([send_buff, 1, MPI.INT32_T], [recv_buff, 1, MPI.INT32_T])
        # Write to file
        amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
        fh = MPI.File.Open(comm, filename, amode)
        nbytes = a.itemsize * self._pencil_aligned_orbs.subshape[1]
        offset = recv_buff[: comm.rank].sum() * nbytes
        fh.Write_at_all(offset, a[indx])
        fh.Close()
