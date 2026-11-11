r"""Pulay (DIIS) mixing to reach self-consistency of the density matrix."""
from __future__ import annotations

from typing import Any, Callable, Optional

from numpy.typing import NDArray

import numpy as np
from numba import njit, prange
from qtpyt.base._kernels import dagger
from qtpyt.parallel.tools import comm_sum
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.tools import tri2full

diag = lambda m: m.diagonal(axis1=1, axis2=2)


@njit("(f8,f8[:],c16[:,:,:,::1],c16[:,:,:,::1],c16[:,:,::1])", parallel=True)
def _mix(alpha, coeff, inp, out, new):
    """Mix input-output arrays.
         __
         \
    new =     (1-alpha) coeff[n] inp[n] + alpha coeff[n] out[n]
         /_ n  
    """
    ne, m, n = out.shape[1:]
    nk = len(coeff)
    for e in prange(ne):
        for i in range(m):
            for j in range(n):
                new[e, i, j] = 0.0
                for k in range(nk):
                    new[e, i, j] += (1 - alpha) * coeff[k] * inp[
                        k, e, i, j
                    ] + alpha * coeff[k] * out[k, e, i, j]


class ArrayList:
    """Ring buffer that keeps the last ``N`` arrays of an iteration.

    Parameters
    ----------
    N : int
        Number of arrays kept.

    Attributes
    ----------
    array : ndarray, shape (N, ...)
        Storage; ``array[indices]`` lists the kept arrays from oldest to
        newest.
    indices : list of int
    """

    def __init__(self, N: int) -> None:
        self.array = None
        self.N = N
        self.indices = []
        self.i = -1

    def append(self, X: NDArray) -> None:
        """Store ``X``, dropping the oldest array when full.

        Parameters
        ----------
        X : ndarray
        """
        if self.array is None:
            self.array = np.empty((self.N,) + X.shape, X.dtype)
        self.i += 1
        self.i %= self.N
        self.indices.append(self.i)
        if len(self) > self.N:
            self.indices.pop(0)
        self.array[self.i] = X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.array[i]


# class Pulay:
#     """Pulay mixing scheme.

#          __
#          \
#     new =     (1-alpha) coeff[n] inp[n] + alpha coeff[n] out[n]
#          /_ n

#     Args:
#         alpha : float
#             mixing factor
#         N : # of iterations to store
#     """

#     def __init__(self, alpha=0.4, N=3):
#         self.alpha = alpha
#         self.N = N
#         self.Gr_inp = []  # ArrayList(N)
#         self.Gr_out = []  # ArrayList(N)
#         self.eps = []

#     def _append(self, G_list, G):
#         G_list.append(G)
#         if len(G_list) > self.N:
#             G_list.pop(0)

#     def append_input(self, gf: DistGreenFunction):
#         """Append input."""
#         self._append(self.Gr_inp, gf.arrays["r"].copy())

#     def append_output(self, gf: DistGreenFunction):
#         """Append output."""
#         self._append(self.Gr_out, gf.arrays["r"].copy())
#         # Compute inner product with all previous (and current) residuals.
#         N = len(self.Gr_out)
#         i = N - 1
#         eps = [None] * N
#         for j in range(N):
#             eps[j] = np.dot(
#                 (self.Gr_out[i] - self.Gr_inp[i]).flat,
#                 (self.Gr_out[j] - self.Gr_inp[j]).flat,
#             ) + np.dot(
#                 (self.Gr_out[i].real - self.Gr_inp[i].real).flat,
#                 (self.Gr_out[j].real - self.Gr_inp[j].real).flat,
#             )
#         # Append lower(upper) triangular entries of Pulay matrix.
#         self.eps.append(eps)
#         if len(self.eps) > self.N:
#             self.eps.pop(0)  # Discard row
#             for eps in self.eps[:-1]:
#                 eps.pop(0)  # Discard column

#     def compute_new_coeff(self):
#         """Compute Pulay's mixing coefficients."""
#         N = len(self.Gr_out)
#         A = np.zeros((N + 1, N + 1))
#         B = np.zeros(N + 1)
#         B[N] = 1.0
#         # Fill lower triangular.
#         for i, eps in enumerate(self.eps):
#             A[i, : i + 1] = eps
#         # Fill symmetric.
#         tri2full(A)
#         A = comm_sum(A)
#         A[:N, N] = 1.0
#         A[N, :N] = 1.0
#         self.A = A
#         self.coeff = np.linalg.solve(A, B)[:-1]
#         return self.coeff

#     def compute_new_input(self, G_inp: ArrayList, G_out: ArrayList, new=None):
#         """Mix previous inputs and outputs and produce new input."""
#         if new is None:
#             new = np.zeros_like(G_inp[0])
#         else:
#             new[:] = 0.0
#         coeff = self.compute_new_coeff()
#         for i in range(len(coeff)):
#             new += (1 - self.alpha) * coeff[i] * G_inp[i] + self.alpha * coeff[
#                 i
#             ] * G_out[i]
#         # _mix(self.alpha, coeff, G_inp.array, G_out.array, new)
#         return new

#     def update_input(self, gf: DistGreenFunction):
#         """Update Green's function arrays with input computed from Pulay."""
#         Gr = gf.arrays["r"]
#         Gl = gf.arrays["l"]
#         self.compute_new_input(self.Gr_inp, self.Gr_out, Gr)
#         fermi = gf.gf0.fermi
#         for e, energy in enumerate(gf.energies):
#             Gl[e] = -fermi(energy) * (Gr[e] - dagger(Gr[e]))
#         self.append_input(gf)

#     def initialize(self, gf: DistGreenFunction, **kwargs):
#         """Update Green's function and initialize Pulay."""
#         child = None
#         if gf.parent is not None:
#             child = gf
#             gf = gf.parent
#         gf.update(lg=False, **kwargs)
#         self.append_input(gf)
#         gf.convert_retarded()
#         if child is not None:
#             child.update_from_parent(gf)

#     def step(self, gf: DistGreenFunction, **kwargs):
#         """Update Green's function and Pulay's internals."""
#         child = None
#         if gf.parent is not None:
#             child = gf
#             gf = gf.parent
#         gf.update(lg=False, **kwargs)
#         self.append_output(gf)
#         self.update_input(gf)  # << Gr, G<
#         gf.convert_retarded()  # << G<, G>
#         if child is not None:
#             child.update_from_parent(gf)


class PulayMixer:
    r"""Pulay (DIIS) mixing of density matrices for a self-consistent loop.

    Each step maps an input density matrix :math:`D^\mathrm{in}` to an output
    :math:`D^\mathrm{out}`. The next input combines the last ``N`` steps,

    .. math::

        D^\mathrm{new} = \sum_n c_n \left[D^\mathrm{in}_n
        + \alpha (D^\mathrm{out}_n - D^\mathrm{in}_n)\right],
        \qquad \sum_n c_n = 1,

    with :math:`c_n` chosen to minimize the norm of the combined residual.
    Residuals are measured on the diagonal (orbital occupations) only.

    Parameters
    ----------
    alpha : float, default 0.4
        Fraction of the output mixed in.
    N : int, default 3
        Number of previous steps kept.
    """

    def __init__(self, alpha: float = 0.4, N: int = 3) -> None:
        self.alpha = alpha
        self.N = N
        self.inp = []  # ArrayList(N)
        self.out = []  # ArrayList(N)
        self.err = []
        self.coeff = None

    @property
    def eps(self) -> float:
        """Squared norm of the latest residual (infinite before the first step)."""
        if not hasattr(self, "A"):
            return np.inf
        return self.A[-2, -2]

    def _append(self, G_list, G):
        G_list.append(G)
        if len(G_list) > self.N:
            G_list.pop(0)

    def append_input(self, D: np.ndarray) -> None:
        """Record an input density matrix.

        Parameters
        ----------
        D : ndarray
        """
        self._append(self.inp, D)

    def append_output(self, D: np.ndarray) -> None:
        """Record the output density matrix of the latest input and its residual.

        Parameters
        ----------
        D : ndarray
        """
        self._append(self.out, D)
        # Compute inner product with all previous (and current) residuals.
        N = len(self.out)
        i = N - 1
        err = [None] * N
        for j in range(N):
            err[j] = np.inner(
                (self.out[i] - self.inp[i]).diagonal(),
                (self.out[j] - self.inp[j]).diagonal(),
            ).real
        # Append lower(upper) triangular entries of Pulay matrix.
        self.err.append(err)
        if len(self.err) > self.N:
            self.err.pop(0)  # Discard row
            for err in self.err[:-1]:
                err.pop(0)  # Discard column

    def compute_new_coeff(self) -> NDArray[np.float64]:
        r"""Mixing coefficients :math:`c_n` that minimize the residual norm with :math:`\sum_n c_n = 1`."""
        from random import random

        N = len(self.out)
        A = np.zeros((N + 1, N + 1))
        B = np.zeros(N + 1)
        A[N, :N] = 1
        A[:N, N] = 1
        B[N] = 1
        # Fill lower triangular.
        # for i in range(N):
        #     for j in range(i + 1):
        #         A[i, j] = A[j, i] = np.vdot(
        #             self.out[i] - self.inp[i], self.out[j] - self.inp[j],
        #         ).real
        for i, err in enumerate(self.err):
            A[i, : i + 1] = err
        # Fill symmetric.
        # A[N, :N] = A[:N, N] = 1.0
        # A[:N, N] = 1.0
        tri2full(A)
        # A = comm_sum(A)
        self.coeff = np.linalg.solve(A, B)[:N]
        self.A = A
        return self.coeff

    def compute_new_input(self, new: Optional[NDArray] = None) -> NDArray:
        """Next input density matrix from the stored steps.

        Parameters
        ----------
        new : ndarray, optional
            Array for the result.
        """
        if new is None:
            new = np.zeros_like(self.inp[0])
        else:
            new[:] = 0.0
        coeff = self.compute_new_coeff()
        for i in range(len(coeff)):
            # new += (1 - self.alpha) * coeff[i] * self.inp[i] + self.alpha * coeff[
            #     i
            # ] * self.out[i]
            new += self.coeff[i] * (
                self.inp[i] + self.alpha * (self.out[i] - self.inp[i])
            )
        return new

    def __call__(self, func: Callable, D_inp: NDArray, tol: float = 1e-5, max_iter: int = 100, max_outer: int = 1, **kwargs: Any) -> Any:
        """Iterate ``D -> func(D)`` with Pulay mixing until the residual falls below ``tol``.

        Parameters
        ----------
        func : callable
            ``func(D, **kwargs)`` returns the output density matrix for input ``D``.
        D_inp : ndarray
            Starting density matrix.
        tol : float, default 1e-5
            Convergence threshold on :attr:`eps`.
        max_iter : int, default 100
            Steps per cycle.
        max_outer : int, default 1
            Cycles; each restarts the mixing history.
        **kwargs
            Passed to ``func``.

        Returns
        -------
        D : ndarray
            Last input density matrix.
        """
        iter_outer = 0
        while self.eps > tol and iter_outer < max_outer:
            iter = 0
            self.inp = []
            self.out = []
            self.err = []
            self.append_input(D_inp)
            while self.eps > tol and iter < max_iter:
                iter += 1
                D_out = func(D_inp, **kwargs)
                self.append_output(D_out)
                # self.eps = abs((D_out - D_inp).real.sum())
                D_inp = self.compute_new_input()
                self.append_input(D_inp)
            iter_outer += 1
        return D_inp
