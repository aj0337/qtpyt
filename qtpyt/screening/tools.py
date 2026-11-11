r"""Occupations, Langreth identities and energy-grid utilities for the many-body module."""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

from numpy.typing import NDArray

import numpy as np
from numba import njit, prange
from scipy.constants import e, k
from scipy.interpolate import interp1d
from scipy.linalg.blas import get_blas_funcs

kB = k / e  # Boltzmann in eV.


def fermidistribution(energy: Union[float, NDArray[np.float64]], kt: float) -> Union[float, NDArray[np.float64]]:
    r"""Fermi–Dirac occupation :math:`f(x) = 1/(1 + e^{x/k_BT})` of an energy measured from the chemical potential.

    Parameters
    ----------
    energy : float or ndarray
        :math:`x = E - \mu`, same unit as ``kt``.
    kt : float
        Thermal energy :math:`k_B T`, non-negative. At zero the result is a
        step equal to 1 for :math:`x \le 0`.

    Returns
    -------
    f : float or ndarray

    Raises
    ------
    AssertionError
        If ``kt`` is negative.
    """
    # fermi level is fixed to zero
    # energy can be a single number or a list
    assert kt >= 0.0, "Negative temperature encountered!"

    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2.0 <= 0)
        else:
            return (energy / 2.0 <= 0).astype(int)
    else:
        return 1.0 / (1.0 + np.exp(energy / kt))


@njit("(c16[:,:,::1],c16[:,:,::1])", parallel=True, fastmath=True)
def lesser_from_retarded(r: NDArray[np.complex128], g: NDArray[np.complex128]) -> None:
    r"""Overwrite :math:`X^r` with :math:`X^< = X^> - X^r + X^a` at every energy.

    Parameters
    ----------
    r : ndarray of complex128, shape (n_E, n, n)
        Retarded component, overwritten with the lesser one.
    g : ndarray of complex128, shape (n_E, n, n)
        Greater component.
    """
    # < = > - r + a
    ne, n = r.shape[:2]
    for e in prange(ne):
        a = np.empty((n, n), r.dtype)
        # diag
        for i in range(n):
            r[e, i, i] = g[e, i, i] - r[e, i, i] + np.conjugate(r[e, i, i])
        # utrid
        for i in range(n - 1):
            for j in range(i + 1, n):
                a[i, j] = np.conjugate(r[e, i, j])
                r[e, i, j] = g[e, i, j] - r[e, i, j] + np.conjugate(r[e, j, i])
        # ltrid
        for i in range(1, n):
            for j in range(i):
                r[e, i, j] = g[e, i, j] - r[e, i, j] + a[j, i]


@njit("(c16[:,:,::1],c16[:,:,::1])", parallel=True, fastmath=True)
def greater_from_retarded(r: NDArray[np.complex128], l: NDArray[np.complex128]) -> None:
    r"""Overwrite :math:`X^r` with :math:`X^> = X^< + X^r - X^a` at every energy.

    Parameters
    ----------
    r : ndarray of complex128, shape (n_E, n, n)
        Retarded component, overwritten with the greater one.
    l : ndarray of complex128, shape (n_E, n, n)
        Lesser component.
    """
    # > = < + r - a
    ne, n = r.shape[:2]
    for e in prange(ne):
        a = np.empty((n, n), r.dtype)
        # diag
        for i in range(n):
            r[e, i, i] = l[e, i, i] + r[e, i, i] - np.conjugate(r[e, i, i])
        # utrid
        for i in range(n - 1):
            for j in range(i + 1, n):
                a[i, j] = np.conjugate(r[e, i, j])
                r[e, i, j] = l[e, i, j] + r[e, i, j] - np.conjugate(r[e, j, i])
        # ltrid
        for i in range(1, n):
            for j in range(i):
                r[e, i, j] = l[e, i, j] + r[e, i, j] - a[j, i]


def translate_along_axis(array: NDArray, translation: int) -> NDArray:
    """Copy of ``array`` cyclically shifted by ``translation`` along the first axis.

    Parameters
    ----------
    array : ndarray
    translation : int
    """
    newarray = array.copy()
    if translation == 0:
        return newarray
    newarray[:translation] = array[-translation:]
    newarray[translation:] = array[:-translation]
    return newarray


def roll(a: NDArray, shift: int) -> NDArray:
    """Cyclic shift of ``a`` along the first axis, in place, as :func:`numpy.roll`.

    Used to move :math:`t = 0` (or :math:`E = 0`) to the first grid point
    before a Fourier transform.

    Parameters
    ----------
    a : ndarray
    shift : int
    """
    n = a.shape[0]
    shift = shift % n
    perm = np.roll(np.arange(n), shift)
    tmpa = np.atleast_1d(np.empty_like(a[0]))

    def swap(a, i, j, tmp):
        tmp[:] = a[i]
        a[i] = a[j]
        a[j] = tmp[:]

    for i in range(n):
        swap(a, i, perm[i], tmpa)
        argmin = np.where((perm - i) == 0)[0][-1]
        perm[argmin] = perm[i]
        perm[i] = 0

    return a


def rotate(a: NDArray, u: NDArray, work: Optional[NDArray] = None, out: Optional[NDArray] = None, overwrite_a: bool = False) -> NDArray:
    r"""Product :math:`U A U^\dagger`, for example :math:`W^r P^> W^a`.

    Parameters
    ----------
    a, u : ndarray, shape (n, n)
    work : ndarray, optional
        Scratch array.
    out : ndarray, optional
        Array for the result.
    overwrite_a : bool, default False
        Write the result into ``a``.
    """

    gemm = get_blas_funcs("gemm", (a, u))
    if work is None:
        work = np.empty_like(a)
    if overwrite_a:
        out = a
    elif out is None:  # overwrite_a == False
        out = np.empty_like(a)
    # U.dot(A).dot(U.T)
    gemm(1.0, a.T, u.T, 0.0, work.T, overwrite_c=True)
    gemm(1.0, u.T, work.T, 0.0, out.T, trans_a=2, overwrite_c=True)
    return out


def increase2pow2(N: int) -> int:
    """Smallest power of two not below ``N``, the length used for zero-padded FFTs.

    Parameters
    ----------
    N : int
    """
    count = 0
    while N != 1:
        if N % 2 == 1:
            N += 1
        N /= 2
        count += 1
    return 2 ** count


def get_extended_energies(energies: NDArray[np.float64], oversample: int = 10) -> NDArray[np.float64]:
    """Energy grid with the same start and spacing, extended to a power-of-two length.

    Parameters
    ----------
    energies : ndarray, shape (n_E,)
        Uniform grid.
    oversample : int, default 10
        The new length is the next power of two above ``oversample * n_E``.
    """
    ne = len(energies)

    # Make sure that grid lenght is a power of 2
    ne_ext = increase2pow2(oversample * ne)

    begin = energies[0]
    de = energies[1] - begin
    extenergies = np.arange(ne_ext) * de + begin

    return np.around(extenergies, 8)


def interpolate(X: NDArray, energies: NDArray[np.float64], indx2eval: NDArray[np.int_]) -> None:
    """Fill grid points that were skipped by linear interpolation from the computed ones, in place.

    Parameters
    ----------
    X : ndarray, shape (n_E, ...)
        Values on the grid; entries at ``indx2eval`` are overwritten.
    energies : ndarray, shape (n_E,)
    indx2eval : ndarray of int
        Grid indices to fill. All other indices must hold computed values.
    """
    assert X.shape[0] == len(energies), "X.shape[0] differs from len(energies)."
    indx = np.setdiff1d(range(len(energies)), indx2eval)
    X_interp = interp1d(energies[indx], X[indx], kind="slinear", axis=0)
    for e, energy in zip(indx2eval, energies[indx2eval]):
        X[e] = X_interp(energy)


def get_interp_indices(energies: NDArray[np.float64]) -> NDArray[np.int_]:
    """Grid indices for a sparse sampling that is dense near zero energy and coarse far from it.

    The step between kept indices grows quadratically with the distance
    from zero, scaled by a tenth of the grid ends.

    Parameters
    ----------
    energies : ndarray, shape (n_E,)
    """
    a = 2
    b = energies[0] / 10.0
    c = energies[-1] / 10.0
    indices = []
    count = 0
    while count < len(energies):
        indices.append(count)
        x = energies[count]
        if x <= 0:
            y = int(1 + pow(abs(x), a) / pow(abs(b), a))
        else:
            y = int(1 + pow(abs(x), a) / pow(abs(c), a))
        count += y
    return np.array(indices)


def linear_interp(a1: NDArray, indices: NDArray[np.int_], size: int) -> NDArray:
    """Values on a full grid from values at selected indices, by linear interpolation.

    Points outside the selected range are extrapolated linearly.

    Parameters
    ----------
    a1 : ndarray, shape (len(indices), ...)
        Values at ``indices``.
    indices : ndarray of int
        Increasing grid indices.
    size : int
        Length of the full grid.
    """
    # Interpolate from array a1 to array a2. a1 is sampled on indices
    # Treat first index as special case

    # Don't do anything if a1 and a2 are identical
    if len(a1) == size:
        return a1
    s = (size,) + a1.shape[1:]
    a2 = np.zeros(s, a1.dtype)
    slope0 = (a1[1] - a1[0]) / (1.0 * indices[1] - indices[0])
    for i in range(indices[0]):
        a2[i] = a1[0] + slope0 * (1.0 * i - indices[0])
    for i in range(1, len(indices)):
        for j in range(indices[i - 1], indices[i]):
            a2[j] = a1[i - 1] + (a1[i] - a1[i - 1]) * (1.0 * j - indices[i - 1]) / (
                indices[i] - indices[i - 1]
            )
    slope1 = (a1[-1] - a1[-2]) / (1.0 * indices[-1] - indices[-2])
    for i in range(indices[-1], a2.shape[0]):
        a2[i] = a1[-1] + slope1 * (1.0 * i - indices[-1])
    return a2


def finer(x: NDArray[np.float64], A: NDArray, oversample: float = 2) -> Tuple[NDArray[np.float64], NDArray]:
    """Matrices on a grid ``oversample`` times finer, by cubic interpolation.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Uniform grid.
    A : ndarray, shape (n, m, m)
        A matrix at each grid point.
    oversample : float, default 2
        Ratio of old to new grid spacing.

    Returns
    -------
    x2 : ndarray
        The finer grid.
    A2 : ndarray
        Matrices on it.
    """
    dx = x[1] - x[0]
    dx /= oversample
    x2 = np.arange(x[0], x[-1] + dx / 2.0, dx)
    A2 = interp1d(x, A, kind="cubic", axis=0)(x2)
    return x2, A2


def smooth(x: NDArray[np.float64], A: NDArray, cutfreq: float = 10.0, oversample: float = 2) -> NDArray:
    """Remove fast oscillations from a function of energy, such as a self-energy on a coarse grid.

    The function is interpolated to a finer grid, Fourier components above
    the highest frequency divided by ``cutfreq`` are discarded, and the
    result is brought back to the original grid.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Uniform energy grid.
    A : ndarray, shape (n, m, m)
    cutfreq : float, default 10.0
        Keep frequencies up to the maximum divided by this number.
    oversample : float, default 2
        Refinement before filtering, see :func:`finer`.

    Returns
    -------
    A : ndarray, shape (n, m, m)
    """
    #
    if oversample != 1:
        x, A = finer(x, A, oversample)
    x_fft = np.fft.fftfreq(len(x), d=x[1] - x[0])
    A_fft = np.fft.fft(A, axis=0)
    cutoff = x_fft.max() / cutfreq
    A_fft[np.abs(x_fft) > cutoff] = 0
    A = np.fft.ifft(A_fft, axis=0).astype(A.dtype)
    if oversample != 1:
        x, A = finer(x, A, 1 / oversample)
    return A
