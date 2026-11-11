r"""Lesser, greater and retarded components of two-time functions, and conversions between them.

In a steady state every Green's function or self-energy depends on one
time difference, or one energy. The Langreth components are related by

.. math::

    X^r(t) = \theta(t)\,[X^>(t) - X^<(t)], \qquad
    X^> - X^< = X^r - X^a,

and these relations convert between the pair stored in memory and the
pair a calculation needs next.
"""
from __future__ import annotations

from functools import wraps
from math import pi
from typing import Any, Callable, Optional

from numpy.typing import NDArray

import numpy as np

from qtpyt.elph.hilbert import hilbert
from qtpyt.parallel.egrid import GridDesc
from qtpyt.screening import fft, ifft
from qtpyt.screening.fourier import fourier_integral, get_fourier_data
from qtpyt.screening.tools import (
    finer,
    get_extended_energies,
    get_interp_indices,
    increase2pow2,
    lesser_from_retarded,
    linear_interp,
    roll,
    smooth,
)


def get_retarded_from_lesser_and_greater(
    less: NDArray, great: NDArray, energies: NDArray[np.float64], ret: Optional[NDArray] = None, oversample: int = 10
) -> NDArray[np.complex128]:
    r"""Retarded component in energy from the lesser and greater components in time.

    Applies :math:`X^r(t) = \theta(t)[X^>(t) - X^<(t)]` and transforms back
    to energy on a zero-padded grid, with end-point corrections.

    Parameters
    ----------
    less, great : ndarray, shape (n_E, n_q)
        :math:`X^<(t)` and :math:`X^>(t)`, sampled at the times conjugate to
        ``energies``.
    energies : ndarray, shape (n_E,)
        Uniform energy grid.
    ret : ndarray, shape (n_E, n_q), optional
        Array for the result.
    oversample : int, default 10
        Zero-padding factor; the padded length is the next power of two.

    Returns
    -------
    ret : ndarray, shape (n_E, n_q)
        :math:`X^r(E)`.
    """
    extenergies = get_extended_energies(energies, oversample)
    fourierdata = get_fourier_data(len(extenergies))
    de_ext = extenergies[1] - extenergies[0]
    Ne = len(energies)
    # Ne_ext = increase2pow2(oversample * Ne)  # len(extenergies)
    Ne_ext = len(extenergies)
    Nq = less.shape[1]
    assert len(less) == Ne

    if ret is None:
        ret = np.zeros((Ne, Nq), complex)

    for i in range(Nq):
        # Make the difference (> - <)
        # g = fft(great[:, i], n=Ne_ext)
        # g -= fft(less[:, i], n=Ne_ext)
        # ret_i = g
        # ret_i = fft(great[:, i], n=Ne_ext) - fft(less[:, i], n=Ne_ext)
        # ret_i = fft(less[:, i], n=Ne_ext) - fft(great[:, i], n=Ne_ext)
        ret_i = great[:, i] - less[:, i]

        # Pad by zeros to extended grid, and FFT to time
        ret_i = fft(ret_i, n=Ne_ext)

        # multiply by theta(t)
        ret_i[Ne_ext // 2 :] = 0.0  # this will zero all t<0
        # ret_i[: Ne_ext // 2] = 0.0  # this will zero all t<0

        # iFFT back to energy using endpoint corrections
        # ret_i = ifft(ret_i)  # <- the equivalent without end-point cor
        ret_i = fourier_integral(ret_i, de_ext, Ne_ext, sign=1, data=fourierdata) / (
            de_ext * Ne_ext
        )
        # Reduce ext grid to full grid and then take nu grid only to store
        ret[:, i] = ret_i[:Ne]
    return ret


def assert_domain(*domains: Optional[str]) -> Callable:
    """Decorator: allow a method only when the arrays are in one of ``domains``.

    Parameters
    ----------
    *domains : {'e', 't', None}
    """

    def inner(f):
        @wraps(f)
        def wrapper(self, *a, **k):
            assert (
                self.domain in domains
            ), f"{self.domain} is invalid for transormation {f.__name__}"
            return f(self, *a, **k)

        return wrapper

    return inner


def change_domain(domain: str) -> Callable:
    """Decorator: mark the arrays as being in ``domain`` after the method runs.

    Parameters
    ----------
    domain : {'e', 't'}
    """

    def inner(f):
        @wraps(f)
        def wrapper(self, *a, **k):
            out = f(self, *a, **k)
            self.domain = domain
            return out

        return wrapper

    return inner


def swap_domain(f: Callable) -> Callable:
    """Decorator: switch the recorded domain between energy and time after the method runs.

    Parameters
    ----------
    f : callable
        The method to wrap.
    """

    @wraps(f)
    def wrapper(self, *a, **k):
        assert self.domain is not None, f"{self.__name__} not initialized."
        out = f(self, *a, **k)
        self.domain = "et".replace(self.domain, "")
        return out

    return wrapper


class LangrethPair(GridDesc):
    r"""Two Langreth components of a function, stored on a distributed energy or time grid.

    Two components fully determine the third, so only two arrays are kept,
    in :attr:`arrays` under the keys ``'l'`` (lesser), ``'g'`` (greater),
    ``'r'`` (retarded) or ``'a'`` (advanced). The conversion methods reuse
    the same memory and relabel the keys, and :attr:`domain` records whether
    the arrays hold energies (``'e'``), times (``'t'``) or nothing yet
    (``None``). A conversion called in the wrong domain raises an
    ``AssertionError``.

    Parameters
    ----------
    energies : ndarray, shape (n_E,)
        Uniform energy grid, same unit as the Hamiltonian.
    no : int
        Matrix size: orbitals for one-particle functions, pair orbitals for
        two-particle ones.
    dtype : data-type, default complex
    oversample : int, default 10
        Zero-padding factor for the time–energy transforms.

    Attributes
    ----------
    arrays : dict of str to ndarray
        The two stored components, each of shape ``(n_E_local, no, no)``.
    domain : {None, 'e', 't'}
    """

    _valid_keys = set(["l", "g", "r", "a"])
    _valid_domains = set([None, "e", "t"])

    def __init__(self, energies: NDArray[np.float64], no: int, dtype: Any = complex, oversample: int = 10) -> None:
        super().__init__(energies, no, dtype)
        self.arrays = {
            "l": self.empty_aligned_orbs(),
            "g": self.empty_aligned_orbs(),
        }
        self.oversample = oversample
        self.domain = None

    @assert_domain("e", "t")
    @swap_domain
    def convert_domain(self) -> None:
        """Fourier transform both components between energy and time."""
        # x(t), y(t) <</>> x(e), y(e)
        # _fft = fft if self.domain == "e" else ifft
        # if self.domain == "e":
        # indices = get_interp_indices(self.global_energies)
        # indices = (self.global_energies <= -50.0) | (self.global_energies >= 50.0)
        for a in self.arrays.values():
            A = self.collect_energies(a)
            if self.domain == "e":
                # A[indices] = 0.0
                # A = smooth(self.global_energies, A, 1.0, 6.0)
                A = fft(A, axis=0)
                # A = fft(linear_interp(A[indices], indices, self.ne), axis=0)
            else:
                A = ifft(A, axis=0)
            self.collect_orbitals(A, a)

    @assert_domain("e")
    def convert_retarded(self) -> None:
        r"""Replace the retarded component by the lesser one, :math:`X^< = X^> - X^r + X^a`, in energy."""
        # l(e) << r(e)
        # l(e) = g(e) - r(e) + a(e)
        lesser_from_retarded(self.arrays["r"], self.arrays["g"])
        self.arrays["l"] = self.arrays.pop("r")

    @assert_domain("t")
    @change_domain("e")
    def convert_less_and_great_to_ret(self, zero_index: Optional[int] = None, override: str = "l") -> None:
        r"""Retarded component in energy from lesser and greater components in time.

        Uses :func:`get_retarded_from_lesser_and_greater`. Afterwards the
        arrays hold the retarded component and the one not overwritten, in
        energy.

        Parameters
        ----------
        zero_index : int, optional
            Grid index of :math:`t = 0`; defaults to the index of the energy
            closest to zero.
        override : {'l', 'g'}, default 'l'
            Which component's memory receives the retarded component.
        """
        # r(e), g(e) << l(t), g(t)
        # r(t) = theta(t) [ <(t) - >(t) ]
        if zero_index is None:
            zero_index = self.zero_index
        l = self.collect_energies(self.arrays["l"])
        g = self.collect_energies(self.arrays["g"])
        ####  -1-  ####
        # l = ifft(l, axis=0)
        # g = ifft(g, axis=0)
        # r = 0.5 * (g - l)
        # r -= 1.0j * hilbert(r, oversample=self.oversample)
        ###############

        ####  -2-  ####
        l = roll(ifft(l, axis=0), zero_index)
        g = roll(ifft(g, axis=0), zero_index)
        if override == "l":
            r = l
            x = g
            keep = "g"
        else:
            r = g
            x = l
            keep = "l"
        get_retarded_from_lesser_and_greater(
            l, g, self.global_energies, r, oversample=self.oversample
        )
        ###############
        self.arrays["r"] = self.arrays.pop(override)
        self.collect_orbitals(r, self.arrays["r"])
        self.collect_orbitals(x, self.arrays[keep])

    def __lshift__(self, other):
        old_k = tuple(self.arrays.keys())
        new_k = tuple(other.arrays.keys())
        for ok, nk in zip(old_k, new_k):
            self.arrays[ok][:] = other.arrays[nk]
        self.arrays = {new_k[0]: self.arrays[old_k[0]], new_k[1]: self.arrays[old_k[1]]}
        self.domain = other.domain
