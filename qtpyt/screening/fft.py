r"""FFT wrappers with a common interface for pyFFTW and NumPy."""
from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

import numpy as np

from . import use_pyfftw


class pyFFTwrap:
    """Forward and backward FFT of length ``n`` with planned pyFFTW transforms.

    Parameters
    ----------
    n : int
        Transform length.
    dtype : data-type
    """

    def __init__(self, n: int, dtype: Any) -> None:
        self.n = n
        self.a = pyfftw.empty_aligned(n, dtype)
        self.b = pyfftw.empty_aligned(n, dtype)
        self.fft = pyfftw.FFTW(
            self.a, self.b, flags=("FFTW_DESTROY_INPUT", "FFTW_MEASURE")
        )
        self.ifft = pyfftw.FFTW(
            self.b,
            self.a,
            direction="FFTW_BACKWARD",
            flags=("FFTW_DESTROY_INPUT", "FFTW_MEASURE"),
        )

    def __eq__(self, inp):
        self.fft.input_array[: self.n] = inp

    def forward(self, a: NDArray) -> NDArray:
        """Forward transform.

        Parameters
        ----------
        a : ndarray
        """
        return self.fft(a)

    def backward(self, b: NDArray) -> NDArray:
        """Backward transform.

        Parameters
        ----------
        b : ndarray
        """
        return self.ifft(b)


class npFFTwrap:
    """Same interface as :class:`pyFFTwrap`, with :mod:`numpy.fft`.

    Parameters
    ----------
    n : int
        Transform length; the input is zero-padded to it.
    dtype : data-type
    """

    def __init__(self, n: int, dtype: Any) -> None:
        self.n = n
        self.dtype = dtype

    def __eq__(self, a):
        self.a = a

    def forward(self, a: NDArray) -> NDArray:
        """Forward transform, zero-padded to length ``n``.

        Parameters
        ----------
        a : ndarray
        """
        return np.fft.fft(a, n=self.n)

    def backward(self, b: NDArray) -> NDArray:
        """Backward transform.

        Parameters
        ----------
        b : ndarray
        """
        return np.fft.ifft(b)
