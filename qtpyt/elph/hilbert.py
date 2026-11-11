r"""Hilbert transform on a uniform grid, for the real part of a retarded function from its imaginary part.

The transform is defined here as

.. math::

    \mathcal H[f](y) = \frac{1}{\pi}\,\mathrm{p.v.}\!\int_{-\infty}^{\infty}
    \frac{f(x)}{x - y}\,dx,

the opposite sign of the convention on Wikipedia. It is evaluated as a
convolution by FFT, with zero-padding to suppress the periodic images.
"""
from __future__ import annotations

from typing import Optional

from numpy.typing import NDArray

import numpy as np
from qtpyt.screening import fft, ifft
from qtpyt.screening.tools import increase2pow2
from scipy import ifft

"""
Methods for perfoming the Hilbert transform of a function::

                     +oo
            1       /     f(x) 
  H[f](y) = -- p.v. | dx -----
            pi      /    x - y  
                     -oo

See en.wikipedia.org/wiki/Hilbert_transform (opposite sign)
"""


def hilbert_kernel_simple(n: int) -> NDArray[np.complex128]:
    """Hilbert kernel in Fourier space for ``n`` points: the discrete transform of :math:`1/x`.

    Parameters
    ----------
    n : int
    """
    ker = np.zeros(n, dtype=complex)
    ker[1 : n // 2] = 1.0j
    ker[n // 2 + 1 :] = -1.0j
    return ker


def hilbert_kernel_interpolate(n: int) -> NDArray[np.complex128]:
    r"""Hilbert kernel in Fourier space for ``n`` points, for piecewise-linear data.

    The exact Hilbert transform of the linear interpolation kernel
    :math:`L(s) = (1 - |s|)\,\Theta(1 - |s|)`, so the result is exact for a
    function that is linear between grid points.

    Parameters
    ----------
    n : int
    """
    # middle grid point
    mid = (n + 1) // 2

    # Make auxiliary array
    aux = np.arange(mid + 1, dtype=float)
    np.multiply(aux[1:], np.log(aux[1:]), aux[1:])

    # Make kernel
    ker = np.zeros(n, float)
    ker[1:mid] = aux[2:] - 2 * aux[1:-1] + aux[:-2]
    ker[-1:-mid:-1] = -ker[1:mid]

    return -fft(ker) / np.pi


def hilbert(f: NDArray, oversample: int = 10, axis: int = 0, ker: Optional[NDArray] = None, kerneltype: str = "interpolate") -> NDArray[np.complex128]:
    r"""Hilbert transform :math:`\mathcal H[f]` of data on a uniform grid.

    For a retarded function with imaginary part :math:`-\Gamma/2`, the real
    part follows from the Kramers–Kronig relation as a Hilbert transform.

    Parameters
    ----------
    f : ndarray
        Samples on a uniform grid along ``axis``.
    oversample : int, default 10
        Zero-padding factor; the padded length is the next power of two.
        ``0`` disables padding.
    axis : int, default 0
        Grid axis.
    ker : ndarray, optional
        Precomputed kernel of the padded length.
    kerneltype : {'interpolate', 'simple'}, default 'interpolate'
        :func:`hilbert_kernel_interpolate` or :func:`hilbert_kernel_simple`.

    Returns
    -------
    h : ndarray
        :math:`\mathcal H[f]` on the original grid. Only the first ``n``
        entries along the first axis are kept, so use ``axis=0``.
    """
    # Number of transform grid points
    n = f.shape[axis]

    # Generate new kernel if needed
    if ker == None:
        if oversample > 0:
            nfft = increase2pow2(oversample * n)
        else:
            nfft = n    
        ker = eval("hilbert_kernel_" + kerneltype)(nfft)
    else:
        assert ker.size == nfft

    # Reshape kernel
    ker_shape = [1,] * len(f.shape)
    ker_shape[axis] = nfft
    ker.shape = tuple(ker_shape)

    # Make convolution of f and kernel
    hil = ifft(fft(f, n=nfft, axis=axis) * ker, axis=axis)
    return hil[0:n]
