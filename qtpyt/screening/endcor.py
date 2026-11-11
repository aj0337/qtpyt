r"""Fourier integrals with end-point corrections; an alternative implementation of :mod:`qtpyt.screening.fourier`."""
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray

import numpy as np
from qtpyt.screening import fft, ifft


def dftint(f: NDArray, N: int, sign: int = 1, Wcor: Optional[Tuple[NDArray, NDArray]] = None) -> NDArray:
    """End-point corrected Fourier integral of ``f``, without the grid-spacing factor.

    Parameters
    ----------
    f : ndarray
    N : int
        Transform length.
    sign : {1, -1}, default 1
        Direction, as in :func:`qtpyt.screening.fourier.fourier_integral`.
    Wcor : tuple of ndarray, optional
        Precomputed :func:`get_dftcor` for this ``N``.
    """
    if Wcor is None:
        Wcor = get_dftcor(N)
    W, cor = Wcor
    if sign == -1:
        W, cor = np.fft.ifftshift(W, axes=0), np.fft.ifftshift(cor, axes=0)
    shift = (
        cor.dot(f[:4]) + cor.real.dot(f[-1:-5:-1]) - 1.0j * cor.imag.dot(f[-1:-5:-1])
    )
    if sign == 1:
        F = (W * N * ifft(f, N) + shift) / N
    else:
        F = W * fft(f, N) + shift
    return F


def get_dftcor(N: int) -> Tuple[NDArray, NDArray]:
    """End-point correction weights for a transform of length ``N``, centred order.

    Parameters
    ----------
    N : int
    """
    theta = np.arange(-N / 2, N / 2) * 2 * np.pi / N
    W, cor = dftcor(theta)  # end point corrections
    # FFTshift end point corrections
    return W, cor
    # return np.fft.ifftshift(W, axes=[0]), np.fft.ifftshift(cor, axes=[0])


def dftcor(theta: NDArray[np.float64]) -> Tuple[NDArray, NDArray]:
    """End-point correction weights, Numerical Recipes in C, section 13.9, eq. 13.9.14.

    Parameters
    ----------
    theta : ndarray of float64
        Angle at each output point.

    Returns
    -------
    W : ndarray, shape (n,)
    cor : ndarray, shape (n, 4)
    """
    theta_small = 5.0e-2
    N = len(theta)
    cor = np.zeros((N, 4), complex)

    mask_s = abs(theta) < theta_small
    mask_b = abs(theta) >= theta_small

    # tb = theta.copy()
    # tb[len(tb) // 2] = 1  # Avoid division by zero!

    t = theta
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2

    cth = np.cos(theta)
    sth = np.sin(theta)
    ctth = cth * cth - sth * sth
    stth = 2.0 * sth * cth
    tmth2 = 3.0 - t2
    spth2 = 6.0 + t2
    sth4i = 1.0 / (6.0 * t4)
    tth4i = 2.0 * sth4i

    W = (1.0 - (11.0 / 720.0) * t4 + (23.0 / 15120.0) * t6) * mask_s
    W += (tth4i * spth2 * (3.0 - 4.0 * cth + ctth)) * mask_b

    cor[:, 0].real = (
        (-2.0 / 3.0) + t2 / 45.0 + (103.0 / 15120.0) * t4 - (169.0 / 226800.0) * t6
    ) * mask_s
    cor[:, 0].imag = (
        t
        * (
            2.0 / 45.0
            + (2.0 / 105.0) * t2
            - (8.0 / 2835.0) * t4
            + (86.0 / 467775.0) * t6
        )
    ) * mask_s
    cor[:, 1].real = (
        (7.0 / 24.0) - (7.0 / 180.0) * t2 + (5.0 / 3456.0) * t4 - (7.0 / 259200.0) * t6
    ) * mask_s
    cor[:, 1].imag = (
        t * (7.0 / 72.0 - t2 / 168.0 + (11.0 / 72576.0) * t4 - (13.0 / 5987520.0) * t6)
    ) * mask_s
    cor[:, 2].real = (
        (-1.0 / 6.0) + t2 / 45.0 - (5.0 / 6048.0) * t4 + t6 / 64800.0
    ) * mask_s
    cor[:, 2].imag = (
        t * (-7.0 / 90.0 + t2 / 210.0 - (11.0 / 90720.0) * t4 + (13.0 / 7484400.0) * t6)
    ) * mask_s
    cor[:, 3].real = (
        (1.0 / 24.0) - t2 / 180.0 + (5.0 / 24192.0) * t4 - t6 / 259200.0
    ) * mask_s
    cor[:, 3].imag = (
        t
        * (7.0 / 360.0 - t2 / 840.0 + (11.0 / 362880.0) * t4 - (13.0 / 29937600.0) * t6)
    ) * mask_s

    cor[:, 0].real += (
        sth4i * (-42.0e0 + 5.0e0 * t2 + spth2 * (8.0e0 * cth - ctth))
    ) * mask_b
    cor[:, 0].imag += (sth4i * (t * (-12.0e0 + 6.0e0 * t2) + spth2 * stth)) * mask_b
    cor[:, 1].real += (sth4i * (14.0e0 * tmth2 - 7.0e0 * spth2 * cth)) * mask_b
    cor[:, 1].imag += (sth4i * (30.0e0 * t - 5.0e0 * spth2 * sth)) * mask_b
    cor[:, 2].real += (tth4i * (-4.0e0 * tmth2 + 2.0e0 * spth2 * cth)) * mask_b
    cor[:, 2].imag += (tth4i * (-12.0e0 * t + 2.0e0 * spth2 * sth)) * mask_b
    cor[:, 3].real += (sth4i * (2.0e0 * tmth2 - spth2 * cth)) * mask_b
    cor[:, 3].imag += (sth4i * (6.0e0 * t - spth2 * sth)) * mask_b

    return W, cor
