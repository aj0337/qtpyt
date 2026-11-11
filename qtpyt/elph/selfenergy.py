r"""Electron–phonon self-energy in the lowest-order (Fock) approximation.

An electron emits or absorbs a phonon of energy :math:`\hbar\omega_\lambda`
and continues at an energy shifted by that amount. The resulting
self-energy broadens and shifts the electronic levels and produces the
inelastic features of the transmission (inelastic electron tunnelling
spectroscopy). Energies and phonon frequencies are in eV.
"""
from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

import numpy as np
from qtpyt.elph.hilbert import hilbert, hilbert_kernel_interpolate
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.langreth import LangrethPair, get_retarded_from_lesser_and_greater
from qtpyt.screening.tools import increase2pow2, roll

kB = 8.6e-5  # Boltzmann
nB = lambda w, kt: 1.0 / (np.exp(w / kt) - 1.0)


def translate(a: NDArray, s: int, out: NDArray, w: float = 1.0) -> None:
    r"""Add a copy of ``a`` shifted by ``s`` grid points, weighted by ``w``, to ``out``.

    ``s > 0`` gives :math:`a(E - s\,\delta E)` and ``s < 0`` gives
    :math:`a(E + |s|\,\delta E)`; values shifted in from beyond the grid are
    zero.

    Parameters
    ----------
    a : ndarray, shape (n_E, ...)
    s : int
    out : ndarray, shape (n_E, ...)
        Accumulated in place.
    w : float, default 1.0
    """
    if s > 0:
        # f^-(x) = f(x-y)
        out[s:] += w * a[:-s]
    elif s < 0:
        # f^+(x) = f(x+y)
        s = abs(s)
        out[:-s] += w * a[s:]
    else:
        out += w * s


def setzeros(*arrays: NDArray) -> None:
    """Set every element of the given arrays to zero, in place.

    Parameters
    ----------
    *arrays : ndarray
    """
    for a in arrays:
        a[:] = 0.0


class ElphFockSelfEnergy(LangrethPair):
    r"""Electron–phonon self-energy to lowest order in the coupling (Fock diagram).

    For each vibrational mode :math:`\lambda` with energy :math:`\omega_\lambda`,
    coupling matrix :math:`M_\lambda` and Bose occupation :math:`n_\lambda`,

    .. math::

        \Sigma^<(E) &= \sum_\lambda M_\lambda \left[ n_\lambda G^<(E - \omega_\lambda)
        + (1 + n_\lambda) G^<(E + \omega_\lambda) \right] M_\lambda, \\
        \Sigma^>(E) &= \sum_\lambda M_\lambda \left[ n_\lambda G^>(E + \omega_\lambda)
        + (1 + n_\lambda) G^>(E - \omega_\lambda) \right] M_\lambda.

    The :math:`n_\lambda` terms describe phonon absorption, the
    :math:`1 + n_\lambda` terms emission. The retarded self-energy follows from
    :math:`\Sigma^> - \Sigma^<` with the Hilbert transform for its real part.
    Shifts by :math:`\omega_\lambda` that fall between grid points are
    interpolated linearly. The Hartree (tadpole) diagram is not included,
    and the phonons stay in equilibrium.

    Parameters
    ----------
    gf : DistGreenFunction
        Electron Green's function on the energy grid. Its temperature
        ``gf.gf0.kt`` sets the phonon occupations.
    g_lii : ndarray, shape (n_modes, n_orb, n_orb)
        Electron–phonon coupling matrices :math:`M_\lambda`, in eV.
    w_l : ndarray, shape (n_modes,)
        Phonon energies :math:`\hbar\omega_\lambda`, in eV.
    oversample : int, default 10
        Zero-padding factor of the Hilbert transform.

    Notes
    -----
    After :meth:`update` the arrays hold :math:`\Sigma^r(E)` and :math:`\Sigma^<(E)`.
    """

    def __init__(
        self, gf: DistGreenFunction, g_lii: np.ndarray, w_l: np.ndarray, oversample: int = 10
    ) -> None:
        self.gf = gf
        self.g_lii = g_lii
        self.w_l = w_l
        self.nph_l = nB(w_l, self.gf.gf0.kt)
        super().__init__(
            gf.global_energies, gf.no, dtype=complex, oversample=oversample
        )
        # We need retarded and lesser
        self.arrays["r"] = self.arrays.pop("g")
        setzeros(*(a for a in self.arrays.values()))

    def update(self) -> None:
        r"""Recompute :math:`\Sigma^r(E)` and :math:`\Sigma^<(E)` from the current :math:`G^<`, :math:`G^>`."""

        Gl = self.collect_energies(self.gf.arrays["l"])
        Gg = self.collect_energies(self.gf.arrays["g"])

        Gl_minus = np.empty_like(Gl)
        Gl_plus = np.empty_like(Gl)
        Gg_minus = np.empty_like(Gl)
        Gg_plus = np.empty_like(Gl)

        Sr = self.arrays["r"]
        Sl = self.arrays["l"]
        setzeros(Sr, Sl)

        for w, g_ii, nph in zip(self.w_l, self.g_lii, self.nph_l):

            s = w / self.de
            r = s % 1  # number from (0,1)
            s = int(s - r)

            setzeros(Gl_minus, Gl_plus, Gg_minus, Gg_plus)

            # interpolate
            for weight in (1 - r, r):
                if round(weight, 5) > 0.0:
                    translate(Gl, s, Gl_minus, weight)
                    translate(Gg, s, Gg_minus, weight)
                    translate(Gl, -s, Gl_plus, weight)
                    translate(Gg, -s, Gg_plus, weight)
                s += 1

            l = nph * Gl_minus + (1 + nph) * Gl_plus
            g = nph * Gg_plus + (1 + nph) * Gg_minus

            r = 0.5 * (g - l)
            r -= 1.0j * hilbert(r, oversample=self.oversample)

            l = self.collect_orbitals(l)
            r = self.collect_orbitals(r)

            l = np.einsum("ij,ejk,kl->eil", g_ii, l, g_ii, optimize=True)
            r = np.einsum("ij,ejk,kl->eil", g_ii, r, g_ii, optimize=True)

            Sl += l
            Sr += r

    def retarded(self, energy: float) -> NDArray[np.complex128]:
        r"""Retarded self-energy :math:`\Sigma^r` at the first grid energy not below ``energy``.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, on the grid; no interpolation is done.
        """
        return self.arrays["r"][np.searchsorted(self.energies, energy)]
