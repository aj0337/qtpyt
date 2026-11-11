r"""Electron occupations from the Green's function by contour integration.

The number of electrons in each orbital is the energy integral of the
spectral function weighted by the Fermi function. Integrating along the
real axis needs a fine grid because of sharp resonances; here the
integral is evaluated instead from the Green's function at a small set of
complex energies, the poles of the continued-fraction representation of
the Fermi function [1]_.

References
----------
.. [1] T. Ozaki, Phys. Rev. B 75, 035123 (2007).
"""
from __future__ import annotations

from typing import Any, Tuple

from numpy.typing import NDArray
from functools import singledispatch

import numpy as np
import scipy.linalg as la

from qtpyt.base._kernels import dotdiag
from qtpyt.basis import Basis
from qtpyt.block_tridiag.greenfunction import GreenFunction as BTGF

"""References:

http://www1.spms.ntu.edu.sg/~ydchong/teaching/08_contour_integration.pdf
"""


def zero_fermi(nzp: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Poles and residues of the continued-fraction expansion of the Fermi function.

    The Fermi function is written as
    :math:`f(x) = \tfrac12 + \sum_p R_p \left[(x - i z_p)^{-1} + (x + i z_p)^{-1}\right]`
    with :math:`x = \beta(E - \mu)`. The poles :math:`z_p` and residues
    :math:`R_p` are found from a generalized eigenvalue problem [1]_.

    Parameters
    ----------
    nzp : int
        Number of poles :math:`N`.

    Returns
    -------
    zp : ndarray, shape (nzp,)
        Poles :math:`z_p`, positive and increasing (the first is :math:`\pi`).
    Rp : ndarray, shape (nzp,)
        Residues :math:`R_p`.
    """
    N = nzp
    M = 2 * N

    A = np.zeros((2 * M, 2 * M))
    B = np.zeros((2 * M, 2 * M))

    zp = np.zeros(2 + M)
    Rp = np.zeros(2 + M)

    for i in range(1, M + 1):
        B[i, i] = 2 * i - 1

    for i in range(1, M):
        A[i, i + 1] = -0.5
        A[i + 1, i] = -0.5

    a = np.zeros(M * M)
    b = np.zeros(M * M)

    for i in range(M):
        for j in range(M):
            a[j * M + i] = A[i + 1, j + 1]
            b[j * M + i] = B[i + 1, j + 1]

    a.shape = (M, M)
    b.shape = (M, M)

    eigvals, eigvecs = la.eigh(a, b)

    zp[:M] = eigvals

    for i in range(M, 0, -1):
        zp[i] = zp[i - 1]

    for i in range(1, M + 1):
        zp[i] = 1.0 / zp[i]

    a = eigvecs.T.flatten()

    for i in range(0, M):
        Rp[i + 1] = -a[i * M] * a[i * M] * zp[i + 1] * zp[i + 1] * 0.250

    zp = -zp[1 : N + 1]
    Rp = Rp[1 : N + 1]

    return zp, Rp


@singledispatch
def _integrate(G, mu=0, beta=38.68, nzp=100, R=1e10):
    zp, Rp = zero_fermi(nzp)
    N = nzp

    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * dotdiag(G.retarded(1j * R), G.S)

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += dotdiag(G.retarded(a_p[i]), G.S) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


@_integrate.register(BTGF)
def _(G, mu=0, beta=38.68, nzp=100, R=1e10):
    zp, Rp = zero_fermi(nzp)
    N = nzp

    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * G.retarded(1j * R).dotdiag(G.S)

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += G.retarded(a_p[i]).dotdiag(G.S) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


def get_ao_charge(G: Any, mu: float = 0, beta: float = 38.68, nzp: int = 100, R: float = 1e10) -> NDArray[np.float64]:
    r"""Number of electrons in each orbital, including both spin directions.

    A fully occupied orbital gives 2, an orbital whose level sits at
    :math:`\mu` gives 1. The occupation is the Mulliken-type diagonal
    :math:`(\rho S)_{ii}`, so the values add up to the total electron
    count of the region.

    Parameters
    ----------
    G : GreenFunction
        Green's function of the region, either
        :class:`qtpyt.base.greenfunction.GreenFunction` or the block
        tridiagonal :class:`qtpyt.block_tridiag.greenfunction.GreenFunction`.
        Its ``retarded`` must accept complex energies. Its broadening
        ``eta`` is set to zero during the integration and restored
        afterwards.
    mu : float, default 0
        Chemical potential :math:`\mu`, same energy unit as the Hamiltonian.
    beta : float, default 38.68
        Inverse temperature :math:`\beta = 1/k_B T` in inverse energy units.
        The default is 300 K for energies in eV.
    nzp : int, default 100
        Number of Fermi-function poles.
    R : float, default 1e10
        Not used; the radius of the large-energy contribution is fixed to
        1e10.

    Returns
    -------
    n : ndarray, shape (n_orb,)
        Electrons per orbital.

    Notes
    -----
    With :math:`z_p, R_p` from :func:`zero_fermi` and the moment at large :math:`|z|`,
    :math:`\mu_0 = \lim_{R\to\infty} iR\,G(iR)S`,

    .. math::

        n_i = \operatorname{Re}\,(\mu_0)_{ii}
        + \operatorname{Im}\Big[-\frac{4i}{\beta}\sum_p R_p
        \big(G(\mu + i z_p/\beta)\,S\big)_{ii}\Big].

    The result is exact for an equilibrium Green's function that is
    analytic in the upper half plane; it ignores any lesser self-energy.
    """
    return _integrate(G, mu, beta, nzp, R)


def get_atomic_charge(basis: Basis, G: Any, mu: float = 0, beta: float = 38.68, nzp: int = 100, R: float = 1e10) -> NDArray[np.float64]:
    """Number of electrons on each atom, the orbital occupations summed per atom.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the region, used to group orbitals by atom.
    G : GreenFunction
        Green's function of the region.
    mu, beta, nzp, R
        See :func:`get_ao_charge`.

    Returns
    -------
    n : ndarray, shape (n_atoms,)
        Electrons per atom, both spins.
    """
    rho = _integrate(G, mu, beta, nzp, R)
    return basis.sum_aos_atoms(rho)


def get_total_charge(G: Any, mu: float = 0, beta: float = 38.68, nzp: int = 100, R: float = 1e10) -> float:
    """Total number of electrons in the region, both spins.

    Parameters
    ----------
    G : GreenFunction
        Green's function of the region.
    mu, beta, nzp, R
        See :func:`get_ao_charge`.
    """
    return _integrate(G, mu, beta, nzp, R).sum()
