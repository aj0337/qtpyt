from functools import singledispatch

import numpy as np
import scipy.linalg as la

from qtpyt.base._kernels import dotdiag
from qtpyt.basis import Basis
from qtpyt.block_tridiag.greenfunction import GreenFunction as BTGF

"""References:

http://www1.spms.ntu.edu.sg/~ydchong/teaching/08_contour_integration.pdf
"""


def zero_fermi(nzp):
    """Compute poles (zp) and residues (Rp) of Fermi function."""
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


def get_ao_charge(G, mu=0, beta=38.68, nzp=100, R=1e10):
    """Get orbital charge density."""
    return _integrate(G, mu, beta, nzp, R)


def get_atomic_charge(basis: Basis, G, mu=0, beta=38.68, nzp=100, R=1e10):
    """Get atomic charge density."""
    rho = _integrate(G, mu, beta, nzp, R)
    return basis.sum_aos_atoms(rho)


def get_total_charge(G, mu=0, beta=38.68, nzp=100, R=1e10):
    """Get total charge density."""
    return _integrate(G, mu, beta, nzp, R).sum()
