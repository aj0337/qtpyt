r"""Small matrix kernels shared by the Green's function classes.

Traces and diagonals of matrix products are computed without forming the
full product, which matters for the density of states and transmission of
large devices.
"""
from __future__ import annotations

from typing import Optional

from numba import vectorize
from numpy.typing import NDArray

from qtpyt import xp

dagger = lambda mat: xp.conj(mat.T)

target = "cpu" if xp.__name__ == "numpy" else "cuda"

kwargs = {"target": target}
if target == "cpu":
    kwargs["fastmath"] = True

# @vectorize(['c16(c16,c16,c16)'],**kwargs)
def _mul(z, h, s):
    return z * s - h


# @vectorize(['c16(c16,c16,c16)'],**kwargs)
def _mulc(z, h, s):
    return z * s.conj() - h.conj()
    # return z*s.conjugate()-h.conjugate()


def get_lambda(sigma: NDArray) -> NDArray:
    r"""Broadening :math:`\Gamma = i(\Sigma - \Sigma^\dagger)` of a retarded self-energy.

    See :meth:`~qtpyt.base.selfenergy.BaseSelfEnergy.get_lambda`.

    Parameters
    ----------
    sigma : ndarray, shape (n, n)
        Retarded self-energy :math:`\Sigma^r`.

    Returns
    -------
    Gamma : ndarray, shape (n, n)
    """
    return 1.0j * (sigma - sigma.T.conj())


def dotdiag(x: NDArray, y: NDArray, res: Optional[NDArray] = None) -> NDArray:
    r"""Diagonal of the product :math:`(XY)_{ii}`, without forming :math:`XY`.

    Parameters
    ----------
    x : ndarray, shape (n, m)
    y : ndarray, shape (m, n)
    res : ndarray, shape (n,), optional
        Array the diagonal is added to. A zero array is used if omitted.

    Returns
    -------
    res : ndarray, shape (n,)
        ``res + diag(x @ y)``.
    """
    if res is None:
        res = xp.zeros(x.shape[0], x.dtype)
    for i in range(x.shape[0]):
        res[i] += xp.sum(x[i, :] * y[:, i])
    return res


def dottrace(x: NDArray, y: NDArray, res: complex = 0.0) -> complex:
    r"""Trace of the product :math:`\operatorname{Tr}[XY]`, without forming :math:`XY`.

    Parameters
    ----------
    x : ndarray, shape (n, m)
    y : ndarray, shape (m, n)
    res : complex, default 0.0
        Value the trace is added to.

    Returns
    -------
    trace : complex
        ``res + trace(x @ y)``.
    """
    for i in range(x.shape[0]):
        res += xp.sum(x[i, :] * y[:, i])
    return res

def dots(*args: NDArray) -> NDArray:
    r"""Ordered matrix product :math:`A_1 A_2 \cdots A_n`.

    Parameters
    ----------
    *args : ndarray
        The factors, multiplied left to right.

    Returns
    -------
    product : ndarray
    """
    x = args[0]
    for M in args[1:]:
        x = xp.dot(x, M)
    return x
