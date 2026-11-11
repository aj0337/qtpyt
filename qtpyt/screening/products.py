r"""Index contractions of one- and two-particle matrices that build polarizations and self-energies.

Two-particle matrices carry a pair index :math:`(ij)` on each side and are
stored as :math:`n_\mathrm{orb}^2 \times n_\mathrm{orb}^2` matrices, or as
:math:`n_q \times n_q` matrices in a reduced pair-orbital basis reached with a
rotation :math:`U` of shape :math:`(n_\mathrm{orb}^2, n_q)`. All functions
work on NumPy or CuPy arrays.
"""
from __future__ import annotations

from typing import Optional, Union

from numpy.typing import NDArray

import numpy as np
from qtpyt import xp


def polarization_product(G: NDArray, Gd: NDArray, pre: Union[float, complex] = 1.0, out: Optional[NDArray] = None, rot: Optional[NDArray] = None, work1: Optional[NDArray] = None, work2: Optional[NDArray] = None) -> NDArray:
    r"""Bubble product :math:`P_{ij,kl} = c\,G_{ik}\,\tilde G_{jl}`, the building block of the polarization.

    Parameters
    ----------
    G, Gd : ndarray, shape (n, n)
        The two propagators :math:`G` and :math:`\tilde G`, for example
        :math:`G^<(t)` and :math:`[G^>(t)]^*`.
    pre : float or complex, default 1.0
        Prefactor :math:`c`.
    out : ndarray, optional
        Array for the result.
    rot : ndarray, shape (n**2, n_q), optional
        Pair-orbital rotation :math:`U`; the result is then
        :math:`U^\dagger P U`, of shape ``(n_q, n_q)``.
    work1, work2 : ndarray, optional
        Scratch arrays of the shape of ``rot``.

    Returns
    -------
    out : ndarray, shape (n**2, n**2) or (n_q, n_q)
    """
    #
    #                  -->--
    #              i  |     | k
    #  U       (t,r)   xxxxx   (t',r')   U
    #   q,ij       j  |     | l           kl,q'
    #                  --<--
    #
    #                +
    #          G   G    =  G   Gd
    #           ik  lj      ik   jl
    #
    #
    Ni = len(G)
    if rot is None:
        if out is None:
            out = xp.zeros((Ni ** 2, Ni ** 2), dtype=G.dtype)
        # out[:] = pre * xp.kron(G, Gd)
        out_ijkl = out.reshape(Ni, Ni, Ni, Ni)
        for i, k in np.ndindex(Ni, Ni):
            out_ijkl[i, :, k, :] = G[i, k] * Gd
    else:
        Nq = rot.shape[1]

        if out is None:
            out = xp.zeros((Nq, Nq), dtype=G.dtype)

        # Temporary work arrays
        if work1 is None:
            work1 = xp.zeros_like(rot)
        if work2 is None:
            work2 = work1.copy()

        # Contiguous views with packed format unfolded, p -> ii
        work1 = work1.reshape(Ni, Ni, Nq)
        work2 = work2.reshape(Ni, Ni * Nq)
        rot = rot.reshape(Ni, Ni, Nq)

        # work1_kj,q' = sum_l B_jl rot_kl,q'
        for work1_iq, U_iq in zip(work1, rot):
            Gd.dot(U_iq, out=work1_iq)

        # work2_ij,q' = sum_k A_ik work1_kj,q'
        G.dot(work1.reshape(Ni, -1), out=work2)
        # work2 = xp.tensordot(G,work1,axes=([1],[0]))

        # out_qq' = sum_ij rot^dag_q,ij work2_ij,q'
        # rotd.dot(work2.reshape(-1, Nq), out=out)
        rot.reshape(-1, Nq).T.dot(work2.reshape(-1, Nq), out=out)

    out *= pre
    return out


def exchange_product(G: NDArray, V: NDArray, pre: Union[float, complex] = 1.0, out: Optional[NDArray] = None, rot: Optional[NDArray] = None, work1: Optional[NDArray] = None, work2: Optional[NDArray] = None) -> NDArray:
    r"""Exchange-type contraction :math:`\Sigma_{ij} = c\sum_{kl} G_{kl}\,V_{ik,jl}`.

    With :math:`G` the density matrix and :math:`V` the bare interaction this
    is the Fock self-energy; with :math:`G^\lessgtr(t)` and
    :math:`W^\lessgtr(t)` it is the GW self-energy.

    Parameters
    ----------
    G : ndarray, shape (n, n)
    V : ndarray, shape (n**2, n**2) or (n_q, n_q)
        Two-particle matrix, in the reduced basis when ``rot`` is given.
    pre : float or complex, default 1.0
        Prefactor :math:`c`.
    out : ndarray, shape (n, n), optional
        Array for the result.
    rot : ndarray, shape (n**2, n_q), optional
        Pair-orbital rotation :math:`U`; :math:`U V U^\dagger` is used for :math:`V`.
    work1, work2 : ndarray, optional
        Scratch arrays.

    Returns
    -------
    out : ndarray, shape (n, n)
    """
    assert G.dtype == V.dtype
    if out is None:
        out = xp.zeros_like(G)

    Ni = len(G)
    if rot is None:
        V_ikjl = V.reshape(Ni, Ni, Ni, Ni)
        # out = np.tensordot(G, V, axes=([1,2],[0,1]))
        for i, k in xp.ndindex(Ni, Ni):
            out[i] += V_ikjl[i, k].dot(G[k])
    else:
        Nq = len(V)
        if work1 is None:
            work1 = xp.zeros(Ni ** 2 * Nq, dtype=G.dtype)
        if work2 is None:
            work2 = xp.zeros(Ni ** 2 * Nq, dtype=G.dtype)

        # work2_q,jl = sum_q' V_qq' rot^dag_q',jl
        V.dot(rot.T, out=work2.reshape(Nq, -1))

        # work1_k,qj = sum_l A_kl work2_qj,l
        G.dot(work2.reshape(-1, Ni).T, out=work1.reshape(Ni, -1))

        # out_ij = sum_kq rot_i,kq work1_kq,j
        rot.reshape(Ni, -1).dot(work1.reshape(-1, Ni), out=out)

    out *= pre
    return out


def hartree_product(G: NDArray, V: NDArray, pre: Union[float, complex] = 1.0, out: Optional[NDArray] = None, rot: Optional[NDArray] = None, work1: Optional[NDArray] = None, work2: Optional[NDArray] = None) -> NDArray:
    r"""Hartree-type contraction :math:`\Sigma_{ij} = c\sum_{kl} G_{kl}\,V_{ij,kl}`.

    With :math:`G` the density matrix this is the electrostatic (Hartree)
    potential of the electron density.

    Parameters
    ----------
    G : ndarray, shape (n, n)
    V : ndarray, shape (n**2, n**2) or (n_q, n_q)
        Two-particle matrix, in the reduced basis when ``rot`` is given.
    pre : float or complex, default 1.0
        Prefactor :math:`c`.
    out : ndarray, shape (n, n), optional
        Array for the result.
    rot : ndarray, shape (n**2, n_q), optional
        Pair-orbital rotation :math:`U`; :math:`U V U^\dagger` is used for :math:`V`.
    work1, work2 : ndarray, optional
        Scratch arrays.

    Returns
    -------
    out : ndarray, shape (n, n)
    """
    assert G.dtype == V.dtype
    if out is None:
        out = np.zeros_like(G)

    Ni = len(G)
    if rot is None:
        V.dot(G.reshape(-1), out=out.reshape(-1))
    else:
        Nq = len(V)
        if work1 is None:
            work1 = np.zeros(Nq, dtype=G.dtype)
        if work2 is None:
            work2 = np.zeros(Nq, dtype=G.dtype)

        # work1_q = sum_kl rot^conj_q,kl G_kl
        rot.T.dot(G.reshape(-1), out=work1)

        # work2_q = sum_q' V_qq' work1_q'
        V.dot(work1, out=work2)

        # out_ij = sum_q rot_ij,q work2_q
        rot.dot(work2, out=out.reshape(-1))

    out *= pre
    return out
