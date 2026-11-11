r"""k-point grids, time-reversal symmetry and Bloch sums for periodic electrodes.

k-points are in units of the reciprocal lattice vectors and lattice vectors
in units of the direct lattice vectors. The Fourier convention is that of
:mod:`qtpyt.tightbinding`. Time-reversal symmetry is used in the form
:math:`A(-\mathbf k) = A(\mathbf k)^T`, valid for Hermitian matrices in a basis
of real orbitals without spin–orbit coupling.
"""
from __future__ import annotations

from functools import partial
from typing import List, Optional, Sequence, Tuple

from numpy.typing import NDArray

import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from numba import njit
from scipy.spatial.distance import cdist


def monkhorst_pack(size: Sequence[int], offset: Sequence[float] = (0.0, 0.0, 0.0)) -> NDArray[np.float64]:
    r"""Monkhorst–Pack grid of k-points.

    Parameters
    ----------
    size : sequence of three int
        Number of k-points along each reciprocal lattice vector.
    offset : sequence of three float, default (0, 0, 0)
        Shift of the grid, in units of the reciprocal lattice vectors.

    Returns
    -------
    kpts : ndarray, shape (prod(size), 3)
        The grid points :math:`(i + 1/2)/N - 1/2 + \mathrm{offset}`, so an odd
        ``size`` contains Γ and an even one does not.
    """
    kpts = np.indices(size).reshape(len(size), -1).T
    kpts = (kpts + 0.5) / size - 0.5 + offset
    return kpts


def _get_dirs(direction="x"):
    """Get indices parallel and transverse directions.
    
    Args:
        direction : (str) parallel direction. 
            One of 'x', 'y' or 'z'
    """
    dirs_t = [0, 1, 2]
    dir_p = dirs_t.pop("xyz".index(direction))
    return dir_p, dirs_t


@njit
def _index(array, item):
    """Find item in array."""
    for idx, val in enumerate(array):
        if val == item:
            return idx


def map_ik2k(kpts: NDArray[np.float64], k2ik: NDArray[np.int_]) -> Tuple[NDArray[np.int_], NDArray[np.bool_]]:
    r"""For every k-point of the full grid, the irreducible k-point it is obtained from.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Full k-point grid.
    k2ik : ndarray of int
        Positions in ``kpts`` of the irreducible k-points,
        ``kpts[k2ik] = ikpts``.

    Returns
    -------
    ik2k : ndarray of int, shape (n_k,)
        Index into the irreducible k-points for each k-point.
    time_reverse : ndarray of bool, shape (n_k,)
        True where the k-point is the time-reversed partner
        :math:`-\mathbf k` of its irreducible k-point.

    Raises
    ------
    IndexError
        If a k-point has no time-reversed partner in the irreducible set.
    """
    # Find if there's a minimum distance with time-reversed k-point.
    # distances = cdist(kpts, np.dot(kpts, -np.eye(3).T)) % 1
    distances = cdist(kpts % 1, -kpts % 1)  #% 1
    equivalent = np.where(distances.min(1) < 1e-7, distances.argmin(1), -1)
    ik2k = np.empty(kpts.shape[0], int)
    ik2k[k2ik] = np.arange(len(k2ik))
    diff = np.setdiff1d(range(kpts.shape[0]), k2ik)
    for idx, val in enumerate(diff):
        try:
            ik2k[val] = _index(k2ik, equivalent[val])
        except:
            raise IndexError
    # ik2k[diff] = [_index(k2ik, item) for item in equivalent[diff]]
    time_reverse = np.zeros(kpts.shape[0], bool)
    time_reverse[diff] = True
    return ik2k, time_reverse


def apply_inversion_symm(kpts: NDArray[np.float64], direction: str = "x") -> Tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.int_], NDArray[np.bool_]]:
    r"""Irreducible k-points under time reversal :math:`\mathbf k \to -\mathbf k`.

    Of each pair :math:`(\mathbf k, -\mathbf k)` one point is kept: for
    ``direction='x'``, the one whose first non-zero component is positive.
    Γ is always kept. Matrices on the full grid then follow from those on
    the irreducible points as ``B = A[ik2k]`` with ``B[time_reverse]``
    transposed.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Full k-point grid.
    direction : {'x', 'y', 'z'}, default 'x'
        Selects which non-zero component (first, second or third) decides
        which point of a pair is kept.

    Returns
    -------
    ikpts : ndarray, shape (n_ik, 3)
        Irreducible k-points.
    k2ik : ndarray of int
        Positions of ``ikpts`` in ``kpts``.
    ik2k, time_reverse : ndarray
        See :func:`map_ik2k`.
    """
    p_dir = "xyz".index(direction)
    ikpts = []
    k2ik = []
    for bzk_index, bzk_k in enumerate(kpts):
        try:
            if bzk_k[np.nonzero(bzk_k)[0][p_dir]] > 0:
                ikpts.append(bzk_k)
                k2ik.append(bzk_index)
            else:
                continue
        # zero case
        except IndexError:
            ikpts.append(bzk_k)
            k2ik.append(bzk_index)
    ikpts = np.array(ikpts)
    k2ik = np.array(k2ik)
    ik2k, time_reverse = map_ik2k(kpts, k2ik)
    return ikpts, k2ik, ik2k, time_reverse


def expand_inversion_symm(kpts: NDArray[np.float64], A: NDArray, direction: str = "x") -> Tuple[NDArray[np.float64], NDArray]:
    r"""Matrices on the full k-grid from matrices on its irreducible part.

    Uses :math:`A(-\mathbf k) = A(\mathbf k)^T`.

    Parameters
    ----------
    kpts : ndarray, shape (n_k, 3)
        Full k-point grid.
    A : ndarray, shape (n_ik, n, n)
        Matrices at the irreducible k-points of :func:`apply_inversion_symm`.
    direction : {'x', 'y', 'z'}, default 'x'
        As in :func:`apply_inversion_symm`.

    Returns
    -------
    kpts : ndarray, shape (n_k, 3)
        k-point of each returned matrix.
    B : ndarray, shape (n_k, n, n)
        Matrices on the full grid.
    """
    assert A.shape[0] < kpts.shape[0], "A already in reducible BZ"
    ikpts, k2ik, ik2k, time_reverse = apply_inversion_symm(kpts, direction)
    B = A[ik2k]
    B[time_reverse] = np.transpose(B[time_reverse], axes=(0, 2, 1))
    kpts = ikpts[ik2k]
    kpts[time_reverse] = -kpts[time_reverse]
    return kpts, B


def build_partial_kpts(size: Sequence[int], offset: Sequence[float] = (0.0, 0.0, 0.0), direction: str = "x") -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a Monkhorst–Pack grid into its transport and transverse parts.

    Parameters
    ----------
    size, offset
        Grid, see :func:`monkhorst_pack`.
    direction : {'x', 'y', 'z'}, default 'x'
        Transport direction.

    Returns
    -------
    kpts_p : ndarray
        k-points along the transport direction only.
    kpts_t : ndarray
        k-points in the transverse plane only.
    """
    size = np.asarray(size)
    offset = np.asarray(offset)

    dir_p, dirs_t = _get_dirs(direction)

    size_p = np.ones(3, int)
    size_p[dir_p] = size[dir_p]
    offset_p = np.zeros(3)
    offset_p[dir_p] = offset[dir_p]

    kpts_p = monkhorst_pack(size_p, offset_p)

    size_t = np.ones(3, int)
    size_t[dirs_t] = size[dirs_t]
    offset_t = np.zeros(3)
    offset_t[dirs_t] = offset[dirs_t]

    kpts_t = monkhorst_pack(size_t, offset_t)

    return kpts_p, kpts_t


def fourier_sum(A: NDArray, kpts: NDArray[np.float64], R: NDArray, out: Optional[NDArray] = None, inverse: bool = False) -> NDArray:
    r"""Fourier transform between k-space matrices and real-space lattice matrices.

    Parameters
    ----------
    A : ndarray, shape (n_k, n, m) or (n_R, n, m)
        Matrices at the k-points (forward) or at the lattice vectors
        (``inverse``).
    kpts : ndarray, shape (3,) or (n_k, 3)
        k-points.
    R : ndarray, shape (3,) or (n_R, 3)
        Lattice vectors.
    out : ndarray, optional
        Array for the result.
    inverse : bool, default False
        Go from real space to k-space instead.

    Returns
    -------
    B : ndarray
        Forward: :math:`A(\mathbf R) = N_k^{-1}\sum_{\mathbf k} A(\mathbf k)\,e^{2\pi i\mathbf k\cdot\mathbf R}`.
        Inverse: :math:`A(\mathbf k) = \sum_{\mathbf R} A(\mathbf R)\,e^{-2\pi i\mathbf k\cdot\mathbf R}`.
        The first axis is dropped when ``R`` is a single vector.
    """
    # np.asarray avoids copy
    kpts = np.asarray(kpts)
    R = np.asarray(R)
    if kpts.ndim < 2:
        kpts = kpts[None, :]
    squeeze = False
    if R.ndim < 2:
        squeeze = True
        R = R[None, :]
    shape = (R.shape[0],) + A.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=A.dtype)
    if out.ndim < 3:
        out = out[None, :]

    out.shape = (out.shape[0], -1)
    A.shape = (A.shape[0], -1)
    if inverse:
        phase = np.exp(-2.0j * np.pi * np.dot(kpts, R.T))
    else:
        phase = np.exp(2.0j * np.pi * np.dot(R, kpts.T)) / kpts.shape[0]
    np.sum(phase[..., None] * A[None, ...], axis=1, out=out)
    A.shape = (kpts.shape[0],) + shape[1:]
    out.shape = shape

    if squeeze:
        return out[0]
    return out


def partial_bloch_sum(A: NDArray, kpts: NDArray[np.float64], R: NDArray, kpts_p: NDArray[np.float64], kpts_t: NDArray[np.float64], out: Optional[NDArray] = None) -> NDArray:
    r"""Fourier transform along the transport direction only, at each transverse k-point.

    Gives the principal-layer matrices :math:`A_{\mathbf k_\perp}(R_\parallel)`:
    real space along transport, Bloch momentum across. These are the inputs
    of :class:`~qtpyt.surface.principallayer.PrincipalLayer`.

    Parameters
    ----------
    A : ndarray, shape (n_k, n, n)
        Matrices on the 3D grid, or on its irreducible part (then
        :math:`A(-\mathbf k) = A(\mathbf k)^T` fills the rest).
    kpts : ndarray, shape (n_k, 3)
        k-points of ``A``.
    R : ndarray, shape (3,) or (n_R, 3)
        Lattice vectors along the transport direction.
    kpts_p, kpts_t : ndarray
        Transport and transverse parts of the grid, from
        :func:`build_partial_kpts`.
    out : ndarray, optional
        Array for the result.

    Returns
    -------
    A_tR : ndarray, shape (n_kt, n_R, n, n) or (n_kt, n, n)
        One set of real-space matrices per transverse k-point.
    """
    distance = partial(np.linalg.norm, axis=1)

    R = np.asarray(R)
    squeeze = False
    if R.ndim < 2:
        squeeze = True
    R = np.array(R, ndmin=2)
    if out is None:
        out = np.empty((kpts_t.shape[0], R.shape[0]) + A.shape[1:], A.dtype)

    for j, kpt_t in enumerate(kpts_t):
        _kpts = kpts_p + kpt_t
        A_t = []
        for k in _kpts:
            try:
                i = np.where(distance(k - kpts) < 1e-7)[0][0]
                A_t.append(A[i])
            except IndexError:
                # kpts are irreducible representation (ikpts).
                # look for time-reversed k-point k = -k + N.
                i = np.where(distance(((-k) % 1) - kpts % 1) < 1e-7)[0][0]
                A_t.append(A[i].T)
        fourier_sum(np.array(A_t), _kpts, R, out[j])

    if squeeze:
        return out[:, 0, :, :]
    return out


def build_lattice_vectors(size: Sequence[int]) -> NDArray[np.int_]:
    r"""Integer lattice vectors :math:`(i, j, k)` with :math:`0 \le i < N_1` and so on.

    Parameters
    ----------
    size : sequence of three int
    """
    return np.indices(size).reshape(len(size), -1).T


def unfold_kpts(Kpts: NDArray[np.float64], kpts: NDArray[np.float64]) -> List[List[int]]:
    """Fine-grid k-points that fold onto each k-point of a coarse grid.

    A supercell :math:`N` times larger than the primitive cell has a
    Brillouin zone :math:`N` times smaller; each of its k-points :math:`K`
    collects :math:`N` k-points of the primitive cell. This finds them.

    Parameters
    ----------
    Kpts : ndarray, shape (n_K, 3)
        Coarse grid, of the supercell.
    kpts : ndarray, shape (n_k, 3)
        Fine grid, of the primitive cell. Its size must be a multiple of
        that of ``Kpts`` along each direction.

    Returns
    -------
    k_groups : list of list of int
        ``k_groups[K]`` holds the indices into ``kpts`` that fold onto
        ``Kpts[K]``.
    """
    distance = partial(np.linalg.norm, axis=1)
    Ksize, Koffset = get_monkhorst_pack_size_and_offset(Kpts)
    ksize, koffset = get_monkhorst_pack_size_and_offset(kpts)

    # assert np.allclose(Koffset, koffset), f'Incompatible mesh grids.\
    # They have different offsets {Koffset} vs. {koffset}. \
    #    Force including the gamma point to match the offsets.'

    Nr = ksize // Ksize
    assert np.allclose(
        Nr, Nr.astype(int)
    ), f"\
Incompatible mesh grids. Fine mesh with size {ksize} is \
not a multiple of coarse mesh with size {Ksize}."

    k_groups = []
    for K in Kpts:
        group_indices = []
        for i, j, k in np.ndindex(tuple(Nr)):
            kpt = (K % 1 + (i, j, k)) / Nr
            idx = np.where(distance(kpt - kpts % 1) < 1e-7)[0][0]
            group_indices.append(idx)
        k_groups.append(group_indices)

    return k_groups
