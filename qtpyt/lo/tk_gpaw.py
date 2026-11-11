r"""Local-orbital tools that read the orbital layout from a GPAW calculator.

Counterparts of :mod:`qtpyt.lo.tools` for code that works with a GPAW
calculator instead of a :class:`~qtpyt.basis.Basis`. Requires GPAW.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from numpy.typing import NDArray

import numpy as np
from scipy import linalg as la
from ase.calculators.singlepoint import SinglePointCalculator
from gpaw.setup import types2atomtypes
from gpaw.basis_data import Basis
from .tools import (
    subdiagonalize,
    get_orthonormal_subspace,
    get_subspace,
    rotate_matrix,
    dagger,
)
from .block import block


def get_atom_indices(calc: Any, a: Union[int, str, Sequence[Union[int, str]]]) -> List[int]:
    """Atom indices from indices and chemical symbols.

    Parameters
    ----------
    calc : gpaw.GPAW
        Calculator holding the atoms.
    a : int, str or sequence of int and str
        Atom indices, or element symbols standing for all atoms of that
        element (also in a repeated ``calc.atoms``).

    Returns
    -------
    indices : list of int
        Sorted atom indices.
    """
    from .tk_calc import get_info

    M_a, nao_a, nao = get_info(calc, ["M_a", "nao_a", "nao"])
    if isinstance(a, (int, str)):
        a = [a]
    ai = []
    for elem in a:
        if isinstance(elem, int):
            ai.append(elem)
        elif isinstance(elem, str):
            symbol = elem
            for a0 in range(len(calc.atoms)):
                offset = 0
                if a0 >= len(M_a):  # original number of atoms,
                    # e.g. allows calc.atoms = atoms.repeat(..)
                    offset += (a0 // len(M_a)) * len(M_a)
                    a0 %= len(M_a)
                if calc.atoms[a0].symbol == symbol:
                    ai.append(a0 + offset)
        else:
            raise Warning("Non valid input {}".format(elem))
    ai = sorted(ai)
    return ai


def get_bfs_indices(calc: Any, a: Union[int, Sequence[int]], method: str = "extend") -> List[Any]:
    """Orbital indices of one or more atoms.

    Parameters
    ----------
    calc : gpaw.GPAW
    a : int or sequence of int
        Atom indices.
    method : {'extend', 'append'}, default 'extend'
        ``'extend'`` returns one flat list, ``'append'`` one list per atom.
    """
    from .tk_calc import get_info

    M_a, nao_a, nao = get_info(calc, ["M_a", "nao_a", "nao"])
    if isinstance(a, (int, np.int32, np.int64)):
        a = [a]
    Mvalues = []
    Mattr = getattr(Mvalues, method)
    for a0 in a:
        M = 0
        M += M_a[a0]
        Mattr(list(range(M, M + nao_a[a0])))
    return Mvalues


def get_bfs_atoms(atoms: Any, basis: Optional[dict] = None, default: str = "dzp", method: str = "extend") -> List[Any]:
    """Orbital indices of every atom, from the GPAW basis-set sizes.

    Parameters
    ----------
    atoms : ase.Atoms
    basis : dict, optional
        ``{symbol: basis-set name}``; ``default`` for all elements if omitted.
    default : str, default 'dzp'
        Basis set of elements missing from ``basis``.
    method : {'extend', 'append'}, default 'extend'
        One flat list, or one list per atom.
    """
    if basis is None:
        basis = {symbol: default for symbol in set(atoms.symbols)}
    symbols = atoms.get_chemical_symbols()
    basis_a = types2atomtypes(symbols, basis, default)

    nao_a = [Basis(symbol, type).nao for symbol, type in zip(symbols, basis_a)]

    Mvalues = []
    Mattr = getattr(Mvalues, method)
    i0 = 0
    for nao in nao_a:
        i1 = i0 + nao
        Mattr(list(range(i0, i1)))
        i0 = i1
    return Mvalues


def subdiagonalize_atoms(calc: Any, h_ii: NDArray, s_ii: NDArray, a: Optional[Union[int, Sequence[int]]] = None) -> Tuple[NDArray, List[NDArray]]:
    """Rotation that diagonalizes the Hamiltonian on each selected atom, see :func:`qtpyt.lo.tools.subdiagonalize_atoms`.

    Parameters
    ----------
    calc : gpaw.GPAW
    h_ii, s_ii : ndarray, shape (n, n)
        Hamiltonian and overlap.
    a : int or sequence of int, optional
        Atoms to treat; all if omitted.
    """
    from scipy.linalg import block_diag

    if a is None:
        a = range(len(calc.atoms))
    if isinstance(a, int):
        a = [a]
    U_mm = []
    e_aj = []
    na_tot = len(calc.atoms)
    for a0 in range(na_tot):
        bfs = get_bfs_indices(calc, a0)
        if a0 in a:
            v_jj, e_j = get_orthonormal_subspace(h_ii, s_ii, bfs)
            U_mm.append(v_jj)
            e_aj.append(e_j)
        else:
            U_mm.append(np.eye(len(bfs)))
    U_mm = block_diag(*U_mm)
    return U_mm, e_aj


def subdiagonalize_blocks(calc: Any, h_ii: NDArray, s_ii: NDArray, a: Optional[Sequence[int]] = None) -> Tuple[NDArray, List[NDArray]]:
    """Block diagonal rotation built only from the listed atoms, in the order given.

    Unlike :func:`subdiagonalize_atoms`, atoms not in ``a`` get no block,
    so the rotation spans only the orbitals of ``a``.

    Parameters
    ----------
    calc : gpaw.GPAW
    h_ii, s_ii : ndarray, shape (n, n)
    a : sequence of int
        Atoms to treat.
    """
    from scipy.linalg import block_diag

    U_mm = []
    e_aj = []
    for a0 in a:
        bfs = get_bfs_indices(calc, a0)
        v_jj, e_j = get_orthonormal_subspace(h_ii, s_ii, bfs)
        U_mm.append(v_jj)
        e_aj.append(e_j)
    U_mm = block_diag(*U_mm)
    return U_mm, e_aj


def get_bf_centers(atoms: Any, basis: Optional[dict] = None) -> NDArray[np.float64]:
    """Position of every orbital, the position of its atom.

    Parameters
    ----------
    atoms : ase.Atoms
        With a GPAW calculator attached, or without one (then the
        ``basis`` sets, default ``'dzp'``, give the orbital counts).
    basis : dict, optional
        ``{symbol: basis-set name}``.

    Returns
    -------
    centers : ndarray, shape (n_orb, 3)
        In Å.
    """

    from ase.calculators.singlepoint import SinglePointCalculator
    from gpaw.setup import types2atomtypes

    calc = atoms.get_calculator()
    if calc is None or isinstance(calc, SinglePointCalculator):
        symbols = atoms.get_chemical_symbols()
        basis_a = types2atomtypes(symbols, basis, "dzp")
        nao_a = [Basis(symbol, type).nao for symbol, type in zip(symbols, basis_a)]
    else:
        if not calc.initialized:
            calc.initialize(atoms)
        nao_a = [calc.wfs.setups[a].nao for a in range(len(atoms))]
    pos_ic = []
    for pos, nao in zip(atoms.get_positions(), nao_a):
        pos_ic.extend(pos[None].repeat(nao, 0))
    return np.array(pos_ic)


def sum_bf_atom(calc: Any, X_i: NDArray, n_a: int = 0) -> NDArray:
    """Sum an orbital-resolved quantity over the orbitals of each atom.

    For example orbital occupations to atomic charges, or a PDOS to an
    atom-resolved DOS.

    Parameters
    ----------
    calc : gpaw.GPAW
    X_i : ndarray, shape (n_orb,)
    n_a : int, default 0
        Number of atoms to include; all if 0.
    """
    if n_a == 0:
        n_a = len(calc.atoms)
    X_a = np.zeros(n_a, dtype=X_i.dtype)
    i0 = 0
    for a0 in range(n_a):
        i1 = i0 + calc.wfs.setups[a0].nao
        X_a[a0] = sum(X_i[i0:i1])
        i0 = i1
    return X_a


def flatten(iterables: Iterable[Iterable[Any]]) -> Iterator[Any]:
    """Elements of nested iterables, one level deep.

    Parameters
    ----------
    iterables : iterable of iterable
    """
    return (elem for iterable in iterables for elem in iterable)


def get_U1(bfs_m: Any, bfs_i: Any, c_MM: Optional[NDArray] = None, apply: bool = False) -> NDArray:
    """Reordering that puts the active orbitals last, see :func:`qtpyt.lo.tools.get_U1`.

    Parameters
    ----------
    bfs_m, bfs_i : ndarray of int
        Active and remaining orbitals.
    c_MM : ndarray, optional
        Rotation whose columns are reordered.
    apply : bool, default False
        Overwrite ``bfs_m`` and ``bfs_i`` with their new positions.
    """
    nbf = len(bfs_m) + len(bfs_i)
    if c_MM is None:
        c_MM = np.eye(nbf)
    U_mm = c_MM.take(bfs_m, axis=1)
    U_ii = c_MM.take(bfs_i, axis=1)
    U1 = np.array(block([[1, 0], [U_ii, U_mm]]))
    """If apply is True permute inplace bfs_m bfs_i"""
    if apply:
        bfs_m[:] = range(nbf - len(bfs_m), nbf)
        bfs_i[:] = range(nbf - len(bfs_m))
    return U1


def get_U2(s1_MM: NDArray, bfs_m: Sequence[int], bfs_i: Sequence[int]) -> NDArray:
    """Rotation that makes the remaining orbitals orthogonal to the active ones, see :func:`qtpyt.lo.tools.get_U2`.

    Parameters
    ----------
    s1_MM : ndarray, shape (n, n)
        Overlap, with the active orbitals last.
    bfs_m, bfs_i : sequence of int
    """
    s_mm = get_subspace(s1_MM, bfs_m)
    s_mi = s1_MM.take(bfs_m, axis=0).take(bfs_i, axis=1)
    U_mi = -la.inv(s_mm).dot(s_mi)
    U_ii = np.eye(len(bfs_i))
    U2 = np.array(block([[U_ii, 0], [U_mi, np.eye(len(bfs_m))]]))
    return U2


def get_mm_ii_im(h_MM: NDArray, s_MM: NDArray, bfs_m: Sequence[int], bfs_i: Sequence[int]) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Blocks on the active orbitals, on the rest, and between them, see :func:`qtpyt.lo.tools.split_subspaces`.

    Parameters
    ----------
    h_MM, s_MM : ndarray, shape (n, n)
    bfs_m, bfs_i : sequence of int
    """
    h_ii = get_subspace(h_MM, bfs_i)
    s_ii = get_subspace(s_MM, bfs_i)
    h_mm = get_subspace(h_MM, bfs_m)
    s_mm = get_subspace(s_MM, bfs_m)
    h_im = h_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    s_im = s_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    return (h_mm, s_mm), (h_ii, s_ii), (h_im, s_im)


def extract_orthogonal_subspaces(
    h_MM: NDArray, s_MM: NDArray, bfs_m: Any, bfs_i: Any, c_MM: Optional[NDArray] = None, orthogonal: bool = True, apply: bool = True
) -> Any:
    """Split into an active subspace and the rest, see :func:`qtpyt.lo.tools.extract_orthogonal_subspaces`.

    Parameters
    ----------
    h_MM, s_MM : ndarray, shape (n, n)
    bfs_m, bfs_i : ndarray of int
        Active and remaining orbitals, overwritten with their new positions.
    c_MM : ndarray, optional
        Rotation applied first.
    orthogonal : bool, default True
        Make the rest orthogonal to the active subspace.
    apply : bool, default True
        Return the blocks; with ``False`` return the rotation only.
    """
    U1 = get_U1(bfs_m, bfs_i, c_MM, apply=True)  # Modifies bfs_m, bfs_i
    h1_MM = rotate_matrix(h_MM, U1)
    s1_MM = rotate_matrix(s_MM, U1)
    """If orthogonal is True set s_im to zeros."""
    if orthogonal:
        U2 = get_U2(s1_MM, bfs_m, bfs_i)
        h2_MM = rotate_matrix(h_MM, U1.dot(U2))
        s2_MM = rotate_matrix(s_MM, U1.dot(U2))
    else:
        h2_MM = h1_MM
        s2_MM = s1_MM
    if apply is False:
        return U1.dot(U2)
    return get_mm_ii_im(h2_MM, s2_MM, bfs_m, bfs_i)


def get_orthogonal_subspaces(calc: Any, h_MM: NDArray, s_MM: NDArray, a: Any = None, cutoff: float = np.inf, orthogonal: bool = True) -> Any:
    r"""Active subspace of local orbitals on chosen atoms, see :func:`qtpyt.lo.tools.get_orthogonal_subspaces`.

    Parameters
    ----------
    calc : gpaw.GPAW
    h_MM, s_MM : ndarray, shape (n, n)
    a : int or sequence of int
        Atoms that host the active orbitals.
    cutoff : float, default inf
        Keep local orbitals with :math:`|\varepsilon| < \mathrm{cutoff}`.
    orthogonal : bool, default True
        Make the rest orthogonal to the active subspace.
    """
    nbf = len(h_MM)
    c_MM, e_aj = subdiagonalize_atoms(calc, h_MM, s_MM, a)
    bfs_imp = get_bfs_indices(calc, a)
    bfs_m = [
        bfs_imp[i] for i, eigval in enumerate(flatten(e_aj)) if abs(eigval) < cutoff
    ]
    bfs_i = list(np.setdiff1d(range(nbf), bfs_m))
    return extract_orthogonal_subspaces(h_MM, s_MM, bfs_m, bfs_i, c_MM, orthogonal)


def get_ll_mm_rr(H: NDArray, S: NDArray, nbf_l: int, nbf_m: int, nbf_r: int) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Central block and its couplings, see :func:`qtpyt.lo.tools.get_ll_mm_rr`.

    Parameters
    ----------
    H, S : ndarray, shape (n, n)
    nbf_l, nbf_m, nbf_r : int
    """
    rng_l = list(range(nbf_l))
    rng_m = list(range(nbf_l, nbf_l + nbf_m))
    rng_r = list(range(nbf_l + nbf_m, nbf_l + nbf_m + nbf_r))
    s_lm = S.take(rng_l, 0).take(rng_m, 1)
    h_lm = H.take(rng_l, 0).take(rng_m, 1)
    s_rm = S.take(rng_r, 0).take(rng_m, 1)
    h_rm = H.take(rng_r, 0).take(rng_m, 1)
    h_mm = get_subspace(H, rng_m)
    s_mm = get_subspace(S, rng_m)
    return (h_lm, s_lm), (h_mm, s_mm), (h_rm, s_rm)


def get_lead_orthogonal(greenfunction: Any) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Device and lead blocks after making the lead layers orthogonal to the device.

    Assembles lead–device–lead :math:`H` and :math:`S`, removes the overlap
    between the device and each lead principal layer, and returns the
    blocks of :func:`get_ll_mm_rr`.

    Parameters
    ----------
    greenfunction : GreenFunction
        Device Green's function whose first two self-energies are the leads,
        with attributes ``h_ii``, ``s_ii``, ``h_im``, ``s_im``.
    """
    s_ll = greenfunction.selfenergies[0].s_ii
    s_rr = greenfunction.selfenergies[1].s_ii
    h_ll = greenfunction.selfenergies[0].h_ii
    h_rr = greenfunction.selfenergies[1].h_ii
    s_lm = greenfunction.selfenergies[0].s_im
    s_rm = greenfunction.selfenergies[1].s_im
    h_lm = greenfunction.selfenergies[0].h_im
    h_rm = greenfunction.selfenergies[1].h_im
    h_mm = greenfunction.H
    s_mm = greenfunction.S
    nbf_l = len(h_ll)
    nbf_r = len(h_rr)
    nbf_m = len(h_mm)
    s_ml = dagger(s_lm)
    s_mr = dagger(s_rm)
    h_ml = dagger(h_lm)
    h_mr = dagger(h_rm)
    U_lm = -la.inv(s_ll).dot(s_lm)
    U_rm = -la.inv(s_rr).dot(s_rm)
    U = np.array(
        block(
            [[np.eye(nbf_l), U_lm, 0], [0, np.eye(nbf_m), 0], [0, U_rm, np.eye(nbf_r)]]
        )
    )
    S = np.array(block([[s_ll, s_lm, 0], [s_ml, s_mm, s_mr], [0, s_rm, s_rr]]))
    H = np.array(block([[h_ll, h_lm, 0], [h_ml, h_mm, h_mr], [0, h_rm, h_rr]]))
    S = rotate_matrix(S, U)
    H = rotate_matrix(H, U)
    return get_ll_mm_rr(H, S, nbf_l, nbf_m, nbf_r)
