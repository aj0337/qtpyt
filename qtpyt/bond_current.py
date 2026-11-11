r"""Local (bond) currents: where the electrons flow inside the device.

The transmission says how much current passes; the bond currents say
along which bonds it passes. They are computed from the partial spectral
function :math:`A_\alpha = G^r \Gamma_\alpha G^a` of one lead (see
:meth:`~qtpyt.base.greenfunction.GreenFunction.get_spectrals`) and the
Hamiltonian, at a single energy. Plotted as arrows on the structure, they
show current paths, loops and interference [1]_.

References
----------
.. [1] S. Sanz et al., "Crossed graphene nanoribbons as beam splitters and
   mirrors for electron quantum optics".
"""
from __future__ import annotations

from itertools import accumulate
from typing import Any, List, Optional, Sequence, Tuple

from numpy.typing import NDArray
import numpy as np
from numba import njit
from functools import singledispatch

#
from qtpyt.block_tridiag.btmatrix import BTMatrix, _BTBuffer

# NeighborList
from qtpyt.basis import get_neighbors

"""
Module's reference:

    Crossed graphene nanoribbons as beam splitters and mirrors
    for electron quantum optics. Sofia Sanz, et al.

"""


@njit
def _orbital_current(A, H, index_i, index_j, A_dag, H_dag):
    """Compute local current matrix between subset of indices.
    
    J   = H   A    -  H    A
     ij     ji  ij      ij   ji
    """
    ni = len(index_i)
    nj = len(index_j)
    J_ij = np.zeros((ni, nj))
    for i, j in np.ndindex(ni, nj):
        ii = index_i[i]
        jj = index_j[j]
        J_ij[i, j] = np.imag(H_dag[jj, ii] * A[ii, jj] - H[ii, jj] * A_dag[jj, ii])
    return J_ij


def _orbital_current_btd(A, H, index_i, index_j):
    """Block-tridiagonal version of _orbitals_current"""
    nodes = np.cumsum([m.shape[0] for m in H.m_qii])
    # Indices of atoms in recursive block
    # np.searchsorted(..,side='right') because
    # if indices_i[0] == nodes[q=0], then atom i
    # goes to next block q=1
    #
    qi = np.searchsorted(nodes, index_i, side="right")
    qj = np.searchsorted(nodes, index_j, side="right")

    # Check
    assert len(set(qi)) == 1
    assert len(set(qj)) == 1
    # If OK, all orbitals belong to same block
    qi = qi[0]
    qj = qj[0]

    M_q = np.insert(nodes, 0, 0)[:-1]
    index_i = np.asarray(index_i) - M_q[qi]
    index_j = np.asarray(index_j) - M_q[qj]

    if qi == qj:
        A_ij = A.m_qii[qi]
        H_ij = H.m_qii[qi]
        A_ji = A_ij
        H_ji = H_ij
    elif qj == qi + 1:
        A_ij = A.m_qij[qi]
        H_ij = H.m_qij[qi]
        A_ji = A.m_qji[qi]
        H_ji = H.m_qji[qi]
    elif qi == qj + 1:
        A_ij = A.m_qji[qj]
        H_ij = H.m_qji[qj]
        A_ji = A.m_qij[qj]
        H_ji = H.m_qij[qj]

    return _orbital_current(A_ij, H_ij, index_i, index_j, A_ji, H_ji)


def orbital_current(A: Any, H: Any, index_i: Sequence[int], index_j: Sequence[int]) -> NDArray[np.float64]:
    r"""Current between each orbital of one set and each orbital of another.

    Parameters
    ----------
    A : ndarray or BTMatrix, shape (n_orb, n_orb)
        Partial spectral function :math:`A_\alpha(E)` of one lead.
    H : ndarray or BTMatrix, shape (n_orb, n_orb)
        Hamiltonian of the device, of the same kind as ``A``.
    index_i, index_j : sequence of int
        Orbital indices of the two sets, usually the orbitals of two
        bonded atoms. For block tridiagonal matrices each set must lie
        within one block, and the two blocks must be equal or adjacent.

    Returns
    -------
    J_ij : ndarray, shape (len(index_i), len(index_j))
        Orbital currents, without the prefactor :math:`e/h`.

    Notes
    -----
    .. math::

        J_{ij} = \operatorname{Im}\left[H^\dagger_{ji} A_{ij} - H_{ij} A^\dagger_{ji}\right].

    Only :math:`H` enters, not :math:`H - ES`, so the overlap between
    orbitals is neglected. The sign gives the direction of flow between
    the two orbitals.
    """
    if isinstance(A, (BTMatrix, _BTBuffer)):
        return _orbital_current_btd(A, H, index_i, index_j)
    else:
        return _orbital_current(A, H, index_i, index_j, A, H)


def bond_current(A: Any, H: Any, bfs_ai: Sequence[Sequence[int]], nlists: Sequence[Sequence[int]]) -> List[List[float]]:
    r"""Current along every bond, the orbital currents summed over the orbitals of the two atoms.

    Parameters
    ----------
    A : ndarray or BTMatrix
        Partial spectral function :math:`A_\alpha(E)` of one lead.
    H : ndarray or BTMatrix
        Hamiltonian of the device.
    bfs_ai : sequence of sequence of int
        ``bfs_ai[a]`` holds the orbital indices of atom ``a``.
    nlists : sequence of sequence of int
        ``nlists[a]`` holds the atoms bonded to atom ``a``, as from
        :func:`~qtpyt.basis.get_neighbors`.

    Returns
    -------
    J_aa : list of list of float
        ``J_aa[a][n]`` is the current from atom ``a`` to its ``n``-th
        neighbour ``nlists[a][n]``, see :func:`orbital_current`.
    """
    J_aa = [[None for _ in nlist] for nlist in nlists]
    for a0, nlist in enumerate(nlists):
        index_i = bfs_ai[a0]
        for j, a1 in enumerate(nlist):
            index_j = bfs_ai[a1]
            J_aa[a0][j] = orbital_current(A, H, index_i, index_j).sum()
    return J_aa


def quiver_plot(atoms: Any, J_aa: Sequence[Sequence[float]], nlists: Sequence[Sequence[int]], **kwargs: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Arrows for drawing bond currents with :func:`matplotlib.pyplot.quiver`.

    Every bond gives one arrow pointing in the direction of the current.
    A bond with negative current gets an arrow starting at the bond
    midpoint and pointing back towards the first atom.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure the currents were computed for.
    J_aa : sequence of sequence of float
        Bond currents from :func:`bond_current`.
    nlists : sequence of sequence of int
        Neighbour lists used for ``J_aa``.
    **kwargs
        Ignored.

    Returns
    -------
    X : ndarray, shape (n_bonds, 3)
        Arrow origins, in Å.
    U : ndarray, shape (n_bonds, 3)
        Arrow vectors, as long as the bond.
    W : ndarray, shape (n_bonds,)
        Magnitude of each bond current.
    """
    X = []
    U = []
    W = []
    for a in atoms:
        nlist = nlists[a.index]
        pos = np.tile(a.position, (len(nlist), 1))
        dist = atoms[nlist].positions - a.position  # \
        # / atoms.get_distances(a.index, nlist)[:,None]
        X.extend(pos.tolist())
        U.extend(dist.tolist())
        W.extend(J_aa[a.index])

    X = np.asarray(X)
    U = np.asarray(U)
    W = np.asarray(W)
    Wneg = W < 0
    # Shift arrow to moddle of plot
    X[Wneg] += U[Wneg] / 2
    # Reverse arrow
    U[Wneg] *= -1
    W = np.abs(W)
    return X, U, W


def normalize_weigths(W: NDArray[np.float64], normalize: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None, clip: bool = False) -> NDArray[np.float64]:
    """Rescale bond-current magnitudes for a colour map.

    Parameters
    ----------
    W : ndarray
        Bond-current magnitudes.
    normalize : bool, default False
        Map ``W`` linearly onto [0, 1].
    vmin, vmax : float, optional
        Clipping limits, used when ``clip`` is set.
    clip : bool, default False
        Clip ``W`` to ``[vmin, vmax]`` after normalizing.

    Returns
    -------
    W : ndarray
    """
    if normalize:
        W = (W - W.min()) / (W.max() - W.min())
    if clip:
        W = np.clip(W, vmin, vmax)
    return W


class DensityCurrent(object):
    """Bond currents of a device at one energy, one map per lead.

    Parameters
    ----------
    gf : GreenFunction
        Green's function of the device with leads attached, providing
        :meth:`~qtpyt.base.greenfunction.GreenFunction.get_spectrals`.
    basis : Basis
        Orbital basis of the device.
    indices : sequence of int, optional
        Atoms to include. All atoms if omitted.

    Notes
    -----
    Bonds are found from covalent radii with periodic boundaries switched
    off, so no bond crosses the cell boundary. Adjust them with
    :meth:`add_neighbors` and :meth:`pop_neighbors`.
    """

    def __init__(self, gf: Any, basis: Any, indices: Optional[Sequence[int]] = None) -> None:
        self.gf = gf
        self.indices = indices or np.arange(len(basis.atoms))
        self.atoms = basis.atoms[self.indices]
        self.bfs_ai = [basis.get_indices(a) for a in self.indices]
        self.energy = None
        self._set_nlists()

    def __call__(self, energy: float) -> List[List[List[float]]]:
        """Bond currents at energy :math:`E` for each lead, see :func:`bond_current`."""
        # Spectral functions
        if energy != self.energy:
            gf = self.gf
            self.sepectrals = gf.get_spectrals(energy)
        J = []
        for A in self.sepectrals:
            J.append(bond_current(A, gf.H, self.bfs_ai, self.nlists))

        return J

    def _set_nlists(self):
        atoms = self.atoms
        pbc = atoms.pbc.copy()
        # Hack to avoid arrow between boundaries
        atoms.set_pbc(False)
        self.nlists = get_neighbors(atoms)
        atoms.set_pbc(pbc)

    def add_neighbors(self, index: int, neighbors: Sequence[int]) -> None:
        """Add bonds from atom ``index`` to the atoms ``neighbors``.

        Parameters
        ----------
        index : int
            Atom index, within the selected atoms.
        neighbors : sequence of int
            Atoms to bond to it.
        """
        self.nlists[index] = np.insert(self.nlists[index], 0, neighbors)

    def pop_neighbors(self, index: int, neighbors: Sequence[int]) -> None:
        """Remove bonds of atom ``index``.

        Parameters
        ----------
        index : int
            Atom index, within the selected atoms.
        neighbors : sequence of int
            Positions in ``nlists[index]`` of the bonds to remove.
        """
        self.nlists[index] = np.delete(self.nlists[index], neighbors)

    def display(
        self,
        J_aa: Sequence[Sequence[float]],
        nlists: Sequence[Sequence[int]],
        projection: str = "2d",
        plane: str = "xy",
        scale: float = 2,
        normalize: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        clip: bool = False,
    ) -> Any:
        """Draw bond currents as arrows coloured by magnitude.

        Parameters
        ----------
        J_aa : sequence of sequence of float
            Bond currents of one lead.
        nlists : sequence of sequence of int
            Neighbour lists used for ``J_aa``.
        projection : {'2d', '3d'}, default '2d'
            Draw in a plane or in 3D.
        plane : str, default 'xy'
            Plane of the 2D drawing.
        scale : float, default 2
            Cell length in Å per inch of figure size.
        normalize, vmin, vmax, clip
            Colour scaling, see :func:`normalize_weigths`.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm

        # from matplotlib.colors import Normalize
        atoms = self.atoms

        X, U, W = quiver_plot(atoms, J_aa, nlists)
        W = normalize_weigths(W, normalize, vmin, vmax, clip)
        colormap = cm.YlOrBr

        fig = plt.figure(figsize=(atoms.cell[0, 0] / scale, atoms.cell[1, 1] / scale))
        if projection is "3d":
            ax = fig.gca(projection=projection)
            ax.quiver(
                X[:, 0], X[:, 1], X[:, 2], U[:, 0], U[:, 1], U[:, 2], color=colormap(W)
            )
        else:
            ax = fig.gca()
            a = "xyz".index(plane[0])
            b = "xyz".index(plane[1])
            ax.quiver(X[:, a], X[:, b], U[:, a], U[:, b], color=colormap(W))
            ax.scatter(atoms.positions[:, a], atoms.positions[:, b])
        return ax
