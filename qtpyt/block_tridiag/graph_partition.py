r"""Splitting a device into principal layers from the connectivity of its Hamiltonian.

To use the block tridiagonal solvers, the orbitals of the device must be
grouped into layers such that each layer couples only to its two
neighbours. Starting from the atoms of the left and right principal layers,
a breadth-first search over the bonds (pairs of atoms with a matrix element
above a cutoff) assigns every atom to a layer: the atoms one bond away from
the left layer form the next layer, and so on, from both sides towards the
middle.
"""
from __future__ import annotations

from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from numpy.typing import NDArray

import numpy as np
from collections import deque, namedtuple, defaultdict
from functools import singledispatch

cutoff = 1e-4


def get_graph(basis: Any, mat: NDArray, cutoff: float = cutoff) -> Dict[int, Any]:
    """Bond graph of the atoms: two atoms are bonded when they share a large matrix element.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the device.
    mat : ndarray, shape (n_orb, n_orb)
        Matrix that defines the bonds, usually the Hamiltonian.
    cutoff : float, default 1e-4
        Two atoms are bonded when any element of their block of ``mat``
        exceeds ``cutoff``. The comparison is on the signed value, not on
        the magnitude.

    Returns
    -------
    graph : dict
        Vertex for each atom, with its list of bonded atoms in
        ``neighbors`` and its layer in ``tag`` (zero until assigned).
    """

    num_vertices = len(basis.atoms)
    # bfs_ai = get_bfs_indices(basis, range(num_vertices), 'append')

    # Vertex
    class vertex:
        """
            :tag       level set index (block index in matrix)
            :neighbors neighborlist vertices (neighborlist atoms)
            :index     vertex in graph (atom index in atoms)
        """

        def __init__(self, tag, neighbors):
            self.tag = tag
            self.neighbors = neighbors
            self.index = 0

    # Graph ::dict of vertices
    graph = defaultdict(lambda: vertex(0, []))

    # Start
    for a0 in range(num_vertices):
        # Find neighborlist for a1 > a0
        for a1 in range(a0, len(basis.atoms)):
            h_ij = mat.take(basis.get_indices(a0), axis=0).take(
                basis.get_indices(a1), axis=1
            )
            if np.any(h_ij > cutoff):
                graph[a0].neighbors.append(a1)
                graph[a0].index = a0
                # Do not double count a0
                if a0 != a1:
                    # insert neighbor a0 in a1.neighborlist
                    graph[a1].neighbors.append(a0)

    return graph


def BFS(Q: Deque[int], graph: Dict[int, Any], subset: Iterable[int]) -> None:
    """Assign layers by breadth-first search from the given starting atoms.

    Each atom reached gets the layer of the atom it was reached from, plus
    one step away from the start (tags grow for positive starts and
    decrease for negative ones). Tags are written into ``graph``.

    Parameters
    ----------
    Q : collections.deque of int
        Starting atoms, whose tags are already set.
    graph : dict
        Bond graph from :func:`get_graph`, modified in place.
    subset : iterable of int
        Only atoms in this set are visited.
    """

    subset = list(subset)
    # Start
    while Q:
        v = Q.pop()
        adjacentEdges = graph[v].neighbors  # grapht_neighbors(v)[0]
        #
        for w in adjacentEdges:
            # Atom not discovered and in subset
            if (not graph[w].tag) & (w in subset):
                # Assign parent previous level set
                graph[w].tag = (
                    graph[v].tag + 1 if graph[v].tag > 0 else graph[v].tag - 1
                )
                Q.appendleft(w)


def get_tridiagonal_nodes(basis: Any, mat: NDArray, pl1: int, pl2: Optional[int] = None, cutoff: float = cutoff) -> List[int]:
    """Orbital indices where the device splits into nearest-neighbour layers.

    The first ``pl1`` atoms form the first layer and the last ``pl2``
    atoms the last one. The left half of the atoms is layered from the
    left, the right half from the right, and the two meet in the middle.
    Atoms must be ordered along the transport direction.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the device.
    mat : ndarray, shape (n_orb, n_orb)
        Matrix that defines the bonds, see :func:`get_graph`.
    pl1 : int
        Number of atoms in the left principal layer.
    pl2 : int, optional
        Number of atoms in the right principal layer, ``pl1`` if omitted.
    cutoff : float, default 1e-4
        Bond threshold, see :func:`get_graph`.

    Returns
    -------
    nodes : list of int
        ``N + 1`` orbital indices; layer ``q`` holds orbitals
        ``nodes[q]`` to ``nodes[q + 1] - 1``.
    """

    if pl2 is None:
        pl2 = pl1

    # Construct graph
    graph = get_graph(basis, mat, cutoff)

    # Number of vertices
    num_vertices = len(graph)

    # Pinn first and last l.s.
    v0 = range(pl1)
    vN = range(num_vertices - pl2, num_vertices)
    _set_tags(graph, v0, 1)
    _set_tags(graph, vN, -1)

    # BFS from left for left subset vertices
    Q = deque(range(pl1))
    subset = range(num_vertices // 2)
    BFS(Q, graph, subset)
    # BFS from right for right subset vertices
    Q = deque(range(num_vertices - 1, num_vertices - pl2 - 1, -1))
    subset = range(num_vertices // 2, num_vertices)
    BFS(Q, graph, subset)

    # from IPython import embed
    # embed()

    # Assign last l.s. from right to last l.s. from left
    tags = _get_tags(graph)
    tmin = min(tags)
    tmax = max(tags)
    for a in graph.values():
        if a.tag == tmin:
            a.tag = tmax

    # Rename positive
    tags = _get_tags(graph)
    u = np.unique(tags).tolist()
    for a in graph.values():
        # Start from 0 index
        if a.tag > 0:
            a.tag -= 1
        # Map negative to positive
        else:
            a.tag %= len(u)

    # Get nodes:= cut edges
    tags = _get_tags(graph)
    u = np.unique(tags).tolist()
    nodes = [0]  # N+1 nodes
    for u0 in u:
        indices = _get_indices(graph, u0)
        bfs_i = basis.get_indices(int(np.max(indices)))
        nodes.append(np.max(bfs_i) + 1)

    return nodes


def split_tridiagonal(mat: NDArray, nodes: Sequence[int]) -> Tuple[List[NDArray], List[NDArray]]:
    """Diagonal and upper off-diagonal blocks of a matrix for given layer boundaries.

    Parameters
    ----------
    mat : ndarray, shape (n_orb, n_orb)
    nodes : sequence of int
        Layer boundaries, see :func:`get_tridiagonal_nodes`.

    Returns
    -------
    mat_list_ii : list of ndarray
        Blocks :math:`M_{qq}`.
    mat_list_ij : list of ndarray
        Blocks :math:`M_{q,q+1}`. Elements beyond the neighbouring layer are
        dropped.
    """

    N = len(nodes) - 1

    mat_list_ii = [None for _ in range(N)]
    mat_list_ij = [None for _ in range(N - 1)]

    for q in range(N):
        q0 = nodes[q]
        q1 = nodes[q + 1]
        row = mat.take(range(q0, q1), axis=0)
        mat_list_ii[q] = row.take(range(q0, q1), axis=1)
        if q < (N - 1):
            q2 = nodes[q + 2]
            mat_list_ij[q] = row.take(range(q1, q2), axis=1)

    return mat_list_ii, mat_list_ij


def _tridiagonalize(nodes, h, s):
    h_list_ii, h_list_ij = split_tridiagonal(h, nodes)
    s_list_ii, s_list_ij = split_tridiagonal(s, nodes)

    hs_list_ii = [np.stack([h, s]) for h, s in zip(h_list_ii, s_list_ii)]
    hs_list_ij = [np.stack([h, s]) for h, s in zip(h_list_ij, s_list_ij)]

    return hs_list_ii, hs_list_ij


@singledispatch
def tridiagonalize(basis: Any, h: NDArray, s: NDArray, pl1: int, pl2: Optional[int] = None, cutoff: float = cutoff) -> Tuple[List[NDArray], List[NDArray]]:
    """Split :math:`H` and :math:`S` into layers found from the bond graph.

    Parameters
    ----------
    basis : Basis
        Orbital basis of the device.
    h, s : ndarray, shape (n_orb, n_orb)
        Hamiltonian and overlap.
    pl1, pl2, cutoff
        See :func:`get_tridiagonal_nodes`.

    Returns
    -------
    hs_list_ii, hs_list_ij : list of ndarray, shape (2, ., .)
        Stacked ``(H, S)`` blocks in the form taken by
        :class:`~qtpyt.block_tridiag.greenfunction.GreenFunction`.
    """
    nodes = get_tridiagonal_nodes(basis, h, pl1, pl2, cutoff)
    return _tridiagonalize(nodes, h, s)


@tridiagonalize.register(list)
def tridiagonalize(nodes: List[int], h: NDArray, s: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
    """Split :math:`H` and :math:`S` into layers at given orbital boundaries.

    Parameters
    ----------
    nodes : list of int
        Layer boundaries, see :func:`get_tridiagonal_nodes`.
    h, s : ndarray, shape (n_orb, n_orb)
        Hamiltonian and overlap.

    Returns
    -------
    hs_list_ii, hs_list_ij : list of ndarray, shape (2, ., .)
        Stacked ``(H, S)`` blocks in the form taken by
        :class:`~qtpyt.block_tridiag.greenfunction.GreenFunction`.
    """
    return _tridiagonalize(nodes, h, s)


################# Helpers ###################


def _set_tags(graph, indices, tag):
    if isinstance(indices, int):
        indices = [indices]
    for i in indices:
        graph[i].tag = tag


def _get_tags(graph):
    return [a.tag for a in graph.values()]


def _get_indices(graph, tag):
    if isinstance(tag, int):
        tag = [tag]
    return sorted([a.index for a in graph.values() if a.tag in tag])
