import numbers

import ase.neighborlist
import numpy as np

#### NeighborList
from ase.data import covalent_radii
from ase.neighborlist import NeighborList


def get_neighbors(atoms):
    """Get a list of neighboring atoms for each atom in the given structure based on covalent radii.

    The covalent radii are used to define an effective bonding distance within which neighboring atoms
    are identified. The `NeighborList` class from ASE is employed here to create and manage these lists.

    Args:
        atoms (ase.Atoms): The atomic structure containing atomic positions and types.

    Returns:
        list[list[int]]: A list of lists where each inner list contains the indices of neighboring atoms for each atom.

    Logic:
        - Retrieve covalent radii for each atom type to define neighbor distances.
        - Use the `NeighborList` object to calculate neighbors within these distances.
        - For each atom, `nl.get_neighbors` provides the list of neighboring atom indices, which are stored in `nlists`.

    """
    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nl.update(atoms)

    nlists = []
    for a in atoms:
        nlists.append(nl.get_neighbors(a.index)[0])

    return nlists


def argsort(positions, order="xyz"):
    """Sorts indices of atoms based on their positions in a specific coordinate order.

    This function lexicographically sorts atom positions, i.e., ordering atoms by x, then y, then z.
    The primary, secondary, and tertiary sorting coordinates are determined by the `order` argument.

    Args:
        positions (np.ndarray): 2D array with atomic positions as rows.
        order (str): Specifies sorting order by axes, e.g., "xyz" to sort by x first, then y, then z.

    Returns:
        np.ndarray: Sorted indices for atomic positions based on the specified order.

    Logic:
        - Use numpy's `lexsort` for multi-key sorting.
        - `order` determines the primary, secondary, and tertiary axes for sorting by converting each axis into its corresponding index.

    """
    i, j, k = ["xyz".index(i) for i in order]
    idx = np.lexsort((positions[:, k], positions[:, j], positions[:, i]))
    return idx


class Basis:
    """Represents a basis function descriptor for atomic orbitals in a structure.

    This class stores and manages information about atomic basis functions for a system.
    It is designed to track the number of atomic orbitals (basis functions) per atom,
    and provide methods to manipulate and query this data.

    Attributes:
        atoms (ase.Atoms): The atomic structure object.
        nao_a (np.ndarray): Number of atomic orbitals per atom (basis functions).
        M_a (np.ndarray): Cumulative orbital indices per atom, starting with zero.
        nao (int): Total number of atomic orbitals in the system.

    """

    def __init__(self, atoms, nao_a, M_a=None):
        """Initializes the Basis object with atomic orbitals for each atom.

        Args:
            atoms (ase.Atoms): The atomic structure object.
            nao_a (np.ndarray): The number of atomic orbitals per atom.
            M_a (np.ndarray, optional): Cumulative orbital indices for atoms. Defaults to None, where it will be calculated.

        Logic:
            - `nao_a` is a list of orbital counts per atom.
            - `M_a` provides the cumulative count of orbitals up to each atom.
            - The total number of atomic orbitals (`nao`) is calculated as the sum of `nao_a`.
            - If `M_a` is not provided, it is initialized as the cumulative sum of `nao_a` with a zero start.

        """
        self.atoms = atoms
        self.nao_a = np.array(nao_a, ndmin=1)
        if M_a is None:
            M_a = np.cumsum(np.insert(self.nao_a[:-1], 0, 0))
        self.M_a = np.array(M_a, ndmin=1)
        self.nao = np.sum(self.nao_a)  # Total # of atomic orbitals

    @classmethod
    def from_dictionary(cls, atoms, basis):
        """Constructs a Basis object from an atoms object and a dictionary.

        Args:
            atoms (ase.Atoms): The atomic structure object.
            basis (dict): Dictionary mapping atomic symbols to the number of atomic orbitals per atom.

        Returns:
            Basis: Initialized Basis object.

        Logic:
            - The method converts a dictionary mapping atomic symbols to basis functions into a list of orbital counts per atom.
            - This is passed to initialize a `Basis` object using `nao_a` from the basis dictionary.

        """
        nao_a = [basis[symbol] for symbol in atoms.symbols]
        return cls(atoms, nao_a)

    @property
    def centers(self):
        """Expands atomic positions to correspond with the number of basis functions.

        Returns:
            np.ndarray: Array of atomic positions, repeated to match the number of basis functions per atom.

        Logic:
            - Each atomic position is repeated by the number of orbitals for that atom (`nao_a`).
            - The helper function `_expand` handles the repetition.

        """
        return self._expand(self.atoms.positions)

    def repeat(self, N):
        """Creates a repeated basis object over the specified periodicity.

        Args:
            N (tuple of int): Pattern for repeating the atomic structure along each axis.

        Returns:
            Basis: New Basis object with repeated atomic and orbital information.

        Logic:
            - Repeats the orbital counts and atomic positions using `nao_a` and `atoms.repeat(N)`.

        """
        nao_a = np.tile(self.nao_a, np.prod(N))
        atoms = self.atoms.repeat(N)
        return Basis(atoms, nao_a)

    def argsort(self, order="xyz"):
        """Sort basis indices using corresponding atomic positions.

        The first character is used as primary coordinate and so on.
        NOTE : Opposite to np.lexsort

        Example:
        In [2]: basis.nao_a
        Out[2]: array([2, 1, 2, 1])

        In [38]: basis.atoms.positions
        Out[38]:
        array([[0. , 0. , 0. ],
            [0.5, 0.5, 0.5],
            [0. , 1. , 0. ],
            [0.5, 1.5, 0.5]])

        In [39]: basis.atoms.positions[basis.argsort()]
        Out[39]:
        array([[0. , 0. , 0. ],
            [0. , 1. , 0. ],
            [0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5]])

        In [41]: basis.nao_a[basis.argsort()]
        Out[41]: array([2, 2, 1, 1])

        """
        return argsort(self.atoms.positions, order)

    def get_indices(self, indices=None):
        """Get basis function indices.

        Args:
            indices : (optional) take subset/sorted atomic indices.
        """
        if indices is None:
            M_a, nao_a, nao = self.M_a, self.nao_a, self.nao
        elif isinstance(indices, numbers.Integral):
            return np.arange(self.nao_a[indices]) + self.M_a[indices]
        else:
            M_a = self.M_a[indices]
            nao_a = self.nao_a[indices]
            nao = np.sum(nao_a)

        idx = self._expand(M_a, nao_a)
        i0 = M_a[0]
        for i in range(1, nao):
            if idx[i] == i0:
                idx[i] = idx[i - 1] + 1
            else:
                i0 = idx[i]
        return idx

    def take(self, basis_dict):
        """Take basis function indices for subset of elements' kind.

        Args:
            basis_dict : dictionary of element keys and indices value(s).
                { 'O' : [3, .. ], ..  'C' : 2, .. }
        """
        idxlist = []
        for kind, idx in basis_dict.items():
            if isinstance(idx, numbers.Integral):
                idx = [idx]
            for a in self.atoms.symbols.search(kind):
                for i in idx:
                    idxlist.append(self.M_a[a] + i)
        idxlist = np.asarray(idxlist)
        idxlist.sort()
        return idxlist

    def extract(self):
        """Extract a subset of the atoms starting from zero index."""
        return self.__class__(self.atoms, self.nao_a)

    def __getitem__(self, i):
        """Return a subset of the atoms."""
        return self.__class__(self.atoms[i], self.nao_a[i], self.M_a[i])

    def __len__(self):
        return self.nao

    def _expand(self, array, repeats=None):
        """Helper function. Expand atoms array to basis functions."""
        if repeats is None:
            repeats = self.nao_a
        return np.repeat(array, repeats, axis=0)
