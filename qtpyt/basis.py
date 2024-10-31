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
        """Sorts basis indices using corresponding atomic positions based on the given order.

        Args:
            order (str): Coordinate order for sorting (e.g., "xyz").

        Returns:
            np.ndarray: Sorted indices based on specified order.

        Logic:
            - Uses the `argsort` function to lexicographically order atoms based on specified coordinates.

        """
        return argsort(self.atoms.positions, order)

    def get_indices(self, indices=None):
        """Gets basis function indices for a subset of atoms.

        Args:
            indices (int or list of int, optional): Atomic indices for which to retrieve basis functions. Defaults to all.

        Returns:
            np.ndarray: Basis function indices for the specified subset.

        Logic:
            - If `indices` is None, uses all atoms; if integer, returns a single atom’s basis functions; otherwise, uses subset.
            - The `_expand` method is applied to expand cumulative orbital indices (`M_a`) based on `nao_a`.

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
        """Gets indices for a subset of basis functions based on atomic element types.

        Args:
            basis_dict (dict): Dictionary mapping atomic symbols to list of basis indices.

        Returns:
            np.ndarray: Sorted indices for the specified basis functions.

        Logic:
            - Maps atomic symbols in `basis_dict` to atom indices using the basis indices for each symbol.
            - The indices are sorted before returning.

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
        """Extracts a subset of atoms and basis functions starting from zero index.

        Returns:
            Basis: A new Basis object with the subset.

        Logic:
            - Returns a new instance of Basis, retaining only the starting subset.

        """
        return self.__class__(self.atoms, self.nao_a)

    def __getitem__(self, i):
        """Accesses a subset of atoms and corresponding basis functions.

        Args:
            i (int): Index for the subset.

        Returns:
            Basis: Subset of atoms and basis functions.

        Logic:
            - Accesses the indexed subset by slicing `atoms`, `nao_a`, and `M_a`.

        """
        return self.__class__(self.atoms[i], self.nao_a[i], self.M_a[i])

    def __len__(self):
        """Gets the total number of basis functions.

        Returns:
            int: Total number of basis functions.

        Logic:
            - Simply returns the attribute `nao`, calculated during initialization.

        """
        return self.nao

    def _expand(self, array, repeats=None):
        """Expands an atomic property array to the size of the basis functions.

        Args:
            array (np.ndarray): Array of atomic properties.
            repeats (np.ndarray, optional): Number of repetitions for each atom, defaults to `nao_a`.

        Returns:
            np.ndarray: Expanded array.

        Logic:
            - Uses `np.repeat` to duplicate each element in `array` according to `repeats`, matching the total number of basis functions.

        """
        if repeats is None:
            repeats = self.nao_a
        return np.repeat(array, repeats, axis=0)
