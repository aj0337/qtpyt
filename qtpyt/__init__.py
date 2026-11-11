r"""qtpyt: quantum transport through nanoscale junctions with non-equilibrium Green's functions.

qtpyt computes the electronic transport properties of a device (a molecule,
a nanoribbon, a defect) connected to two electrodes. The device and the
electrodes are described by a Hamiltonian :math:`H` and overlap :math:`S` in
a localized orbital basis, typically from a DFT calculation. From these the
package builds the retarded Green's function of the device with the
electrodes attached as self-energies, and derives the transmission, density
of states, occupations and local currents.

Array backend
-------------
``qtpyt.xp`` is CuPy when CuPy is installed and NumPy otherwise, and
``qtpyt.xla`` is the matching linear-algebra module. Matrices built inside
the package live on that backend, so with CuPy installed the arrays returned
by Green's functions and self-energies are ``cupy.ndarray`` and stay on the
GPU. Every array type written ``ndarray`` in this documentation means an
array of the active backend.

Units
-----
The package does not convert units. Energies (including ``eta``, ``kt``,
``mu`` and biases) are in the unit of the Hamiltonian you provide, eV when
it comes from ASE or GPAW. Positions are in Å, as in ASE.
"""
# from .calculators import TransportCalculator
# __all__ = ['TransportCalculator']
import numpy as np
from scipy import linalg

xp = np
xla = linalg

try:
    import cupy as cp
except:
    pass
else:
    xp = cp
    xla = cp.linalg


# try:
#    import _cppmodule as _cpp
# except ImportError as e:
#    if "GLIBCXX" in str(e):
#        msg = ("Cannot import __cppmodule")
#        raise ImportError(msg).with_traceback(e.__traceback__)
#    else:
#        raise
