r"""Many-body self-energies on a real-energy grid: Hartree, Fock, RPA screening and GW.

Green's functions, polarizations and self-energies are tabulated on a
uniform energy grid and distributed over MPI ranks
(:class:`~qtpyt.parallel.egrid.GridDesc`). Convolutions in energy are done as
products in time, with fast Fourier transforms (``mkl_fft`` when installed,
NumPy otherwise). Two-particle quantities such as the bare interaction
:math:`V` and the polarization :math:`P` are matrices in the pair-orbital
index :math:`(ij)`, optionally reduced to a smaller set of optimized pair
orbitals by a rotation ``U_pq``. Spin degeneracy is assumed throughout.
"""
import numpy as np

try:
    import mkl_fft
except:
    fft = np.fft.fft
    ifft = np.fft.ifft
else:
    fft = mkl_fft.fft
    ifft = mkl_fft.ifft
