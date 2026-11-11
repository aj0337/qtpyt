# qtpyt

Quantum transport through nanoscale junctions with non-equilibrium Green's
functions (NEGF), in Python.

qtpyt takes the Hamiltonian and overlap of a device and its electrodes in a
localized orbital basis, typically from a DFT calculation, and computes the
transmission, density of states, occupations and local currents. Many-body
corrections (GW, electron–phonon) enter as self-energies.

```{toctree}
:maxdepth: 2
:hidden:

installation
theory/index
api
```

- [Installation](installation.md): dependencies and building the package.
- [Theory](theory/index.md): the NEGF quantities qtpyt computes and its conventions.
- [API reference](api.md): every module, class and function.
