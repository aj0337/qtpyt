# Theory

The central quantity is the retarded Green's function of the device with the
electrodes attached as self-energies,

$$
G^r(E) = \left[(E + i\eta) S - H - \Sigma_L(E) - \Sigma_R(E)\right]^{-1},
$$

from which the transmission
$T(E) = \operatorname{Tr}[\Gamma_L G^r \Gamma_R G^a]$ follows, with
$\Gamma_\alpha = i(\Sigma_\alpha - \Sigma_\alpha^\dagger)$. See
{class}`qtpyt.base.greenfunction.GreenFunction`.

Units: qtpyt does not convert units. Energies are in the unit of the
Hamiltonian you provide (eV from ASE/GPAW), positions in Å.
