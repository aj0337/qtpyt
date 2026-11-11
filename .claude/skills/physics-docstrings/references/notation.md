# qtpyt notation and conventions

Symbols used across the API docs. Use these exact symbols and signs so pages agree with each other. The "Home" column names the object whose docstring carries the full explanation; every other docstring gives one clause and links there.

Everything below was read off the code. When you document a module that introduces a new reusable symbol, add a row, again taken from the code.

## Energies, broadening, occupation

| Symbol | Code | Meaning | Home |
|---|---|---|---|
| E | `energy` | Real energy at which a quantity is evaluated, same unit as `H` (eV when H comes from ASE/GPAW). | `BaseGreenFunction` |
| η | `eta` | Small positive imaginary shift. Gives every level a Lorentzian width η, makes G retarded (causal). Too large blurs features; too small gives spiky DOS on a finite energy grid. | `BaseGreenFunction` |
| z | `z(energy)` | Complex energy z = E + iη. In leads with a bias, z = E − V + iη (`SanchoRubio.z`). | `BaseGreenFunction.z` |
| μ | `mu` | Chemical potential of a reservoir. | `BaseGreenFunction` |
| k_B T | `kt` | Thermal energy, same unit as `H`. `kt=0` is a sharp step. | `BaseGreenFunction` |
| f(E) | `fermi(energy)` | Fermi–Dirac occupation f(E − μ) at temperature `kt`. | `BaseGreenFunction.fermi` |
| V | `bias` | Energy shift of a lead's bands. Check the sign convention against `SanchoRubio.z` and `SanchoRubio.fermi` before documenting; report any inconsistency. | `SanchoRubio` |

## Basis and Hamiltonian

| Symbol | Code | Meaning | Home |
|---|---|---|---|
| H | `H`, `h_ii` | Hamiltonian matrix in a localized, generally non-orthogonal orbital basis. | `BaseGreenFunction` |
| S | `S`, `s_ii` | Overlap matrix S_μν = ⟨φ_μ\|φ_ν⟩. Identity for an orthogonal basis. | `BaseGreenFunction` |
| H_ij, S_ij | `hs_ij`, `h_ij`, `s_ij` | Coupling and overlap between neighbouring blocks (principal layers, or lead and device). | `BaseSelfEnergy` |
| n_orb | — | Number of orbitals in the region a matrix lives on. Use it in shape strings. | — |

## Green's functions

| Symbol | Code | Definition | Home |
|---|---|---|---|
| G^r | `retarded` | G^r(E) = [zS − H − Σ(E)]^{-1}, Σ the sum of all attached self-energies. | `GreenFunction.retarded` |
| G^a | `advanced` | G^a = (G^r)^†. | `BaseGreenFunction.advanced` |
| G^< | `lesser` | Equilibrium: G^< = −f(E)(G^r − G^a). Out of equilibrium: G^< = G^r Σ^< G^a. | `GreenFunction.lesser` |
| G^> | `greater` | Equilibrium: G^> = (1 − f)(G^r − G^a). Otherwise G^> = G^< + G^r − G^a. | `GreenFunction.greater` |
| g_s | `SanchoRubio.retarded` | Surface Green's function of a semi-infinite lead, from decimation. | `SanchoRubio` |

## Self-energies and couplings

| Symbol | Code | Definition | Home |
|---|---|---|---|
| Σ | `retarded` on self-energy objects | Effect of an eliminated region (lead, environment) folded onto the region it couples to. For a lead: Σ = τ_ji g τ_ij with τ_ij = z S_ij − H_ij. | `BaseSelfEnergy.retarded` |
| Γ | `get_lambda` | Broadening matrix Γ = i(Σ − Σ^†). ħ/Γ sets the time an electron stays before escaping into that lead. Called "lambda" in the code. | `BaseSelfEnergy.get_lambda` |
| Δ(z) | `Hybridization.retarded` | Hybridization of a subspace with the rest: Δ(z) = g_0^{-1}(z) − G^{-1}(z) (code comment). | `Hybridization` |

## Observables

| Symbol | Code | Definition | Home |
|---|---|---|---|
| D(E) | `get_dos` | D(E) = −(1/π) Im Tr[G^r(E) S]. | `BaseGreenFunction.get_dos` |
| D_i(E) | `get_pdos` | −(1/π) Im[(S G^r S)_ii] / S_ii, density of states projected on the normalized orbital φ_i. | `BaseGreenFunction.get_pdos` |
| T(E) | `get_transmission` | Elastic: T(E) = Tr[Γ_L G^r Γ_R G^a]. Optional vertex corrections (`ferretti`, `brazilian`) add an inelastic term; see the code and Ferretti et al., Phys. Rev. B 72, 125114 (2005) (URL in the code comment). | `GreenFunction.get_transmission` |

| n_i | `get_ao_charge` | Electrons in orbital i, both spins (a filled orbital gives 2), Mulliken-type (ρS)_ii. | `qtpyt.continued_fraction.get_ao_charge` |
| A_α | `get_spectrals` | Partial spectral function A_α = G^r Γ_α G^a of lead α. | `GreenFunction.get_spectrals` |

## Many-body (screening, elph)

| Symbol | Code | Definition | Home |
|---|---|---|---|
| D | `get_density_matrix` | One-spin density matrix D = −(i/2π) ∫ G^<(E) dE. | `DistGreenFunction.get_density_matrix` |
| V | `V_qq` | Bare Coulomb matrix in the pair-orbital basis, V_{ij,kl}. | `qtpyt.screening` (module) |
| U_pq | `U_pq` | Rotation from all orbital pairs to optimized pair orbitals. | `qtpyt.screening` (module) |
| P | `Chi` | Polarization, spin-summed bubble P^≶_{ij,kl}(t) = 2i G^≶_{ik}(t) [G^≷_{jl}(t)]*. | `Chi` |
| W | `WRPA` | RPA screened interaction W^r = [1 − V P^r]^{-1} V, W^> = W^r P^> W^a. | `WRPA` |
| Σ^GW | `GW` | Σ^≶_{ij}(t) = i Σ_kl G^≶_{kl}(t) W^≶_{ik,jl}(t). | `GW` |
| U | `cRPA` | Partially screened interaction of a correlated subspace (constrained RPA). | `cRPA` |
| M_λ, ω_λ, n_λ | `g_lii`, `w_l`, `nph_l` | Electron–phonon coupling, phonon energy (eV), Bose occupation of mode λ. | `ElphFockSelfEnergy` |

Conductance in the Landauer picture is G = G_0 T(E_F) with G_0 = 2e²/h for spin-degenerate channels. Only state this where the code works with conductance, and say whether spin degeneracy is assumed.
