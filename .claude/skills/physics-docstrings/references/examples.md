# Worked examples from qtpyt

These are written against the code as it stands. Copy the approach, not the wording.

## 1. A class that owns several shared concepts

`BaseGreenFunction` is the home for η, μ, `kt`, H and S (see `notation.md`). Its docstring explains them fully so subclasses can link instead of repeating.

Before:

```python
class BaseGreenFunction:
    """Base class for the Green's function."""

    def __init__(self, H, S, eta=1e-5, mu=0.0, kt=0.0) -> None:
```

After:

```python
class BaseGreenFunction:
    r"""Single-particle Green's function of a region described by a Hamiltonian and an overlap.

    The Green's function answers the question: if an electron of energy
    :math:`E` is injected into orbital :math:`\nu`, what amplitude
    appears in orbital :math:`\mu`? Its poles sit at the energy levels
    of the region and its imaginary part gives the density of states.
    Subclasses decide how the inverse :math:`G^{-1}(E)` is built, for
    example by adding lead self-energies or by surface decimation.

    Parameters
    ----------
    H : array_like, shape (n_orb, n_orb)
        Hamiltonian in a localized orbital basis, in any energy unit
        (eV when built from ASE/GPAW). All energies passed to this
        object use the same unit.
    S : array_like, shape (n_orb, n_orb)
        Overlap matrix :math:`S_{\mu\nu} = \langle\phi_\mu|\phi_\nu\rangle`.
        Pass the identity for an orthogonal basis.
    eta : float, default 1e-5
        Imaginary shift :math:`\eta` in :math:`z = E + i\eta`. Each level
        is broadened into a Lorentzian of half-width :math:`\eta`. Choose
        it smaller than the features you want to resolve but larger than
        the spacing of your energy grid, or the density of states turns
        into isolated spikes.
    mu : float, default 0.0
        Chemical potential of the reservoir that fixes the occupation.
    kt : float, default 0.0
        Thermal energy :math:`k_B T`. Zero gives a sharp Fermi step.

    Attributes
    ----------
    energy : float or None
        Energy of the most recent evaluation; results for this energy
        are reused on the next call.

    Notes
    -----
    In the non-orthogonal basis the retarded Green's function is

    .. math::

        G^r(E) = \left[(E + i\eta) S - H - \Sigma(E)\right]^{-1},

    where :math:`\Sigma` is zero here and supplied by subclasses. The
    advanced function is :math:`G^a = (G^r)^\dagger`.
    """

    def __init__(
        self,
        H: ArrayLike,
        S: ArrayLike,
        eta: float = 1e-5,
        mu: float = 0.0,
        kt: float = 0.0,
    ) -> None:
```

What changed: the summary names the physics; the extended summary gives an intuitive picture in four sentences; each parameter says what it means physically and how to choose it; the defining equation sits in Notes. The words "base class" disappear because a physicist reading the page does not need them (Sphinx already shows the inheritance).

## 2. A method where the physics needs care

The PDOS in a non-orthogonal basis is a place where users get confused, because the projected densities do not add up to the total. The docstring is the place to say so.

Before:

```python
    def get_pdos(self, energy):
        """Projected density of states."""
```

After:

```python
    def get_pdos(self, energy: float) -> NDArray[np.float64]:
        r"""Density of states projected onto each basis orbital.

        Gives, for every orbital :math:`\phi_i`, how many states per unit
        energy at :math:`E` have weight on that orbital. Plot it to see
        which atoms or orbital characters a transmission peak comes from.

        Parameters
        ----------
        energy : float
            Energy :math:`E`, same unit as :attr:`H`.

        Returns
        -------
        pdos : ndarray, shape (n_orb,)
            :math:`D_i(E)` in states per energy unit, one value per orbital.

        See Also
        --------
        get_dos : Total density of states.

        Notes
        -----
        The projection uses the normalized orbital itself,

        .. math::

            D_i(E) = -\frac{1}{\pi}\,
            \frac{\operatorname{Im}\,(S\,G^r(E)\,S)_{ii}}{S_{ii}}.

        In a non-orthogonal basis the orbitals overlap, so
        :math:`\sum_i D_i(E)` is not equal to :meth:`get_dos`. The sum
        double-counts weight that lives in the overlap between orbitals.
        For an orthogonal basis (:math:`S = 1`) the two agree.
        """
```

What changed: the formula is read from the code (`dotdiag(S, G S) / S.diagonal()`), not assumed. The Notes warn about the one thing a physicist would get wrong. `See Also` names one related method, not every method in the class.

## 3. An override that differs from its parent in one way

`SanchoRubio.retarded` does what `BaseGreenFunction.retarded` describes, only the inverse comes from decimation. Its docstring says only that, and the class docstring carries the algorithm.

```python
    def retarded(self, energy: float) -> NDArray[np.complex128]:
        """Surface Green's function :math:`g_s(E)` of the semi-infinite lead.

        Inverts the decimated :meth:`get_Ginv`. See :class:`SanchoRubio`
        for the method and its convergence criterion.
        """
```

If an override behaves exactly like its parent, write no docstring at all; autodoc shows the parent's.

## 4. Style failures to avoid

```python
    """Seamlessly computes the transmission.

    This powerful method leverages the Green's function to compute the
    transmission. It handles elastic transport, inelastic corrections,
    and multiple leads. It's worth noting that the Ferretti correction
    is used by default.

    In summary, this method gives the transmission.
    """
```

Problems: banned words ("seamlessly", "powerful", "leverages", "it's worth noting"); a reflexive three-item list; a closing summary sentence that adds nothing; no equation, no units, no statement of what the correction approximates. The fact buried in the middle (Ferretti is on by default, so the default result is *not* plain Landauer transmission when extra self-energies are attached) is the most important sentence and should come first.
