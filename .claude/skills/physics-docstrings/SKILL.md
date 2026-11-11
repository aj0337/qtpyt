---
name: physics-docstrings
description: Write or revise NumPy-style docstrings and type hints for qtpyt (quantum transport / NEGF code) so the Sphinx-built ReadTheDocs API reads like physics documentation for physicists, not like software documentation. Use this whenever the user asks to document, add docstrings, add type hints, annotate, explain, or clean up the docs of any qtpyt module, class, function or method, when preparing code for the Sphinx/ReadTheDocs API pages, or when reviewing existing docstrings, even if they only say "document this file" or "add types here".
---

# Physics-first docstrings for qtpyt

The readers are physicists: experimentalists comparing to measured conductance, theorists who know Green's functions but not this code, and students. They open the API page to find out *which physical quantity a call gives them, under which approximation, in which units and conventions*. They do not care how memory is managed unless it changes what they can compute. Every docstring is written for that reader.

The rendered output is Sphinx (autodoc + napoleon, NumPy style) with MathJax, hosted on ReadTheDocs. Equations will be typeset, and cross-references become links, so use both freely.

## Workflow

1. **Read before writing.** Read the target object, its parent classes, the functions it calls and the docstrings of nearby objects. You need the implementation to know what is actually computed, and the neighbours to know what is already explained elsewhere (see "Say it once").
2. **Find the equation the code evaluates.** Write it down from the code itself, signs and factors included (e.g. `get_lambda` gives `1j * (Sigma - Sigma^†)`, so Γ = i(Σ − Σ†)). Compare to the textbook form you expect. If they differ in sign, factor, shift or ordering, document what the code does and report the discrepancy to the user at the end; do not silently write the textbook version, and do not change the code.
3. **Check the notation file.** Read `references/notation.md` for the symbols and conventions already fixed for qtpyt. Use them. If you introduce a new symbol that other objects will reuse, add it there with its owning object.
4. **Add type hints** to the signature (rules below). Never change behaviour while doing this: no new defaults, no fixing mutable default arguments, no renaming. Mention such issues to the user instead.
5. **Write the docstring** using the structure below.
6. **Run the checker**: `python .claude/skills/physics-docstrings/scripts/check_docstrings.py <files>`. It flags banned phrases, missing hints, undocumented parameters and non-raw docstrings that contain backslashes. Fix what it reports; it is a floor, not the whole review.
7. **Report** briefly: what you documented, every point where code and physics expectation disagree, every unit or reference you could not confirm.

## Docstring structure (NumPy style)

Use only the sections that carry information for this object, in numpydoc order:

```
Summary line
Extended summary
Parameters
Returns / Yields
Raises
See Also
Notes
References
Examples
```

Classes additionally take `Attributes` after `Parameters`. Document `__init__` arguments in the class docstring, not in `__init__`.

**Summary line.** One line, names the physical quantity or action: "Retarded Green's function of the scattering region." or "Surface Green's function of a semi-infinite lead by Sancho–Rubio decimation." Not "This method returns…", not the function name restated.

**Extended summary.** The intuitive physics, two to six sentences of prose. What the quantity means for an electron in the device, what you would see if you plotted it, when a user reaches for it. Intuition, not derivation. An analogy is fine when it is accurate and short.

**Parameters / Returns.** `name : type-and-shape` then a description of the *physical* meaning, units and allowed range. Give shapes in terms of physical counts: `(n_orb, n_orb)` where `n_orb` is the number of basis orbitals in the region, `(n_E,)` for an energy grid. State units as "same energy unit as `H` (eV when built from ASE/GPAW)". State default values only when the default carries physics (`eta=1e-5` is a broadening, say what it does).

**Notes.** Where the equations live. Defining equation as a `.. math::` block, the approximation made, sign conventions, limits (η → 0, T → 0, equilibrium vs. bias), and the few numerical facts a physicist needs (convergence criterion, which energy the result is stored for). Keep derivations short and point to references for the rest.

**References.** numpydoc `.. [1]` entries. Cite only references that appear in the code or comments, or that you are certain of (authors, journal, volume, page, year). If unsure, write the reference you think it is in your report to the user and leave it out of the docstring.

**Examples.** Optional. Only include an example you have run; a minimal tight-binding chain is ideal. A wrong example in published docs is worse than none.

Use raw strings (`r"""..."""`) for any docstring containing LaTeX backslashes, otherwise `\t`, `\n`, `\b` in `\theta`, `\nu`, `\beta` get mangled.

## Physics over computer science

Translate implementation facts into physics, or leave them out.

| Instead of | Write |
|---|---|
| "Caches the result" | "The result is kept for the last energy, so repeated calls at the same E cost nothing." (only if it matters to the user) |
| "Instantiates a GreenFunction for the lead" | "Builds the Green's function of one principal layer of the lead." |
| "Wrapper around `xp.linalg.inv`" | (omit; say what is inverted: "Inverts zS − H − Σ(E).") |
| "Returns a complex ndarray" | "G^r(E) in the orbital basis of the region." |
| "Iterates until convergence" | "Each decimation step doubles the effective length of the lead; stops once the residual inter-layer coupling falls below `conv`." |
| "`kt` : float, temperature parameter" | "`kt` : float. Thermal energy k_B T of the reservoir, in the energy unit of `H`. `kt=0` gives a sharp Fermi step." |

CS detail belongs in a docstring only when it changes what a physicist can do: memory or time that grows steeply with system size, GPU vs CPU arrays, MPI distribution. Say it in physical terms ("memory grows as the square of the number of orbitals in the device").

Name approximations explicitly: non-interacting, equilibrium, wide-band, nearest-neighbour principal layers, Γ-point only, spin-degenerate. A physicist needs to know what a result is *not* valid for.

## Say it once

The API reference is read as a website, and repeated paragraphs make it long and hard to scan. Each concept has one home and everything else links to it.

- Explain a concept at the most general object that uses it (η and `kt` in `BaseGreenFunction`, Γ in `BaseSelfEnergy.get_lambda`). Elsewhere write one clause and a link: "Broadening, see :class:`~qtpyt.base.greenfunction.BaseGreenFunction`."
- Large explanations (what NEGF is, how leads attach) belong on a narrative page of the docs site; link with ``:doc:`theory/negf` `` once that page exists, or ask the user where it should live.
- An override that behaves like its parent needs no docstring (autodoc inherits it). If it differs, its docstring says only what differs.
- Do not restate the class story in each method, and do not repeat the summary line in `Returns`.
- Use `See Also` for genuinely related objects, not as a list of everything in the module.

Cross-reference syntax: ``:class:`~qtpyt.base.selfenergy.LeadSelfEnergy` ``, ``:meth:`retarded` ``, ``:func:`numpy.linalg.inv` ``. The `~` shows only the short name.

## Describe the code as it is now

Docstrings get written during active development, but the reader (a user or a developer new to the code) has never seen earlier versions and was not part of the conversation that produced the change. A docstring describes the current behaviour as if it had always been that way. Nothing in it should only make sense to someone who knows the history.

- No history: not "previously this returned…", "now uses…", "was changed to…", "no longer computes…", "the old implementation", "fixed the sign of…", "renamed from…".
- No traces of the request or conversation: not "as requested", "the new option", "the corrected formula", "added to support…".
- When the code changes, rewrite the affected docstring so it describes the new behaviour on its own. Do not add a sentence about the change to the old text.
- If a behaviour is easy to get wrong, state it as a current fact: "The inelastic term is zero unless `ferretti` or `brazilian` is set." Not: "The inelastic term is now set to zero when no correction is requested."
- A deprecation is current information, because it tells the reader what to use today. Write it with Sphinx's `.. deprecated::` directive, naming the replacement, and leave out the reason for the change.

History goes in commit messages and the changelog. If you think the user should hear about a behaviour change, tell them in your report, not in the docstring.

## Type hints

- Add `from __future__ import annotations` at the top of an edited module so hints are not evaluated at import (the package still claims Python ≥ 3.7).
- Use `typing.Optional`, `Union`, `Sequence`, `Tuple` rather than `X | None` and `tuple[...]`, for the same reason.
- Arrays: `numpy.typing.NDArray[np.complex128]` / `NDArray[np.float64]` for arrays the function returns or stores; `numpy.typing.ArrayLike` for inputs that are converted with `xp.asarray`. When CuPy is installed `xp` is CuPy and the arrays are `cupy.ndarray`; state this once at the module or package level, not in every docstring.
- Energies: `float` when a scalar is required. If a function also accepts an array of energies, hint `Union[float, NDArray[np.float64]]` and say so.
- `(H, S)` pairs: `Tuple[ArrayLike, ArrayLike]`.
- Self-energy collections: `Sequence[Tuple[Any, BaseSelfEnergy]]` where the first element is the orbital index selector; describe it physically in the docstring ("orbitals of the region the lead couples to").
- `__init__` returns `-> None`. Keep hints exactly as specific as the code requires; do not narrow a type the code accepts more broadly.

## Writing style

Write like a careful physicist writing a methods section: precise, plain, specific. Each sentence either defines something, states a condition, or tells the reader what to expect. Delete any sentence that does none of these.

Banned words and phrases (the checker greps for them): "delve", "seamlessly", "it's worth noting", "it is worth noting", "note that", "it is important to", "crucial", "robust", "powerful", "leverage", "utilize", "facilitate", "comprehensive", "in essence", "essentially", "simply", "basically", "under the hood", "a wide range of", "cutting-edge", "state-of-the-art", "streamline", "elegant", "plays a key role", "in summary", "to summarize", "in short", "in conclusion", and "Overall," opening a sentence. ("Just above the Fermi level" and "overall phase" are fine; those are physics.)

Banned habits:
- **Lists of three by reflex.** Use the number of items the content has. Most explanations read better as prose; use a list only for genuinely parallel items such as allowed values of an option.
- **A closing summary line** at the end of a section or docstring. The last sentence should add information, not restate the previous ones.
- Restating the summary line in the extended summary.
- Marketing adjectives ("efficient", "fast", "flexible") without a number or a comparison behind them.
- Hedging filler ("can be used to", "is designed to", "allows the user to"). Say what it does.
- Emojis, bold for emphasis, exclamation marks.

## Worked examples

See `references/examples.md` for before/after docstrings from this codebase: a class with physics parameters, a method whose physics needs care (PDOS in a non-orthogonal basis), and an override that defers to its parent. Read it the first time you use this skill in a session.

## Sphinx settings these docstrings assume

See `references/sphinx.md` when the user is setting up or changing `docs/conf.py`, or when cross-references or math do not render.
