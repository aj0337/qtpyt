#!/usr/bin/env python3
"""Lint qtpyt docstrings against the physics-docstrings skill rules.

Usage: python check_docstrings.py FILE [FILE ...]

Reports, per public function/method/class:
  - banned phrases in the docstring
  - development-history wording ("previously", "now returns", "was changed", ...)
  - missing docstring (unless it overrides a parent method; check by hand)
  - parameters without type hints, missing return hint
  - parameters missing from the ``Parameters`` section
  - backslashes in a docstring that is not a raw string
Exit status 1 if anything was reported.
"""
import ast
import re
import sys

BANNED = [
    r"delve", r"seamless(ly)?", r"it'?s worth noting", r"it is worth noting",
    r"note that", r"it is important to", r"crucial", r"robust", r"powerful",
    r"leverag(e|es|ed|ing)", r"utili[sz](e|es|ed|ing)", r"facilitat(e|es|ed|ing)",
    r"comprehensive", r"in essence", r"essentially", r"simply", r"basically",
    r"under the hood", r"a wide range of", r"cutting[- ]edge", r"state[- ]of[- ]the[- ]art",
    r"streamlin(e|es|ed|ing)", r"elegant(ly)?", r"plays? a key role", r"in summary",
    r"to summari[sz]e", r"in short", r"in conclusion",
]
BANNED_RE = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.IGNORECASE)
OVERALL_RE = re.compile(r"(^|[.!?]\s+)Overall,", re.MULTILINE)
# Development-history wording. "no longer" is left out: it is common physics prose.
HISTORY_RE = re.compile(
    r"\b(previously|formerly|was (changed|updated|modified|replaced|renamed|removed|added|fixed)"
    r"|has been (changed|updated|modified|replaced|renamed|removed|added|fixed)"
    r"|now (uses|returns|supports|accepts|computes|sets|includes|gives|raises)"
    r"|renamed from|old (implementation|version|behaviou?r)|as requested|per (the )?request"
    r"|new(ly)? added|fixe[sd] (a|the) bug|bug ?fix|refactor(ed|ing)?)\b",
    re.IGNORECASE,
)


def documented_params(doc):
    """Names listed in the numpydoc Parameters section."""
    lines = doc.splitlines()
    names, inside = set(), False
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.fullmatch(r"\s*-{3,}\s*", nxt):  # a section header
            inside = line.strip() in ("Parameters", "Other Parameters")
            continue
        m = re.match(r"(\s*)(\*{0,2}[A-Za-z_]\w*)(\s*,\s*\*{0,2}[A-Za-z_]\w*)*\s*(:|$)", line)
        if inside and m and not re.fullmatch(r"\s*-{3,}\s*", line):
            for name in re.split(r"\s*,\s*", line.split(":")[0].strip()):
                names.add(name.lstrip("*"))
    return names


def check_node(node, src, path, report, in_class):
    name = node.name
    if name.startswith("_") and name != "__init__":
        return
    if any(isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter") for d in getattr(node, "decorator_list", [])):
        return  # documented on the property getter
    doc = ast.get_docstring(node, clean=True)
    where = f"{path}:{node.lineno} {name}"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        params = args.posonlyargs + args.args + args.kwonlyargs
        if in_class and params and params[0].arg in ("self", "cls"):
            params = params[1:]
        params += [a for a in (args.vararg, args.kwarg) if a]
        for a in params:
            if a.annotation is None:
                report(where, f"parameter '{a.arg}' has no type hint")
        if node.returns is None:
            report(where, "no return type hint")
        if name == "__init__":
            return  # parameters are documented in the class docstring
        if doc is not None:
            missing = {a.arg for a in params} - documented_params(doc)
            for m in sorted(missing):
                report(where, f"parameter '{m}' not in Parameters section")

    if doc is None:
        if not isinstance(node, ast.ClassDef) and in_class:
            report(where, "no docstring (fine only if it overrides an unchanged parent method)")
        else:
            report(where, "no docstring")
        return

    for m in BANNED_RE.finditer(doc):
        report(where, f"banned phrase '{m.group(0)}'")
    if OVERALL_RE.search(doc):
        report(where, "banned sentence opener 'Overall,'")
    for m in HISTORY_RE.finditer(doc):
        report(where, f"development history '{m.group(0)}': describe current behaviour only")

    raw = ast.get_source_segment(src, node.body[0].value) or ""
    # Check the source, not the parsed text: "\v" in "\varepsilon" is a valid escape and vanishes from it.
    if "\\" in raw and not raw.lstrip()[:1] in ("r", "R"):
        report(where, "docstring contains backslashes but is not a raw string (use r\"\"\")")

    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is not None:
            a = init.args
            params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs][1:]
            for m in sorted(set(params) - documented_params(doc)):
                report(where, f"__init__ parameter '{m}' not in class Parameters section")


def check_file(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    problems = []
    report = lambda where, msg: problems.append(f"{where}: {msg}")

    mod_doc = ast.get_docstring(tree)
    if mod_doc is None:
        report(f"{path}:1 <module>", "no module docstring")
    elif BANNED_RE.search(mod_doc):
        report(f"{path}:1 <module>", f"banned phrase '{BANNED_RE.search(mod_doc).group(0)}'")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_node(node, src, path, report, in_class=False)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue  # private class
            check_node(node, src, path, report, in_class=False)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check_node(sub, src, path, report, in_class=True)
    return problems


def _self_test():
    doc = "Summary.\n\nParameters\n----------\nH, S : ndarray\n    Matrices.\n*args : tuple\n    Extra.\n\nReturns\n-------\nx : float\n"
    assert documented_params(doc) == {"H", "S", "args"}, documented_params(doc)
    assert BANNED_RE.search("We leverage G.") and not BANNED_RE.search("just above E_F, overall phase")
    assert OVERALL_RE.search("Fine. Overall, good.") and not OVERALL_RE.search("an overall phase")
    assert HISTORY_RE.search("This now returns G^r.") and HISTORY_RE.search("Previously it used S.")
    assert not HISTORY_RE.search("Current is no longer conserved; the term is set to zero.")


if __name__ == "__main__":
    _self_test()
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    found = [p for f in sys.argv[1:] for p in check_file(f)]
    print("\n".join(found) if found else "No problems found.")
    sys.exit(1 if found else 0)
