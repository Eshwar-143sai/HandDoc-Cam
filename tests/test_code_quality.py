#!/usr/bin/env python3
"""Keeps the code base readable for reviewers: every module, class and function has a docstring, every parameter and
return value has a type hint, Config's typed attributes match its settings table, and (if mypy is installed) the
type checker is clean.   Run:  python tests/test_code_quality.py"""
import ast, sys, subprocess, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODULES = ["handdoc_cam.py", "config.py", "perf.py"]
fails = 0
def check(label, ok, extra=""):
    global fails
    fails += (not ok); print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  -> ' + str(extra)) if extra else ''}")

print("docstrings and type hints:")
for name in MODULES:
    tree = ast.parse((ROOT / name).read_text()); problems, n_funcs = [], 0
    if ast.get_docstring(tree) is None and name != "handdoc_cam.py":  # the main script's docstring sits under the shebang
        problems.append("module docstring")
    if name == "handdoc_cam.py" and ast.get_docstring(tree) is None:
        problems.append("module docstring")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and ast.get_docstring(node) is None:
            problems.append(f"class {node.name}: docstring")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_funcs += 1
            if node.name.startswith("__") and node.name not in ("__init__", "__call__"):
                continue
            a = node.args
            params = [x for x in a.posonlyargs + a.args + a.kwonlyargs if x.arg not in ("self", "cls")]
            if a.vararg: params.append(a.vararg)
            if a.kwarg: params.append(a.kwarg)
            missing = [p.arg for p in params if p.annotation is None]
            if missing: problems.append(f"{node.name}(): no type hint for {missing}")
            if node.returns is None: problems.append(f"{node.name}(): no return type")
            if ast.get_docstring(node) is None: problems.append(f"{node.name}(): no docstring")
    check(f"{name}: {n_funcs} functions/methods fully documented and typed", not problems, "; ".join(problems[:4]))

print("\nconfig typing:")
import config
ann = config.Config.__annotations__
want = {p.name: ("int" if p.kind == "int" else "float") for p in config.PARAMS}
check("every setting has a typed attribute on Config (autocomplete / mypy)", {k: ann.get(k) for k in want} == want,
      [k for k in want if ann.get(k) != want[k]][:5])
check("no stale attributes on Config", set(ann) == set(want), sorted(set(ann) - set(want)))
check("every setting has a help text and a sane range", all(p.help and p.lo < p.hi and p.lo <= p.default <= p.hi for p in config.PARAMS))
check("setting names are unique", len({p.name for p in config.PARAMS}) == len(config.PARAMS))

print("\nstatic analysis:")
if importlib.util.find_spec("mypy"):
    r = subprocess.run([sys.executable, "-m", "mypy", "--ignore-missing-imports", "--python-version", "3.10", *MODULES],
                       capture_output=True, text=True, cwd=str(ROOT))
    check("mypy: no issues", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])
else:
    print("  skip  mypy not installed (pip install mypy)")
if importlib.util.find_spec("pyflakes"):
    r = subprocess.run([sys.executable, "-m", "pyflakes", *MODULES], capture_output=True, text=True, cwd=str(ROOT))
    check("pyflakes: no unused imports / undefined names", r.returncode == 0, r.stdout.strip()[:100])
else:
    print("  skip  pyflakes not installed (pip install pyflakes)")

print("\nALL CODE-QUALITY CHECKS PASSED" if not fails else f"\n{fails} FAILURE(S)"); sys.exit(1 if fails else 0)
