"""Process setup and access to the original scoring library in lib/.

The library files are the ones that produced the paper's numbers and are checked
against lib/SHA256SUMS. They import each other by bare name, so lib/ must be on
sys.path. Renamed views are seeded with string hashes, so Python must start with
PYTHONHASHSEED=0.
"""

import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib"
# release_2026/, so that `other_models` imports as a package.
RELEASE = HERE.parents[2]
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))


def setup():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Start Python with PYTHONHASHSEED=0")


def install_fcntl():
    """decmech.py imports fcntl; on Windows, provide flock with msvcrt locking."""
    if os.name != "nt" or "fcntl" in sys.modules:
        return
    import msvcrt
    import types

    module = types.ModuleType("fcntl")
    module.LOCK_EX = 2
    module.LOCK_UN = 8

    def flock(fd, operation):
        if hasattr(fd, "fileno"):
            fd = fd.fileno()
        position = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if operation == module.LOCK_EX:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            elif operation == module.LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                raise ValueError("Only exclusive lock and unlock are supported")
        finally:
            os.lseek(fd, position, os.SEEK_SET)

    module.flock = flock
    sys.modules["fcntl"] = module


def lib_module(name):
    import warnings

    # Parsing CodeSearchNet functions raises many SyntaxWarnings for regex strings.
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    install_fcntl()
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != LIB.resolve():
        raise RuntimeError(
            f"{name} was imported from {module.__file__}, not from {LIB}"
        )
    return module


def verify_lib():
    """Compare lib/ with SHA256SUMS, ignoring CRLF line endings from a Windows checkout."""
    import hashlib

    for line in (LIB / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        data = (LIB / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"lib/{name} differs from lib/SHA256SUMS")


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def json_safe_raw(raw):
    """Replace non-finite floats with null and return the paths that were replaced."""
    import math

    paths = []

    def visit(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            paths.append(path)
            return None
        if isinstance(value, dict):
            return {key: visit(item, path + "." + key) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    return visit(raw, "raw"), paths


def quiet_command(function):
    """Recompile a library command without its print calls.

    The library's progress lines format undefined retention with %.4f, which fails
    before the results are stored when a rate has an empty denominator. The scoring
    statements are unchanged. Empty paired masks (b0, bN) are given a boolean dtype
    so that `&` works when a tokenizer has no multi-piece targets.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))

    class Diagnostics(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "print"
            ):
                return ast.Pass()
            return self.generic_visit(node)

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if any(
                isinstance(t, ast.Name) and t.id in ("b0", "bN") for t in node.targets
            ):
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "np"
                    and call.func.attr == "array"
                    and not any(k.arg == "dtype" for k in call.keywords)
                ):
                    call.keywords.append(
                        ast.keyword(
                            arg="dtype", value=ast.Name(id="bool", ctx=ast.Load())
                        )
                    )
            return node

    tree = ast.fix_missing_locations(Diagnostics().visit(tree))
    namespace = {}
    exec(compile(tree, "<quiet-command>", "exec"), function.__globals__, namespace)
    return namespace[function.__name__]
