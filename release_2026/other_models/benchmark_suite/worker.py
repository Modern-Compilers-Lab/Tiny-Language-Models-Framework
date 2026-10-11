"""Small cross-platform worker; only called by execution.crux_success."""

import ast
import builtins
import contextlib
import io
import json
import sys


def main():
    payload = json.loads(sys.stdin.read())
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    except ImportError:
        pass  # Windows: parent enforces wall time.
    allowed_modules = {
        "math",
        "re",
        "collections",
        "itertools",
        "functools",
        "string",
        "random",
        "statistics",
        "bisect",
        "heapq",
    }
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name not in allowed_modules:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    safe = {
        name: getattr(builtins, name)
        for name in (
            "abs",
            "all",
            "any",
            "bin",
            "bool",
            "bytes",
            "bytearray",
            "chr",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "hash",
            "hex",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "object",
            "oct",
            "ord",
            "pow",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "slice",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "zip",
            "Exception",
            "ValueError",
            "IndexError",
            "KeyError",
        )
    }
    safe["__import__"] = guarded_import
    tree = ast.parse(payload["code"])
    if any(
        isinstance(n, ast.Attribute)
        and n.attr.startswith("__")
        or isinstance(n, ast.Name)
        and n.id.startswith("__")
        for n in ast.walk(tree)
    ):
        return
    env = {"__builtins__": safe}
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exec(compile(tree, "<cruxeval>", "exec"), env)
            actual = eval(payload["input_text"], env)
            expected = ast.literal_eval(payload["output"])
            correct = expected == actual
        if correct:
            sys.stdout.write("passed")
    except BaseException:
        pass


if __name__ == "__main__":
    main()
