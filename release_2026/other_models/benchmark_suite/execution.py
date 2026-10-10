"""CRUXEval checks with literal answers and a disposable, time-limited worker.

The worker is a reliability guard, not an OS security sandbox. Predictions cannot
contain executable calls other than f(literal arguments); output uses literal_eval.
"""

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def expression(text, mode):
    text = text.strip()
    if text.startswith("[ANSWER]"):
        text = text.removeprefix("[ANSWER]").split("[/ANSWER]", 1)[0].strip()
    text = text.strip("`").removeprefix("python\n").strip()
    try:
        tree = ast.parse(text)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
            compare = tree.body[0].test
            if (
                not isinstance(compare, ast.Compare)
                or len(compare.ops) != 1
                or not isinstance(compare.ops[0], ast.Eq)
            ):
                return None
            node = compare.left if mode == "input" else compare.comparators[0]
        else:
            node = ast.parse(text, mode="eval").body
        if mode == "output":
            return ("literal", ast.literal_eval(node))
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "f"
        ):
            return None
        args = [ast.literal_eval(a) for a in node.args]
        kwargs = {
            k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg is not None
        }
        if len(kwargs) != len(node.keywords):
            return None
        return ("call", args, kwargs)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def crux_success(row, text, mode, timeout=3):
    answer = expression(text, mode)
    if answer is None:
        return False
    try:
        expected = ast.literal_eval(row["output"])
    except (SyntaxError, ValueError, TypeError):
        raise ValueError("Invalid CRUXEval reference output")
    if mode == "output":
        return expected == answer[1]
    payload = dict(
        code=row["code"],
        input_text=ast.unparse(
            ast.Call(
                func=ast.Name(id="f", ctx=ast.Load()),
                args=[
                    ast.Constant(value=x)
                    if isinstance(x, (str, int, float, bool, type(None)))
                    else ast.parse(repr(x), mode="eval").body
                    for x in answer[1]
                ],
                keywords=[
                    ast.keyword(arg=k, value=ast.parse(repr(v), mode="eval").body)
                    for k, v in answer[2].items()
                ],
            )
        ),
        output=row["output"],
    )
    # Run no model code in the evaluator process. -I ignores PYTHONPATH/site customizations.
    with tempfile.TemporaryDirectory(prefix="cruxeval-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).with_name("worker.py"))],
                input=json.dumps(payload),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                timeout=timeout,
            )
            return proc.returncode == 0 and proc.stdout.strip() == "passed"
        except subprocess.TimeoutExpired:
            return False
