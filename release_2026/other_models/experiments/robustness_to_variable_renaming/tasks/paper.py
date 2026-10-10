"""Immutable paper metrics through the shared tokenizer/causal-LM interface.

Each task requires a fresh process: importing variable misuse changes a library global.
Custom adapters must implement tokenizer offsets and HF-compatible causal_lm forward.
"""

import argparse, json, os
from pathlib import Path
from types import SimpleNamespace
from ..registry import TASKS
from ....runtime import write_json


def json_safe_raw(raw):
    """Export undefined legacy diagnostics as null; retain all finite scoring numerics."""
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
    """Keep scoring arithmetic; remove diagnostics and type empty boolean masks.

    Legacy diagnostics format undefined retention with %.4f, raising before store().
    The scoring functions remain verbatim and null rates remain null in JSON.
    """
    import ast, inspect

    tree = ast.parse(inspect.getsource(function))

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
            # cmd_idcurve's paired correctness vectors are boolean. NumPy gives
            # [] a float dtype, which makes '&' fail when a tokenizer has no
            # multi-piece continuation targets. Nonempty values retain their
            # existing dtype and scoring; empty support now produces null rates.
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
                ):
                    if not any(k.arg == "dtype" for k in call.keywords):
                        call.keywords.append(
                            ast.keyword(
                                arg="dtype", value=ast.Name(id="bool", ctx=ast.Load())
                            )
                        )
            return node

    tree = ast.fix_missing_locations(Diagnostics().visit(tree))
    namespace = {}
    exec(
        compile(tree, "<paper-no-diagnostics>", "exec"), function.__globals__, namespace
    )
    return namespace[function.__name__]


def score_paper(model, task, run, n=0, max_len=320, batch=16, n_sample=0):
    if n < 0 or n_sample < 0:
        raise ValueError("paper limits must be nonnegative")
    from ..legacy import library
    import torch

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("start Python with PYTHONHASHSEED=0")
    DM = library("decmech", run)
    library("whyscale", run).CACHE = str(Path(run) / "eval_corpus")
    torch.set_num_threads(4)
    # The arithmetic and item construction are immutable. Only inject the model adapter.
    DM.load_lm = lambda path, device: (model.tokenizer, model.causal_lm)
    if task in ("identifier_reproduction", "variable_misuse_localization"):
        # Original source readers slice [:n], so translate all into the actual
        # prepared corpus size rather than forwarding an empty-slice sentinel.
        corpus = Path(run) / "eval_corpus/real6_corpus.json"
        available = len(json.loads(corpus.read_text())["recs"])
        n = min(n or available, available)
        if n == 0:
            raise ValueError("Empty prepared reference corpus")
    if task == "cloze_all_paper":
        cache = Path(run) / "eval_corpus" / f"cloze_all_items_n{n_sample}.json"
        if not cache.exists():
            raise ValueError(
                "Prepare the shared cloze data and experiment views before scoring"
            )
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached["n_sample"] != n_sample or cached["seed"] != DM.BENCH_SEED:
            raise ValueError("Prepared cloze cache configuration differs")
        if not cached["items"]:
            raise ValueError("No eligible prepared cloze items; use a larger subset")
    tmp = Path(run) / "tmp" / f"{task}.json"
    tmp.unlink(missing_ok=True)
    a = SimpleNamespace(
        model="tinylmframework_model",
        device=model.device,
        threads=4,
        limit=n,
        n=n,
        n_items=n,
        n_sample=n_sample,
        max_len=max_len,
        batch=batch,
        write_smoke=True,
        out=str(tmp),
        variant="all",
        per_item=False,
    )
    if task == "identifier_reproduction":
        quiet_command(DM.cmd_idcurve)(a)
        key = "idcurves"
    elif task == "variable_misuse_localization":
        quiet_command(DM.cmd_vmloc)(a)
        key = "varmisuse_locrepair"
    elif task == "cloze_all_paper":
        quiet_command(DM.cmd_cloze)(a)
        key = "cloze_all"
    else:
        raise ValueError(task)
    raw = json.loads(tmp.read_text())[key]["tinylmframework_model"]
    if task == "identifier_reproduction":
        m = raw["A_reproduction"]["N0_accuracy"]
    elif task == "variable_misuse_localization":
        m = raw["mean"]["N0_localization"]
    else:
        m = raw["N0_accuracy_safe"]
    clean, nonfinite = json_safe_raw(raw)
    result = dict(accuracy=m["rate"], correct=m["k"], count=m["n"], raw=clean)
    if nonfinite:
        result["nonfinite_raw_fields"] = nonfinite
    return result


for name in [
    "identifier_reproduction",
    "variable_misuse_localization",
    "cloze_all_paper",
]:
    TASKS.add(name, score_paper)


def main():
    from ....models import HFModel

    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--task")
    p.add_argument("--run")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--n-sample", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    model = HFModel(a.model, a.device, a.dtype, a.batch)
    result = score_paper(
        model, a.task, a.run, a.n, a.max_len, batch=a.batch, n_sample=a.n_sample
    )
    write_json(Path(a.run) / "results" / (a.task + ".json"), result)


if __name__ == "__main__":
    main()
