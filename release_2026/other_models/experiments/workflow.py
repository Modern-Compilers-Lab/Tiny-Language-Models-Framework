"""Standalone entry points for the reusable codelab experiment workflows."""

import argparse
import importlib
import json

from ..runtime import setup_run, write_json

UTILITIES = {
    "prepare": "prepare_benchmarks",
    "prepare-views": "llama_csn",
    "evaluate": "llama_csn",
    "score": "llama_csn",
    "prepare-training": "prepare_training",
    "prepare-data": "prepare_data",
    "token-cache": "token_cache",
    "diversity": "measure_diversity",
    "freeze": "freeze",
    "campaign": "campaign",
    "report": "report",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "transform", *UTILITIES])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "list":
        from .robustness_to_variable_renaming.registry import builtins, TASKS

        builtins()
        result = dict(
            experiments=["robustness_to_variable_renaming"],
            modes=["plain", "halfvar", "highdiv"],
            rename_caps=[0, 1, 2, 4, 8],
            tasks=sorted(TASKS),
            utilities=list(UTILITIES),
            default_limit=0,
        )
        print(json.dumps(result, indent=2))
        return result
    if args.action == "transform":
        sub = argparse.ArgumentParser(description="Prepare a controlled renaming treatment")
        sub.add_argument("--input", required=True)
        sub.add_argument("--run", required=True)
        sub.add_argument("--transform", "--mode", dest="mode", choices=["plain", "halfvar", "highdiv"], default="halfvar")
        sub.add_argument("--seed", type=int, default=2)
        options = sub.parse_args(args.arguments)
        from .robustness_to_variable_renaming.data_transformations.renaming import transform

        run = setup_run(options.run, vars(options))
        result = transform(options.input, run / "data/transformed.jsonl", options.mode, options.seed)
        write_json(run / "results/transformation.json", result)
        print(json.dumps(result, indent=2))
        return result
    module = importlib.import_module(".robustness_to_variable_renaming." + UTILITIES[args.action], __package__)
    forwarded = ([args.action] if args.action in ("prepare-views", "evaluate", "score") else []) + args.arguments
    return module.main(forwarded)
