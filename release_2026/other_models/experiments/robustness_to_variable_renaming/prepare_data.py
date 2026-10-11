"""Use the framework's complete-source preparation, optionally apply a treatment."""

import argparse
from pathlib import Path

from ...cli import main as framework_cli
from ...models import resolve_dataset
from ...runtime import write_json
from .data_transformations.renaming import transform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--arch")
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-cache")
    parser.add_argument("--reference-cache")
    parser.add_argument("--n-docs", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--transform", choices=["plain", "halfvar", "highdiv"])
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args(argv)
    name = resolve_dataset(args.arch, args.dataset)
    command = ["prepare", "--dataset", name, "--run", args.run, "--n-docs", str(args.n_docs)]
    for option, value in (("--cache", args.source_cache), ("--reference-cache", args.reference_cache), ("--max-files", args.max_files)):
        if value is not None:
            command.extend([option, str(value)])
    framework_cli(command)
    run = Path(args.run)
    source = run / "data" / f"{name}.jsonl"
    if args.transform:
        result = transform(source, run / "data/transformed.jsonl", args.transform, args.seed)
        write_json(run / "results/transformation.json", result)
        return result
    return source
