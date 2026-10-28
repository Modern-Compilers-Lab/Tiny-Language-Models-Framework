"""Rename local variables in a training JSONL to create a training condition.

    python transform.py --input TRAIN.jsonl --output OUT.jsonl --mode halfvar --seed 2

plain    unchanged code
halfvar  about half of the eligible local bindings of each document get fresh names
highdiv  every eligible local binding gets a fresh name

Fresh names are unique across the corpus (lib/corpus.py, FreshNames and
rename_document_halfvar, the treatment used for the paper). Only identifier spans
change; comments, strings, attributes and imports are kept. Row order, IDs and all
other fields are preserved. Train each output with the framework's
`train --data OUT.jsonl` command.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402

MODES = ("plain", "halfvar", "highdiv")


def transform(source, destination, mode="halfvar", seed=2):
    from other_models.runtime import digest, read_jsonl, write_json, write_jsonl

    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("The output must differ from the input")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(MODES)}")
    C = env.lib_module("corpus")
    fresh = C.FreshNames(key=seed)
    rng = random.Random(seed)
    stats = Counter()
    before, after = set(), set()

    def rows():
        for row in read_jsonl(source):
            if not isinstance(row.get("code"), str):
                raise ValueError("Every row needs a string 'code' field")
            code = out = row["code"]
            status, names = "plain", 0
            if mode != "plain":
                out, status, names, _sites = C.rename_document_halfvar(
                    code, fresh, rng.random, mix_frac=0.5 if mode == "halfvar" else 1.0
                )
            stats["rows"] += 1
            stats[status] += 1
            stats["changed"] += out != code
            stats["renamed_bindings"] += names
            # Distinct variable names over the first 2,000 rows, before and after.
            if stats["rows"] <= 2000:
                try:
                    before.update(C.extract_variables(code, "python"))
                    after.update(C.extract_variables(out, "python"))
                except Exception:
                    stats["diversity_parse_errors"] += 1
            yield dict(row, code=out)

    write_jsonl(destination, rows())
    report = dict(
        stats,
        seed=seed,
        mode=mode,
        source_sha256=digest(source),
        output_sha256=digest(destination),
        distinct_before=len(before),
        distinct_after=len(after),
    )
    write_json(str(destination) + ".manifest.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=MODES, default="halfvar")
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(transform(a.input, a.output, a.mode, a.seed), indent=2))
