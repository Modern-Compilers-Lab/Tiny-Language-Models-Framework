"""Preserve JSONL row schema, IDs, order, metadata and all non-code fields."""

import random
from collections import Counter
from ..registry import TRANSFORMS
from ....runtime import read_jsonl, write_jsonl, write_json, digest


def transform(source, destination, mode="halfvar", seed=2):
    from pathlib import Path

    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Transformation output must differ from its source")
    if mode not in ("plain", "halfvar", "highdiv"):
        raise ValueError(mode)
    from ..legacy import library

    C = library("corpus")
    fresh = C.FreshNames(key=seed)
    rng = random.Random(seed)
    stats = Counter()
    before = set()
    after = set()

    def rows():
        for row in read_jsonl(source):
            if not isinstance(row.get("code"), str):
                raise ValueError("code must be a string")
            code = row["code"]
            out = code
            status = "plain"
            names = sites = 0
            if mode != "plain":
                out, status, names, sites = C.rename_document_halfvar(
                    code, fresh, rng.random, mix_frac=0.5 if mode == "halfvar" else 1.0
                )
            stats["rows"] += 1
            stats[status] += 1
            stats["changed"] += out != code
            stats["renamed_bindings"] += names
            # Unique variables in a fixed prefix, plus exact intervention accounting.
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
        recipe="paper-span-fresh-v1",
    )
    write_json(str(destination) + ".manifest.json", report)
    return report


for _name in ("plain", "halfvar", "highdiv"):
    TRANSFORMS.add(_name, transform)
