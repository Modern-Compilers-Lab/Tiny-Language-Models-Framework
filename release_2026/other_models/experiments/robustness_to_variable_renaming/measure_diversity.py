"""Supplementary manipulation checks at the actual truncated training-token boundary."""

import argparse, io, json, keyword, os, tokenize
from collections import Counter
from itertools import islice
from pathlib import Path
from ...runtime import read_jsonl, write_json, setup_run, digest
from .token_cache import TokenTable


def names_before(code, cut):
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    counts = Counter()
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            start = offsets[token.start[0] - 1] + token.start[1]
            if start >= cut:
                break
            end = offsets[token.end[0] - 1] + token.end[1]
            if (
                token.type == tokenize.NAME
                and end <= cut
                and not keyword.iskeyword(token.string)
            ):
                counts[token.string] += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return counts


def measure(config, run, cache, baseline_cache, n=0):
    if n < 0:
        raise ValueError("n must be nonnegative; 0 means all rows")
    from transformers import AutoTokenizer
    import numpy as np

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "2"
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    counts = Counter()
    rows = 0
    stream = iter(islice(read_jsonl(config["data"]), n or None))
    while batch := list(islice(stream, 32)):
        codes = [r["code"] for r in batch]
        enc = tokenizer(
            codes,
            add_special_tokens=False,
            truncation=True,
            max_length=config["max_len"],
            return_offsets_mapping=True,
        )
        for code, spans in zip(codes, enc["offset_mapping"]):
            counts.update(names_before(code, max((b for a, b in spans), default=0)))
            rows += 1
    table = TokenTable(cache)
    baseline = TokenTable(baseline_cache)
    if len(table) != len(baseline):
        raise ValueError("corpus row count mismatch")
    changed = sum(not np.array_equal(table[i], baseline[i]) for i in range(len(table)))
    result = dict(
        protocol="supplementary token-visible lexical-identifier check; no significance claim",
        sample_rows=rows,
        max_len=config["max_len"],
        distinct_visible_identifiers=len(counts),
        visible_identifier_occurrences=sum(counts.values()),
        all_rows=len(table),
        rows_with_changed_training_ids=changed,
        changed_training_fraction=changed / len(table) if len(table) else None,
        data_sha256=digest(config["data"]),
        cache=str(cache),
        baseline_cache=str(baseline_cache),
    )
    write_json(Path(run) / "results/diversity_visible.json", result)
    if os.getenv("ORX_METRICS_PATH"):
        write_json(os.environ["ORX_METRICS_PATH"], result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--baseline-cache", required=True)
    p.add_argument("--n", type=int, default=0, help="row cap; 0 means all")
    a = p.parse_args(argv)
    config = json.loads(Path(a.config).read_text())
    run = setup_run(a.run, dict(arguments=vars(a), training_config=config))
    print(json.dumps(measure(config, run, a.cache, a.baseline_cache, a.n)), flush=True)


if __name__ == "__main__":
    main()
