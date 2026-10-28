"""Prepare the evaluation corpus read by every experiment (--corpus).

    python prepare_corpus.py --out CORPUS --n 2000 [--cloze 3000]

real6_corpus.json  CodeSearchNet Python TEST functions that pass the paper's filters
                   (120 < length < 2000 characters, parses, at least three local
                   variables and one integer literal, names of at most 20 characters),
                   in source order, first --n of them. The filters are the library's
                   (whyscale._real6_parse). The source is a local CodeSearchNet download
                   under other_models/runs, --source, or a new download of the test split.
cloze_all_items_n<N>.json
                   The renamed ClozeTest-all views used by the bench group of
                   exp_fig6_fig16_tasks.py, built from the framework's shared CodeXGLUE
                   preparation (first N TEST rows) with the library's renaming rule.
                   The paper sampled 3,000 rows with a stride over the whole set; the
                   shared preparation takes a prefix.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env  # noqa: E402

OTHER_MODELS = HERE.parents[1]


def find_source():
    """A complete local CodeSearchNet download that contains the Python test split."""
    from other_models.source_downloads import MANIFEST, verify_source

    for marker in sorted((OTHER_MODELS / "runs").glob(f"**/sources/*/{MANIFEST}")):
        path = marker.parent
        if list(path.glob("python/test-*.parquet")):
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("name") != "codesearchnet":
                continue
            verify_source(path, "codesearchnet", require_full=False)
            return path
    return None


def prepare_reference(out, n, source=None):
    from other_models.datasets import csn_rows
    from other_models.runtime import digest, write_json
    from other_models.source_downloads import download_dataset

    W = env.lib_module("whyscale")
    source = Path(source) if source else find_source()
    if source is None:
        source = download_dataset(
            "codesearchnet",
            out / "reference-source",
            allow_patterns=["python/test-*.parquet"],
        )
    recs, scanned = [], 0
    for row in csn_rows(source, "test"):
        scanned += 1
        code = row["func_code_string"]
        if not 120 < len(code) < 2000:
            continue
        parsed = W._real6_parse(code)
        if parsed is None:
            continue
        src, names, others, ints = parsed
        if len(names) < 3 or ints < 1 or max(map(len, names)) > W.REAL6_LMAX:
            continue
        recs.append(dict(src=src, names=names, others=others, ints=ints))
        if n and len(recs) >= n:
            break
    if not recs:
        raise ValueError("No CodeSearchNet test function passed the filters")
    corpus = out / "real6_corpus.json"
    # n_items must equal the --n given to the experiments, or the library rebuilds the corpus.
    write_json(
        corpus,
        dict(
            n_items=len(recs), min_vars=3, lmax=W.REAL6_LMAX, scanned=scanned, recs=recs
        ),
    )
    return dict(
        source=str(source), functions=len(recs), scanned=scanned, sha256=digest(corpus)
    )


def cloze_items(rows, variant="all"):
    """Renamed views of each ClozeTest row, with the library's seeds and collision rule."""
    dm = env.lib_module("decmech")
    items, skipped = [], 0
    for row in rows:
        tokens = list(row["pl_tokens"])
        mask = row.get("mask_index", tokens.index("<mask>"))
        tokens[mask] = row["gold"]
        if mask < 2 or mask > len(tokens) - 2:
            skipped += 1
            continue
        eligible = dm.py150_renameable(tokens)
        names = sorted({t for t, ok in zip(tokens, eligible) if ok})
        views = {0: tokens}
        for n in (1, 2, 4, 8):
            rng = random.Random(
                dm.hash_seed("cloze", dm.BENCH_SEED, variant, row["id"], n)
            )
            picked = list(names)
            rng.shuffle(picked)
            mapping, taken = {}, set(tokens)
            for name in picked[:n]:
                new, attempts = dm._rand_letters(rng, len(name)), 0
                while (new in taken or new in mapping.values()) and attempts < 256:
                    new = dm._rand_letters(rng, len(name))
                    attempts += 1
                mapping[name] = new
            views[n] = [
                mapping.get(t, t) if ok else t for t, ok in zip(tokens, eligible)
            ]
        # A view that renames the answer itself is not a valid item.
        if any(view[mask] != tokens[mask] for view in views.values()):
            skipped += 1
            continue
        items.append(
            dict(
                id=row["id"],
                mask=mask,
                gold=row["gold"],
                safe=not eligible[mask],
                views=views,
                n_names=len(names),
            )
        )
    answers = Counter(item["gold"] for item in items)
    meta = dict(
        n_items=len(items),
        n_safe=sum(item["safe"] for item in items),
        skipped=skipped,
        majority_answer=answers.most_common(1)[0] if answers else None,
        distinct_answers=len(answers),
        frac_safe=sum(item["safe"] for item in items) / len(items) if items else None,
        sampling="prefix of the shared prepared TEST split",
    )
    return items, meta


def prepare_cloze(out, n, benchmark_run, prepared=None):
    from other_models.benchmark_suite.workflow import ensure_data, manifest
    from other_models.runtime import digest, read_jsonl, write_json

    name = "codexglue_cloze_all"
    data = ensure_data(name, benchmark_run, n, prepared, search_root=benchmark_run)
    meta = manifest(data)
    rows = list(read_jsonl(data / "test.jsonl"))[:n]
    if any("pl_tokens" not in row for row in rows):
        if not meta.get("raw_source"):
            raise ValueError(
                f"{data} has no token boundaries (pl_tokens); prepare it again "
                "with a fresh --benchmark-run"
            )
        from other_models.benchmark_suite.data import restore_cloze_tokens

        rows = restore_cloze_tokens(rows, meta["raw_source"])
    items, stats = cloze_items(rows)
    dm = env.lib_module("decmech")
    stats.update(
        benchmark=name,
        prepared=str(data),
        test_sha256=meta["splits"]["test"]["sha256"],
        n_dataset=meta["splits"]["test"]["count"],
        n_sampled=len(rows),
    )
    path = out / f"cloze_all_items_n{n}.json"
    write_json(path, dict(n_sample=n, seed=dm.BENCH_SEED, items=items, meta=stats))
    return dict(
        path=str(path), items=len(items), safe=stats["n_safe"], sha256=digest(path)
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="corpus folder to create")
    p.add_argument("--n", type=int, default=2000, help="functions; 0 keeps all")
    p.add_argument(
        "--source", help="local CodeSearchNet download with python/test-*.parquet"
    )
    p.add_argument(
        "--cloze", type=int, default=0, help="ClozeTest rows; 0 skips the cache"
    )
    p.add_argument(
        "--benchmark-run",
        default=str(OTHER_MODELS / "runs" / "benchmarks"),
        help="shared benchmark storage used for the ClozeTest download",
    )
    p.add_argument(
        "--benchmark-data", help="an existing prepared codexglue_cloze_all folder"
    )
    a = p.parse_args(argv)
    env.setup()
    if a.n < 0 or a.cloze < 0:
        p.error("--n and --cloze must be nonnegative")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = dict(reference=prepare_reference(out, a.n, a.source))
    if a.cloze:
        result["cloze"] = prepare_cloze(
            out, a.cloze, Path(a.benchmark_run), a.benchmark_data
        )
    from other_models.runtime import write_json

    write_json(out / "manifest.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
