"""Download/preprocess code-only benchmark inputs into a run, never into the package."""

import ast, io, itertools, json, tokenize, urllib.request, urllib.parse
from pathlib import Path
from ...runtime import write_json, write_jsonl, digest


def strip_documentation(code):
    """Blank comments/docstring bytes, keep code and line positions (best-effort Python 3)."""
    lines = code.splitlines(keepends=True)
    spans = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.COMMENT:
                spans.append((t.start, t.end))
        tree = ast.parse(code)
        for n in ast.walk(tree):
            body = getattr(n, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                e = body[0]
                a = len(lines[e.lineno - 1].encode()[: e.col_offset].decode())
                b = len(lines[e.end_lineno - 1].encode()[: e.end_col_offset].decode())
                spans.append(((e.lineno, a), (e.end_lineno, b)))
    except (SyntaxError, tokenize.TokenError, IndentationError):
        pass
    for (l, a), (r, b) in sorted(spans, reverse=True):
        if l == r:
            lines[l - 1] = lines[l - 1][:a] + " " * (b - a) + lines[l - 1][b:]
        else:
            lines[l - 1] = lines[l - 1][:a] + "\n"
            for k in range(l, r - 1):
                lines[k] = "\n"
            lines[r - 1] = " " * b + lines[r - 1][b:]
    return "".join(lines)


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=90) as src, open(tmp, "wb") as dst:
            while block := src.read(1 << 20):
                dst.write(block)
        tmp.replace(path)
    return path


def fetch_prefix(url, path, n):
    """Read enough complete JSONL records with bounded HTTP range requests."""
    path = Path(path)
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        # Never mistake a cached prefix for a complete release file.
        return fetch(url, path.with_name(path.stem + ".full" + path.suffix))
    path = path.with_name(path.stem + f".n{n}" + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    size = max(1 << 20, n * 8192)
    while True:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(size)
            total = int(response.headers.get("Content-Range", "0/0").split("/")[-1])
        at_end = (total > 0 and len(raw) >= total) or len(raw) < size
        complete = raw if at_end else raw.rsplit(b"\n", 1)[0] + b"\n"
        if complete and not complete.endswith(b"\n"):
            complete += b"\n"
        if complete.count(b"\n") >= n or at_end:
            path.write_bytes(complete)
            return path
        size *= 2


def cubert_objects(prefix):
    """List every object page; large releases must not stop at the first page."""
    query = {"prefix": prefix}
    while True:
        url = (
            "https://storage.googleapis.com/storage/v1/b/cubert/o?"
            + urllib.parse.urlencode(query)
        )
        listing = json.load(urllib.request.urlopen(url))
        yield from listing.get("items", [])
        if not listing.get("nextPageToken"):
            break
        query["pageToken"] = listing["nextPageToken"]


SHARED_BENCHMARKS = {
    "py150_token": "codexglue_py150",
    "cloze_maxmin": "codexglue_cloze_maxmin",
    "cloze_all_multitoken": "codexglue_cloze_all",
    "cloze_all_paper": "codexglue_cloze_all",
}


def _lexemes(text):
    """Recover old prepared prefixes losslessly, keeping quoted whitespace intact."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                raise ValueError("Old cloze comments have ambiguous token boundaries; reprepare from the existing raw benchmark source")
            if token.type not in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                tokens.append(token.string)
    except tokenize.TokenError as error:
        if "EOF in multi-line statement" not in str(error):
            raise ValueError("Cannot reconstruct old cloze tokens; reprepare from the existing raw benchmark source") from error
    if " ".join(tokens) != text.strip():
        raise ValueError("Cannot reconstruct old cloze tokens losslessly; reprepare from the existing raw benchmark source")
    return tokens


def cloze_tokens(row):
    if "pl_tokens" in row:
        tokens = list(row["pl_tokens"])
        mask = row.get("mask_index", tokens.index("<mask>"))
        if tokens.count("<mask>") != 1 or tokens[mask] != "<mask>":
            raise ValueError("Invalid saved cloze mask")
        if " ".join(tokens[:mask]) + " " != row["prefix"] or " ".join(tokens[mask + 1:]) != row["suffix"]:
            raise ValueError("Saved cloze tokens differ from the prepared prefix/suffix")
    else:
        if "suffix" not in row:
            raise ValueError("Renaming requires cloze suffix/tokens; reprepare using the shared benchmark readers")
        prefix, suffix = _lexemes(row["prefix"]), _lexemes(row["suffix"])
        tokens, mask = prefix + ["<mask>"] + suffix, len(prefix)
    tokens[mask] = row["gold"]
    return tokens, mask


def cloze_items(rows, variant):
    """Apply codelab's original intervention to labels from the shared benchmark.

    Same seeds, candidate bindings and collision rule as the immutable scorer;
    sampling and gold recovery are owned by benchmark_suite, not duplicated here.
    """
    import random
    from collections import Counter
    from .legacy import library

    dm = library("decmech")
    items, skipped = [], 0
    for row in rows:
        tokens, mask = cloze_tokens(row)
        if mask < 2 or mask > len(tokens) - 2:
            skipped += 1
            continue
        eligible = dm.py150_renameable(tokens)
        names = sorted({t for t, ok in zip(tokens, eligible) if ok})
        views = {0: tokens}
        for n in (1, 2, 4, 8):
            rng = random.Random(dm.hash_seed("cloze", dm.BENCH_SEED, variant, row["id"], n))
            picked = list(names)
            rng.shuffle(picked)
            mapping, taken = {}, set(tokens)
            for name in picked[:n]:
                new, attempts = dm._rand_letters(rng, len(name)), 0
                while (new in taken or new in mapping.values()) and attempts < 256:
                    new = dm._rand_letters(rng, len(name))
                    attempts += 1
                mapping[name] = new
            views[n] = [mapping.get(t, t) if ok else t for t, ok in zip(tokens, eligible)]
        if any(view[mask] != tokens[mask] for view in views.values()):
            skipped += 1
            continue
        items.append(dict(id=row["id"], mask=mask, gold=row["gold"], safe=not eligible[mask], views=views, n_names=len(names)))
    answers = Counter(item["gold"] for item in items)
    return items, dict(n_items=len(items), n_safe=sum(item["safe"] for item in items), skipped_unindexable_or_changed_answer=skipped,
                       majority_answer=answers.most_common(1)[0] if answers else None,
                       distinct_answers=len(answers), frac_safe=sum(item["safe"] for item in items) / len(items) if items else None,
                       sampling="prefix from shared prepared TEST split", gold_recovery="verified benchmark_suite labels")


def prepare(run, cache=None, n=0, benchmark_run=None, benchmark_data=None, benchmark_sources=None, tasks=None):
    if n < 0:
        raise ValueError("n must be nonnegative; 0 means all examples")
    from ...runtime import read_jsonl
    from ...benchmark_suite.workflow import ensure_data, manifest as shared_manifest
    from .registry import builtins, TASKS
    from .legacy import library

    builtins()
    selected = set(tasks) if tasks is not None else set(TASKS)
    if not selected or selected - TASKS.keys():
        raise ValueError("Unknown or empty task selection")
    run = Path(run).resolve()
    data = run / "data"
    data.mkdir(exist_ok=True)
    (run / "eval_corpus").mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    benchmark_run = Path(benchmark_run) if benchmark_run else root / "runs/benchmarks"
    benchmark_data, benchmark_sources = benchmark_data or {}, benchmark_sources or {}
    manifest = dict(n_requested=n, full=n == 0, subset=n != 0, tasks=sorted(selected), sources={}, subsets={},
                    notes=["Shared CodeXGLUE TEST rows, IDs and gold labels; no separate experiment download or label recovery",
                           "All available source rows for n=0; positive limits use the shared source-order prefix",
                           "Original eligibility filters and variable-renaming scorer are retained"])

    def save(name, rows, source):
        count = 0
        def counted():
            nonlocal count
            for row in rows:
                count += 1
                yield row
        write_jsonl(data / (name + ".jsonl"), counted())
        manifest["sources"][name] = source
        manifest["subsets"][name] = dict(count=count, sha256=digest(data / (name + ".jsonl")))

    for name in sorted({SHARED_BENCHMARKS[task] for task in selected if task in SHARED_BENCHMARKS}):
        prepared = ensure_data(name, benchmark_run, n, benchmark_data.get(name), benchmark_sources.get(name), search_root=root / "runs")
        meta = shared_manifest(prepared)
        source = dict(benchmark=name, prepared=str(prepared), manifest_sha256=digest(prepared / "manifest.json"),
                      test_sha256=meta["splits"]["test"]["sha256"], raw_source=meta.get("raw_source"), prepared_subset=meta["subset"])
        rows = itertools.islice(read_jsonl(prepared / "test.jsonl"), n or None)
        if name == "codexglue_py150":
            save("py150_token", rows, source)
            continue
        variant = "all" if name.endswith("_all") else "maxmin"
        rows = list(rows)
        if any("pl_tokens" not in row for row in rows) and meta.get("raw_source"):
            from ...benchmark_suite.data import restore_cloze_tokens
            from ...source_downloads import MANIFEST, verify_source

            raw = Path(meta["raw_source"])
            verify_source(raw, name, "http_files")
            if meta.get("raw_source_manifest_sha256") and digest(raw / MANIFEST) != meta["raw_source_manifest_sha256"]:
                raise ValueError("Raw cloze source changed after benchmark preparation")
            rows = restore_cloze_tokens(rows, raw)
            source["token_source_manifest_sha256"] = digest(raw / MANIFEST)
        items, stats = cloze_items(rows, variant)
        choices = rows[0]["choices"] if rows else []
        if any(row["choices"] != choices for row in rows):
            raise ValueError("Inconsistent shared cloze vocabulary")
        dm = library("decmech")
        write_json(run / "eval_corpus" / f"cloze_{variant}_items_n{n}.json",
                   dict(n_sample=n, seed=dm.BENCH_SEED, items=items, meta=dict(stats, n_dataset=meta["splits"]["test"]["count"], n_sampled=len(rows), shared_source=source)))
        save("cloze_" + variant,
             (dict(id=item["id"], prefix=" ".join(item["views"][0][:item["mask"]]) + " ", gold=item["gold"], choices=choices, safe=item["safe"])
              for item in items), dict(source, **stats))
    from ...runtime import read_jsonl

    for task in [
        "exception",
        "variable_misuse",
        "swapped_operands",
        "wrong_binary_operator",
    ]:
        if "cubert_" + task not in selected:
            continue
        for split in ["dev", "eval"]:
            prefix = f"20200621_Python/{task}_datasets/{split}.jsontxt"
            source = []

            def rows():
                count = 0
                for info in sorted(cubert_objects(prefix), key=lambda x: x["name"]):
                    u = "https://storage.googleapis.com/cubert/" + info["name"]
                    p = fetch_prefix(
                        u,
                        run
                        / "tmp"
                        / Path(info["name"]).name.replace(split, task + "_" + split, 1),
                        n,
                    )
                    source.append(
                        {"url": u, "prefix_sha256": digest(p), "prefix_only": n != 0}
                    )
                    for row in read_jsonl(p):
                        yield dict(
                            id=str(count),
                            code=strip_documentation(row["function"]),
                            label=row["label"],
                            info=row.get("info", ""),
                        )
                        count += 1
                        if n and count >= n:
                            return

            save(
                "cubert_" + task + "_" + split,
                rows(),
                {
                    "sources": source,
                    "split": split,
                    "preprocess": "blank comments/docstrings",
                },
            )
    write_json(data / "manifest.json", manifest)
    return manifest
