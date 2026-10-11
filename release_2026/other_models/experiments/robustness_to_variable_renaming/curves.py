"""Paired N=0,1,2,4,8 evaluation. Paper tasks retain their immutable scorer.

Additional tasks use the paper's same-length random-name intervention and stable
item IDs. Curves always report requested AND achieved rename counts; N is a cap.
"""

import json, random, re
from pathlib import Path
from ...runtime import read_jsonl, write_jsonl, write_json, digest

NS = (0, 1, 2, 4, 8)


def lexical_views(tokens, key):
    from .legacy import library

    dm = library("decmech")
    ok = dm.py150_renameable(tokens)
    names = sorted({t for t, k in zip(tokens, ok) if k})
    views = {"0": tokens}
    counts = {"0": 0}
    for n in NS[1:]:
        rng = random.Random(dm.hash_seed("py150", dm.BENCH_SEED, key, n))
        picked = list(names)
        rng.shuffle(picked)
        mapping = {}
        taken = set(tokens)
        for v in picked[:n]:
            # Collision rejection, including Python keywords/builtins.
            for attempt in range(10000):
                new = dm._rand_letters(rng, len(v))
                if (
                    new not in taken
                    and new not in mapping.values()
                    and not dm._kwlike(new)
                ):
                    break
            else:
                raise ValueError("fresh-name space exhausted")
            mapping[v] = new
        views[str(n)] = [mapping.get(t, t) if o else t for t, o in zip(tokens, ok)]
        counts[str(n)] = len(mapping)
    return views, counts


def code_views(code, key):
    from .legacy import library

    ws = library("whyscale")
    dm = library("decmech")
    parsed = ws._real6_parse(code)
    if parsed is None:
        return None
    src, names, *_ = parsed
    views = {"0": src}
    counts = {"0": 0}
    for n in NS[1:]:
        rng = random.Random(dm.hash_seed("curveren", dm.BENCH_SEED, key, n))
        picked = list(names)
        rng.shuffle(picked)
        mapping = {}
        taken = set(ws.IDENT_RE.findall(src))
        for v in picked[:n]:
            for attempt in range(10000):
                new = dm._rand_letters(rng, len(v))
                if (
                    new not in taken
                    and new not in mapping.values()
                    and not dm._kwlike(new)
                ):
                    break
            else:
                raise ValueError("fresh-name space exhausted")
            mapping[v] = new
        new = ws._real6_apply(src, mapping) if mapping else src
        if not ws._alpha_ok(src, new, mapping):
            return None
        views[str(n)] = new
        counts[str(n)] = len(mapping)
    return views, counts


def prepare_views(data, destination, limit=0, tasks=None):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    from .registry import builtins, TASKS
    from .evaluation import PAPER

    builtins()
    data = Path(data)
    destination = Path(destination)
    if data.resolve() == destination.resolve():
        raise ValueError("Views directory must differ from the original data directory")
    manifest = {}
    selected = set(tasks) if tasks is not None else set(TASKS)
    if selected - TASKS.keys():
        raise ValueError(f"Unknown tasks: {selected - TASKS.keys()}")
    for task in sorted(selected - PAPER):
        rows = []
        dropped = 0
        file = (
            "cloze_all"
            if task == "cloze_all_multitoken"
            else task + ("_eval" if task.startswith("cubert_") else "")
        )
        from itertools import islice

        source = list(islice(read_jsonl(data / (file + ".jsonl")), limit or None))
        cached = {}
        if task.startswith("cloze_"):
            variant = "all" if task == "cloze_all_multitoken" else "maxmin"
            size = json.loads((data / "manifest.json").read_text())["n_requested"]
            cached = {
                str(r["id"]): r
                for r in json.loads(
                    (
                        data.parent
                        / "eval_corpus"
                        / f"cloze_{variant}_items_n{size}.json"
                    ).read_text()
                )["items"]
            }
        for index, row in enumerate(source):
            if task.startswith("cloze_"):
                original = cached[str(row["id"])]
                if not original["safe"]:
                    dropped += 1
                    continue
                views = {
                    str(n): " ".join(original["views"][str(n)][: original["mask"]])
                    + " "
                    for n in NS
                }
                counts = {
                    str(n): len(
                        {
                            a
                            for a, b in zip(
                                original["views"]["0"], original["views"][str(n)]
                            )
                            if a != b
                        }
                    )
                    for n in NS
                }
                views = {k: dict(row, prefix=v) for k, v in views.items()}
            else:
                result = (
                    lexical_views(row["code"], index)
                    if task == "py150_token"
                    else code_views(row["code"], index)
                )
                if result is None:
                    dropped += 1
                    continue
                raw, counts = result
                views = {k: dict(row, code=v) for k, v in raw.items()}
            rows.append(dict(id=row["id"], views=views, renamed=counts))
        write_jsonl(destination / (task + ".jsonl"), rows)
        meta = dict(
            source=len(source),
            kept=len(rows),
            dropped=dropped,
            sha256=digest(destination / (task + ".jsonl")),
            mean_renamed={
                str(n): sum(r["renamed"][str(n)] for r in rows) / len(rows)
                if rows
                else None
                for n in NS
            },
        )
        if task.startswith("cubert_") and task != "cubert_exception":
            dev = []
            for i, row in enumerate(
                islice(read_jsonl(data / (task + "_dev.jsonl")), limit or None)
            ):
                result = code_views(row["code"], i)
                if result:
                    dev.append(dict(row, code=result[0]["0"]))
            write_jsonl(destination / (task + "_dev.jsonl"), dev)
            meta["dev_sha256"] = digest(destination / (task + "_dev.jsonl"))
        manifest[task] = meta
    write_json(destination / "manifest.json", manifest)
    return manifest


def score_curves(predictions):
    """Pure scoring of paired prediction rows; no model or input-label access."""
    from .tasks.scoring import fraction

    by = {str(n): {} for n in NS}
    for r in predictions:
        n = str(r["N"])
        key = str(r["id"])
        if n not in by or key in by[n]:
            raise ValueError("invalid N or duplicate prediction ID")
        by[n][key] = bool(r["correct"])
    base = by["0"]
    out = {}
    for n, cur in by.items():
        keys = base.keys() & cur.keys()
        correct = [key for key in keys if base[key]]
        retention = fraction(cur[key] for key in correct)
        out[n] = dict(
            **fraction(cur.values()),
            paired_count=len(keys),
            retained=retention["correct"],
            baseline_correct_paired=retention["count"],
            retention=retention["accuracy"],
        )
    return out


def predict_curves(model, task, path, options):
    from .registry import TASKS
    from .tasks.bugs import threshold

    options = dict(options)
    from itertools import islice

    limit = options.get("limit", 0)
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    rows = list(islice(read_jsonl(path), limit or None))
    identifiers = set()
    for row in rows:
        identifier = str(row["id"])
        if identifier in identifiers:
            raise ValueError("Duplicate view ID")
        identifiers.add(identifier)
        for n in NS:
            if str(row["views"][str(n)]["id"]) != identifier:
                raise ValueError("Paired view IDs differ")
    if task.startswith("cubert_") and task != "cubert_exception":
        dev = options["dev"]
        labels = [r["label"] == "Correct" for r in dev]
        if len(set(labels)) != 2:
            raise ValueError("calibration requires both dev classes")
        options["threshold"] = threshold(
            model.sequence_scores(
                [r["code"] for r in dev], options.get("max_len", 320)
            ),
            labels,
        )
    result = []
    for n in NS:
        predictions = TASKS[task](model, [r["views"][str(n)] for r in rows], options)
        result.extend(dict(r, N=n) for r in predictions)
    return result


def paper_curves(task, result):
    r = result["raw"]
    out = {}
    for n in NS:
        if task == "identifier_reproduction":
            a = r["A_reproduction"][f"N{n}_accuracy"]
            ret = r["A_reproduction"].get(
                f"N{n}_retention",
                dict(rate=1.0 if a["k"] else None, k=a["k"], n=a["k"]),
            )
        elif task == "cloze_all_paper":
            a = (
                r["N0_accuracy_safe"]
                if n == 0
                else dict(
                    rate=r[f"N{n}"]["accuracy_safe"],
                    k=None,
                    n=r["N0_accuracy_safe"]["n"],
                )
            )
            ret = (
                r[f"N{n}"]["retention_safe"]
                if n
                else dict(rate=1.0 if a["k"] else None, k=a["k"], n=a["k"])
            )
        else:
            a = (
                r["mean"]["N0_localization"]
                if n == 0
                else dict(
                    rate=r["mean"][f"N{n}"]["localization_accuracy"],
                    k=None,
                    n=result["count"],
                )
            )
            ret = (
                r["mean"][f"N{n}"]["retention_localization"]
                if n
                else dict(rate=1.0 if a["k"] else None, k=a["k"], n=a["k"])
            )
        out[str(n)] = dict(
            accuracy=a["rate"],
            correct=a.get("k"),
            count=a["n"],
            retention=ret["rate"],
            retained=ret["k"],
            baseline_correct_paired=ret["n"],
        )
    return out
