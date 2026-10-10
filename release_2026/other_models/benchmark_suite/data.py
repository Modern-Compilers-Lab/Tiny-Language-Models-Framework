"""Prepare only requested splits and rows, retaining source/split provenance."""

import io
import itertools
import json
import sqlite3
import tarfile
import tempfile
import uuid
import zipfile
from pathlib import Path

from ..runtime import digest, read_jsonl, write_json, write_jsonl
from . import BENCHMARKS
from .downloads import Budget, json_items, parquet_rows, seekable, stream

RAW = "https://raw.githubusercontent.com"
HF = "https://huggingface.co/datasets"
BIFI_DATA = "https://nlp.stanford.edu/projects/myasu/BIFI/"
MT4PY = "https://zenodo.org/records/5244636/files/ManyTypes4PyDataset-v0.7.tar.gz?download=1"


def take(rows, limit):
    return rows if limit == 0 else itertools.islice(rows, limit)


def local_splits(source, splits):
    source = Path(source)
    for split in splits:
        path = source / (split + ".jsonl")
        if not path.exists():
            raise FileNotFoundError(path)
        yield split, read_jsonl(path)


def py150_splits(budget, include_train, source=None):
    root = (
        str(Path(source) / "python") + "/"
        if source
        else HF + "/google/code_x_glue_cc_code_completion_token/resolve/main/python/"
    )

    def rows(split):
        if split == "test":
            raw = parquet_rows(root + "test-00000-of-00001.parquet", budget)
        else:
            # CodeXGLUE takes the final 5k of PY150's 100k training files as DEV.
            raw = (
                itertools.chain(
                    parquet_rows(root + "train-00000-of-00002.parquet", budget),
                    parquet_rows(
                        root + "train-00001-of-00002.parquet", budget, stop=45000
                    ),
                )
                if split == "train"
                else parquet_rows(
                    root + "train-00001-of-00002.parquet", budget, start=45000
                )
            )
        for i, r in enumerate(raw):
            tokens = r.get("code", r.get("item"))
            if isinstance(tokens, str):
                tokens = (
                    json.loads(tokens) if tokens.startswith("[") else tokens.split()
                )
            yield dict(id=f"{split}:{i}", code=tokens, split=split)

    for split in ["train", "dev", "test"] if include_train else ["test"]:
        yield split, rows(split)


def crux_rows(budget, source=None):
    url = (
        str(Path(source) / "cruxeval.jsonl")
        if source
        else RAW + "/facebookresearch/cruxeval/main/data/cruxeval.jsonl"
    )
    with stream(url, budget) as handle:
        # No full read: stop downloading immediately after the requested prefix.
        for line in io.TextIOWrapper(io.BufferedReader(_Raw(handle)), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                # Upstream JSON contains escaped backslash-n rather than newlines.
                r["code"] = r["code"].replace("\\n", "\n")
                yield dict(r, split="test")


class _Raw(io.RawIOBase):
    def __init__(self, handle):
        self.handle = handle

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.handle.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def recover_cloze(rows, budget, source=None):
    """Labels are withheld upstream; match exact tokens against CSN heldout splits.

    Unrecovered and ambiguous rows cause an error; never silently select easy rows.
    """
    pending, found = {}, {}
    for r in rows:
        t = r["pl_tokens"]
        if t.count("<mask>") != 1:
            raise ValueError("expected one cloze mask")
        m = t.index("<mask>")
        key = (len(t), tuple(t[:m]), tuple(t[m + 1 :]))
        pending.setdefault(len(t), []).append((r["id"], m, key))
    for split in ("test", "validation"):
        url = (
            HF
            + f"/code-search-net/code_search_net/resolve/main/python/{split}-00000-of-00001.parquet"
        )
        if source:
            url = str(Path(source) / "csn" / f"{split}-00000-of-00001.parquet")
        for r in parquet_rows(url, budget, columns=["func_code_tokens"]):
            t = r["func_code_tokens"]
            for key_id, m, key in pending.get(len(t), []):
                if key == (len(t), tuple(t[:m]), tuple(t[m + 1 :])):
                    if key_id in found and found[key_id] != t[m]:
                        raise ValueError(f"ambiguous cloze gold: {key_id}")
                    found[key_id] = t[m]
    missing = {r["id"] for r in rows} - found.keys()
    if missing:
        raise ValueError(
            f"Cannot recover {len(missing)} cloze labels; provide labeled local test.jsonl"
        )
    return found


def cloze_rows(name, budget, limit, source=None):
    variant = "all" if name.endswith("_all") else "maxmin"
    root = (
        RAW
        + f"/microsoft/CodeXGLUE/main/Code-Code/ClozeTesting-{variant}/data/cloze-{variant}/"
    )
    raw = (
        str(Path(source) / "clozeTest.json")
        if source
        else root + "python/clozeTest.json"
    )
    with stream(raw, budget) as handle:
        rows = [dict(r, id=r["idx"]) for r in take(json_items(handle), limit)]
    if variant == "all":
        words = (
            str(Path(source) / "cloze_test_words.txt")
            if source
            else root + "cloze_test_words.txt"
        )
        with stream(words, budget) as handle:
            choices = handle.read().decode().splitlines()
    else:
        choices = ["max", "min"]
    # The repository's evaluator/answers files contain placeholder token1, token2,
    # etc. They are format examples, not benchmark labels.
    gold = recover_cloze(rows, budget, source)
    for r in rows:
        m = r["pl_tokens"].index("<mask>")
        if gold[r["id"]] not in choices:
            raise ValueError("gold outside official candidate vocabulary")
        yield dict(
            id=r["id"],
            prefix=" ".join(r["pl_tokens"][:m]) + " ",
            suffix=" ".join(r["pl_tokens"][m + 1 :]),
            gold=gold[r["id"]],
            choices=choices,
            split="test",
        )


def zip_member(archive, suffix):
    matches = [n for n in archive.namelist() if n.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one archive member ending in {suffix}, found {len(matches)}"
        )
    return archive.open(matches[0])


def bifi_splits(source, budget, include_train, limit):
    source = source or BIFI_DATA + ("data.zip" if include_train else "data_minimal.zip")
    with seekable(source, budget) as handle, zipfile.ZipFile(handle) as archive:
        if include_train:
            for split in ("train", "dev"):
                # Original round0 pairs; no breaker training or test-generated pairs.
                with (
                    zip_member(archive, f"round0/data_paired/{split}.bad") as bad,
                    zip_member(archive, f"round0/data_paired/{split}.good") as good,
                ):

                    def pairs():
                        for i, pair in enumerate(itertools.zip_longest(bad, good)):
                            a, b = pair
                            if a is None or b is None:
                                raise ValueError("Unaligned BIFI training pairs")
                            yield dict(
                                id=f"{split}:{i}",
                                buggy_tokens=a.decode().strip(),
                                fixed_tokens=b.decode().strip(),
                                split=split,
                            )

                    yield split, take(pairs(), limit)

        def test():
            from .metrics import paren_ok

            # Sample each of the two heldout chunks, with a shared total limit.
            per_chunk = None if not limit else ((limit + 1) // 2, limit // 2)
            for index, chunk in enumerate((3, 4)):
                with (
                    zip_member(archive, f"orig_bad_code/orig.{chunk}.bad") as codes,
                    zip_member(archive, f"orig_bad_code/orig.{chunk}.id") as ids,
                ):
                    for i, pair in enumerate(itertools.zip_longest(codes, ids)):
                        if per_chunk is not None and i >= per_chunk[index]:
                            break
                        code, key = pair
                        if code is None or key is None:
                            raise ValueError("Unaligned BIFI test IDs")
                        tokens = code.decode().strip()
                        yield dict(
                            id=key.decode().strip(),
                            buggy_tokens=tokens,
                            chunk=chunk,
                            split="test",
                            error_type="unbalanced (){}[]"
                            if not paren_ok(tokens.split())
                            else "syntax/indentation",
                        )

        yield "test", test()


def manytypes_slots(project):
    """LibSA4Py project JSON: use untyped_seq, never typed_seq in prompts."""
    for repo, obj in project.items():
        for path, file in obj.get("src_files", {}).items():
            split = {"valid": "dev", "validation": "dev"}.get(
                file.get("set"), file.get("set")
            )
            if split not in ("train", "dev", "test"):
                raise ValueError(f"Missing official split for {path}")
            context = file["untyped_seq"]
            if isinstance(context, list):
                context = " ".join(context)

            def slots(node, scope):
                for field, kind in (("params", "Parameter"), ("variables", "Variable")):
                    for target, gold in node.get(field, {}).items():
                        if (
                            gold
                            and gold
                            not in (
                                "Any",
                                "typing.Any",
                                "None",
                                "object",
                                "self",
                                "cls",
                            )
                            and target not in ("self", "cls")
                        ):
                            yield scope, target, kind, gold
                if node.get("ret_type") and node["ret_type"] not in (
                    "Any",
                    "typing.Any",
                    "None",
                ):
                    yield scope, node.get("name", scope), "Return", node["ret_type"]
                for fn in node.get("funcs", []):
                    yield from slots(fn, fn.get("q_name", fn["name"]))
                for cls in node.get("classes", []):
                    yield from slots(cls, cls.get("q_name", cls["name"]))

            for scope, target, kind, gold in slots(file, "<module>"):
                key = f"{repo}:{path}:{scope}:{kind}:{target}"
                yield dict(
                    id=key,
                    code=context,
                    scope=scope,
                    target=target,
                    kind=kind,
                    gold=gold,
                    split=split,
                    repo=repo,
                    file=path,
                )


def manytypes_splits(source, budget, splits, limit, version="clean", spool_dir=None):
    counts = dict.fromkeys(splits, 0)
    with tempfile.TemporaryDirectory(prefix="types-", dir=spool_dir) as spool:
        paths = {s: Path(spool) / (s + ".jsonl") for s in splits}
        outputs = {s: paths[s].open("w", encoding="utf-8") for s in splits}
        try:
            with (
                stream(source or MT4PY, budget) as handle,
                tarfile.open(fileobj=handle, mode="r|gz") as archive,
            ):
                for member in archive:
                    if (
                        not member.isfile()
                        or not member.name.endswith(".json")
                        or f"processed_projects_{version}/" not in member.name
                    ):
                        continue
                    # One project in memory; selected rows are spooled, never the archive.
                    if member.size > 64 << 20 and limit:
                        raise ValueError(
                            "Project JSON exceeds 64 MiB; use normalized local splits"
                        )
                    with archive.extractfile(member) as inp:
                        for r in manytypes_project_rows(inp):
                            s = r["split"]
                            if s in counts and (not limit or counts[s] < limit):
                                outputs[s].write(
                                    json.dumps(r, ensure_ascii=False) + "\n"
                                )
                                counts[s] += 1
                    if limit and all(n >= limit for n in counts.values()):
                        break
        finally:
            for output in outputs.values():
                output.close()
        for s in splits:
            rows = read_jsonl(paths[s])
            try:
                yield s, rows
            finally:
                rows.close()


def manytypes_project_rows(handle):
    """Stream one source-file object at a time, including very large projects."""
    import ijson
    from ijson.common import ObjectBuilder

    events = iter(ijson.parse(handle))
    repo, filename = None, None
    for prefix, event, value in events:
        if prefix == "" and event == "map_key":
            repo = value
        elif prefix == f"{repo}.src_files" and event == "map_key":
            filename = value
        elif (
            filename is not None
            and prefix == f"{repo}.src_files.{filename}"
            and event == "start_map"
        ):
            builder = ObjectBuilder()
            builder.event(event, value)
            depth = 1
            for _, nested, item in events:
                builder.event(nested, item)
                depth += int(nested in ("start_map", "start_array"))
                depth -= int(nested in ("end_map", "end_array"))
                if depth == 0:
                    break
            yield from manytypes_slots({repo: {"src_files": {filename: builder.value}}})
            filename = None


def validate_row(name, row, split):
    if row.get("split") != split:
        raise ValueError("Row split does not match source split")
    if "id" not in row:
        raise ValueError("Missing benchmark row ID")
    required = (
        ("code",)
        if name == "codexglue_py150"
        else (
            ("buggy_tokens", "error_type")
            if split == "test"
            else ("buggy_tokens", "fixed_tokens")
        )
        if name == "bifi"
        else ("code", "gold", "kind", "target", "scope")
        if name == "manytypes4py"
        else ("code", "input", "output")
        if name.startswith("cruxeval_")
        else ("prefix", "gold", "choices")
    )
    if not all(k in row for k in required):
        raise ValueError(f"Missing fields for {name}: {required}")
    if name == "bifi" and split != "test" and not row.get("fixed_tokens"):
        raise ValueError("BIFI train/dev require repair pairs")
    if name == "bifi" and split == "test" and row.get("chunk") not in (3, 4):
        raise ValueError("BIFI test must use heldout chunks 3/4")
    if name.startswith("codexglue_cloze") and row["gold"] not in row["choices"]:
        raise ValueError("Cloze label outside choices")
    if name == "codexglue_py150" and (
        not isinstance(row["code"], list)
        or not all(isinstance(t, str) for t in row["code"])
    ):
        raise ValueError("PY150 requires a lexical token list")


def prepare(
    name,
    run,
    source=None,
    limit=None,
    include_train=None,
    max_download_mb=None,
    version="clean",
    full=None,
):
    if full is None:
        full = limit is None or limit == 0
    if limit is None:
        limit = 0 if full else 128
    if limit < 0:
        raise ValueError("limit must be nonnegative (0 means full split)")
    spec = BENCHMARKS[name]
    if full:
        limit = 0
    if include_train is None:
        include_train = full and spec.trainable
    if max_download_mb is None:
        max_download_mb = 0 if full else 64
    if max_download_mb < 0:
        raise ValueError("max_download_mb must be nonnegative")
    if include_train and not spec.trainable:
        raise ValueError(f"{name} is evaluation-only; no official training split")
    dest = Path(run) / "data" / name
    if (dest / "manifest.json").exists():
        raise ValueError("Benchmark already prepared; choose a new run directory")
    budget = Budget(max_download_mb << 20)
    splits = ["train", "dev", "test"] if include_train else ["test"]
    from ..source_downloads import MANIFEST, verify_source

    raw_source = None
    raw_meta = None
    if source and Path(source).is_dir() and (Path(source) / MANIFEST).exists():
        raw_meta = verify_source(source, name, "http_files")
        raw_source = Path(source)
    elif full and not source:
        from .sources import download

        raw_source = download(name, run, budget)
        raw_meta = json.loads((raw_source / MANIFEST).read_text(encoding="utf-8"))
    if raw_source and name == "codexglue_py150":
        providers = py150_splits(budget, include_train, raw_source)
    elif raw_source and name.startswith("codexglue_cloze"):
        providers = [("test", cloze_rows(name, budget, limit, raw_source))]
    elif raw_source and name.startswith("cruxeval_"):
        providers = [("test", crux_rows(budget, raw_source))]
    elif source and Path(source).is_dir() and not raw_source:
        providers = local_splits(source, splits)
    elif name == "bifi":
        providers = bifi_splits(
            raw_source / "data.zip" if raw_source else source,
            budget,
            include_train,
            limit,
        )
    elif name == "manytypes4py":
        spool_dir = Path(run) / "tmp"
        spool_dir.mkdir(parents=True, exist_ok=True)
        providers = manytypes_splits(
            raw_source / "ManyTypes4PyDataset-v0.7.tar.gz" if raw_source else source,
            budget,
            splits,
            limit,
            version,
            spool_dir,
        )
    elif source:
        raise ValueError(
            "For this benchmark --source must be a directory with normalized split JSONL files"
        )
    elif name == "codexglue_py150":
        providers = py150_splits(budget, include_train)
    elif name.startswith("codexglue_cloze"):
        providers = [("test", cloze_rows(name, budget, limit))]
    else:
        providers = [("test", crux_rows(budget))]
    manifest = dict(
        benchmark=name,
        source=str(source or spec.source),
        limit=limit,
        full=full,
        max_download_mb=max_download_mb,
        raw_source=str(raw_source) if raw_source else None,
        raw_source_manifest_sha256=digest(raw_source / MANIFEST)
        if raw_source
        else None,
        protocol=spec.protocol,
        trainable=spec.trainable,
        splits={},
        units="metrics are fractions [0,1]",
        version=version if name == "manytypes4py" else None,
        sampling="prefix in each heldout BIFI chunk"
        if name == "bifi"
        else "prefix in source order per split",
    )
    seen = set()
    connection, index = None, None
    if limit == 0:
        directory = Path(run) / "tmp"
        directory.mkdir(parents=True, exist_ok=True)
        index = directory / ("benchmark-ids-" + uuid.uuid4().hex + ".sqlite")
        connection = sqlite3.connect(index)
        connection.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")

    def unique(key):
        if connection is not None:
            return (
                connection.execute(
                    "INSERT OR IGNORE INTO seen VALUES (?)", (key,)
                ).rowcount
                == 1
            )
        if key in seen:
            return False
        seen.add(key)
        return True

    labels = set()
    common_counts = {}
    try:
        for split, rows in providers:
            path = dest / (split + ".jsonl")

            def checked():
                for row in take(rows, limit):
                    validate_row(name, row, split)
                    key = str(row["id"])
                    if not unique(key):
                        raise ValueError(f"Duplicate or overlapping split ID: {key}")
                    if name == "manytypes4py" and split == "train":
                        labels.add(row["gold"])
                        common_counts[row["gold"]] = (
                            common_counts.get(row["gold"], 0) + 1
                        )
                    yield row

            try:
                write_jsonl(path, checked())
            finally:
                if hasattr(rows, "close"):
                    rows.close()
            count = sum(1 for _ in read_jsonl(path))
            if not count:
                raise ValueError(f"Empty {split} split for {name}")
            manifest["splits"][split] = dict(count=count, sha256=digest(path))
    finally:
        if connection is not None:
            connection.close()
            index.unlink(missing_ok=True)
    if labels:
        write_json(dest / "type_vocabulary.json", sorted(labels))
        manifest["type_vocabulary_sha256"] = digest(dest / "type_vocabulary.json")
        write_json(dest / "type_train_counts.json", common_counts)
    manifest["download_bytes"] = budget.used
    manifest["raw_source_bytes"] = raw_meta["total_bytes"] if raw_meta else 0
    manifest["subset"] = limit != 0
    write_json(dest / "manifest.json", manifest)
    return dest
