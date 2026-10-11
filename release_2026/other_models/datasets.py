"""Python training corpora with shared JSONL and codelab leakage filters."""

import ast
import gzip
import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path

from .runtime import digest, write_json, write_jsonl
from .source_downloads import MANIFEST, REPOSITORIES, download_dataset, verify_source


def normalize_repo(repo):
    repo = (
        str(repo or "")
        .lower()
        .strip()
        .removeprefix("https://")
        .removeprefix("http://")
        .removeprefix("github.com/")
        .removesuffix(".git")
        .strip("/")
    )
    return "/".join(repo.split("/")[:2])


def canonical_code(code):
    try:
        return ast.dump(ast.parse(code), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        return re.sub(r"\s+", "", code)


def code_hash(code):
    return hashlib.sha256(canonical_code(code).encode()).hexdigest()


def csn_rows(cache, split="train"):
    parquet = sorted(Path(cache).glob(f"python/{split}-*.parquet"))
    if parquet:
        import pyarrow.parquet as pq

        for path in parquet:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        return
    files = sorted(
        Path(cache).glob(
            f"datasets/code_search_net/python/*/*/code_search_net-{split}*.arrow"
        )
    )
    if not files:
        files = sorted(
            Path(cache).glob(
                f"datasets/code-search-net___code_search_net/python/*/*/code_search_net-{split}*.arrow"
            )
        )
    if files:
        from datasets import Dataset

        for path in files:
            yield from Dataset.from_file(str(path))
    else:
        raise ValueError(f"No CodeSearchNet Python {split} files in {cache}")


def source_rows(name, cache, scale="full"):
    cache = Path(cache)
    if name == "codesearchnet":
        for r in csn_rows(cache):
            yield (
                r["func_code_string"],
                r["repository_name"],
                r.get("func_path_in_repository", ""),
            )
    elif name == "thevault":
        import pyarrow.parquet as pq

        files = sorted(
            cache.glob(
                "data/train/"
                + ("small" if scale == "small" else "*")
                + "/python-*.parquet"
            )
        ) or sorted(
            cache.glob(
                "hub/datasets--Fsoft-AIC--the-vault-function/snapshots/*/data/train/"
                + ("small" if scale == "small" else "*")
                + "/python-*.parquet"
            )
        )
        if not files:
            raise ValueError(f"No The Vault Python training files in {cache}")
        for f in files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=1024):
                for r in b.to_pylist():
                    yield r["original_string"], r.get("repo", ""), r.get("path", "")
    elif name == "codeparrot":
        files = sorted(cache.glob("file-*.json.gz")) or sorted(
            cache.glob(
                "hub/datasets--codeparrot--codeparrot-clean/snapshots/*/file-*.json.gz"
            )
        )
        if not files:
            raise ValueError(f"No CodeParrot training files in {cache}")
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8") as inp:
                for line in inp:
                    r = json.loads(line)
                    yield (
                        r["content"],
                        r.get("repo_name", r.get("repo", "")),
                        r.get("path", ""),
                    )
    else:
        raise ValueError(name)


def prepare(
    name,
    run,
    cache=None,
    scale="full",
    n_docs=None,
    exclude_repos=None,
    reference_cache=None,
    max_files=None,
    revision="main",
):
    run = Path(run)
    dest = run / "data" / f"{name}.jsonl"
    n_docs = 0 if n_docs is None else n_docs
    if n_docs < 0:
        raise ValueError("n_docs must be nonnegative; 0 requests complete sources")
    full = n_docs == 0
    if name not in REPOSITORIES:
        raise ValueError(f"Unknown dataset: {name}")
    if scale not in ("small", "full"):
        raise ValueError("scale must be small or full")
    if cache is not None and max_files is not None:
        raise ValueError(
            "max_files applies to downloads; omit it when using a local cache"
        )
    source_meta = None
    reference_meta = None
    if cache is None:
        cache = download_dataset(name, run, revision=revision, max_files=max_files)
    cache = Path(cache)
    if (cache / MANIFEST).exists():
        source_meta = verify_source(cache, name, "hf_repository", require_full=False)
    # Reuse the complete test split for repository/function leakage protection.
    if reference_cache is not None:
        held_cache = Path(reference_cache)
    elif name == "codesearchnet" and list(cache.glob("python/test-*.parquet")):
        held_cache = cache
    else:
        held_cache = download_dataset(
            "codesearchnet",
            run / "reference",
            allow_patterns=["python/test-*.parquet"],
            revision=source_meta["revision"]
            if name == "codesearchnet" and source_meta
            else "main",
        )
    if (held_cache / MANIFEST).exists():
        reference_meta = verify_source(held_cache, "codesearchnet", require_full=False)
    repos, hashes = set(), set()
    for r in csn_rows(held_cache, "test"):
        repos.add(normalize_repo(r["repository_name"]))
        hashes.add(code_hash(r["func_code_string"]))
    repos.update(normalize_repo(r) for r in (exclude_repos or []))
    stats = dict(
        dataset=name,
        scale=scale,
        n_requested=n_docs,
        scanned=0,
        kept=0,
        excluded_repo=0,
        excluded_exact_function=0,
        missing_repo=0,
        duplicate=0,
        too_short=0,
        too_long=0,
        heldout_repos=len(repos),
        sampling="all eligible rows" if full else "first eligible in source order",
        full=full
        and scale == "full"
        and bool(source_meta and source_meta.get("full", True)),
        all_available_rows=full,
        source_full=source_meta.get("full", True) if source_meta else None,
        source_scope="all available Python training partitions"
        if scale == "full"
        else "small Python training partitions",
        source_revision=source_meta["revision"] if source_meta else None,
        reference_cache=str(held_cache),
        reference_revision=reference_meta["revision"] if reference_meta else None,
        source_cache=str(cache),
    )
    seen = set()
    connection, index = None, None
    if full:
        directory = run / "tmp"
        directory.mkdir(parents=True, exist_ok=True)
        index = directory / ("dedup-" + uuid.uuid4().hex + ".sqlite")
        connection = sqlite3.connect(index)
        connection.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")

    def unique(h):
        if connection is not None:
            return (
                connection.execute(
                    "INSERT OR IGNORE INTO seen VALUES (?)", (h,)
                ).rowcount
                == 1
            )
        if h in seen:
            return False
        seen.add(h)
        return True

    def rows():
        for code, repo, path in source_rows(name, cache, scale):
            stats["scanned"] += 1
            repo = normalize_repo(repo)
            if not repo:
                stats["missing_repo"] += 1
                continue
            if repo in repos:
                stats["excluded_repo"] += 1
                continue
            if len(code) < 60:
                stats["too_short"] += 1
                continue
            if len(code) > 200000:
                stats["too_long"] += 1
                continue
            h = code_hash(code)
            # Function hashes protect copied functions even when repository spelling differs.
            try:
                fn_hashes = {
                    code_hash(ast.get_source_segment(code, n) or "")
                    for n in ast.walk(ast.parse(code))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
            except (SyntaxError, ValueError, RecursionError):
                fn_hashes = set()
            if h in hashes or fn_hashes & hashes:
                stats["excluded_exact_function"] += 1
                continue
            if not unique(h):
                stats["duplicate"] += 1
                continue
            stats["kept"] += 1
            yield dict(
                id=h,
                code=code,
                repo=repo,
                split="train",
                meta={"source": name, "path": path},
            )
            if n_docs and stats["kept"] >= n_docs:
                break
        if not stats["kept"]:
            raise ValueError("no eligible rows")

    try:
        write_jsonl(dest, rows())
    finally:
        if connection is not None:
            connection.close()
            index.unlink(missing_ok=True)
    stats["sha256"] = digest(dest)
    write_json(str(dest) + ".manifest.json", stats)
    write_json(run / "data" / "heldout_repos.json", sorted(repos))
    return dest
