"""Prepare external benchmarks; optionally retain a pinned historical reference corpus."""

import argparse, json, shutil
from pathlib import Path
from unittest.mock import patch
from ...runtime import setup_run, write_json, digest


def with_native_loader(provider, native_loader):
    """A legacy loader shim must not recursively intercept a modern streaming fallback."""

    def rows(cache, split="train"):
        import datasets

        with patch.object(datasets, "load_dataset", native_loader):
            yield from provider(cache, split)

    return rows


def prepare_reference(run, cache, n, pinned=None):
    if n < 0:
        raise ValueError("n must be nonnegative; 0 means all examples")
    target = Path(run) / "eval_corpus"
    target.mkdir(exist_ok=True)
    if pinned:
        source = Path(pinned) / "real6_corpus.json"
        shutil.copy2(source, target / source.name)
        provenance = dict(kind="pinned historical corpus", source=str(source))
    else:
        import datasets
        from ...datasets import csn_rows
        from .legacy import library
        if list(Path(cache).glob("csn/test-*.parquet")):
            import pyarrow.parquet as pq
            records = [row for file in sorted(Path(cache).glob("csn/test-*.parquet"))
                       for batch in pq.ParquetFile(file).iter_batches(batch_size=1024) for row in batch.to_pylist()]
        else:
            records = list(csn_rows(cache, "test"))
        native = datasets.load_dataset

        def loader(name, *args, **kwargs):
            if name in ("code_search_net", "code-search-net/code_search_net"):
                return iter(records)
            return native(name, *args, **kwargs)

        W = library("whyscale", run)
        W.CACHE = str(target)
        if n:
            with patch.object(datasets, "load_dataset", loader):
                W._real6_corpus(n)
        else:
            # Retain the original source filters without its finite scan/sample caps.
            recs = []
            for row in records:
                code = row["func_code_string"]
                if not 120 < len(code) < 2000:
                    continue
                parsed = W._real6_parse(code)
                if parsed is None:
                    continue
                src, names, others, ints = parsed
                if len(names) < 3 or ints < 1:
                    continue
                if max(map(len, names)) > W.REAL6_LMAX:
                    continue
                recs.append(dict(src=src, names=names, others=others, ints=ints))
            write_json(
                target / "real6_corpus.json",
                dict(
                    n_items=len(recs),
                    min_vars=3,
                    lmax=W.REAL6_LMAX,
                    scanned=len(records),
                    recs=recs,
                    full=True,
                    n_requested=0,
                ),
            )
        provenance = dict(
            kind="fresh model-independent CodeSearchNet test corpus",
            source="code-search-net/code_search_net/python/test",
            not_historical_numeric_reproduction=True,
        )
    corpus = target / "real6_corpus.json"
    provenance.update(
        sha256=digest(corpus),
        rows=len(json.loads(corpus.read_text())["recs"]),
        full=n == 0,
    )
    write_json(target / "reference_provenance.json", provenance)
    return provenance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--cache", default="", help="optional read-only source cache")
    p.add_argument(
        "--pinned-eval", help="use archived corpus for exact historical comparisons"
    )
    p.add_argument(
        "--n",
        type=int,
        default=0,
        help="row cap; default 0 means all available examples",
    )
    p.add_argument("--corpus-only", action="store_true")
    p.add_argument("--tasks", help="Comma-separated experiment tasks; default all ten")
    p.add_argument("--benchmark-run", help="Shared benchmark storage root; defaults to other_models/runs/benchmarks")
    p.add_argument("--benchmark-data", action="append", default=[], metavar="NAME=PATH", help="Reuse a prepared benchmark directory; repeat for multiple benchmarks")
    p.add_argument("--benchmark-source", action="append", default=[], metavar="NAME=PATH", help="Reuse an existing raw benchmark download; repeat for multiple benchmarks")
    a = p.parse_args(argv)
    if a.n < 0:
        p.error("--n must be nonnegative; 0 means all examples")
    from .registry import builtins, TASKS
    from .benchmarks import prepare, SHARED_BENCHMARKS
    builtins()
    tasks = a.tasks.split(",") if a.tasks else sorted(TASKS)
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - TASKS.keys():
        p.error("--tasks requires unique registered task names")

    def mapping(values):
        result = {}
        for value in values:
            name, separator, path = value.partition("=")
            if not separator or name not in SHARED_BENCHMARKS.values() or not path or name in result:
                p.error("Benchmark paths must be unique NAME=PATH entries for a shared CodeXGLUE benchmark")
            result[name] = path
        return result

    prepared, sources = mapping(a.benchmark_data), mapping(a.benchmark_source)
    if prepared.keys() & sources.keys():
        p.error("Select a prepared directory or a raw source for each benchmark, not both")
    run = setup_run(a.run, vars(a))
    provenance = None
    if a.corpus_only or set(tasks) & {"identifier_reproduction", "variable_misuse_localization"}:
        cache = a.cache
        if not cache and not a.pinned_eval:
            from ...source_downloads import MANIFEST, download_dataset, verify_source
            root = Path(__file__).resolve().parents[2]
            for marker in sorted((root / "runs").glob(f"**/sources/*/{MANIFEST}")):
                path = marker.parent
                if list(path.glob("python/test-*.parquet")) or list(path.glob("csn/test-*.parquet")):
                    meta = json.loads(marker.read_text(encoding="utf-8"))
                    verify_source(path, meta["name"], require_full=False)
                    cache = str(path)
                    break
            if not cache:
                cache = str(download_dataset("codesearchnet", run / "reference-source", allow_patterns=["python/test-*.parquet"]))
        provenance = prepare_reference(run, cache, a.n, a.pinned_eval)
    if a.corpus_only:
        result = dict(n_requested=a.n, full=a.n == 0, reference=provenance, corpus_only=True)
    else:
        result = prepare(run, a.cache, a.n, a.benchmark_run, prepared, sources, tasks)
        if provenance:
            result["reference"] = provenance
    write_json(run / "data/manifest.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
