"""Optional checkpoint fine-tuning and benchmark-specific evaluation."""

import hashlib
import json
import math
import sqlite3
import tempfile
from pathlib import Path

from ..runtime import digest, read_jsonl, setup_run, write_json, write_jsonl
from . import BENCHMARKS
from .data import take, validate_row
from .execution import crux_success
from .metrics import (
    MetricAccumulator,
    crux_metrics,
    repair_metrics,
    repair_result,
    type_metrics,
)
from .utils import fraction


def prompt(row, name):
    if name == "bifi":
        return (
            "# Repair the Python token sequence. Return only the complete repaired token sequence.\n# Bad:\n"
            + row["buggy_tokens"]
            + "\n# Fixed:\n"
        )
    if name == "manytypes4py":
        return (
            row["code"] + f"\n# Type of {row['kind']} {row['scope']}:{row['target']}: "
        )
    if name == "cruxeval_input":
        return (
            row["code"]
            + f"\n# Find a function call producing {row['output']}. Return only f(literal arguments).\n"
        )
    if name == "cruxeval_output":
        return row["code"] + f"\nassert f({row['input']}) == "
    raise ValueError(name)


def manifest(data):
    data = Path(data)
    meta = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    name = meta["benchmark"]
    if name not in BENCHMARKS:
        raise ValueError("unknown benchmark manifest")
    for split, info in meta["splits"].items():
        if split not in ("train", "dev", "test"):
            raise ValueError("unknown split")
        if digest(data / (split + ".jsonl")) != info["sha256"]:
            raise ValueError(f"{split} data changed after preparation")
    if (
        meta.get("type_vocabulary_sha256")
        and digest(data / "type_vocabulary.json") != meta["type_vocabulary_sha256"]
    ):
        raise ValueError("Type vocabulary changed after preparation")
    return meta


def ensure_data(name, run, limit=0, prepared=None, source=None, search_root=None):
    """Reuse validated prepared/raw releases before downloading through shared readers.

    Explicit paths fail on invalid data. Automatic discovery never serves a sampled
    preparation for an all-example request, or a smaller sample for a larger request.
    """
    if name not in BENCHMARKS or limit < 0:
        raise ValueError("Invalid benchmark or limit")
    run = Path(run).resolve()

    def eligible(path, explicit=False):
        meta = manifest(path)
        if meta["benchmark"] != name or not meta["splits"].get("test", {}).get("count"):
            raise ValueError(f"Prepared data does not contain {name} TEST rows")
        if limit == 0 and (meta.get("subset", True) or not meta.get("full", False)):
            if explicit:
                raise ValueError("Full evaluation cannot reuse a prepared subset; set a positive limit or supply full data")
            return False
        return explicit or limit == 0 or meta["splits"]["test"]["count"] >= limit

    if prepared:
        path = Path(prepared).resolve()
        eligible(path, explicit=True)
        return path
    search = Path(search_root).resolve() if search_root else run
    candidates = [] if source else [run / "data" / name]
    if search.exists() and not source:
        candidates.extend(p.parent for p in sorted(search.glob(f"**/data/{name}/manifest.json")))
    for path in dict.fromkeys(candidates):
        if not (path / "manifest.json").exists():
            continue
        try:
            if eligible(path):
                return path.resolve()
        except (ValueError, KeyError, FileNotFoundError):
            # An automatic cache candidate is optional; explicit inputs fail above.
            continue
    raw = Path(source).resolve() if source else None
    if raw is None:
        from ..source_downloads import MANIFEST, verify_source

        sources = [run / "sources" / name]
        if search.exists():
            sources.extend(p.parent for p in sorted(search.glob(f"**/sources/{name}/{MANIFEST}")))
        for path in dict.fromkeys(sources):
            if not (path / MANIFEST).exists():
                continue
            try:
                verify_source(path, name, "http_files")
                raw = path.resolve()
                break
            except (ValueError, KeyError, FileNotFoundError):
                continue
    if raw is None and limit == 0:
        from .sources import download

        raw = download(name, run)
    from .data import prepare

    # Separate preparation sizes while sharing one raw source store.
    target = run / "prepared" / ("full" if limit == 0 else f"subset-{limit}")
    path = target / "data" / name
    if (path / "manifest.json").exists():
        cached = manifest(path)
        if source and (cached.get("raw_source") or cached.get("source")) != str(raw):
            raise ValueError("Benchmark source changed; choose a fresh benchmark run")
        if not eligible(path):
            raise ValueError("Cached preparation has insufficient rows")
        return path
    path = prepare(name, target, source=str(raw) if raw else None, limit=limit)
    eligible(path, explicit=True)
    return path


def training_data(name, rows):
    from .utils import render

    for r in rows:
        validate_row(name, r, "train")
        if name == "codexglue_py150":
            yield dict(id=r["id"], code=render(r["code"])[0], split="train")
        else:
            yield dict(
                id=r["id"],
                split="train",
                prompt=prompt(r, name),
                target=r["fixed_tokens"] if name == "bifi" else r["gold"],
            )


def finetune(data, model, config, run):
    from ..training import train, validate_config

    data = Path(data)
    meta = manifest(data)
    name = meta["benchmark"]
    if not BENCHMARKS[name].trainable:
        raise ValueError(
            f"{name} has no official training split; fine-tuning on evaluation data is forbidden"
        )
    if "train" not in meta["splits"]:
        raise ValueError(
            "Prepare this benchmark with --include-train before fine-tuning"
        )
    config = dict(config)
    forbidden = {
        "data",
        "pretrained",
        "tokenizer",
        "supervised",
        "arch",
        "model",
        "benchmark",
    } & config.keys()
    if forbidden:
        raise ValueError(f"Fine-tuning config cannot override {sorted(forbidden)}")
    if "steps" not in config:
        raise ValueError("Set an explicit fine-tuning steps budget")
    config.setdefault("device", "cpu")
    config.setdefault("warmup", 0)
    config.setdefault("lr", 5e-5)
    config.setdefault("batch_size", 2)
    validate_config(config)
    run = Path(run).resolve()
    local_checkpoint = Path(model)
    if local_checkpoint.is_dir() and (
        run.is_relative_to(local_checkpoint.resolve())
        or (run / "model").resolve() == local_checkpoint.resolve()
    ):
        raise ValueError("Fine-tuning requires a separate output checkpoint directory")
    if (run / "model/config.json").exists() or (run / "results/training.json").exists():
        raise ValueError(
            "Fine-tuning output already exists; choose a new run directory"
        )
    # Check full TRAIN against heldout splits without holding the corpus in RAM.
    with tempfile.TemporaryDirectory(prefix="benchmark-overlap-") as directory:
        connection = sqlite3.connect(str(Path(directory) / "overlap.sqlite"))
        try:
            connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE inputs (hash TEXT PRIMARY KEY)")

            def input_hash(row):
                value = row.get(
                    "buggy_tokens", json.dumps(row.get("code"), sort_keys=True)
                )
                return hashlib.sha256(value.encode("utf-8")).hexdigest()

            for row in read_jsonl(data / "train.jsonl"):
                validate_row(name, row, "train")
                connection.execute(
                    "INSERT OR IGNORE INTO ids VALUES (?)", (str(row["id"]),)
                )
                connection.execute(
                    "INSERT OR IGNORE INTO inputs VALUES (?)", (input_hash(row),)
                )
            for split in ("dev", "test"):
                if split not in meta["splits"]:
                    continue
                for row in read_jsonl(data / (split + ".jsonl")):
                    if connection.execute(
                        "SELECT 1 FROM ids WHERE id=?", (str(row["id"]),)
                    ).fetchone():
                        raise ValueError("Train/heldout ID overlap")
                    if connection.execute(
                        "SELECT 1 FROM inputs WHERE hash=?", (input_hash(row),)
                    ).fetchone():
                        raise ValueError("Train/heldout input overlap")
        finally:
            connection.close()
    from transformers import AutoConfig

    config.update(
        arch=AutoConfig.from_pretrained(model).model_type,
        pretrained=str(model),
        tokenizer=str(model),
        supervised=name != "codexglue_py150",
        mask_eos_like_pilot=False,
        benchmark=name,
        benchmark_manifest_sha256=digest(data / "manifest.json"),
    )
    if local_checkpoint.is_dir():
        config["source_checkpoint_hashes"] = {
            p.name: digest(p)
            for p in sorted(local_checkpoint.iterdir())
            if p.suffix in (".safetensors", ".bin", ".json")
        }
    config["data"] = str(Path(run).resolve() / "data" / "finetune.jsonl")
    run = setup_run(run, config)
    write_jsonl(config["data"], training_data(name, read_jsonl(data / "train.jsonl")))
    train(config, run)
    return run / "model"


def predict(name, model, rows, options):
    out = []
    max_new = options.get("max_new_tokens", 128)
    if name == "codexglue_py150":
        for row in rows:
            for result in model.lexical_predictions(
                row["code"], options.get("max_len", 320)
            ):
                out.append(dict(result, id=f"{row['id']}:{result['index']}"))
    elif name.startswith("codexglue_cloze"):
        for row in rows:
            scores = model.continuation_scores(row["prefix"], row["choices"])
            if (
                len(scores) != len(row["choices"])
                or not scores
                or not all(math.isfinite(s) for s in scores)
            ):
                raise ValueError("Invalid candidate scores")
            # Match upstream argmax semantics: first candidate wins a tie.
            winner = max(range(len(scores)), key=scores.__getitem__)
            answer = row["choices"][winner]
            out.append(
                dict(
                    id=row["id"],
                    prediction=answer,
                    gold=row["gold"],
                    correct=answer == row["gold"],
                )
            )
    elif name == "bifi":
        for row in rows:
            generated = model.sample(prompt(row, name), max_new_tokens=max_new)[0]
            out.append(repair_result(row, generated))
    elif name == "manytypes4py":
        vocabulary = options["type_vocabulary"]
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("type vocabulary must be nonempty and unique")
        for row in rows:
            scores = model.continuation_scores(prompt(row, name), vocabulary)
            if len(scores) != len(vocabulary) or not all(
                math.isfinite(s) for s in scores
            ):
                raise ValueError("Invalid type scores")
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:10]
            out.append(
                dict(
                    id=row["id"],
                    kind=row["kind"],
                    gold=row["gold"],
                    predictions=[vocabulary[i] for i in ranked],
                )
            )
    else:
        mode = name.removeprefix("cruxeval_")
        for row in rows:
            generated = model.sample(
                prompt(row, name),
                n=options.get("samples", 1),
                max_new_tokens=max_new,
                temperature=options.get("temperature", 0.0),
                top_p=options.get("top_p", 0.95),
            )
            if len(generated) != options.get("samples", 1):
                raise ValueError("Incorrect generation count")
            successes = [
                crux_success(row, g, mode, options.get("timeout", 3)) for g in generated
            ]
            out.append(dict(id=row["id"], generations=generated, successes=successes))
    return out


def metrics(name, rows):
    if name == "bifi":
        return repair_metrics(rows)
    if name == "manytypes4py":
        return type_metrics(rows)
    if name.startswith("cruxeval_"):
        return crux_metrics(rows)
    result = fraction(r["correct"] for r in rows)
    if name == "codexglue_py150":
        result["token_accuracy"] = result.pop("accuracy")
    return result


def evaluate(args, run):
    from ..models import HFModel

    data = Path(args.data)
    meta = manifest(data)
    name = meta["benchmark"]
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.samples < 1 or args.temperature < 0:
        raise ValueError("invalid generation settings")
    if (
        args.max_len < 2
        or args.batch < 1
        or args.max_new_tokens < 1
        or args.timeout <= 0
    ):
        raise ValueError("Invalid context, batch size, generation length, or timeout")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if args.samples > 1 and args.temperature == 0:
        raise ValueError("Multiple samples require --temperature > 0")
    options = dict(
        max_len=args.max_len,
        max_new_tokens=args.max_new_tokens,
        samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
    )
    if name == "manytypes4py":
        vocab = (
            Path(args.type_vocabulary)
            if args.type_vocabulary
            else data / "type_vocabulary.json"
        )
        if not vocab.exists():
            raise ValueError(
                "Provide a frozen --type-vocabulary or prepare --include-train; test labels cannot define candidates"
            )
        options["type_vocabulary"] = json.loads(vocab.read_text(encoding="utf-8"))
        options["type_vocabulary_sha256"] = digest(vocab)
    model_path = args.model
    if args.finetune_config:
        model_path = finetune(
            data,
            model_path,
            json.loads(Path(args.finetune_config).read_text(encoding="utf-8")),
            Path(run) / "finetune",
        )
    if not meta["splits"].get("test", {}).get("count"):
        raise ValueError("Empty evaluation split")
    import torch

    torch.manual_seed(args.seed)
    model = HFModel(model_path, args.device, args.dtype, args.batch)
    model.context = min(model.context, args.max_len)
    accumulator = MetricAccumulator(name)
    evaluated_examples = 0

    def predictions():
        nonlocal evaluated_examples
        for row in take(read_jsonl(data / "test.jsonl"), args.limit):
            validate_row(name, row, "test")
            evaluated_examples += 1
            for prediction in predict(name, model, [row], options):
                accumulator.add(prediction)
                yield prediction

    write_jsonl(Path(run) / "results" / (name + ".predictions.jsonl"), predictions())
    if not evaluated_examples:
        raise ValueError("Empty evaluation split")
    result = accumulator.result()
    result.update(
        benchmark=name,
        model=str(model_path),
        finetuned=bool(args.finetune_config),
        protocol=BENCHMARKS[name].protocol,
        source=meta["source"],
        units="fraction",
        evaluated_examples=evaluated_examples,
        subset=meta["subset"] or evaluated_examples < meta["splits"]["test"]["count"],
        data_sha256=digest(data / "test.jsonl"),
        generation=dict(
            samples=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        ),
        context=model.context,
    )
    result["benchmark_manifest_sha256"] = digest(data / "manifest.json")
    result["model_revision"] = getattr(model.causal_lm.config, "_commit_hash", None)
    if Path(model_path).is_dir():
        result["checkpoint_hashes"] = {
            p.name: digest(p)
            for p in sorted(Path(model_path).iterdir())
            if p.suffix in (".safetensors", ".bin", ".json")
        }
    if "type_vocabulary_sha256" in options:
        result["type_vocabulary_sha256"] = options["type_vocabulary_sha256"]
    write_json(Path(run) / "results" / "metrics.json", result)
    print(json.dumps(result, indent=2))
    return result


def add_cli(sub):
    parser = sub.add_parser(
        "benchmark",
        help="standard small-model benchmark preparation, fine-tuning, evaluation",
    )
    actions = parser.add_subparsers(dest="benchmark_action", required=True)
    actions.add_parser("list")
    raw = actions.add_parser(
        "download", help="download complete raw benchmark releases"
    )
    raw.add_argument("--benchmark", required=True, choices=[*sorted(BENCHMARKS), "all"])
    raw.add_argument("--run", required=True)
    raw.add_argument(
        "--max-download-mb",
        type=int,
        default=0,
        help="Explicit aggregate transfer cap in MiB; default 0 is unlimited",
    )
    prep = actions.add_parser("prepare")
    prep.add_argument(
        "--benchmark", required=True, choices=[*sorted(BENCHMARKS), "all"]
    )
    prep.add_argument("--run", required=True)
    prep.add_argument("--source")
    counts = prep.add_mutually_exclusive_group()
    counts.add_argument(
        "--limit",
        type=int,
        help="per-split row cap; default 0 downloads complete releases",
    )
    counts.add_argument(
        "--full",
        action="store_true",
        default=None,
        help="complete raw sources and all available splits; no row/byte cap unless explicitly set",
    )
    splits = prep.add_mutually_exclusive_group()
    splits.add_argument("--include-train", action="store_true", default=None)
    splits.add_argument("--test-only", action="store_false", dest="include_train")
    prep.add_argument(
        "--max-download-mb",
        type=int,
        help="streamed download cap in MiB; 0 disables it; default unlimited (64 with a positive --limit)",
    )
    prep.add_argument("--version", choices=["clean", "complete"], default="clean")
    ft = actions.add_parser("finetune")
    ft.add_argument("--data", required=True)
    ft.add_argument("--model", required=True)
    ft.add_argument("--config", required=True)
    ft.add_argument("--run", required=True)
    ev = actions.add_parser(
        "evaluate",
        help="evaluate all prepared TEST examples, optionally after fine-tuning",
        description="Evaluate all prepared TEST examples by default. Use --limit N for an explicit subset.",
    )
    ev.add_argument("--data", required=True, help="prepared benchmark directory")
    ev.add_argument(
        "--model", required=True, help="local Hugging Face checkpoint or model ID"
    )
    ev.add_argument("--run", required=True)
    ev.add_argument(
        "--finetune-config",
        help="optional fine-tuning recipe JSON; default evaluates supplied checkpoint",
    )
    ev.add_argument(
        "--device", default="auto", help="Evaluation device (default: auto selects CUDA when available, otherwise CPU)"
    )
    ev.add_argument("--dtype", default="float32")
    ev.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum evaluation examples; default 0 scores all prepared examples",
    )
    ev.add_argument("--max-len", type=int, default=1024)
    ev.add_argument("--batch", type=int, default=4)
    ev.add_argument("--max-new-tokens", type=int, default=128)
    ev.add_argument("--samples", type=int, default=1)
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--top-p", type=float, default=0.95)
    ev.add_argument("--seed", type=int, default=2)
    ev.add_argument("--timeout", type=float, default=3)
    ev.add_argument("--type-vocabulary")


def cli(args):
    if args.benchmark_action == "list":
        from dataclasses import asdict

        print(json.dumps({k: asdict(v) for k, v in BENCHMARKS.items()}, indent=2))
        return
    if args.benchmark_action == "finetune":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        print(finetune(args.data, args.model, config, args.run))
        return
    if args.benchmark_action == "prepare" and args.benchmark == "all" and args.source:
        raise ValueError("--source requires a single benchmark")
    if args.benchmark_action == "download" and args.max_download_mb < 0:
        raise ValueError("max_download_mb must be nonnegative")
    run = setup_run(args.run, vars(args))
    if args.benchmark_action == "download":
        from .downloads import Budget
        from .sources import download

        budget = Budget(args.max_download_mb << 20)
        for name in BENCHMARKS if args.benchmark == "all" else [args.benchmark]:
            print(download(name, run, budget))
    elif args.benchmark_action == "prepare":
        from .data import prepare

        for name in BENCHMARKS if args.benchmark == "all" else [args.benchmark]:
            print(
                prepare(
                    name,
                    run,
                    args.source,
                    args.limit,
                    args.include_train,
                    args.max_download_mb,
                    args.version,
                    args.full,
                )
            )
    else:
        evaluate(args, run)
