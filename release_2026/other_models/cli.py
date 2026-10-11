"""Download, prepare, train and load the three codelab model families."""

import argparse
import json
from pathlib import Path

from .models import ALIASES, MODEL_DEFAULTS, TOKENIZER, resolve_arch, resolve_dataset
from .source_downloads import REPOSITORIES, download_dataset

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from .benchmark_suite.workflow import add_cli

    add_cli(commands)
    sub = commands.add_parser("experiment", help="Codelab renaming experiments and campaign utilities")
    sub.add_argument("experiment_args", nargs=argparse.REMAINDER)
    commands.add_parser("list", help="Show model presets and training corpora")
    for command in ("download", "prepare"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", choices=[*REPOSITORIES, "all"])
        sub.add_argument("--arch", choices=[*MODEL_DEFAULTS, *ALIASES])
        sub.add_argument("--run", type=Path, default=ROOT / "runs/datasets")
        sub.add_argument("--revision", default="main")
        sub.add_argument(
            "--max-files",
            type=int,
            help="Download at most N training shards (default: entire repository)",
        )
        if command == "download":
            sub.add_argument(
                "--allow-pattern",
                action="append",
                help="Download matching files; repeat for multiple patterns",
            )
            sub.add_argument(
                "--dry-run",
                action="store_true",
                help="Preview pinned revision and download size",
            )
        else:
            sub.add_argument(
                "--cache",
                type=Path,
                help="Already downloaded dataset directory; no source download",
            )
            sub.add_argument(
                "--reference-cache",
                type=Path,
                help="CodeSearchNet source with the Python test split",
            )
            sub.add_argument(
                "--n-docs",
                type=int,
                default=0,
                help="Maximum eligible documents; 0 (default) prepares all",
            )
            sub.add_argument("--exclude-repo", action="append", default=[])
    sub = commands.add_parser("train")
    sub.add_argument("--arch", choices=[*MODEL_DEFAULTS, *ALIASES])
    sub.add_argument("--config", type=Path, help="JSON recipe; CLI options override it")
    sub.add_argument("--dataset", choices=list(REPOSITORIES))
    sub.add_argument("--data", type=Path, help="Prepared training JSONL")
    sub.add_argument("--data-run", type=Path, default=ROOT / "runs/datasets")
    sub.add_argument("--run", type=Path, help="Training output directory")
    sub.add_argument("--tokenizer")
    sub.add_argument("--pretrained", help="Continue training from an HF checkpoint")
    for key in (
        "steps",
        "batch-size",
        "micro-batch",
        "max-len",
        "warmup",
        "seed",
        "save-every",
        "threads",
    ):
        sub.add_argument(f"--{key}", type=int)
    sub.add_argument("--lr", type=float)
    sub.add_argument("--device", choices=["auto", "cpu", "cuda"])
    sub.add_argument("--precision", choices=["float32", "bfloat16"])
    sub = commands.add_parser(
        "generate", help="Load a trained checkpoint and generate code"
    )
    sub.add_argument("--model", required=True)
    sub.add_argument("--prompt", required=True)
    sub.add_argument("--max-new-tokens", type=int, default=32)
    sub.add_argument("--device", default="auto")
    sub.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "experiment":
            from .experiments.workflow import main as experiment_main

            return experiment_main(args.experiment_args)
        elif args.command == "benchmark":
            from .benchmark_suite.workflow import cli

            return cli(args)
        elif args.command == "list":
            print(
                json.dumps(
                    dict(
                        models=MODEL_DEFAULTS,
                        datasets=REPOSITORIES,
                        tokenizer=TOKENIZER,
                    ),
                    indent=2,
                )
            )
        elif args.command in ("download", "prepare"):
            if args.max_files is not None and args.max_files < 1:
                raise ValueError("max_files must be positive")
            if args.command == "prepare" and args.n_docs < 0:
                raise ValueError("n_docs must be nonnegative")
            names = (
                list(REPOSITORIES)
                if args.dataset == "all"
                else [resolve_dataset(args.arch, args.dataset)]
            )
            if args.command == "prepare" and len(names) > 1 and args.cache:
                raise ValueError("--cache requires a single dataset")
            results = {}
            for name in names:
                if args.command == "download":
                    value = download_dataset(
                        name,
                        args.run,
                        revision=args.revision,
                        dry_run=args.dry_run,
                        max_files=args.max_files,
                        allow_patterns=args.allow_pattern,
                    )
                else:
                    from .datasets import prepare

                    value = prepare(
                        name,
                        args.run,
                        cache=args.cache,
                        n_docs=args.n_docs,
                        reference_cache=args.reference_cache,
                        max_files=args.max_files,
                        revision=args.revision,
                        exclude_repos=args.exclude_repo,
                    )
                results[name] = str(value) if isinstance(value, Path) else value
            print(json.dumps(results, indent=2))
        elif args.command == "train":
            config = (
                json.loads(args.config.read_text(encoding="utf-8"))
                if args.config
                else {}
            )
            arch = resolve_arch(args.arch or config.get("arch"))
            filenames = {
                "gpt_bigcode": "starcoder-small",
                "llama": "llama-small",
                "gpt_neox": "gpt_neox-small",
            }
            recipe = json.loads(
                (ROOT / "configs" / f"{filenames[arch]}.json").read_text(
                    encoding="utf-8"
                )
            )
            recipe.update(config)
            recipe["arch"] = arch
            # Changing architecture also changes the default corpus unless explicitly selected.
            recipe["dataset"] = resolve_dataset(
                arch, args.dataset or config.get("dataset")
            )
            for key in (
                "steps",
                "batch_size",
                "micro_batch",
                "max_len",
                "warmup",
                "seed",
                "save_every",
                "threads",
                "lr",
                "device",
                "precision",
                "tokenizer",
                "pretrained",
            ):
                if getattr(args, key) is not None:
                    recipe[key] = getattr(args, key)
            from .training import train, validate_config

            validate_config(recipe)
            explicit_data = args.data or (
                config.get("data") if args.dataset is None else None
            )
            if explicit_data:
                data = Path(explicit_data)
                if not data.is_file():
                    raise FileNotFoundError(data)
            else:
                data = args.data_run / "data" / f"{recipe['dataset']}.jsonl"
                if not data.exists():
                    from .datasets import prepare

                    prepare(recipe["dataset"], args.data_run)
            recipe["data"] = str(data.resolve())
            report = train(recipe, args.run or ROOT / "runs" / arch)
            print(json.dumps(report, indent=2))
        else:
            import torch

            from .models import load_model

            if args.max_new_tokens < 1:
                raise ValueError("max_new_tokens must be positive")
            model, tokenizer = load_model(
                args.model, args.device, local_files_only=args.local_files_only
            )
            ids = tokenizer.encode(args.prompt, add_special_tokens=False) or [
                tokenizer.eos_token_id
            ]
            if len(ids) + args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Prompt plus generation exceeds model context")
            inputs = torch.tensor([ids], device=model.device)
            with torch.inference_mode():
                output = model.generate(
                    inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            print(tokenizer.decode(output[0, len(ids) :], skip_special_tokens=True))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
