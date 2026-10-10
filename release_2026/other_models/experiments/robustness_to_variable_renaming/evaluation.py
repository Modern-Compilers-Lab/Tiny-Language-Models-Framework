import json, os, shutil, subprocess, sys
from pathlib import Path
from ...models import HFModel
from ...runtime import read_jsonl, write_jsonl, write_json, digest
from .registry import TASKS
from .tasks.scoring import score_predictions

PAPER = {"identifier_reproduction", "variable_misuse_localization", "cloze_all_paper"}


def evaluate(args, run):
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    tasks = args.tasks.split(",")
    unknown = set(tasks) - TASKS.keys()
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    data = Path(args.data)
    all_results = {}
    model = None
    for name in tasks:
        if name in PAPER:
            target = run / "eval_corpus"
            target.mkdir(exist_ok=True)
            for file in (data.parent / "eval_corpus").glob("*.json"):
                shutil.copy2(file, target / file.name)
            env = dict(
                os.environ,
                PYTHONHASHSEED="0",
                WHYSCALE_CACHE=str(target),
                RENAMING_WORK=str(run),
                PYTHONPATH=str(Path(__file__).resolve().parents[3]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            )
            command = [
                sys.executable,
                "-m",
                __package__ + ".tasks.paper",
                "--model",
                args.model,
                "--task",
                name,
                "--run",
                str(run),
                "--n",
                str(args.limit),
                "--max-len",
                str(args.max_len),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--batch",
                str(args.batch),
                "--n-sample",
                str(json.loads((data / "manifest.json").read_text())["n_requested"]),
            ]
            # Release resident model before worker acquires GPU memory.
            if model is not None:
                del model
                model = None
                import gc, torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            with open(run / "logs" / (name + ".log"), "w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            result = json.loads((run / "results" / (name + ".json")).read_text())
        else:
            filename = name
            if name == "cloze_all_multitoken":
                filename = "cloze_all"
            if name.startswith("cubert_"):
                filename += "_eval"
            path = data / (filename + ".jsonl")
            if not path.exists():
                raise FileNotFoundError(path)
            rows = list(read_jsonl(path))
            rows = rows[: args.limit] if args.limit else rows
            if not rows:
                result = {
                    "status": "unavailable",
                    "reason": "no labeled input rows",
                    "accuracy": None,
                    "count": 0,
                }
            else:
                if model is None:
                    model = HFModel(args.model, args.device, args.dtype, args.batch)
                    model.context = min(model.context, args.max_len)
                options = {"max_len": args.max_len}
                if name.startswith("cubert_") and name != "cubert_exception":
                    options["dev"] = list(read_jsonl(data / (name + "_dev.jsonl")))[
                        : args.limit or None
                    ]
                predictions = TASKS[name](model, rows, options)
                write_jsonl(
                    run / "results" / (name + ".predictions.jsonl"), predictions
                )
                result = score_predictions(predictions)
                result["data_sha256"] = digest(path)
            write_json(run / "results" / (name + ".json"), result)
        all_results[name] = result
        print(name, {k: v for k, v in result.items() if k != "raw"}, flush=True)
    write_json(run / "results" / "metrics.json", all_results)
    if os.getenv("ORX_METRICS_PATH"):
        write_json(
            os.environ["ORX_METRICS_PATH"],
            {
                k: {a: b for a, b in v.items() if a != "raw"}
                for k, v in all_results.items()
            },
        )
    return all_results
