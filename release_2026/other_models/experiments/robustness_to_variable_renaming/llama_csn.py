"""Llama/CSN curve pipeline: prepare-views, evaluate raw predictions, then pure score."""

import argparse, json, os
from pathlib import Path
from ...runtime import setup_run, read_jsonl, write_jsonl, write_json, digest
from .curves import (
    prepare_views,
    predict_curves,
    score_curves,
    paper_curves,
)
from .registry import builtins, TASKS
from .evaluation import PAPER, evaluate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare-views", "evaluate", "score"])
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--model")
    p.add_argument("--views")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="row cap; default 0 means all prepared examples",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks")
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    a = p.parse_args(argv)
    if a.limit < 0:
        p.error("--limit must be nonnegative")
    builtins()
    tasks = a.tasks.split(",") if a.tasks else sorted(TASKS)
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - TASKS.keys():
        p.error("--tasks requires unique registered task names")
    if a.max_len < 2 or a.batch < 1:
        p.error("--max-len must be at least 2 and --batch must be positive")
    if a.action == "prepare-views":
        m = prepare_views(a.data, a.run, a.limit, tasks)
        if os.getenv("ORX_METRICS_PATH"):
            write_json(os.environ["ORX_METRICS_PATH"], m)
        return
    if a.action == "score":
        run = Path(a.run)
        manifest = json.loads((run / "results/prediction_manifest.json").read_text())
        result = {}
        for task in tasks:
            if task in PAPER:
                result[task] = paper_curves(
                    task, json.loads((run / "results" / (task + ".json")).read_text())
                )
            else:
                path = run / "results" / (task + ".curves.predictions.jsonl")
                if digest(path) != manifest[task]["prediction_sha256"]:
                    raise ValueError(f"Prediction hash changed: {task}")
                result[task] = score_curves(
                    read_jsonl(path)
                )
        write_json(run / "results/curves.json", result)
        if os.getenv("ORX_METRICS_PATH"):
            write_json(os.environ["ORX_METRICS_PATH"], result)
        return result
    if not a.model:
        p.error("evaluate requires --model")
    if set(tasks) - PAPER and not a.views:
        p.error("evaluate requires --views for non-paper tasks")
    a.data = str(Path(a.data).resolve())
    if a.views:
        a.views = str(Path(a.views).resolve())
    if Path(a.model).exists():
        a.model = str(Path(a.model).resolve())
    run = setup_run(a.run, vars(a))
    from types import SimpleNamespace

    paper = [t for t in tasks if t in PAPER]
    if paper:
        evaluate(
            SimpleNamespace(
                tasks=",".join(paper),
                data=a.data,
                model=a.model,
                limit=a.limit,
                max_len=a.max_len,
                device=a.device,
                dtype=a.dtype,
                batch=a.batch,
            ),
            run,
        )
    from ...models import HFModel

    model = HFModel(a.model, a.device, a.dtype, a.batch) if set(tasks) - PAPER else None
    if model is not None:
        model.context = min(model.context, a.max_len)
    output = {}
    for task in tasks:
        if task in PAPER:
            continue
        options = {"max_len": a.max_len, "limit": a.limit}
        path = Path(a.views) / (task + ".jsonl")
        if task.startswith("cubert_") and task != "cubert_exception":
            options["dev"] = list(read_jsonl(Path(a.views) / (task + "_dev.jsonl")))[
                : a.limit or None
            ]
        predictions = predict_curves(model, task, path, options)
        dest = run / "results" / (task + ".curves.predictions.jsonl")
        write_jsonl(dest, predictions)
        output[task] = dict(
            predictions=len(predictions),
            prediction_sha256=digest(dest),
            views_sha256=digest(path),
        )
        print(task, output[task], flush=True)
    write_json(run / "results/prediction_manifest.json", output)
    if os.getenv("ORX_METRICS_PATH"):
        write_json(os.environ["ORX_METRICS_PATH"], output)
    return output


if __name__ == "__main__":
    main()
