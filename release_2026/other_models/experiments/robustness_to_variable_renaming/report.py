"""Export verified prediction curves to JSON and CSV, with optional seed aggregates."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from .curves import NS, paper_curves, score_curves
from .evaluation import PAPER
from ...runtime import digest, read_jsonl, write_json


def verify_results(run):
    run = Path(run)
    curves = json.loads((run / "results/curves.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "results/prediction_manifest.json").read_text(encoding="utf-8"))
    for task, curve in curves.items():
        if task in PAPER:
            actual = paper_curves(task, json.loads((run / "results" / f"{task}.json").read_text(encoding="utf-8")))
        else:
            path = run / "results" / f"{task}.curves.predictions.jsonl"
            if digest(path) != manifest[task]["prediction_sha256"]:
                raise ValueError(f"Prediction hash changed: {task}")
            actual = score_curves(read_jsonl(path))
        if actual != curve:
            raise ValueError(f"Scored curves differ from predictions: {task}")
    return curves


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="One evaluation run, or a root containing arm/seed runs")
    parser.add_argument("--out", required=True, help="Output directory for report.json and curves.csv")
    args = parser.parse_args(argv)
    base, out = Path(args.run).resolve(), Path(args.out).resolve()
    runs = [base] if (base / "results/curves.json").exists() else sorted(p.parent.parent for p in base.rglob("results/curves.json"))
    if not runs:
        raise ValueError("No scored experiment runs found")
    rows = []
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        training = {}
        if config.get("model"):
            recipe = Path(config["model"]).parent / "config.json"
            if recipe.exists():
                training = json.loads(recipe.read_text(encoding="utf-8"))
        for task, curve in verify_results(run).items():
            for n in NS:
                rows.append(dict(run=str(run), arch=training.get("arch"), dataset=training.get("dataset"), mode=training.get("mode", config.get("mode")), seed=training.get("seed", config.get("seed")), max_len=config.get("max_len"), task=task, N=n, **curve[str(n)]))
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        # Unlabelled runs remain individual rather than mixing unrelated checkpoints.
        condition = row["mode"] or row["run"]
        groups.setdefault((row["arch"], row["dataset"], condition, row["task"], row["N"], row["max_len"]), []).append(row)
    aggregate = []
    for (arch, dataset, condition, task, n, max_len), group in groups.items():
        item = dict(arch=arch, dataset=dataset, condition=condition, task=task, N=n, max_len=max_len, runs=len(group))
        for metric in ("accuracy", "retention"):
            values = [r[metric] for r in group if r[metric] is not None]
            item[metric + "_mean"] = statistics.mean(values) if values else None
            item[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            item[metric + "_defined_runs"] = len(values)
        aggregate.append(item)
    result = dict(verified_runs=[str(run) for run in runs], rows=rows, aggregates=aggregate)
    write_json(out / "report.json", result)
    print(json.dumps(dict(runs=len(runs), rows=len(rows), output=str(out)), indent=2))
    return result


if __name__ == "__main__":
    main()
