"""Aggregate mechanistic runs by training condition (mean and sample SD over seeds).

    python report.py --runs RUN_A RUN_B ... --out REPORT_DIR

Each run must contain the four mechanistic results (run.py). Runs are grouped by
their --arm label; a duplicate (arm, training seed) pair is rejected. Writes
summary.json, summary.csv and curves.csv to REPORT_DIR.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import summarize  # noqa: E402

MODULES = ("copyscore", "meanablate", "force", "firsterror")


def aggregate(values):
    v = [float(x) for x in values if x is not None]
    if any(not math.isfinite(x) for x in v):
        raise ValueError("Non-finite report input")
    return dict(
        mean=statistics.mean(v) if v else None,
        sd=statistics.stdev(v) if len(v) > 1 else None,
        n_seeds=len(v),
    )


def load(path):
    run = Path(path)
    row = dict(run=str(run), config=json.loads((run / "config.json").read_text()))
    for name in MODULES:
        row[name] = json.loads((run / "results" / (name + ".json")).read_text())
    row["metrics"] = summarize(run)
    return row


def report(paths, out):
    rows = [load(p) for p in paths]
    if not rows:
        raise ValueError("No runs given")
    seen = [(r["config"]["arm"], r["config"]["training_seed"]) for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError("Duplicate arm and training seed")
    arms = sorted({r["config"]["arm"] for r in rows})
    groups = {arm: [r for r in rows if r["config"]["arm"] == arm] for arm in arms}
    fields = sorted(set.intersection(*(set(r["metrics"]) for r in rows)))
    stats = {
        arm: {f: aggregate([r["metrics"][f] for r in g]) for f in fields}
        for arm, g in groups.items()
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False) + "\n"
    )
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "metric", "mean", "sample_sd", "n_seeds"])
        for arm, ss in stats.items():
            for key, v in ss.items():
                w.writerow([arm, key, v["mean"], v["sd"], v["n_seeds"]])
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["experiment", "arm", "k", "measure", "mean", "sample_sd", "n_seeds"]
        )
        for exp in ("copyscore", "meanablate"):
            for arm, g in groups.items():
                for k in sorted({x["k"] for r in g for x in r[exp]["curve"]}):
                    for field in (
                        "acc_id",
                        "acc_ood",
                        "random_id_mean",
                        "random_ood_mean",
                    ):
                        values = [
                            x[field]
                            for r in g
                            for x in r[exp]["curve"]
                            if x["k"] == k and field in x
                        ]
                        if values:
                            v = aggregate(values)
                            w.writerow(
                                [exp, arm, k, field, v["mean"], v["sd"], v["n_seeds"]]
                            )
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(report(a.runs, a.out), indent=2))
