"""NL-free generative adaptation of CuBERT classification. Calibrate on DEV only."""

from .scoring import fraction
import math


def threshold(dev_scores, dev_labels):
    if len(dev_scores) != len(dev_labels) or not dev_scores:
        raise ValueError("invalid dev set")
    if not all(math.isfinite(s) for s in dev_scores):
        raise ValueError("nonfinite calibration score")
    values = sorted(set(dev_scores))
    candidates = (
        [values[0] - 1]
        + [(a + b) / 2 for a, b in zip(values, values[1:])]
        + [values[-1] + 1]
    )
    # fixed orientation: high likelihood = unmodified. Ties choose smallest threshold.
    return max(
        candidates,
        key=lambda t: sum((s >= t) == y for s, y in zip(dev_scores, dev_labels)),
    )


def classify(model, rows, options):
    dev = options["dev"]
    maxlen = options.get("max_len", 320)
    labels = [r["label"] == "Correct" for r in dev]
    if len(set(labels)) < 2:
        raise ValueError("calibration requires both dev classes")
    cut = (
        options["threshold"]
        if "threshold" in options
        else threshold(model.sequence_scores([r["code"] for r in dev], maxlen), labels)
    )
    scores = model.sequence_scores([r["code"] for r in rows], maxlen)
    if len(scores) != len(rows) or not all(math.isfinite(s) for s in scores):
        raise ValueError("invalid prediction scores")
    return [
        dict(
            id=r["id"],
            gold=r["label"] == "Correct",
            prediction=s >= cut,
            score=s,
            threshold=cut,
            calibration_n=len(dev),
            correct=(s >= cut) == (r["label"] == "Correct"),
        )
        for r, s in zip(rows, scores)
    ]
