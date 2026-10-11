"""Pure metrics: frozen independently of prediction/model code."""

import math


def choice(scores, gold):
    if not scores or not 0 <= gold < len(scores):
        raise ValueError("invalid choices")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("nonfinite score")
    return bool(all(scores[gold] > s for i, s in enumerate(scores) if i != gold))


def fraction(values):
    values = list(values)
    k = sum(bool(v) for v in values)
    n = len(values)
    return dict(correct=k, count=n, accuracy=k / n if n else None)


def edit_similarity(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b), 1)


def score_predictions(rows):
    rows = list(rows)
    result = fraction(r["correct"] for r in rows)
    if rows and "edit_similarity" in rows[0]:
        result["edit_similarity"] = sum(r["edit_similarity"] for r in rows) / len(rows)
    return result
