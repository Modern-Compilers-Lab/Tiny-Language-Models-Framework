"""Decoder-only prefix conditional candidate ranking (supports multi-token answers)."""

from ..registry import TASKS
from .scoring import choice


def cloze(model, rows, options):
    out = []
    for r in rows:
        scores = model.continuation_scores(r["prefix"], r["choices"])
        gold = r["choices"].index(r["gold"])
        out.append(
            dict(
                id=r["id"],
                gold=gold,
                scores=scores,
                prediction=max(range(len(scores)), key=scores.__getitem__),
                correct=choice(scores, gold),
            )
        )
    return out


TASKS.add("cloze_maxmin", cloze)
TASKS.add("cloze_all_multitoken", cloze)
