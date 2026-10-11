"""Original benchmark metrics, expressed as fractions with explicit denominators."""

import ast
import math
import re

from .utils import fraction


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def paren_ok(tokens):
    # BIFI's critic counts bracket types independently (not a nesting stack).
    counts = [0, 0, 0]
    for token in tokens:
        if token in ("(", "{", "["):
            counts[("(", "{", "[").index(token)] += 1
        elif token in (")", "}", "]"):
            i = (")", "}", "]").index(token)
            counts[i] -= 1
            if counts[i] < 0:
                return False
    return counts == [0, 0, 0]


def bifi_code(tokens):
    """BIFI token serialization, including default anonymized literal restoration."""
    defaults = {"<unk>": "unk", "<COMMENT>": "##", "<NUMBER>": "1", "<STRING>": '"str"'}
    out = []
    indent = 0
    start = True
    for token in tokens:
        if token == "<INDENT>":
            indent += 1
            continue
        if token == "<DEDENT>":
            indent -= 1
            continue
        if token in ("<NEWLINE>", "<NL>"):
            out.append("\n")
            start = True
            continue
        if start:
            indent = max(0, indent)
            out.append("    " * indent)
            start = False
        else:
            out.append(" ")
        out.append(defaults.get(token, token))
    return "".join(out)


def repair_result(row, prediction):
    source = row["buggy_tokens"].split()
    target = prediction.strip().split()
    edits = distance(source, target)
    critic = False
    if target:
        if row["error_type"] == "unbalanced (){}[]":
            critic = paren_ok(target)
        else:
            try:
                ast.parse(bifi_code(target))
                critic = True
            except (SyntaxError, ValueError, RecursionError):
                pass
    return dict(
        id=row["id"],
        prediction=prediction,
        error_type=row["error_type"],
        token_edits=edits,
        critic_success=critic,
        correct=critic and 0 < edits <= 4,
    )


def repair_metrics(rows):
    result = fraction(r["correct"] for r in rows)
    result["repair_accuracy"] = result.pop("accuracy")
    result["by_error_type"] = {
        group: fraction(r["correct"] for r in rows if r["error_type"] == group)
        for group in sorted({r["error_type"] for r in rows})
    }
    return result


def parametric_match(gold, candidates):
    pattern = r"(.+)\[(.+)\]"
    match = re.match(pattern, gold)
    if not match:
        return gold in candidates
    outer = match.group(1)
    return any(
        p == gold
        or (re.match(pattern, p) and re.match(pattern, p).group(1) == outer)
        or (not re.match(pattern, p) and p.lower() == outer.lower())
        for p in candidates
    )


def type_metrics(rows):
    def group(items):
        n = len(items)
        result = dict(count=n)
        for k in (1, 5, 10):
            result[f"exact_match@{k}"] = (
                sum(r["gold"] in r["predictions"][:k] for r in items) / n if n else None
            )
        reciprocal = [
            next(
                (
                    1 / (i + 1)
                    for i, p in enumerate(r["predictions"][:10])
                    if p == r["gold"]
                ),
                0.0,
            )
            for r in items
        ]
        result["mrr@10"] = sum(reciprocal) / n if n else None
        result["parametric_match@10"] = (
            sum(parametric_match(r["gold"], r["predictions"][:10]) for r in items) / n
            if n
            else None
        )
        return result

    result = group(rows)
    result["by_kind"] = {
        kind: group([r for r in rows if r["kind"] == kind])
        for kind in ("Parameter", "Return", "Variable")
    }
    return result


def pass_at_k(n, c, k):
    if not 0 <= c <= n or k < 1:
        raise ValueError("invalid pass@k arguments")
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1 - math.prod(1 - k / i for i in range(n - c + 1, n + 1))


def crux_metrics(rows):
    result = dict(count=len(rows), samples=sum(len(r["successes"]) for r in rows))
    for k in (1, 5):
        values = [pass_at_k(len(r["successes"]), sum(r["successes"]), k) for r in rows]
        result[f"pass@{k}"] = (
            sum(values) / len(values)
            if values and all(v is not None for v in values)
            else None
        )
    return result


class MetricAccumulator:
    """Aggregate complete evaluations without retaining every prediction in RAM."""

    def __init__(self, name):
        self.name = name
        self.count = self.correct = self.samples = 0
        self.groups = {}
        self.totals = {}
        self.available = {1: True, 5: True}

    def add(self, row):
        self.count += 1
        if self.name.startswith("cruxeval_"):
            n, c = len(row["successes"]), sum(row["successes"])
            self.samples += n
            for k in (1, 5):
                value = pass_at_k(n, c, k)
                self.available[k] &= value is not None
                self.totals[k] = self.totals.get(k, 0) + (value or 0)
        elif self.name == "manytypes4py":
            values = type_metrics([row])
            group = self.groups.setdefault(row["kind"], {"count": 0})
            group["count"] += 1
            for key, value in values.items():
                if key in ("count", "by_kind"):
                    continue
                self.totals[key] = self.totals.get(key, 0) + value
                group[key] = group.get(key, 0) + value
        else:
            self.correct += bool(row["correct"])
            if self.name == "bifi":
                group = self.groups.setdefault(
                    row["error_type"], {"count": 0, "correct": 0}
                )
                group["count"] += 1
                group["correct"] += bool(row["correct"])

    def result(self):
        if self.name.startswith("cruxeval_"):
            return dict(
                count=self.count,
                samples=self.samples,
                **{
                    f"pass@{k}": self.totals.get(k, 0) / self.count
                    if self.count and self.available[k]
                    else None
                    for k in (1, 5)
                },
            )
        if self.name == "manytypes4py":
            keys = (
                "exact_match@1",
                "exact_match@5",
                "exact_match@10",
                "mrr@10",
                "parametric_match@10",
            )

            def values(group):
                n = group["count"]
                return dict(
                    count=n,
                    **{key: group.get(key, 0) / n if n else None for key in keys},
                )

            result = values(dict(self.totals, count=self.count))
            result["by_kind"] = {
                kind: values(self.groups.get(kind, {"count": 0}))
                for kind in ("Parameter", "Return", "Variable")
            }
            return result
        key = (
            "repair_accuracy"
            if self.name == "bifi"
            else "token_accuracy"
            if self.name == "codexglue_py150"
            else "accuracy"
        )
        result = dict(count=self.count, correct=self.correct)
        result[key] = self.correct / self.count if self.count else None
        if self.name == "bifi":
            result["by_error_type"] = {
                group: dict(values, accuracy=values["correct"] / values["count"])
                for group, values in sorted(self.groups.items())
            }
        return result
