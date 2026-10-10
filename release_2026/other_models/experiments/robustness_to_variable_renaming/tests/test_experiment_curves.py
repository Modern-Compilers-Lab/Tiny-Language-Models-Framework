import pytest
from other_models.experiments.robustness_to_variable_renaming.curves import (
    score_curves,
    code_views,
    lexical_views,
    NS,
    paper_curves,
)


def test_paired_retention_hand_calculated_and_missing_targets():
    rows = [
        dict(N=n, id=i, correct=c)
        for n, values in [(0, [True, True, False]), (1, [True, False, True])]
        for i, c in enumerate(values)
    ]
    out = score_curves(rows)
    assert out["0"]["accuracy"] == pytest.approx(2 / 3)
    assert out["1"]["accuracy"] == pytest.approx(2 / 3)
    assert out["1"]["retention"] == 0.5 and out["1"]["baseline_correct_paired"] == 2
    assert out["2"]["retention"] is None
    assert score_curves([dict(N=0, id=0, correct=False)])["0"]["retention"] is None
    with pytest.raises(ValueError):
        score_curves(rows + rows)


def test_code_views_semantics_literals_and_determinism():
    code = 'def f(value, other):\n    answer = value + other\n    extra = answer * 2\n    label = "value other answer"\n    return extra + len(label)\n'
    got = code_views(code, 7)
    assert got == code_views(code, 7)
    views, counts = got
    for n in NS:
        ns = {}
        exec(views[str(n)], ns)
        assert ns["f"](2, 3) == 28
        assert "value other answer" in views[str(n)]
        assert counts[str(n)] <= n
    assert views["0"] != views["1"]
    assert code_views("invalid syntax (", 0) is None


def test_lexical_preserves_attributes_keywords_strings_and_lengths():
    tokens = [
        "<s>",
        "def",
        "f",
        "(",
        "value",
        ")",
        ":",
        "<EOL>",
        "return",
        "value",
        "+",
        "obj",
        ".",
        "value",
        "+",
        '"value"',
        "</s>",
    ]
    views, counts = lexical_views(tokens, 1)
    assert views == lexical_views(tokens, 1)[0]
    for n in NS:
        row = views[str(n)]
        assert row[13] == "value" and row[15] == '"value"' and row[1] == "def"
        assert all(len(a) == len(b) for a, b in zip(tokens, row))
        assert counts[str(n)] <= n


def test_paper_curve_adapter_scalar_accuracy():
    raw = {"N0_accuracy_safe": dict(k=1, n=2, rate=0.5)}
    for n in NS[1:]:
        raw[f"N{n}"] = dict(accuracy_safe=0.5, retention_safe=dict(k=1, n=1, rate=1.0))
    out = paper_curves("cloze_all_paper", dict(raw=raw, count=2))
    assert out["8"]["accuracy"] == 0.5


def test_curve_limit_applies_to_every_rename_view(tmp_path, monkeypatch):
    from other_models.experiments.robustness_to_variable_renaming.curves import predict_curves
    from other_models.experiments.robustness_to_variable_renaming.registry import TASKS
    from other_models.runtime import write_jsonl

    monkeypatch.setitem(
        TASKS,
        "test_limit",
        lambda model, rows, options: [dict(id=r["id"], correct=True) for r in rows],
    )
    path = tmp_path / "views.jsonl"
    write_jsonl(
        path,
        [
            dict(id=i, views={str(n): dict(id=i, code="x") for n in NS})
            for i in range(3)
        ],
    )
    got = predict_curves(None, "test_limit", path, {"limit": 1})
    assert len(got) == 5 and {r["id"] for r in got} == {0}


