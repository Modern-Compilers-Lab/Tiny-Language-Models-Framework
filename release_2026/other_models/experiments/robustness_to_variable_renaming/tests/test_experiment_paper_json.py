import json, math


def test_undefined_paper_intervals_are_explicit_json_null():
    from other_models.experiments.robustness_to_variable_renaming.tasks.paper import json_safe_raw

    raw = {
        "diagnostic": {
            "k": -60,
            "n": 108,
            "rate": -5 / 9,
            "ci_lo": float("nan"),
            "ci_hi": float("inf"),
        }
    }
    clean, paths = json_safe_raw(raw)
    assert clean == {
        "diagnostic": {"k": -60, "n": 108, "rate": -5 / 9, "ci_lo": None, "ci_hi": None}
    }
    assert paths == ["raw.diagnostic.ci_lo", "raw.diagnostic.ci_hi"]
    assert math.isnan(raw["diagnostic"]["ci_lo"])
    json.dumps(clean, allow_nan=False)


def test_finite_paper_values_remain_exact():
    from other_models.experiments.robustness_to_variable_renaming.tasks.paper import json_safe_raw

    raw = {
        "rate": 1 / 3,
        "counts": [0, 1, 7],
        "missing": None,
        "flag": False,
        "nested": {"ci_lo": 0.000123456789},
    }
    clean, paths = json_safe_raw(raw)
    assert clean == raw and not paths
    assert json.dumps(clean) == json.dumps(raw)
