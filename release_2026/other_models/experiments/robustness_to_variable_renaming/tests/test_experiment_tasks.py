import pytest
from other_models.experiments.robustness_to_variable_renaming.tasks.bugs import threshold, classify
from other_models.experiments.robustness_to_variable_renaming.tasks.cloze import cloze
from other_models.experiments.robustness_to_variable_renaming.tasks.completion import render, token_completion
from other_models.models import build_config
from other_models.datasets import normalize_repo, code_hash


class Fake:
    def continuation_scores(self, prefix, choices):
        return [0.0, 2.0]

    def sequence_scores(self, texts, max_length):
        return [float(t) for t in texts]

    def generate(self, prompt, max_new_tokens):
        return "return x\nignored"


def test_cloze_multitoken_and_ties():
    got = cloze(
        Fake(),
        [dict(id="1", prefix="code", choices=["a", "two tokens"], gold="two tokens")],
        {},
    )
    assert got[0]["correct"] and got[0]["prediction"] == 1


def test_threshold_no_test_labels():
    dev = [dict(code="-3", label="Bad"), dict(code="-1", label="Correct")]
    test = [
        dict(id="0", code="-4", label="Bad"),
        dict(id="1", code="0", label="Correct"),
    ]
    out = classify(Fake(), test, {"dev": dev})
    assert all(x["correct"] for x in out)
    assert out[0]["threshold"] == -2
    test[0]["label"] = "Correct"
    assert classify(Fake(), test, {"dev": dev})[0]["threshold"] == -2


def test_token_alignment_excludes_truncation():
    class M:
        def token_predictions(self, texts, max_length):
            return [([4, 5, 6], [(0, 1), (2, 3), (4, 5)], [5, 6, 0])]

    out = token_completion(
        M(), [dict(id=1, code=["x", "=", "long_name"])], {"max_len": 3}
    )
    assert len(out) == 1 and out[0]["correct"]


def test_repository_normalization():
    assert normalize_repo("https://github.com/OWNER/REPO.git") == "owner/repo"


def test_function_hash_ignores_formatting():
    assert code_hash("def f(x):\n return x+1") == code_hash(
        "def f(x):\n    return x + 1"
    )


def noisy_metric(values):
    rate = sum(values) / len(values) if values else None
    print("accuracy %.4f" % rate)
    return {"rate": rate, "count": len(values)}


def test_paper_diagnostics_do_not_change_metrics():
    from other_models.experiments.robustness_to_variable_renaming.tasks.paper import quiet_command

    quiet = quiet_command(noisy_metric)
    assert quiet([True, False]) == noisy_metric([True, False])
    assert quiet([]) == {"rate": None, "count": 0}


def boolean_pairs(values):
    import numpy as np

    b0 = np.array([x for x, _ in values])
    bN = np.array([y for _, y in values])
    return {"joint": int((b0 & bN).sum()), "baseline": int(b0.sum())}


def test_paper_empty_boolean_support_and_defined_values():
    from other_models.experiments.robustness_to_variable_renaming.tasks.paper import quiet_command

    wrapped = quiet_command(boolean_pairs)
    assert wrapped([(True, False), (True, True)]) == boolean_pairs(
        [(True, False), (True, True)]
    )
    assert wrapped([]) == {"joint": 0, "baseline": 0}


def test_continuation_moves_boundary_whitespace_into_answer():
    import torch
    from types import SimpleNamespace
    from other_models.models import HFModel

    class Tok:
        bos_token_id = 0
        eos_token_id = 0

        def encode(self, text, **kw):
            return {
                "return": [1],
                "return ": [1, 2],
                "max": [3],
                "min": [5],
                " max": [4],
                " min": [6],
            }[text]

        def pad(self, batch, **kw):
            ids = torch.tensor(batch["input_ids"])
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids))

    seen = []

    def forward(**batch):
        seen.append(batch["input_ids"].tolist())
        return SimpleNamespace(logits=torch.zeros(*batch["input_ids"].shape, 10))

    model = HFModel.__new__(HFModel)
    model.tokenizer = Tok()
    model.causal_lm = forward
    model.context = 32
    model.batch = 16
    model.device = "cpu"
    scores = model.continuation_scores("return ", ["max", "min"])
    assert seen == [[[1, 4], [1, 6]]]
    assert scores == pytest.approx([-2.302585093, -2.302585093])


def test_token_alignment_excludes_piece_crossing_right_lexical_boundary():
    class M:
        def token_predictions(self, texts, max_length):
            # One hypothetical tokenizer piece spans two lexical targets, '= y'.
            return [([4, 8], [(0, 1), (2, 5)], [8, 0])]

    out = token_completion(M(), [dict(id=1, code=["x", "=", "y"])], {"max_len": 3})
    assert out == []


def test_token_alignment_excludes_partial_first_lexical_target():
    class M:
        def token_predictions(self, texts, max_length):
            return [([4, 5, 6], [(0, 2), (2, 4), (5, 6)], [5, 6, 0])]

    out = token_completion(M(), [dict(id=1, code=["long", "="])], {"max_len": 3})
    assert [r["id"] for r in out] == ["1:1"]


def test_paper_cli_batch_reaches_model_and_scorer(tmp_path, monkeypatch):
    import sys
    import other_models.models as models
    import other_models.experiments.robustness_to_variable_renaming.tasks.paper as paper

    calls = []
    monkeypatch.setattr(
        models,
        "HFModel",
        lambda path, device, dtype, batch: calls.append(("model", batch)),
    )

    def score(*args, **kwargs):
        calls.append(("score", kwargs["batch"]))
        return {"accuracy": None, "count": 0}

    monkeypatch.setattr(paper, "score_paper", score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper",
            "--model",
            "toy",
            "--task",
            "identifier_reproduction",
            "--run",
            str(tmp_path),
            "--batch",
            "7",
        ],
    )
    paper.main()
    assert calls == [("model", 7), ("score", 7)]
