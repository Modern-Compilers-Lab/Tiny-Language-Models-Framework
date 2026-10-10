import json
import subprocess
import sys
from pathlib import Path

import pytest

from other_models.cli import main
from other_models.experiments.robustness_to_variable_renaming.curves import NS, prepare_views
from other_models.experiments.robustness_to_variable_renaming.legacy import verify
from other_models.runtime import read_jsonl, write_json, write_jsonl


CODE = "def f(value, other):\n    total = value + other + 1\n    adjusted = total + value\n    result = adjusted + other\n    return result + total\n"


def test_vendor_integrity_and_registry():
    verify()
    result = main(["experiment", "list"])
    assert len(result["tasks"]) == 10
    assert result["default_limit"] == 0


@pytest.mark.parametrize("limit,expected", [(0, 140), (2, 2)])
def test_reference_preparation_from_direct_launcher_outside_project(tmp_path, limit, expected):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache = tmp_path / "source"
    (cache / "python").mkdir(parents=True)
    records = [dict(func_code_string=CODE.replace("def f(", f"def f{i}(")) for i in range(140)]
    pq.write_table(pa.Table.from_pylist(records), cache / "python/test-0.parquet")
    run = tmp_path / "prepared"
    script = Path(__file__).resolve().parents[3] / "run.py"
    command = [sys.executable, str(script), "experiment", "prepare", "--run", str(run), "--cache", str(cache), "--corpus-only"]
    if limit:
        command.extend(["--n", str(limit)])
    proc = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    corpus = json.loads((run / "eval_corpus/real6_corpus.json").read_text())
    assert len(corpus["recs"]) == expected
    assert json.loads((run / "data/manifest.json").read_text())["n_requested"] == limit


def test_selected_views_full_default_and_source_preservation(tmp_path):
    data = tmp_path / "source/data"
    write_jsonl(data / "py150_token.jsonl", [dict(id=i, code=["<s>", "value", "=", "1", "</s>"]) for i in range(140)])
    result = prepare_views(data, tmp_path / "views", tasks=["py150_token"])
    assert result["py150_token"]["kept"] == 140
    assert len(list(read_jsonl(data / "py150_token.jsonl"))) == 140
    with pytest.raises(ValueError, match="differ"):
        prepare_views(data, data, tasks=["py150_token"])


def test_all_seven_view_tasks_and_calibration(tmp_path):
    data = tmp_path / "benchmarks/data"
    write_json(data / "manifest.json", dict(n_requested=2))
    write_jsonl(data / "py150_token.jsonl", [dict(id="p", code=["def", "f", "(", "value", ")", ":", "return", "value"])])
    for variant in ("all", "maxmin"):
        row = dict(id="c", prefix="return ", choices=["max", "min"], gold="max", safe=True)
        write_jsonl(data / f"cloze_{variant}.jsonl", [row])
        write_json(data.parent / "eval_corpus" / f"cloze_{variant}_items_n2.json", dict(items=[dict(id="c", mask=1, safe=True, views={str(n): ["return", "max"] for n in NS})]))
    for name in ("exception", "variable_misuse", "swapped_operands", "wrong_binary_operator"):
        for split in ("dev", "eval"):
            code = CODE + ("" if name != "exception" else "")
            if name == "exception":
                code = CODE.replace("return result + total", "raise __HOLE__(result + total)")
            write_jsonl(data / f"cubert_{name}_{split}.jsonl", [dict(id=str(i), code=code, label=label) for i, label in enumerate(["Correct", "Bad"])])
    result = prepare_views(data, tmp_path / "views")
    assert len(result) == 7
    assert all(item["kept"] > 0 and item["dropped"] == 0 for item in result.values())
    for row in read_jsonl(tmp_path / "views/cubert_exception.jsonl"):
        assert all("__HOLE__" in row["views"][str(n)]["code"] for n in NS)


def test_training_cache_rejects_evaluation_rows(tmp_path):
    from other_models.experiments.robustness_to_variable_renaming.token_cache import prepare
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<eos>": 0, "<unk>": 1}, unk_token="<unk>")), eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    source = tmp_path / "evaluation.jsonl"
    write_jsonl(source, [dict(code=CODE, split="test")])
    config = dict(tokenizer=str(tmp_path / "tokenizer"), data=str(source), max_len=32)
    with pytest.raises(ValueError, match="train rows"):
        prepare(config, tmp_path, tmp_path / "tokens")


def test_score_and_report_reject_tampered_predictions(tmp_path, capsys):
    from other_models.runtime import digest
    from other_models.experiments.robustness_to_variable_renaming.curves import score_curves
    from other_models.experiments.robustness_to_variable_renaming.report import verify_results

    run = tmp_path / "evaluation"
    predictions = [dict(N=n, id="p", correct=n == 0) for n in NS]
    path = run / "results/py150_token.curves.predictions.jsonl"
    write_jsonl(path, predictions)
    write_json(run / "results/prediction_manifest.json", {"py150_token": dict(prediction_sha256=digest(path))})
    write_json(run / "results/curves.json", {"py150_token": score_curves(predictions)})
    assert verify_results(run)["py150_token"]["1"]["retention"] == 0
    predictions[1]["correct"] = True
    write_jsonl(path, predictions)
    with pytest.raises(ValueError, match="hash changed"):
        verify_results(run)
    with pytest.raises(SystemExit):
        main(["experiment", "score", "--run", str(run), "--data", str(tmp_path), "--tasks", "py150_token"])
    assert "hash changed" in capsys.readouterr().err


def test_freeze_inside_data_verifies_then_detects_changes(tmp_path, capsys):
    data = tmp_path / "data"
    write_jsonl(data / "py150_token.jsonl", [dict(id=1, code=["x"] )])
    frozen = data / "frozen.json"
    args = ["experiment", "freeze", "--root", str(tmp_path), "--data", str(data), "--out", str(frozen)]
    main(args)
    main(args + ["--verify"])
    write_jsonl(data / "py150_token.jsonl", [dict(id=1, code=["y"])])
    with pytest.raises(SystemExit):
        main(args + ["--verify"])
    assert "Frozen" in capsys.readouterr().err


def test_prefix_download_keeps_last_line_without_newline(tmp_path, monkeypatch):
    import io
    import urllib.request
    from other_models.experiments.robustness_to_variable_renaming.benchmarks import fetch_prefix

    def response(*args, **kwargs):
        stream = io.BytesIO(b'{"id":1}\n{"id":2}')
        stream.headers = {}
        return stream

    monkeypatch.setattr(urllib.request, "urlopen", response)
    base = tmp_path / "records.jsonl"
    first = fetch_prefix("https://fixture.invalid/data", base, 2)
    assert len(list(read_jsonl(first))) == 2
    assert first != fetch_prefix("https://fixture.invalid/data", base, 4)
