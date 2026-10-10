"""Small fixtures only: official formats, metrics, isolation, actual CPU SFT."""

import io
import json
import tarfile
import zipfile
from types import SimpleNamespace

import pytest
from other_models.benchmark_suite import BENCHMARKS
from other_models.benchmark_suite.data import manytypes_slots, prepare
from other_models.benchmark_suite.downloads import (
    BoundedStream,
    Budget,
    HTTPRangeReader,
)
from other_models.benchmark_suite.execution import crux_success
from other_models.benchmark_suite.metrics import (
    crux_metrics,
    pass_at_k,
    repair_metrics,
    repair_result,
    type_metrics,
)
from other_models.benchmark_suite.workflow import manifest, metrics, predict
from other_models.runtime import read_jsonl, setup_run, write_jsonl


def test_framework_exposes_both_workflows(capsys):
    from other_models.cli import main

    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    assert "benchmark" in help_text and "train" in help_text


def test_all_benchmark_download_and_preparation_cli_defaults(tmp_path, monkeypatch):
    from other_models.benchmark_suite import data, sources
    from other_models.cli import main

    downloads = []
    preparations = []
    monkeypatch.setattr(
        sources,
        "download",
        lambda name, run, budget: downloads.append((name, budget.maximum)),
    )
    monkeypatch.setattr(
        data, "prepare", lambda name, *args: preparations.append((name, args))
    )
    main(
        ["benchmark", "download", "--benchmark", "all", "--run", str(tmp_path / "raw")]
    )
    main(
        [
            "benchmark",
            "prepare",
            "--benchmark",
            "all",
            "--run",
            str(tmp_path / "prepared"),
        ]
    )
    assert {name for name, maximum in downloads if maximum is None} == set(BENCHMARKS)
    assert {name for name, _ in preparations} == set(BENCHMARKS)
    assert all(options[2] is None and options[3] is None for _, options in preparations)


@pytest.mark.parametrize("arch", ["gpt_bigcode", "llama", "gpt_neox"])
@pytest.mark.parametrize("name", ["codexglue_py150", "bifi", "manytypes4py"])
def test_standalone_finetuning_saves_new_checkpoint_for_each_model_and_task(
    tmp_path, arch, name
):
    import torch
    from other_models.benchmark_suite.workflow import finetune
    from other_models.runtime import digest
    from transformers import AutoModelForCausalLM

    base, _ = tiny_checkpoint(tmp_path, arch)
    original_files = {
        path.name: digest(path) for path in base.iterdir() if path.is_file()
    }
    before = AutoModelForCausalLM.from_pretrained(base).state_dict()
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "dev", "test"):
        row = dict(id=split, split=split)
        if name == "codexglue_py150":
            row["code"] = [split, "x", "=", "1"]
        elif name == "bifi":
            row.update(
                buggy_tokens=f"{split} = ( 1 <NEWLINE>",
                fixed_tokens=f"{split} = ( 1 ) <NEWLINE>",
            )
            if split == "test":
                row.update(chunk=3, error_type="unbalanced (){}[]")
        else:
            row.update(
                code=f"{split} x = 1",
                scope="<module>",
                target="x",
                kind="Variable",
                gold="int",
            )
        write_jsonl(source / (split + ".jsonl"), [row])
    data = prepare(name, tmp_path / "prepared", source)
    tuned = finetune(
        data,
        base,
        dict(steps=1, batch_size=1, max_len=32, threads=1),
        tmp_path / "tuned",
    )
    assert tuned.resolve() != base.resolve()
    assert tuned.joinpath("model.safetensors").is_file()
    assert tuned.joinpath("tokenizer.json").is_file()
    after = AutoModelForCausalLM.from_pretrained(tuned).state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert original_files == {
        path.name: digest(path) for path in base.iterdir() if path.is_file()
    }
    assert json.loads((tmp_path / "tuned/config.json").read_text())["arch"] == arch


@pytest.mark.parametrize(
    "name",
    [
        "codexglue_cloze_all",
        "codexglue_cloze_maxmin",
        "cruxeval_input",
        "cruxeval_output",
    ],
)
def test_evaluation_only_finetuning_rejected(tmp_path, name):
    from other_models.benchmark_suite.workflow import finetune

    source = tmp_path / "source"
    source.mkdir()
    row = dict(id="test", split="test")
    if name.startswith("codexglue_cloze"):
        row.update(prefix="return ", gold="max", choices=["max", "min"])
    else:
        row.update(code="def f(x): return x", input="1", output="1")
    write_jsonl(source / "test.jsonl", [row])
    data = prepare(name, tmp_path / "prepared", source)
    with pytest.raises(ValueError, match="no official training"):
        finetune(data, "missing-checkpoint", {"steps": 1}, tmp_path / "tuned")
    assert not (tmp_path / "tuned/model").exists()


def test_finetuning_rejects_base_checkpoint_output(tmp_path):
    from other_models.benchmark_suite.workflow import finetune

    base, _ = tiny_checkpoint(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "dev", "test"):
        write_jsonl(
            source / (split + ".jsonl"),
            [dict(id=split, code=[split, "x"], split=split)],
        )
    data = prepare("codexglue_py150", tmp_path / "prepared", source)
    with pytest.raises(ValueError, match="separate output"):
        finetune(data, base, {"steps": 1}, base)


def test_finetuning_rechecks_heldout_inputs(tmp_path):
    from other_models.benchmark_suite.workflow import finetune

    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "dev", "test"):
        write_jsonl(
            source / (split + ".jsonl"),
            [dict(id=split, code=["x", "=", "1"], split=split)],
        )
    data = prepare("codexglue_py150", tmp_path / "prepared", source)
    with pytest.raises(ValueError, match="input overlap"):
        finetune(data, "missing-checkpoint", {"steps": 1}, tmp_path / "tuned")


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_streaming_metrics_match_reference_metrics(name):
    from other_models.benchmark_suite.metrics import MetricAccumulator

    if name == "manytypes4py":
        rows = [
            dict(kind="Parameter", gold="int", predictions=["str", "int"]),
            dict(kind="Return", gold="List[int]", predictions=["List[str]"]),
            dict(kind="Variable", gold="int", predictions=["int"]),
        ]
    elif name.startswith("cruxeval_"):
        rows = [
            dict(successes=[True, False, False, True, False]),
            dict(successes=[False] * 5),
        ]
    elif name == "bifi":
        rows = [
            dict(correct=True, error_type="syntax"),
            dict(correct=False, error_type="syntax"),
            dict(correct=True, error_type="brackets"),
        ]
    else:
        rows = [dict(correct=True), dict(correct=False), dict(correct=True)]
    accumulator = MetricAccumulator(name)
    for row in rows:
        accumulator.add(row)
    assert accumulator.result() == metrics(name, rows)


def test_metrics_match_hand_calculated_official_definitions():
    row = dict(id="a", buggy_tokens="x = ( 1 <NEWLINE>", error_type="unbalanced (){}[]")
    fixed = repair_result(row, "x = ( 1 ) <NEWLINE>")
    assert fixed["correct"] and fixed["token_edits"] == 1
    assert not repair_result(row, row["buggy_tokens"])["correct"]
    assert not repair_result(row, "a b c d e f g h")["correct"]
    syntax = dict(
        row,
        buggy_tokens="def f ( ) <NEWLINE> <INDENT> return 1 <NEWLINE> <DEDENT>",
        error_type="syntax/indentation",
    )
    assert repair_result(
        syntax, "def f ( ) : <NEWLINE> <INDENT> return 1 <NEWLINE> <DEDENT>"
    )["correct"]
    assert repair_metrics([fixed])["repair_accuracy"] == 1
    rows = [
        dict(kind="Parameter", gold="int", predictions=["str", "int"]),
        dict(kind="Return", gold="List[int]", predictions=["List[str]"]),
        dict(kind="Variable", gold="Unseen", predictions=["str"]),
    ]
    got = type_metrics(rows)
    assert got["exact_match@1"] == 0 and got["exact_match@5"] == pytest.approx(1 / 3)
    assert got["mrr@10"] == pytest.approx(1 / 6)
    assert got["parametric_match@10"] == pytest.approx(2 / 3)
    assert pass_at_k(10, 2, 5) == pytest.approx(1 - 56 / 252)
    assert pass_at_k(1, 1, 5) is None
    assert crux_metrics([dict(successes=[True, False])])["pass@1"] == 0.5


def test_crux_value_equality_and_nonunique_inputs():
    row = dict(code="def f(x):\n    return abs(x)", input="2", output="2")
    assert crux_success(row, "f(-2)", "input")
    assert crux_success(row, "[ANSWER]assert f(-2) == 2[/ANSWER]", "input")
    assert not crux_success(row, "f(3)", "input")
    assert crux_success(row, "2.0", "output")
    assert not crux_success(row, "f(2)", "output")
    assert not crux_success(row, '__import__("os").system("something")', "input")
    assert not crux_success(
        dict(row, code="def f(x):\n    while True: pass"), "f(2)", "input", timeout=0.15
    )
    assert crux_success(
        dict(row, output="'# done [ANSWER]'"), "'# done [ANSWER]'", "output"
    )


def test_bifi_selects_heldout_chunks_and_real_training_pairs(tmp_path):
    source = tmp_path / "bifi.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for split in ("train", "dev"):
            archive.writestr(
                f"data/round0/data_paired/{split}.bad", "x = ( 1\ny = ( 2\n"
            )
            archive.writestr(
                f"data/round0/data_paired/{split}.good", "x = ( 1 )\ny = ( 2 )\n"
            )
        for chunk in (3, 4):
            archive.writestr(
                f"data/orig_bad_code/orig.{chunk}.bad",
                "x = ( 1 <NEWLINE>\ny = ( 2 <NEWLINE>\n",
            )
            archive.writestr(
                f"data/orig_bad_code/orig.{chunk}.id", f"{chunk}-a\n{chunk}-b\n"
            )
        archive.writestr("data/huge-model.bin", b"x" * 1000)
    run = setup_run(tmp_path / "run", {})
    data = prepare("bifi", run, source, limit=3, include_train=True)
    rows = list(read_jsonl(data / "test.jsonl"))
    assert [r["chunk"] for r in rows] == [3, 3, 4]
    assert manifest(data)["download_bytes"] == 0
    assert manifest(data)["subset"] and not manifest(data)["full"]
    assert manifest(data)["max_download_mb"] == 64
    assert not list(run.rglob("*.bin"))
    assert len(list(read_jsonl(data / "train.jsonl"))) == 2


def project():
    return {
        "owner/repo": {
            "src_files": {
                f"{s}.py": dict(
                    set=s,
                    untyped_seq=f"def f_{s} ( x ) : return x",
                    typed_seq="SECRET LABELS",
                    variables={"v": "float"},
                    classes=[],
                    funcs=[
                        dict(
                            name="f",
                            q_name="f",
                            params={"x": "int", "self": "Any"},
                            ret_type="str",
                            variables={},
                        )
                    ],
                )
                for s in ("train", "valid", "test")
            }
        }
    }


def test_manytypes_official_schema_splits_and_train_only_vocabulary(tmp_path):
    rows = list(manytypes_slots(project()))
    assert len(rows) == 9 and {r["split"] for r in rows} == {"train", "dev", "test"}
    assert all("SECRET" not in r["code"] for r in rows)
    source = tmp_path / "mt4py.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        raw = json.dumps(project()).encode()
        member = tarfile.TarInfo("data/processed_projects_clean/p.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
        member = tarfile.TarInfo("../../escape.py")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    data = prepare(
        "manytypes4py",
        setup_run(tmp_path / "run", {}),
        source,
        limit=3,
        include_train=True,
    )
    assert json.loads((data / "type_vocabulary.json").read_text()) == [
        "float",
        "int",
        "str",
    ]
    assert not (tmp_path / "escape.py").exists()
    assert manifest(data)["splits"]["dev"]["count"] == 3


def test_budget_stream_stops_at_cap():
    b = Budget(4)
    stream = BoundedStream(io.BytesIO(b"12345678"), b)
    assert stream.read(2) == b"12"
    with pytest.raises(ValueError, match="budget"):
        stream.read(100)
    assert b.used == 5


def test_range_reader_and_no_large_fallback(monkeypatch):
    import urllib.request

    payload = b"0123456789"
    calls = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, a, b):
            super().__init__(payload[a : b + 1])
            self.headers = {"Content-Range": f"bytes {a}-{b}/{len(payload)}"}

    def request(req, timeout):
        a, b = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((a, b))
        return Response(a, b)

    monkeypatch.setattr(urllib.request, "urlopen", request)
    reader = HTTPRangeReader("https://example.invalid/data", Budget(32))
    reader.seek(-3, 2)
    assert reader.read() == b"789"
    reader.seek(-3, 2)
    assert reader.read() == b"789" and len(calls) == 2

    def unsupported(*a, **k):
        r = Response(0, 9)
        r.status = 200
        return r

    monkeypatch.setattr(urllib.request, "urlopen", unsupported)
    with pytest.raises(ValueError, match="ranges"):
        HTTPRangeReader("https://example.invalid/data", Budget(32))


@pytest.mark.parametrize(
    "name",
    [
        "codexglue_cloze_all",
        "codexglue_cloze_maxmin",
        "cruxeval_input",
        "cruxeval_output",
    ],
)
def test_evaluation_only_rejects_training_before_download(tmp_path, name):
    with pytest.raises(ValueError, match="evaluation-only"):
        prepare(name, tmp_path, include_train=True)


def test_rejects_split_overlap_and_changed_prepared_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "dev", "test"):
        write_jsonl(
            source / (split + ".jsonl"), [dict(id="same", code=["x"], split=split)]
        )
    with pytest.raises(ValueError, match="overlapping"):
        prepare("codexglue_py150", tmp_path / "run", source, include_train=True)
    source2 = tmp_path / "source2"
    source2.mkdir()
    write_jsonl(source2 / "test.jsonl", [dict(id="one", code=["x"], split="test")])
    data = prepare("codexglue_py150", tmp_path / "second", source2, include_train=False)
    (data / "test.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        manifest(data)


def test_all_benchmark_prediction_routes():
    class Model:
        def sample(self, prompt, n=1, **kw):
            if "Bad:" in prompt:
                return ["x = ( 1 ) <NEWLINE>"]
            if "Find a function" in prompt:
                return ["f(-2)"] * n
            return ["2"] * n

        def continuation_scores(self, prefix, choices):
            return [0.0] + [-1.0] * (len(choices) - 1)

        def lexical_predictions(self, tokens, max_length):
            return [dict(index=0, gold="x", prediction="x", correct=True)]

    rows = dict(
        codexglue_py150=[dict(id="a", code=["x"])],
        codexglue_cloze_all=[
            dict(id="a", prefix="return ", choices=["max", "min"], gold="max")
        ],
        codexglue_cloze_maxmin=[
            dict(id="a", prefix="return ", choices=["max", "min"], gold="max")
        ],
        bifi=[
            dict(
                id="a", buggy_tokens="x = ( 1 <NEWLINE>", error_type="unbalanced (){}[]"
            )
        ],
        manytypes4py=[
            dict(
                id="a",
                code="x = 1",
                kind="Variable",
                scope="<module>",
                target="x",
                gold="int",
            )
        ],
        cruxeval_input=[
            dict(id="a", code="def f(x):\n    return abs(x)", input="2", output="2")
        ],
        cruxeval_output=[
            dict(id="a", code="def f(x):\n    return abs(x)", input="2", output="2")
        ],
    )
    for name in BENCHMARKS:
        result = predict(
            name, Model(), rows[name], dict(type_vocabulary=["int", "str"], samples=1)
        )
        assert metrics(name, result)["count"] == 1
        key = BENCHMARKS[name].metrics[0]
        assert metrics(name, result)[key] == 1


def tiny_checkpoint(tmp_path, arch="llama"):
    from other_models.models import build_config
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    raw = Tokenizer(
        models.WordLevel(
            {
                "<eos>": 0,
                "<unk>": 1,
                "x": 2,
                "=": 3,
                "1": 4,
                "int": 5,
                "str": 6,
                "Type": 7,
            },
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, eos_token="<eos>", pad_token="<eos>", unk_token="<unk>"
    )
    checkpoint = tmp_path / "base"
    tok.save_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_config(
        build_config(arch, len(tok), hidden=16, layers=1, heads=1, context=64)
    )
    model.save_pretrained(checkpoint)
    return checkpoint, tok


def test_prompt_masking_preserves_answer_and_eos(tmp_path):
    from other_models.training import encode_training_rows

    _, tok = tiny_checkpoint(tmp_path)
    ids, labels = encode_training_rows(
        tok, [dict(split="train", prompt="x " * 20, target="int")], 4, True
    )
    assert labels[0][-2:] == [5, 0] and labels[0][:-2] == [-100, -100]
    assert len(ids[0]) == 4
    with pytest.raises(ValueError, match="train rows"):
        encode_training_rows(
            tok, [dict(split="test", prompt="x", target="int")], 4, True
        )
    with pytest.raises(ValueError, match="exceeds context"):
        encode_training_rows(
            tok, [dict(split="train", prompt="x", target="int " * 4)], 4, True
        )


def test_actual_checkpoint_finetune_and_cli_evaluation(tmp_path, capsys):
    import torch
    from other_models.cli import main
    from transformers import AutoModelForCausalLM

    base, _ = tiny_checkpoint(tmp_path)
    before = {
        k: v.clone()
        for k, v in AutoModelForCausalLM.from_pretrained(base).state_dict().items()
    }
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "dev", "test"):
        write_jsonl(
            source / (split + ".jsonl"),
            [
                dict(
                    id=split,
                    code=f"{split} x = 1",
                    scope="<module>",
                    target="x",
                    kind="Variable",
                    gold="int",
                    split=split,
                )
            ],
        )
    data = prepare(
        "manytypes4py", tmp_path / "prepare", source, include_train=True, limit=1
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(dict(steps=1, batch_size=1, max_len=32, save_every=1, threads=1))
    )
    main(
        [
            "benchmark",
            "evaluate",
            "--model",
            str(base),
            "--data",
            str(data),
            "--run",
            str(tmp_path / "eval"),
            "--finetune-config",
            str(recipe),
            "--max-len",
            "32",
            "--max-new-tokens",
            "4",
        ]
    )
    result = json.loads((tmp_path / "eval/results/metrics.json").read_text())
    assert result["finetuned"] and result["count"] == 1
    after = AutoModelForCausalLM.from_pretrained(
        tmp_path / "eval/finetune/model"
    ).state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    original = AutoModelForCausalLM.from_pretrained(base).state_dict()
    assert all(torch.equal(before[k], original[k]) for k in before)
    training = json.loads(
        (tmp_path / "eval/finetune/results/training.json").read_text()
    )
    assert training["config"]["pretrained"] == str(base) and math_finite(
        training["final_loss"]
    )


def math_finite(value):
    import math

    return math.isfinite(value)


def test_actual_py150_scores_first_target_and_full_sliding_sequence(tmp_path):
    from other_models.models import HFModel

    base, _ = tiny_checkpoint(tmp_path)
    model = HFModel(base)
    tokens = ["<s>"] + ["x", "=", "1", "<EOL>"] * 12 + ["</s>"]
    result = model.lexical_predictions(tokens, max_length=8)
    assert len(result) == 36 and result[0]["index"] == 1 and result[-1]["index"] == 47


@pytest.mark.parametrize("limit, count", [(None, 140), (0, 140), (2, 2)])
@pytest.mark.parametrize("tuned", [False, True])
def test_benchmark_cli_scores_all_by_default(
    tmp_path, monkeypatch, limit, count, tuned
):
    from other_models import models
    from other_models.benchmark_suite import workflow
    from other_models.benchmark_suite.data import prepare
    from other_models.cli import main

    source = tmp_path / "source"
    source.mkdir()
    write_jsonl(
        source / "test.jsonl",
        [dict(id=str(i), split="test", code=["x", str(i)]) for i in range(140)],
    )
    data = prepare("codexglue_py150", tmp_path / "data", source, include_train=False)

    class Model:
        context = 1024
        causal_lm = SimpleNamespace(config=SimpleNamespace())

        def lexical_predictions(self, tokens, max_length):
            return [dict(index=0, gold="x", prediction="x", correct=True)]

    monkeypatch.setattr(models, "HFModel", lambda *a, **k: Model())
    args = [
        "benchmark",
        "evaluate",
        "--data",
        str(data),
        "--model",
        "local-test-model",
        "--run",
        str(tmp_path / "eval"),
    ]
    if tuned:
        config = tmp_path / "finetune.json"
        config.write_text("{}")
        monkeypatch.setattr(workflow, "finetune", lambda *a: "local-tuned-model")
        args += ["--finetune-config", str(config)]
    if limit is not None:
        args += ["--limit", str(limit)]
    main(args)
    result = json.loads((tmp_path / "eval/results/metrics.json").read_text())
    assert result["evaluated_examples"] == count
    assert result["subset"] == (count < 140)
    assert result["finetuned"] == tuned
