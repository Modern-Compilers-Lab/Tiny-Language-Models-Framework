import json
import urllib.request

import pytest

from other_models.benchmark_suite.data import prepare as prepare_shared
from other_models.benchmark_suite.workflow import ensure_data
from other_models.runtime import read_jsonl, setup_run, write_jsonl
from other_models.experiments.robustness_to_variable_renaming.benchmarks import cloze_items, cloze_tokens, prepare


def shared_rows(name, count=4):
    for i in range(count):
        if name == "codexglue_py150":
            yield dict(id=f"test:{i}", code=["def", "f", "(", "value", ")", ":", "return", "value"], split="test")
        else:
            tokens = ["def", f"f{i}", "(", "value", ")", ":", "return", "<mask>", "(", "value", ",", "'a b'", ")"]
            yield dict(id=f"test:{i}", pl_tokens=tokens, mask_index=7,
                       prefix=" ".join(tokens[:7]) + " ", suffix=" ".join(tokens[8:]),
                       gold="max", choices=["max", "min"], split="test")


def shared_data(tmp_path, name, count=4, limit=0):
    source = tmp_path / "normalized" / name
    write_jsonl(source / "test.jsonl", shared_rows(name, count))
    return prepare_shared(name, tmp_path / f"prepared-{limit}", source=source, limit=limit, include_train=False)


@pytest.mark.parametrize("limit", [0, 2])
def test_all_codexglue_inputs_reused_without_network(tmp_path, monkeypatch, limit):
    names = ["codexglue_py150", "codexglue_cloze_all", "codexglue_cloze_maxmin"]
    paths = {name: shared_data(tmp_path, name) for name in names}
    before = {name: (path / "test.jsonl").read_bytes() for name, path in paths.items()}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected redownload"))
    run = setup_run(tmp_path / "experiment", {})
    meta = prepare(run, n=limit, benchmark_data=paths, tasks=["py150_token", "cloze_maxmin", "cloze_all_multitoken", "cloze_all_paper"])
    assert set(meta["subsets"]) == {"py150_token", "cloze_all", "cloze_maxmin"}
    for task, source in meta["sources"].items():
        assert source["manifest_sha256"]
        assert meta["subsets"][task]["count"] == (limit or 4)
        name = source["benchmark"]
        assert (paths[name] / "test.jsonl").read_bytes() == before[name]
    assert list(read_jsonl(run / "data/py150_token.jsonl"))[0]["id"] == "test:0"
    cloze = json.loads((run / "eval_corpus" / f"cloze_maxmin_items_n{limit}.json").read_text())
    assert all(item["gold"] == "max" and "'a b'" in item["views"]["0"] for item in cloze["items"])


def test_full_request_rejects_prepared_subset(tmp_path, monkeypatch):
    path = shared_data(tmp_path, "codexglue_py150", limit=2)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected redownload"))
    with pytest.raises(ValueError, match="prepared subset"):
        ensure_data("codexglue_py150", tmp_path / "benchmarks", prepared=path)
    assert ensure_data("codexglue_py150", tmp_path / "benchmarks", limit=2, prepared=path) == path.resolve()


def test_automatic_reuse_and_tamper_detection(tmp_path, monkeypatch):
    path = shared_data(tmp_path, "codexglue_cloze_maxmin")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected redownload"))
    assert ensure_data("codexglue_cloze_maxmin", tmp_path / "new", search_root=tmp_path) == path.resolve()
    with (path / "test.jsonl").open("a") as output:
        output.write("{}\n")
    with pytest.raises(ValueError, match="data changed"):
        ensure_data("codexglue_cloze_maxmin", tmp_path / "new", prepared=path)


def test_legacy_prepared_cloze_preserves_quoted_whitespace():
    row = next(shared_rows("codexglue_cloze_maxmin"))
    expected = cloze_tokens(row)
    del row["pl_tokens"], row["mask_index"]
    assert cloze_tokens(row) == expected


def test_legacy_flattened_comments_require_original_tokens():
    row = next(shared_rows("codexglue_cloze_maxmin"))
    del row["pl_tokens"], row["mask_index"]
    row["prefix"] = "def f ( value ) : # comment without a recorded boundary "
    with pytest.raises(ValueError, match="ambiguous token boundaries"):
        cloze_tokens(row)


def test_old_comment_rows_restored_locally_without_label_recovery(tmp_path, monkeypatch):
    from other_models.benchmark_suite.data import restore_cloze_tokens
    from other_models.runtime import write_json

    row = next(shared_rows("codexglue_cloze_maxmin"))
    tokens = row["pl_tokens"][:7] + ["# a comment containing spaces"] + row["pl_tokens"][7:]
    row.update(prefix=" ".join(tokens[:8]) + " ", suffix=" ".join(tokens[9:]))
    del row["pl_tokens"], row["mask_index"]
    write_json(tmp_path / "clozeTest.json", [dict(idx=row["id"], pl_tokens=tokens)])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected redownload"))
    restored = restore_cloze_tokens([row], tmp_path)[0]
    assert restored["gold"] == row["gold"]
    assert restored["pl_tokens"] == tokens and restored["mask_index"] == 8
    assert cloze_tokens(restored)[0][8] == "max"


@pytest.mark.parametrize("variant", ["all", "maxmin"])
def test_cloze_intervention_matches_immutable_builder(tmp_path, monkeypatch, variant):
    import datasets
    from other_models.experiments.robustness_to_variable_renaming.legacy import library

    rows = list(shared_rows("codexglue_cloze_maxmin", 3))
    expected, _ = cloze_items(rows, variant)
    masked = [dict(id=row["id"], pl_tokens=row["pl_tokens"]) for row in rows]
    gold = [dict(func_code_tokens=cloze_tokens(row)[0]) for row in rows]
    def load(name, *args, **kwargs):
        if "cloze_testing" in name:
            return {"train": datasets.Dataset.from_list(masked)}
        return iter(gold)
    monkeypatch.setattr(datasets, "load_dataset", load)
    corpus = tmp_path / "eval_corpus"
    corpus.mkdir()
    library("whyscale").CACHE = str(corpus)
    actual, _ = library("decmech").cloze_items(variant, 3)
    assert actual == expected
