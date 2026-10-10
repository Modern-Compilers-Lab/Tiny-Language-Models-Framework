"""Complete renaming coverage on tiny local sources, never public downloads."""

import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from other_models.runtime import setup_run, read_jsonl, write_json, write_jsonl


@pytest.fixture
def release(tmp_path, monkeypatch):
    import urllib.request
    from other_models.benchmark_suite.data import prepare

    count = 140
    paths = {}
    for name in ["codexglue_py150", "codexglue_cloze_all", "codexglue_cloze_maxmin"]:
        source = tmp_path / "normalized" / name
        if name == "codexglue_py150":
            rows = [dict(id=f"test:{i}", code=["return", "value", str(i)], split="test") for i in range(count)]
        else:
            choices = ["max", "min"] if name.endswith("maxmin") else ["max"] + [f"word{i}" for i in range(929)]
            tokens = ["def", "f", "(", "value", ")", ":", "return", "<mask>", "(", "value", ")"]
            rows = [dict(id=f"test:{i}", pl_tokens=tokens, mask_index=7,
                         prefix=" ".join(tokens[:7]) + " ", suffix=" ".join(tokens[8:]),
                         gold="max", choices=choices, split="test") for i in range(count)]
        write_jsonl(source / "test.jsonl", rows)
        paths[name] = prepare(name, tmp_path / "shared", source=source, include_train=False)
    calls = []

    def response(request, *a, **k):
        url = request.full_url if hasattr(request, "full_url") else request
        calls.append(url)
        parsed = urlparse(url)
        if "storage/v1" in url:
            query = parse_qs(parsed.query)
            page = query.get("pageToken", ["first"])[0]
            payload = dict(
                items=[
                    dict(name=query["prefix"][0] + (".a" if page == "first" else ".b"))
                ]
            )
            if page == "first":
                payload["nextPageToken"] = "second"
            raw = json.dumps(payload).encode()
        elif "cloze_test_words" in url:
            raw = ("max\n" + "\n".join(f"word{i}" for i in range(929)) + "\n").encode()
        else:
            raw = b"".join(
                json.dumps(
                    dict(
                        function=f"def f{i}(value):\n    return value + 1\n",
                        label="Correct",
                        info="fixture",
                    )
                ).encode()
                + b"\n"
                for i in range(count // 2)
            )
        stream = io.BytesIO(raw)
        stream.headers = {"Content-Range": f"bytes 0-{len(raw) - 1}/{len(raw)}"}
        return stream

    monkeypatch.setattr(urllib.request, "urlopen", response)
    return paths, calls, count


@pytest.mark.parametrize("cap,expected", [(0, 140), (2, 2)])
def test_renaming_preparation_reads_all_shards_and_pages_or_explicit_cap(
    tmp_path, release, cap, expected
):
    from other_models.experiments.robustness_to_variable_renaming.benchmarks import prepare

    paths, calls, count = release
    run = setup_run(tmp_path / "prepared", {})
    meta = prepare(run, n=cap, benchmark_data=paths)
    assert meta["full"] == (cap == 0) and meta["subset"] == (cap != 0)
    assert meta["n_requested"] == cap
    assert len(meta["subsets"]) == 11
    assert all(info["count"] == expected for info in meta["subsets"].values())
    assert len(list(read_jsonl(run / "data/py150_token.jsonl"))) == expected
    for task in [
        "exception",
        "variable_misuse",
        "swapped_operands",
        "wrong_binary_operator",
    ]:
        for split in ["dev", "eval"]:
            source = meta["sources"][f"cubert_{task}_{split}"]["sources"]
            assert len(source) == (2 if cap == 0 else 1)
            assert all(row["prefix_only"] == (cap != 0) for row in source)
    if cap == 0:
        assert any("pageToken=second" in call for call in calls)
        for variant in ["all", "maxmin"]:
            payload = json.loads(
                (run / "eval_corpus" / f"cloze_{variant}_items_n0.json").read_text()
            )
            assert payload["n_sample"] == 0 and len(payload["items"]) == count
            assert payload["meta"]["n_dataset"] == payload["meta"]["n_sampled"] == count


def test_full_cubert_download_does_not_reuse_a_partial_prefix(tmp_path, monkeypatch):
    import other_models.experiments.robustness_to_variable_renaming.benchmarks as bench

    path = tmp_path / "source.jsonl"
    path.write_text('{"old": 1}\n')
    requested = []
    monkeypatch.setattr(
        bench, "fetch", lambda url, target: requested.append(target) or target
    )
    target = bench.fetch_prefix("https://fixture.invalid/data", path, 0)
    assert target != path and requested == [target]


