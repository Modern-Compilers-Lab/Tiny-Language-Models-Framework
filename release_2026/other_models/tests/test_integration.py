import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from other_models.cli import main
from other_models.datasets import code_hash, prepare
from other_models.models import build_model, load_model, resolve_dataset
from other_models.runtime import read_jsonl
from other_models.source_downloads import (
    MANIFEST,
    REPOSITORIES,
    download_dataset,
    verify_source,
)


def program(name, operator="+"):
    return f"def {name}(value, other):\n    result = value {operator} other\n    return result + value\n"


def parquet(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), output)
    return output.getvalue()


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    train = [
        dict(func_code_string=program(f"f{i}"), repository_name="fixture/train")
        for i in range(3)
    ]
    sources = {
        REPOSITORIES["codesearchnet"]: {
            "README.md": b"fixture",
            "python/train-0.parquet": parquet(train),
            "python/test-0.parquet": parquet(
                [
                    dict(
                        func_code_string=program("held", "-"),
                        repository_name="fixture/held",
                    )
                ]
            ),
            "ruby/train-0.parquet": b"other language",
        },
        REPOSITORIES["thevault"]: {
            "data/train/small/python-0.parquet": parquet(
                [dict(original_string=program("small"), repo="fixture/train")]
            ),
            "data/train/full/python-0.parquet": parquet(
                [
                    dict(original_string=program("full"), repo="fixture/train"),
                    dict(original_string=program("small"), repo="fixture/train"),
                ]
            ),
            "README.md": b"fixture",
        },
        REPOSITORIES["codeparrot"]: {
            "file-0.json.gz": gzip.compress(
                (
                    "\n".join(
                        json.dumps(
                            dict(content=program(f"p{i}"), repo_name="fixture/train")
                        )
                        for i in range(3)
                    )
                    + "\n"
                ).encode()
            ),
            "README.md": b"fixture",
        },
    }
    calls = dict(info=[], snapshots=[], omit=None)

    class API:
        def dataset_info(self, repo, revision, files_metadata):
            calls["info"].append((repo, revision))
            return SimpleNamespace(
                sha="a" * 40,
                siblings=[
                    SimpleNamespace(rfilename=p, size=len(b))
                    for p, b in sources[repo].items()
                ],
            )

    def snapshot(repo, **options):
        calls["snapshots"].append((repo, options))
        root = Path(options["local_dir"])
        for name, data in sources[repo].items():
            if options.get("allow_patterns") and name not in options["allow_patterns"]:
                continue
            if name == calls["omit"]:
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return str(root)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    return calls, sources


@pytest.mark.parametrize("dataset", list(REPOSITORIES))
def test_full_download_and_offline_reuse(tmp_path, hub, dataset):
    calls, sources = hub
    root = download_dataset(dataset, tmp_path)
    metadata = verify_source(root, dataset)
    assert metadata["full"] and metadata["complete"]
    assert len(metadata["files"]) == len(sources[REPOSITORIES[dataset]])
    assert "allow_patterns" not in calls["snapshots"][0][1]
    assert calls["snapshots"][0][1]["revision"] == "a" * 40
    download_dataset(dataset, tmp_path)
    assert len(calls["snapshots"]) == 1


def test_download_preview_and_partial_shards(tmp_path, hub):
    calls, _ = hub
    preview = download_dataset("thevault", tmp_path, dry_run=True)
    assert preview["full"] and preview["file_count"] == 3
    assert not tmp_path.joinpath("sources").exists() and not calls["snapshots"]
    source = download_dataset("thevault", tmp_path, max_files=1)
    metadata = verify_source(source, "thevault", require_full=False)
    assert not metadata["full"] and len(metadata["files"]) == 1
    assert metadata["files"][0]["path"].endswith("python-0.parquet")
    with pytest.raises(ValueError, match="subset"):
        verify_source(source, "thevault")
    with pytest.raises(ValueError, match="changed"):
        download_dataset("thevault", tmp_path)


def test_incomplete_download_retry_pins_revision(tmp_path, hub):
    calls, _ = hub
    calls["omit"] = "ruby/train-0.parquet"
    with pytest.raises(ValueError, match="omitted"):
        download_dataset("codesearchnet", tmp_path)
    assert not json.loads((tmp_path / "sources/codesearchnet" / MANIFEST).read_text())[
        "complete"
    ]
    calls["omit"] = None
    source = download_dataset("codesearchnet", tmp_path)
    assert calls["info"][-1][1] == "a" * 40
    (source / "ruby/train-0.parquet").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="Incomplete"):
        verify_source(source, "codesearchnet")


@pytest.mark.parametrize(
    "dataset,count", [("codesearchnet", 3), ("thevault", 2), ("codeparrot", 3)]
)
def test_full_preparation_visits_all_partitions_and_deduplicates(
    tmp_path, hub, dataset, count
):
    path = prepare(dataset, tmp_path)
    rows = list(read_jsonl(path))
    assert len(rows) == count
    assert all(
        row["split"] == "train" and row["meta"]["source"] == dataset for row in rows
    )
    metadata = json.loads(Path(str(path) + ".manifest.json").read_text())
    assert metadata["full"] and metadata["source_full"]
    assert metadata["source_revision"] == "a" * 40
    assert not list(tmp_path.glob("tmp/*.sqlite"))


def test_local_partial_preparation_and_leakage_filters(tmp_path, hub):
    calls, sources = hub
    # Add a copied heldout function, a heldout repository and a duplicate.
    rows = [
        dict(func_code_string=program("held", "-"), repository_name="fixture/copied"),
        dict(
            func_code_string=program("leak"),
            repository_name="https://github.com/Fixture/Held.git",
        ),
        dict(func_code_string=program("ok"), repository_name="fixture/train"),
        dict(func_code_string=program("ok"), repository_name="fixture/train"),
        dict(func_code_string=program("also_ok"), repository_name="fixture/train"),
    ]
    sources[REPOSITORIES["codesearchnet"]]["python/train-0.parquet"] = parquet(rows)
    source = download_dataset("codesearchnet", tmp_path / "raw")
    calls["snapshots"].clear()
    path = prepare(
        "codesearchnet",
        tmp_path / "prepared",
        cache=source,
        reference_cache=source,
        n_docs=2,
    )
    metadata = json.loads(Path(str(path) + ".manifest.json").read_text())
    assert metadata["kept"] == 2 and metadata["duplicate"] == 1
    assert metadata["excluded_repo"] == metadata["excluded_exact_function"] == 1
    assert not metadata["full"] and not calls["snapshots"]
    assert code_hash(list(read_jsonl(path))[0]["code"]) != code_hash(
        program("held", "-")
    )


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("starcoder", "codesearchnet"),
        ("llama", "codesearchnet"),
        ("gpt_neox", "thevault"),
    ],
)
def test_cli_model_defaults_and_overrides(tmp_path, hub, arch, expected):
    main(["prepare", "--arch", arch, "--run", str(tmp_path), "--n-docs", "1"])
    assert len(list(read_jsonl(tmp_path / "data" / f"{expected}.jsonl"))) == 1
    assert resolve_dataset(arch, "codeparrot") == "codeparrot"


def test_cli_all_datasets(tmp_path, hub):
    main(["prepare", "--dataset", "all", "--run", str(tmp_path)])
    assert all((tmp_path / "data" / f"{name}.jsonl").is_file() for name in REPOSITORIES)


@pytest.mark.parametrize(
    "arch,expected",
    [("gpt_bigcode", 18919552), ("llama", 18292416), ("gpt_neox", 18334848)],
)
def test_original_parameter_counts(arch, expected):
    assert sum(p.numel() for p in build_model(arch).parameters()) == expected


@pytest.mark.parametrize(
    "arch,family",
    [
        ("starcoder", "GPTBigCodeForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
    ],
)
def test_real_architecture_gradients_and_causality(arch, family):
    import torch

    model = build_model(arch, vocab_size=17, hidden=32, heads=2, layers=1, context=32)
    assert type(model).__name__ == family
    ids = torch.tensor([[2, 3, 4, 5]])
    loss = model(ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    model.eval()
    with torch.no_grad():
        original = model(ids).logits
        changed = model(torch.tensor([[2, 3, 4, 9]])).logits
    assert torch.allclose(original[:, :3], changed[:, :3], atol=1e-6)


@pytest.fixture
def local_tokenizer(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        WordLevel(
            {
                "<eos>": 0,
                "<unk>": 1,
                "def": 2,
                "return": 3,
                "value": 4,
                "other": 5,
                "result": 6,
                "=": 7,
                "+": 8,
                "(": 9,
                ")": 10,
                ":": 11,
                ",": 12,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<eos>",
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token="<unk>",
    )
    fast.save_pretrained(tmp_path / "tokenizer")
    return tmp_path / "tokenizer"


@pytest.mark.parametrize("arch", ["gpt_bigcode", "llama", "gpt_neox"])
def test_training_save_reload_and_generation(tmp_path, local_tokenizer, arch):
    import torch
    from other_models.training import train

    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(dict(code=program("train"), split="train")) + "\n", encoding="utf-8"
    )
    config = dict(
        arch=arch,
        dataset="codeparrot",
        data=str(data),
        tokenizer=str(local_tokenizer),
        model=dict(hidden=32, heads=2, layers=1, context=32),
        steps=2,
        batch_size=2,
        micro_batch=1,
        max_len=24,
        warmup=0,
        device="cpu",
        threads=1,
        save_every=1,
    )
    report = train(config, tmp_path / "run")
    assert report["steps"] == 2 and report["config"]["dataset"] == "codeparrot"
    assert report["final_loss"] > 0
    assert (tmp_path / "run/checkpoints/step-1/model.safetensors").exists()
    model, tokenizer = load_model(
        tmp_path / "run/model", device="cpu", local_files_only=True
    )
    assert model.config.use_cache
    inputs = tokenizer("def value", return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2, pad_token_id=0)
    assert output.shape[1] > inputs["input_ids"].shape[1]


def test_cli_training_dataset_override_replaces_recipe_data(
    tmp_path, local_tokenizer, monkeypatch
):
    from other_models import training

    captured = {}
    data_run = tmp_path / "datasets"
    (data_run / "data").mkdir(parents=True)
    (data_run / "data/thevault.jsonl").write_text("{}\n")
    config = tmp_path / "recipe.json"
    config.write_text(
        json.dumps(dict(arch="llama", dataset="codesearchnet", data="old-corpus.jsonl"))
    )
    monkeypatch.setattr(
        training, "train", lambda recipe, run: captured.update(recipe) or {}
    )
    main(
        [
            "train",
            "--config",
            str(config),
            "--dataset",
            "thevault",
            "--data-run",
            str(data_run),
        ]
    )
    assert captured["dataset"] == "thevault" and captured["data"].endswith(
        "thevault.jsonl"
    )


def test_invalid_selection_fails_before_download(tmp_path, hub):
    with pytest.raises(SystemExit):
        main(
            [
                "prepare",
                "--dataset",
                "codeparrot",
                "--n-docs",
                "-1",
                "--run",
                str(tmp_path),
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "download",
                "--dataset",
                "codeparrot",
                "--max-files",
                "0",
                "--run",
                str(tmp_path),
            ]
        )
    assert not hub[0]["info"] and not tmp_path.joinpath("sources").exists()


def test_invalid_training_fails_before_automatic_download(tmp_path, hub):
    with pytest.raises(SystemExit):
        main(["train", "--arch", "llama", "--steps", "-1", "--data-run", str(tmp_path)])
    assert not hub[0]["info"]


def test_empty_preparation_does_not_publish_an_empty_dataset(tmp_path, hub):
    with pytest.raises(ValueError, match="no eligible"):
        prepare("codeparrot", tmp_path, exclude_repos=["fixture/train"])
    assert not (tmp_path / "data/codeparrot.jsonl").exists()
    assert not list(tmp_path.glob("data/*.tmp"))


def test_launcher_works_outside_repository(tmp_path):
    import subprocess
    import sys

    launcher = Path(__file__).resolve().parents[1] / "run.py"
    result = subprocess.run(
        [sys.executable, str(launcher), "list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["models"]["gpt_neox"]["dataset"] == "thevault"
