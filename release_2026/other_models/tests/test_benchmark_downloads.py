"""Complete benchmark release and subset paths, with no public downloads."""

import hashlib
import io
import json
import tarfile
import zipfile

import pytest
from other_models.cli import main
from other_models.runtime import write_json
from other_models.source_downloads import download_file, verify_source


def parquet(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), output)
    return output.getvalue()


def test_complete_partial_file_finalizes_without_another_request(tmp_path, monkeypatch):
    import urllib.request

    target = tmp_path / "source"
    (tmp_path / "source.part").write_bytes(b"complete")
    write_json(
        tmp_path / "source.part.json",
        dict(url="https://fixture.invalid/data", validator='"same"', total_bytes=8),
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("Already received the complete body"),
    )
    assert download_file("https://fixture.invalid/data", target)["bytes"] == 8
    assert target.read_bytes() == b"complete"


class Response(io.BytesIO):
    def __init__(self, payload, status=200, **headers):
        super().__init__(payload)
        self.status = status
        self.headers = {
            "Content-Length": str(len(payload)),
            "ETag": '"fixture"',
            **headers,
        }


def test_http_partial_download_resumes_with_validator(tmp_path, monkeypatch):
    import urllib.request

    requests = []

    def fetch(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return Response(b"abcd", **{"Content-Length": "8"})
        assert request.get_header("Range") == "bytes=4-"
        assert request.get_header("If-range") == '"fixture"'
        return Response(b"efgh", 206, **{"Content-Range": "bytes 4-7/8"})

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    target = tmp_path / "archive.zip"
    with pytest.raises(IOError, match="Incomplete"):
        download_file("https://fixture.invalid/archive", target)
    assert not target.exists()
    result = download_file("https://fixture.invalid/archive", target)
    assert target.read_bytes() == b"abcdefgh"
    assert result["sha256"] == hashlib.sha256(b"abcdefgh").hexdigest()


def test_http_server_ignoring_resume_restarts_cleanly(tmp_path, monkeypatch):
    import urllib.request

    target = tmp_path / "data.zip"
    (tmp_path / "data.zip.part").write_bytes(b"OLD")
    write_json(
        tmp_path / "data.zip.part.json",
        dict(url="https://fixture.invalid/data", validator='"old"'),
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: Response(b"new data")
    )
    download_file("https://fixture.invalid/data", target)
    assert target.read_bytes() == b"new data"


@pytest.fixture
def releases():
    bifi = io.BytesIO()
    with zipfile.ZipFile(bifi, "w") as archive:
        for split in ("train", "dev"):
            archive.writestr(f"round0/data_paired/{split}.bad", "x = ( 1\n" * 3)
            archive.writestr(f"round0/data_paired/{split}.good", "x = ( 1 )\n" * 3)
        for chunk in (3, 4):
            archive.writestr(
                f"orig_bad_code/orig.{chunk}.bad", "x = ( 1 <NEWLINE>\n" * 3
            )
            archive.writestr(
                f"orig_bad_code/orig.{chunk}.id",
                "".join(f"{chunk}:{i}\n" for i in range(3)),
            )
    types = io.BytesIO()
    project = {
        "owner/repo.with.dots": {
            "src_files": {
                f"{split}.py": dict(
                    set=split,
                    untyped_seq=f"def f_{split}(x): return x",
                    variables={"value": "int"},
                    funcs=[],
                    classes=[],
                )
                for split in ("train", "valid", "test")
            }
        }
    }
    with tarfile.open(fileobj=types, mode="w:gz") as archive:
        raw = json.dumps(project).encode()
        entry = tarfile.TarInfo("processed_projects_clean/project.json")
        entry.size = len(raw)
        archive.addfile(entry, io.BytesIO(raw))
    return dict(
        bifi={"data.zip": bifi.getvalue()},
        manytypes4py={"ManyTypes4PyDataset-v0.7.tar.gz": types.getvalue()},
        codexglue_py150={
            "python/train-00000-of-00002.parquet": parquet(
                [dict(code=["return", f"v{i}"]) for i in range(2)]
            ),
            "python/train-00001-of-00002.parquet": parquet(
                [dict(code=["return", f"w{i}"]) for i in range(45002)]
            ),
            "python/test-00000-of-00001.parquet": parquet(
                [dict(code=["return", f"t{i}"]) for i in range(3)]
            ),
        },
    )


@pytest.mark.parametrize(
    "benchmark",
    [
        "codexglue_py150",
        "codexglue_cloze_all",
        "codexglue_cloze_maxmin",
        "bifi",
        "manytypes4py",
        "cruxeval_input",
        "cruxeval_output",
    ],
)
def test_full_benchmark_download_and_offline_reuse(
    tmp_path, monkeypatch, releases, benchmark
):
    import urllib.request

    from other_models.benchmark_suite import BENCHMARKS
    from other_models.benchmark_suite.sources import source_files

    files = releases.get(benchmark)
    if benchmark.startswith("cruxeval_"):
        files = {
            "cruxeval.jsonl": "".join(
                json.dumps(
                    dict(
                        id=str(i),
                        code="def f(x):\n    return x+1",
                        input="1",
                        output="2",
                    )
                )
                + "\n"
                for i in range(140)
            ).encode()
        }
    if benchmark.startswith("codexglue_cloze"):
        all_words = benchmark.endswith("_all")
        prefix, gold = ("identity", "token0") if all_words else ("return", "max")
        files = {
            "clozeTest.json": json.dumps(
                [dict(idx="one", pl_tokens=[prefix, "<mask>"])]
            ).encode(),
            "csn/test-00000-of-00001.parquet": parquet(
                [dict(func_code_tokens=[prefix, gold])]
            ),
            "csn/validation-00000-of-00001.parquet": parquet(
                [dict(func_code_tokens=["assert", "False"])]
            ),
        }
        if all_words:
            files["cloze_test_words.txt"] = "\n".join(
                f"token{i}" for i in range(930)
            ).encode()
    routes = {url: files[path] for path, url in source_files(benchmark)}
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        return Response(routes[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    raw_run = tmp_path / "raw"
    main(["benchmark", "download", "--benchmark", benchmark, "--run", str(raw_run)])
    raw = raw_run / "sources" / benchmark
    assert verify_source(raw, benchmark, "http_files")["complete"]
    assert len(calls) == len(routes)
    # Repeat raw download reuses verified files without network access.
    main(["benchmark", "download", "--benchmark", benchmark, "--run", str(raw_run)])
    assert len(calls) == len(routes)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("Offline preparation attempted network access"),
    )
    run = tmp_path / "prepared"
    main(
        [
            "benchmark",
            "prepare",
            "--benchmark",
            benchmark,
            "--source",
            str(raw),
            "--run",
            str(run),
        ]
    )
    data = run / "data" / benchmark
    meta = json.loads((data / "manifest.json").read_text())
    assert not meta["subset"] and meta["full"] and meta["max_download_mb"] == 0
    assert set(meta["splits"]) == (
        {"train", "dev", "test"} if BENCHMARKS[benchmark].trainable else {"test"}
    )
    if benchmark.startswith("cruxeval_"):
        assert meta["splits"]["test"]["count"] == 140
    if benchmark == "codexglue_py150":
        assert meta["splits"]["train"]["count"] == 45002
        assert meta["splits"]["dev"]["count"] == 2


@pytest.mark.parametrize("flags", [[], ["--full"], ["--limit", "0"]])
def test_full_preparation_downloads_complete_source_automatically(
    tmp_path, monkeypatch, flags
):
    import urllib.request

    from other_models.benchmark_suite.sources import source_files

    payload = (
        json.dumps(
            dict(id="one", code="def f(x): return x", input="1", output="1")
        ).encode()
        + b"\n"
    )
    url = source_files("cruxeval_output")[0][1]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: (
            Response(payload)
            if request.full_url == url
            else pytest.fail("Unexpected URL")
        ),
    )
    main(
        [
            "benchmark",
            "prepare",
            "--benchmark",
            "cruxeval_output",
            *flags,
            "--run",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "sources/cruxeval_output/cruxeval.jsonl").read_bytes() == payload
    meta = json.loads((tmp_path / "data/cruxeval_output/manifest.json").read_text())
    assert meta["download_bytes"] == len(payload)


def test_unlimited_budget_and_explicit_full_download_cap(tmp_path, monkeypatch):
    import urllib.request

    from other_models.benchmark_suite.downloads import BoundedStream, Budget

    budget = Budget(0)
    assert BoundedStream(io.BytesIO(b"12345"), budget).read() == b"12345"
    assert budget.used == 5
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response(b"12345"))
    with pytest.raises(ValueError, match="budget"):
        download_file("https://fixture.invalid/data", tmp_path / "limited", Budget(4))
    assert not (tmp_path / "limited").exists()


def test_positive_benchmark_limit_preserves_sampled_defaults(tmp_path):
    from other_models.benchmark_suite.data import prepare

    source = tmp_path / "source"
    source.mkdir()
    (source / "test.jsonl").write_text(
        "".join(
            json.dumps(dict(id=str(i), code=["x"], split="test")) + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    output = prepare("codexglue_py150", tmp_path / "prepared", source, limit=2)
    meta = json.loads((output / "manifest.json").read_text())
    assert meta["subset"] and not meta["full"]
    assert meta["max_download_mb"] == 64
    assert set(meta["splits"]) == {"test"}
    assert meta["splits"]["test"]["count"] == 2
    assert not (tmp_path / "prepared/sources").exists()
