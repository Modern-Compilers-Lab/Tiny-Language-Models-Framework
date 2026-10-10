"""Full or explicitly selected raw sources pinned to Hub commits."""

import fnmatch
import hashlib
import json
import urllib.request
from pathlib import Path, PurePosixPath

from .runtime import digest, write_json

REPOSITORIES = {
    "codesearchnet": "code-search-net/code_search_net",
    "thevault": "Fsoft-AIC/the-vault-function",
    "codeparrot": "codeparrot/codeparrot-clean",
}
MANIFEST = ".source-manifest.json"
TRAIN_PATTERNS = {
    "codesearchnet": "python/train-*.parquet",
    "thevault": "data/train/*/python-*.parquet",
    "codeparrot": "file-*.json.gz",
}


def source_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid source filename: {name}")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Source filename escapes its directory")
    return path


def verify_source(directory, name, kind=None, require_full=True):
    directory = Path(directory)
    meta = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if (
        not meta.get("complete")
        or meta["name"] != name
        or (kind and meta["kind"] != kind)
    ):
        raise ValueError("Source manifest does not describe this complete dataset")
    if require_full and not meta.get("full", True):
        raise ValueError("Source is a subset; use a new run for a full download")
    if meta["kind"] == "hf_repository" and meta.get("repo_id") != REPOSITORIES.get(
        name
    ):
        raise ValueError("Source repository does not match this dataset")
    if not meta.get("files"):
        raise ValueError("Complete source has no files")
    for item in meta["files"]:
        path = source_file(directory, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Incomplete source: {path}; rerun its download command")
        if item.get("sha256") and digest(path) != item["sha256"]:
            raise ValueError(f"Changed source file: {path}")
    return meta


def download_dataset(
    name, run, revision="main", dry_run=False, max_files=None, allow_patterns=None
):
    """Download every repository file by default; record explicit subsets."""
    from huggingface_hub import HfApi, snapshot_download

    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be positive")
    repo = REPOSITORIES[name]
    patterns = list(allow_patterns or [])
    if max_files and not patterns:
        patterns = [TRAIN_PATTERNS[name]]
    selection = dict(max_files=max_files, allow_patterns=patterns)
    full = not patterns and max_files is None
    run = Path(run)
    dest = run / "sources" / name
    marker = dest / MANIFEST
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (
            previous.get("repo_id") != repo
            or previous.get("requested_revision") != revision
            or previous.get("selection", dict(max_files=None, allow_patterns=[]))
            != selection
        ):
            raise ValueError("Source request changed; choose a new run")
        # Reuse the pinned revision on retry, even if the branch has advanced.
        commit = previous["revision"]
        try:
            cached = verify_source(dest, name, "hf_repository", require_full=full)
            if dry_run:
                return dict(
                    name=name,
                    revision=cached["revision"],
                    file_count=len(cached["files"]),
                    total_bytes=cached["total_bytes"],
                    already_downloaded=True,
                    full=cached.get("full", True),
                )
            return dest
        except ValueError:
            pass
    else:
        commit = revision
    info = HfApi().dataset_info(repo, revision=commit, files_metadata=True)
    entries = []
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        if patterns and not any(
            fnmatch.fnmatchcase(sibling.rfilename, p) for p in patterns
        ):
            continue
        if sibling.rfilename == MANIFEST:
            raise ValueError("Repository uses the reserved source-manifest filename")
        source_file(dest, sibling.rfilename)
        entries.append(dict(path=sibling.rfilename, expected_bytes=sibling.size))
    if max_files:
        entries = entries[:max_files]
    if not entries:
        raise ValueError("Dataset repository has no files")
    if dry_run:
        return dict(
            name=name,
            repo_id=repo,
            revision=info.sha,
            file_count=len(entries),
            total_bytes=sum(item["expected_bytes"] or 0 for item in entries),
            sizes_complete=all(item["expected_bytes"] is not None for item in entries),
            already_downloaded=False,
            full=full,
        )
    meta = dict(
        name=name,
        kind="hf_repository",
        repo_id=repo,
        requested_revision=revision,
        revision=info.sha,
        scope="entire upstream repository, all languages and splits"
        if full
        else "explicit file selection",
        full=full,
        selection=selection,
        complete=False,
        files=entries,
    )
    write_json(marker, meta)
    options = {} if full else {"allow_patterns": [item["path"] for item in entries]}
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=info.sha,
        local_dir=str(dest),
        cache_dir=str(run / "cache" / "hub"),
        **options,
    )
    for item in entries:
        path = source_file(dest, item["path"])
        if not path.is_file():
            raise ValueError(f"Snapshot omitted {item['path']}")
        item["bytes"] = path.stat().st_size
        if (
            item["expected_bytes"] is not None
            and item["bytes"] != item["expected_bytes"]
        ):
            raise ValueError(f"Incomplete downloaded file: {path}")
    meta["complete"] = True
    meta["total_bytes"] = sum(item["bytes"] for item in entries)
    write_json(marker, meta)
    write_json(
        run / "results" / f"download-{name}.json",
        dict(meta, directory=str(dest.resolve())),
    )
    write_json(
        run / "results" / "download.json", dict(meta, directory=str(dest.resolve()))
    )
    return dest


def download_file(url, target, budget=None):
    """Stream to disk; resume only with a matching server validator and range."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    state_path = part.with_name(part.name + ".json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    offset = (
        part.stat().st_size
        if part.exists() and state.get("url") == url and state.get("validator")
        else 0
    )
    if offset and state.get("total_bytes") == offset:
        checksum = digest(part)
        part.replace(target)
        state_path.unlink(missing_ok=True)
        return dict(bytes=offset, sha256=checksum)
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers.update(Range=f"bytes={offset}-", **{"If-Range": state["validator"]})
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as response:
        resumed = offset and response.status == 206
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError("Invalid download resume range")
            total = int(content_range.split("/")[-1])
            validator = response.headers.get("ETag") or response.headers.get(
                "Last-Modified"
            )
            if validator and validator != state["validator"]:
                raise ValueError("Source changed during resumed download")
        else:
            if response.status != 200:
                raise ValueError(f"Unexpected download status: {response.status}")
            offset = 0
            total = (
                int(response.headers["Content-Length"])
                if response.headers.get("Content-Length")
                else None
            )
        validator = response.headers.get("ETag") or response.headers.get(
            "Last-Modified"
        )
        if (
            budget is not None
            and budget.maximum is not None
            and total is not None
            and total - offset > budget.maximum - budget.used
        ):
            raise ValueError(
                "Download budget exceeded; explicitly raise --max-download-mb or use 0"
            )
        write_json(state_path, dict(url=url, validator=validator, total_bytes=total))
        checksum = hashlib.sha256()
        if offset:
            with part.open("rb") as previous:
                for block in iter(lambda: previous.read(1 << 20), b""):
                    checksum.update(block)
        count = offset
        with part.open("ab" if offset else "wb") as output:

            def read():
                size = 1 << 20
                if budget is not None and budget.maximum is not None:
                    size = min(size, budget.maximum - budget.used + 1)
                block = response.read(size)
                if budget is not None:
                    budget.consume(len(block))
                return block

            for block in iter(read, b""):
                output.write(block)
                checksum.update(block)
                count += len(block)
        if total is not None and count != total:
            raise IOError(
                f"Incomplete download: expected {total} bytes, received {count}"
            )
    part.replace(target)
    state_path.unlink(missing_ok=True)
    return dict(bytes=count, sha256=checksum.hexdigest())


def download_files(name, run, files, budget=None):
    """Persist complete release files; do not extract archive contents."""
    run = Path(run)
    dest = run / "sources" / name
    marker = dest / MANIFEST
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    old = {item["path"]: item for item in previous.get("files", [])}
    entries = []
    for relative, url in files:
        target = source_file(dest, relative)
        item = old.get(relative)
        if not (
            item
            and item.get("url") == url
            and target.is_file()
            and target.stat().st_size == item["bytes"]
            and digest(target) == item["sha256"]
        ):
            item = dict(path=relative, url=url, **download_file(url, target, budget))
        entries.append(item)
        write_json(
            marker, dict(name=name, kind="http_files", complete=False, files=entries)
        )
    meta = dict(
        name=name,
        kind="http_files",
        complete=True,
        files=entries,
        scope="complete Python benchmark release files and required label sources",
        total_bytes=sum(item["bytes"] for item in entries),
    )
    write_json(marker, meta)
    write_json(
        run / "results" / "download.json", dict(meta, directory=str(dest.resolve()))
    )
    return dest
