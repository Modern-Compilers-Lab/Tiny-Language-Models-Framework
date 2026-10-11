import hashlib
import json
import os
from pathlib import Path


def setup_run(path, config):
    """Keep benchmark outputs local and preserve the recorded run recipe."""
    run = Path(path).resolve()
    config = json.loads(json.dumps(config, default=str))
    snapshot = run / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Run config changed; choose a new run directory: {run}")
    for sub in ("logs", "results", "tmp", "data", "cache"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        write_json(snapshot, config)
    return run


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
