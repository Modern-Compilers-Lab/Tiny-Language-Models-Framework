"""Boundary around the immutable scoring library; no source-tree imports."""

import importlib, os, sys
from pathlib import Path

VENDOR = Path(__file__).parent / "_vendor"
LIB = VENDOR / "renaming/lib"


def library(name, run=None):
    from .windows_lock import install

    install()
    if run:
        os.environ["RENAMING_WORK"] = str(Path(run).resolve())
        os.environ["WHYSCALE_CACHE"] = str(Path(run).resolve() / "eval_corpus")
    for path in [str(VENDOR), str(LIB)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(name)


def verify():
    from ...runtime import digest

    for line in (LIB / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert digest(LIB / name) == expected, name
