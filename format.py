"""Format Python sources across the repository with Ruff."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true", help="Apply formatting")
    modes.add_argument("--check", action="store_true", help="Check without changing files")
    modes.add_argument("--diff", action="store_true", help="Show proposed changes without writing")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories relative to the repository root; default: whole project",
    )
    args = parser.parse_args(argv)
    paths = []
    for value in args.paths:
        path = (ROOT / value).resolve()
        if not path.is_relative_to(ROOT):
            parser.error(f"Path is outside this repository: {value}")
        if not path.exists():
            parser.error(f"Path does not exist: {value}")
        paths.append(str(path))
    if importlib.util.find_spec("ruff") is None:
        parser.error(
            "Install the formatter first: python -m pip install -r "
            f'"{ROOT / "requirements-dev.txt"}"'
        )
    command = [
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--config",
        str(ROOT / "ruff.toml"),
    ]
    if args.check:
        command.append("--check")
    elif args.diff:
        command.append("--diff")
    command.extend(paths)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
