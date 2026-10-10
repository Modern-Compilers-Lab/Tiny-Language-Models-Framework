"""Execute explicit experiment cells sequentially with local logs and status records."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ...runtime import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def execute(cells_path, status_path=None):
    cells_path = Path(cells_path).resolve()
    cells = json.loads(cells_path.read_text(encoding="utf-8"))
    names = [cell["name"] for cell in cells]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate campaign cell names")
    frozen_path = cells_path.parent / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if any(digest(path) != checksum for path, checksum in frozen.items()):
            raise ValueError("Frozen campaign inputs changed")
    status_path = Path(status_path) if status_path else cells_path.with_suffix(".status.json")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    completed = []
    for cell in cells:
        run = Path(cell["run"]).resolve()
        (run / "logs").mkdir(parents=True, exist_ok=True)
        if cell.get("script"):
            command = [sys.executable, str(Path(cell["script"]).resolve())]
        else:
            command = [sys.executable, "-m", cell.get("module", "other_models")]
        command.extend(map(str, cell["args"]))
        env["ORX_METRICS_PATH"] = str(run / "results/cell-metrics.json")
        with (run / "logs/driver.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
        completed.append(dict(name=cell["name"], run=str(run), returncode=result.returncode, cells_sha256=digest(cells_path)))
        write_json(status_path, completed)
        print(cell["name"], result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f"Cell failed: {cell['name']}; see {run / 'logs/driver.log'}")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--status")
    args = parser.parse_args(argv)
    return execute(args.cells, args.status)


if __name__ == "__main__":
    main()
