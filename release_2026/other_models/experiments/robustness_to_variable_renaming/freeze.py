"""Freeze/verify evaluator closure and exact benchmark inputs before final runs."""

import argparse, json
from pathlib import Path
from ...runtime import digest, write_json

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    data = Path(a.data).resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    paths = sorted(
        [x for x in root.rglob("*.py") if "runs" not in x.relative_to(root).parts]
        + list(data.glob("*.json*"))
        + list((data.parent / "eval_corpus").glob("*.json"))
    )
    state = {str(x): digest(x) for x in paths if x.resolve() != Path(a.out).resolve()}
    if a.verify:
        expected = json.loads(Path(a.out).read_text())
        if expected != state:
            raise ValueError("Frozen code or data changed")
        print(f"verified {len(state)} frozen files")
    else:
        write_json(a.out, state)
        print(f"froze {len(state)} files")


if __name__ == "__main__":
    main()
