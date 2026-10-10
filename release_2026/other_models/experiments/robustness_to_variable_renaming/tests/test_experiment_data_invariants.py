import json
from pathlib import Path
import pytest
from other_models.runtime import write_jsonl, read_jsonl
from other_models.experiments.robustness_to_variable_renaming.data_transformations.renaming import transform


@pytest.mark.parametrize(
    "code",
    [
        'def f(value):\n    # value stays in a comment\n    answer = value + 1\n    return answer, "value"\n',
        "def f(value):\n    def inner(extra):\n        return value + extra\n    answer = inner(2)\n    return answer\n",
        'def f(value):\n    text = "Ï€ and value"\n    answer = value + len(text)\n    return answer\n',
        "def f(value):\n    import math\n    answer = math.sqrt(value)\n    return answer\n",
    ],
)
def test_renaming_semantics_and_noncode(tmp_path, code):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    row = dict(
        id="x",
        code=code,
        repo="a/b",
        split="train",
        meta={"keep": [1, 2]},
        extra="preserve",
    )
    write_jsonl(src, [row])
    stats = transform(src, out, "highdiv", 2)
    new = list(read_jsonl(out))[0]
    before = {}
    after = {}
    exec(code, before)
    exec(new["code"], after)
    assert before["f"](4) == after["f"](4)
    assert {k: v for k, v in row.items() if k != "code"} == {
        k: v for k, v in new.items() if k != "code"
    }
    if "# value" in code:
        assert "# value stays in a comment" in new["code"]
    if '"value"' in code:
        assert '"value"' in new["code"]
    assert stats["changed"] == 1


def test_transform_reproducible(tmp_path):
    row = dict(
        id="x",
        code="def f(value):\n    answer = value + 1\n    return answer\n",
        repo="a/b",
        split="train",
        meta={},
    )
    src = tmp_path / "in.jsonl"
    write_jsonl(src, [row])
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    transform(src, a, "halfvar", 4)
    transform(src, b, "halfvar", 4)
    assert a.read_bytes() == b.read_bytes()


