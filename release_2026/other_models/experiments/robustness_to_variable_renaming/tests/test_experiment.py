import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from adapter import Layout, resolve_checkpoint, select_heads

HERE = Path(__file__).resolve().parents[1]


def tiny_model(arch, kv):
    from transformers import (
        AutoModelForCausalLM,
        GPTBigCodeConfig,
        GPTNeoXConfig,
        LlamaConfig,
    )

    common = dict(
        vocab_size=40,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    if arch == "gpt_bigcode":
        cfg = GPTBigCodeConfig(
            vocab_size=40,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_positions=64,
            multi_query=kv == 1,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
        )
    elif arch == "llama":
        cfg = LlamaConfig(**common, intermediate_size=96, num_key_value_heads=kv)
    else:
        cfg = GPTNeoXConfig(
            **common, intermediate_size=96, rotary_pct=0.25, hidden_dropout=0
        )
    cfg._attn_implementation = "eager"
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.mark.parametrize(
    "arch,kv",
    [
        ("gpt_bigcode", 1),
        ("gpt_bigcode", 4),
        ("llama", 4),
        ("llama", 2),
        ("gpt_neox", 4),
    ],
)
def test_value_hooks_reconstruct_head_outputs(arch, kv):
    torch.manual_seed(3)
    torch.set_num_threads(1)
    model = tiny_model(arch, kv)
    layout = Layout(model)
    vals, captured = {}, {}
    handles = layout.value_hooks(lambda l, v: vals.__setitem__(l, v))
    for l, p in enumerate(layout.projections):
        handles.append(
            p.register_forward_pre_hook(
                lambda mod, args, l=l: captured.__setitem__(l, args[0].detach().clone())
            )
        )
    ids = torch.tensor([[1, 3, 7, 8, 2], [4, 8, 1, 2, 3]])
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    for l in range(2):
        rebuilt = torch.einsum("bhqk,bkhd->bqhd", out.attentions[l], vals[l]).flatten(
            -2
        )
        torch.testing.assert_close(rebuilt, captured[l], atol=2e-6, rtol=2e-5)
    for h in handles:
        h.remove()
    assert not any(p._forward_pre_hooks for p in layout.projections)


def test_unsupported_architecture_and_checkpoint_resolution(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        Layout(SimpleNamespace(config=SimpleNamespace(model_type="unknown")))
    (tmp_path / "config.json").write_text('{"training": true}')
    (tmp_path / "model").mkdir()
    (tmp_path / "model/config.json").write_text("{}")
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path / "model")
    (tmp_path / "model/config.json").unlink()
    with pytest.raises(ValueError, match="No Hugging Face"):
        resolve_checkpoint(str(tmp_path))


def test_head_selection_and_control_budget():
    layout = SimpleNamespace(L=2, H=4)
    pairs = [(l, h) for l in range(2) for h in range(4)]
    ranked = [(1 - i / 10, l, h) for i, (l, h) in enumerate(pairs)]
    a = SimpleNamespace(layers=[1], heads=None, kmax=2)
    assert {r[1] for r in select_heads(ranked, a, layout)} == {1}
    a.kmax = 3
    with pytest.raises(ValueError, match="too small"):
        select_heads(ranked, a, layout)
    a.layers = [2]
    with pytest.raises(ValueError, match="out of range"):
        select_heads(ranked, a, layout)


def test_rates_and_aggregation():
    from report import aggregate
    from score import rate

    assert rate(2, 4) == 0.5
    assert rate(0, 0) is None
    with pytest.raises(ValueError):
        rate(3, 2)
    assert aggregate([1, 2, 3]) == dict(mean=2, sd=1, n_seeds=3)
    assert aggregate([None, 2]) == dict(mean=2, sd=None, n_seeds=1)
    assert aggregate([None]) == dict(mean=None, sd=None, n_seeds=0)
    with pytest.raises(ValueError):
        aggregate([float("nan")])


def test_library_matches_checksums():
    import env

    env.verify_lib()


def test_json_safe_raw_and_quiet_command():
    import env

    clean, paths = env.json_safe_raw({"a": float("nan"), "b": [1.0, float("inf")]})
    assert clean == {"a": None, "b": [1.0, None]} and paths == ["raw.a", "raw.b[1]"]

    def noisy(x):
        print("progress %.4f" % None)
        return x + 1

    assert env.quiet_command(noisy)(1) == 2


@pytest.mark.parametrize("script", sorted(p.name for p in HERE.glob("exp_*.py")))
def test_entry_point_help(script):
    environment = dict(os.environ, PYTHONHASHSEED="0")
    result = subprocess.run(
        [sys.executable, str(HERE / script), "--help"],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--model", "--corpus", "--run"):
        assert option in result.stdout


def test_toy_models_run_all_mechanistic_protocols(tmp_path):
    environment = dict(os.environ, PYTHONHASHSEED="0")
    result = subprocess.run(
        [sys.executable, str(HERE / "toy.py"), "--run", str(tmp_path)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    summary = json.loads((tmp_path / "verification.json").read_text())
    assert set(summary) == {"gpt_bigcode", "llama", "gpt_neox"}
    for cell in summary.values():
        assert cell["items"] > 0
        assert cell["selfcheck"]["max_abs_diff"] < 5e-5
