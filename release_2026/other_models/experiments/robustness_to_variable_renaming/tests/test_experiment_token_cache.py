"""Check cached token IDs and exact parameter equality after real optimizer updates."""

import importlib.util, json
from pathlib import Path
import torch
from other_models.runtime import setup_run, write_jsonl


def test_cached_training_identical(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
    from other_models.training import train
    from other_models.experiments.robustness_to_variable_renaming.registry import builtins

    builtins()
    from other_models.experiments.robustness_to_variable_renaming import token_cache as module
    t = Tokenizer(
        models.WordLevel(
            {
                "<eos>": 0,
                "<unk>": 1,
                "def": 2,
                "f": 3,
                "(": 4,
                ")": 5,
                ":": 6,
                "return": 7,
                "x": 8,
                "+": 9,
                "1": 10,
            },
            unk_token="<unk>",
        )
    )
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=t, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>"
    )
    tok.save_pretrained(tmp_path / "tok")
    data = tmp_path / "data.jsonl"
    texts = ["def f(x): return x + 1", "return x", "def f(x): return 1"]
    write_jsonl(data, [dict(code=x) for x in texts])
    cfg = dict(
        arch="llama",
        scale="small",
        model=dict(hidden=64, layers=1, heads=1, context=64),
        tokenizer=str(tmp_path / "tok"),
        data=str(data),
        device="cpu",
        seed=2,
        steps=2,
        warmup=1,
        batch_size=2,
        max_len=32,
        threads=1,
    )
    direct = setup_run(tmp_path / "direct", cfg)
    cached = setup_run(tmp_path / "cached", cfg)
    prefix = cached / "data/tokens"
    module.prepare(cfg, cached, prefix, batch_size=2)
    table = module.TokenTable(prefix)
    assert [row.tolist() for row in table] == tok(
        texts, add_special_tokens=False, truncation=True, max_length=32
    )["input_ids"]
    train(cfg, direct)
    module.train_cached(cfg, cached, prefix)
    a = AutoModelForCausalLM.from_pretrained(direct / "model")
    b = AutoModelForCausalLM.from_pretrained(cached / "model")
    assert all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )


def test_cache_offsets_corruption_rejected(tmp_path):
    import pytest, numpy as np

    test_cached_training_identical(tmp_path)
    from other_models.experiments.robustness_to_variable_renaming import token_cache as module
    cached = tmp_path / "cached"
    prefix = cached / "data/tokens"
    cfg = json.loads((cached / "config.json").read_text())
    offsets = np.load(str(prefix) + ".offsets.npy")
    offsets[1] = offsets[-1]
    np.save(str(prefix) + ".offsets.npy", offsets)
    with pytest.raises(ValueError, match="offset"):
        module.train_cached(cfg, cached, prefix)
