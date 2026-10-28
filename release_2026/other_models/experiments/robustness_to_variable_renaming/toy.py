"""End-to-end check of the four mechanistic protocols on small random models.

Builds one random two-layer model per architecture (GPT-BigCode, Llama with
grouped-query attention, GPT-NeoX) with a byte-level tokenizer, runs run.py on
each with tiny budgets and checks the outputs. The models are untrained, so the
numbers only test the code paths.

    python toy.py --run RUN --corpus fixtures
"""

import argparse
import json
from pathlib import Path

from run import main, write_json


def execute(root, corpus):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        AutoModelForCausalLM,
        GPTBigCodeConfig,
        GPTNeoXConfig,
        LlamaConfig,
        PreTrainedTokenizerFast,
    )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<eos>": 0, **{s: i + 1 for i, s in enumerate(alphabet)}}
    t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=t, eos_token="<eos>", pad_token="<eos>"
    )
    n = len(tok)
    configs = {
        "gpt_bigcode": GPTBigCodeConfig(
            vocab_size=n,
            n_embd=64,
            n_head=4,
            n_layer=2,
            n_positions=320,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            multi_query=True,
        ),
        "llama": LlamaConfig(
            vocab_size=n,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=96,
            max_position_embeddings=320,
        ),
        "gpt_neox": GPTNeoXConfig(
            vocab_size=n,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=96,
            max_position_embeddings=320,
            rotary_pct=0.25,
            attention_dropout=0,
            hidden_dropout=0,
        ),
    }
    summary = {}
    for arch, cfg in configs.items():
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg).eval()
        checkpoint = root / (arch + "-checkpoint")
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
        dest = root / arch
        main(
            ["--model", str(checkpoint), "--corpus", str(corpus), "--run", str(dest)]
            + ["--n", "40", "--limit", "12", "--kmax", "2", "--ks", "1,2"]
            + [
                "--n-random",
                "2",
                "--n-random-k3",
                "2",
                "--batch",
                "4",
                "--threads",
                "1",
            ]
        )
        results = {
            p.stem: json.loads(p.read_text()) for p in (dest / "results").glob("*.json")
        }
        assert set(results) == {
            "copyscore",
            "meanablate",
            "firsterror",
            "force",
            "metrics",
        }
        assert results["firsterror"]["n_items"] > 0
        for name in ("copyscore", "meanablate"):
            assert len(results[name]["curve"]) == 2
            assert results[name]["ranked_top16"] == results["force"]["ranked_top16"]
        summary[arch] = {
            "items": results["firsterror"]["n_items"],
            "selfcheck": results["force"]["selfcheck"],
            "modules": 4,
        }
    write_json(root / "verification.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument(
        "--corpus", default=str(Path(__file__).resolve().parent / "fixtures")
    )
    a = p.parse_args()
    print(json.dumps(execute(a.run, a.corpus), indent=2))
