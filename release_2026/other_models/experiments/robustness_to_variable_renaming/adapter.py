"""Load framework checkpoints and expose per-head attention tensors.

Supported architectures are the three families of other_models/models.py:
GPT-BigCode (multi-query or multi-head), Llama (multi-head or grouped-query)
and GPT-NeoX. Dimensions are read from the model config.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import env  # noqa: F401  (puts release_2026/ on sys.path)


def resolve_checkpoint(path):
    """Accept a Hugging Face directory or a training run directory with model/."""
    p = Path(path).expanduser()
    config = p / "config.json"
    is_hf = config.exists() and "model_type" in json.loads(config.read_text())
    if p.is_dir() and not is_hf:
        if (p / "model" / "config.json").exists():
            return str(p / "model")
        raise ValueError(f"No Hugging Face config.json in {p} or {p / 'model'}")
    return str(path)


def load_model(source, device="cpu"):
    """Return (tokenizer, causal LM, Layout) from a path or an HFModel object."""
    if hasattr(source, "causal_lm") and hasattr(source, "tokenizer"):
        interface = source
    else:
        from transformers.utils import logging

        from other_models.models import HFModel

        logging.disable_progress_bar()
        interface = HFModel(resolve_checkpoint(source), device=device, dtype="float32")
    model = interface.causal_lm.eval()
    # Attention weights are only returned by the eager implementation.
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"
    tok = interface.tokenizer
    if not getattr(tok, "is_fast", False):
        raise ValueError("A fast tokenizer with character offsets is required")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    tok.padding_side = "right"
    return tok, model, Layout(model)


class Layout:
    """Per-layer attention modules and output projections of one model.

    Value tensors passed to value_hooks callbacks are [batch, token, head, head_dim],
    with shared key/value heads repeated for every query head.
    """

    def __init__(self, model):
        self.model = model
        c = model.config
        self.arch = c.model_type
        if self.arch not in ("gpt_bigcode", "llama", "gpt_neox"):
            raise ValueError(
                f"Unsupported attention architecture {self.arch!r}; "
                "supported: gpt_bigcode, llama, gpt_neox"
            )
        self.H = c.num_attention_heads
        self.L = c.num_hidden_layers
        self.E = c.hidden_size
        self.D = getattr(c, "head_dim", None) or self.E // self.H
        self.context = c.max_position_embeddings
        if self.arch == "gpt_bigcode":
            self.attentions = [b.attn for b in model.transformer.h]
            self.projections = [m.c_proj for m in self.attentions]
        elif self.arch == "llama":
            self.attentions = [b.self_attn for b in model.model.layers]
            self.projections = [m.o_proj for m in self.attentions]
        else:
            self.attentions = [b.attention for b in model.gpt_neox.layers]
            self.projections = [m.dense for m in self.attentions]
        if len(self.projections) != self.L:
            raise ValueError("Unsupported nonuniform layer layout")
        if any(p.weight.shape[1] != self.H * self.D for p in self.projections):
            raise ValueError("Unsupported output projection or head dimension layout")
        # The scoring code reads GPT-2 style names; model.config itself is not changed.
        self.config = SimpleNamespace(
            n_layer=self.L,
            n_head=self.H,
            n_embd=self.H * self.D,
            n_positions=self.context,
            vocab_size=c.vocab_size,
            multi_query=getattr(c, "multi_query", False),
        )

    def value_module(self, attention):
        if self.arch == "llama":
            return attention.v_proj
        if self.arch == "gpt_bigcode":
            return attention.c_attn
        return attention.query_key_value

    def value_hooks(self, callback):
        handles = []
        for li, m in enumerate(self.attentions):

            def hook(mod, args, out, li=li):
                b, t = out.shape[:2]
                if self.arch == "llama":
                    v = out.reshape(b, t, -1, self.D)
                    v = v.repeat_interleave(self.H // v.shape[2], dim=2)
                elif self.arch == "gpt_bigcode" and self.model.config.multi_query:
                    v = out[..., self.E + self.D : self.E + 2 * self.D]
                    v = v.unsqueeze(2).expand(-1, -1, self.H, -1)
                else:
                    v = out.reshape(b, t, self.H, 3 * self.D)[..., 2 * self.D :]
                callback(li, v)

            handles.append(self.value_module(m).register_forward_hook(hook))
        return handles

    def validate_k(self, ks):
        for k in ks:
            if k < 1 or 2 * k > self.L * self.H:
                raise ValueError(
                    f"k={k} needs at least {2 * k} heads for disjoint controls; "
                    f"the model has {self.L * self.H}"
                )


def select_heads(ranked, a, layout):
    """Restrict ranked (score, layer, head) rows to --layers and --heads."""
    layers = set(a.layers) if getattr(a, "layers", None) else set(range(layout.L))
    heads = set(tuple(x) for x in a.heads) if getattr(a, "heads", None) else None
    if not layers <= set(range(layout.L)):
        raise ValueError("Layer index out of range")
    if heads is not None and not heads <= {
        (l, h) for l in layers for h in range(layout.H)
    }:
        raise ValueError("Head index out of range or outside the selected layers")
    selected = [
        r for r in ranked if r[1] in layers and (heads is None or (r[1], r[2]) in heads)
    ]
    ks = list(range(1, a.kmax + 1)) if hasattr(a, "kmax") else getattr(a, "ks", [])
    if ks and 2 * max(ks) > len(selected):
        raise ValueError(
            "Selected head pool too small for k and disjoint random controls"
        )
    return selected


def source_label(source):
    if isinstance(source, (str, os.PathLike)) or hasattr(source, "__fspath__"):
        return os.path.abspath(os.fspath(source))
    return getattr(source, "name", type(source).__name__)
