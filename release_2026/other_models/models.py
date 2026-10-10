"""Original codelab architecture presets and Hugging Face checkpoint loading."""

PRESETS = {
    "small": dict(hidden=192, layers=20, heads=3, context=1024),
    "large": dict(hidden=768, layers=20, heads=12, context=2048),
}

# Preferred preparation recipes; model factories remain independent of corpora.
MODEL_DEFAULTS = {
    "gpt_bigcode": {
        "name": "Small StarCoder",
        "dataset": "codesearchnet",
        "parameters": 18919552,
    },
    "llama": {
        "name": "Small Llama",
        "dataset": "codesearchnet",
        "parameters": 18292416,
    },
    "gpt_neox": {
        "name": "Small GPT-NeoX",
        "dataset": "thevault",
        "parameters": 18334848,
    },
}

TOKENIZER = "bigcode/tiny_starcoder_py"
ALIASES = {"starcoder": "gpt_bigcode", "gpt-neox": "gpt_neox"}


def resolve_arch(arch):
    arch = ALIASES.get(arch, arch)
    if arch not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown architecture: {arch}")
    return arch


def resolve_dataset(arch=None, dataset=None):
    """An explicit corpus overrides the optional built-in model preference."""
    if dataset is not None:
        if dataset not in ("codesearchnet", "thevault", "codeparrot"):
            raise ValueError(f"Unknown dataset: {dataset}")
        return dataset
    arch = resolve_arch(arch)
    return MODEL_DEFAULTS[arch]["dataset"]


def build_config(arch, vocab_size, scale="small", **overrides):
    from transformers import GPTBigCodeConfig, GPTNeoXConfig, LlamaConfig

    arch = resolve_arch(arch)
    if scale not in PRESETS:
        raise ValueError(f"Unknown model scale: {scale}")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    unknown = set(overrides) - set(PRESETS[scale])
    if unknown:
        raise ValueError(f"Unknown model overrides: {sorted(unknown)}")
    p = dict(PRESETS[scale])
    p["context"] = 8192 if arch == "gpt_bigcode" else p["context"]
    p.update({k: v for k, v in overrides.items() if v is not None})
    hidden, layers, heads, context = (
        p[k] for k in ("hidden", "layers", "heads", "context")
    )
    if min(hidden, layers, heads, context) < 1:
        raise ValueError("Model dimensions must be positive")
    if hidden % heads:
        raise ValueError("hidden must be divisible by heads")
    shared = dict(
        vocab_size=vocab_size, tie_word_embeddings=True, bos_token_id=0, eos_token_id=0
    )
    if arch == "gpt_bigcode":
        return GPTBigCodeConfig(
            **shared,
            n_embd=hidden,
            n_layer=layers,
            n_head=heads,
            n_inner=4 * hidden,
            n_positions=context,
            multi_query=True,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
        )
    common = dict(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        max_position_embeddings=context,
    )
    if arch == "llama":
        return LlamaConfig(
            **shared,
            **common,
            intermediate_size=8 * hidden // 3,
            num_key_value_heads=heads,
            rms_norm_eps=1e-5,
            attention_dropout=0.1,
        )
    if arch == "gpt_neox":
        return GPTNeoXConfig(
            **shared,
            **common,
            intermediate_size=4 * hidden,
            rotary_pct=0.25,
            use_parallel_residual=True,
            layer_norm_eps=1e-5,
            attention_dropout=0.1,
            hidden_dropout=0.1,
        )
    raise ValueError(arch)


def resolve_device(device="auto"):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return str(resolved)


def build_model(arch, vocab_size=49152, **overrides):
    """Create the original codelab architecture with random weights."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_config(build_config(arch, vocab_size, **overrides))


def load_model(path, device="auto", dtype="float32", local_files_only=False):
    """Load a saved HF model and its tokenizer; return (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=getattr(torch, dtype), local_files_only=local_files_only
        )
        .to(device)
        .eval()
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    return model, tokenizer


class HFModel:
    def __init__(self, path, device="auto", dtype="float32", batch=16):
        if batch < 1:
            raise ValueError("batch must be positive")
        self.causal_lm, self.tokenizer = load_model(path, device, dtype)
        self.device = str(self.causal_lm.device)
        self.batch = batch
        self.tokenizer.padding_side = "right"
        self.context = self.causal_lm.config.max_position_embeddings

    def token_predictions(self, texts, max_length=320):
        import torch

        tok = self.tokenizer
        out = []
        for start in range(0, len(texts), self.batch):
            enc = tok(
                texts[start : start + self.batch],
                padding=True,
                truncation=True,
                max_length=min(max_length, self.context),
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = enc.pop("offset_mapping").tolist()
            lens = enc["attention_mask"].sum(1).tolist()
            with torch.inference_mode():
                pred = (
                    self.causal_lm(**{k: v.to(self.device) for k, v in enc.items()})
                    .logits.argmax(-1)
                    .cpu()
                    .tolist()
                )
            for ids, off, pr, n in zip(enc["input_ids"].tolist(), offsets, pred, lens):
                out.append((ids[:n], off[:n], pr[:n]))
        return out

    def sequence_scores(self, texts, max_length=320):
        import torch

        out = []
        for start in range(0, len(texts), self.batch):
            enc = self.tokenizer(
                texts[start : start + self.batch],
                padding=True,
                truncation=True,
                max_length=min(max_length, self.context),
                return_tensors="pt",
                add_special_tokens=False,
            )
            b = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.causal_lm(**b).logits[:, :-1].float()
                lp = (
                    logits.log_softmax(-1)
                    .gather(-1, b["input_ids"][:, 1:, None])
                    .squeeze(-1)
                )
                mask = b["attention_mask"][:, 1:]
                n = mask.sum(1)
                if (n == 0).any():
                    raise ValueError("fewer than two code tokens")
                out.extend(((lp * mask).sum(1) / n).cpu().tolist())
        return out

    def continuation_scores(self, prefix, choices):
        import torch

        scores = []
        tok = self.tokenizer
        # Move trailing prompt whitespace into answer encoding (canonical code BPE boundary).
        context = prefix.rstrip()
        boundary_space = prefix[len(context) :]
        pre = tok.encode(context, add_special_tokens=False) or [
            tok.bos_token_id or tok.eos_token_id
        ]
        encoded = [
            tok.encode(boundary_space + choice, add_special_tokens=False)
            for choice in choices
        ]
        for start in range(0, len(encoded), self.batch):
            group = encoded[start : start + self.batch]
            seqs = []
            bounds = []
            for ans in group:
                if not ans or len(ans) >= self.context:
                    raise ValueError("invalid answer length")
                ctx = pre[-(self.context - len(ans)) :]
                seqs.append(ctx + ans)
                bounds.append(len(ctx))
            batch = tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = self.causal_lm(
                    **{k: v.to(self.device) for k, v in batch.items()}
                ).logits
            for j, (ans, bound) in enumerate(zip(group, bounds)):
                lp = logits[j, bound - 1 : bound - 1 + len(ans)].float().log_softmax(-1)
                scores.append(
                    lp.gather(-1, torch.tensor(ans, device=self.device)[:, None])
                    .sum()
                    .item()
                )
        return scores

    def generate_many(self, prompts, max_new_tokens=32):
        import torch

        tok = self.tokenizer
        out = []
        old = tok.padding_side
        tok.padding_side = "left"
        try:
            for start in range(0, len(prompts), self.batch):
                ids = [
                    tok.encode(p, add_special_tokens=False)[
                        -max(1, self.context - max_new_tokens) :
                    ]
                    or [tok.eos_token_id]
                    for p in prompts[start : start + self.batch]
                ]
                batch = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
                width = batch["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = self.causal_lm.generate(
                        **{k: v.to(self.device) for k, v in batch.items()},
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                out.extend(
                    tok.batch_decode(generated[:, width:], skip_special_tokens=True)
                )
        finally:
            tok.padding_side = old
        return out

    def generate(self, prompt, max_new_tokens=32):
        import torch

        tok = self.tokenizer
        ids = tok.encode(prompt, add_special_tokens=False) or [tok.eos_token_id]
        ids = ids[-max(1, self.context - max_new_tokens) :]
        b = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            out = self.causal_lm.generate(
                b,
                attention_mask=torch.ones_like(b),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0, len(ids) :], skip_special_tokens=True)

    def sample(self, prompt, n=1, max_new_tokens=128, temperature=0.0, top_p=0.95):
        import torch

        if n < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation settings")
        if n > 1 and temperature == 0:
            raise ValueError("multiple samples require a positive temperature")
        if not 0 < max_new_tokens < self.context:
            raise ValueError("max_new_tokens must be less than model context")
        tok = self.tokenizer
        ids = tok.encode(prompt, add_special_tokens=False) or [tok.eos_token_id]
        if len(ids) + max_new_tokens > self.context:
            raise ValueError(
                "benchmark prompt plus generation exceeds context; increase max-len or reduce max-new-tokens"
            )
        b = torch.tensor([ids], device=self.device)
        options = dict(do_sample=temperature > 0)
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
        outputs = []
        # Sequential generation avoids multiplying laptop activation memory by n.
        for _ in range(n):
            with torch.inference_mode():
                out = self.causal_lm.generate(
                    b,
                    attention_mask=torch.ones_like(b),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    **options,
                )
            outputs.append(tok.decode(out[0, len(ids) :], skip_special_tokens=True))
        return outputs

    def lexical_predictions(self, tokens, max_length=320):
        """Score all PY150 lexical targets using overlapping bounded contexts."""
        import torch

        from .benchmark_suite.utils import render

        text, spans = render(tokens)
        tok = self.tokenizer
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if bos is None:
            raise ValueError("PY150 evaluation requires a BOS or EOS token")
        context = min(max_length, self.context)
        if context < 2:
            raise ValueError("PY150 context must be at least two")
        seq = [bos] + ids
        predictions = [None] * len(ids)
        stride = max(1, context // 2)
        for start in range(1, len(seq), stride):
            end = min(start + stride, len(seq))
            left = max(0, end - context)
            batch = torch.tensor([seq[left:end]], device=self.device)
            with torch.inference_mode():
                pred = self.causal_lm(batch).logits.argmax(-1)[0].cpu().tolist()
            for k in range(start, end):
                predictions[k - 1] = pred[k - left - 1]
        out = []
        for j, span in enumerate(spans):
            if span is None:
                continue
            a, b = span
            pieces = [
                k
                for k, (x, y) in enumerate(offsets)
                if y > a and x < b and text[x:y].strip()
            ]
            if not pieces or offsets[pieces[0]][0] > a or offsets[pieces[-1]][1] < b:
                raise ValueError("Tokenizer cannot align complete PY150 lexical target")
            if any(
                text[offsets[k][0] : a].strip() for k in pieces if offsets[k][0] < a
            ) or any(
                text[b : offsets[k][1]].strip() for k in pieces if offsets[k][1] > b
            ):
                raise ValueError("Tokenizer piece crosses PY150 lexical boundaries")
            answer = tok.decode(
                [predictions[k] for k in pieces], skip_special_tokens=False
            ).strip()
            out.append(
                dict(
                    index=j,
                    gold=tokens[j],
                    prediction=answer,
                    correct=answer == tokens[j],
                )
            )
        return out
