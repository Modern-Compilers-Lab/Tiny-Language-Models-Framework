"""Causal training with bounded RAM, independent of the original framework."""

import json
import random
import time
from pathlib import Path

from .models import (
    TOKENIZER,
    build_config,
    resolve_arch,
    resolve_dataset,
    resolve_device,
)
from .runtime import digest, read_jsonl, write_json


def encode_training_rows(tok, rows, max_len, supervised=False):
    """Keep every answer token, mask prompt loss, append EOS for supervised tasks."""
    if max_len < 2:
        raise ValueError("max_len must be at least two")
    if not supervised:
        # Keep the batched tokenizer contract used by experiment token caches.
        encoded = tok(
            [r["code"] for r in rows],
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )["input_ids"]
        ids = [seq for seq in encoded if len(seq) > 1]
        return ids, ids
    ids, labels = [], []
    for row in rows:
        if supervised:
            if row.get("split") != "train":
                raise ValueError("fine-tuning accepts train rows only")
            prefix = row["prompt"].rstrip()
            whitespace = row["prompt"][len(prefix) :]
            pre = tok.encode(prefix, add_special_tokens=False) or [
                tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            ]
            ans = tok.encode(whitespace + row["target"], add_special_tokens=False)
            if not ans:
                raise ValueError("empty supervised target")
            if tok.eos_token_id is not None:
                ans.append(tok.eos_token_id)
            if len(ans) >= max_len:
                raise ValueError(
                    "supervised target exceeds context; raise max_len or use shorter examples"
                )
            pre = pre[-(max_len - len(ans)) :]
            ids.append(pre + ans)
            labels.append([-100] * len(pre) + ans)
    return ids, labels


def training_sequences(
    tokenizer, path, max_len, rng, shuffle_buffer=1024, supervised=False
):
    """Read one epoch, keeping only a bounded buffer of tokenized documents."""
    buffer = []
    for row in read_jsonl(path):
        if row.get("split", "train") != "train":
            raise ValueError("Training data must contain train rows only")
        if not supervised and not isinstance(row.get("code"), str):
            raise ValueError("Training JSONL rows require a string 'code' field")
        ids, labels = encode_training_rows(tokenizer, [row], max_len, supervised)
        if not ids:
            continue
        buffer.append((ids[0], labels[0]))
        if len(buffer) >= shuffle_buffer:
            yield buffer.pop(rng.randrange(len(buffer)))
    rng.shuffle(buffer)
    yield from buffer


def validate_config(config):
    """Reject invalid recipes before any automatic dataset download."""
    steps, batch_size, micro = (
        config.get(k, default)
        for k, default in (("steps", 20000), ("batch_size", 16), ("micro_batch", 1))
    )
    if (
        min(
            steps,
            batch_size,
            micro,
            config.get("save_every", 2000),
            config.get("shuffle_buffer", 1024),
            config.get("threads", 4),
        )
        < 1
    ):
        raise ValueError(
            "steps, batch sizes, save_every, shuffle_buffer, and threads must be positive"
        )
    if batch_size % micro:
        raise ValueError("batch_size must be divisible by micro_batch")
    if (
        config.get("max_len", 320) < 2
        or config.get("lr", 3e-4) <= 0
        or config.get("warmup", 2000) < 0
    ):
        raise ValueError("Invalid max_len, learning rate, or warmup")
    if config.get("precision", "float32") not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")


def train(config, run):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = dict(config)
    config["arch"] = resolve_arch(config["arch"])
    if not config.get("benchmark"):
        config["dataset"] = resolve_dataset(config["arch"], config.get("dataset"))
    validate_config(config)
    steps, batch_size, micro = (
        config.get(k, default)
        for k, default in (("steps", 20000), ("batch_size", 16), ("micro_batch", 1))
    )
    max_len = config.get("max_len", 320)
    save_every = config.get("save_every", 2000)
    buffer_size = config.get("shuffle_buffer", 1024)
    precision = config.get("precision", "float32")
    device = resolve_device(config.get("device", "auto"))
    if (
        precision == "bfloat16"
        and device.startswith("cuda")
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This GPU does not support bfloat16")
    torch.set_num_threads(config.get("threads", 4))
    seed = config.get("seed", 2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = Path(config["data"])
    if not data.is_file():
        raise FileNotFoundError(data)
    run = Path(run)
    if (run / "model/config.json").exists() or (run / "results/training.json").exists():
        raise ValueError("Training output already exists; choose a new run directory")
    run.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "results", "cache"):
        (run / sub).mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer", TOKENIZER), cache_dir=str(run / "cache")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer requires a PAD or EOS token")
    tokenizer.padding_side = "right"
    if config.get("pretrained"):
        model = AutoModelForCausalLM.from_pretrained(config["pretrained"])
    else:
        model_config = build_config(
            config["arch"],
            len(tokenizer),
            config.get("scale", "small"),
            **config.get("model", {}),
        )
        model_config.bos_token_id = tokenizer.bos_token_id
        model_config.eos_token_id = tokenizer.eos_token_id
        model_config.pad_token_id = tokenizer.pad_token_id
        model = AutoModelForCausalLM.from_config(model_config)
    if model.config.model_type != config["arch"]:
        raise ValueError(
            "Checkpoint architecture does not match requested architecture"
        )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    max_len = min(max_len, model.config.max_position_embeddings)
    if max_len < 2:
        raise ValueError("Model context must be at least two tokens")
    model.to(device).train()
    model.config.use_cache = False
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", 3e-4), weight_decay=0.01
    )
    config["data"] = str(data.resolve())
    write_json(run / "config.json", config)
    data_hash = digest(data)

    def infinite_sequences():
        while True:
            count = 0
            for ids in training_sequences(
                tokenizer,
                data,
                max_len,
                rng,
                buffer_size,
                config.get("supervised", False),
            ):
                count += 1
                yield ids
            if not count:
                raise ValueError("No trainable sequences in the training data")

    sequences = infinite_sequences()
    accumulation = batch_size // micro
    started = time.time()
    loss_value = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(accumulation):
            chosen = [next(sequences) for _ in range(micro)]
            batch = tokenizer.pad(
                {"input_ids": [ids for ids, _ in chosen]},
                padding=True,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = torch.full_like(batch["input_ids"], -100)
            for index, (_, label_ids) in enumerate(chosen):
                labels[index, : len(label_ids)] = torch.tensor(label_ids, device=device)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                loss = model(**batch, labels=labels).loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = config.get("lr", 3e-4) * min(
                1, (step + 1) / max(config.get("warmup", 2000), 1)
            )
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            record = dict(step=step + 1, loss=loss_value, seconds=time.time() - started)
            print(json.dumps(record), flush=True)
            with (run / "logs/training.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record) + "\n")
        if (step + 1) % save_every == 0 and step + 1 < steps:
            checkpoint = run / "checkpoints" / f"step-{step + 1}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
    model.config.use_cache = True
    model.save_pretrained(run / "model")
    tokenizer.save_pretrained(run / "model")
    report = dict(
        config=config,
        parameters=sum(p.numel() for p in model.parameters()),
        steps=steps,
        seconds=time.time() - started,
        final_loss=loss_value,
        data_sha256=data_hash,
    )
    write_json(run / "results/training.json", report)
    return report
