"""Bounded-memory tokenization and a cache adapter for the unchanged core trainer.

The JSONL training dataset remains the source of truth; the cache is a run artifact.
All architecture/scale/optimizer/update code stays in other_models.training.train.
"""

import argparse, json, os, time
from pathlib import Path
from ...runtime import read_jsonl, write_json, digest, setup_run


def prepare(config, run, prefix, batch_size=32, threads=2):
    if batch_size < 1 or threads < 1 or config.get("max_len", 320) < 2:
        raise ValueError("Invalid cache batch size, threads or max_len")
    if config.get("supervised"):
        raise ValueError("Renaming token caches accept code-only training data")
    import numpy as np
    from transformers import AutoTokenizer

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source_hash = digest(config["data"])
    max_len = config.get("max_len", 320)
    meta_path = Path(str(prefix) + ".json")
    tok = AutoTokenizer.from_pretrained(config["tokenizer"])
    tok_hash = (
        __import__("hashlib")
        .sha256(tok.backend_tokenizer.to_str().encode())
        .hexdigest()
    )
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta["data_sha256"] == source_hash
            and meta["max_len"] == max_len
            and meta["tokenizer"] == config["tokenizer"]
        ):
            if meta.get("tokenizer_sha256") != tok_hash:
                raise ValueError("tokenizer changed")
            if meta.get("offsets_sha256") != digest(str(prefix) + ".offsets.npy"):
                raise ValueError("corrupt or uncertified cache offsets")
            if meta["bin_sha256"] != digest(str(prefix) + ".bin"):
                raise ValueError("corrupt token cache")
            return meta
        raise ValueError("token cache configuration differs")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    tok = AutoTokenizer.from_pretrained(config["tokenizer"])
    tok_hash = (
        __import__("hashlib")
        .sha256(tok.backend_tokenizer.to_str().encode())
        .hexdigest()
    )
    offsets = [0]
    count = 0
    start = time.time()
    batch = []
    tmp = Path(str(prefix) + ".bin.tmp")
    with tmp.open("wb") as output:

        def flush():
            nonlocal count
            ids = tok(
                batch, truncation=True, max_length=max_len, add_special_tokens=False
            )["input_ids"]
            for seq in ids:
                array = np.asarray(seq, dtype="<u4")
                output.write(array.tobytes())
                offsets.append(offsets[-1] + len(seq))
                count += 1
            batch.clear()

        for row in read_jsonl(config["data"]):
            if row.get("split", "train") != "train":
                raise ValueError("Training token caches accept train rows only")
            batch.append(row["code"])
            if len(batch) >= batch_size:
                flush()
                if count % 1024 == 0:
                    print(
                        json.dumps(dict(rows=count, seconds=time.time() - start)),
                        flush=True,
                    )
        if batch:
            flush()
    tmp.replace(str(prefix) + ".bin")
    with open(str(prefix) + ".offsets.npy", "wb") as f:
        np.save(f, np.asarray(offsets, dtype="<u8"))
    meta = dict(
        rows=count,
        tokens=offsets[-1],
        max_len=max_len,
        data_sha256=source_hash,
        tokenizer=config["tokenizer"],
        tokenizer_sha256=tok_hash,
        seconds=time.time() - start,
        format="uint32-little-endian+uint64-offsets",
        bin_sha256=digest(str(prefix) + ".bin"),
        offsets_sha256=digest(str(prefix) + ".offsets.npy"),
    )
    write_json(meta_path, meta)
    return meta


class TokenTable:
    def __init__(self, prefix):
        import numpy as np

        self.data = np.memmap(str(prefix) + ".bin", dtype="<u4", mode="r")
        self.offsets = np.load(str(prefix) + ".offsets.npy", mmap_mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.data[int(self.offsets[index]) : int(self.offsets[index + 1])]


class TokenizerCacheAdapter:
    def __init__(self, tokenizer, table, max_len):
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "max_len", max_len)

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def __setattr__(self, key, value):
        setattr(self.tokenizer, key, value)

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, texts, **kwargs):
        if len(texts) != len(self.table) or kwargs.get("max_length") != self.max_len:
            raise ValueError("cache request mismatch")
        return {"input_ids": self.table}


def train_cached(config, run, prefix):
    from unittest.mock import patch
    from transformers import AutoTokenizer
    from ... import training as trainer

    meta = json.loads(Path(str(prefix) + ".json").read_text())
    if (
        meta["data_sha256"] != digest(config["data"])
        or meta["max_len"] != config.get("max_len", 320)
        or meta["tokenizer"] != config["tokenizer"]
    ):
        raise ValueError("stale cache")
    if meta["bin_sha256"] != digest(str(prefix) + ".bin"):
        raise ValueError("corrupt token cache")
    if meta.get("offsets_sha256") != digest(str(prefix) + ".offsets.npy"):
        raise ValueError("corrupt or uncertified cache offsets")
    tok = AutoTokenizer.from_pretrained(config["tokenizer"])
    if (
        __import__("hashlib")
        .sha256(tok.backend_tokenizer.to_str().encode())
        .hexdigest()
        != meta["tokenizer_sha256"]
    ):
        raise ValueError("tokenizer changed")
    if config.get("supervised"):
        raise ValueError("Renaming token caches accept code-only training data")
    table = TokenTable(prefix)

    def cached_sequences(tokenizer, path, max_len, rng, shuffle_buffer=1024, supervised=False):
        if supervised or Path(path).resolve() != Path(config["data"]).resolve() or max_len != meta["max_len"]:
            raise ValueError("cache request mismatch")
        buffer = []
        for index in range(len(table)):
            ids = table[index].tolist()
            if len(ids) < 2:
                continue
            buffer.append((ids, ids))
            if len(buffer) >= shuffle_buffer:
                yield buffer.pop(rng.randrange(len(buffer)))
        rng.shuffle(buffer)
        yield from buffer
    write_json(
        Path(run) / "cache_backend.json",
        dict(
            prefix=str(prefix),
            metadata=meta,
            trainer="other_models.training.train with identical streaming sequence order",
        ),
    )
    # Preserve bounded shuffle order, masking and the core trainer's optimizer/batching.
    try:
        with patch.object(trainer, "training_sequences", cached_sequences):
            result = trainer.train(config, run)
            if os.getenv("ORX_METRICS_PATH"):
                write_json(os.environ["ORX_METRICS_PATH"], result)
            return result
    finally:
        # Release file mappings explicitly; Windows cannot rewrite a mapped cache.
        table.data._mmap.close()
        table.offsets._mmap.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "train"])
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    config = json.loads(Path(a.config).read_text())
    run = setup_run(a.run, config)
    if a.action == "prepare":
        print(prepare(config, run, a.cache, threads=a.threads), flush=True)
    else:
        from .registry import builtins

        builtins()
        train_cached(config, run, a.cache)


if __name__ == "__main__":
    main()
