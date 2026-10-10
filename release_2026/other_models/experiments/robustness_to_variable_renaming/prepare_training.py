"""Prepare controlled model/dataset/transform/seed cells, with optional token caches."""

import argparse, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ...runtime import setup_run, write_json


def cell(args):
    source, tokenizer, root, arch, dataset, mode, seed, use_cache, overrides = args
    run = Path(root) / f"{mode}-s{seed}"
    cfg = dict(
        arch=arch,
        dataset=dataset,
        scale="small",
        mode=mode,
        seed=seed,
        steps=20000,
        batch_size=16,
        max_len=320,
        lr=0.0003,
        warmup=2000,
        device="cuda",
        precision="float32",
        threads=4,
        tokenizer=tokenizer,
        data=str(run / "data/train.jsonl"),
        transformation="paper-span-fresh-v1",
    )
    cfg.update(overrides)
    from ...training import validate_config

    validate_config(cfg)
    run = setup_run(run, cfg)
    write_json(run / "spec.json", cfg)
    from .data_transformations.renaming import transform

    stats = transform(source, cfg["data"], mode, seed)
    write_json(run / "results/preparation.json", stats)
    if use_cache:
        from .token_cache import prepare

        prepare(cfg, run, run / "data/tokens", threads=1)
    return dict(
        name=f"train-{mode}-s{seed}",
        seed=seed,
        run=str(run),
        tag="final",
        **(
            dict(
                module="other_models.experiments.robustness_to_variable_renaming.token_cache",
                args=[
                    "train",
                    "--config",
                    str(run / "spec.json"),
                    "--run",
                    str(run),
                    "--cache",
                    str(run / "data/tokens"),
                ],
            )
            if use_cache
            else dict(
                args=["train", "--config", str(run / "spec.json"), "--run", str(run)]
            )
        ),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--arch", default="llama")
    p.add_argument("--scale", choices=["small", "large"], default="small")
    p.add_argument("--dataset", help="explicit dataset; defaults according to --arch")
    p.add_argument("--modes", default="plain,halfvar,highdiv")
    p.add_argument("--seeds", default="2,3,4")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--token-cache", action="store_true")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--pretrained", help="Optional common starting checkpoint for every arm")
    a = p.parse_args(argv)
    from ...models import resolve_dataset

    try:
        a.dataset = resolve_dataset(a.arch, a.dataset)
    except ValueError as error:
        p.error(str(error))
    root = Path(a.run_root).resolve()
    modes = a.modes.split(",")
    seeds = list(map(int, a.seeds.split(",")))
    if a.workers < 1 or set(modes) - {"plain", "halfvar", "highdiv"}:
        p.error("Invalid workers or treatment names")
    if len(modes) != len(set(modes)) or len(seeds) != len(set(seeds)):
        p.error("Duplicate modes or seeds")
    overrides = {key: getattr(a, key) for key in ("steps", "batch_size", "micro_batch", "max_len", "warmup", "device", "scale")}
    if a.pretrained:
        overrides["pretrained"] = str(Path(a.pretrained).resolve()) if Path(a.pretrained).exists() else a.pretrained
    from ...training import validate_config

    validate_config(overrides)
    inputs = [
        (
            str(Path(a.source).resolve()),
            str(Path(a.tokenizer).resolve()) if Path(a.tokenizer).exists() else a.tokenizer,
            str(root),
            a.arch,
            a.dataset,
            m,
            s,
            a.token_cache,
            overrides,
        )
        for m in modes
        for s in seeds
    ]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        cells = list(pool.map(cell, inputs))
    write_json(root / "training-cells.json", cells)


if __name__ == "__main__":
    main()
