"""Run the mechanistic protocols (copy score, mean ablation, forced attention, first error)."""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import env  # noqa: E402
from adapter import load_model, resolve_checkpoint  # noqa: E402
from env import write_json  # noqa: E402

MECHANISTIC = ("copyscore", "meanablate", "force", "firsterror")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for f in self.streams:
            f.write(s)
            f.flush()

    def flush(self):
        for f in self.streams:
            f.flush()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_hashes(model):
    return {
        p.name: digest(p)
        for p in sorted(Path(model).glob("*"))
        if p.is_file() and p.suffix in (".json", ".safetensors", ".bin")
    }


def configure(run, corpus):
    """Create the run folders and point every cache inside the run."""
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "results", "cache"):
        (run / sub).mkdir(exist_ok=True)
    for key in ("HF_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME"):
        os.environ[key] = str(run / "cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RENAMING_WORK"] = str(run)
    os.environ["WHYSCALE_CACHE"] = str(Path(corpus).resolve())
    env.lib_module("whyscale").CACHE = str(Path(corpus).resolve())
    return run


def parser(experiment=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite",
        choices=["mechanistic", "paper"],
        default="mechanistic",
        help="paper runs every exp_*.py entry point, each in its own process",
    )
    p.add_argument("--config", help="JSON defaults; explicit options override them")
    p.add_argument("--model", help="Hugging Face directory or training run with model/")
    p.add_argument("--run", help="output directory (default: runs/<timestamp>)")
    p.add_argument("--corpus", help="folder containing real6_corpus.json")
    p.add_argument("--experiments", default=experiment or ",".join(MECHANISTIC))
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--n", type=int, default=2000, help="evaluation functions")
    p.add_argument("--limit", type=int, default=0, help="first N functions only")
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--kmax", type=int, default=15, help="ablation sweep k = 1..kmax")
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--n-random-k3", type=int, default=50)
    p.add_argument("--ks", default="3,12", help="forced-attention head counts")
    p.add_argument(
        "--n-items", type=int, default=1600, help="suite only: variable-misuse item cap"
    )
    p.add_argument(
        "--n-sample", type=int, default=3000, help="suite only: ClozeTest cache size"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rename-seed", type=int, default=564)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--layers", help="comma-separated zero-based layers to rank")
    p.add_argument("--heads", help="comma-separated L:H pairs to rank")
    p.add_argument("--arm", default="unspecified", help="training condition label")
    p.add_argument("--training-seed", type=int)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip completed modules after checking config and hashes",
    )
    return p


def main(argv=None, experiment=None):
    p = parser(experiment)
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        p.set_defaults(**json.loads(Path(pre.config).read_text()))
    a = p.parse_args(argv)
    if experiment:
        a.experiments = experiment
    if not a.model or not a.corpus:
        p.error("--model and --corpus are required")
    if a.suite == "paper":
        return paper_suite(a)
    if os.environ.get("PYTHONHASHSEED") != "0":
        p.error("Start Python with PYTHONHASHSEED=0")
    run = configure(a.run or HERE / "runs" / time.strftime("%Y%m%d-%H%M%S"), a.corpus)
    import torch
    import transformers

    torch.set_num_threads(a.threads)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible")
    a.model = resolve_checkpoint(a.model)
    a.ks = [int(k) for k in a.ks.split(",")] if isinstance(a.ks, str) else a.ks
    if isinstance(a.layers, str):
        a.layers = [int(x) for x in a.layers.split(",")]
    if isinstance(a.heads, str):
        a.heads = [tuple(map(int, x.split(":"))) for x in a.heads.split(",")]
    if (
        min(a.n, a.batch, a.threads, a.max_len, a.n_random, a.n_random_k3) < 1
        or a.kmax < 0
    ):
        raise ValueError("Sample, batch and sweep options must be positive")
    experiments = a.experiments.split(",")
    if not set(experiments) <= set(MECHANISTIC):
        raise ValueError(f"Unknown experiment; choose from {', '.join(MECHANISTIC)}")
    config = vars(a).copy()
    config.pop("resume")
    config.pop("run")
    config["corpus_sha256"] = digest(Path(a.corpus) / "real6_corpus.json")
    config["source_sha256"] = {
        p.name: digest(p) for p in sorted(HERE.glob("*.py")) if p.name != "report.py"
    }
    config["checkpoint_sha256"] = checkpoint_hashes(a.model)
    if (run / "config.json").exists():
        if not a.resume:
            raise ValueError(
                "Run directory already has a config; use a new run or --resume"
            )
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Resume config, source, data or checkpoint mismatch")
    write_json(run / "config.json", config)
    write_json(
        run / "environment.json",
        dict(
            python=sys.version,
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=a.device,
            host=platform.node(),
        ),
    )
    with (
        open(run / "logs/run.log", "a") as log,
        contextlib.redirect_stdout(Tee(sys.stdout, log)),
    ):
        tok, model, layout = load_model(a.model, a.device)

        # The scoring functions read the model through this object and use the path
        # only as a label.
        class Source:
            tokenizer = tok
            causal_lm = model
            device = a.device

            def __fspath__(self):
                return a.model

        source = Source()
        tracked = list(layout.projections) + [
            layout.value_module(m) for m in layout.attentions
        ]
        for name in experiments:
            dest = run / "results" / (name + ".json")
            if a.resume and dest.exists():
                continue
            args = argparse.Namespace(**vars(a))
            if name in ("force", "firsterror"):
                delattr(args, "kmax")
            if name == "force":
                layout.validate_k(a.ks)
            if name in ("copyscore", "meanablate") and a.kmax:
                layout.validate_k([a.kmax])
            start = time.time()
            print("START", name, flush=True)
            # Remove any hook a module leaves behind, also when it raises.
            old = [
                (dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in tracked
            ]
            try:
                result = importlib.import_module(name).score_snapshot(
                    source, args, a.device
                )
            finally:
                for m, (post, pre_hooks) in zip(tracked, old):
                    for k in set(m._forward_hooks) - set(post):
                        del m._forward_hooks[k]
                    for k in set(m._forward_pre_hooks) - set(pre_hooks):
                        del m._forward_pre_hooks[k]
            result.update(
                experiment=name, seconds=time.time() - start, architecture=layout.arch
            )
            write_json(dest, result)
            print("DONE", name, round(result["seconds"], 1), flush=True)
    from score import summarize

    write_json(run / "results/metrics.json", summarize(run))
    return 0


def paper_suite(a):
    """Run every entry point in its own process; the library has process-wide settings."""
    root = Path(a.run or HERE / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    common = ["--model", a.model, "--corpus", a.corpus, "--device", a.device]
    common += ["--n", str(a.n), "--limit", str(a.limit), "--max-len", str(a.max_len)]
    common += ["--batch", str(a.batch), "--threads", str(a.threads)]
    ks = a.ks if isinstance(a.ks, str) else ",".join(map(str, a.ks))
    entries = [
        ("fig7a", "exp_fig7a_copy_score.py", []),
        ("zero", "exp_zero_ablation.py", []),
        ("mean", "exp_fig7b_mean_ablation.py", []),
        ("force", "exp_table6_forced_attention.py", []),
        ("firsterror", "exp_e2_paired_first_error.py", []),
        ("idrep", "exp_fig6_fig16_tasks.py", ["--group", "idrep"]),
        (
            "bench",
            "exp_fig6_fig16_tasks.py",
            ["--group", "bench", "--n-items", str(a.n_items)]
            + ["--n-sample", str(a.n_sample)],
        ),
        ("comp", "exp_d5_nonidentifier.py", []),
        ("vm", "exp_d5_misuse_detection.py", []),
    ]
    mechanistic = ("fig7a", "zero", "mean", "force", "firsterror")
    for label, script, extra in entries:
        options = common + ["--run", str(root / label)] + extra
        if label in mechanistic:
            options += [
                "--kmax",
                str(a.kmax),
                "--ks",
                ks,
                "--n-random",
                str(a.n_random),
            ]
            options += ["--n-random-k3", str(a.n_random_k3), "--seed", str(a.seed)]
            options += ["--rename-seed", str(a.rename_seed)]
            options += ["--split-seed", str(a.split_seed), "--arm", a.arm]
            if a.training_seed is not None:
                options += ["--training-seed", str(a.training_seed)]
            if a.layers:
                options += ["--layers", a.layers]
            if a.heads:
                options += ["--heads", a.heads]
            if a.resume:
                options += ["--resume"]
        environment = dict(os.environ, PYTHONHASHSEED="0")
        subprocess.run(
            [sys.executable, str(HERE / script), *options], check=True, env=environment
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
