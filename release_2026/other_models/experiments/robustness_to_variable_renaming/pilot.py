"""The two measures of the Appendix D.5 pilot.

    python exp_d5_nonidentifier.py --model CHECKPOINT --corpus CORPUS --run RUN
    python exp_d5_misuse_detection.py --model CHECKPOINT --corpus CORPUS --run RUN

comp  NEXT-TOKEN ACCURACY at non-identifier positions (decmech.cmd_comp). Teacher-forced
      next-token accuracy on the evaluation functions, counted only at tokens that do not
      overlap a variable name. Reported for the original functions ("accuracy on the original
      code" in D.5), and the retention of correct predictions at the same positions after every
      local variable is renamed to a random same-length name.
vm    VARIABLE-MISUSE DETECTION (lib/varmisuse.py, decoder scoring). A two-way choice: the model
      must assign a higher log-likelihood to the correct function than to a copy in which one
      use of a variable is replaced by another variable (chance 0.5; 1,954 items). This is NOT
      the variable-misuse LOCALIZATION task of Section 4.5 (choose the bug position among every
      candidate edit), so the two accuracies are not comparable. Also reported: retention after
      renaming up to 1, 4 and 8 variables.

The two measures run in separate processes because importing lib/varmisuse.py changes a
library setting (WHYSCALE_BLOCK) for the whole process. Models are loaded in fp32 on either
device.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import env
from adapter import load_model, resolve_checkpoint
from run import checkpoint_hashes, configure, digest


def run_comp(a):
    from types import SimpleNamespace

    DM = env.lib_module("decmech")
    tok, model, layout = load_model(a.model, a.device)
    DM.load_lm = lambda path, device: (tok, model)
    tmp = "%s.comp.%d.json" % (a.out, os.getpid())
    ns = SimpleNamespace(
        model=a.model,
        n=a.n,
        seed=564,
        max_len=a.max_len,
        batch=a.batch,
        gen_n=0,
        gen_batch=24,
        no_gen=True,
        device=a.device,
        threads=a.threads,
        limit=a.limit,
        write_smoke=True,
        out=tmp,
    )
    try:
        DM.cmd_comp(ns)
        doc = json.load(open(tmp))["completion"][
            os.path.basename(os.path.normpath(a.model))
        ]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {
        "accuracy_nonident_original": doc["identity"]["unpaired"]["nonident"]["rate"],
        "accuracy_nonident_renamed": doc["samelen"]["unpaired"]["nonident"]["rate"],
        "retention_nonident_renamed": doc["samelen"]["paired_nonident"]["retention"][
            "rate"
        ],
        "raw": doc,
    }


def run_vm(a):
    VM = env.lib_module("varmisuse")

    # The library's scorer loads its own model; give it the framework checkpoint instead.
    def init(sc, path, device, batch_size=8, window=None, amp=None):
        import torch

        sc.torch = torch
        sc.tok, sc.m, _ = load_model(path, device)
        sc.dev = device

    VM.DecScorer.__init__ = init
    # build_items asks for REAL6_N (2,000) functions. With a smaller prepared corpus
    # the library would download CodeSearchNet again and overwrite real6_corpus.json,
    # so the measure uses every prepared function instead.
    W = env.lib_module("whyscale")
    with open(Path(W.CACHE) / "real6_corpus.json") as f:
        available = len(json.load(f)["recs"])
    W.REAL6_N = min(W.REAL6_N, available)
    items, meta = VM.build_items(a.limit or None)
    out = VM.run_model(
        a.model, items, "decoder", a.device, a.vm_batch, 8, amp=(a.amp == "bf16")
    )
    res = VM.summarize(out, items, meta, "decoder", 8)
    d = res["kinds"]["var"]
    return {
        "vm_detection_accuracy": d["accuracy"],
        "vm_detection_retention": {
            str(N): d.get("retention_N%d" % N) for N in (1, 4, 8)
        },
        "n_items": d["n_items"],
        "n_functions": W.REAL6_N,
        "raw": res,
    }


def main(argv=None, what=None):
    env.setup()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--what", default=what, choices=("comp", "vm"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", help="optional output JSON path")
    ap.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vm-batch", type=int, default=160)
    ap.add_argument("--amp", default="off", choices=("off", "bf16"))
    ap.add_argument("--limit", type=int, default=0, help="quick checks only")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if what:
        a.what = what
    if not a.what:
        ap.error("--what is required")
    run = configure(a.run, a.corpus)
    a.model = resolve_checkpoint(a.model)
    a.out = a.out or str(run / "results" / ("pilot_" + a.what + ".json"))
    env.write_json(
        run / "config.json",
        dict(
            vars(a),
            corpus_sha256=digest(Path(a.corpus) / "real6_corpus.json"),
            checkpoint_sha256=checkpoint_hashes(a.model),
        ),
    )
    import torch

    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    res = run_comp(a) if a.what == "comp" else run_vm(a)
    res.update(
        experiment="pilot_" + a.what,
        model=os.path.abspath(a.model),
        limit=a.limit,
        seconds=time.time() - t0,
    )
    env.write_json(a.out, res)
    print({k: v for k, v in res.items() if k != "raw"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
