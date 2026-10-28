"""Entry points of the variable-renaming experiment.

Each action runs one script of experiments/robustness_to_variable_renaming/ in a
new Python process started with PYTHONHASHSEED=0. The scoring library seeds its
renamed views with string hashes and changes process-wide settings, so the
experiments never share a process. Arguments after the action are passed to the
script unchanged; `experiment ACTION --help` lists them.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent / "robustness_to_variable_renaming"

ACTIONS = {
    "prepare": ("prepare_corpus.py", "Build the evaluation corpus and ClozeTest cache"),
    "transform": ("transform.py", "Create a plain, halfvar or highdiv training set"),
    "tasks": (
        "exp_fig6_fig16_tasks.py",
        "Figure 6 / Table 4 tasks (--group idrep|bench)",
    ),
    "copy-score": ("exp_fig7a_copy_score.py", "Figure 7A copy score"),
    "zero-ablation": ("exp_zero_ablation.py", "Section 4.5 zero ablation"),
    "mean-ablation": ("exp_fig7b_mean_ablation.py", "Figure 7B mean ablation"),
    "forced-attention": ("exp_table6_forced_attention.py", "Table 6 forced attention"),
    "first-error": ("exp_e2_paired_first_error.py", "Appendix E.2 paired first error"),
    "nonidentifier": (
        "exp_d5_nonidentifier.py",
        "Appendix D.5 non-identifier accuracy",
    ),
    "misuse-detection": ("exp_d5_misuse_detection.py", "Appendix D.5 misuse detection"),
    "mechanistic": ("run.py", "The four mechanistic protocols in one process"),
    "report": ("report.py", "Mean and sample SD over seeds, by training condition"),
    "toy": ("toy.py", "Check every protocol on small random models"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=["list", "suite", *ACTIONS])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "list":
        result = dict(
            experiment=EXPERIMENT.name,
            actions={
                k: dict(script=v[0], description=v[1]) for k, v in ACTIONS.items()
            },
            suite="every paper entry point, each in its own process (run.py --suite paper)",
        )
        print(json.dumps(result, indent=2))
        return result
    if args.action == "suite":
        script, extra = "run.py", ["--suite", "paper"]
    else:
        script, extra = ACTIONS[args.action][0], []
    environment = dict(os.environ, PYTHONHASHSEED="0")
    command = [sys.executable, str(EXPERIMENT / script), *extra, *args.arguments]
    code = subprocess.run(command, env=environment).returncode
    if code:
        sys.exit(code)
    return 0
