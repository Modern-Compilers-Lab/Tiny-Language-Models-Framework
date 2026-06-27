import os
import random
import numpy as np
import gc
import re
import ast
import pandas as pd
from indistrebution_perturbator import VariableRenamer
from tinypy_code_tracing_generator_parallel import TinyPyCodeTracer


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def extract_folder_name(path):
    """Extracts the eval folder name from a full result path."""
    match = re.search(r'([^/]+)(?=/infers/)', path)
    return match.group(1) if match else None


def get_grouped_evaluation_paths(A):
    """
    Returns a 2D array of existing hard.csv paths, one row per percentage in A.
    """
    base_root = "/data/ia2921/Tiny_language_model_framework/3Evaluations/experiment_reboot/exp2_alpha/results"

    checkpoints = [
        "checkpoint_0.03.pth", "checkpoint_0.07.pth", "checkpoint_0.10.pth",
        "checkpoint_0.13.pth", "checkpoint_0.17.pth", "checkpoint_0.20.pth",
        "checkpoint_0.23.pth", "checkpoint_0.27.pth", "checkpoint_0.30.pth",
        "checkpoint_0.33.pth", "checkpoint_0.37.pth", "checkpoint_0.40.pth",
        "checkpoint_0.43.pth", "checkpoint_0.47.pth", "checkpoint_0.50.pth",
        "checkpoint_0.53.pth", "checkpoint_0.57.pth", "checkpoint_0.60.pth",
        "checkpoint_0.63.pth", "checkpoint_0.67.pth", "checkpoint_0.70.pth",
        "checkpoint_0.73.pth", "checkpoint_0.77.pth", "checkpoint_0.80.pth",
        "checkpoint_0.83.pth", "checkpoint_0.87.pth", "checkpoint_0.90.pth",
        "checkpoint_0.93.pth", "checkpoint_0.97.pth", "checkpoint_1.00.pth",
    ]
    grouped_paths = []
    print(f"Checking paths in: {base_root}\n")

    for a in A:
        current_row = []
        for ckpt in checkpoints:
            folder_name = f"eval_jobs1_{ckpt}_ID"
            full_path   = os.path.join(base_root, folder_name, "infers", "hard.csv")
            if os.path.exists(full_path):
                current_row.append(full_path)
            else:
                print(f"[MISSING] {full_path}")
        grouped_paths.append(current_row)

    print("-" * 60)
    for i, a_val in enumerate(A):
        print(f"Row {i} (A={a_val}): Found {len(grouped_paths[i])} valid paths")
    print("-" * 60)

    return grouped_paths


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
A = ["one"]  # add/remove as results arrive

# FIX 1: proper f-string list comprehension
original_test_sets = [
    f"/data/ia2921/Tiny_language_model_framework/1Datasets/experiment_reboot/alpha_0.X/300m/test.txt"
    for a in A
]

all_paths = get_grouped_evaluation_paths(A)

seed = 564
np.random.seed(seed)
random.seed(seed)

renamer = VariableRenamer(mode="all")

# FIX 2: range(len(A)) instead of range(1)
for i in range(len(A)):
    test_path    = original_test_sets[i]
    subset_paths = all_paths[i]

    for path in subset_paths:
        tpct = TinyPyCodeTracer()

        correct_examples = [
            example.split("#STEP", 1)[0]
            for example in pd.read_csv(path)["example_input"].tolist()
        ]

        correct_test_renamed_var_examples = []
        print(f"[*] Generating test sets for: {extract_folder_name(path)} ...")
        print("[*] Executing variable renaming ...")

        for example in correct_examples:
            tree = ast.parse(example)
            tree = renamer.rename(tree)
            if tree is not False:
                transformed_example = ast.unparse(tree).replace("    ", "\t")
                exec_trace, error_type, _ = tpct.trace_snippet(transformed_example)
                if error_type is None:
                    correct_test_renamed_var_examples.append(exec_trace)

        print(f"[*] Total renamed examples: {len(correct_test_renamed_var_examples)}")

        base_directory = (
            "/data/ia2921/Tiny_language_model_framework/3Evaluations/experiment_reboot/exp2_alpha"
            f"/semantic_preserving_edits_files/{extract_folder_name(path)}/"
        )
        os.makedirs(base_directory, exist_ok=True)

        out_path = os.path.join(base_directory, "test.txt")
        print(f"[*] Writing to {out_path} ...")
        with open(out_path, "w") as f:
            f.write("\n\n".join(correct_test_renamed_var_examples) + "\n\n")

        del correct_test_renamed_var_examples
        gc.collect()

gc.collect()