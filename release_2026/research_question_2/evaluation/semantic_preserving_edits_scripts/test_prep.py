import os
import random
import numpy as np
import gc
from indistrebution_perturbator import VariableRenamer
from tinypy_code_tracing_generator_parallel import TinyPyCodeTracer
import ast
import pandas as pd


import os

import re

def extract_folder_name(path):
    """
    Eats a full path and spits out the evaluation folder name.
    Example Input: .../results/eval_jobs1.../infers/hard.csv
    Example Output: eval_jobs1...
    """
    match = re.search(r'([^/]+)(?=/infers/)', path)
    if match:
        return match.group(1)
    return None

def get_grouped_evaluation_paths():
    """
    Returns a 2D array of file paths.
    Index 0 -> A="10"
    Index 1 -> A="1"
    Index 2 -> A="0.1"
    Index 3 -> A="0.01"
    """
    
    # --- Configuration ---
    base_root = "/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/results_extended"

    X = ["jobs1"]
    
    Y = [
        "checkpoint_0.04.pth", "checkpoint_0.08.pth", "checkpoint_0.12.pth",
        "checkpoint_0.17.pth", "checkpoint_0.21.pth", "checkpoint_0.25.pth",
        "checkpoint_0.29.pth", "checkpoint_0.33.pth", "checkpoint_0.37.pth",
        "checkpoint_0.42.pth", "checkpoint_0.46.pth", "checkpoint_0.50.pth",
        "checkpoint_0.54.pth", "checkpoint_0.58.pth", "checkpoint_0.62.pth",
        "checkpoint_0.67.pth", "checkpoint_0.71.pth", "checkpoint_0.75.pth",
        "checkpoint_0.79.pth", "checkpoint_0.83.pth", "checkpoint_0.87.pth",
        "checkpoint_0.92.pth", "checkpoint_0.96.pth", "checkpoint_1.00.pth"
    ]
    
    # Constraint: Only "ID"
    Z = ["ID"]
    
    # The order of this array defines the order of the output rows
    A = ["80"]

    # This will hold our list of lists
    grouped_paths = []

    print(f"Checking paths in: {base_root}\n")
    
    # --- Outer Loop: Iterate through A to create rows ---
    for a in A:
        current_row = []
        
        # --- Inner Loops: Iterate through X, Y, Z to fill the row ---
        for x in X:
            for y in Y:
                for z in Z:
                    # Construct folder name: eval_X_Y_Z_p_A
                    # e.g. eval_jobs1_checkpoint_0.04.pth_ID_p_10
                    folder_name = f"eval_{x}_{y}_{z}_p_{a}"
                    
                    # Construct full absolute path
                    full_path = os.path.join(base_root, folder_name, "infers", "hard.csv")
                    
                    if os.path.exists(full_path):
                        current_row.append(full_path)
                    else:
                        print(f"[MISSING] {full_path}")

        # Add the completed row to the main array
        grouped_paths.append(current_row)

    # --- Summary ---
    print("-" * 60)
    for i, a_val in enumerate(A):
        count = len(grouped_paths[i])
        print(f"Row {i} (A={a_val}): Found {count} valid paths")
    print("-" * 60)
    print(grouped_paths)
    return grouped_paths

all_paths = get_grouped_evaluation_paths()

A = ["80"]
original_test_sets = [ "/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_80_p/test.txt"]

seed = 564
np.random.seed(seed)
random.seed(seed)

for i in range(1):
    test_path = original_test_sets[i]


# # Loading the all test snippets
# print("[*] Loading the test dataset ...")
# with open(f"/data/ia2921/Tiny_language_model_framework/1Datasets/gamma_1_X_gen/data_200/test.txt", "r") as f:
#     test_data = f.read()
# print("[*] Splitting dataset by examples ...")
# all_test_examples = test_data.split("\n\n")[:-1]
# all_test_examples = [example.split("#STEP", 1)[0] for example in all_test_examples]
# all_test_examples = all_test_examples[:1024]

# print(f"[*] Generating the all test set ...")
# all_test_renamed_var_examples = []

# tpct = TinyPyCodeTracer()

# print("[*] Executing variable renaming ...")
# renamer = VariableRenamer(mode="all")
# for example in all_test_examples:
#     tree = ast.parse(example)
#     tree = renamer.rename(tree)
#     if tree is not False:
#         transformed_example = ast.unparse(tree)
#         exec_trace, error_type, error_msg = tpct.trace_snippet(transformed_example)
#         if error_type is None:
#             all_test_renamed_var_examples.append(exec_trace)
# print(f"[*] Total all test renamed var examples: {len(all_test_renamed_var_examples)}")

renamer = VariableRenamer(mode="all")

for i in range(1):
    test_path = original_test_sets[i]
    subset_paths = all_paths[i]
    for path in subset_paths:

        tpct = TinyPyCodeTracer()

        correct_examples = [example.split("#STEP", 1)[0] for example in pd.read_csv(path)["example_input"].tolist()]

        correct_test_renamed_var_examples = []

        print(f"[*] Generating the test sets for path : {extract_folder_name(path)} ...")

        print("[*] Executing variable renaming ...")
        for example in correct_examples:
            tree = ast.parse(example)
            tree = renamer.rename(tree)
            if tree is not False:
                transformed_example = ast.unparse(tree).replace("    ", "\t")
                exec_trace, error_type, error_msg = tpct.trace_snippet(transformed_example)
                if error_type is None:
                    correct_test_renamed_var_examples.append(exec_trace)
        print(f"[*] Total correct test renamed var examples: {len(correct_test_renamed_var_examples)}")


        base_directory = f"/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files_new/{extract_folder_name(path)}/"
        os.makedirs(base_directory, exist_ok=True)

        # all_test_renamed_var_out_path = base_directory + "test.txt"

        # with open(all_test_renamed_var_out_path, 'w') as f:
        #     f.write("\n\n".join(all_test_renamed_var_examples) + "\n\n")


        print(f"[*] Writting correct test examples ...")
        correct_test_renamed_var_out_path = base_directory + "test.txt"

        with open(correct_test_renamed_var_out_path, 'w') as f:
            f.write("\n\n".join(correct_test_renamed_var_examples) + "\n\n")


        del correct_test_renamed_var_examples


        gc.collect()

# del all_test_renamed_var_examples
gc.collect()