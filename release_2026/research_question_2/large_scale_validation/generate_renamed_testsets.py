#!/usr/bin/env python3
"""
Generate shared, fixed renamed versions of the POJ-104 test set.

Every variable that gets renamed is replaced with ``var_<random number>``
(e.g. ``var_48127``), using the same renaming utilities as
``evaluate_robustness.py``. Output files are produced once, with a fixed seed,
so that EVERY model (the trained checkpoints and the CodeBERT / ContraBERT_C
baselines) is tested against byte-for-byte identical renamed code — a fair
head-to-head comparison.

For each N in --rename_counts it writes ``test_renamed_N{N}.jsonl`` next to the
original test set, preserving file order and the original fields, with the
renamed source in ``code`` and the actual number of variables renamed in
``num_renamed``.

Usage:
    python generate_renamed_testsets.py \
        --data_dir data_cache/poj104 --rename_counts 1,4,8 --seed 123456
"""

import argparse
import json
import os
import random

# Reuse the exact renaming logic (var_xxx) from the evaluator.
from evaluate_robustness import rename_variables, extract_variables


def main():
    parser = argparse.ArgumentParser(
        description='Generate shared fixed renamed POJ-104 test sets (var_xxx).')
    parser.add_argument('--data_dir', type=str, default='data_cache/poj104',
                        help='Directory containing test.jsonl')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to test.jsonl (default: <data_dir>/test.jsonl)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: --data_dir)')
    parser.add_argument('--rename_counts', type=str, default='1,4,8',
                        help='Comma-separated rename counts (default: 1,4,8)')
    parser.add_argument('--seed', type=int, default=123456,
                        help='Fixed seed for reproducible renamings (default: 123456)')
    args = parser.parse_args()

    test_file = args.test_file or os.path.join(args.data_dir, 'test.jsonl')
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    rename_counts = [int(x) for x in args.rename_counts.split(',')]

    # Load the original test set once (file order is preserved throughout).
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test examples from {test_file}")

    for n_renames in rename_counts:
        # Reseed before each N so each output file is independently reproducible.
        random.seed(args.seed)

        out_path = os.path.join(out_dir, f'test_renamed_N{n_renames}.jsonl')
        total_renamed = 0
        sample_names = []
        with open(out_path, 'w') as out:
            for js in test_data:
                renamed_code, actual_renamed = rename_variables(
                    js['code'], n_renames)
                total_renamed += actual_renamed
                rec = {
                    'code': renamed_code,
                    'label': js['label'],
                    'index': js['index'],
                    'num_renamed': actual_renamed,
                }
                out.write(json.dumps(rec) + '\n')

                # Collect a few example new names for a sanity printout.
                if len(sample_names) < 8 and actual_renamed > 0:
                    for tok in renamed_code.split():
                        cand = tok.strip('(){}[];,*&=<>+-/').strip()
                        if cand.startswith('var_') and cand not in sample_names:
                            sample_names.append(cand)
                            if len(sample_names) >= 8:
                                break

        avg = total_renamed / max(len(test_data), 1)
        print(f"  N={n_renames}: wrote {out_path}  "
              f"(avg vars renamed/example: {avg:.2f})")
        if sample_names:
            print(f"           sample new names: {', '.join(sample_names[:8])}")

    print("Done.")


if __name__ == '__main__':
    main()
