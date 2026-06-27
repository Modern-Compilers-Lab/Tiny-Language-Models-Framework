#!/usr/bin/env python3
"""
Build the final fine-tuned robustness table from per-model result JSONs.

Reads every ``*.json`` in --eval_dir (the per-model copies written by
``evaluate_robustness.py``) and emits a table with columns:

    Model | MAP@R | N=1 | N=4 | N=8

Baselines (CodeBERT, ContraBERT_C) are listed first, then the exp1_diverse
checkpoints in ascending epoch order. Writes both a human-readable ``.txt`` and
a ``.csv`` into --eval_dir.
"""

import argparse
import csv
import glob
import json
import os
import re


def label_for(name):
    """Human-readable row label + a sort key (baselines first, then epoch)."""
    if name == 'baseline_codebert':
        return 'CodeBERT (baseline)', (0, 0)
    if name == 'baseline_contrabert':
        return 'ContraBERT_C (baseline)', (0, 1)
    m = re.match(r'finetuned_epoch_(\d+)$', name)
    if m:
        e = int(m.group(1))
        return f'Exp1 (epoch {e})', (1, e)
    return name, (2, name)


def main():
    parser = argparse.ArgumentParser(description='Build the robustness table.')
    parser.add_argument('--eval_dir', type=str, default='eval_results_varname')
    args = parser.parse_args()

    rows = []
    for fp in glob.glob(os.path.join(args.eval_dir, '*.json')):
        with open(fp) as f:
            d = json.load(f)
        name = d.get('experiment_name') or os.path.basename(fp)[:-5]
        label, key = label_for(name)
        rows.append({
            'key': key,
            'label': label,
            'mapr': d.get('MAP@R_N0', float('nan')),
            'n1': d.get('accuracy_N1', float('nan')),
            'n4': d.get('accuracy_N4', float('nan')),
            'n8': d.get('accuracy_N8', float('nan')),
        })

    if not rows:
        print(f"No result JSONs found in {args.eval_dir}/")
        return

    rows.sort(key=lambda r: r['key'])

    header = f"{'Model':<28} {'MAP@R':>8} {'N=1':>8} {'N=4':>8} {'N=8':>8}"
    sep = '-' * len(header)
    lines = [
        '==================================================================',
        '  FINE-TUNED ROBUSTNESS (POJ-104 fine-tune, var_xxx renaming)',
        '  Renamings are shared/fixed across all models.',
        '==================================================================',
        '',
        header,
        sep,
    ]
    prev_group = None
    for r in rows:
        if prev_group is not None and r['key'][0] != prev_group:
            lines.append(sep)
        prev_group = r['key'][0]
        lines.append(f"{r['label']:<28} {r['mapr']:>8.4f} {r['n1']:>8.4f} "
                     f"{r['n4']:>8.4f} {r['n8']:>8.4f}")
    lines.append(sep)
    table = '\n'.join(lines)
    print('\n' + table + '\n')

    txt_path = os.path.join(args.eval_dir, 'finetuned_results.txt')
    with open(txt_path, 'w') as f:
        f.write(table + '\n')

    csv_path = os.path.join(args.eval_dir, 'finetuned_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'MAP@R', 'N=1', 'N=4', 'N=8'])
        for r in rows:
            w.writerow([r['label'],
                        f"{r['mapr']:.4f}", f"{r['n1']:.4f}",
                        f"{r['n4']:.4f}", f"{r['n8']:.4f}"])

    print(f"Wrote {txt_path}")
    print(f"Wrote {csv_path}")


if __name__ == '__main__':
    main()
