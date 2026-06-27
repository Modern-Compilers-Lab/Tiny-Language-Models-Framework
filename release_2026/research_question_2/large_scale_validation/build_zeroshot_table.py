#!/usr/bin/env python3
"""
Build the ZERO-SHOT robustness table from per-model result JSONs.

Reads every ``zeroshot_*.json`` in --eval_dir and emits a table with columns:

    Model | Correct@N=0 (#/%) | N=1 # / retention% | N=4 # / retention% | N=8 # / retention% | N=8 drop

Baselines (CodeBERT, ContraBERT_C) are listed first, then exp1_diverse
checkpoints in ascending epoch order. Writes both a ``.txt`` and ``.csv``.
"""

import argparse
import csv
import glob
import json
import os
import re


def label_for(name):
    """Human-readable row label + sort key (baselines first, then epoch)."""
    if name == 'baseline_codebert':
        return 'CodeBERT (baseline)', (0, 0)
    if name == 'baseline_contrabert':
        return 'ContraBERT_C (baseline)', (0, 1)
    m = re.match(r'epoch_(\d+)$', name)
    if m:
        e = int(m.group(1))
        return f'Exp1 (epoch {e})', (1, e)
    return name, (2, name)


def main():
    parser = argparse.ArgumentParser(description='Build the zero-shot table.')
    parser.add_argument('--eval_dir', type=str, default='eval_results_varname')
    args = parser.parse_args()

    rows = []
    for fp in glob.glob(os.path.join(args.eval_dir, 'zeroshot_*.json')):
        with open(fp) as f:
            d = json.load(f)
        name = (d.get('experiment_name')
                or os.path.basename(fp)[len('zeroshot_'):-len('.json')])
        label, key = label_for(name)
        c0 = int(d.get('num_correct_0', 0))
        n_test = int(d.get('num_test', 12000))
        acc_at_0 = c0 / max(n_test, 1)
        ret = {n: float(d.get(f'accuracy_N{n}', 0.0)) for n in (1, 4, 8)}
        cnt = {n: int(d.get(f'num_correct_N{n}', round(ret[n] * c0)))
               for n in (1, 4, 8)}
        rows.append({
            'key': key, 'label': label,
            'n_test': n_test,
            'c0': c0, 'acc_at_0': acc_at_0,
            'n1c': cnt[1], 'n1r': ret[1],
            'n4c': cnt[4], 'n4r': ret[4],
            'n8c': cnt[8], 'n8r': ret[8],
            'n8_drop': 1.0 - ret[8],
        })

    if not rows:
        print(f"No zeroshot JSONs found in {args.eval_dir}/")
        return

    rows.sort(key=lambda r: r['key'])

    header = (f"{'Model':<28} "
              f"{'N=0 #/%':>18} "
              f"{'N=1 #/ret%':>18} "
              f"{'N=4 #/ret%':>18} "
              f"{'N=8 #/ret%':>18} "
              f"{'N=8 drop':>10}")
    sep = '-' * len(header)
    lines = [
        '=' * len(header),
        '  ZERO-SHOT ROBUSTNESS (raw pre-trained encoder, no fine-tuning)',
        '  Mean-pooled cosine similarity on POJ-104; shared var_xxx renamings.',
        '=' * len(header),
        header,
        sep,
    ]
    prev_group = None
    for r in rows:
        if prev_group is not None and r['key'][0] != prev_group:
            lines.append(sep)
        prev_group = r['key'][0]
        n0_cell = f"{r['c0']:>6,} ({r['acc_at_0']*100:>5.2f}%)"
        n1_cell = f"{r['n1c']:>6,} ({r['n1r']*100:>5.2f}%)"
        n4_cell = f"{r['n4c']:>6,} ({r['n4r']*100:>5.2f}%)"
        n8_cell = f"{r['n8c']:>6,} ({r['n8r']*100:>5.2f}%)"
        lines.append(f"{r['label']:<28} "
                     f"{n0_cell:>18} {n1_cell:>18} {n4_cell:>18} {n8_cell:>18} "
                     f"{r['n8_drop']:>10.4f}")
    lines.append(sep)
    lines.append("N=0 %  = top-1 accuracy on the original test set (out of num_test).")
    lines.append("N>0 % = retention rate = num_correct_N / num_correct_at_0.")
    lines.append("N=8 drop = 1 - retention(N=8).")

    table = '\n'.join(lines)
    print('\n' + table + '\n')

    txt_path = os.path.join(args.eval_dir, 'zeroshot_results.txt')
    with open(txt_path, 'w') as f:
        f.write(table + '\n')

    csv_path = os.path.join(args.eval_dir, 'zeroshot_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'num_test', 'num_correct_at_0', 'acc_at_0',
                    'num_correct_N1', 'retention_N1',
                    'num_correct_N4', 'retention_N4',
                    'num_correct_N8', 'retention_N8',
                    'N8_drop'])
        for r in rows:
            w.writerow([r['label'], r['n_test'], r['c0'], f"{r['acc_at_0']:.4f}",
                        r['n1c'], f"{r['n1r']:.4f}",
                        r['n4c'], f"{r['n4r']:.4f}",
                        r['n8c'], f"{r['n8r']:.4f}",
                        f"{r['n8_drop']:.4f}"])
    print(f"Wrote {txt_path}")
    print(f"Wrote {csv_path}")


if __name__ == '__main__':
    main()
