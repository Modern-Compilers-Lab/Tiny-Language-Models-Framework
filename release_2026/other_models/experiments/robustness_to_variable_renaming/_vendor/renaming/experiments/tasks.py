"""The three robustness tasks of Section 4.5 (Figure 6, Table 4, Figure 16).

    python -m renaming.experiments.tasks --model DIR --group idrep --out idrep.json
    python -m renaming.experiments.tasks --model DIR --group bench --out bench.json

Each model is scored with the original functions (N = 0) and after renaming N = 1, 2, 4, 8
variables of every test function (Appendix D.6). Two numbers are kept per task and N:

    accuracy   over all test items (Table 4)
    retention  of the items the model got right at N = 0, the fraction still right at N
               (Figure 6, Figure 16). Stored as {k, n, rate}: k of n were still right.

The renamings are identical for every model and seed: each draws N eligible variables with a
seed fixed per function and per N, and gives each a random lowercase name of the same length
that does not occur in the function. Every renaming is verified by applying the inverse mapping.

TASK 1  IDENTIFIER REPRODUCTION (group "idrep"; decmech.cmd_idcurve)
    Teacher-forced. Every occurrence of an eligible variable whose name already appeared earlier
    in the function is scored; it is correct when the model's most likely next token matches
    EVERY token of the name.
TASK 2  VARIABLE-MISUSE LOCALIZATION (group "bench"; decmech.cmd_vmloc; CuBERT, Kanade et al.
    2020). One use of a variable is replaced by another variable in scope. Every candidate edit
    (every eligible use x every other variable in scope) is scored by the length-normalised
    log-likelihood of the whole edited function; the best edit must be at the modified position.
    Ties count as wrong. At most 150 candidate edits and 1,600 items.
TASK 3  CLOZE TEST (group "bench"; decmech.cmd_cloze; CodeXGLUE ClozeTest-all, Python).
    3,000 items sampled evenly; answers recovered from CodeSearchNet; only items whose answer is
    not a variable are kept. The model sees the code before the mask and picks the single-token
    candidate with the highest probability.

WHY TWO GROUPS IN TWO PROCESSES. Importing the variable-misuse code (renaming.lib.varmisuse)
sets WHYSCALE_BLOCK=319 for the whole process, while identifier reproduction runs with the
library's default. This is how the paper's numbers were produced, so the two groups must never
share a process. scripts/04_evaluate.sh runs them separately.

The evaluation data (CodeSearchNet test functions, ClozeTest items) must already be cached:
run `python -m renaming.data.eval_sets` once first (scripts/01_download.sh does).
"""
import argparse
import json
import os
import sys
import time

from renaming import env

NS = (1, 2, 4, 8)
TASKS = ('idreproduction', 'vmloc_localization', 'cloze_all')
TITLES = {'idreproduction': 'Identifier reproduction',
          'vmloc_localization': 'Variable-misuse localization',
          'cloze_all': 'Cloze test'}


def _ns(a, model_dir, device, tmp, **kw):
    """The argument namespace the library's cmd_* functions expect."""
    from types import SimpleNamespace
    return SimpleNamespace(model=model_dir, max_len=a.max_len, device=device, threads=a.threads,
                           limit=a.limit, write_smoke=True, out=tmp, **kw)


def score_idrep(model_dir, a, device):
    """Task 1. Returns {N0_accuracy, N0_n, accuracy: {N: rate}, retention: {N: {k, n, rate}}}."""
    import torch
    DM = env.lib_module('decmech')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    tmp = '%s.idcurve.%d.json' % (a.out, os.getpid())
    try:
        DM.cmd_idcurve(_ns(a, model_dir, device, tmp, n=a.n, batch=a.batch))
        doc = json.load(open(tmp))['idcurves'][os.path.basename(os.path.normpath(model_dir))]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    cell = doc['A_reproduction']
    return {'idreproduction': {
        'N0_accuracy': cell['N0_accuracy']['rate'], 'N0_n': cell['N0_accuracy']['n'],
        'N0_correct': cell['N0_accuracy']['k'],
        'accuracy': {str(N): cell['N%d_accuracy' % N]['rate'] for N in NS},
        'retention': {str(N): cell['N%d_retention' % N] for N in NS},
        'baseline_most_recent_variable': doc['baseline_A_most_recent_variable']['rate'],
        'n_functions': (doc.get('meta') or {}).get('n_functions'),
        'meta': doc.get('meta')}}


def score_bench(model_dir, a, device):
    """Tasks 2 and 3, in the same shape as score_idrep."""
    DM = env.lib_module('decmech')
    name = os.path.basename(os.path.normpath(model_dir))
    tmp = '%s.bench.%d.json' % (a.out, os.getpid())
    out = {}
    try:
        DM.cmd_vmloc(_ns(a, model_dir, device, tmp, n=a.n, n_items=a.n_items,
                         batch=a.vmloc_batch))
        vm = json.load(open(tmp))['varmisuse_locrepair'][name]
        os.remove(tmp)
        DM.cmd_cloze(_ns(a, model_dir, device, tmp, variant='all', n_sample=a.n_sample,
                         batch=a.cloze_batch, per_item=False))
        cz = json.load(open(tmp))['cloze_all'][name]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    blk = vm['mean']                     # length-normalised log-likelihood (the paper's scoring)
    a0 = blk['N0_localization']
    out['vmloc_localization'] = {
        'N0_accuracy': a0['rate'], 'N0_n': a0['n'], 'N0_correct': a0['k'],
        'accuracy': {str(N): blk['N%d' % N]['localization_accuracy'] for N in NS},
        'retention': {str(N): blk['N%d' % N]['retention_localization'] for N in NS},
        'raw': vm}
    c0 = cz['N0_accuracy_safe']
    out['cloze_all'] = {
        'N0_accuracy': c0['rate'], 'N0_n': c0['n'], 'N0_correct': c0['k'],
        'accuracy': {str(N): cz['N%d' % N].get('accuracy_safe') for N in NS},
        'retention': {str(N): cz['N%d' % N]['retention_safe'] for N in NS},
        'raw': cz}
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True, help='a HuggingFace model directory (a snapshot)')
    ap.add_argument('--group', required=True, choices=('idrep', 'bench'))
    ap.add_argument('--out', required=True, help='result JSON to write')
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    ap.add_argument('--n', type=int, default=2000, help='evaluation functions (Appendix D.6)')
    ap.add_argument('--n-items', type=int, default=1600, help='variable-misuse item cap')
    ap.add_argument('--n-sample', type=int, default=3000, help='ClozeTest items sampled')
    ap.add_argument('--max-len', type=int, default=320, help='tokens read per function')
    ap.add_argument('--batch', type=int, default=16, help='batch for identifier reproduction')
    ap.add_argument('--vmloc-batch', type=int, default=128)
    ap.add_argument('--cloze-batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0,
                    help='first N functions only (quick checks and tests; not the paper setting)')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    return ap


def main(argv=None):
    env.setup()
    a = build_parser().parse_args(argv)
    import torch
    dev = 'cuda' if (a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available())) \
        else 'cpu'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)  # the library writes a
    t0 = time.time()                                    # temporary file next to it
    res = score_idrep(a.model, a, dev) if a.group == 'idrep' else score_bench(a.model, a, dev)
    res.update({'experiment': 'tasks_' + a.group, 'model': os.path.abspath(a.model),
                'n': a.n, 'max_len': a.max_len, 'limit': a.limit, 'seconds': time.time() - t0})
    env.write_json(a.out, res)
    for t in TASKS:
        if t in res:
            r = res[t]
            print('%-30s N0 %.4f | retention %s' % (TITLES[t], r['N0_accuracy'] or 0, ' '.join(
                'N%s %.3f' % (N, (r['retention'][N] or {}).get('rate') or 0) for N in r['retention'])))
    print('wrote %s (%.1f min)' % (a.out, (time.time() - t0) / 60))
    return 0


if __name__ == '__main__':
    sys.exit(main())
