"""Evaluate one pilot model (Appendix D.5) on the pilot's two measures.

    python -m renaming.pilot.eval_pilot --model DIR --what comp --out comp.json
    python -m renaming.pilot.eval_pilot --model DIR --what vm   --out vm.json

comp  NEXT-TOKEN ACCURACY at non-identifier positions (decmech `comp`). Teacher-forced next-token
      accuracy on the evaluation functions (renaming.data.eval_sets), counted only at tokens that
      do not overlap a variable name. Reported for the original functions ("accuracy on the
      original code" in D.5), and the retention of correct predictions at the same positions
      after every local variable is renamed to a random same-length name.
vm    VARIABLE-MISUSE DETECTION (renaming/lib/varmisuse.py, decoder scoring). A two-way choice:
      the model must assign a higher log-likelihood to the correct function than to a copy in
      which one use of a variable is replaced by another variable (chance 0.5; 1,954 items). This
      is NOT the variable-misuse LOCALIZATION task of Section 4.5 (choose the bug position among
      every candidate edit), so the two accuracies are not comparable. Also reported: retention
      after renaming up to 1, 4 and 8 variables.

The two measures run in separate processes, as they did for the paper, because importing the
variable-misuse code changes a library setting (WHYSCALE_BLOCK) for the whole process.

PRECISION. The paper's pilot numbers were computed on CPU in fp32, which is the default here.
With --device cuda, `comp` runs in bf16 (the library's loader picks bf16 on GPUs), so its numbers
can differ slightly; `vm` stays fp32 unless --amp bf16 is given.
"""
import argparse
import json
import os
import sys
import time

from renaming import env


def run_comp(a):
    from types import SimpleNamespace
    DM = env.lib_module('decmech')
    tmp = '%s.comp.%d.json' % (a.out, os.getpid())
    ns = SimpleNamespace(model=a.model, n=a.n, seed=564, max_len=a.max_len, batch=a.batch,
                         gen_n=0, gen_batch=24, no_gen=True, device=a.device,
                         threads=a.threads, limit=a.limit, write_smoke=True, out=tmp)
    try:
        DM.cmd_comp(ns)
        doc = json.load(open(tmp))['completion'][os.path.basename(os.path.normpath(a.model))]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {'accuracy_nonident_original': doc['identity']['unpaired']['nonident']['rate'],
            'accuracy_nonident_renamed': doc['samelen']['unpaired']['nonident']['rate'],
            'retention_nonident_renamed': doc['samelen']['paired_nonident']['retention']['rate'],
            'raw': doc}


def run_vm(a):
    VM = env.lib_module('varmisuse')
    items, meta = VM.build_items(a.limit or None)
    out = VM.run_model(a.model, items, 'decoder', a.device, a.vm_batch, 8, amp=(a.amp == 'bf16'))
    res = VM.summarize(out, items, meta, 'decoder', 8)
    d = res['kinds']['var']
    return {'vm_detection_accuracy': d['accuracy'],
            'vm_detection_retention': {str(N): d.get('retention_N%d' % N) for N in (1, 4, 8)},
            'n_items': d['n_items'], 'raw': res}


def main(argv=None):
    env.setup()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--what', required=True, choices=('comp', 'vm'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--max-len', type=int, default=320)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--vm-batch', type=int, default=160)
    ap.add_argument('--amp', default='off', choices=('off', 'bf16'))
    ap.add_argument('--limit', type=int, default=0, help='quick checks only')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)  # the library writes a
    t0 = time.time()                                    # temporary file next to it
    res = run_comp(a) if a.what == 'comp' else run_vm(a)
    res.update({'experiment': 'pilot_' + a.what, 'model': os.path.abspath(a.model),
                'limit': a.limit, 'seconds': time.time() - t0})
    env.write_json(a.out, res)
    print({k: v for k, v in res.items() if k != 'raw'})
    return 0


if __name__ == '__main__':
    sys.exit(main())
