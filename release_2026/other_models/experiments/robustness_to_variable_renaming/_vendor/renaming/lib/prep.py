#!/usr/bin/env python3
"""Build both arms of the scale25 corpus in ONE streaming pass.

    python prep.py --out <dir> [--workers 26] [--files N] [--max-docs-per-file N]

Reads the codeparrot/codeparrot-clean json.gz shards in filename order. For every document:
  * keep it iff blake2b(content, key='scale25-v1') % 4 == 0   (the deterministic 25% subset)
  * of the kept documents, 1 in 512 goes to the held-out validation split
  * tokenise the PLAIN text and the RENAMED text with the tiny_starcoder_py tokenizer and
    append both to the arm's uint16 stream, separated by the EOS token

Because both arms are written in the same loop from the same document, the two `.bin` streams
contain exactly the same documents in exactly the same order. Only the identifier bytes differ.

Outputs (few, large files):
    <out>/train_plain.bin  <out>/train_rename.bin
    <out>/val_plain.bin    <out>/val_rename.bin
    <out>/manifest.json
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as CO  # noqa: E402

os.environ.setdefault('HF_HOME', '/scratch/rb4792/hf_cache')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

MODEL = 'bigcode/tiny_starcoder_py'
MAX_CHARS = 200_000          # documents longer than this are truncated (counted in the manifest)
BATCH = 128                  # documents per tokenizer call

_G = {}


def _init(out, n_shards, seed, treatment='rename', mix_frac=0.5):
    from transformers import AutoTokenizer
    _G['tok'] = AutoTokenizer.from_pretrained(MODEL)
    _G['eos'] = _G['tok'].eos_token_id
    _G['out'] = out
    _G['n_shards'] = n_shards
    _G['seed'] = seed
    # TREATMENT of the second arm. 'rename' is the original behaviour (every renameable local
    # of the document replaced); 'halfvar' renames a `mix_frac` fraction of the locals of EVERY
    # function and is the treatment the decoder result is stated for (labnotes 2026-09-20,
    # "[AGENT DEC] FINAL decoder statement: halfvar at 50%"). The treated stream keeps the file
    # name `*_rename.bin` in both cases: it is the second ARM, and train.py/setbudget.py/
    # evalsnap.py address it by that name. `manifest.json` records which treatment produced it.
    _G['treatment'] = treatment
    if treatment == 'halfvar':
        import functools
        _G['renamer'] = functools.partial(CO.rename_document_halfvar, mix_frac=mix_frac)
    else:
        _G['renamer'] = CO.rename_document


def _flush(handles, key, batch_texts):
    if not batch_texts:
        return 0
    enc = _G['tok'](batch_texts)['input_ids']
    eos = _G['eos']
    n = 0
    for ids in enc:
        ids = ids + [eos]
        handles[key].write(np.asarray(ids, dtype=np.uint16).tobytes())
        n += len(ids)
    batch_texts.clear()
    return n


def process_file(job):
    idx, path, max_docs = job
    out, n_shards = _G['out'], _G['n_shards']
    parts = os.path.join(out, 'parts')
    names = {k: os.path.join(parts, f'{k}_{idx:03d}.bin')
             for k in ('train_plain', 'train_rename', 'val_plain', 'val_rename')}
    handles = {k: open(v, 'wb') for k, v in names.items()}

    fresh = CO.FreshNames(shard=idx, n_shards=n_shards, key=_G['seed'])
    rng = random.Random(_G['seed'] * 1000003 + idx)

    st = {'file': os.path.basename(path), 'seen': 0, 'kept': 0, 'val': 0, 'train': 0,
          'truncated': 0, 'decode_fail': 0,
          'rename_status': {'renamed': 0, 'parse_error': 0, 'no_names': 0, 'unchanged': 0},
          'n_names': 0, 'n_sites': 0,
          'tokens': {'train_plain': 0, 'train_rename': 0, 'val_plain': 0, 'val_rename': 0}}
    buf = {k: [] for k in names}
    t0 = time.time()
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            st['seen'] += 1
            if max_docs and st['kept'] >= max_docs:
                break
            try:
                rec = json.loads(line)
            except Exception:
                st['decode_fail'] += 1
                continue
            content = rec.get('content')
            if not content:
                continue
            role = CO.subset_role(content)
            if role == 'skip':
                continue
            st['kept'] += 1
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]
                st['truncated'] += 1
            renamed, status, nn, ns = _G['renamer'](content, fresh, rng.random)
            st['rename_status'][status] = st['rename_status'].get(status, 0) + 1
            st['n_names'] += nn
            st['n_sites'] += ns
            st[role] += 1
            kp, kr = f'{role}_plain', f'{role}_rename'
            buf[kp].append(content)
            buf[kr].append(renamed)
            if len(buf[kp]) >= BATCH:
                st['tokens'][kp] += _flush(handles, kp, buf[kp])
                st['tokens'][kr] += _flush(handles, kr, buf[kr])
    for k in names:
        st['tokens'][k] += _flush(handles, k, buf[k])
        handles[k].close()
    st['seconds'] = time.time() - t0
    st['name_lengths_used'] = {str(L): fresh.counter[L] for L in fresh.counter}
    st['name_space_exhausted'] = fresh.exhausted
    with open(os.path.join(parts, f'stats_{idx:03d}.json'), 'w') as f:
        json.dump(st, f)
    print(f"[{idx:02d}] {st['file']} seen {st['seen']:,} kept {st['kept']:,} "
          f"train_plain {st['tokens']['train_plain']/1e6:.1f}M "
          f"train_rename {st['tokens']['train_rename']/1e6:.1f}M "
          f"{st['seconds']:.0f}s", flush=True)
    return idx


def merge(out, keys, n_files):
    """Concatenate the per-file parts in file order into one .bin per arm, then delete them."""
    parts = os.path.join(out, 'parts')
    sizes = {}
    for k in keys:
        dst = os.path.join(out, f'{k}.bin')
        total = 0
        with open(dst, 'wb') as w:
            for i in range(n_files):
                p = os.path.join(parts, f'{k}_{i:03d}.bin')
                if not os.path.exists(p):
                    continue
                with open(p, 'rb') as r:
                    while True:
                        chunk = r.read(1 << 26)
                        if not chunk:
                            break
                        w.write(chunk)
                        total += len(chunk)
                os.remove(p)
        sizes[k] = total // 2
        print(f'merged {k}: {sizes[k]:,} tokens', flush=True)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--src', default=None, help='dir holding file-*.json.gz')
    ap.add_argument('--workers', type=int, default=26)
    ap.add_argument('--files', type=int, default=0, help='0 = all')
    ap.add_argument('--max-docs-per-file', type=int, default=0, help='0 = all (smoke tests)')
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--subset-mod', type=int, default=CO.SUBSET_MOD,
                    help='keep 1 document in SUBSET-MOD. The default is corpus.SUBSET_MOD (4, '
                         'the 25%% subset). 1 = the whole corpus, which is what a 40,000-step '
                         'run needs if it is not to repeat data.')
    ap.add_argument('--val-mod', type=int, default=CO.VAL_MOD,
                    help='1 kept document in VAL-MOD is held out for validation. The default is '
                         'corpus.VAL_MOD (512); a smoke run over a few hundred documents needs a '
                         'small value or the validation stream comes out empty.')
    ap.add_argument('--treatment', default='rename', choices=('rename', 'halfvar'),
                    help="what the SECOND arm does: 'rename' every local (original behaviour) "
                         "or 'halfvar', a --mix-frac fraction of the locals of every function")
    ap.add_argument('--mix-frac', type=float, default=0.5,
                    help='halfvar only: probability that a given local of a function is renamed')
    a = ap.parse_args()

    CO.VAL_MOD = a.val_mod        # read by CO.subset_role; fork() carries it to the workers
    CO.SUBSET_MOD = a.subset_mod
    src = a.src
    if src is None:
        from huggingface_hub import snapshot_download
        src = snapshot_download('codeparrot/codeparrot-clean', repo_type='dataset',
                                allow_patterns=['*.json.gz'])
    files = sorted(glob.glob(os.path.join(src, 'file-*.json.gz')))
    if a.files:
        files = files[:a.files]
    if not files:
        raise SystemExit(f'no json.gz shards under {src}')
    os.makedirs(os.path.join(a.out, 'parts'), exist_ok=True)
    print(f'{len(files)} shards, {a.workers} workers -> {a.out}', flush=True)

    jobs = [(i, p, a.max_docs_per_file) for i, p in enumerate(files)]
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(min(a.workers, len(jobs)), initializer=_init,
                  initargs=(a.out, len(files), a.seed, a.treatment, a.mix_frac)) as pool:
        for _ in pool.imap_unordered(process_file, jobs):
            pass
    elapsed = time.time() - t0

    keys = ('train_plain', 'train_rename', 'val_plain', 'val_rename')
    sizes = merge(a.out, keys, len(files))

    stats = [json.load(open(os.path.join(a.out, 'parts', f'stats_{i:03d}.json')))
             for i in range(len(files))]
    agg = {'seen': 0, 'kept': 0, 'train': 0, 'val': 0, 'truncated': 0, 'decode_fail': 0,
           'n_names': 0, 'n_sites': 0, 'name_space_exhausted': 0}
    rs = {}
    for s in stats:
        for k in agg:
            agg[k] += s.get(k, 0)
        for k, v in s['rename_status'].items():
            rs[k] = rs.get(k, 0) + v
    man = {
        'model': MODEL, 'seed': a.seed, 'subset_key': CO.SUBSET_KEY.decode(),
        'treatment': a.treatment, 'mix_frac': (a.mix_frac if a.treatment == 'halfvar' else None),
        'subset_mod': CO.SUBSET_MOD, 'val_mod': CO.VAL_MOD,
        'name_lengths': CO.NAME_LENGTHS, 'name_weights': CO.NAME_WEIGHTS,
        'max_chars': MAX_CHARS, 'n_shards': len(files), 'src': src,
        'max_docs_per_file': a.max_docs_per_file,
        'documents': agg, 'rename_status': rs, 'tokens': sizes,
        'token_inflation_rename_over_plain':
            sizes['train_rename'] / sizes['train_plain'] if sizes['train_plain'] else None,
        'prep_seconds': elapsed,
        'per_file': stats,
    }
    with open(os.path.join(a.out, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=1)
    try:
        os.rmdir(os.path.join(a.out, 'parts'))
    except OSError:
        pass
    print(json.dumps({k: v for k, v in man.items() if k != 'per_file'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
