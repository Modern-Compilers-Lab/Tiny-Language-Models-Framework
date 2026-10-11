#!/usr/bin/env python3
"""[AGENT DEC] WHERE does a BPE code decoder break under variable renaming?  (Stage 2: D0, D1)

QUESTION
  The character-level TinyPy tracer fails at one decision: where an unfamiliar name ENDS (q=0.1%:
  99.8% of first errors on failed renamed traces are in a name, 78% exactly at the name end).
  Registered (labnotes 2026-09-19 "REGISTERED (before any run): why MLM encoders and BPE decoders
  fail under renaming"): where does a decoder with a BYTE-PAIR tokenizer fail?

  D0  BRIDGE. The TinyPy tracer trained with a BPE-4096 tokenizer at q=0.1% (ckpt_p0.1sb).
      Prediction: >= 50% of first errors on failed renamed traces are at the name END.
  D1  WHERE on real code. GPT-BigCode replicas of tiny_starcoder_py (20L, width 192, its BPE)
      trained from scratch on CodeSearchNet Python (mech_dec_{plain,highdiv}_s2), scored on the
      real6 CSN-test functions under identity / samelen / permute.
      P-D1: the renamed-minus-original drop concentrates at ENTRIES of COPYABLE occurrences:
            copyable-entry drop >= 3x exit drop; permute <= 1/4 of samelen at copyable entries.
      FALSIFIER: exit drop >= entry drop.

DESIGN (see the "[AGENT DEC] design" labnotes entry, written before any number existed)
  D0. Text = the character tokenizer's reconstruction (spaces dropped), exactly what the BPE model
      was trained on (interp.bpe_boundary makes the same choice). One teacher-forced forward per
      trace; a trace FAILS if any target token (every BPE token after the first '\\n#STEP\\n') is not
      the argmax. Its FIRST wrong token is classified by the ROLE OF THE TARGET TOKEN (primary):
        {code,state}_name_first  first BPE piece of a name
        {code,state}_name_inner  a later piece of the same name
        {code,state}_name_end    the first token AFTER a name (the delimiter); split into
                                 `continue` (the model predicted a token starting with a letter,
                                 i.e. it kept writing the name) and `other`
        value / value_end        digits / '-' of a state-dump value; the ';' or newline closing it
        struct                   everything else (code literals, '@', '$', '#STEP', keywords)
      Name spans are maximal [a-z]+ words of the SPACED text minus the keyword `if`, mapped onto the
      space-free text (so `if kb` -> keyword + name even though the model sees `ifkb`). A name is a
      state name iff it lies after the '$' of its line. Secondary (string level): where the
      predicted token's STRING first departs from the reference text, classified with the char
      model's classes; `seg_only` = the predicted string is consistent with the reference (a
      non-canonical segmentation, not a textual error). Also the first STRING-divergent wrong
      token of each trace.
  D1. Occurrences = every Name / arg / ExceptHandler-name site of a RENAMED local (the only sites
      the real6 views rewrite), located by walking the original and the view ASTs in parallel.
      Self-checks per function: each span spells its name, and the text BETWEEN occurrences is
      byte-identical across identity / samelen / permute (a missed site would break this).
      Functions failing a check are dropped from every view and counted.
      Token classes (a token's start ignores its leading whitespace, since ' x' carries the space):
        entry     first BPE piece of an occurrence;  interior  a later piece
        exit      the token whose raw start is the occurrence's end (first token after it)
        other     everything else;  straddle  a token crossing an occurrence edge (counted, excluded)
      COPYABLE = the same variable occurred earlier in the function (its string is in the left
      context); FIRST = first mention. Never pooled.
      Entry / interior / name-exact / per-occurrence first error: compared PER OCCURRENCE (the
      target itself changes under renaming). Exit: paired per occurrence, and only when the exit
      token id is the same in both texts. Other: paired by aligned character span + same id.
      Context window = 320 tokens (repair.py scale --max-len, the replicas' training length).
      Drops carry a function-level cluster bootstrap 95% CI (2000 resamples).

RUN
  export HF_HOME=/scratch/rb4792/hf_cache XDG_CACHE_HOME=/scratch/rb4792/xdg_cache PYTHONHASHSEED=0
  python decmech.py d0 [--ckpt work/ckpt_p0.1sb/checkpoint_1.00.pth] [--n 2048]
  python decmech.py d1 --model work/mech_dec_plain_s2 [--n 2000] [--max-len 320]
  python decmech.py d1 --model bigcode/tiny_starcoder_py
  --limit N on either = smoke test (results NOT written unless --write-smoke).
OUTPUT
  work/dec_mech.json  {config, d0: {ckpt: {id, rv, contrast}}, d1: {model: {view: ...}}}
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORK = os.path.join(os.path.dirname(HERE), 'work')
OUT = os.path.join(WORK, 'dec_mech.json')
BOOT = 2000


# ===========================================================================
# shared
# ===========================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return {'k': int(k), 'n': 0, 'rate': None}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {'k': int(k), 'n': int(n), 'rate': float(p), 'ci_lo': float(c - h), 'ci_hi': float(c + h)}


def store(keys, value, path=OUT):
    """Merge `value` at doc[keys[0]][keys[1]]... under an exclusive lock on the file itself."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        txt = f.read()
        doc = json.loads(txt) if txt.strip() else {}
        d = doc
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        f.seek(0)
        f.truncate()
        json.dump(doc, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def provenance():
    import platform
    import socket
    return {'host': socket.gethostname(), 'python': platform.python_version(),
            'torch': torch.__version__, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
            'slurm_job': os.environ.get('SLURM_JOB_ID')}


def device_of(name):
    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================================
# D0: first-divergence localisation on the TinyPy BPE tracer
# ===========================================================================
TOK_CLASSES = ('code_name_first', 'code_name_inner', 'code_name_end',
               'state_name_first', 'state_name_inner', 'state_name_end',
               'value', 'value_end', 'struct')
CHAR_CLASSES = ('code_name_start', 'code_name_cont', 'code_name_end',
                'state_name_start', 'state_name_cont', 'state_name_end',
                'value', 'value_end', 'struct')
R_START, R_CONT, R_END, R_VAL, R_VEND, R_STRUCT = 0, 1, 2, 6, 7, 8   # +3 for state names
STEP = '\n#STEP\n'


def d0_layout(example):
    """-> dict(recon, prompt_end, spans[(x,y,is_state)], role[int8 per char], span_of, end_of).

    `role` uses CHAR_CLASSES indices, i.e. the character model's taxonomy on this text.
    """
    spaced = example + '\n\n'
    keep = [i for i, ch in enumerate(spaced) if ch != ' ']
    recon = ''.join(spaced[i] for i in keep)
    pos = np.full(len(spaced) + 1, -1, dtype=np.int64)
    for j, i in enumerate(keep):
        pos[i] = j
    L = len(recon)
    # state regions and values
    in_state = np.zeros(L + 1, dtype=bool)
    role = np.full(L + 1, R_STRUCT, dtype=np.int8)          # index L = "past the end"
    state = val = False
    for c, ch in enumerate(recon):
        if ch == '\n':
            if val:
                role[c] = R_VEND
            state = val = False
            continue
        if ch == '$':
            state, val = True, False
            continue
        if state:
            in_state[c] = True
            if ch == '?':
                val = True
            elif ch == ';':
                role[c] = R_VEND if val else R_STRUCT
                val = False
            elif val and (ch.isdigit() or ch == '-'):
                role[c] = R_VAL
    spans = []
    span_of = np.full(L + 1, -1, dtype=np.int64)
    end_of = np.full(L + 1, -1, dtype=np.int64)
    for m in re.finditer(r'[a-z]+', spaced):
        if m.group() == 'if':
            continue
        x, y = int(pos[m.start()]), int(pos[m.end() - 1]) + 1
        st = bool(in_state[x])
        k = len(spans)
        spans.append((x, y, st))
        span_of[x:y] = k
        off = 3 if st else 0
        role[x] = R_START + off
        role[x + 1:y] = R_CONT + off
        end_of[y] = k
        role[y] = R_END + off
    p = recon.find(STEP)
    if p < 0:
        raise ValueError('no #STEP in example')
    return dict(recon=recon, prompt_end=p + len(STEP), spans=spans, role=role,
                span_of=span_of, end_of=end_of)


def d0_token_class(a, b, lay):
    """Primary (token-level) class of a target BPE token [a, b). -> (class index, straddle)."""
    sid = lay['span_of'][a]
    if sid >= 0:
        x, y, st = lay['spans'][sid]
        off = 3 if st else 0
        return (0 if a == x else 1) + off, b > y
    eid = lay['end_of'][a]
    if eid >= 0:
        return 2 + (3 if lay['spans'][eid][2] else 0), False
    r = int(lay['role'][a])
    return (r if r in (R_VAL, R_VEND) else R_STRUCT), False


def d0_string_class(a, pred_str, lay):
    """Secondary: where the predicted STRING departs from the reference text."""
    ref = lay['recon'][a:]
    if ref.startswith(pred_str):
        return 'seg_only'
    i = 0
    while i < len(pred_str) and i < len(ref) and pred_str[i] == ref[i]:
        i += 1
    return CHAR_CLASSES[int(lay['role'][min(a + i, len(lay['recon']))])]


def _gpt_argmax(model, seqs, device, batch):
    """Teacher-forced argmax for a list of id lists (tiny GPT, ALiBi, right padding)."""
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        T = max(len(seqs[i]) for i in idx)
        inp = torch.zeros((len(idx), T), dtype=torch.long, device=device)
        for j, i in enumerate(idx):
            inp[j, :len(seqs[i])] = torch.tensor(seqs[i], device=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                             enabled=device.type == 'cuda'):
            logits, _ = model(inp)
        pred = logits.argmax(-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = pred[j, :len(seqs[i])]
    return out


def d0_score(model, bpe, examples, device, batch):
    """Per-set D0 statistics."""
    ts = [bpe.id_to_token(i) for i in range(bpe.get_vocab_size())]
    letter_start = np.array([bool(t) and t[0].islower() and t[0].isalpha() for t in ts])
    lays, seqs, offs = [], [], []
    straddles = 0
    for ex in examples:
        lay = d0_layout(ex)
        enc = bpe.encode(lay['recon'])
        if ''.join(enc.tokens) != lay['recon']:
            raise ValueError('BPE round trip failed')
        lays.append(lay)
        seqs.append(enc.ids)
        offs.append(enc.offsets)
    preds = _gpt_argmax(model, seqs, device, batch)
    ok_by = np.zeros(len(TOK_CLASSES), dtype=np.int64)
    n_by = np.zeros(len(TOK_CLASSES), dtype=np.int64)
    end_notletter = [0, 0]                       # interp.bpe_boundary's looser end metric
    first_tok, first_end_sub, first_str, first_len = Counter(), Counter(), Counter(), Counter()
    first_pos_in_name = Counter()
    strfirst = Counter()
    n_fail = n_strfail = 0
    fail_examples = []
    maxlen = 0
    for e, (lay, ids, off, pr) in enumerate(zip(lays, seqs, offs, preds)):
        maxlen = max(maxlen, len(ids))
        first = None
        sfirst = None
        for k in range(1, len(ids)):
            a, b = off[k]
            if a < lay['prompt_end']:
                continue
            c, strad = d0_token_class(a, b, lay)
            straddles += int(strad)
            p = int(pr[k - 1])
            ok = p == ids[k]
            n_by[c] += 1
            ok_by[c] += ok
            if c in (2, 5):
                end_notletter[1] += 1
                end_notletter[0] += int(not letter_start[p])
            if not ok:
                if first is None:
                    first = (k, c, p, a)
                if sfirst is None:
                    sc = d0_string_class(a, ts[p], lay)
                    if sc != 'seg_only':
                        sfirst = sc
        if first is not None:
            n_fail += 1
            k, c, p, a = first
            first_tok[TOK_CLASSES[c]] += 1
            if c in (2, 5):
                first_end_sub['continue' if letter_start[p] else 'other'] += 1
            first_str[d0_string_class(a, ts[p], lay)] += 1
            if c in (0, 1, 3, 4):
                x, y, _ = lay['spans'][lay['span_of'][a]]
                first_len[str(y - x)] += 1
            elif c in (2, 5):
                x, y, _ = lay['spans'][lay['end_of'][a]]
                first_len[str(y - x)] += 1
            if len(fail_examples) < 5:
                fail_examples.append({'example': e, 'class': TOK_CLASSES[c],
                                      'target': ts[ids[k]], 'pred': ts[p],
                                      'context': lay['recon'][max(0, a - 40):a]})
        if sfirst is not None:
            n_strfail += 1
            strfirst[sfirst] += 1

    def hist(cnt, n, keys):
        return {kk: wilson(cnt.get(kk, 0), n) for kk in keys}
    NAME_T = ('code_name_first', 'code_name_inner', 'code_name_end')
    NAME_C = ('code_name_start', 'code_name_cont', 'code_name_end')
    rec = {
        'n_traces': len(examples), 'max_bpe_len': int(maxlen),
        'straddling_target_tokens': int(straddles),
        'hard_tf': wilson(len(examples) - n_fail, len(examples)),
        'n_failed': n_fail,
        'first_error_token_level': {
            'counts': dict(first_tok),
            'shares': hist(first_tok, n_fail, TOK_CLASSES),
            'name_end_code_plus_state': wilson(first_tok['code_name_end']
                                               + first_tok['state_name_end'], n_fail),
            'code_name_any': wilson(sum(first_tok[x] for x in NAME_T), n_fail),
            'state_name_any': wilson(sum(first_tok[x] for x in TOK_CLASSES[3:6]), n_fail),
            'name_end_split': {'continue_name(letter-start pred)': first_end_sub['continue'],
                               'other_pred': first_end_sub['other']},
            'name_char_length_of_first_error': dict(sorted(first_len.items())),
        },
        'first_error_string_level': {
            'definition': 'the SAME first wrong tokens, classed by where the predicted string '
                          'departs from the reference text (char-model classes); seg_only = '
                          'predicted string consistent with the text (non-canonical split)',
            'counts': dict(first_str),
            'shares': hist(first_str, n_fail, CHAR_CLASSES + ('seg_only',)),
            'code_name_any': wilson(sum(first_str[x] for x in NAME_C), n_fail),
        },
        'first_string_divergent_error': {
            'definition': 'per trace, the first target whose predicted string is NOT consistent '
                          'with the reference text; traces with none count as string-correct',
            'n_failed': n_strfail,
            'counts': dict(strfirst),
            'shares': hist(strfirst, n_strfail, CHAR_CLASSES),
        },
        'token_accuracy_by_class': {TOK_CLASSES[i]: wilson(int(ok_by[i]), int(n_by[i]))
                                    for i in range(len(TOK_CLASSES))},
        'name_end_pred_not_letter_start': wilson(*end_notletter),
        'examples_of_first_errors': fail_examples,
    }
    return rec


def cmd_d0(a):
    import interp as I
    from tokenizers import Tokenizer
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    bpe = Tokenizer.from_file(os.path.join(a.data, 'bpe.json'))
    model = I.load_model(a.ckpt, dev)
    V = model.token_embedding_table.weight.shape[0]
    if V != bpe.get_vocab_size():
        raise SystemExit('vocab mismatch %d vs %d' % (V, bpe.get_vocab_size()))
    n = a.limit or a.n
    key = '%s@%s' % (os.path.basename(os.path.dirname(a.ckpt)),
                     os.path.basename(a.ckpt).replace('.pth', ''))
    res = {'checkpoint': a.ckpt, 'data': a.data, 'tokenizer': os.path.join(a.data, 'bpe.json'),
           'n_requested': n, 'provenance': provenance()}
    # sanity: our reconstruction == the char tokenizer's reconstruction
    ex0 = I.load_examples(os.path.join(a.data, 'test.txt'), 3)
    inv = I.TOK.decod_map
    for ex in ex0:
        ids_c, _ = I.encode(ex)
        assert ''.join(inv[i] for i in ids_c) == d0_layout(ex)['recon']
    for setname, fname in (('id', 'test.txt'), ('rv', 'test_rv.txt')):
        ex = I.load_examples(os.path.join(a.data, fname), n)
        t0 = time.time()
        r = d0_score(model, bpe, ex, dev, a.batch)
        r['file'] = fname
        r['seconds'] = round(time.time() - t0, 1)
        res[setname] = r
        ft = r['first_error_token_level']
        print('%s %s: n=%d failed=%d  END(code+state)=%s  first-error counts %s  | string-level %s'
              % (key, setname, r['n_traces'], r['n_failed'],
                 '%.4f' % ft['name_end_code_plus_state']['rate']
                 if ft['name_end_code_plus_state']['rate'] is not None else '-',
                 ft['counts'], r['first_error_string_level']['counts']), flush=True)
    idf, rvf = res['id']['n_failed'], res['rv']['n_failed']
    se = res['rv']['first_error_token_level']['name_end_code_plus_state']
    res['contrast'] = {
        'failed_traces_id_vs_rv': [idf, rvf],
        'hard_tf_drop': res['id']['hard_tf']['rate'] - res['rv']['hard_tf']['rate'],
        'name_end_token_acc_id_vs_rv': [
            res[s]['token_accuracy_by_class']['code_name_end']['rate'] for s in ('id', 'rv')],
        'registered_D0': '>= 50% of first errors on failed renamed traces at the name END',
        'primary_share_rv': se,
        'code_only_share_rv': res['rv']['first_error_token_level']['shares']['code_name_end'],
        'verdict_point': (None if se['rate'] is None
                          else ('HELD' if se['rate'] >= 0.5 else 'FAILED')),
    }
    print('D0 %s verdict (primary, point): %s  share=%s'
          % (key, res['contrast']['verdict_point'], se), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['d0', key], res, a.out)
    print('wrote', a.out, ['d0', key])
    return 0


# ===========================================================================
# D1: where on real code (entry / interior / exit / other; copyable vs first)
# ===========================================================================
def _line_starts(text):
    starts, pos = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
    starts.append(pos)
    return starts


def _off(text, starts, row, col):
    line_end = starts[row] if row < len(starts) else len(text)
    line = text[starts[row - 1]:line_end]
    return starts[row - 1] + len(line.encode('utf-8')[:col].decode('utf-8', errors='strict'))


def _sites(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.append(('name', n, n.id))
        elif isinstance(n, ast.arg):
            out.append(('arg', n, n.arg))
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.append(('exc', n, n.name))
    return out


def occurrences(orig, view, names):
    """Occurrence spans of the renamed locals in `view`, aligned with `orig`.

    -> list of (x, y, key) sorted by x, key = the ORIGINAL name (variable identity), or None if
    the parallel AST walk does not line up or a span does not spell its name.
    """
    names = set(names)
    to, tv = ast.parse(orig), ast.parse(view)
    so, sv = _sites(to), _sites(tv)
    if len(so) != len(sv):
        return None
    st = _line_starts(view)
    occ = []
    for (ko, no, nmo), (kv, nv, nmv) in zip(so, sv):
        if ko != kv:
            return None
        if nmo not in names:
            if nmo != nmv:
                return None
            continue
        if ko == 'exc':
            x0 = _off(view, st, nv.lineno, nv.col_offset)
            m = re.compile(r'\bas\s+(' + re.escape(nmv) + r')\b').search(view, x0)
            if m is None or '\n' in view[x0:m.start(1)]:
                return None
            x, y = m.start(1), m.end(1)
        else:
            x = _off(view, st, nv.lineno, nv.col_offset)
            y = x + len(nmv)
        if view[x:y] != nmv:
            return None
        occ.append((x, y, nmo))
    occ.sort()
    return occ


def segments(text, occ):
    """Text between consecutive occurrences (len(occ)+1 strings)."""
    segs, p = [], 0
    for x, y, _ in occ:
        segs.append(text[p:x])
        p = y
    segs.append(text[p:])
    return segs


def token_layout(text, offsets, occ):
    """Per-token class + per-occurrence token indices for one view.

    info per occurrence: cover (token indices of its pieces), name_ok (all pieces present, no
    token crosses its edges), exit (index of the token whose raw start is the occurrence end, or
    None: end of text / truncated / straddle), straddle.
    """
    T = len(offsets)
    es = np.zeros(T, dtype=np.int64)
    for k, (a, b) in enumerate(offsets):
        s = text[a:b]
        lead = len(s) - len(s.lstrip())
        es[k] = a + lead if lead < len(s) else b          # whitespace-only: starts nowhere
    starts_at = {}
    for k, (a, b) in enumerate(offsets):
        if b > a:
            starts_at.setdefault(a, k)
    cls = np.array(['other'] * T, dtype=object)
    occ_tok = []
    for x, y, key in occ:
        cover = [k for k in range(T) if es[k] < y and offsets[k][1] > x and offsets[k][1] > es[k]]
        if not cover:
            occ_tok.append({'cover': [], 'name_ok': False, 'exit': None, 'straddle': False,
                            'truncated': True})
            continue
        k0 = cover[0]
        strad_in = es[k0] != x
        strad_out = offsets[cover[-1]][1] > y
        for j, k in enumerate(cover):
            if (k == k0 and strad_in) or offsets[k][1] > y:
                cls[k] = 'straddle'
            else:
                cls[k] = 'entry' if j == 0 else 'interior'
        truncated = offsets[cover[-1]][1] < y
        ex = starts_at.get(y)
        if ex is not None and cls[ex] == 'other':
            cls[ex] = 'exit'
        occ_tok.append({'cover': cover, 'name_ok': not (strad_in or strad_out or truncated),
                        'exit': ex, 'straddle': bool(strad_in or strad_out),
                        'truncated': bool(truncated)})
    return cls, occ_tok, es


def load_lm(path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    dt = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(device).eval()
    return tok, model


@torch.no_grad()
def lm_argmax(model, tok, texts, device, batch, max_len):
    enc = tok(texts, truncation=True, max_length=max_len, return_offsets_mapping=True,
              add_special_tokens=False)
    ids_all, off_all = enc['input_ids'], enc['offset_mapping']
    preds = [np.zeros(0, dtype=np.int64)] * len(texts)
    order = [i for i in sorted(range(len(texts)), key=lambda i: len(ids_all[i]), reverse=True)
             if len(ids_all[i]) >= 2]
    pad = tok.pad_token_id
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        T = max(len(ids_all[i]) for i in idx)
        inp = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            inp[j, :len(ids_all[i])] = torch.tensor(ids_all[i])
            att[j, :len(ids_all[i])] = 1
        logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
        pr = logits.argmax(-1).cpu().numpy()
        for j, i in enumerate(idx):
            preds[i] = pr[j, :len(ids_all[i])]
    return ids_all, [[tuple(o) for o in off] for off in off_all], preds


def pieces_bin(n):
    return str(n) if n < 5 else '5+'


def cluster_boot(F_rows, fn, B=BOOT, seed=0):
    """Function-level cluster bootstrap. F_rows: array [F, m] of per-function sums; fn maps the
    column sums to a scalar. -> (lo, hi)."""
    F_rows = np.asarray(F_rows, dtype=np.float64)
    if len(F_rows) == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(F_rows), len(F_rows))
        with np.errstate(divide='ignore', invalid='ignore'):
            vals[b] = fn(F_rows[idx].sum(0))
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _safe_div(a, b):
    return a / b if b else float('nan')


def d1_pair(O, Vw, fn_ids):
    """Compare the original view O with a renamed view Vw (both per-function dicts)."""
    out = {}
    for grp in ('copyable', 'first'):
        # per-function rows: [entry_ok_o, entry_ok_v, n_occ, exact_o, exact_v,
        #                     int_ok_o, int_n_o, int_ok_v, int_n_v, exit_ok_o, exit_ok_v, n_exit]
        rows = []
        fe = {'o': Counter(), 'v': Counter()}
        cond = {'o': [0, 0], 'v': [0, 0]}        # exit ok | name exact
        strata = defaultdict(lambda: np.zeros(8))  # by renamed-view piece bin
        strata_o = defaultdict(lambda: np.zeros(5))  # original view by its own piece bin
        excl = Counter()
        n_fe = 0
        base = {'uniform_prev': [], 'recency_var': [], 'recency_piece_o': [],
                'recency_piece_v': []}
        exit_tokens = Counter()
        for f in fn_ids:
            r = np.zeros(12)
            for oi in range(len(O[f]['occ'])):
                if O[f]['copyable'][oi] != (grp == 'copyable'):
                    continue
                io, iv = O[f]['occ_tok'][oi], Vw[f]['occ_tok'][oi]
                if not (io['name_ok'] and iv['name_ok']):
                    excl['straddle' if io['straddle'] or iv['straddle'] else 'truncated'] += 1
                    continue
                oko, okv = O[f]['ok'], Vw[f]['ok']
                eo, ev = oko[io['cover'][0]], okv[iv['cover'][0]]
                xo = all(oko[k] for k in io['cover'])
                xv = all(okv[k] for k in iv['cover'])
                r[0] += eo; r[1] += ev; r[2] += 1; r[3] += xo; r[4] += xv
                r[5] += sum(oko[k] for k in io['cover'][1:]); r[6] += len(io['cover']) - 1
                r[7] += sum(okv[k] for k in iv['cover'][1:]); r[8] += len(iv['cover']) - 1
                has_exit = io['exit'] is not None and iv['exit'] is not None
                if not has_exit:
                    excl['no_exit_token(end_of_text_or_truncated)'] += 1
                elif O[f]['ids'][io['exit']] == Vw[f]['ids'][iv['exit']]:
                    r[9] += oko[io['exit']]; r[10] += okv[iv['exit']]; r[11] += 1
                    exit_tokens[O[f]['ids'][io['exit']]] += 1
                else:
                    excl['exit_token_differs'] += 1
                if has_exit:
                    n_fe += 1
                    for tag, info, ok in (('o', io, oko), ('v', iv, okv)):
                        seq = [('entry', info['cover'][0])] + \
                              [('interior', k) for k in info['cover'][1:]] + [('exit', info['exit'])]
                        loc = next((nm for nm, k in seq if not ok[k]), 'none')
                        fe[tag][loc] += 1
                        if all(ok[k] for k in info['cover']):
                            cond[tag][1] += 1
                            cond[tag][0] += ok[info['exit']]
                pb = pieces_bin(len(iv['cover']))
                strata[pb] += np.array([eo, ev, 1, xo, xv,
                                        okv[iv['exit']] if (xv and has_exit) else 0,
                                        int(xv and has_exit), int(ev and not xv)])
                pbo = pieces_bin(len(io['cover']))
                strata_o[pbo] += np.array([eo, xo, 1, oko[io['exit']] if (xo and has_exit) else 0,
                                           int(xo and has_exit)])
                if grp == 'copyable':
                    base['uniform_prev'].append(1.0 / O[f]['n_prev_vars'][oi])
                    base['recency_var'].append(O[f]['recent_same'][oi])
                    base['recency_piece_o'].append(O[f]['recent_piece_same'][oi])
                    base['recency_piece_v'].append(Vw[f]['recent_piece_same'][oi])
            rows.append(r)
        rows = np.array(rows) if rows else np.zeros((0, 12))
        S = rows.sum(0)
        g = {
            'n_occurrences_paired': int(S[2]),
            'excluded': dict(excl),
            'entry': {'orig': wilson(S[0], S[2]), 'view': wilson(S[1], S[2]),
                      'drop': _safe_div(S[0] - S[1], S[2]),
                      'drop_ci95_cluster': cluster_boot(rows[:, [0, 1, 2]],
                                                        lambda s: (s[0] - s[1]) / s[2])},
            'interior_per_piece': {'orig': wilson(S[5], S[6]), 'view': wilson(S[7], S[8]),
                                   'drop': _safe_div(S[5], S[6]) - _safe_div(S[7], S[8]),
                                   'drop_ci95_cluster': cluster_boot(
                                       rows[:, [5, 6, 7, 8]],
                                       lambda s: s[0] / s[1] - s[2] / s[3])},
            'name_exact_all_pieces': {'orig': wilson(S[3], S[2]), 'view': wilson(S[4], S[2]),
                                      'drop': _safe_div(S[3] - S[4], S[2])},
            'exit_paired_same_token': {'orig': wilson(S[9], S[11]), 'view': wilson(S[10], S[11]),
                                       'drop': _safe_div(S[9] - S[10], S[11]),
                                       'drop_ci95_cluster': cluster_boot(
                                           rows[:, [9, 10, 11]],
                                           lambda s: (s[0] - s[1]) / s[2]),
                                       'majority_token_baseline': wilson(
                                           exit_tokens.most_common(1)[0][1] if exit_tokens else 0,
                                           S[11])},
            'exit_given_name_exact': {'orig': wilson(*cond['o']), 'view': wilson(*cond['v'])},
            'first_error_location': {
                'n': n_fe, 'definition': 'first wrong position among [entry, interior..., exit], '
                                          'occurrences with an exit token in both texts',
                'orig': {k: wilson(fe['o'][k], n_fe) for k in ('entry', 'interior', 'exit', 'none')},
                'view': {k: wilson(fe['v'][k], n_fe) for k in ('entry', 'interior', 'exit', 'none')}},
            'by_view_pieces': {
                pb: {'n': int(v[2]), 'entry_orig': _safe_div(v[0], v[2]),
                     'entry_view': _safe_div(v[1], v[2]), 'entry_drop': _safe_div(v[0] - v[1], v[2]),
                     'exact_orig': _safe_div(v[3], v[2]), 'exact_view': _safe_div(v[4], v[2]),
                     'view_exit_given_exact': _safe_div(v[5], v[6]),
                     'view_entry_right_but_name_wrong': _safe_div(v[7], v[2])}
                for pb, v in sorted(strata.items())},
            'orig_by_orig_pieces': {
                pb: {'n': int(v[2]), 'entry': _safe_div(v[0], v[2]), 'exact': _safe_div(v[1], v[2]),
                     'exit_given_exact': _safe_div(v[3], v[4])}
                for pb, v in sorted(strata_o.items())},
        }
        if grp == 'copyable' and base['uniform_prev']:
            g['entry_baselines'] = {
                'uniform_over_previous_variables': float(np.mean(base['uniform_prev'])),
                'most_recent_variable': float(np.mean(base['recency_var'])),
                'most_recent_first_piece_orig': float(np.mean(base['recency_piece_o'])),
                'most_recent_first_piece_view': float(np.mean(base['recency_piece_v'])),
                'note': 'upper-bound-ish heuristics that KNOW an identifier comes next and copy '
                        'an earlier variable; model accuracy is over all entry decisions'}
        out[grp] = g
    # other: paired by aligned span and same id
    rows = []
    maj = Counter()
    for f in fn_ids:
        a_map = O[f]['other_index']
        ok_o, ok_v, n = 0, 0, 0
        for key, kv in Vw[f]['other_index'].items():
            ko = a_map.get(key)
            if ko is None:
                continue
            n += 1
            ok_o += O[f]['ok'][ko]
            ok_v += Vw[f]['ok'][kv]
            maj[key[2]] += 1
        rows.append([ok_o, ok_v, n])
    rows = np.array(rows) if rows else np.zeros((0, 3))
    S = rows.sum(0)
    out['other_paired'] = {'orig': wilson(S[0], S[2]), 'view': wilson(S[1], S[2]),
                           'drop': _safe_div(S[0] - S[1], S[2]),
                           'drop_ci95_cluster': cluster_boot(rows, lambda s: (s[0] - s[1]) / s[2]),
                           'majority_token_baseline': wilson(
                               maj.most_common(1)[0][1] if maj else 0, S[2])}
    return out


def d1_view_record(text, occ, copyable, ids, offs, pred, rec_first_piece):
    cls, occ_tok, es = token_layout(text, offs, occ)
    ok = np.zeros(len(ids), dtype=bool)
    ok[1:] = np.asarray(pred[:-1]) == np.asarray(ids[1:])
    ok[0] = False
    # "other" tokens keyed by the segment they sit in and their offset inside it
    segstart = [0] + [y for _, y, _ in occ]
    segend = [x for x, _, _ in occ] + [len(text)]
    other_index = {}
    for k in range(1, len(ids)):
        if cls[k] != 'other':
            continue
        a, b = offs[k]
        if b <= a:
            continue
        for j in range(len(segstart)):
            if segstart[j] <= a and b <= segend[j]:
                other_index[(j, a - segstart[j], int(ids[k]), b - a)] = k
                break
    # the first piece of the most recent previous occurrence (baseline)
    rps = []
    for oi, info in enumerate(occ_tok):
        if oi == 0 or not info['cover'] or not occ_tok[oi - 1]['cover']:
            rps.append(0)
            continue
        cur = text[es[info['cover'][0]]:offs[info['cover'][0]][1]]
        prv = text[es[occ_tok[oi - 1]['cover'][0]]:offs[occ_tok[oi - 1]['cover'][0]][1]]
        rps.append(int(cur == prv))
    return {'ids': list(ids), 'ok': ok, 'occ': occ, 'occ_tok': occ_tok, 'copyable': copyable,
            'other_index': other_index, 'cls': cls, 'recent_piece_same': rps,
            'pred': np.asarray(pred), 'text': text, 'es': es, 'offs': offs}


def d1_diagnostics(O, Vw, fn_ids, tok):
    """POST-HOC (added after the first D1 numbers; not part of the registered test).

    What the model predicts when it is wrong, on exactly the occurrences d1_pair scores:
      copyable ENTRY errors -> other_in_context_variable (first piece of another renamed local
        already in the context: wrong variable) | target_name_other_split (a proper prefix of the
        correct name, incl. a leading-space variant: right name, wrong segmentation) |
        identifier_in_left_context (an identifier-like token that occurs as a word earlier in the
        text: attributes, globals, unrenamed names) | identifier_not_in_context (an identifier-like
        token absent from the left context: a familiar name from training) | keyword | non_identifier
      EXIT errors (both groups) -> continues_identifier (predicted token glued to the name: starts
        with [A-Za-z0-9_], i.e. the model keeps writing the name) | other
    """
    import keyword as _kw
    IDR = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
    dec = {}

    def s_of(i):
        if i not in dec:
            dec[i] = tok.decode([int(i)])
        return dec[i]
    out = {}
    for grp in ('copyable', 'first'):
        ent = {'o': Counter(), 'v': Counter()}
        exi = {'o': Counter(), 'v': Counter()}
        n_ent = n_exit = 0
        for f in fn_ids:
            for oi in range(len(O[f]['occ'])):
                if O[f]['copyable'][oi] != (grp == 'copyable'):
                    continue
                io, iv = O[f]['occ_tok'][oi], Vw[f]['occ_tok'][oi]
                if not (io['name_ok'] and iv['name_ok']):
                    continue
                n_ent += 1
                has_exit = (io['exit'] is not None and iv['exit'] is not None
                            and O[f]['ids'][io['exit']] == Vw[f]['ids'][iv['exit']])
                n_exit += int(has_exit)
                for tag, R, info in (('o', O[f], io), ('v', Vw[f], iv)):
                    x, y, key = R['occ'][oi]
                    text = R['text']
                    k0 = info['cover'][0]
                    if grp == 'copyable' and not R['ok'][k0]:
                        s = s_of(R['pred'][k0 - 1]).strip()
                        name = text[x:y]
                        others = set()
                        for pj in range(oi):
                            px, py, pk = R['occ'][pj]
                            pc = R['occ_tok'][pj]['cover']
                            if pk != key and pc:
                                others.add(text[R['es'][pc[0]]:R['offs'][pc[0]][1]])
                        if s and s in others:
                            c = 'other_in_context_variable'
                        elif s and name.startswith(s):
                            c = 'target_name_other_split'
                        elif IDR.match(s) and _kw.iskeyword(s):
                            c = 'keyword'
                        elif IDR.match(s):
                            c = ('identifier_in_left_context'
                                 if re.search(r'\b' + re.escape(s) + r'\b', text[:x])
                                 else 'identifier_not_in_context')
                        else:
                            c = 'non_identifier'
                        ent[tag][c] += 1
                    if has_exit and not R['ok'][info['exit']]:
                        s = s_of(R['pred'][info['exit'] - 1])
                        exi[tag]['continues_identifier' if s[:1] and (s[0].isalnum() or s[0] == '_')
                                 else 'other'] += 1
        g = {'n_occurrences': n_ent, 'n_exits': n_exit}
        if grp == 'copyable':
            g['entry_error_types'] = {t: {'n_errors': sum(ent[t].values()),
                                          'counts': dict(ent[t]),
                                          'share_of_errors': {k: v / max(1, sum(ent[t].values()))
                                                              for k, v in ent[t].items()},
                                          'share_of_occurrences': {k: v / max(1, n_ent)
                                                                   for k, v in ent[t].items()}}
                                      for t in ('o', 'v')}
        g['exit_error_types'] = {t: {'n_errors': sum(exi[t].values()), 'counts': dict(exi[t]),
                                     'continues_share_of_exits': exi[t]['continues_identifier']
                                     / max(1, n_exit)}
                                 for t in ('o', 'v')}
        out[grp] = g
    return out


def d1_view_totals(R, fn_ids):
    """Unpaired per-view accuracy by class (sanity / context)."""
    tot = defaultdict(lambda: [0, 0])
    for f in fn_ids:
        r = R[f]
        cp_tok = {}
        for oi, info in enumerate(r['occ_tok']):
            g = 'copyable' if r['copyable'][oi] else 'first'
            for k in info['cover']:
                cp_tok[k] = g
            if info['exit'] is not None:
                cp_tok[info['exit']] = g
        for k in range(1, len(r['ids'])):
            c = r['cls'][k]
            key = c if c in ('other', 'straddle') else '%s_%s' % (c, cp_tok.get(k, '?'))
            tot[key][0] += int(r['ok'][k])
            tot[key][1] += 1
    return {k: wilson(*v) for k, v in sorted(tot.items())}


VIEWS = ('identity', 'samelen', 'permute')


def real6_occs(a):
    """real6 functions, the three views (draw 'a') and the self-checked occurrence spans."""
    import whyscale as WS
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    with open(os.path.join(WS.CACHE, 'real6_corpus.json')) as f:
        cached = json.load(f)
    if cached['min_vars'] != 3 or cached['lmax'] != WS.REAL6_LMAX or len(cached['recs']) < a.n:
        raise SystemExit('cached real6 corpus does not match the design')
    recs = cached['recs'][:a.n]
    views, orig, nvars, audit = WS._real6_build(recs, a.seed)
    # recover which rec each kept function came from (to know its renamed names)
    src2names = {r['src']: r['names'] for r in recs}
    texts = {v: views[(v, 'a')] for v in VIEWS}
    assert all(x == y for x, y in zip(texts['identity'], orig))
    fn_all = list(range(len(orig)))
    if a.limit:
        fn_all = fn_all[:a.limit]
    occs, dropped = {}, Counter()
    for f in fn_all:
        names = src2names[orig[f]]
        per = {}
        for v in VIEWS:
            o = occurrences(orig[f], texts[v][f], names)
            if o is None:
                break
            per[v] = o
        if len(per) != len(VIEWS):
            dropped['ast_alignment'] += 1
            continue
        keys = [[k for _, _, k in per[v]] for v in VIEWS]
        if any(k != keys[0] for k in keys):
            dropped['occurrence_order'] += 1
            continue
        sg = [segments(texts[v][f], per[v]) for v in VIEWS]
        if any(s != sg[0] for s in sg):
            dropped['segments_differ'] += 1
            continue
        occs[f] = per
    fn_ids = sorted(occs)
    print('%d functions (%d kept after self-checks; dropped %s)'
          % (len(fn_all), len(fn_ids), dict(dropped)), flush=True)
    return orig, texts, occs, fn_ids, dropped, audit


def entry_split_later(R, fn_ids):
    """POST-HOC, audited ([AGENT AUDIT] decoder Stage 2, probe (b)); not a registered test.

    Copyable occurrences whose name is intact in all three views. Per view: piece count, entry right,
    later pieces right. For piece counts k = 2..5 (each view's OWN count) and separately for occurrences
    whose entry that view got RIGHT / WRONG: per-piece later accuracy samelen vs permute. The summary
    is the samelen-minus-permute gap averaged over k weighted by samelen later pieces.
    """
    recs = []
    for f in fn_ids:
        for oi, c in enumerate(R['identity'][f]['copyable']):
            if not c:
                continue
            infos = {v: R[v][f]['occ_tok'][oi] for v in VIEWS}
            if not all(i['name_ok'] for i in infos.values()):
                continue
            d = {}
            for v in VIEWS:
                cov, ok = infos[v]['cover'], R[v][f]['ok']
                d[v] = (len(cov), bool(ok[cov[0]]), [bool(ok[k]) for k in cov[1:]])
            recs.append(d)
    out = {'n_triple': len(recs),
           'definition': 'later-piece per-piece accuracy, samelen vs permute, matched on own piece '
                         'count 2..5, split by the same view\'s entry correctness; gap weighted by '
                         'samelen later pieces'}
    for split in ('right', 'wrong'):
        want = split == 'right'
        byk, num, den = {}, 0.0, 0
        for k in (2, 3, 4, 5):
            acc = {}
            for v in ('samelen', 'permute', 'identity'):
                lp = [x for d in recs if d[v][0] == k and d[v][1] == want for x in d[v][2]]
                acc[v] = [float(np.mean(lp)) if lp else None, len(lp)]
            byk[str(k)] = acc
            if acc['samelen'][0] is not None and acc['permute'][0] is not None:
                num += acc['samelen'][1] * (acc['samelen'][0] - acc['permute'][0])
                den += acc['samelen'][1]
        out[split] = {'by_pieces': byk,
                      'gap_samelen_minus_permute_weighted': num / den if den else None,
                      'weight_samelen_later_pieces': den}
    return out


def token_form_split(R, fn_ids):
    """POST-HOC (data-fix round 1): entry accuracy split by whether the EXACT entry token already
    appeared as the first piece of an earlier occurrence of the same variable.

    Copyable occurrences whose name is intact in all three views (the same triple set as
    entry_split_later), so the three views are scored on identical occurrences. This is the
    unconditional version of the audited "conversion after a correct look" split: it needs no
    attention, so it can be read for every arm.
    """
    out = {'definition': token_form_split.__doc__.strip(), 'views': {}}
    n_triple = 0
    agg = {v: np.zeros(4) for v in VIEWS}      # k_with, n_with, k_without, n_without
    for f in fn_ids:
        occ_id = R['identity'][f]['occ']
        for oi, c in enumerate(R['identity'][f]['copyable']):
            if not c:
                continue
            if not all(R[v][f]['occ_tok'][oi]['name_ok'] for v in VIEWS):
                continue
            n_triple += 1
            for v in VIEWS:
                rec = R[v][f]
                key = rec['occ'][oi][2]
                k0 = rec['occ_tok'][oi]['cover'][0]
                seen = any(rec['occ_tok'][pj]['cover']
                           and rec['ids'][rec['occ_tok'][pj]['cover'][0]] == rec['ids'][k0]
                           for pj in range(oi) if rec['occ'][pj][2] == key)
                ok = bool(rec['ok'][k0])
                agg[v][0 if seen else 2] += ok
                agg[v][1 if seen else 3] += 1
    out['n_triple'] = n_triple
    for v in VIEWS:
        w = agg[v]
        out['views'][v] = {'frac_exact_token_seen': _safe_div(w[1], w[1] + w[3]),
                           'entry_acc_with_token': wilson(w[0], w[1]),
                           'entry_acc_without_token': wilson(w[2], w[3]),
                           'gap_with_minus_without': _safe_div(w[0], w[1]) - _safe_div(w[2], w[3])}
    return out


def cmd_d1(a):
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    orig, texts, occs, fn_ids, dropped, audit = real6_occs(a)
    print('D1: kept %d' % len(fn_ids), flush=True)
    tok, model = load_lm(a.model, dev)
    R = {}
    for v in VIEWS:
        t0 = time.time()
        tx = [texts[v][f] for f in fn_ids]
        ids, offs, preds = lm_argmax(model, tok, tx, dev, a.batch, a.max_len)
        R[v] = {}
        for j, f in enumerate(fn_ids):
            occ = occs[f][v]
            seen, cop, nprev, rsame, last = set(), [], [], [], None
            for x, y, k in occ:
                cop.append(k in seen)
                nprev.append(max(1, len(seen)))
                rsame.append(int(last == k))
                seen.add(k)
                last = k
            rec = d1_view_record(texts[v][f], occ, cop, ids[j], offs[j], preds[j], None)
            rec['n_prev_vars'] = nprev
            rec['recent_same'] = rsame
            R[v][f] = rec
        print('  scored %s: %d texts in %.0fs' % (v, len(tx), time.time() - t0), flush=True)
    mkey = os.path.basename(os.path.normpath(a.model))
    res = {'model': a.model, 'n_functions_real6': len(orig), 'n_scored': len(fn_ids),
           'dropped_by_self_checks': dict(dropped), 'real6_audit': audit,
           'max_len': a.max_len, 'seed': a.seed, 'draw': 'a', 'provenance': provenance(),
           'truncated_functions': {v: int(sum(1 for f in fn_ids
                                              if R[v][f]['occ_tok'] and
                                              any(i['truncated'] for i in R[v][f]['occ_tok'])))
                                   for v in VIEWS},
           'occurrence_counts': {
               'copyable': int(sum(sum(R['identity'][f]['copyable']) for f in fn_ids)),
               'first': int(sum(len(R['identity'][f]['copyable']) - sum(R['identity'][f]['copyable'])
                                for f in fn_ids))}}
    for v in VIEWS:
        res[v] = {'per_class_unpaired': d1_view_totals(R[v], fn_ids)}
        if v != 'identity':
            res[v]['vs_identity'] = d1_pair(R['identity'], R[v], fn_ids)
            res[v]['diagnostics_posthoc'] = d1_diagnostics(R['identity'], R[v], fn_ids, tok)
    res['posthoc_audited_entry_split'] = entry_split_later(R, fn_ids)
    res['posthoc_token_form'] = token_form_split(R, fn_ids)
    # registered checks
    sl, pm = res['samelen']['vs_identity'], res['permute']['vs_identity']
    ce, cx = sl['copyable']['entry']['drop'], sl['copyable']['exit_paired_same_token']['drop']
    fx = sl['first']['exit_paired_same_token']['drop']
    pe = pm['copyable']['entry']['drop']
    res['registered'] = {
        'P-D1': 'copyable-entry drop >= 3x exit drop; permute <= 1/4 of samelen at copyable '
                'entries. FALSIFIER: exit drop >= entry drop',
        'copyable_entry_drop_samelen': ce, 'copyable_exit_drop_samelen': cx,
        'first_mention_exit_drop_samelen': fx,
        'other_drop_samelen': sl['other_paired']['drop'],
        'ratio_entry_over_copyable_exit': _safe_div(ce, cx),
        'permute_copyable_entry_drop': pe,
        'permute_over_samelen_entry': _safe_div(pe, ce),
        'cond1_entry_ge_3x_copyable_exit': bool(ce >= 3 * cx),
        'cond1_secondary_vs_first_mention_exit': bool(ce >= 3 * fx),
        'cond2_permute_le_quarter': bool(pe <= 0.25 * ce),
        'falsifier_exit_ge_entry': bool(cx >= ce),
    }
    rg = res['registered']
    rg['verdict_point'] = ('FAILED (falsifier)' if rg['falsifier_exit_ge_entry'] else
                           'HELD' if rg['cond1_entry_ge_3x_copyable_exit']
                           and rg['cond2_permute_le_quarter'] else 'FAILED')
    for v in ('samelen', 'permute'):
        g = res[v]['vs_identity']
        for grp in ('copyable', 'first'):
            e, x = g[grp]['entry'], g[grp]['exit_paired_same_token']
            print('  %-8s %-8s n=%5d  entry %.4f->%.4f (drop %+.4f %s)  interior drop %+.4f  '
                  'exit %.4f->%.4f (drop %+.4f, n=%d)'
                  % (v, grp, g[grp]['n_occurrences_paired'], e['orig']['rate'] or 0,
                     e['view']['rate'] or 0, e['drop'], e['drop_ci95_cluster'],
                     g[grp]['interior_per_piece']['drop'], x['orig']['rate'] or 0,
                     x['view']['rate'] or 0, x['drop'], x['orig']['n']), flush=True)
        print('  %-8s other drop %+.4f (n=%d)' % (v, g['other_paired']['drop'],
                                                   g['other_paired']['orig']['n']), flush=True)
    print('  REGISTERED:', json.dumps(rg), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['d1', mkey], res, a.out)
    print('wrote', a.out, ['d1', mkey])
    return 0


# ===========================================================================
# Stage 3 (b): FIND OR WRITE?  attention at copyable entries
# ===========================================================================
def _topk_summary(S, items_B, ok, fB, top):
    """S[v]: [N, L*H]; returns per-view split-B means for the selected heads, correct/wrong split,
    samelen/original and permute/original ratios with function-cluster CIs."""
    out = {}
    fidx = np.unique(fB)
    pos = {f: i for i, f in enumerate(fidx)}
    for v in VIEWS:
        sc = S[v][items_B][:, top].mean(1)
        okB = ok[v][items_B]
        rows = np.zeros((len(fidx), 4))
        for x, o, f in zip(sc, okB, fB):
            r = rows[pos[f]]
            if o:
                r[0] += x; r[1] += 1
            else:
                r[2] += x; r[3] += 1
        d = {'mean': float(sc.mean()), 'n': int(len(sc)),
             'correct': float(sc[okB].mean()) if okB.any() else None, 'n_correct': int(okB.sum()),
             'wrong': float(sc[~okB].mean()) if (~okB).any() else None, 'n_wrong': int((~okB).sum())}
        if okB.any() and (~okB).any():
            d['correct_minus_wrong'] = d['correct'] - d['wrong']
            d['correct_minus_wrong_ci95_cluster'] = cluster_boot(
                rows, lambda t: t[0] / t[1] - t[2] / t[3])
        out[v] = d
    base = S['identity'][items_B][:, top].mean(1)
    for v in ('samelen', 'permute'):
        sc = S[v][items_B][:, top].mean(1)
        rows = np.zeros((len(fidx), 2))
        for x, y, f in zip(sc, base, fB):
            rows[pos[f]] += (x, y)
        out[v + '_over_original'] = float(sc.sum() / base.sum())
        out[v + '_over_original_ci95_cluster'] = cluster_boot(rows, lambda t: t[0] / t[1])
    return out


def _pb_verdict(summ):
    r = summ['samelen_over_original']
    d = summ['samelen'].get('correct_minus_wrong')
    if r < 0.5:
        return 'FALSIFIED (ratio < 0.5)'
    if d is not None and abs(d) < 0.10:
        return 'HELD'
    return 'FAILED (ratio >= 0.5 but |correct - wrong| >= 0.10)'


def audited_followup(Ac, Amx, nC, nPrev, qp1, exTok, Aov, Aovn, ok, isB, sel, item_f):
    """POST-HOC (audited follow-up of [AGENT AUDIT] decoder attention (Stage 3b)); not registered.

    (a) top-5 samelen/original ratio (split B, heads chosen on original split A) under four masses:
        summed (registered), per-token (mass / tokens of the correct variable in context), per name
        piece (mass x earlier occurrences / tokens), enrichment over uniform (mass x (q+1) / tokens),
        max single token (largest weight on any one token of the correct variable).
    (b) three-way split of the entry drop with look_correct defined on summed mass (and per-token):
        conv  = P_o(look) * (acc_o|look - acc_v|look)
        nolook= (1 - P_o(look)) * (acc_o|other - acc_v|other)
        rate  = drop - conv - nolook = (P_o - P_v) * (acc_v|look - acc_v|other)
    (c) accuracy after a correct look (summed definition), split by whether the exact entry token
        already appeared as the first piece of an earlier occurrence of the same variable.
    """
    out = {'definition': audited_followup.__doc__.strip()}
    fB = item_f[isB]
    masses = {
        'summed': {v: Ac[v] for v in VIEWS},
        'per_token': {v: Ac[v] / nC[v][:, None] for v in VIEWS},
        'per_name_piece': {v: Ac[v] * nPrev[v][:, None] / nC[v][:, None] for v in VIEWS},
        'enrichment_over_uniform': {v: Ac[v] * qp1[v][:, None] / nC[v][:, None] for v in VIEWS},
        'max_single_token': {v: Amx[v] for v in VIEWS}}
    out['a_ratios_top5_splitB'] = {}
    for name, S in masses.items():
        t = _topk_summary(S, isB, ok, fB, sel)
        out['a_ratios_top5_splitB'][name] = {
            'samelen_over_original': t['samelen_over_original'],
            'ci95_cluster': t['samelen_over_original_ci95_cluster'],
            'permute_over_original': t['permute_over_original'],
            'means': {v: t[v]['mean'] for v in VIEWS},
            'samelen_correct_minus_wrong': t['samelen'].get('correct_minus_wrong')}
    idxB = np.where(isB)[0]

    def looks(v, per_token):
        c5 = Ac[v][isB][:, sel].mean(1)
        if per_token:
            c5 = c5 / nC[v][isB]
        om = []
        for i in idxB:
            if len(Aov[v][i]) == 0:
                om.append(0.0)
                continue
            m = Aov[v][i][:, sel].mean(1)
            if per_token:
                m = m / Aovn[v][i]
            om.append(m.max())
        return c5 > np.array(om)

    out['b_decomposition'] = {}
    for per_token in (False, True):
        lk = {v: looks(v, per_token) for v in VIEWS}
        okB = {v: ok[v][isB] for v in VIEWS}
        st = {}
        for v in VIEWS:
            st[v] = {'look_rate': float(lk[v].mean()), 'acc_look': float(okB[v][lk[v]].mean()),
                     'acc_other': float(okB[v][~lk[v]].mean()), 'acc': float(okB[v].mean())}
        dec = {'per_view': st}
        o = st['identity']
        for v in ('samelen', 'permute'):
            w = st[v]
            drop = o['acc'] - w['acc']
            conv = o['look_rate'] * (o['acc_look'] - w['acc_look'])
            nol = (1 - o['look_rate']) * (o['acc_other'] - w['acc_other'])
            rate = drop - conv - nol
            dec[v] = {'drop': drop, 'conversion_after_correct_look': conv, 'no_correct_look': nol,
                      'look_rate': rate,
                      'shares': {'conversion_after_correct_look': conv / drop,
                                 'no_correct_look': nol / drop, 'look_rate': rate / drop}}
        out['b_decomposition']['per_token_look' if per_token else 'summed_look'] = dec
        if not per_token:
            lk_summed = lk
    out['c_conversion_by_exact_token'] = {}
    for v in VIEWS:
        lk, okB, ex = lk_summed[v], ok[v][isB], exTok[v][isB]
        out['c_conversion_by_exact_token'][v] = {
            'frac_items_exact_token_seen': float(ex.mean()),
            'acc_look_correct_with_token': float(okB[lk & ex].mean()), 'n_with': int((lk & ex).sum()),
            'acc_look_correct_without_token': float(okB[lk & ~ex].mean()),
            'n_without': int((lk & ~ex).sum())}
    return out


def cmd_attn(a):
    """(b) Attention from the position that predicts a copyable entry piece to the earlier
    occurrences (all their pieces) of the CORRECT variable and of OTHER renamed variables."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    orig, texts, occs, fn_ids, dropped, audit = real6_occs(a)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation='eager',
                                                 dtype=torch.float32).to(dev).eval()
    L, H = model.config.n_layer, model.config.n_head
    TL, lay = {}, {}
    for v in VIEWS:
        enc = tok([texts[v][f] for f in fn_ids], truncation=True, max_length=a.max_len,
                  return_offsets_mapping=True, add_special_tokens=False)
        TL[v] = {f: (enc['input_ids'][j], [tuple(o) for o in enc['offset_mapping'][j]])
                 for j, f in enumerate(fn_ids)}
        lay[v] = {f: token_layout(texts[v][f], TL[v][f][1], occs[f][v]) for f in fn_ids}
    items = {}
    for f in fn_ids:
        seen, it = set(), []
        for oi, (_, _, k) in enumerate(occs[f]['identity']):
            if k in seen and all(lay[v][f][1][oi]['name_ok'] and lay[v][f][1][oi]['cover'][0] >= 1
                                 for v in VIEWS):
                it.append(oi)
            seen.add(k)
        items[f] = it
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(fn_ids))
    splitA = {fn_ids[i] for i in perm[:len(fn_ids) // 2]}
    order = [f for f in sorted(fn_ids, key=lambda f: len(TL['identity'][f][0]), reverse=True)
             if items[f]]
    item_f = np.array([f for f in order for _ in items[f]])
    Ac, Ao, Ap, ok, ubC, ubO, nC, Aov = {}, {}, {}, {}, {}, {}, {}, {}
    Amx, nPrev, qp1, exTok, Aovn = {}, {}, {}, {}, {}
    wl = {}
    for v in VIEWS:
        t0 = time.time()
        rc, ro, rp, rok, uc, uo, nc, rov = [], [], [], [], [], [], [], []
        rmx, rnp, rq, rex, rovn = [], [], [], [], []
        wlv = Counter()
        for s0 in range(0, len(order), a.batch):
            fb = order[s0:s0 + a.batch]
            T = max(len(TL[v][f][0]) for f in fb)
            inp = torch.full((len(fb), T), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(fb), T), dtype=torch.long)
            for j, f in enumerate(fb):
                ids = TL[v][f][0]
                inp[j, :len(ids)] = torch.tensor(ids)
                att[j, :len(ids)] = 1
            with torch.no_grad():
                out = model(input_ids=inp.to(dev), attention_mask=att.to(dev), output_attentions=True)
            A = torch.stack(out.attentions, 1).float().cpu()            # [B, L, H, T, T]
            pred = out.logits.argmax(-1).cpu().numpy()
            for j, f in enumerate(fb):
                ids, offs = TL[v][f]
                cls, occ_tok, es = lay[v][f]
                occ, text = occs[f][v], texts[v][f]
                for oi in items[f]:
                    q = occ_tok[oi]['cover'][0] - 1
                    key = occ[oi][2]
                    C = sorted({t for pj in range(oi) if occ[pj][2] == key
                                for t in occ_tok[pj]['cover'] if t <= q})
                    O = sorted({t for pj in range(oi) if occ[pj][2] != key
                                for t in occ_tok[pj]['cover'] if t <= q})
                    row = A[j, :, :, q, :]
                    rc.append(row[:, :, C].sum(-1).reshape(-1).numpy())
                    rmx.append(row[:, :, C].max(-1).values.reshape(-1).numpy())
                    prev_same = [pj for pj in range(oi) if occ[pj][2] == key]
                    rnp.append(len(prev_same))
                    rq.append(q + 1)
                    k0 = occ_tok[oi]['cover'][0]
                    rex.append(any(occ_tok[pj]['cover'] and ids[occ_tok[pj]['cover'][0]] == ids[k0]
                                   for pj in prev_same))
                    per_other = {}
                    for pj in range(oi):
                        if occ[pj][2] != key:
                            per_other.setdefault(occ[pj][2], set()).update(
                                t for t in occ_tok[pj]['cover'] if t <= q)
                    rov.append(np.array([row[:, :, sorted(ts)].sum(-1).reshape(-1).numpy()
                                         for ts in per_other.values() if ts]).reshape(-1, L * H))
                    rovn.append(np.array([len(ts) for ts in per_other.values() if ts], dtype=float))
                    ro.append(row[:, :, O].sum(-1).reshape(-1).numpy() if O else np.zeros(L * H))
                    good = int(pred[j, q]) == ids[q + 1]
                    rok.append(good)
                    uc.append(len(C) / (q + 1)); uo.append(len(O) / (q + 1)); nc.append(len(C))
                    # secondary: which in-context variable did a wrong prediction name?
                    pv = np.full(L * H, np.nan)
                    if not good:
                        ps = tok.decode([int(pred[j, q])]).strip()
                        last = {}
                        for pj in range(oi):
                            if occ[pj][2] != key and occ_tok[pj]['cover']:
                                last[occ[pj][2]] = pj
                        match = [k2 for k2, pj in last.items()
                                 if ps and text[es[occ_tok[pj]['cover'][0]]:
                                                offs[occ_tok[pj]['cover'][0]][1]] == ps]
                        if len(match) == 1:
                            P = sorted({t for pj in range(oi) if occ[pj][2] == match[0]
                                        for t in occ_tok[pj]['cover'] if t <= q})
                            pv = row[:, :, P].sum(-1).reshape(-1).numpy()
                            wlv['predicted_other_variable_identified'] += 1
                        else:
                            wlv['ambiguous' if len(match) > 1 else 'not_an_in_context_variable'] += 1
                    rp.append(pv)
        Ac[v], Ao[v], Ap[v] = np.array(rc), np.array(ro), np.array(rp)
        Aov[v] = rov
        Amx[v], nPrev[v], qp1[v] = np.array(rmx), np.array(rnp, dtype=float), np.array(rq, dtype=float)
        exTok[v], Aovn[v] = np.array(rex, dtype=bool), rovn
        ok[v] = np.array(rok, dtype=bool)
        ubC[v], ubO[v], nC[v] = np.array(uc), np.array(uo), np.array(nc)
        wl[v] = dict(wlv)
        print('  attn %s: %d items in %.0fs, entry acc %.4f' % (v, len(rok), time.time() - t0,
                                                                  ok[v].mean()), flush=True)
    isB = np.array([f not in splitA for f in item_f])
    isA = ~isB
    sel = np.argsort(-Ac['identity'][isA].mean(0))[:5]
    heads = [{'layer': int(h // H), 'head': int(h % H),
              'splitA_original': float(Ac['identity'][isA].mean(0)[h]),
              'splitB': {v: float(Ac[v][isB].mean(0)[h]) for v in VIEWS}} for h in sel]
    allh = np.arange(L * H)
    fB = item_f[isB]
    res = {'model': a.model, 'n_functions': len(fn_ids), 'n_items': int(len(item_f)),
           'n_items_splitA': int(isA.sum()), 'n_items_splitB': int(isB.sum()),
           'layers': L, 'heads_per_layer': H, 'max_len': a.max_len, 'provenance': provenance(),
           'split': 'functions, numpy default_rng(0) permutation, first half = A',
           'correctness_source': 'argmax of the eager-attention forward used here',
           'entry_accuracy': {v: float(ok[v].mean()) for v in VIEWS},
           'uniform_baseline_mass': {'correct_var': {v: float(ubC[v].mean()) for v in VIEWS},
                                     'other_vars': {v: float(ubO[v].mean()) for v in VIEWS}},
           'mean_correct_var_pieces_in_context': {v: float(nC[v].mean()) for v in VIEWS},
           'top5_heads_selected_on_original_splitA': heads}
    res['correct_var'] = {
        'top5_splitB': _topk_summary(Ac, isB, ok, fB, sel),
        'head_average_splitB': _topk_summary(Ac, isB, ok, fB, allh),
        'head_average_all_items': {v: float(Ac[v].mean()) for v in VIEWS},
        'per_layer_head_mean_all_items': {v: Ac[v].mean(0).reshape(L, H).round(5).tolist()
                                          for v in VIEWS}}
    res['other_vars'] = {
        'top5_splitB': _topk_summary(Ao, isB, ok, fB, sel),
        'head_average_splitB': _topk_summary(Ao, isB, ok, fB, allh),
        'head_average_all_items': {v: float(Ao[v].mean()) for v in VIEWS},
        'per_layer_head_mean_all_items': {v: Ao[v].mean(0).reshape(L, H).round(5).tolist()
                                          for v in VIEWS}}
    wlook = {}
    for v in VIEWS:
        m = isB & ~ok[v] & ~np.isnan(Ap[v][:, 0])
        wlook[v] = {'counts_all_wrong_items': wl[v], 'n_splitB': int(m.sum()),
                    'top5_to_correct_var': float(Ac[v][m][:, sel].mean()) if m.any() else None,
                    'top5_to_predicted_var': float(Ap[v][m][:, sel].mean()) if m.any() else None,
                    'frac_items_predicted_gt_correct': float(
                        (Ap[v][m][:, sel].mean(1) > Ac[v][m][:, sel].mean(1)).mean()) if m.any() else None}
    res['wrong_look_secondary'] = wlook
    # POST-HOC (declared in labnotes after the registered P-B numbers were seen): find vs write.
    # look_correct = top-5 mass on the correct variable exceeds the top-5 mass on EVERY single other
    # earlier variable. Then entry accuracy given each kind of look, and accuracy in bins of the
    # top-5 correct-variable mass (quintile edges from ORIGINAL, split B).
    idxB = np.where(isB)[0]
    c5o = Ac['identity'][isB][:, sel].mean(1)
    edges = np.quantile(c5o, [0.2, 0.4, 0.6, 0.8])
    fw = {'definition': 'look_correct: top-5 mass on the correct variable > top-5 mass on each other '
                        'earlier variable (items with no other variable count as look_correct if mass > 0)',
          'quintile_edges_from_original': edges.tolist()}
    for v in VIEWS:
        c5 = Ac[v][isB][:, sel].mean(1)
        om = np.array([Aov[v][i][:, sel].mean(1).max() if len(Aov[v][i]) else 0.0 for i in idxB])
        look = c5 > om
        okB = ok[v][isB]
        b = np.digitize(c5, edges)
        fw[v] = {'n': int(len(c5)), 'n_no_other_variable': int(sum(len(Aov[v][i]) == 0 for i in idxB)),
                 'look_correct_rate': float(look.mean()),
                 'acc_given_look_correct': float(okB[look].mean()), 'n_look_correct': int(look.sum()),
                 'acc_given_look_other': float(okB[~look].mean()), 'n_look_other': int((~look).sum()),
                 'acc_by_mass_quintile': [[float(okB[b == i].mean()) if (b == i).any() else None,
                                           int((b == i).sum())] for i in range(5)]}
    res['posthoc_find_vs_write'] = fw
    res['posthoc_audited_followup'] = audited_followup(Ac, Amx, nC, nPrev, qp1, exTok, Aov, Aovn,
                                                       ok, isB, sel, item_f)
    print('  audited follow-up:', json.dumps(res['posthoc_audited_followup'])[:3000], flush=True)
    print('  posthoc find/write:', json.dumps(fw), flush=True)
    res['P-B_decoder'] = {
        'rule': 'HELD iff samelen/original >= 0.5 and within samelen |correct - wrong| < 0.10; '
                'FALSIFIED if samelen/original < 0.5. Primary summary = top-5 heads (split B).',
        'primary_top5': {'samelen_over_original': res['correct_var']['top5_splitB']['samelen_over_original'],
                         'samelen_correct_minus_wrong':
                             res['correct_var']['top5_splitB']['samelen'].get('correct_minus_wrong'),
                         'verdict': _pb_verdict(res['correct_var']['top5_splitB'])},
        'secondary_head_average': {
            'samelen_over_original': res['correct_var']['head_average_splitB']['samelen_over_original'],
            'samelen_correct_minus_wrong':
                res['correct_var']['head_average_splitB']['samelen'].get('correct_minus_wrong'),
            'verdict': _pb_verdict(res['correct_var']['head_average_splitB'])}}
    for kind in ('correct_var', 'other_vars'):
        for sm in ('top5_splitB', 'head_average_splitB'):
            t = res[kind][sm]
            print('  %-11s %-19s orig %.4f samelen %.4f permute %.4f | sam/orig %.3f %s perm/orig %.3f'
                  ' | samelen correct %.4f (n=%d) wrong %.4f (n=%d)'
                  % (kind, sm, t['identity']['mean'], t['samelen']['mean'], t['permute']['mean'],
                     t['samelen_over_original'], t['samelen_over_original_ci95_cluster'],
                     t['permute_over_original'], t['samelen']['correct'] or 0,
                     t['samelen']['n_correct'], t['samelen']['wrong'] or 0, t['samelen']['n_wrong']),
                  flush=True)
    print('  top5 heads', [(h['layer'], h['head'], round(h['splitA_original'], 4)) for h in heads])
    print('  P-B decoder:', json.dumps(res['P-B_decoder']), flush=True)
    print('  wrong look:', json.dumps(wlook), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    mkey = os.path.basename(os.path.normpath(a.model))
    store(['stage3b_attention', mkey], res, a.out)
    print('wrote', a.out, ['stage3b_attention', mkey])
    return 0


def cmd_seeds(a):
    """Aggregate D1 over seeds: per-seed values, mean +- sd, and whether each verdict holds in >= 2/3."""
    with open(a.out) as f:
        doc = json.load(f)
    import whyscale as WS
    res = {}
    for arm in ('plain', 'highdiv'):
        rows = {}
        for sd in a.seeds:
            m = 'mech_dec_%s_s%d' % (arm, sd)
            if m not in doc.get('d1', {}):
                continue
            d = doc['d1'][m]
            rg = d['registered']
            es = d.get('posthoc_audited_entry_split', {})
            sl = d['samelen']['vs_identity']['copyable']
            row = {k: rg[k] for k in ('copyable_entry_drop_samelen', 'copyable_exit_drop_samelen',
                                      'first_mention_exit_drop_samelen', 'other_drop_samelen',
                                      'ratio_entry_over_copyable_exit', 'permute_copyable_entry_drop',
                                      'permute_over_samelen_entry', 'cond1_entry_ge_3x_copyable_exit',
                                      'cond2_permute_le_quarter', 'falsifier_exit_ge_entry',
                                      'verdict_point')}
            row['entry_orig'] = sl['entry']['orig']['rate']
            row['entry_samelen'] = sl['entry']['view']['rate']
            row['later_gap_entry_right'] = es.get('right', {}).get('gap_samelen_minus_permute_weighted')
            row['later_gap_entry_wrong'] = es.get('wrong', {}).get('gap_samelen_minus_permute_weighted')
            try:
                with open(os.path.join(WS.CACHE, 'whyscale_real6__%s.json' % m)) as f:
                    w = json.load(f)
                row['stage1_retrieval_samelen'] = w['views']['samelen']['paired_top1']
                row['stage1_retrieval_permute'] = w['views']['permute']['paired_top1']
            except (OSError, KeyError):
                row['stage1_retrieval_samelen'] = None
            rows['s%d' % sd] = row
        if not rows:
            continue
        summ = {}
        for k in next(iter(rows.values())):
            vals = [r[k] for r in rows.values()]
            if all(isinstance(x, bool) for x in vals):
                summ[k] = {'n_true': int(sum(vals)), 'n': len(vals),
                           'kept_true_in_ge_2_of_3': bool(sum(vals) >= 2)}
            elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals if x is not None) \
                    and any(x is not None for x in vals):
                xs = np.array([x for x in vals if x is not None], dtype=float)
                summ[k] = {'mean': float(xs.mean()), 'sd': float(xs.std(ddof=1)) if len(xs) > 1 else None,
                           'n': int(len(xs))}
            else:
                summ[k] = Counter(vals)
        v = [r['verdict_point'] for r in rows.values()]
        summ['P-D1_FAILED_kept'] = sum(x.startswith('FAILED') for x in v) >= 2
        summ['P-D1_HELD_kept'] = sum(x == 'HELD' for x in v) >= 2
        res[arm] = {'per_seed': rows, 'summary': summ}
        print(arm, json.dumps(res[arm], default=str, indent=0)[:4000], flush=True)
    if 'plain' in res and 'highdiv' in res:
        # Same seed does NOT reproduce the same model (labnotes, [LEAD] seed-3 replicas land), so plain
        # and highdiv sharing a seed are not matched draws: UNPAIRED difference of means, with each arm's
        # run-to-run sd. Per-seed differences are listed for reference only and are NOT paired.
        con = {'note': 'UNPAIRED: mean(highdiv) - mean(plain) over independent runs; per-seed '
                       'differences listed for reference only, not a paired test'}
        P, Hd = res['plain']['per_seed'], res['highdiv']['per_seed']
        for k in ('entry_orig', 'entry_samelen', 'copyable_entry_drop_samelen',
                  'copyable_exit_drop_samelen', 'first_mention_exit_drop_samelen',
                  'permute_copyable_entry_drop', 'stage1_retrieval_samelen'):
            xp = np.array([r[k] for r in P.values() if r.get(k) is not None], dtype=float)
            xh = np.array([r[k] for r in Hd.values() if r.get(k) is not None], dtype=float)
            if len(xp) and len(xh):
                con[k] = {'highdiv_minus_plain': float(xh.mean() - xp.mean()),
                          'sd_plain': float(xp.std(ddof=1)) if len(xp) > 1 else None,
                          'sd_highdiv': float(xh.std(ddof=1)) if len(xh) > 1 else None,
                          'n_plain': int(len(xp)), 'n_highdiv': int(len(xh)),
                          'per_seed_differences_not_paired': {
                              s: Hd[s][k] - P[s][k] for s in P if s in Hd
                              and P[s].get(k) is not None and Hd[s].get(k) is not None}}
        res['highdiv_minus_plain_unpaired'] = con
        print('contrast', json.dumps(con, indent=0)[:3000], flush=True)
    store(['d1_seeds'], res, a.out)
    print('wrote', a.out, ['d1_seeds'])
    return 0


def cmd_fuseeds(a):
    """Aggregate the audited Stage-3b follow-up over runs s2/s3/s4 with the pre-declared readings
    (labnotes "[AGENT DEC] POST-HOC (audited follow-up)"); a reading is KEPT iff it holds in >= 2/3."""
    with open(a.out) as f:
        doc = json.load(f)['stage3b_attention']
    res = {}
    for arm in ('plain', 'highdiv'):
        runs = {}
        for sd in a.seeds:
            m = 'mech_dec_%s_s%d' % (arm, sd)
            if m not in doc or 'posthoc_audited_followup' not in doc[m]:
                continue
            fu = doc[m]['posthoc_audited_followup']
            r = {}
            for k, v in fu['a_ratios_top5_splitB'].items():
                r['ratio_' + k] = v['samelen_over_original']
            for dk, dv in fu['b_decomposition'].items():
                for part, x in dv['samelen']['shares'].items():
                    r['%s_share_%s' % (dk, part)] = x
                r[dk + '_drop'] = dv['samelen']['drop']
                for v in VIEWS:
                    for q in ('look_rate', 'acc_look', 'acc_other'):
                        r['%s_%s_%s' % (dk, v, q)] = dv['per_view'][v][q]
            for v in VIEWS:
                c = fu['c_conversion_by_exact_token'][v]
                r['c_%s_with' % v] = c['acc_look_correct_with_token']
                r['c_%s_without' % v] = c['acc_look_correct_without_token']
                r['c_%s_frac_seen' % v] = c['frac_items_exact_token_seen']
            pb = doc[m]['P-B_decoder']['primary_top5']
            r['registered_PB_verdict'] = pb['verdict']
            r['registered_samelen_correct_minus_wrong'] = pb['samelen_correct_minus_wrong']
            rd = {
                'R-a1 summed ratio >= 0.5': r['ratio_summed'] >= 0.5,
                'R-a2 per-token ratio >= 0.5': r['ratio_per_token'] >= 0.5,
                'R-a3 per-name-piece ratio >= 0.5': r['ratio_per_name_piece'] >= 0.5,
                'R-a4 enrichment ratio >= 0.5': r['ratio_enrichment_over_uniform'] >= 0.5,
                'R-a5 max-token ratio >= 0.5': r['ratio_max_single_token'] >= 0.5,
                'R-a6 max-token ratio >= 0.9': r['ratio_max_single_token'] >= 0.9,
                'R-b1 conversion share 0.35-0.65':
                    0.35 <= r['summed_look_share_conversion_after_correct_look'] <= 0.65,
                'R-b2 no-look share >= 0.25': r['summed_look_share_no_correct_look'] >= 0.25,
                'R-b3 look-rate share < 0.20': r['summed_look_share_look_rate'] < 0.20,
                'R-c1 original |with - without| < 0.05':
                    abs(r['c_identity_with'] - r['c_identity_without']) < 0.05,
                'R-c2 samelen without lower by >= 0.10': r['c_samelen_with'] - r['c_samelen_without'] >= 0.10,
                'R-c3 samelen-with below original-with by >= 0.05':
                    r['c_identity_with'] - r['c_samelen_with'] >= 0.05}
            r['readings'] = rd
            runs['s%d' % sd] = r
        if not runs:
            continue
        summ = {}
        for k in next(iter(runs.values())):
            if k == 'readings' or isinstance(next(iter(runs.values()))[k], str):
                continue
            xs = np.array([r[k] for r in runs.values()], dtype=float)
            summ[k] = {'mean': float(xs.mean()), 'sd': float(xs.std(ddof=1)) if len(xs) > 1 else None}
        keep = {}
        for rk in next(iter(runs.values()))['readings']:
            n = sum(bool(r['readings'][rk]) for r in runs.values())
            keep[rk] = {'holds_in': n, 'of': len(runs), 'KEPT': n >= 2}
        res[arm] = {'per_run': runs, 'mean_sd': summ, 'readings': keep,
                    'registered_PB_verdicts': [r['registered_PB_verdict'] for r in runs.values()]}
        print(arm, json.dumps(keep, indent=0), flush=True)
    store(['stage3b_followup_seeds'], res, a.out)
    print('wrote', a.out, ['stage3b_followup_seeds'])
    return 0


# ===========================================================================
# Evaluation item (1): CODE COMPLETION under renaming
# ===========================================================================
def all_ident_spans(text):
    """Character spans of every ast Name / arg / ExceptHandler-name occurrence (renamed or not)."""
    tree = ast.parse(text)
    st = _line_starts(text)
    out = []
    for kind, n, nm in _sites(tree):
        if kind == 'exc':
            x0 = _off(text, st, n.lineno, n.col_offset)
            m = re.compile(r'\bas\s+(' + re.escape(nm) + r')\b').search(text, x0)
            if m is None:
                continue
            out.append((m.start(1), m.end(1)))
        else:
            x = _off(text, st, n.lineno, n.col_offset)
            if text[x:x + len(nm)] == nm:
                out.append((x, x + len(nm)))
    return sorted(set(out))


def comp_view_record(text, occ, ids, offs, pred):
    """Per-token correctness, identifier flag and the paired key for one view."""
    ok = np.zeros(len(ids), dtype=bool)
    ok[1:] = np.asarray(pred[:-1]) == np.asarray(ids[1:])
    spans = all_ident_spans(text)
    segstart = [0] + [y for _, y, _ in occ]
    segend = [x for x, _, _ in occ] + [len(text)]
    isid, key = np.zeros(len(ids), dtype=bool), [None] * len(ids)
    for k in range(1, len(ids)):
        a, b = offs[k]
        if b <= a:
            continue
        s = text[a:b]
        es = a + (len(s) - len(s.lstrip())) if s.strip() else b
        isid[k] = any(not (b <= x or es >= y) for x, y in spans)
        for j in range(len(segstart)):
            if segstart[j] <= a and b <= segend[j]:
                key[k] = (j, a - segstart[j], int(ids[k]), b - a)
                break
    return {'ok': ok, 'isid': isid, 'key': key, 'ids': ids}


def _truncation_point(text, frac=0.60):
    """-> (prompt, reference line content, line index) using the registered rule."""
    lines = text.split('\n')
    starts, p = [], 0
    for ln in lines:
        starts.append(p)
        p += len(ln) + 1
    cut = frac * len(text)
    cand = [i for i, ln in enumerate(lines) if starts[i] >= cut and ln.strip()]
    if not cand:
        cand = [i for i, ln in enumerate(lines) if ln.strip()]
        if not cand:
            return None
        cand = [cand[-1]]
    i = cand[0]
    ln = lines[i]
    lead = len(ln) - len(ln.lstrip())
    return text[:starts[i] + lead], ln[lead:].rstrip(), i


@torch.no_grad()
def greedy_lines(model, tok, prompts, device, batch, max_new=32):
    """Greedy continuation of each prompt, cut at the first newline. Left padding."""
    outs = [''] * len(prompts)
    enc = [tok(p, add_special_tokens=False)['input_ids'][-448:] for p in prompts]
    order = [i for i in sorted(range(len(prompts)), key=lambda i: len(enc[i]), reverse=True)
             if len(enc[i]) >= 1]
    pad = tok.pad_token_id
    for s0 in range(0, len(order), batch):
        idx = order[s0:s0 + batch]
        T = max(len(enc[i]) for i in idx)
        inp = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            inp[j, T - len(enc[i]):] = torch.tensor(enc[i])
            att[j, T - len(enc[i]):] = 1
        gen = model.generate(input_ids=inp.to(device), attention_mask=att.to(device),
                             max_new_tokens=max_new, do_sample=False, num_beams=1,
                             pad_token_id=pad)
        for j, i in enumerate(idx):
            txt = tok.decode(gen[j, T:], skip_special_tokens=True)
            outs[i] = txt.split('\n')[0].strip()
    return outs


def edit_sim(a, b):
    """1 - levenshtein / max(len) (CodeXGLUE line-completion style), on short strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def cmd_comp(a):
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    orig, texts, occs, fn_ids, dropped, audit = real6_occs(a)
    tok, model = load_lm(a.model, dev)
    mkey = os.path.basename(os.path.normpath(a.model))
    res = {'model': a.model, 'n_functions': len(fn_ids), 'max_len': a.max_len,
           'dropped_by_self_checks': dict(dropped), 'provenance': provenance(),
           'definition': 'teacher-forced next-token accuracy on real6; identifier position = token '
                         'overlapping any ast Name/arg/ExceptHandler span; paired non-identifier '
                         'positions are aligned across views (same segment offset, same token id)'}
    R = {}
    for v in VIEWS:
        t0 = time.time()
        ids, offs, preds = lm_argmax(model, tok, [texts[v][f] for f in fn_ids], dev, a.batch, a.max_len)
        R[v] = {}
        for j, f in enumerate(fn_ids):
            rec = comp_view_record(texts[v][f], occs[f][v], ids[j], offs[j], preds[j])
            rec['occ_tok'] = token_layout(texts[v][f], offs[j], occs[f][v])[1]
            R[v][f] = rec
        tot = {'all': [0, 0], 'ident': [0, 0], 'nonident': [0, 0]}
        for f in fn_ids:
            r = R[v][f]
            for k in range(1, len(r['ids'])):
                tot['all'][0] += int(r['ok'][k]); tot['all'][1] += 1
                g = 'ident' if r['isid'][k] else 'nonident'
                tot[g][0] += int(r['ok'][k]); tot[g][1] += 1
        res[v] = {'unpaired': {g: wilson(*w) for g, w in tot.items()}}
        print('  tf %-9s all %.4f ident %.4f NONIDENT %.4f  (%.0fs)'
              % (v, tot['all'][0] / tot['all'][1], tot['ident'][0] / max(1, tot['ident'][1]),
                 tot['nonident'][0] / max(1, tot['nonident'][1]), time.time() - t0), flush=True)
    # paired positions + retention
    for v in ('samelen', 'permute'):
        rows_n, rows_a = [], []
        for f in fn_ids:
            o, w = R['identity'][f], R[v][f]
            wmap = {k: i for i, k in enumerate(w['key']) if k is not None}
            rn = np.zeros(6)      # ok_o, ok_v, n, both, ok_o_all, ok_v_all  (n for nonident)
            ra = np.zeros(4)      # ok_o, ok_v, n, both (all paired)
            for i, k in enumerate(o['key']):
                if k is None or i == 0:
                    continue
                j = wmap.get(k)
                if j is None:
                    continue
                oo, ov = bool(o['ok'][i]), bool(w['ok'][j])
                ra[0] += oo; ra[1] += ov; ra[2] += 1; ra[3] += oo and ov
                if not o['isid'][i] and not w['isid'][j]:
                    rn[0] += oo; rn[1] += ov; rn[2] += 1; rn[3] += oo and ov
            rows_n.append(rn); rows_a.append(ra)
        Sn, Sa = np.array(rows_n).sum(0), np.array(rows_a).sum(0)
        res[v]['paired_nonident'] = {
            'orig': wilson(Sn[0], Sn[2]), 'view': wilson(Sn[1], Sn[2]),
            'drop': _safe_div(Sn[0] - Sn[1], Sn[2]),
            'drop_ci95_cluster': cluster_boot(np.array(rows_n)[:, [0, 1, 2]],
                                              lambda t: (t[0] - t[1]) / t[2]),
            'retention': wilson(Sn[3], Sn[0]),
            'retention_ci95_cluster': cluster_boot(np.array(rows_n)[:, [3, 0]],
                                                   lambda t: t[0] / t[1])}
        res[v]['paired_all_positions'] = {
            'orig': wilson(Sa[0], Sa[2]), 'view': wilson(Sa[1], Sa[2]),
            'drop': _safe_div(Sa[0] - Sa[1], Sa[2]), 'retention': wilson(Sa[3], Sa[0])}
        # occurrence-level retention of the entry piece (copyable occurrences intact in both views)
        k_o = k_both = k_n = 0
        for f in fn_ids:
            seen = set()
            for oi, (_, _, key) in enumerate(occs[f]['identity']):
                cop = key in seen
                seen.add(key)
                io, iv = R['identity'][f]['occ_tok'][oi], R[v][f]['occ_tok'][oi]
                if not (cop and io['name_ok'] and iv['name_ok']):
                    continue
                oo = bool(R['identity'][f]['ok'][io['cover'][0]])
                ov = bool(R[v][f]['ok'][iv['cover'][0]])
                k_n += 1
                k_o += oo
                k_both += oo and ov
        res[v]['identifier_occurrence_entry'] = {
            'n_occurrences': k_n, 'orig_correct': k_o,
            'retention': wilson(k_both, k_o),
            'note': 'occurrence level: the target token itself changes under renaming'}
    # generative
    if not a.no_gen:
        N = len(fn_ids)
        m = min(a.gen_n, N)
        sub = [fn_ids[int(round(i * (N - 1) / (m - 1)))] for i in range(m)]
        prompts, refs, has_var = {}, {}, {}
        for v in VIEWS:
            ps, rs, hv = [], [], []
            for f in sub:
                cut = _truncation_point(texts[v][f])
                ps.append(cut[0]); rs.append(cut[1])
                lo, hi = len(cut[0]), len(cut[0]) + len(cut[1])
                hv.append(any(lo <= x and y <= hi for x, y, _ in occs[f][v]))
            prompts[v], refs[v], has_var[v] = ps, rs, hv
        gen, match, esim = {}, {}, {}
        for v in VIEWS:
            t0 = time.time()
            gen[v] = greedy_lines(model, tok, prompts[v], dev, a.gen_batch)
            match[v] = np.array([g == r.strip() for g, r in zip(gen[v], refs[v])])
            esim[v] = np.array([edit_sim(g, r.strip()) for g, r in zip(gen[v], refs[v])])
            print('  gen %-9s exact-line %.4f  edit-sim %.4f (n=%d, %.0fs)'
                  % (v, match[v].mean(), esim[v].mean(), len(sub), time.time() - t0), flush=True)
        g = {'n_functions': len(sub), 'max_new_tokens': 32, 'decoding': 'greedy, cut at first newline',
             'truncation_rule': 'first non-empty line starting at or after 60% of the characters; '
                                'prompt includes that line indentation; the generated line is '
                                'stripped before comparison, because a space-prefixed first BPE '
                                'token is a tokenizer artefact, not an error',
             'sample': 'even stride over the 1962 real6 functions'}
        for v in VIEWS:
            g[v] = {'exact_line': wilson(int(match[v].sum()), len(sub)),
                    'edit_similarity': float(esim[v].mean())}
        for v in ('samelen', 'permute'):
            both = int((match['identity'] & match[v]).sum())
            g[v]['drop'] = float(match['identity'].mean() - match[v].mean())
            g[v]['edit_similarity_drop'] = float(esim['identity'].mean() - esim[v].mean())
            g[v]['retention'] = wilson(both, int(match['identity'].sum()))
            hv = np.array(has_var['identity'])
            g[v]['by_line_contains_renamed_var'] = {
                'with': {'orig': float(match['identity'][hv].mean()) if hv.any() else None,
                         'view': float(match[v][hv].mean()) if hv.any() else None, 'n': int(hv.sum())},
                'without': {'orig': float(match['identity'][~hv].mean()) if (~hv).any() else None,
                            'view': float(match[v][~hv].mean()) if (~hv).any() else None,
                            'n': int((~hv).sum())}}
        g['examples'] = [{'prompt_tail': prompts['identity'][i][-60:], 'ref': refs['identity'][i],
                          'gen_orig': gen['identity'][i], 'gen_samelen': gen['samelen'][i],
                          'ref_samelen': refs['samelen'][i]} for i in range(3)]
        res['generative'] = g
    print('  RETENTION nonident samelen %.4f permute %.4f'
          % (res['samelen']['paired_nonident']['retention']['rate'],
             res['permute']['paired_nonident']['retention']['rate']), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['completion', mkey], res, a.out)
    print('wrote', a.out, ['completion', mkey])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p0 = sub.add_parser('d0')
    p0.add_argument('--ckpt', default=os.path.join(WORK, 'ckpt_p0.1sb', 'checkpoint_1.00.pth'))
    p0.add_argument('--data', default=os.path.join(WORK, 'data_p0.1sb'))
    p0.add_argument('--n', type=int, default=2048)
    p0.add_argument('--batch', type=int, default=32)
    p1 = sub.add_parser('d1')
    p1.add_argument('--model', required=True)
    p1.add_argument('--n', type=int, default=2000)
    p1.add_argument('--seed', type=int, default=564)
    p1.add_argument('--max-len', type=int, default=320)
    p1.add_argument('--batch', type=int, default=16)
    p2 = sub.add_parser('attn')
    p2.add_argument('--model', required=True)
    p2.add_argument('--n', type=int, default=2000)
    p2.add_argument('--seed', type=int, default=564)
    p2.add_argument('--max-len', type=int, default=320)
    p2.add_argument('--batch', type=int, default=8)
    p3 = sub.add_parser('seeds')
    p3.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4])
    pg = sub.add_parser('vmloc')
    pg.add_argument('--model', required=True)
    pg.add_argument('--n', type=int, default=2000, help='real6 sources to scan')
    pg.add_argument('--n-items', type=int, default=400)
    pg.add_argument('--max-len', type=int, default=320)
    pg.add_argument('--batch', type=int, default=64)
    pf = sub.add_parser('codenet')
    pf.add_argument('--model', required=True)
    pf.add_argument('--n-problems', type=int, default=100)
    pf.add_argument('--per-problem', type=int, default=5)
    pf.add_argument('--batch', type=int, default=16)
    pe = sub.add_parser('cloze')
    pe.add_argument('--model', required=True)
    pe.add_argument('--variant', default='maxmin', choices=('maxmin', 'all'))
    pe.add_argument('--n-sample', type=int, default=3000)
    pe.add_argument('--max-len', type=int, default=320)
    pe.add_argument('--batch', type=int, default=16)
    pe.add_argument('--per-item', action='store_true',
                    help='same run, plus per-item correctness at every N and a report stratified '
                         'by gold token kind; writes cloze_{variant}_bykind instead of cloze_'
                         '{variant}')
    pd = sub.add_parser('py150')
    pd.add_argument('--model', required=True)
    pd.add_argument('--n-files', type=int, default=500)
    pd.add_argument('--line-n', type=int, default=300)
    pd.add_argument('--max-len', type=int, default=320)
    pd.add_argument('--batch', type=int, default=16)
    pd.add_argument('--gen-batch', type=int, default=24)
    pd.add_argument('--warmup', type=int, default=8)
    pd.add_argument('--no-line', action='store_true')
    pd.add_argument('--examples', action='store_true', help='store the manual rename-rule check')
    pc = sub.add_parser('idcurve')
    pc.add_argument('--model', required=True)
    pc.add_argument('--n', type=int, default=2000)
    pc.add_argument('--max-len', type=int, default=320)
    pc.add_argument('--batch', type=int, default=16)
    pb = sub.add_parser('sel')
    pb.add_argument('--model', required=True)
    pb.add_argument('--task', default='all', choices=('all',) + SEL_TASKS)
    pb.add_argument('--n-items', type=int, default=300)
    pb.add_argument('--batch', type=int, default=16)
    pb.add_argument('--curve', action='store_true', help='also score N = 1,2,4,8')
    pa = sub.add_parser('probecurve')
    pa.add_argument('--model', required=True)
    pa.add_argument('--tasks', default='useassign,binop,operands')
    pa.add_argument('--batch', type=int, default=16)
    p9 = sub.add_parser('probe')
    p9.add_argument('--model', required=True)
    p9.add_argument('--task', default='all', choices=('all',) + PROBE_TASKS)
    p9.add_argument('--n-items', type=int, default=300)
    p9.add_argument('--batch', type=int, default=16)
    p9.add_argument('--nextline-unseen', action='store_true',
                    help='draw next-line distractors only from lines AFTER the cut, so no candidate '
                         'has already appeared in the prompt')
    p8 = sub.add_parser('docmatch')
    p8.add_argument('--model', required=True)
    p8.add_argument('--n-items', type=int, default=800)
    p8.add_argument('--batch', type=int, default=16)
    p8.add_argument('--fmt', default='docstring_slot',
                    choices=('docstring_slot', 'trailing_comment'))
    p8.add_argument('--pick-format', action='store_true',
                    help='score BOTH formats on the dev slice (never part of the evaluation set)')
    p7 = sub.add_parser('curves')
    p7.add_argument('--model', required=True)
    p7.add_argument('--n', type=int, default=2000)
    p7.add_argument('--max-len', type=int, default=320)
    p7.add_argument('--batch', type=int, default=16)
    p6 = sub.add_parser('bench')
    p6.add_argument('--model', required=True)
    p6.add_argument('--task', default='all', choices=('all', 'py150', 'mbpp', 'vmmbpp'))
    p6.add_argument('--n-py150', type=int, default=500)
    p6.add_argument('--n-mbpp', type=int, default=500)
    p6.add_argument('--max-len', type=int, default=320)
    p6.add_argument('--batch', type=int, default=16)
    p6.add_argument('--gen-batch', type=int, default=24)
    p5 = sub.add_parser('comp')
    p5.add_argument('--model', required=True)
    p5.add_argument('--n', type=int, default=2000)
    p5.add_argument('--seed', type=int, default=564)
    p5.add_argument('--max-len', type=int, default=320)
    p5.add_argument('--batch', type=int, default=16)
    p5.add_argument('--gen-n', type=int, default=300)
    p5.add_argument('--gen-batch', type=int, default=24)
    p5.add_argument('--no-gen', action='store_true')
    p4 = sub.add_parser('fuseeds')
    p4.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4])
    for p in (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, pa, pb, pc, pd, pe, pf, pg):
        p.add_argument('--device', default=None)
        p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
        p.add_argument('--limit', type=int, default=0)
        p.add_argument('--write-smoke', action='store_true')
        p.add_argument('--out', default=OUT)
    a = ap.parse_args()
    return {'d0': cmd_d0, 'd1': cmd_d1, 'attn': cmd_attn, 'seeds': cmd_seeds,
            'fuseeds': cmd_fuseeds, 'comp': cmd_comp, 'bench': cmd_bench, 'curves': cmd_curves, 'docmatch': cmd_docmatch, 'probe': cmd_probe, 'probecurve': cmd_probecurve, 'sel': cmd_sel, 'idcurve': cmd_idcurve, 'py150': cmd_py150, 'cloze': cmd_cloze, 'codenet': cmd_codenet, 'vmloc': cmd_vmloc}[a.cmd](a)




# ===========================================================================
# PUBLISHED-BENCHMARK SUITE (three tasks: PY150 token completion, MBPP line
# completion, CuBERT-style variable misuse on MBPP)
# ===========================================================================
BENCH_NS = (1, 4, 8)
BENCH_SEED = 564


def _kwlike(t):
    import builtins
    import keyword
    return keyword.iskeyword(t) or keyword.issoftkeyword(t) or hasattr(builtins, t)


def py150_renameable(toks):
    """Token-level decision: which tokens of a CodeXGLUE PY150 file may be renamed.

    A token is renameable iff it matches ^[A-Za-z_]\\w*$ and it is NOT
      - a Python keyword / soft keyword / builtin;
      - in attribute position (previous token '.');
      - the name in a def/class header (previous token 'def' or 'class');
      - anywhere inside an import statement (between an 'import'/'from' and the next <EOL>);
      - in call/definition position (next token '(');
      - a keyword-argument name (previous token '(' or ',' and next token '=').
    Everything left is treated as a variable occurrence. The heuristics for kwargs and calls are
    conservative: they drop some real variables rather than rename something that is not one.
    """
    ident = re.compile(r'^[A-Za-z_]\w*$')
    out, in_import = [], False
    for i, t in enumerate(toks):
        if t == '<EOL>':
            in_import = False
            out.append(False)
            continue
        if t in ('import', 'from'):
            in_import = True
        prev = toks[i - 1] if i else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        ok = bool(ident.match(t)) and not _kwlike(t) and prev != '.' \
            and prev not in ('def', 'class') and not in_import and nxt != '(' \
            and not (nxt == '=' and prev in ('(', ','))
        out.append(ok)
    return out


def py150_render(toks):
    """-> (text, spans) where spans[j] is the character span of code token j ('' for structure)."""
    parts, spans, pos, line_start = [], [], 0, True
    for t in toks:
        if t in ('<s>', '</s>'):
            spans.append(None)
            continue
        if t == '<EOL>':
            parts.append('\n')
            pos += 1
            line_start = True
            spans.append(None)
            continue
        if not line_start:
            parts.append(' ')
            pos += 1
        parts.append(t)
        spans.append((pos, pos + len(t)))
        pos += len(t)
        line_start = False
    return ''.join(parts), spans


def py150_corpus(n=500, seed=BENCH_SEED, max_tokens=900, ns=BENCH_NS):
    """n files by even stride over the 50,000-file test split, with views N = 0,1,4,8."""
    from datasets import load_dataset
    ds = load_dataset('google/code_x_glue_cc_code_completion_token', 'python', split='test')
    idx = [int(round(i * (len(ds) - 1) / (n - 1))) for i in range(n)]
    out, meta = [], {'n_requested': n, 'split': 'test', 'stride_sample': True,
                     'n_dataset': len(ds), 'max_code_tokens': max_tokens, 'renamed': {}}
    ren_counts = {N: [] for N in ns}
    for j, i in enumerate(idx):
        toks = list(ds[i]['code'])[:max_tokens]
        ok = py150_renameable(toks)
        names = sorted({t for t, o in zip(toks, ok) if o})
        views = {0: toks}
        taken = set(toks)
        for N in ns:
            rng = random.Random(hash_seed('py150', seed, j, N))
            picked = list(names)
            rng.shuffle(picked)
            picked = picked[:N]
            mp = {}
            for v in picked:
                nn, g = _rand_letters(rng, len(v)), 0
                while (nn in taken or nn in mp.values()) and g < 256:
                    nn = _rand_letters(rng, len(v))
                    g += 1
                mp[v] = nn
            views[N] = [mp.get(t, t) if o else t for t, o in zip(toks, ok)]
            ren_counts[N].append(len(mp))
        out.append({'i': i, 'toks': views, 'n_names': len(names),
                    'n_renameable_tokens': int(sum(ok)), 'n_tokens': len(toks)})
    meta['renamed'] = {str(N): float(np.mean(ren_counts[N])) for N in ns}
    meta['mean_renameable_TOKENS_per_file'] = float(np.mean([r['n_renameable_tokens'] for r in out]))
    meta['frac_tokens_renameable'] = float(np.mean([r['n_renameable_tokens'] / max(1, r['n_tokens'])
                                                    for r in out]))
    meta['mean_code_tokens'] = float(np.mean([len(r['toks'][0]) for r in out]))
    meta['mean_renameable_names'] = float(np.mean([r['n_names'] for r in out]))
    return out, meta


def hash_seed(*p):
    import hashlib
    return int(hashlib.blake2b('|'.join(str(x) for x in p).encode(), digest_size=6).hexdigest(), 16)


def _rand_letters(rng, k):
    return ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(max(1, k)))


def bench_py150(model, tok, dev, recs, meta, batch, max_len, warmup=8):
    """Teacher-forced next-token accuracy on PY150, per view, plus code-token retention."""
    views = (0,) + BENCH_NS
    texts, spans = {}, {}
    for N in views:
        texts[N] = [py150_render(r['toks'][N])[0] for r in recs]
        spans[N] = [py150_render(r['toks'][N])[1] for r in recs]
    keep = [j for j in range(len(recs))
            if min(len(tok(texts[N][j], add_special_tokens=False)['input_ids']) for N in views) >= 2]
    if len(keep) < len(recs):
        meta = dict(meta); meta['dropped_too_short'] = len(recs) - len(keep)
        recs = [recs[j] for j in keep]
        texts = {N: [texts[N][j] for j in keep] for N in views}
        spans = {N: [spans[N][j] for j in keep] for N in views}
    res = {'meta': meta, 'warmup_code_tokens': warmup, 'views': {}}
    per_tok = {}
    for N in views:
        ids, offs, preds = lm_argmax(model, tok, texts[N], dev, batch, max_len)
        tot = {'all': [0, 0], 'ident': [0, 0], 'nonident': [0, 0]}
        maj, cmaj = Counter(), Counter()
        firstpiece = {}
        for j, r in enumerate(recs):
            sp, tk, off, pr = spans[N][j], texts[N][j], offs[j], preds[j]
            idmap = np.zeros(len(ids[j]), dtype=bool)
            start_of = {}
            ren = {}
            okflag = py150_renameable(r['toks'][0])
            IDRE = re.compile(r'^[A-Za-z_]\w*$')
            isid_tok = [bool(IDRE.match(t)) and not _kwlike(t) for t in r['toks'][N]]
            for cj, s in enumerate(sp):
                if s is None:
                    continue
                for k in range(1, len(ids[j])):
                    a, b = off[k]
                    if b <= a:
                        continue
                    es = a + (len(tk[a:b]) - len(tk[a:b].lstrip()))
                    if es < s[1] and b > s[0]:
                        if isid_tok[cj]:
                            idmap[k] = True
                        start_of.setdefault(cj, k)
                        ren[cj] = okflag[cj]
            ok = np.zeros(len(ids[j]), dtype=bool)
            ok[1:] = np.asarray(pr[:-1]) == np.asarray(ids[j][1:])
            lo = min([k for cj, k in start_of.items() if cj >= warmup] or [len(ids[j])])
            for k in range(max(1, lo), len(ids[j])):
                g = 'ident' if idmap[k] else 'nonident'
                tot['all'][0] += int(ok[k]); tot['all'][1] += 1
                tot[g][0] += int(ok[k]); tot[g][1] += 1
                maj[int(ids[j][k])] += 1
            for cj, k in start_of.items():
                if cj >= warmup and k >= 1:
                    cmaj[r['toks'][N][cj]] += 1
            for cj, k in start_of.items():
                if cj >= warmup and k >= 1:
                    per_tok.setdefault((j, cj), {})[N] = (bool(ok[k]), bool(ren.get(cj, False)))
        res['views'][str(N)] = {
            'next_token_acc': {g: wilson(*v) for g, v in tot.items()},
            'majority_token_baseline': wilson(maj.most_common(1)[0][1] if maj else 0, tot['all'][1]),
            'majority_code_token_baseline': wilson(cmaj.most_common(1)[0][1] if cmaj else 0,
                                                   sum(cmaj.values())),
            'majority_code_token': cmaj.most_common(1)[0][0] if cmaj else None}
        print('  py150 N=%d  all %.4f  ident %.4f  nonident %.4f  (n=%d, majority %.4f)'
              % (N, tot['all'][0] / max(1, tot['all'][1]), tot['ident'][0] / max(1, tot['ident'][1]),
                 tot['nonident'][0] / max(1, tot['nonident'][1]), tot['all'][1],
                 res['views'][str(N)]['majority_token_baseline']['rate'] or 0), flush=True)
    # code-token level: first piece correct, aligned across views
    keys = [k for k, d in per_tok.items() if all(N in d for N in views)]
    base = np.array([per_tok[k][0][0] for k in keys])
    isren = {N: np.array([per_tok[k][N][1] for k in keys]) for N in views}
    res['code_token'] = {'n_aligned': len(keys),
                         'first_piece_acc_N0': wilson(int(base.sum()), len(keys))}
    for N in BENCH_NS:
        cur = np.array([per_tok[k][N][0] for k in keys])
        unchanged = ~isren[N]
        res['code_token']['N%d' % N] = {
            'first_piece_acc': wilson(int(cur.sum()), len(keys)),
            'retention_all': wilson(int((base & cur).sum()), int(base.sum())),
            'retention_unchanged_tokens': wilson(int((base & cur & unchanged).sum()),
                                                 int((base & unchanged).sum())),
            'retention_renamed_tokens': wilson(int((base & cur & isren[N]).sum()),
                                               int((base & isren[N]).sum())),
            'n_renamed_tokens': int(isren[N].sum())}
        print('  py150 code-token N=%d acc %.4f retention %.4f (unchanged %.4f, renamed %.4f)'
              % (N, cur.mean(), res['code_token']['N%d' % N]['retention_all']['rate'] or 0,
                 res['code_token']['N%d' % N]['retention_unchanged_tokens']['rate'] or 0,
                 res['code_token']['N%d' % N]['retention_renamed_tokens']['rate'] or 0), flush=True)
    return res


def mbpp_corpus(n=500, seed=BENCH_SEED):
    """MBPP test programs, canonicalised with ast.unparse, with views N = 0,1,4,8.

    EVERY view including N=0 is ast.unparse output, so the views differ only in identifier names
    (the same convention real6 uses). Renameable locals come from whyscale._py_renameable and the
    rename is certified by whyscale._alpha_ok.
    """
    import whyscale as WS
    from datasets import load_dataset
    ds = load_dataset('google-research-datasets/mbpp', 'full', split='test')
    out, drops = [], Counter()
    ren = {N: [] for N in BENCH_NS}
    for i, r in enumerate(ds):
        src = r['code'].replace('\r\n', '\n')
        try:
            code = ast.unparse(ast.parse(src))
            ast.parse(code)
        except Exception:
            drops['unparseable'] += 1
            continue
        got = WS._py_renameable(code)
        if got is None:
            drops['no_symtable'] += 1
            continue
        names = got[0]
        views, ok = {0: code}, True
        for N in BENCH_NS:
            rng = random.Random(hash_seed('mbpp', seed, i, N))
            picked = list(names)
            rng.shuffle(picked)
            picked = picked[:N]
            taken = set(WS.IDENT_RE.findall(code))
            mp = {}
            for v in picked:
                nn, g = _rand_letters(rng, len(v)), 0
                while (nn in taken or nn in mp.values()) and g < 256:
                    nn = _rand_letters(rng, len(v))
                    g += 1
                mp[v] = nn
            new = WS._real6_apply(code, mp) if mp else code
            if mp and not WS._alpha_ok(code, new, mp):
                drops['rename_failed'] += 1
                ok = False
                break
            views[N] = new
            ren[N].append(len(mp))
        if ok:
            out.append({'task_id': r['task_id'], 'views': views, 'n_names': len(names)})
    if len(out) > n:
        st = max(1, len(out) // n)
        out = out[::st][:n]
    meta = {'n_items': len(out), 'drops': dict(drops), 'seed': seed,
            'mean_renameable_locals': float(np.mean([r['n_names'] for r in out])),
            'mean_renamed': {str(N): float(np.mean(ren[N])) for N in BENCH_NS},
            'canonicalised': 'every view is ast.unparse output'}
    return out, meta


def mbpp_cut(text, frac=0.50):
    """Target line = the first non-blank line with index >= 1 whose start is at or after `frac` of
    the characters; if none, the last non-blank line with index >= 1. Prompt includes that line's
    indentation; the reference is the rest of the line. Also returns the previous non-blank line's
    content (the copy baseline)."""
    lines = text.split('\n')
    starts, p = [], 0
    for ln in lines:
        starts.append(p)
        p += len(ln) + 1
    cand = [i for i in range(1, len(lines)) if lines[i].strip() and starts[i] >= frac * len(text)]
    if not cand:
        cand = [i for i in range(1, len(lines)) if lines[i].strip()]
        if not cand:
            return None
        cand = [cand[-1]]
    i = cand[0]
    ln = lines[i]
    lead = len(ln) - len(ln.lstrip())
    prev = next((lines[k].strip() for k in range(i - 1, -1, -1) if lines[k].strip()), '')
    return text[:starts[i] + lead], ln[lead:].rstrip(), prev, i


def bench_mbpp(model, tok, dev, recs, meta, gen_batch):
    """Greedy line completion on MBPP: exact match and edit similarity, with a copy baseline."""
    views = (0,) + BENCH_NS
    cut = {N: [mbpp_cut(r['views'][N]) for r in recs] for N in views}
    keep = [j for j in range(len(recs)) if all(cut[N][j] is not None for N in views)]
    res = {'meta': meta, 'n_scored': len(keep),
           'truncation_rule': mbpp_cut.__doc__.split('Also returns')[0].strip(),
           'decoding': 'greedy, 32 new tokens, cut at the first newline, both sides stripped',
           'views': {}}
    match = {}
    for N in views:
        prompts = [cut[N][j][0] for j in keep]
        refs = [cut[N][j][1] for j in keep]
        prev = [cut[N][j][2] for j in keep]
        gen = greedy_lines(model, tok, prompts, dev, gen_batch)
        m = np.array([g == r.strip() for g, r in zip(gen, refs)])
        es = np.array([edit_sim(g, r.strip()) for g, r in zip(gen, refs)])
        bm = np.array([p == r.strip() for p, r in zip(prev, refs)])
        bes = np.array([edit_sim(p, r.strip()) for p, r in zip(prev, refs)])
        match[N] = m
        res['views'][str(N)] = {
            'exact_line': wilson(int(m.sum()), len(m)), 'edit_similarity': float(es.mean()),
            'baseline_copy_prev_line_exact': wilson(int(bm.sum()), len(bm)),
            'baseline_copy_prev_line_edit_sim': float(bes.mean())}
        print('  mbpp N=%d exact %.4f edit-sim %.4f (copy-prev baseline %.4f / %.4f, n=%d)'
              % (N, m.mean(), es.mean(), bm.mean(), bes.mean(), len(m)), flush=True)
        if N == 0:
            res['examples'] = [{'prompt_tail': prompts[k][-60:], 'ref': refs[k], 'gen': gen[k]}
                               for k in range(3)]
    for N in BENCH_NS:
        res['views'][str(N)]['retention'] = wilson(int((match[0] & match[N]).sum()),
                                                   int(match[0].sum()))
        res['views'][str(N)]['drop'] = float(match[0].mean() - match[N].mean())
    return res


def vm_mbpp_items(recs, seed=BENCH_SEED):
    """CuBERT-style variable-misuse items on MBPP, built with varmisuse.py's own corruption and
    rename machinery (imported, not copied); the corpus is the only thing that changes."""
    import varmisuse as VM
    import whyscale as WS
    items = {k: [] for k in VM.KINDS}
    drops = {k: Counter() for k in VM.KINDS}
    for i, r in enumerate(recs):
        code = r['views'][0]
        names = sorted({n for n in WS.IDENT_RE.findall(code)} & set(
            WS._py_renameable(code)[0] if WS._py_renameable(code) else []))
        rng = random.Random(hash_seed('vmmbpp', seed, i))
        pair = None
        if len(names) >= 2:
            nodes = VM._load_names(code)
            elig = [k for k, n in enumerate(nodes) if n.id in names]
            rng.shuffle(elig)
            for k in elig:
                orig = nodes[k].id
                others = [n for n in names if n != orig]
                new = others[rng.randrange(len(others))]
                bad = VM.corrupt_var_at(code, k, new)
                if bad is not None:
                    pair = {'k': k, 'orig': orig, 'new': new, 'bad': bad}
                    break
        if pair is None:
            drops['var']['no_pair'] += 1
        op = VM.corrupt_op(code, random.Random(hash_seed('vmmbppop', seed, i)))
        if op is None:
            drops['op']['no_pair'] += 1
        for kind, made in (('var', pair), ('op', op)):
            if made is None:
                continue
            it = {'i': i, 'names': names, 'good': {0: code}, 'nren': {0: 0}, 'kind': kind,
                  'touched': {}}
            if kind == 'var':
                it.update({'k': made['k'], 'orig': made['orig'], 'new': made['new'],
                           'bad': {0: made['bad']}})
            else:
                it['bad'] = {0: made[0]}
            ok = True
            for N in VM.NS:
                rr = random.Random(hash_seed('vmren', seed, i, N))
                picked = list(names)
                rr.shuffle(picked)
                picked = picked[:N]
                taken = set(WS.IDENT_RE.findall(code))
                mp = {}
                for v in picked:
                    nn, g = _rand_letters(rr, len(v)), 0
                    while (nn in taken or nn in mp.values()) and g < 256:
                        nn = _rand_letters(rr, len(v))
                        g += 1
                    mp[v] = nn
                g_ren = WS._real6_apply(code, mp) if mp else code
                if mp and not WS._alpha_ok(code, g_ren, mp):
                    drops[kind]['rename_failed'] += 1
                    ok = False
                    break
                if kind == 'var':
                    b_ren = VM.corrupt_var_at(g_ren, it['k'], mp.get(it['new'], it['new']))
                    it['touched'][N] = (it['orig'] in mp) or (it['new'] in mp)
                else:
                    b_ren = WS._real6_apply(it['bad'][0], mp) if mp else it['bad'][0]
                    it['touched'][N] = False
                if b_ren is None or b_ren == g_ren:
                    drops[kind]['rename_failed'] += 1
                    ok = False
                    break
                if mp and not WS._alpha_ok(it['bad'][0], b_ren, mp):
                    drops[kind]['alpha_bad'] += 1
                    ok = False
                    break
                it['good'][N], it['bad'][N], it['nren'][N] = g_ren, b_ren, len(mp)
            if ok:
                items[kind].append(it)
    meta = {'corpus': 'MBPP test (ast.unparse canonical)', 'seed': seed,
            'n_items': {k: len(items[k]) for k in VM.KINDS},
            'drops': {k: dict(v) for k, v in drops.items()},
            'mean_renamed': {k: {str(N): float(np.mean([it['nren'][N] for it in items[k]]))
                                 for N in (0,) + VM.NS} for k in VM.KINDS if items[k]},
            'note': 'built with varmisuse.corrupt_var_at / corrupt_op and whyscale rename+alpha_ok'}
    return items, meta


def cmd_bench(a):
    """Three published-benchmark tasks on one model: PY150 token completion, MBPP line completion,
    CuBERT-style variable misuse on MBPP."""
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    mkey = os.path.basename(os.path.normpath(a.model))
    tasks = ('py150', 'mbpp', 'vmmbpp') if a.task == 'all' else (a.task,)
    out = {'model': a.model, 'provenance': provenance(), 'max_len': a.max_len}
    if 'py150' in tasks:
        recs, meta = py150_corpus(a.n_py150)
        tok, model = load_lm(a.model, dev)
        t0 = time.time()
        out['py150'] = bench_py150(model, tok, dev, recs, meta, a.batch, a.max_len)
        out['py150']['seconds'] = round(time.time() - t0, 1)
        del model
    if 'mbpp' in tasks or 'vmmbpp' in tasks:
        recs, meta = mbpp_corpus(a.n_mbpp)
        if 'mbpp' in tasks:
            tok, model = load_lm(a.model, dev)
            t0 = time.time()
            out['mbpp'] = bench_mbpp(model, tok, dev, recs, meta, a.gen_batch)
            out['mbpp']['seconds'] = round(time.time() - t0, 1)
            del model
        if 'vmmbpp' in tasks:
            import varmisuse as VM
            items, vmeta = vm_mbpp_items(recs)
            print('  vm-mbpp items %s drops %s' % (vmeta['n_items'], vmeta['drops']), flush=True)
            t0 = time.time()
            raw = VM.run_model(a.model, items, 'decoder', str(dev), a.batch, None)
            s = VM.summarize(raw, items, vmeta, 'decoder', None)
            s['seconds'] = round(time.time() - t0, 1)
            out['vmmbpp'] = s
            for k in VM.KINDS:
                d = s['kinds'][k]
                print('  vm-mbpp %-3s n=%d acc %.4f | ret %s' % (
                    k, d['n_items'], d['accuracy'],
                    ' '.join('%s' % (None if d['retention_N%d' % N] is None
                                     else round(d['retention_N%d' % N], 4)) for N in VM.NS)),
                    flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    for t in tasks:
        if t in out:
            store(['bench_' + t, mkey], {**out[t], 'model': a.model,
                                         'provenance': out['provenance']}, a.out)
            print('wrote', a.out, ['bench_' + t, mkey], flush=True)
    return 0




# ===========================================================================
# RETENTION CURVES: retrieval / completion at N = 1,2,4,8 renames
# (variable misuse gets its curve from varmisuse.py --ns 1,2,4,8)
# ===========================================================================
CURVE_NS = (0, 1, 2, 4, 8)


def curve_views(a, seed=BENCH_SEED):
    """One shared set of N-rename views for BOTH curve tasks: the real6 functions, with N renameable
    locals rewritten to random SAME-LENGTH names (whyscale._real6_apply, certified by _alpha_ok).
    N=0 is the canonical source. Identical scheme, seed and item set for retrieval and completion.
    """
    import whyscale as WS
    with open(os.path.join(WS.CACHE, 'real6_corpus.json')) as f:
        cached = json.load(f)
    recs = cached['recs'][:a.n]
    out, drops, ren = [], Counter(), {N: [] for N in CURVE_NS}
    for i, r in enumerate(recs):
        code, names = r['src'], r['names']
        views, ok = {0: code}, True
        for N in CURVE_NS[1:]:
            rng = random.Random(hash_seed('curveren', seed, i, N))
            picked = list(names)
            rng.shuffle(picked)
            picked = picked[:N]
            taken = set(WS.IDENT_RE.findall(code))
            mp = {}
            for v in picked:
                nn, g = _rand_letters(rng, len(v)), 0
                while (nn in taken or nn in mp.values()) and g < 256:
                    nn = _rand_letters(rng, len(v))
                    g += 1
                mp[v] = nn
            new = WS._real6_apply(code, mp) if mp else code
            if mp and not WS._alpha_ok(code, new, mp):
                drops['rename_failed'] += 1
                ok = False
                break
            views[N] = new
            ren[N].append(len(mp))
        if not ok:
            continue
        occ = {}
        for N in CURVE_NS:
            o = occurrences(code, views[N], names)
            if o is None:
                occ = None
                break
            occ[N] = o
        if occ is None:
            drops['ast_alignment'] += 1
            continue
        if any([segments(views[N], occ[N]) != segments(code, occ[0]) for N in CURVE_NS]):
            drops['segments_differ'] += 1
            continue
        out.append({'i': i, 'views': views, 'occ': occ, 'names': names})
    meta = {'n_functions': len(out), 'drops': dict(drops), 'seed': seed,
            'scheme': 'samelen: N renameable locals -> random same-length names, _alpha_ok certified',
            'mean_renamed': {str(N): float(np.mean(ren[N])) for N in CURVE_NS[1:]},
            'mean_renameable_locals': float(np.mean([len(r['names']) for r in out]))}
    return out, meta


def curve_retrieval(a, recs, meta):
    """Retention of paired retrieval: query = the renamed function, pool = the 1962 originals."""
    import whyscale as WS
    tok, enc = WS.load_encoder(a.model, str(device_of(a.device)))
    codes = [r['views'][0] for r in recs]
    blank = [set() for _ in codes]
    E0, _ = WS.embed(tok, enc, codes, blank, str(device_of(a.device)), a.batch, normalize=False)
    out = {'meta': meta, 'definition': 'retention = P(own original is the nearest neighbour after N '
                                       'renames | it was with original names); pool = all originals'}
    base = WS._paired(E0, E0)
    out['N0_self_retrieval'] = {'rate': float(base.mean()), 'n': int(len(base))}
    for N in CURVE_NS[1:]:
        P, _ = WS.embed(tok, enc, [r['views'][N] for r in recs], blank,
                        str(device_of(a.device)), a.batch, normalize=False)
        hit = WS._paired(P, E0)
        out['N%d' % N] = {'paired_top1': float(hit.mean()),
                          'retention': wilson(int((hit & base).sum()), int(base.sum()))}
        print('  retrieval N=%d  top1 %.4f  retention %.4f (n=%d)'
              % (N, hit.mean(), out['N%d' % N]['retention']['rate'], int(base.sum())), flush=True)
    del enc
    return out


def curve_completion(a, recs, meta):
    """Retention of correct next-token predictions at paired NON-IDENTIFIER positions."""
    dev = device_of(a.device)
    tok, model = load_lm(a.model, dev)
    R = {}
    for N in CURVE_NS:
        ids, offs, preds = lm_argmax(model, tok, [r['views'][N] for r in recs], dev, a.batch, a.max_len)
        R[N] = [comp_view_record(recs[j]['views'][N], recs[j]['occ'][N], ids[j], offs[j], preds[j])
                for j in range(len(recs))]
    out = {'meta': meta, 'definition': 'retention = P(next token still correct after N renames | it '
                                       'was correct with original names), paired non-identifier '
                                       'positions (same token id at the same aligned offset)'}
    tot0 = sum(int(r['ok'][k]) for r in R[0] for k in range(1, len(r['ids']))
               if not r['isid'][k] and r['key'][k] is not None)
    out['N0_nonident_correct'] = tot0
    # DENOMINATOR, added 2026-09-20 by [AGENT PKG] (additive: no existing key changes). The
    # retention rows are conditioned on the N=0 successes, so the N=0 ACCURACY they are
    # conditioned on has to be reportable beside them; without this key only the numerator was.
    tot0_n = sum(1 for r in R[0] for k in range(1, len(r['ids']))
                 if not r['isid'][k] and r['key'][k] is not None)
    out['N0_nonident_total'] = tot0_n
    out['N0_nonident_accuracy'] = wilson(tot0, tot0_n)
    for N in CURVE_NS[1:]:
        k_both = k_base = 0
        for o, w in zip(R[0], R[N]):
            wmap = {kk: i for i, kk in enumerate(w['key']) if kk is not None}
            for i, kk in enumerate(o['key']):
                if kk is None or i == 0 or o['isid'][i]:
                    continue
                j = wmap.get(kk)
                if j is None or w['isid'][j]:
                    continue
                if o['ok'][i]:
                    k_base += 1
                    k_both += int(w['ok'][j])
        out['N%d' % N] = {'retention': wilson(k_both, k_base)}
        print('  completion N=%d retention %.4f (n=%d)'
              % (N, k_both / max(1, k_base), k_base), flush=True)
    del model
    return out


def cmd_curves(a):
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    recs, meta = curve_views(a)
    print('curve views: %d functions, %s' % (len(recs), meta['mean_renamed']), flush=True)
    mkey = os.path.basename(os.path.normpath(a.model))
    res = {'model': a.model, 'provenance': provenance(), 'max_len': a.max_len,
           'retrieval': curve_retrieval(a, recs, meta),
           'completion': curve_completion(a, recs, meta)}
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['curves', mkey], res, a.out)
    print('wrote', a.out, ['curves', mkey])
    return 0




# ===========================================================================
# DOCSTRING-CODE MATCHING (CuBERT's function-docstring mismatch, zero-shot)
# ===========================================================================
DOC_CACHE = 'docmatch_items.json'


def doc_split(code):
    """-> (canonical body with the docstring removed, docstring, renameable locals) or None."""
    import whyscale as WS
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(tree.body) != 1 or not fns:
        return None
    fn = fns[0]
    doc = ast.get_docstring(fn)
    if not doc or len(doc.split()) < 4 or len(doc) > 400:
        return None
    b = fn.body
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
            and isinstance(b[0].value.value, str):
        fn.body = b[1:] or [ast.Pass()]
    try:
        body = ast.unparse(tree)
        ast.parse(body)
    except Exception:
        return None
    if len(body.strip()) < 40:
        return None
    got = WS._py_renameable(body)
    if got is None or not got[0]:
        return None
    return body, ' '.join(doc.split()), got[0]


def doc_items(n_items=800, seed=BENCH_SEED, dev_n=150, max_scan=20000):
    """Fixed, seeded item set shared by every model and every N.

    Distractor rule: the distractor docstring is drawn (seeded) from the items whose docstring
    LENGTH IN CHARACTERS is within +-20% of this item's, excluding itself and any identical string,
    so docstring length cannot be the cue. An item with no such candidate is dropped and counted.
    The first `dev_n` items are the DEV slice used only to pick the scoring format; the evaluation
    set is disjoint from it.
    """
    import whyscale as WS
    cache = os.path.join(WS.CACHE, DOC_CACHE)
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        if d['n_requested'] == n_items and d['seed'] == seed and d['dev_n'] == dev_n:
            print('docmatch items from cache: %d' % len(d['items']), flush=True)
            return d['items'], d['meta']
    from datasets import load_dataset
    ds = load_dataset('code_search_net', 'python', split='test', streaming=True,
                      trust_remote_code=True)
    raw, scanned, drops = [], 0, Counter()
    for r in ds:
        scanned += 1
        s = doc_split(r['func_code_string'])
        if s is None:
            drops['filtered'] += 1
        else:
            raw.append({'body': s[0], 'doc': s[1], 'names': s[2]})
        if len(raw) >= n_items + dev_n or scanned >= max_scan:
            break
    rng = random.Random(seed)
    items = []
    for i, it in enumerate(raw):
        L = len(it['doc'])
        cand = [j for j, o in enumerate(raw)
                if j != i and o['doc'] != it['doc'] and 0.8 * L <= len(o['doc']) <= 1.25 * L]
        if not cand:
            drops['no_length_matched_distractor'] += 1
            continue
        j = cand[random.Random(hash_seed('docdistract', seed, i)).randrange(len(cand))]
        items.append({'i': i, 'body': it['body'], 'doc': it['doc'], 'names': it['names'],
                      'other': raw[j]['doc'],
                      'mentions_any_renameable': any(re.search(r'\b%s\b' % re.escape(v), it['doc'])
                                                     for v in it['names'])})
    meta = {'corpus': 'code_search_net python test (func_code_string), streamed',
            'scanned': scanned, 'n_raw': len(raw), 'n_items': len(items), 'drops': dict(drops),
            'frac_rejected_no_distractor': drops['no_length_matched_distractor'] / max(1, len(raw)),
            'dev_n': dev_n, 'seed': seed,
            'distractor_rule': doc_items.__doc__.split('Distractor rule:')[1].split('The first')[0].strip(),
            'body_is_canonical_ast_unparse': True,
            'frac_mentions_any_renameable': float(np.mean([it['mentions_any_renameable']
                                                           for it in items]))}
    os.makedirs(WS.CACHE, exist_ok=True)
    with open(cache, 'w') as f:
        json.dump({'n_requested': n_items, 'seed': seed, 'dev_n': dev_n, 'items': items,
                   'meta': meta}, f)
    print('docmatch items: %d from %d scanned -> %s' % (len(items), scanned, cache), flush=True)
    return items, meta


def doc_views(items, ns=CURVE_NS, seed=BENCH_SEED):
    """N-rename views of each BODY (docstrings are the candidates and are never renamed), same
    machinery and seeding discipline as decmech `curves`."""
    import whyscale as WS
    out, drops = [], Counter()
    for it in items:
        views, ok, mapped = {0: it['body']}, True, {0: []}
        for N in ns[1:]:
            rng = random.Random(hash_seed('docren', seed, it['i'], N))
            picked = list(it['names'])
            rng.shuffle(picked)
            picked = picked[:N]
            taken = set(WS.IDENT_RE.findall(it['body']))
            mp = {}
            for v in picked:
                nn, g = _rand_letters(rng, len(v)), 0
                while (nn in taken or nn in mp.values()) and g < 256:
                    nn = _rand_letters(rng, len(v))
                    g += 1
                mp[v] = nn
            new = WS._real6_apply(it['body'], mp) if mp else it['body']
            if mp and not WS._alpha_ok(it['body'], new, mp):
                drops['rename_failed'] += 1
                ok = False
                break
            views[N], mapped[N] = new, sorted(mp)
        if ok:
            out.append({**it, 'views': views, 'renamed_names': mapped})
    return out, dict(drops)


@torch.no_grad()
def doc_score(model, tok, pairs, dev, batch, max_len=280, cont_max=48, fmt='docstring_slot',
              reduce='mean'):
    """Length-normalised log P(docstring tokens | code) for a list of (body, doc).

    fmt='docstring_slot': the docstring is scored where a docstring actually goes -- the function
        header, then a triple-quoted string. Prefix = 'def ...():\\n    \"\"\"'.
    fmt='trailing_comment': the pilot's format, prefix = body + '\\n# '.
    """
    seqs = []
    for body, doc in pairs:
        if fmt == 'trailing_comment_raw':
            pre = tok(body, truncation=True, max_length=max_len).input_ids
        elif fmt == 'docstring_slot':
            head, rest = body.split('\n', 1) if '\n' in body else (body, '')
            pre = tok(head + '\n    """', truncation=True, max_length=max_len).input_ids
        else:
            pre = tok(body + '\n# ', truncation=True, max_length=max_len).input_ids
        cont = tok(doc, add_special_tokens=False).input_ids[:cont_max]
        seqs.append((pre, cont))
    out = np.full(len(seqs), -1e9)
    order = [i for i in sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]),
                               reverse=True) if seqs[i][1]]
    pad = tok.pad_token_id
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
        inp = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            ids = seqs[i][0] + seqs[i][1]
            inp[j, :len(ids)] = torch.tensor(ids)
            att[j, :len(ids)] = 1
        lg = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float().log_softmax(-1)
        for j, i in enumerate(idx):
            pre, cont = seqs[i]
            pos = torch.arange(len(pre) - 1, len(pre) - 1 + len(cont))
            lp = lg[j, pos, torch.tensor(cont)].sum().item()
            out[i] = float(lp / len(cont)) if reduce == 'mean' else float(lp)
    return out


def doc_accuracy(model, tok, recs, N, dev, batch, fmt):
    """-> bool array: is the TRUE docstring scored above the distractor, for view N."""
    pairs = [(r['views'][N], r['doc']) for r in recs] + [(r['views'][N], r['other']) for r in recs]
    s = doc_score(model, tok, pairs, dev, batch, fmt=fmt)
    n = len(recs)
    return s[:n] > s[n:]


def cmd_docmatch(a):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    items, meta = doc_items(a.n_items)
    dev_items = items[:meta['dev_n']]
    ev_items = items[meta['dev_n']:]
    recs, drops = doc_views(ev_items)
    meta = {**meta, 'rename_drops': drops, 'n_scored': len(recs)}
    tok, model = load_lm(a.model, dev)
    res = {'model': a.model, 'meta': meta, 'provenance': provenance(), 'format': a.fmt}
    if a.pick_format:
        dv, _ = doc_views(dev_items)
        for f in ('docstring_slot', 'trailing_comment'):
            h = doc_accuracy(model, tok, dv, 0, dev, a.batch, f)
            print('  DEV format %-18s accuracy %.4f (n=%d)' % (f, h.mean(), len(h)), flush=True)
            res.setdefault('dev_format_choice', {})[f] = {'accuracy': float(h.mean()), 'n': len(h)}
    hits = {}
    for N in CURVE_NS:
        hits[N] = doc_accuracy(model, tok, recs, N, dev, a.batch, a.fmt)
        print('  docmatch N=%d accuracy %.4f (n=%d)' % (N, hits[N].mean(), len(recs)), flush=True)
    base = hits[0]
    res['N0_accuracy'] = wilson(int(base.sum()), len(base))
    res['frac_mentions_any_renameable'] = meta['frac_mentions_any_renameable']
    for N in CURVE_NS[1:]:
        ment = np.array([any(re.search(r'\b%s\b' % re.escape(v), r['doc'])
                             for v in r['renamed_names'][N]) for r in recs])
        cell = {'accuracy': float(hits[N].mean()),
                'retention': wilson(int((base & hits[N]).sum()), int(base.sum())),
                'mentions': {'retention': wilson(int((base & hits[N] & ment).sum()),
                                                 int((base & ment).sum())),
                             'n_items': int(ment.sum())},
                'no_mention': {'retention': wilson(int((base & hits[N] & ~ment).sum()),
                                                   int((base & ~ment).sum())),
                               'n_items': int((~ment).sum())},
                'frac_docstring_mentions_a_renamed_name': float(ment.mean())}
        res['N%d' % N] = cell
        print('    N=%d retention %.4f | mentions %.4f (n=%d) | no-mention %.4f (n=%d)'
              % (N, cell['retention']['rate'], cell['mentions']['retention']['rate'] or 0,
                 cell['mentions']['retention']['n'], cell['no_mention']['retention']['rate'] or 0,
                 cell['no_mention']['retention']['n']), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['docmatch', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['docmatch', os.path.basename(os.path.normpath(a.model))])
    return 0




# ===========================================================================
# CODE-ONLY TASK PILOTS (feasibility round; no retention curves)
# ===========================================================================
PROBE_TASKS = ('exctype', 'stmtswap', 'binop', 'operands', 'useassign', 'nextline')
CTRL = (ast.Return, ast.Raise, ast.Break, ast.Continue, ast.Global, ast.Nonlocal, ast.If,
        ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.Import, ast.ImportFrom, ast.Yield)
NONCOMM = (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift)
CMP_DIR = (ast.Lt, ast.Gt, ast.LtE, ast.GtE)


def _blocks(tree):
    for n in ast.walk(tree):
        for f in ('body', 'orelse', 'finalbody'):
            b = getattr(n, f, None)
            if isinstance(b, list) and len(b) >= 2:
                yield n, f, b


def _nm(node, ctx):
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ctx)}


def real6_sources(n=2000):
    import whyscale as WS
    with open(os.path.join(WS.CACHE, 'real6_corpus.json')) as f:
        return [r['src'] for r in json.load(f)['recs'][:n]]


def probe_stmtswap(srcs, seed, cap=400):
    """Swap two ADJACENT statements of one block when neither is control flow / import / def and
    neither writes a name the other reads or writes (statically data-independent). The swapped
    version must parse and differ textually."""
    out = []
    for i, src in enumerate(srcs):
        tree = ast.parse(src)
        cands = []
        for node, field, b in _blocks(tree):
            for k in range(len(b) - 1):
                s1, s2 = b[k], b[k + 1]
                if any(isinstance(s, CTRL) for s in (s1, s2)):
                    continue
                w1, w2 = _nm(s1, (ast.Store, ast.Del)), _nm(s2, (ast.Store, ast.Del))
                r1, r2 = _nm(s1, ast.Load), _nm(s2, ast.Load)
                if (w1 & (r2 | w2)) or (w2 & (r1 | w1)):
                    continue
                cands.append((node, field, k))
        if not cands:
            continue
        node, field, k = cands[random.Random(hash_seed('swap', seed, i)).randrange(len(cands))]
        b = getattr(node, field)
        b[k], b[k + 1] = b[k + 1], b[k]
        try:
            bad = ast.unparse(tree)
            ast.parse(bad)
        except Exception:
            continue
        good = ast.unparse(ast.parse(src))
        if bad != good:
            out.append({'good': good, 'bad': bad, 'src_index': i})
        if len(out) >= cap:
            break
    return out, {'rule': probe_stmtswap.__doc__.strip()}


def probe_operands(srcs, seed, cap=400):
    """Swap the operands of a NON-COMMUTATIVE binary op (- / // % ** << >>) or of a directional
    comparison (< > <= >=), when the two sides differ textually. Must parse and differ."""
    out = []
    for i, src in enumerate(srcs):
        tree = ast.parse(src)
        cands = []
        for n in ast.walk(tree):
            if isinstance(n, ast.BinOp) and isinstance(n.op, NONCOMM) \
                    and ast.unparse(n.left) != ast.unparse(n.right):
                cands.append(n)
            elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], CMP_DIR) \
                    and ast.unparse(n.left) != ast.unparse(n.comparators[0]):
                cands.append(n)
        if not cands:
            continue
        n = cands[random.Random(hash_seed('operands', seed, i)).randrange(len(cands))]
        if isinstance(n, ast.BinOp):
            n.left, n.right = n.right, n.left
        else:
            n.left, n.comparators[0] = n.comparators[0], n.left
        try:
            bad = ast.unparse(tree)
            ast.parse(bad)
        except Exception:
            continue
        good = ast.unparse(ast.parse(src))
        if bad != good:
            out.append({'good': good, 'bad': bad, 'src_index': i})
        if len(out) >= cap:
            break
    return out, {'rule': probe_operands.__doc__.strip()}


def probe_useassign(srcs, seed, cap=400):
    """Move a statement that READS a variable above the assignment that defines it, inside the same
    block. The moved statement is not control flow, and it must not write anything the statements it
    jumps over read, so the only defect introduced is the use before assignment."""
    out = []
    for i, src in enumerate(srcs):
        tree = ast.parse(src)
        cands = []
        for node, field, b in _blocks(tree):
            for k, s in enumerate(b):
                if not (isinstance(s, ast.Assign) and len(s.targets) == 1
                        and isinstance(s.targets[0], ast.Name)):
                    continue
                tgt = s.targets[0].id
                for j in range(k + 1, len(b)):
                    if isinstance(b[j], CTRL):
                        continue
                    if tgt not in _nm(b[j], ast.Load):
                        continue
                    jumped = set().union(*[_nm(b[m], ast.Load) for m in range(k, j)]) if j > k else set()
                    if _nm(b[j], (ast.Store, ast.Del)) & jumped:
                        continue
                    cands.append((node, field, k, j, tgt))
        if not cands:
            continue
        node, field, k, j, tgt = cands[random.Random(hash_seed('useassign', seed, i)).randrange(len(cands))]
        b = getattr(node, field)
        mv = b.pop(j)
        b.insert(k, mv)
        try:
            bad = ast.unparse(tree)
            ast.parse(bad)
        except Exception:
            continue
        good = ast.unparse(ast.parse(src))
        if bad != good:
            out.append({'good': good, 'bad': bad, 'var': tgt, 'src_index': i})
        if len(out) >= cap:
            break
    return out, {'rule': probe_useassign.__doc__.strip()}


def probe_binop(srcs, seed, cap=400):
    """Flip one binary operator (+/-, */ /, ==/!=, </>), via varmisuse.corrupt_op -- the same
    corruption as the operator control, promoted to a first-class task."""
    import varmisuse as VM
    out = []
    for i, src in enumerate(srcs):
        got = VM.corrupt_op(ast.unparse(ast.parse(src)), random.Random(hash_seed('binop', seed, i)))
        if got is None:
            continue
        out.append({'good': ast.unparse(ast.parse(src)), 'bad': got[0], 'op': got[1],
                    'src_index': i})
        if len(out) >= cap:
            break
    return out, {'rule': probe_binop.__doc__.strip()}


def probe_nextline(srcs, seed, cap=400, k=4, unseen=False):
    """Truncate at a line boundary (first non-blank line at or after 50% of the characters, never
    the signature); the candidates are the true next line and k-1 OTHER non-blank lines of the SAME
    function, so identifiers cannot give the answer away. Prompt ends with the line's indentation."""
    out = []
    for i, src in enumerate(srcs):
        code = ast.unparse(ast.parse(src))
        cut = _truncation_point(code, 0.50)
        if cut is None:
            continue
        prompt, ref, li = cut
        lines = [x.strip() for x in code.split('\n')]
        rng_pool = range(li + 1, len(lines)) if unseen else range(len(lines))
        pool = sorted({lines[j] for j in rng_pool if lines[j] and j != li and lines[j] != ref})
        if len(pool) < k - 1 or not ref:
            continue
        rng = random.Random(hash_seed('nextline', seed, i))
        distract = rng.sample(pool, k - 1)
        cands = [ref] + distract
        order = list(range(k))
        rng.shuffle(order)
        out.append({'prompt': prompt, 'cands': [cands[j] for j in order],
                    'true': order.index(0)})
        if len(out) >= cap:
            break
    return out, {'rule': probe_nextline.__doc__.strip(), 'k': k,
                 'distractors': 'lines AFTER the cut (never seen in the prompt)' if unseen
                 else 'any other line of the same function'}


def probe_exctype(n_items, seed, max_scan=8000, n_labels=20):
    """`except <Type>:` with the type blanked; the model ranks the types that actually occur in the
    corpus. Streams CodeSearchNet test, because real6 has only 118 such functions."""
    from datasets import load_dataset
    ds = load_dataset('code_search_net', 'python', split='test', streaming=True)
    raw, counts, scanned = [], Counter(), 0
    for r in ds:
        scanned += 1
        code = r['func_code_string']
        try:
            tree = ast.parse(code)
        except Exception:
            continue
        hs = [x for x in ast.walk(tree) if isinstance(x, ast.ExceptHandler)
              and isinstance(x.type, ast.Name)]
        if len(hs) != 1:
            continue
        h = hs[0]
        lines = code.split('\n')
        li, c0, c1 = h.type.lineno - 1, h.type.col_offset, h.type.end_col_offset
        if li >= len(lines) or lines[li][c0:c1] != h.type.id:
            continue
        prefix = '\n'.join(lines[:li]) + ('\n' if li else '') + lines[li][:c0]
        counts[h.type.id] += 1
        raw.append({'prefix': prefix, 'true': h.type.id})
        if len(raw) >= n_items * 3 or scanned >= max_scan:
            break
    labels = [t for t, _ in counts.most_common(n_labels)]
    items = [r for r in raw if r['true'] in labels][:n_items]
    top = counts.most_common(1)[0][0]
    meta = {'rule': probe_exctype.__doc__.strip(), 'scanned': scanned, 'labels': labels,
            'n_labels': len(labels), 'most_frequent_label': top,
            'baseline_most_frequent': float(np.mean([it['true'] == top for it in items])),
            'label_counts': dict(counts.most_common(n_labels))}
    return items, meta


@torch.no_grad()
def seq_ll(model, tok, texts, dev, batch, max_len=320):
    """Length-normalised conditional log-likelihood of each whole text (varmisuse's decoder score,
    batched)."""
    enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=False)['input_ids']
    out = np.full(len(texts), -1e9)
    order = [i for i in sorted(range(len(texts)), key=lambda i: len(enc[i]), reverse=True)
             if len(enc[i]) >= 2]
    pad = tok.pad_token_id
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        T = max(len(enc[i]) for i in idx)
        inp = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            inp[j, :len(enc[i])] = torch.tensor(enc[i])
            att[j, :len(enc[i])] = 1
        lg = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float().log_softmax(-1)
        for j, i in enumerate(idx):
            ids = torch.tensor(enc[i])
            out[i] = float(lg[j, torch.arange(len(ids) - 1), ids[1:]].sum().item() / (len(ids) - 1))
    return out


def cmd_probe(a):
    """Feasibility pilots for code-only candidate tasks on ONE model (no retention curves)."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    tasks = PROBE_TASKS if a.task == 'all' else (a.task,)
    srcs = real6_sources()
    tok, model = load_lm(a.model, dev)
    res = {'model': a.model, 'provenance': provenance(), 'corpus': 'real6 sources (CSN python test)'}
    for t in tasks:
        t0 = time.time()
        if t == 'exctype':
            items, meta = probe_exctype(a.n_items, BENCH_SEED)
            scores = {}
            for kind in ('sum', 'first', 'norm'):
                hit = []
                for it in items:
                    pre = tok(it['prefix'], truncation=True, max_length=300)['input_ids']
                    best, arg = -1e18, None
                    texts = [it['prefix'] + lab for lab in meta['labels']]
                    enc = tok(texts, truncation=True, max_length=320, add_special_tokens=False)
                    ids = enc['input_ids']
                    L = max(len(x) for x in ids)
                    inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
                    att = torch.zeros((len(ids), L), dtype=torch.long)
                    for j, x in enumerate(ids):
                        inp[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
                    with torch.no_grad():
                        lg = model(input_ids=inp.to(dev), attention_mask=att.to(dev)
                                   ).logits.float().log_softmax(-1).cpu()
                    for j, lab in enumerate(meta['labels']):
                        cont = tok(lab, add_special_tokens=False)['input_ids']
                        n_pre = len(ids[j]) - len(cont)
                        if n_pre < 1:
                            continue
                        pos = torch.arange(n_pre - 1, n_pre - 1 + len(cont))
                        lp = lg[j, pos, torch.tensor(cont)]
                        s = float(lp.sum()) if kind == 'sum' else (
                            float(lp[0]) if kind == 'first' else float(lp.mean()))
                        if s > best:
                            best, arg = s, lab
                    hit.append(arg == it['true'])
                scores[kind] = wilson(int(np.sum(hit)), len(hit))
                print('  exctype %-5s accuracy %.4f (n=%d, baseline %.4f)'
                      % (kind, np.mean(hit), len(hit), meta['baseline_most_frequent']), flush=True)
            res['exctype'] = {'meta': meta, 'n': len(items), 'scoring_variants': scores,
                              'baseline_most_frequent': meta['baseline_most_frequent'],
                              'best_variant': max(scores, key=lambda k: scores[k]['rate']),
                              'seconds': round(time.time() - t0, 1)}
        elif t == 'nextline':
            items, meta = probe_nextline(srcs, BENCH_SEED, unseen=a.nextline_unseen)
            texts, idx = [], []
            for it in items:
                for c in it['cands']:
                    texts.append((it['prompt'], c))
            k = meta['k']
            true = np.array([it['true'] for it in items])
            acc = {}
            for red in ('mean', 'sum'):
                sc = doc_score(model, tok, texts, dev, a.batch, fmt='trailing_comment_raw',
                               reduce=red).reshape(len(items), k)
                h = sc.argmax(1) == true
                acc[red] = wilson(int(h.sum()), len(h))
                print('  nextline (%s-reduced) accuracy %.4f' % (red, h.mean()), flush=True)
                if red == 'mean':
                    hit = h
            allc = Counter(c for it in items for c in it['cands'])
            maj = np.mean([it['cands'][int(np.argmax([allc[c] for c in it['cands']]))]
                           == it['cands'][it['true']] for it in items])
            short = np.mean([int(np.argmin([len(c) for c in it['cands']])) == it['true']
                             for it in items])
            res['nextline'] = {'meta': meta, 'n': len(items),
                               'accuracy_by_reduction': acc,
                               'accuracy': max(acc.values(), key=lambda v: v['rate']),
                               'baseline_chance': 1.0 / k,
                               'baseline_most_frequent_candidate': float(maj),
                               'baseline_shortest_candidate': float(short),
                               'seconds': round(time.time() - t0, 1)}
            print('  nextline best %.4f (mean %.4f / sum %.4f) n=%d k=%d chance %.3f '
                  'most-frequent %.3f shortest %.3f  [%s]'
                  % (max(v['rate'] for v in acc.values()), acc['mean']['rate'], acc['sum']['rate'],
                     len(items), k, 1.0 / k, maj, short, meta['distractors']), flush=True)
        else:
            build = {'stmtswap': probe_stmtswap, 'operands': probe_operands,
                     'useassign': probe_useassign, 'binop': probe_binop}[t]
            items, meta = build(srcs, BENCH_SEED)
            g = seq_ll(model, tok, [it['good'] for it in items], dev, a.batch)
            b = seq_ll(model, tok, [it['bad'] for it in items], dev, a.batch)
            hit = g > b
            res[t] = {'meta': meta, 'n': len(items), 'accuracy': wilson(int(hit.sum()), len(hit)),
                      'ties': int((g == b).sum()), 'baseline_chance': 0.5,
                      'seconds': round(time.time() - t0, 1)}
            print('  %-9s accuracy %.4f (n=%d, chance 0.5, ties %d)'
                  % (t, hit.mean(), len(hit), int((g == b).sum())), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['probe', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['probe', os.path.basename(os.path.normpath(a.model))])
    return 0




def probe_curve_views(items, seed=BENCH_SEED, ns=CURVE_NS):
    """N-rename views of BOTH members of each pair, one mapping per (item, N), certified by
    whyscale._alpha_ok on each member -- the construction varmisuse.py uses."""
    import whyscale as WS
    out, drops = [], Counter()
    for it in items:
        got = WS._py_renameable(it['good'])
        names = got[0] if got else []
        if not names:
            drops['no_renameable'] += 1
            continue
        views, ok, touched = {0: (it['good'], it['bad'])}, True, {0: False}
        for N in ns[1:]:
            rng = random.Random(hash_seed('probeRen', seed, it['src_index'], N))
            picked = list(names)
            rng.shuffle(picked)
            picked = picked[:N]
            taken = set(WS.IDENT_RE.findall(it['good']))
            mp = {}
            for v in picked:
                nn, g = _rand_letters(rng, len(v)), 0
                while (nn in taken or nn in mp.values()) and g < 256:
                    nn = _rand_letters(rng, len(v))
                    g += 1
                mp[v] = nn
            try:
                gN = WS._real6_apply(it['good'], mp) if mp else it['good']
                bN = WS._real6_apply(it['bad'], mp) if mp else it['bad']
            except Exception:
                drops['apply_failed'] += 1
                ok = False
                break
            if mp and not (WS._alpha_ok(it['good'], gN, mp) and WS._alpha_ok(it['bad'], bN, mp)):
                drops['alpha_bad'] += 1
                ok = False
                break
            if gN == bN:
                drops['collapsed'] += 1
                ok = False
                break
            views[N] = (gN, bN)
            touched[N] = it.get('var') in mp
        if ok:
            out.append({**it, 'views': views, 'touched': touched})
    return out, dict(drops)


def cmd_probecurve(a):
    """Retention curves for the passing code-only pair tasks (same protocol as `curves`)."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    srcs = real6_sources()
    tok, model = load_lm(a.model, dev)
    res = {'model': a.model, 'provenance': provenance(),
           'definition': 'accuracy = the ORIGINAL scores above the corrupted one (length-normalised '
                         'whole-function log-likelihood, ties wrong); retention = P(correct after N '
                         'renames | correct at N=0); both members renamed by one mapping per (item, N)'}
    for t in a.tasks.split(','):
        build = {'useassign': probe_useassign, 'binop': probe_binop, 'operands': probe_operands,
                 'stmtswap': probe_stmtswap}[t]
        items, meta = build(srcs, BENCH_SEED)
        recs, drops = probe_curve_views(items)
        hits = {}
        for N in CURVE_NS:
            g = seq_ll(model, tok, [r['views'][N][0] for r in recs], dev, a.batch)
            b = seq_ll(model, tok, [r['views'][N][1] for r in recs], dev, a.batch)
            hits[N] = g > b
        base = hits[0]
        cell = {'meta': {**meta, 'rename_drops': drops}, 'n_items': len(recs),
                'N0_accuracy': wilson(int(base.sum()), len(base))}
        for N in CURVE_NS[1:]:
            c = {'accuracy': float(hits[N].mean()),
                 'retention': wilson(int((base & hits[N]).sum()), int(base.sum()))}
            if t == 'useassign':
                tch = np.array([r['touched'][N] for r in recs])
                c['touched'] = {'retention': wilson(int((base & hits[N] & tch).sum()),
                                                    int((base & tch).sum())),
                                'frac_items': float(tch.mean())}
                c['untouched'] = {'retention': wilson(int((base & hits[N] & ~tch).sum()),
                                                      int((base & ~tch).sum()))}
            cell['N%d' % N] = c
        res[t] = cell
        print('  %-9s N0 %.4f (n=%d) | retention %s' % (
            t, cell['N0_accuracy']['rate'], len(recs),
            ' '.join('N%d %.4f' % (N, cell['N%d' % N]['retention']['rate']) for N in CURVE_NS[1:])),
            flush=True)
        if t == 'useassign':
            print('            touched   %s' % ' '.join(
                'N%d %.4f(n=%d)' % (N, cell['N%d' % N]['touched']['retention']['rate'] or 0,
                                    cell['N%d' % N]['touched']['retention']['n'])
                for N in CURVE_NS[1:]), flush=True)
            print('            untouched %s' % ' '.join(
                'N%d %.4f(n=%d)' % (N, cell['N%d' % N]['untouched']['retention']['rate'] or 0,
                                    cell['N%d' % N]['untouched']['retention']['n'])
                for N in CURVE_NS[1:]), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['probe_curves', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['probe_curves', os.path.basename(os.path.normpath(a.model))])
    return 0




# ===========================================================================
# IDENTIFIER-SELECTION TASKS (produce/identify an identifier, not judge plausibility)
# ===========================================================================
SEL_TASKS = ('varsel', 'retsel', 'argsel')


def _binding_positions(tree, names):
    """-> {name: earliest (lineno, col_offset) at which it is bound} for the given names.

    Bindings counted: function parameters (position of the def), assignment / augmented /
    annotated-assignment targets, for-targets, with-targets, except-handler names, comprehension
    targets, and walrus targets. This is the practical notion of 'in scope at this point' for a
    single-function snippet: a candidate must have been bound EARLIER in the text.
    """
    pos = {}
    def put(n, node):
        if n in names:
            p = (node.lineno, node.col_offset)
            if n not in pos or p < pos[n]:
                pos[n] = p
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs) + \
                    ([args.vararg] if args.vararg else []) + ([args.kwarg] if args.kwarg else []):
                put(a.arg, node)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            put(node.id, node)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            put(node.name, node)
    return pos


def _same_statement_text(tree, node):
    """The source segment of the smallest statement containing `node` (leak check)."""
    best = None
    for s in ast.walk(tree):
        if isinstance(s, ast.stmt) and s.lineno <= node.lineno \
                and getattr(s, 'end_lineno', s.lineno) >= getattr(node, 'end_lineno', node.lineno):
            if best is None or (s.lineno, -getattr(s, 'end_lineno', s.lineno)) >= \
                    (best.lineno, -getattr(best, 'end_lineno', best.lineno)):
                best = s
    return best


def substitute_var_at(code, k, new):
    """Replace the k-th Load Name (ast.walk order) by character span with `new`; identity allowed."""
    import varmisuse as VM
    import poolfix as P
    nodes = VM._load_names(code)
    if k >= len(nodes):
        return None
    node = nodes[k]
    lines, starts = P._line_index(code)
    a = P._pos(lines, starts, node.lineno, node.col_offset)
    b = P._pos(lines, starts, node.end_lineno, node.end_col_offset)
    if code[a:b] != node.id:
        return None
    out = code[:a] + new + code[b:]
    try:
        ast.parse(out)
    except SyntaxError:
        return None
    return out


def build_selection_items(srcs, kind, seed=BENCH_SEED, cap=300):
    """Items for the identifier-selection tasks.

    Site: a Load occurrence of a RENAMEABLE local that was bound EARLIER in the text (so the answer
    is recoverable from the function) and that already occurred earlier (copyable).
      varsel  any such site
      retsel  the site must be the value of a `return` statement
      argsel  the site must be a positional argument of a call
    Candidates: every renameable local bound strictly before the site (the in-scope set), >= 2.
    LEAK RULES (items dropped and counted): the true name must not appear anywhere else in the
    SAME STATEMENT as the site, and not anywhere else on the same LINE.
    """
    import varmisuse as VM
    import whyscale as WS
    out, drops = [], Counter()
    for i, src in enumerate(srcs):
        code = ast.unparse(ast.parse(src))
        got = WS._py_renameable(code)
        if not got or len(got[0]) < 2:
            drops['too_few_locals'] += 1
            continue
        names = set(got[0])
        tree = ast.parse(code)
        binds = _binding_positions(tree, names)
        nodes = VM._load_names(code)
        ret_nodes, arg_nodes = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Name):
                ret_nodes.add((n.value.lineno, n.value.col_offset))
            if isinstance(n, ast.Call):
                for a_ in n.args:
                    if isinstance(a_, ast.Name):
                        arg_nodes.add((a_.lineno, a_.col_offset))
        lines = code.split('\n')
        cands_all = []
        seen = Counter()
        for k, nd in enumerate(nodes):
            key = (nd.lineno, nd.col_offset)
            occ_before = seen[nd.id]
            seen[nd.id] += 1
            if nd.id not in names or occ_before == 0:
                continue
            if kind == 'retsel' and key not in ret_nodes:
                continue
            if kind == 'argsel' and key not in arg_nodes:
                continue
            here = (nd.lineno, nd.col_offset)
            inscope = sorted([v for v, p in binds.items() if p < here])
            if nd.id not in inscope or len(inscope) < 2:
                continue
            st = _same_statement_text(tree, nd)
            seg = ast.get_source_segment(code, st) if st is not None else lines[nd.lineno - 1]
            if seg is None:
                seg = lines[nd.lineno - 1]
            if len(re.findall(r'\b%s\b' % re.escape(nd.id), seg)) > 1:
                drops['leak_same_statement'] += 1
                continue
            if len(re.findall(r'\b%s\b' % re.escape(nd.id), lines[nd.lineno - 1])) > 1:
                drops['leak_same_line'] += 1
                continue
            # most-recently-mentioned in-scope variable strictly before this site
            prev = [(m.lineno, m.col_offset, m.id) for m in ast.walk(tree)
                    if isinstance(m, ast.Name) and m.id in inscope and (m.lineno, m.col_offset) < here]
            recent = max(prev)[2] if prev else None
            cands_all.append({'k': k, 'true': nd.id, 'cands': inscope, 'recent': recent})
        if not cands_all:
            continue
        it = cands_all[random.Random(hash_seed('sel', seed, kind, i)).randrange(len(cands_all))]
        out.append({'src_index': i, 'code': code, **it})
        if len(out) >= cap:
            break
    meta = {'rule': build_selection_items.__doc__.strip(), 'kind': kind, 'drops': dict(drops),
            'n_items': len(out), 'mean_k': float(np.mean([len(o['cands']) for o in out])) if out else None,
            'k_distribution': dict(Counter(len(o['cands']) for o in out)),
            'baseline_uniform': float(np.mean([1.0 / len(o['cands']) for o in out])) if out else None,
            'baseline_most_recent': float(np.mean([o['recent'] == o['true'] for o in out])) if out else None}
    return out, meta


def selection_views(items, seed=BENCH_SEED, ns=CURVE_NS):
    """N-rename views: one mapping per (item, N) over the function's renameable locals; the true
    answer and every candidate are mapped through it, so the ANSWER is renaming-invariant."""
    import whyscale as WS
    out, drops = [], Counter()
    for it in items:
        got = WS._py_renameable(it['code'])
        names = got[0] if got else []
        views, ok = {0: (it['code'], it['true'], it['cands'], it['recent'])}, True
        for N in ns[1:]:
            rng = random.Random(hash_seed('selRen', seed, it['src_index'], N))
            picked = list(names)
            rng.shuffle(picked)
            picked = picked[:N]
            taken = set(WS.IDENT_RE.findall(it['code']))
            mp = {}
            for v in picked:
                nn, g = _rand_letters(rng, len(v)), 0
                while (nn in taken or nn in mp.values()) and g < 256:
                    nn = _rand_letters(rng, len(v))
                    g += 1
                mp[v] = nn
            new = WS._real6_apply(it['code'], mp) if mp else it['code']
            if mp and not WS._alpha_ok(it['code'], new, mp):
                drops['alpha_bad'] += 1
                ok = False
                break
            views[N] = (new, mp.get(it['true'], it['true']),
                        [mp.get(c, c) for c in it['cands']],
                        mp.get(it['recent'], it['recent']))
        if ok:
            out.append({**it, 'views': views,
                        'touched': {N: (it['true'] in
                                        {c for c, m in zip(it['cands'], views[N][2]) if c != m})
                                    for N in ns[1:]}})
    return out, dict(drops)


def score_selection(model, tok, recs, N, dev, batch, max_len=320):
    """-> bool array: does the TRUE variable win the likelihood comparison at view N?"""
    texts, owner = [], []
    for j, r in enumerate(recs):
        code, true, cands, _ = r['views'][N]
        for c in cands:
            t = substitute_var_at(code, r['k'], c)
            texts.append(t if t is not None else code)
            owner.append((j, c == true))
    s = seq_ll(model, tok, texts, dev, batch, max_len)
    best = {}
    for (j, is_true), v in zip(owner, s):
        if j not in best or v > best[j][0]:
            best[j] = (v, is_true)
    return np.array([best[j][1] for j in range(len(recs))])


def cmd_sel(a):
    """Identifier-selection tasks: N=0 gate, and the retention curve when asked for."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    srcs = real6_sources()
    tok, model = load_lm(a.model, dev)
    res = {'model': a.model, 'provenance': provenance(),
           'definition': 'one in-scope variable is blanked at a copyable use site; the candidate '
                         'substituted into the slot that maximises the whole-function length-'
                         'normalised log-likelihood wins; correct when that is the true variable'}
    for kind in (SEL_TASKS if a.task == 'all' else (a.task,)):
        items, meta = build_selection_items(srcs, kind, cap=a.n_items)
        recs, rdrops = selection_views(items)
        meta['rename_drops'] = rdrops
        hits = {0: score_selection(model, tok, recs, 0, dev, a.batch)}
        cell = {'meta': meta, 'n': len(recs),
                'N0_accuracy': wilson(int(hits[0].sum()), len(hits[0])),
                'baseline_uniform': meta['baseline_uniform'],
                'baseline_most_recent': meta['baseline_most_recent']}
        print('  %-7s N0 %.4f (n=%d) | uniform %.4f most-recent %.4f | mean k %.2f | drops %s'
              % (kind, cell['N0_accuracy']['rate'], len(recs), meta['baseline_uniform'],
                 meta['baseline_most_recent'], meta['mean_k'], meta['drops']), flush=True)
        if a.curve:
            base = hits[0]
            for N in CURVE_NS[1:]:
                hits[N] = score_selection(model, tok, recs, N, dev, a.batch)
                tch = np.array([r['touched'][N] for r in recs])
                cell['N%d' % N] = {
                    'accuracy': float(hits[N].mean()),
                    'retention': wilson(int((base & hits[N]).sum()), int(base.sum())),
                    'touched': {'retention': wilson(int((base & hits[N] & tch).sum()),
                                                    int((base & tch).sum())),
                                'frac_items': float(tch.mean())},
                    'untouched': {'retention': wilson(int((base & hits[N] & ~tch).sum()),
                                                      int((base & ~tch).sum()))}}
                print('    N=%d acc %.4f retention %.4f (touched %.4f n=%d | untouched %.4f n=%d)'
                      % (N, hits[N].mean(), cell['N%d' % N]['retention']['rate'],
                         cell['N%d' % N]['touched']['retention']['rate'] or 0,
                         cell['N%d' % N]['touched']['retention']['n'],
                         cell['N%d' % N]['untouched']['retention']['rate'] or 0,
                         cell['N%d' % N]['untouched']['retention']['n']), flush=True)
        res[kind] = cell
        if not (a.limit and not a.write_smoke):
            # one entry PER TASK, so running the tasks in separate invocations does not clobber
            key = 'selection_curves' if a.curve else 'selection'
            mk = os.path.basename(os.path.normpath(a.model))
            store([key, mk, kind], {**cell, 'model': a.model, 'provenance': res['provenance'],
                                    'definition': res['definition']}, a.out)
            print('wrote', a.out, [key, mk, kind], flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
    return 0




# ===========================================================================
# IDENTIFIER PRODUCTION: (A) reproduction of a copyable name, (B) continuation given piece 1
# ===========================================================================
def cmd_idcurve(a):
    """Teacher-forced production of a copyable identifier, and of its continuation, at N=0,1,2,4,8.

    A  reproduction: every BPE piece of the occurrence must be the argmax (exact name).
    B  continuation: the first piece is given (teacher forcing), every LATER piece must be the
       argmax; restricted to occurrences of two or more pieces in that view.
    Items, seeding and rename machinery are decmech `curves`'s, so these sit beside the other curves.
    """
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    recs, meta = curve_views(a)
    tok, model = load_lm(a.model, dev)
    per = {}
    for N in CURVE_NS:
        ids, offs, preds = lm_argmax(model, tok, [r['views'][N] for r in recs], dev, a.batch,
                                     a.max_len)
        for j, r in enumerate(recs):
            cls, occ_tok, es = token_layout(r['views'][N], offs[j], r['occ'][N])
            ok = np.zeros(len(ids[j]), dtype=bool)
            ok[1:] = np.asarray(preds[j][:-1]) == np.asarray(ids[j][1:])
            seen = set()
            for oi, (_, _, key) in enumerate(r['occ'][N]):
                cop = key in seen
                seen.add(key)
                info = occ_tok[oi]
                if not (cop and info['name_ok'] and info['cover'] and info['cover'][0] >= 1):
                    continue
                cov = info['cover']
                per.setdefault((j, oi), {})[N] = {
                    'all': bool(all(ok[k] for k in cov)),
                    'cont': bool(all(ok[k] for k in cov[1:])) if len(cov) > 1 else None,
                    'pieces': len(cov), 'n_ok': int(sum(int(ok[k]) for k in cov)),
                    'first_id': int(ids[j][cov[0]]),
                    'cont_ids': tuple(int(ids[j][k]) for k in cov[1:])}
    res = {'model': a.model, 'meta': meta, 'provenance': provenance(),
           'definition': cmd_idcurve.__doc__.strip()}
    # trivial baselines
    rec_hit, n_rec = 0, 0
    for r in recs:
        seen, last = set(), None
        for oi, (_, _, key) in enumerate(r['occ'][0]):
            if key in seen:
                n_rec += 1
                rec_hit += int(last == key)
            seen.add(key)
            last = key
    res['baseline_A_most_recent_variable'] = wilson(rec_hit, n_rec)
    for task, key in (('A_reproduction', 'all'), ('B_continuation', 'cont')):
        cell = {}
        for N in CURVE_NS:
            items = [(k, v[N]) for k, v in per.items() if N in v and v[N][key] is not None]
            hit = np.array([v[key] for _, v in items])
            cell['N%d_accuracy' % N] = wilson(int(hit.sum()), len(hit))
            piece_ok = sum(v['n_ok'] if key == 'all' else v['n_ok'] - 1 for _, v in items)
            piece_n = sum(v['pieces'] if key == 'all' else v['pieces'] - 1 for _, v in items)
            cell['N%d_per_piece' % N] = wilson(piece_ok, piece_n)
            by = defaultdict(lambda: [0, 0])
            for _, v in items:
                b = by[pieces_bin(v['pieces'])]
                b[0] += int(v[key]); b[1] += 1
            cell['N%d_by_pieces' % N] = {k: wilson(*v) for k, v in sorted(by.items())}
            if key == 'cont':   # only meaningful for the continuation task (>= 2 pieces)
                maj = defaultdict(Counter)
                for _, v in items:
                    maj[v['first_id']][v['cont_ids']] += 1
                cell['N%d_baseline_majority_continuation' % N] = wilson(
                    sum(c.most_common(1)[0][1] for c in maj.values()), len(items))
        for N in CURVE_NS[1:]:
            both = [(v[0][key], v[N][key]) for v in per.values()
                    if 0 in v and N in v and v[0][key] is not None and v[N][key] is not None]
            b0 = np.array([x for x, _ in both]); bN = np.array([y for _, y in both])
            cell['N%d_retention' % N] = wilson(int((b0 & bN).sum()), int(b0.sum()))
            cell['N%d_paired_n' % N] = len(both)
        res[task] = cell
        # PRINTING ONLY (added 2026-09-21 by [AGENT PKG]): `wilson` returns rate None when a cell
        # has no support, which a small --n can produce (B_continuation needs multi-piece
        # occurrences). '%.4f' % None then raises and loses the whole run over a log line. `_r`
        # prints nan for those; no stored number changes.
        def _r(d, k='rate'):
            v = (d or {}).get(k)
            return float('nan') if v is None else v
        print('  %-14s N0 acc %.4f | retention %s' % (
            task, _r(cell['N0_accuracy']),
            ' '.join('N%d %.4f(n=%d)' % (N, _r(cell['N%d_retention' % N]),
                                         cell['N%d_retention' % N]['n']) for N in CURVE_NS[1:])),
            flush=True)
        print('%18s per-piece N0 %.4f | %s | by pieces %s'
              % ('', _r(cell['N0_per_piece']),
                 ('majority-continuation baseline N0 %.4f'
                  % _r(cell['N0_baseline_majority_continuation']))
                 if 'N0_baseline_majority_continuation' in cell else 'baseline: see A row',
                 {k: round(_r(v), 3) for k, v in cell['N0_by_pieces'].items()}), flush=True)
    _b = res['baseline_A_most_recent_variable']
    print('  baseline A (most recent variable) %.4f (n=%d)'
          % (float('nan') if _b['rate'] is None else _b['rate'], _b['n']), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['idcurves', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['idcurves', os.path.basename(os.path.normpath(a.model))])
    return 0




def py150_examples(n=20, seed=BENCH_SEED):
    """Manual check of the lexical rename rule: what it touches and what it leaves alone."""
    recs, meta = py150_corpus(n, seed, ns=(8,))
    out = []
    for r in recs[:n]:
        base, new = r['toks'][0], r['toks'][8]
        ch = [(j, base[j], new[j]) for j in range(len(base)) if base[j] != new[j]]
        ctx = [{'token': b, 'became': v, 'prev': base[j - 1] if j else None,
                'next': base[j + 1] if j + 1 < len(base) else None} for j, b, v in ch[:4]]
        untouched_idents = [t for j, t in enumerate(base)
                            if re.match(r'^[A-Za-z_]\w*$', t) and base[j] == new[j]][:8]
        out.append({'file': r['i'], 'n_changed_tokens': len(ch), 'changed': ctx,
                    'untouched_identifier_tokens': untouched_idents})
    return out, meta


def cmd_py150(a):
    """CodeXGLUE token-level completion on PY150 with the renaming retention curve, plus the
    line-completion PROTOCOL (the official line split ships no gold: its `gt` field is empty)."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    recs, meta = py150_corpus(a.n_files, ns=CURVE_NS[1:])
    tok, model = load_lm(a.model, dev)
    IDRE = re.compile(r'^[A-Za-z_]\w*$')
    per, maj = {}, Counter()
    for N in CURVE_NS:
        texts, spans = [], []
        for r in recs:
            t, sp = py150_render(r['toks'][N])
            texts.append(t); spans.append(sp)
        ids, offs, preds = lm_argmax(model, tok, texts, dev, a.batch, a.max_len)
        for j, r in enumerate(recs):
            ok = np.zeros(len(ids[j]), dtype=bool)
            ok[1:] = np.asarray(preds[j][:-1]) == np.asarray(ids[j][1:])
            cover = defaultdict(list)
            for k in range(1, len(ids[j])):
                aa, bb = offs[j][k]
                if bb <= aa:
                    continue
                st = texts[j][aa:bb]
                es = aa + (len(st) - len(st.lstrip())) if st.strip() else bb
                for cj, sp in enumerate(spans[j]):
                    if sp and es < sp[1] and bb > sp[0]:
                        cover[cj].append(k)
                        break
            for cj, ks in cover.items():
                if cj < a.warmup or not ks or ks[0] < 1:
                    continue
                t_txt = r['toks'][N][cj]
                per.setdefault((j, cj), {})[N] = {
                    'exact': bool(all(ok[k] for k in ks)), 'first': bool(ok[ks[0]]),
                    'ident': bool(IDRE.match(t_txt)) and not _kwlike(t_txt),
                    'renameable': py150_renameable(r['toks'][0])[cj], 'tok': t_txt}
                if N == 0:
                    maj[t_txt] += 1
    res = {'model': a.model, 'meta': meta, 'provenance': provenance(),
           'benchmark': 'CodeXGLUE code_x_glue_cc_code_completion_token, python, TEST split',
           'metric': 'next-token accuracy at the CODE-TOKEN level (all BPE pieces exact); '
                     'first-piece accuracy reported as a secondary',
           'rename_rule': py150_renameable.__doc__.strip()}
    keys = [k for k, v in per.items() if all(N in v for N in CURVE_NS)]
    res['n_aligned_code_tokens'] = len(keys)
    base_tok = [per[k][0]['tok'] for k in keys]
    res['baseline_most_frequent_code_token'] = {
        'token': maj.most_common(1)[0][0] if maj else None,
        'rate': float(np.mean([t == maj.most_common(1)[0][0] for t in base_tok])) if maj else None}
    for slice_name, sel in (('all', lambda v: True), ('identifier', lambda v: v['ident']),
                            ('non_identifier', lambda v: not v['ident']),
                            ('renameable_identifier', lambda v: v['renameable'])):
        idx = [k for k in keys if sel(per[k][0])]
        cell = {'n': len(idx)}
        for N in CURVE_NS:
            e = np.array([per[k][N]['exact'] for k in idx])
            f = np.array([per[k][N]['first'] for k in idx])
            cell['N%d_accuracy' % N] = wilson(int(e.sum()), len(e))
            cell['N%d_first_piece' % N] = wilson(int(f.sum()), len(f))
        b0 = np.array([per[k][0]['exact'] for k in idx])
        for N in CURVE_NS[1:]:
            bN = np.array([per[k][N]['exact'] for k in idx])
            cell['N%d_retention' % N] = wilson(int((b0 & bN).sum()), int(b0.sum()))
        res[slice_name] = cell
        print('  %-22s n=%6d  N0 %.4f | ret %s' % (
            slice_name, len(idx), cell['N0_accuracy']['rate'],
            ' '.join('N%d %.4f' % (N, cell['N%d_retention' % N]['rate']) for N in CURVE_NS[1:])),
            flush=True)
    # line-completion PROTOCOL on the same files
    if not a.no_line:
        sub = list(range(0, len(recs), max(1, len(recs) // a.line_n)))[:a.line_n]
        match, esim = {}, {}
        for N in CURVE_NS:
            prompts, refs = [], []
            for j in sub:
                t, _ = py150_render(recs[j]['toks'][N])
                cut = _truncation_point(t, 0.50)
                if cut is None:
                    continue
                prompts.append(cut[0]); refs.append(cut[1])
            gen = greedy_lines(model, tok, prompts, dev, a.gen_batch)
            match[N] = np.array([g == r.strip() for g, r in zip(gen, refs)])
            esim[N] = np.array([edit_sim(g, r.strip()) for g, r in zip(gen, refs)])
        line = {'n': int(len(match[0])), 'protocol': 'the official line split ships only `train` '
                'with an EMPTY gt field, so this is the CodeXGLUE line-completion PROTOCOL applied '
                'to the token-level TEST files, not the official split',
                'truncation': 'first non-blank line at or after 50% of the characters; prompt '
                              'includes its indentation; greedy, 32 tokens, cut at the newline'}
        for N in CURVE_NS:
            line['N%d' % N] = {'exact_match': wilson(int(match[N].sum()), len(match[N])),
                               'edit_similarity': float(esim[N].mean())}
        for N in CURVE_NS[1:]:
            line['N%d' % N]['retention'] = wilson(int((match[0] & match[N]).sum()),
                                                  int(match[0].sum()))
        res['line_protocol'] = line
        print('  line protocol   n=%d  EM %.4f edit-sim %.4f | ret %s' % (
            line['n'], line['N0']['exact_match']['rate'], line['N0']['edit_similarity'],
            ' '.join('N%d %s' % (N, ('%.4f' % line['N%d' % N]['retention']['rate'])
                                 if line['N%d' % N]['retention']['rate'] is not None else '-')
                     for N in CURVE_NS[1:])), flush=True)
    if a.examples:
        res['rename_rule_examples'], _ = py150_examples()
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['py150', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['py150', os.path.basename(os.path.normpath(a.model))])
    return 0




# ===========================================================================
# CodeXGLUE ClozeTest (CT-all and CT-maxmin), Python portion
# ===========================================================================
def cloze_items(variant='maxmin', n_sample=3000, seed=BENCH_SEED, ns=CURVE_NS[1:]):
    """ClozeTest items with the GOLD RECOVERED from CodeSearchNet.

    The HF ClozeTest datasets ship id / idx / nl_tokens / pl_tokens and NO answer: the gold is
    withheld. The items are CodeSearchNet functions with one token masked, so the answer is
    recovered by matching the item's token list against CSN python (test then validation, then
    train if needed): a function of the same length whose tokens agree everywhere except at the
    mask index supplies the masked token. Items whose mask is at position < 2 or at the last
    position are not indexed (they cannot be keyed) and are counted.
    SAFE / UNSAFE: an item is SAFE for a renaming curve iff our lexical rename rule would NOT touch
    the masked token in its context (keyword, builtin, operator, literal, attribute after '.', call
    before '(', import name, kwarg name); otherwise renaming changes the correct answer and the item
    is excluded from the curve.
    """
    import whyscale as WS
    from datasets import load_dataset
    # The cache path carries n_sample. It used to be one file per variant, which meant asking for
    # a different n_sample REBUILT it in place and threw away the item set (and the CodeSearchNet
    # pass that produced it) belonging to the other size -- a smoke run at a small n_sample would
    # silently destroy a real run's items. Sizes now coexist. A smaller n_sample is NOT a prefix
    # of a larger one (the sampler strides through the benchmark, so the stride differs), so
    # serving a prefix the way _real6_corpus does would be wrong here; separate files is the fix.
    cache = os.path.join(WS.CACHE, 'cloze_%s_items_n%d.json' % (variant, n_sample))
    legacy = os.path.join(WS.CACHE, 'cloze_%s_items.json' % variant)   # pre-2026-09-21 layout
    for cand in (cache, legacy):
        if not os.path.exists(cand):
            continue
        d = json.load(open(cand))
        if d['n_sample'] == n_sample and d['seed'] == seed:
            # JSON has no integer keys: restore the view dict's int keys
            for it in d['items']:
                it['views'] = {int(k): v for k, v in it['views'].items()}
            print('cloze %s items from cache: %d (%s)'
                  % (variant, len(d['items']), os.path.basename(cand)), flush=True)
            return d['items'], d['meta']
    name = ('google/code_x_glue_cc_cloze_testing_%s'
            % ('all' if variant == 'all' else 'maxmin'))
    cz = load_dataset(name, 'python')['train']
    idx = list(range(0, len(cz), max(1, len(cz) // n_sample)))[:n_sample]
    sub = cz.select(idx)
    need, skipped = {}, 0
    for r in sub:
        t = list(r['pl_tokens'])
        m = t.index('<mask>')
        if m < 2 or m > len(t) - 2:
            skipped += 1
            continue
        need.setdefault((len(t), t[0], t[1], t[-1]), []).append((r['id'], m, tuple(t)))
    found = {}
    for split in ('test', 'validation', 'train'):
        if len(found) >= sum(len(v) for v in need.values()):
            break
        for r in load_dataset('code_search_net', 'python', split=split, streaming=True):
            tk = r['func_code_tokens']
            k = (len(tk), tk[0], tk[1], tk[-1]) if len(tk) > 2 else None
            if k in need:
                for cid, m, ct in need[k]:
                    if cid in found:
                        continue
                    if all(x == y for i, (x, y) in enumerate(zip(ct, tk)) if i != m):
                        found[cid] = tk[m]
    items, unsafe = [], 0
    for key, lst in need.items():
        for cid, m, ct in lst:
            if cid not in found:
                continue
            toks = list(ct)
            toks[m] = found[cid]
            ok = py150_renameable(toks)
            safe = not ok[m]
            if not safe:
                unsafe += 1
            views = {0: toks}
            taken = set(toks)
            names = sorted({t for t, o in zip(toks, ok) if o})
            good = True
            for N in ns:
                rng = random.Random(hash_seed('cloze', seed, variant, cid, N))
                picked = list(names)
                rng.shuffle(picked)
                picked = picked[:N]
                mp = {}
                for v in picked:
                    nn, g = _rand_letters(rng, len(v)), 0
                    while (nn in taken or nn in mp.values()) and g < 256:
                        nn = _rand_letters(rng, len(v))
                        g += 1
                    mp[v] = nn
                views[N] = [mp.get(t, t) if o else t for t, o in zip(toks, ok)]
                if views[N][m] != toks[m]:
                    good = False          # the answer itself moved: unsafe by construction
            if not good:
                continue
            items.append({'id': cid, 'mask': m, 'gold': found[cid], 'safe': bool(safe),
                          'views': views, 'n_names': len(names)})
    meta = {'benchmark': name, 'config': 'python', 'split': 'train (the only split HF ships)',
            'n_dataset': len(cz), 'n_sampled': len(idx), 'skipped_unindexable': skipped,
            'recovered': len(found), 'n_items': len(items),
            'n_safe': int(sum(it['safe'] for it in items)),
            'frac_safe': float(np.mean([it['safe'] for it in items])) if items else None,
            'unsafe_answer_is_a_renameable_local': unsafe,
            'gold_recovery': cloze_items.__doc__.strip(),
            'distinct_answers': len({it['gold'] for it in items}),
            'majority_answer': Counter(it['gold'] for it in items).most_common(1)[0] if items else None}
    json.dump({'n_sample': n_sample, 'seed': seed, 'items': items, 'meta': meta}, open(cache, 'w'))
    print('cloze %s: %d items (%d safe) -> %s' % (variant, len(items), meta['n_safe'], cache),
          flush=True)
    return items, meta


CLOZE_KINDS = ('identifier', 'builtin', 'keyword', 'other')
CLOZE_KIND_RULE = ("gold token kind, in this order: keyword.iskeyword(g) -> 'keyword' (hard "
                   "keywords only; soft keywords such as 'match'/'type' are NOT keywords here); "
                   "else hasattr(builtins, g) -> 'builtin'; else g starts with a letter or '_' -> "
                   "'identifier' (attribute names, called function names, import names); else "
                   "'other' (punctuation / operator / literal)")


def cloze_gold_kind(g):
    """Kind of the masked gold token. See CLOZE_KIND_RULE."""
    import builtins as _bi
    import keyword as _kw
    if _kw.iskeyword(g):
        return 'keyword'
    if hasattr(_bi, g):
        return 'builtin'
    if g[:1].isalpha() or g[:1] == '_':
        return 'identifier'
    return 'other'


def cmd_cloze(a):
    """ClozeTest accuracy and the renaming retention curve on the SAFE subset."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    items, meta = cloze_items(a.variant, a.n_sample)
    tok, model = load_lm(a.model, dev)
    cands = sorted({it['gold'] for it in items})
    res = {'model': a.model, 'meta': meta, 'provenance': provenance(), 'variant': a.variant,
           'candidates': len(cands),
           'protocol': ('full-sequence length-normalised log-likelihood with each candidate '
                        'substituted at the mask (2 candidates)') if a.variant == 'maxmin' else
                       ('prefix-only ranking: one forward per item, candidates ranked by the '
                        'log-probability of their first BPE piece at the mask slot; restricted to '
                        'candidates that are a single BPE token'),
           'protocol_note': 'a causal LM ranks from the LEFT context; the benchmark baselines are '
                            'bidirectional encoders, so absolute numbers are not comparable with '
                            'the leaderboard'}
    safe = [it for it in items if it['safe']]
    hits = {}
    for N in CURVE_NS:
        if a.variant == 'maxmin':
            texts, owner = [], []
            for j, it in enumerate(items):
                for c in ('max', 'min'):
                    t = list(it['views'][N])
                    t[it['mask']] = c
                    texts.append(py150_render(t)[0])
                    owner.append((j, c))
            s = seq_ll(model, tok, texts, dev, a.batch, a.max_len)
            best = {}
            for (j, c), v in zip(owner, s):
                if j not in best or v > best[j][0]:
                    best[j] = (v, c)
            hits[N] = np.array([best[j][1] == items[j]['gold'] for j in range(len(items))])
        else:
            cid = {}
            for c in cands:
                p = tok(' ' + c, add_special_tokens=False)['input_ids']
                if len(p) == 1:
                    cid[c] = p[0]
            res['candidates_single_token'] = len(cid)
            texts, spans = [], []
            for it in items:
                t, sp = py150_render(it['views'][N])
                texts.append(t); spans.append(sp[it['mask']])
            enc = tok(texts, truncation=True, max_length=a.max_len, return_offsets_mapping=True,
                      add_special_tokens=False)
            hit = []
            keys = list(cid); ids_t = torch.tensor([cid[c] for c in keys])
            for s0 in range(0, len(texts), a.batch):
                chunk = list(range(s0, min(s0 + a.batch, len(texts))))
                T = max(len(enc['input_ids'][i]) for i in chunk)
                inp = torch.full((len(chunk), T), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(chunk), T), dtype=torch.long)
                for j, i in enumerate(chunk):
                    ii = enc['input_ids'][i]
                    inp[j, :len(ii)] = torch.tensor(ii); att[j, :len(ii)] = 1
                with torch.no_grad():
                    lg = model(input_ids=inp.to(dev), attention_mask=att.to(dev)
                               ).logits.float().log_softmax(-1).cpu()
                for j, i in enumerate(chunk):
                    sp = spans[i]
                    pos = next((k for k, (aa, bb) in enumerate(enc['offset_mapping'][i])
                                if bb > aa and aa <= sp[0] < bb), None)
                    if pos is None or pos < 1:
                        hit.append(False)
                        continue
                    sc = lg[j, pos - 1, ids_t]
                    hit.append(keys[int(sc.argmax())] == items[i]['gold'])
            hits[N] = np.array(hit)
    base = hits[0]
    sm = np.array([it['safe'] for it in items])
    res['N0_accuracy_all_items'] = wilson(int(base.sum()), len(base))
    res['N0_accuracy_safe'] = wilson(int((base & sm).sum()), int(sm.sum()))
    res['baseline_most_frequent_answer'] = float(np.mean(
        [it['gold'] == meta['majority_answer'][0] for it in items]))
    for N in CURVE_NS[1:]:
        res['N%d' % N] = {
            'accuracy_safe': float(hits[N][sm].mean()),
            'retention_safe': wilson(int((base & hits[N] & sm).sum()), int((base & sm).sum()))}
    print('  cloze-%-6s N0 all %.4f | N0 safe %.4f (n=%d) | baseline %.4f | ret %s'
          % (a.variant, res['N0_accuracy_all_items']['rate'], res['N0_accuracy_safe']['rate'],
             int(sm.sum()), res['baseline_most_frequent_answer'],
             ' '.join('N%d %.4f' % (N, res['N%d' % N]['retention_safe']['rate'])
                      for N in CURVE_NS[1:])), flush=True)
    if getattr(a, 'per_item', False):
        # SAME item set, SAME safety filter, SAME views, SAME scoring: only extra bookkeeping.
        kinds = [cloze_gold_kind(it['gold']) for it in items]
        karr = np.array(kinds)
        res['stratification'] = CLOZE_KIND_RULE
        res['curve_ns'] = list(CURVE_NS)
        strata = {}
        for kd in CLOZE_KINDS:
            mk = sm & (karr == kd)
            b = base & mk
            s = {'n_safe_items': int(mk.sum()),
                 'N0_accuracy': wilson(int(b.sum()), int(mk.sum()))}
            for N in CURVE_NS[1:]:
                s['N%d' % N] = {
                    'accuracy': wilson(int((hits[N] & mk).sum()), int(mk.sum())),
                    'retention': wilson(int((base & hits[N] & mk).sum()), int(b.sum()))}
            strata[kd] = s
        res['strata'] = strata
        res['per_item'] = [{'id': it['id'], 'gold': it['gold'], 'kind': kinds[i],
                            'hits': ''.join('1' if hits[N][i] else '0' for N in CURVE_NS)}
                           for i, it in enumerate(items) if it['safe']]
        for kd in CLOZE_KINDS:
            s = strata[kd]
            print('    %-10s n=%-5d N0 %s | ret %s' % (
                kd, s['n_safe_items'],
                ('%.4f' % s['N0_accuracy']['rate']) if s['n_safe_items'] else '   n/a',
                ' '.join('N%d %s' % (N, ('%.4f' % s['N%d' % N]['retention']['rate'])
                                      if s['N%d' % N]['retention']['rate'] is not None else 'n/a')
                         for N in CURVE_NS[1:])), flush=True)
        if a.limit and not a.write_smoke:
            print('smoke run: not written')
            return 0
        key = ['cloze_%s_bykind' % a.variant, os.path.basename(os.path.normpath(a.model))]
        store(key, res, a.out)
        print('wrote', a.out, key)
        return 0
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['cloze_' + a.variant, os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['cloze_' + a.variant, os.path.basename(os.path.normpath(a.model))])
    return 0




# ===========================================================================
# PROJECT CODENET (Python): problem-based retrieval, the POJ-104 protocol
# ===========================================================================
def codenet_items(n_problems=100, per_problem=5, seed=BENCH_SEED, ns=CURVE_NS[1:],
                  max_scan=120000):
    """Accepted Python-3 CodeNet submissions grouped by problem, with N-rename views.

    Two submissions to the same problem are semantic clones, which is the POJ-104 setting. Filters:
    Accepted, original_language Python3, 150 < len(code) < 1500, parses, at least two renameable
    locals (so a rename is meaningful). The first `n_problems` problems that reach `per_problem`
    usable submissions are kept. Renaming: whyscale._real6_apply with random same-length names,
    certified by _alpha_ok, seeded per (submission, N).
    """
    import whyscale as WS
    from datasets import load_dataset
    cache = os.path.join(WS.CACHE, 'codenet_items.json')
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d['n_problems'] == n_problems and d['per_problem'] == per_problem and d['seed'] == seed:
            for it in d['items']:
                it['views'] = {int(k): v for k, v in it['views'].items()}
            print('codenet items from cache: %d' % len(d['items']), flush=True)
            return d['items'], d['meta']
    ds = load_dataset('iNeil77/CodeNet', 'Python', split='train', streaming=True)
    by, scanned, drops = defaultdict(list), 0, Counter()
    for r in ds:
        scanned += 1
        if scanned > max_scan or len([p for p, v in by.items() if len(v) >= per_problem]) >= n_problems:
            break
        if r['status'] != 'Accepted' or r['original_language'] != 'Python3':
            continue
        c = r['code']
        if not (150 < len(c) < 1500):
            continue
        try:
            c = ast.unparse(ast.parse(c))     # canonical, like every other corpus we rename
            ast.parse(c)
        except Exception:
            drops['unparseable'] += 1
            continue
        got = WS._py_renameable(c)
        if not got or len(got[0]) < 2:
            drops['too_few_locals'] += 1
            continue
        if len(by[r['p_id']]) < per_problem:
            by[r['p_id']].append({'s_id': r['s_id'], 'code': c, 'names': got[0]})
    probs = [p for p, v in by.items() if len(v) >= per_problem][:n_problems]
    items = []
    for p in probs:
        for sub in by[p][:per_problem]:
            views, ok = {0: sub['code']}, True
            for N in ns:
                rng = random.Random(hash_seed('codenet', seed, sub['s_id'], N))
                picked = list(sub['names'])
                rng.shuffle(picked)
                picked = picked[:N]
                taken = set(WS.IDENT_RE.findall(sub['code']))
                mp = {}
                for v in picked:
                    nn, g = _rand_letters(rng, len(v)), 0
                    while (nn in taken or nn in mp.values()) and g < 256:
                        nn = _rand_letters(rng, len(v))
                        g += 1
                    mp[v] = nn
                new = WS._real6_apply(sub['code'], mp) if mp else sub['code']
                if mp and not WS._alpha_ok(sub['code'], new, mp):
                    drops['rename_failed'] += 1
                    ok = False
                    break
                views[N] = new
            if ok:
                items.append({'p_id': p, 's_id': sub['s_id'], 'views': views,
                              'n_names': len(sub['names'])})
    meta = {'dataset': 'iNeil77/CodeNet (Project CodeNet mirror), config Python, 568 MB in 4 parquet '
                       'shards, streamed', 'scanned': scanned, 'n_problems': len(probs),
            'per_problem': per_problem, 'n_items': len(items), 'drops': dict(drops),
            'filters': codenet_items.__doc__.strip(),
            'mean_renameable_locals': float(np.mean([it['n_names'] for it in items]))}
    json.dump({'n_problems': n_problems, 'per_problem': per_problem, 'seed': seed,
               'items': items, 'meta': meta}, open(cache, 'w'))
    print('codenet: %d items over %d problems (scanned %d) -> %s'
          % (len(items), len(probs), scanned, cache), flush=True)
    return items, meta


def _map_at_r(sim, labels, exclude_self=True):
    """MAP@R (the POJ-104 metric): for each query, R = number of other items with its label."""
    n = len(labels)
    out = []
    top1 = np.zeros(n, dtype=bool)
    for i in range(n):
        s = sim[i].copy()
        if exclude_self:
            s[i] = -np.inf
        order = np.argsort(-s)
        rel = np.array([labels[j] == labels[i] for j in order])
        R = int((np.array(labels) == labels[i]).sum() - (1 if exclude_self else 0))
        if R <= 0:
            continue
        hits = rel[:R]
        prec = np.cumsum(hits) / np.arange(1, R + 1)
        out.append(float((prec * hits).sum() / R))
        top1[i] = bool(rel[0])
    return float(np.mean(out)), top1


def cmd_codenet(a):
    """CodeNet problem-based retrieval (POJ-104 protocol) with the renaming retention curve."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    import whyscale as WS
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = str(device_of(a.device))
    items, meta = codenet_items(a.n_problems, a.per_problem)
    labels = [it['p_id'] for it in items]
    tok, enc = WS.load_encoder(a.model, dev)
    blank = [set() for _ in items]
    E0, _ = WS.embed(tok, enc, [it['views'][0] for it in items], blank, dev, a.batch,
                     normalize=False)
    U0 = WS.unit(E0)
    res = {'model': a.model, 'meta': meta, 'provenance': provenance(),
           'protocol': 'query = a RENAMED submission, pool = all ORIGINAL submissions; the query own '
                       'original is excluded, so a hit is another submission to the SAME PROBLEM '
                       '(the POJ-104 formulation, which is the published one). Self-retrieval '
                       '(our task 1 formulation) is reported separately.',
           'metric': 'MAP@R (POJ-104) and top-1; retention is defined on the top-1 bit'}
    # length baseline (no model): nearest by |len difference|
    L = np.array([len(it['views'][0]) for it in items], dtype=float)
    dl = -np.abs(L[:, None] - L[None, :])
    m_len, t_len = _map_at_r(dl, labels)
    res['baseline_length_nearest'] = {'map_at_r': m_len, 'top1': float(t_len.mean())}
    res['baseline_chance_top1'] = (a.per_problem - 1) / (len(items) - 1)
    hits = {}
    for N in CURVE_NS:
        P, _ = WS.embed(tok, enc, [it['views'][N] for it in items], blank, dev, a.batch,
                        normalize=False)
        UP = WS.unit(P)
        sim = UP @ U0.T
        np.fill_diagonal(sim, -np.inf)          # exclude the query's own original
        mapr, top1 = _map_at_r(sim, labels, exclude_self=False)
        self_sim = UP @ U0.T
        self_top1 = np.array([int(np.argmax(self_sim[i])) == i for i in range(len(items))])
        hits[N] = top1
        res['N%d' % N] = {'map_at_r': mapr, 'top1_same_problem': float(top1.mean()),
                          'top1_self_retrieval': float(self_top1.mean())}
        print('  codenet N=%d  MAP@R %.4f | top1 same-problem %.4f | self-retrieval %.4f'
              % (N, mapr, top1.mean(), self_top1.mean()), flush=True)
    base = hits[0]
    for N in CURVE_NS[1:]:
        res['N%d' % N]['retention_top1'] = wilson(int((base & hits[N]).sum()), int(base.sum()))
        res['N%d' % N]['map_ratio'] = res['N%d' % N]['map_at_r'] / max(1e-9, res['N0']['map_at_r'])
    print('  retention(top1) %s | MAP ratios %s'
          % (' '.join('N%d %.4f' % (N, res['N%d' % N]['retention_top1']['rate'])
                      for N in CURVE_NS[1:]),
             ' '.join('N%d %.3f' % (N, res['N%d' % N]['map_ratio']) for N in CURVE_NS[1:])),
          flush=True)
    del enc
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['codenet', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['codenet', os.path.basename(os.path.normpath(a.model))])
    return 0

# ===========================================================================
# VARIABLE-MISUSE LOCALIZATION AND REPAIR (the CuBERT pointer task), zero-shot
# ===========================================================================
VMLOC_MAX_EDITS = 150
VMLOC_TOK_MARGIN = 24


def vmloc_items(srcs, tok, seed=BENCH_SEED, cap=400, max_len=320, max_edits=VMLOC_MAX_EDITS,
                min_sites=2, ns=CURVE_NS):
    """CuBERT-style variable-misuse LOCALIZATION + REPAIR items on the real6 corpus.

    ITEM. One Load occurrence of a renameable local that is BOUND EARLIER in the text and has at
    least one other in-scope variable available is replaced by one of those other in-scope
    variables (seeded). The model sees only the corrupted function and must point at WHICH
    occurrence is wrong (localization) and at which variable repairs it (repair).
    SEARCH SPACE. Every Load occurrence of a renameable local that is bound earlier and has >= 2
    in-scope variables is a candidate POSITION; at each position the candidates are the in-scope
    variables other than the one currently written there. The true answer is
    (bug position, original variable); the identity edit is never a candidate, so the buggy program
    itself is not in the search space and "the edit that most raises the likelihood" is exactly
    the argmax over this set.
    RENAMING-INVARIANCE. Views are built by `selection_views`, the existing builder (same seed
    namespace `selRen`, same mapping construction, `whyscale._alpha_ok`-certified), so the renamed
    programs are byte-identical to the ones the `sel` tasks score. `cands` is set to ALL renameable
    locals of the function, so `views[N][2]` hands back the full mapping. Four things are then
    checked BYTE-EXACTLY at every N and any failure drops the item and is counted:
      1. the renamed function has the same Load-Name sequence, position for position, with every
         id mapped through the mapping  -> the bug POSITION index is invariant;
      2. corrupting the renamed function at that position with the mapped wrong variable equals
         applying the mapping to the corrupted original -> the BUG is the alpha-renamed bug;
      3. repairing the renamed corrupted function at that position with the mapped original equals
         the renamed original -> the REPAIR variable is invariant;
      4. the mapping is injective on every position's in-scope set -> the candidate sets
         correspond one-to-one across views.
    TRUNCATION. Items are dropped unless every view fits in `max_len` tokens with `VMLOC_TOK_MARGIN`
    to spare, so no candidate edit can fall outside the scored window in any view.
    """
    import whyscale as WS
    import varmisuse as VM
    drops = Counter()
    raw = []
    for i, src in enumerate(srcs):
        try:
            code = ast.unparse(ast.parse(src))
        except SyntaxError:
            drops['unparseable'] += 1
            continue
        got = WS._py_renameable(code)
        if not got or len(got[0]) < 2:
            drops['too_few_locals'] += 1
            continue
        names = set(got[0])
        tree = ast.parse(code)
        binds = _binding_positions(tree, names)
        nodes = VM._load_names(code)
        sites = []
        for k, nd in enumerate(nodes):
            if nd.id not in names:
                continue
            here = (nd.lineno, nd.col_offset)
            ins = sorted(v for v, p in binds.items() if p < here)
            if nd.id not in ins or len(ins) < 2:
                continue
            sites.append({'k': k, 'id': nd.id, 'ins': ins})
        if len(sites) < min_sites:
            drops['too_few_sites'] += 1
            continue
        n_edits = sum(len(s['ins']) - 1 for s in sites)
        if n_edits > max_edits:
            drops['too_many_edits'] += 1
            continue
        rng = random.Random(hash_seed('vmloc', seed, i))
        bs = sites[rng.randrange(len(sites))]
        alts = [v for v in bs['ins'] if v != bs['id']]
        wrong = alts[rng.randrange(len(alts))]
        bad0 = substitute_var_at(code, bs['k'], wrong)
        if bad0 is None or bad0 == code:
            drops['corruption_failed'] += 1
            continue
        bn = VM._load_names(bad0)
        if len(bn) != len(nodes) or any(
                bn[j].id != (wrong if j == bs['k'] else nodes[j].id) for j in range(len(nodes))):
            drops['corruption_moved_sites'] += 1
            continue
        raw.append({'src_index': i, 'code': code, 'true': bs['id'], 'recent': None,
                    'cands': sorted(names), 'k_bug': bs['k'], 'wrong': wrong,
                    'sites': sites, 'n_edits': n_edits, 'bad0': bad0})
    recs, rdrops = selection_views(raw, seed=seed, ns=ns)
    for k, v in rdrops.items():
        drops['view_' + k] += v
    # ---- renaming-invariance certification, byte-exact, at every N
    out, violations, vdetail = [], 0, Counter()
    for r in recs:
        import whyscale as WS2
        import varmisuse as VM2
        nodes0 = VM2._load_names(r['code'])
        ok = True
        for N in ns[1:]:
            code_N, _, mapped, _ = r['views'][N]
            mp = {a: b for a, b in zip(r['cands'], mapped) if a != b}
            nN = VM2._load_names(code_N)
            if len(nN) != len(nodes0) or any(
                    nN[j].id != mp.get(nodes0[j].id, nodes0[j].id) for j in range(len(nodes0))):
                vdetail['load_sequence_moved'] += 1
                ok = False
                break
            bad_N = substitute_var_at(code_N, r['k_bug'], mp.get(r['wrong'], r['wrong']))
            expect = WS2._real6_apply(r['bad0'], mp) if mp else r['bad0']
            if bad_N is None or bad_N != expect:
                vdetail['bug_not_alpha_image'] += 1
                ok = False
                break
            if substitute_var_at(bad_N, r['k_bug'], mp.get(r['true'], r['true'])) != code_N:
                vdetail['repair_does_not_restore'] += 1
                ok = False
                break
            if any(len({mp.get(c, c) for c in s['ins']}) != len(s['ins']) for s in r['sites']):
                vdetail['candidate_set_collapsed'] += 1
                ok = False
                break
        if not ok:
            violations += 1
            continue
        out.append(r)
    # ---- truncation filter (model tokenizer; identical for all three checkpoints)
    kept = []
    for r in out:
        lens = [len(tok(r['views'][N][0], add_special_tokens=False)['input_ids']) for N in ns]
        if max(lens) + VMLOC_TOK_MARGIN > max_len:
            drops['too_long_for_window'] += 1
            continue
        r['max_view_tokens'] = int(max(lens))
        kept.append(r)
    kept = kept[:cap]
    meta = {
        'definition': vmloc_items.__doc__.strip(),
        'corpus': 'whyscale real6 = CodeSearchNet Python TEST split, streamed in order, '
                  'ast.unparse canonical form (repro/whyscale.py:_real6_corpus, split="test")',
        'provenance_contamination': 'the decoders are trained on code_search_net python split='
                                    '"train" (repro/repair.py:624); the items come from split='
                                    '"test", so no item was in training',
        'n_sources_scanned': len(srcs), 'drops': dict(drops),
        'renaming_invariance_violations': int(violations),
        'renaming_invariance_violation_kinds': dict(vdetail),
        'n_items': len(kept), 'max_len': max_len, 'tok_margin': VMLOC_TOK_MARGIN,
        'ns': list(ns),
        'mean_sites': float(np.mean([len(r['sites']) for r in kept])) if kept else None,
        'mean_edits': float(np.mean([r['n_edits'] for r in kept])) if kept else None,
        'sites_distribution': dict(Counter(len(r['sites']) for r in kept)),
        'mean_view_tokens': float(np.mean([r['max_view_tokens'] for r in kept])) if kept else None,
        'items_hash': hashlib.md5(json.dumps(
            [(r['src_index'], r['k_bug'], r['true'], r['wrong'],
              [s['k'] for s in r['sites']]) for r in kept], sort_keys=True).encode()).hexdigest(),
    }
    if kept:
        meta['baseline_localization_uniform'] = float(np.mean([1.0 / len(r['sites']) for r in kept]))
        meta['baseline_joint_uniform'] = float(np.mean([1.0 / r['n_edits'] for r in kept]))
        # model-free heuristics, computed on the N=0 corrupted program
        loc_h, joint_h = [], []
        for r in kept:
            cur = {s['k']: (r['wrong'] if s['k'] == r['k_bug'] else s['id']) for s in r['sites']}
            freq = Counter(n.id for n in VM._load_names(r['bad0']))
            pick = min(r['sites'], key=lambda s: (freq[cur[s['k']]], s['k']))
            loc_h.append(pick['k'] == r['k_bug'])
            # repair heuristic: the most recently BOUND in-scope variable other than the one there
            alt = [v for v in pick['ins'] if v != cur[pick['k']]]
            binds = _binding_positions(ast.parse(r['code']), set(r['cands']))
            rep = max(alt, key=lambda v: binds.get(v, (0, 0))) if alt else None
            joint_h.append(pick['k'] == r['k_bug'] and rep == r['true'])
        meta['baseline_localization_rarest_name'] = float(np.mean(loc_h))
        meta['baseline_joint_rarest_plus_latest_binding'] = float(np.mean(joint_h))
    return kept, meta


def vmloc_score(model, tok, recs, N, dev, batch, max_len=320):
    """-> dict of bool arrays: does the argmax edit localize / localize+repair correctly at view N?

    One likelihood per candidate edit. Reported under both scorings that this project uses: the
    length-normalised log-likelihood (`mean`, what `sel`/`varmisuse` use) and the summed
    log-likelihood (`sum`). A tie at the maximum counts as WRONG and is counted.
    """
    texts, owner = [], []
    for j, r in enumerate(recs):
        code_N, _, mapped, _ = r['views'][N]
        mp = {a: b for a, b in zip(r['cands'], mapped)}
        bad = substitute_var_at(code_N, r['k_bug'], mp.get(r['wrong'], r['wrong']))
        for s in r['sites']:
            k_id = r['wrong'] if s['k'] == r['k_bug'] else s['id']
            cur = mp.get(k_id, k_id)
            for v in s['ins']:
                vv = mp.get(v, v)
                if vv == cur:
                    continue
                t = substitute_var_at(bad, s['k'], vv)
                if t is None:
                    continue
                texts.append(t)
                owner.append((j, s['k'], v))
    mean = seq_ll(model, tok, texts, dev, batch, max_len)
    ntok = np.array([len(tok(t, truncation=True, max_length=max_len,
                             add_special_tokens=False)['input_ids']) for t in texts])
    total = mean * np.maximum(ntok - 1, 1)
    res = {}
    for how, sc in (('mean', mean), ('sum', total)):
        best = {}
        for (j, k, v), val in zip(owner, sc):
            b = best.get(j)
            if b is None or val > b[0] + 1e-12:
                best[j] = (val, k, v, 1)
            elif abs(val - b[0]) <= 1e-12:
                best[j] = (b[0], b[1], b[2], b[3] + 1)
        loc, joint, ties, miss = [], [], 0, 0
        for j, r in enumerate(recs):
            b = best.get(j)
            if b is None:
                miss += 1
                loc.append(False)
                joint.append(False)
                continue
            uniq = b[3] == 1
            ties += (not uniq)
            loc.append(bool(uniq and b[1] == r['k_bug']))
            joint.append(bool(uniq and b[1] == r['k_bug'] and b[2] == r['true']))
        res[how] = {'loc': np.array(loc), 'joint': np.array(joint), 'ties': int(ties),
                    'no_candidate': int(miss)}
    res['n_texts'] = len(texts)
    return res


def cmd_vmloc(a):
    """Variable-misuse localization and repair: N=0 accuracy vs baselines, and the retention curve."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise SystemExit('set PYTHONHASHSEED=0')
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    tok, model = load_lm(a.model, dev)
    srcs = real6_sources(a.n)
    recs, meta = vmloc_items(srcs, tok, cap=(a.limit or a.n_items), max_len=a.max_len)
    print('vmloc: %d items | sites %.2f edits %.2f | invariance violations %d | drops %s'
          % (len(recs), meta['mean_sites'], meta['mean_edits'],
             meta['renaming_invariance_violations'], meta['drops']), flush=True)
    res = {'model': a.model, 'provenance': provenance(), 'meta': meta,
           'task': 'CuBERT variable-misuse LOCALIZATION and REPAIR (Kanade et al., ICML 2020), '
                   'instantiated zero-shot on held-out CodeSearchNet Python test functions',
           'protocol': 'the corrupted function is edited at every (position, in-scope variable) '
                       'candidate and the edit with the highest whole-function log-likelihood '
                       'wins; localization is correct when its position is the bug position, '
                       'joint localization+repair when the variable is also the original one; '
                       'ties at the maximum count as wrong',
           'tokenizer_md5_note': 'the three checkpoints share one tokenizer.json, so items_hash '
                                 'is comparable across models'}
    hits = {0: vmloc_score(model, tok, recs, 0, dev, a.batch, a.max_len)}
    res['n_scored_texts_per_view'] = hits[0]['n_texts']
    for how in ('mean', 'sum'):
        cell = {
            'N0_localization': wilson(int(hits[0][how]['loc'].sum()), len(recs)),
            'N0_joint_localization_repair': wilson(int(hits[0][how]['joint'].sum()), len(recs)),
            'N0_ties': hits[0][how]['ties'],
            'baseline_localization_uniform': meta.get('baseline_localization_uniform'),
            'baseline_localization_rarest_name': meta.get('baseline_localization_rarest_name'),
            'baseline_joint_uniform': meta.get('baseline_joint_uniform'),
            'baseline_joint_rarest_plus_latest_binding':
                meta.get('baseline_joint_rarest_plus_latest_binding')}
        res[how] = cell
        print('  [%s] N0 loc %.4f (uniform %.4f, rarest-name %.4f) | joint %.4f '
              '(uniform %.4f, heuristic %.4f) | ties %d'
              % (how, cell['N0_localization']['rate'], cell['baseline_localization_uniform'],
                 cell['baseline_localization_rarest_name'],
                 cell['N0_joint_localization_repair']['rate'], cell['baseline_joint_uniform'],
                 cell['baseline_joint_rarest_plus_latest_binding'], cell['N0_ties']), flush=True)
    for N in CURVE_NS[1:]:
        hits[N] = vmloc_score(model, tok, recs, N, dev, a.batch, a.max_len)
        tch = np.array([r['touched'][N] for r in recs])
        for how in ('mean', 'sum'):
            b_l, b_j = hits[0][how]['loc'], hits[0][how]['joint']
            c_l, c_j = hits[N][how]['loc'], hits[N][how]['joint']
            res[how]['N%d' % N] = {
                'localization_accuracy': float(c_l.mean()),
                'joint_accuracy': float(c_j.mean()),
                'retention_localization': wilson(int((b_l & c_l).sum()), int(b_l.sum())),
                'retention_joint': wilson(int((b_j & c_j).sum()), int(b_j.sum())),
                'retention_joint_repair_var_renamed':
                    wilson(int((b_j & c_j & tch).sum()), int((b_j & tch).sum())),
                'retention_joint_repair_var_untouched':
                    wilson(int((b_j & c_j & ~tch).sum()), int((b_j & ~tch).sum())),
                'frac_repair_var_renamed': float(tch.mean()),
                'ties': hits[N][how]['ties']}
        m = res['mean']['N%d' % N]
        print('    N=%d [mean] loc acc %.4f ret %.4f | joint acc %.4f ret %.4f (n=%d)'
              % (N, m['localization_accuracy'], m['retention_localization']['rate'],
                 m['joint_accuracy'], m['retention_joint']['rate'],
                 m['retention_joint']['n']), flush=True)
    if a.limit and not a.write_smoke:
        print('smoke run: not written')
        return 0
    store(['varmisuse_locrepair', os.path.basename(os.path.normpath(a.model))], res, a.out)
    print('wrote', a.out, ['varmisuse_locrepair', os.path.basename(os.path.normpath(a.model))])
    return 0


if __name__ == '__main__':
    sys.exit(main())
