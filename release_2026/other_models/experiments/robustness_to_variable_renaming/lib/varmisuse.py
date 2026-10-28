#!/usr/bin/env python3
"""VARIABLE MISUSE detection under variable renaming, for both model families.

QUESTION
  One occurrence of a variable is replaced by another variable of the same function. The model is
  correct when it gives the ORIGINAL function the higher likelihood (chance 0.5, zero-shot, no head).
  Renaming variables must not change that judgement, so the headline is retention: the fraction of
  pairs judged correctly at N = 0 that are still correct after N = 1, 4, 8 variables are renamed.
  CONTROL: the same protocol corrupting a binary OPERATOR instead. Renaming touches no operator, so
  a drop there means renaming broke the model generally, not its variable binding.

DESIGN registered in labnotes ("[AGENT ENC] ... VARIABLE-MISUSE design, fixed BEFORE any number").
Written fresh from the lead's pilot (scratchpad/varmisuse_pilot.py); it differs deliberately:
character-span AST corruption rather than line/column string surgery, pairs built ONCE and reused by
every arm and view, both kinds and all views in one pass, and the substitute drawn from the
function's RENAMEABLE locals so that a rename maps both names involved.

  items       whyscale real6 = CodeSearchNet Python TEST functions (seed 564), the same functions as
              every other number we report.
  corruption  `var`: the k-th Load-context ast.Name (seeded) replaced, by character span, with
              another renameable local of the same function; must parse and differ.
              `op` : one binary operator swapped on the AST (+<->-, *<->/, ==<->!=, <<->>).
  views       N in {0,1,4,8} renameable locals renamed to random same-length names
              (whyscale._rand_name + _real6_apply). The corrupted member is rebuilt by applying the
              SAME corruption to the renamed original (same node index, substitute's mapped name),
              and both members are certified with whyscale._alpha_ok against the same mapping, which
              is what proves the corruption relationship survived the rename unchanged.
  scoring     encoder: pseudo-log-likelihood, each token masked in turn, length-normalised. Full-
              sequence PLL is ~230 forwards per member, i.e. millions over items x views x members,
              so it is WINDOWED: the corrupted span's tokens plus --window tokens on each side
              (default 8), in each member's own tokenisation. The corrupted-span-only score is
              recorded too, from the same forwards. This is the usual MLM approximation and is
              applied identically to every arm.
              decoder: exact length-normalised conditional log-likelihood of the whole function,
              one forward per member. Exposed so [AGENT DEC] calls this same code.
  metric      accuracy (ties count as wrong and are counted), then retention_N; support < 30 -> None.

RUN
  export HF_HOME=/scratch/rb4792/hf_cache XDG_CACHE_HOME=/scratch/rb4792/xdg_cache PYTHONHASHSEED=0
  python varmisuse.py --models mech_enc_plain_s2,mech_enc_highdiv_s2 --device cuda
  python varmisuse.py --models <gptbigcode dir> --family decoder --device cuda
  python varmisuse.py --models mech_enc_plain_s2 --limit 100          # smoke, never writes
Results merge into tiny_scale/work/enc_mech.json under `varmisuse` -> family -> model.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault('WHYSCALE_BLOCK', '319')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import whyscale as W                                            # noqa: E402
import poolfix as P                                             # noqa: E402

OUT = os.path.join(W.WORK, 'enc_mech.json')
SEED = 564
NS = (1, 4, 8)
KINDS = ('var', 'op')
MIN_SUPPORT = 30
OP_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
           ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.Gt, ast.Gt: ast.Lt}


def _h(*p):
    return int(hashlib.md5('|'.join(str(x) for x in p).encode()).hexdigest()[:8], 16)


def _load_names(code):
    """Load-context Name nodes, in ast.walk order (stable under alpha-renaming)."""
    return [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)]


def corrupt_var_at(code, k, new):
    """Replace the k-th Load Name by character span with `new`. None if impossible."""
    nodes = _load_names(code)
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
    return out if out != code else None


def corrupt_op(code, rng):
    """Swap one binary operator on the AST and re-unparse. Returns (new_code, description)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    cands = []
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and type(n.op) in OP_SWAP:
            cands.append((n, 'op'))
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in OP_SWAP:
            cands.append((n, 'cmp'))
    if not cands:
        return None
    node, what = cands[rng.randrange(len(cands))]
    if what == 'op':
        old = type(node.op)
        node.op = OP_SWAP[old]()
    else:
        old = type(node.ops[0])
        node.ops = [OP_SWAP[old]()]
    try:
        out = ast.unparse(tree)
        ast.parse(out)
    except Exception:
        return None
    base = ast.unparse(ast.parse(code))
    if out == base:
        return None
    return out, old.__name__


def build_items(limit=None):
    """Pairs and rename views, model-independent, built once and reused by every arm."""
    recs, scanned = W._real6_corpus(W.REAL6_N)
    views, codes, nvars, audit = W._real6_build(recs, SEED)
    src2names = {}
    for r in recs:
        src2names.setdefault(r['src'], set(r['names']))
    items = {k: [] for k in KINDS}
    drops = {k: {'no_pair': 0, 'rename_failed': 0, 'alpha_bad': 0} for k in KINDS}
    for i, code in enumerate(codes):
        names = sorted(src2names[code])
        rng = random.Random(_h('vm', SEED, i))
        # ---------- var pair
        pair = None
        if len(names) >= 2:
            nodes = _load_names(code)
            elig = [k for k, n in enumerate(nodes) if n.id in names]
            rng.shuffle(elig)
            for k in elig:
                orig = nodes[k].id
                others = [n for n in names if n != orig]
                new = others[rng.randrange(len(others))]
                bad = corrupt_var_at(code, k, new)
                if bad is not None:
                    pair = {'k': k, 'orig': orig, 'new': new, 'bad': bad}
                    break
        if pair is None:
            drops['var']['no_pair'] += 1
        # ---------- op pair
        op = corrupt_op(code, random.Random(_h('vmop', SEED, i)))
        if op is None:
            drops['op']['no_pair'] += 1
        for kind, made in (('var', pair), ('op', op)):
            if made is None:
                continue
            it = {'i': i, 'names': names, 'good': {0: code}, 'nren': {0: 0}, 'kind': kind}
            if kind == 'var':
                it.update({'k': made['k'], 'orig': made['orig'], 'new': made['new']})
                it['bad'] = {0: made['bad']}
            else:
                it['bad'] = {0: made[0]}
                it['op_kind'] = made[1]
            ok = True
            it['touched'] = {}
            for N in NS:
                rr = random.Random(_h('ren', SEED, i, N))
                picked = list(names)
                rr.shuffle(picked)
                picked = picked[:N]
                taken = set(W.IDENT_RE.findall(code))
                mp = {}
                for v in picked:
                    nn, g = W._rand_name(rr, len(v)), 0
                    while (nn in taken or nn in mp.values()) and g < 256:
                        nn = W._rand_name(rr, len(v))
                        g += 1
                    mp[v] = nn
                g_ren = W._real6_apply(code, mp)
                if not W._alpha_ok(code, g_ren, mp):
                    drops[kind]['rename_failed'] += 1
                    ok = False
                    break
                if kind == 'var':
                    b_ren = corrupt_var_at(g_ren, it['k'], mp.get(it['new'], it['new']))
                    it['touched'][N] = (it['orig'] in mp) or (it['new'] in mp)
                else:
                    b_ren = W._real6_apply(it['bad'][0], mp)
                    it['touched'][N] = False
                if b_ren is None or b_ren == g_ren:
                    drops[kind]['rename_failed'] += 1
                    ok = False
                    break
                # the corruption relationship must be the SAME one, alpha-renamed
                if not W._alpha_ok(it['bad'][0], b_ren, mp):
                    drops[kind]['alpha_bad'] += 1
                    ok = False
                    break
                it['good'][N], it['bad'][N], it['nren'][N] = g_ren, b_ren, len(mp)
            if ok:
                items[kind].append(it)
    if limit:
        for k in KINDS:
            st = max(1, len(items[k]) // limit)
            items[k] = items[k][::st][:limit]
    meta = {'n_real6_codes': len(codes), 'scanned': scanned, 'seed': SEED, 'limit': limit,
            'n_items': {k: len(items[k]) for k in KINDS}, 'drops': drops,
            'mean_renamed': {k: {str(N): float(np.mean([it['nren'][N] for it in items[k]]))
                                 for N in (0,) + NS} for k in KINDS},
            'frac_rename_touched_the_pair': {
                str(N): float(np.mean([it['touched'][N] for it in items['var']])) for N in NS},
            'verification': 'both members certified by whyscale._alpha_ok against the same mapping; '
                            'the corrupted member of a renamed view is rebuilt by applying the same '
                            'corruption (same node index, mapped substitute) to the renamed original'}
    return items, meta


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

class EncScorer:
    """Windowed pseudo-log-likelihood: mask each scored position in turn."""

    def __init__(self, path, device, batch_size=64, window=8, amp=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, trim_offsets=False)
        self.m = AutoModelForMaskedLM.from_pretrained(path).float().eval().to(device)
        cap = self.m.config.max_position_embeddings - (self.m.config.pad_token_id or 0) - 2
        assert cap >= W.BLOCK_SIZE, 'block %d > capacity %d' % (W.BLOCK_SIZE, cap)
        self.dev, self.bs, self.win = device, batch_size, window
        # bf16 autocast for the forward only; the logits are upcast before log_softmax, so the
        # arithmetic that decides good-vs-bad stays fp32. Verified against fp32 on a GPU smoke run.
        self.amp = bool(amp) and str(device).startswith('cuda')

    def _positions(self, text, other):
        """Token ids plus the scored window: the tokens whose characters differ, +- window."""
        e = self.tok(text, truncation=True, max_length=W.BLOCK_SIZE,
                     return_offsets_mapping=True, return_special_tokens_mask=True)
        ids = np.asarray(e['input_ids'])
        offs = np.asarray(e['offset_mapping'])
        spec = np.asarray(e['special_tokens_mask'], dtype=bool)
        lo = 0
        while lo < min(len(text), len(other)) and text[lo] == other[lo]:
            lo += 1
        hi_t, hi_o = len(text), len(other)
        while hi_t > lo and hi_o > lo and text[hi_t - 1] == other[hi_o - 1]:
            hi_t -= 1
            hi_o -= 1
        core = np.flatnonzero((offs[:, 0] < max(hi_t, lo + 1)) & (lo < offs[:, 1]) & ~spec)
        if len(core) == 0:
            return ids, None, None
        a, b = int(core[0]), int(core[-1])
        s = max(1, a - self.win)
        e_ = min(len(ids) - 2, b + self.win)
        return ids, np.arange(s, e_ + 1), (a, b)

    def score_many(self, pairs):
        """[(text, other), ...] -> [(windowed PLL, corrupted-span-only PLL), ...].

        Every masked row of every text in `pairs` goes through ONE batched pass, which is the only
        difference from scoring them one at a time: the rows, the masks and the arithmetic are
        identical."""
        torch = self.torch
        plan, rows, owner = [], [], []
        for t, (text, other) in enumerate(pairs):
            ids, pos, core = self._positions(text, other)
            if pos is None or len(pos) == 0:
                plan.append(None)
                continue
            plan.append((ids, pos, core))
            for p in pos:
                r = ids.copy()
                r[p] = self.tok.mask_token_id
                rows.append(r)
                owner.append((t, int(p)))
        if not rows:
            return [(None, None)] * len(pairs)
        T = max(len(r) for r in rows)
        pad = self.tok.pad_token_id
        got = {}
        for s in range(0, len(rows), self.bs):
            chunk = rows[s:s + self.bs]
            bid = np.full((len(chunk), T), pad, dtype=np.int64)
            att = np.zeros((len(chunk), T), dtype=np.int64)
            for b, r in enumerate(chunk):
                bid[b, :len(r)] = r
                att[b, :len(r)] = 1
            # Only ONE position per row is ever read, so the masked position is gathered BEFORE
            # the softmax. Scoring the whole (batch, seq, vocab) tensor and copying it to the host
            # was ~7 GB per forward and made this 100x slower; the numbers are unchanged.
            pos_b = torch.from_numpy(np.array([owner[s + b][1] for b in range(len(chunk))])).to(
                self.dev)
            tgt_b = torch.from_numpy(np.array(
                [plan[owner[s + b][0]][0][owner[s + b][1]] for b in range(len(chunk))])).to(
                self.dev)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                 enabled=self.amp):
                lg = self.m(input_ids=torch.from_numpy(bid).to(self.dev),
                            attention_mask=torch.from_numpy(att).to(self.dev)).logits
            r = torch.arange(len(chunk), device=lg.device)
            sel = lg[r, pos_b].float()                       # (batch, vocab)
            lsel = torch.log_softmax(sel, -1)[r, tgt_b].cpu().numpy()
            for b in range(len(chunk)):
                t, p = owner[s + b]
                got[(t, p)] = float(lsel[b])
        out = []
        for t, pl in enumerate(plan):
            if pl is None:
                out.append((None, None))
                continue
            ids, pos, core = pl
            lp = np.array([got[(t, int(p))] for p in pos])
            in_core = (pos >= core[0]) & (pos <= core[1])
            out.append((float(lp.mean()),
                        float(lp[in_core].mean()) if in_core.any() else None))
        return out

    def score(self, text, other):
        return self.score_many([(text, other)])[0]


class DecScorer:
    """Exact length-normalised conditional log-likelihood of the whole function."""

    def __init__(self, path, device, batch_size=8, window=None, amp=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token
        self.m = AutoModelForCausalLM.from_pretrained(path).float().eval().to(device)
        self.dev = device

    def score_many(self, pairs):
        return [self.score(t, o) for t, o in pairs]

    def score(self, text, other=None):
        torch = self.torch
        ids = self.tok(text, truncation=True, max_length=W.BLOCK_SIZE,
                       return_tensors='pt')['input_ids'].to(self.dev)
        if ids.shape[1] < 2:
            return None, None
        with torch.no_grad():
            lg = self.m(input_ids=ids).logits[0].float().log_softmax(-1)
        t = ids[0]
        v = float(sum(lg[i, t[i + 1]].item() for i in range(ids.shape[1] - 1)) /
                  (ids.shape[1] - 1))
        return v, None


def run_model(path, items, family, device, batch_size, window, amp=True):
    sc = (EncScorer if family == 'encoder' else DecScorer)(path, device, batch_size, window, amp)
    out = {}
    for kind in KINDS:
        res = {N: {'correct': [], 'tie': 0, 'core': []} for N in (0,) + NS}
        for it in items[kind]:
            pairs = []
            for N in (0,) + NS:
                pairs.append((it['good'][N], it['bad'][N]))
                pairs.append((it['bad'][N], it['good'][N]))
            sc_all = sc.score_many(pairs)
            for j, N in enumerate((0,) + NS):
                (g, gc), (b, bc) = sc_all[2 * j], sc_all[2 * j + 1]
                if g is None or b is None:
                    res[N]['correct'].append(False)
                    continue
                res[N]['correct'].append(bool(g > b))
                res[N]['tie'] += int(g == b)
                if gc is not None and bc is not None:
                    res[N]['core'].append(bool(gc > bc))
        out[kind] = res
        print('  %s done' % kind, flush=True)
    return out


def summarize(out, items, meta, family, window):
    res = {'family': family, 'window_tokens': window if family == 'encoder' else None,
           'chance': 0.5, 'items_meta': meta, 'kinds': {}}
    for kind in KINDS:
        r = out[kind]
        c0 = np.array(r[0]['correct'])
        sup = int(c0.sum())
        d = {'n_items': len(c0), 'accuracy': float(c0.mean()), 'ties_N0': r[0]['tie'],
             'support_N0': sup,
             'accuracy_corrupted_span_only': (float(np.mean(r[0]['core']))
                                              if r[0]['core'] else None)}
        for N in NS:
            cn = np.array(r[N]['correct'])
            d['accuracy_N%d' % N] = float(cn.mean())
            d['retention_N%d' % N] = (float((c0 & cn).sum() / sup) if sup >= MIN_SUPPORT else None)
            d['retention_support_N%d' % N] = sup
            d['ties_N%d' % N] = r[N]['tie']
            if kind == 'var':
                t = np.array([it['touched'][N] for it in items['var']])
                s2 = int((c0 & t).sum())
                d['retention_N%d_pair_touched' % N] = (
                    float((c0 & cn & t).sum() / s2) if s2 >= MIN_SUPPORT else None)
                d['retention_N%d_pair_touched_support' % N] = s2
        res['kinds'][kind] = d
    return res


def merge(family, name, payload, out=None):
    fd = os.open(out or OUT, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        raw = f.read()
        d = json.loads(raw) if raw.strip() else {}
        d.setdefault('varmisuse', {}).setdefault(family, {})[name] = payload
        f.seek(0)
        f.truncate()
        json.dump(d, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    print('merged varmisuse/%s/%s' % (family, name), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', required=True,
                    help='comma-separated work/ dir names, codebert_base, or absolute paths')
    ap.add_argument('--family', default='encoder', choices=('encoder', 'decoder'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=160)
    ap.add_argument('--window', type=int, default=8,
                    help='encoder PLL window: tokens scored on each side of the corruption')
    ap.add_argument('--amp', default='bf16', choices=('bf16', 'off'),
                    help='bf16 autocast for the forward on CUDA (log-softmax stays fp32)')
    ap.add_argument('--limit', type=int, default=None, help='smoke test; never writes')
    ap.add_argument('--ns', default=None,
                    help='comma-separated rename counts to score instead of the default 1,4,8. '
                         'Added by [AGENT DEC] 2026-09-20 for the retention CURVE (needs N=2). '
                         'Each N draws its own seeded mapping, so the 1/4/8 numbers are unchanged.')
    ap.add_argument('--out', default=None,
                    help='write the merged document here instead of work/enc_mech.json. Added by '
                         '[AGENT DEC] 2026-09-20: five parallel decoder jobs lost 7 of 14 writes to '
                         'the cross-node flock race, so parallel runs write per-model files that are '
                         'merged afterwards. Default behaviour is unchanged.')
    a = ap.parse_args()
    if a.ns:
        globals()['NS'] = tuple(int(x) for x in a.ns.split(','))
        print('NS overridden: %s' % (NS,), flush=True)
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',
                                             os.environ.get('ENC_THREADS', 4))))
    items, meta = build_items(a.limit)
    print('varmisuse: items %s, drops %s, mean renamed %s, rename touches the pair %s'
          % (meta['n_items'], meta['drops'], meta['mean_renamed'],
             meta['frac_rename_touched_the_pair']), flush=True)
    for name in a.models.split(','):
        path = ('microsoft/codebert-base' if name == 'codebert_base'
                else name if os.path.isabs(name) else os.path.join(W.WORK, name))
        t0 = time.time()
        out = run_model(path, items, a.family, a.device, a.batch_size, a.window,
                        amp=(a.amp == 'bf16'))
        res = summarize(out, items, meta, a.family, a.window)
        res.update({'model': name, 'checkpoint': path, 'amp': a.amp,
                    'seconds': round(time.time() - t0, 1)})
        for kind in KINDS:
            d = res['kinds'][kind]
            print('[%s | %s] n %d acc %.4f (span-only %s) | ret N1/4/8 %s'
                  % (name, kind, d['n_items'], d['accuracy'],
                     None if d['accuracy_corrupted_span_only'] is None
                     else round(d['accuracy_corrupted_span_only'], 4),
                     ' '.join(str(None if d['retention_N%d' % N] is None
                                  else round(d['retention_N%d' % N], 4)) for N in NS)), flush=True)
        if not a.limit:
            merge(a.family, name, res, a.out)


if __name__ == '__main__':
    main()
