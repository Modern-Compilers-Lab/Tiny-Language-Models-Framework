#!/usr/bin/env python3
"""Does excluding identifier tokens from the mean pool fix renaming brittleness at real scale?

The tiny-scale `matched_pool` block of work/patch.json says renaming a TinyPy program destroys the
representation OF THE RENAMED IDENTIFIER TOKENS and nothing else: with no patching at all, and
query and corpus pooled over the SAME position subset, paired self-retrieval is 0.171 over all
positions, 0.008 over identifier positions, and 0.926 over non-identifier positions.

This file asks the same question of real code encoders, reusing `whyscale.real6` (1962
CodeSearchNet Python functions, six AST-certified views, paired self-retrieval) unchanged for the
corpus and the perturbations. The new work is entirely in the MASK: subword tokenizers do not put
token boundaries where the renamer put character boundaries, so the identifier mask is built from
`ast` character spans of the renameable locals in BOTH texts of a pair, intersected with the
tokenizer's offset mapping. Nothing is guessed; every step is counted and cross-checked.

Everything here ADDS to whyscale.py, it does not modify it.

    python poolfix.py real6 --model answerdotai/ModernBERT-base
    python poolfix.py poj   --model answerdotai/ModernBERT-base
    python poolfix.py merge                       # -> work/poolfix.json
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import whyscale as W                                            # noqa: E402

CACHE = W.CACHE
POOLS = ('all', 'ident', 'nonident', 'randi', 'randn')


# ---------------------------------------------------------------------------
# 1. Character spans of the renameable locals, in BOTH texts of a pair
# ---------------------------------------------------------------------------

def _line_index(src):
    lines = src.split('\n')
    starts, acc = [], 0
    for ln in lines:
        starts.append(acc)
        acc += len(ln) + 1
    return lines, starts


def _pos(lines, starts, lineno, col):
    """ast col_offset is a UTF-8 BYTE offset; tokenizer offsets are CHARACTER offsets."""
    b = lines[lineno - 1].encode('utf-8')
    return starts[lineno - 1] + len(b[:col].decode('utf-8', 'ignore'))


_AS_RE = re.compile(r'\bas\s+([A-Za-z_]\w*)')


def span_pair(s0, s1, names):
    """Character spans of every occurrence of a renameable local, in s0 and in s1.

    s1 is a view of s0 built by `_real6_view`, which is either an alpha-renaming certified by
    `_alpha_ok` (samelen / ph_seen / ph_ood / permute), a literal rewrite (constants), or s0
    itself (identity). In every case the two ASTs are ISOMORPHIC, so `ast.walk` visits them in
    lockstep and the i-th node of one is the i-th node of the other. That is asserted node by
    node, not assumed: any type mismatch, any length mismatch, and any span whose text does not
    equal the identifier it is supposed to be, rejects the snippet.

    Returns (spans0, spans1, mapping, diag) or (None, None, None, reason).
    """
    try:
        t0, t1 = ast.parse(s0), ast.parse(s1)
    except SyntaxError:
        return None, None, None, 'parse'
    n0, n1 = list(ast.walk(t0)), list(ast.walk(t1))
    if len(n0) != len(n1):
        return None, None, None, 'walk_len'
    for a, b in zip(n0, n1):
        if type(a) is not type(b):
            return None, None, None, 'walk_type'
    L0, S0 = _line_index(s0)
    L1, S1 = _line_index(s1)
    sp0, sp1, mp = [], [], {}
    diag = {'handler_sites': 0, 'handler_found': 0, 'partial_span': 0}
    for a, b in zip(n0, n1):
        if isinstance(a, ast.Name):
            ida, idb = a.id, b.id
        elif isinstance(a, ast.arg):
            ida, idb = a.arg, b.arg
        elif isinstance(a, ast.ExceptHandler):
            if not a.name or a.name not in names:
                continue
            diag['handler_sites'] += 1
            h0 = _pos(L0, S0, a.lineno, a.col_offset)
            h1 = _pos(L1, S1, b.lineno, b.col_offset)
            m0 = _AS_RE.search(s0, h0)
            m1 = _AS_RE.search(s1, h1)
            if not (m0 and m1 and m0.group(1) == a.name and m1.group(1) == b.name):
                return None, None, None, 'handler'
            diag['handler_found'] += 1
            sp0.append(m0.span(1))
            sp1.append(m1.span(1))
            if mp.setdefault(a.name, b.name) != b.name:
                return None, None, None, 'inconsistent'
            continue
        else:
            continue
        if ida not in names:
            continue
        # `ast.arg.end_col_offset` covers the ANNOTATION too (`def f(state: State)` reports the
        # span `state: State`), so an arg's span is taken as the identifier length from its start.
        # Found by printing every mismatch; without it 92 of 1962 snippets were rejected.
        if isinstance(a, ast.arg):
            s0a = _pos(L0, S0, a.lineno, a.col_offset)
            s1a = _pos(L1, S1, b.lineno, b.col_offset)
            p0 = (s0a, s0a + len(ida))
            p1 = (s1a, s1a + len(idb))
        else:
            p0 = (_pos(L0, S0, a.lineno, a.col_offset),
                  _pos(L0, S0, a.end_lineno, a.end_col_offset))
            p1 = (_pos(L1, S1, b.lineno, b.col_offset),
                  _pos(L1, S1, b.end_lineno, b.end_col_offset))
        if s0[p0[0]:p0[1]] != ida or s1[p1[0]:p1[1]] != idb:
            return None, None, None, 'span_text'
        sp0.append(p0)
        sp1.append(p1)
        if mp.setdefault(ida, idb) != idb:
            return None, None, None, 'inconsistent'
    return sp0, sp1, mp, diag


def tokenize_spans(src, names):
    """Independent second opinion on the spans, from the `tokenize` module rather than `ast`.

    Every NAME token whose string is a renameable local. This over-counts by construction --
    `obj.attr` and `f(kw=1)` are NAME tokens in other namespaces and `ast` correctly refuses
    them -- so it is a strict superset check, and the size of the difference is reported.
    """
    import io
    import tokenize as TK
    lines, starts = _line_index(src)
    out = []
    try:
        for t in TK.generate_tokens(io.StringIO(src).readline):
            if t.type == TK.NAME and t.string in names:
                out.append((_pos(lines, starts, t.start[0], t.start[1]),
                            _pos(lines, starts, t.end[0], t.end[1])))
    except (TK.TokenError, IndentationError, SyntaxError):
        return None
    return out


# ---------------------------------------------------------------------------
# 2. Token masks and the five pooled readouts, from ONE forward pass
# ---------------------------------------------------------------------------

def _rand_subset(k, universe, seed):
    if k <= 0 or len(universe) == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, len(universe))
    rs = np.random.RandomState(seed % (2 ** 31 - 1))
    return universe[rs.choice(len(universe), size=k, replace=False)]


def pooled(tok, enc, texts, spans, device, batch_size, side, seed=20260916):
    """Pooled hidden states under all five position subsets, plus mask bookkeeping.

    The five pools share one forward pass, so the ONLY thing that differs between them is which
    positions enter the mean. `all` uses the attention mask exactly as `whyscale.embed` does, so
    the `all` row here is directly comparable to the published real6 numbers.
    """
    import torch
    outs = {p: [] for p in POOLS}
    stat = {'n_tok': 0, 'n_ident': 0, 'n_real': 0, 'n_span': 0, 'n_span_covered': 0,
            'n_ident_partial': 0, 'n_empty_ident': 0, 'n_empty_nonident': 0, 'n_trunc': 0}
    tokids, identmask = [], []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        sp = spans[s:s + batch_size]
        enc_in = tok(list(chunk), truncation=True, max_length=W.BLOCK_SIZE, padding='max_length',
                     return_offsets_mapping=True, return_tensors='pt')
        offs = enc_in.pop('offset_mapping').numpy()
        am = enc_in['attention_mask'].numpy()
        ids = enc_in['input_ids'].numpy()
        im = np.zeros_like(am, dtype=bool)
        for b in range(len(chunk)):
            a, z = offs[b, :, 0], offs[b, :, 1]
            real = (z > a) & (am[b] > 0)
            stat['n_real'] += int(real.sum())
            stat['n_tok'] += int(am[b].sum())
            if am[b].sum() >= W.BLOCK_SIZE:
                stat['n_trunc'] += 1
            if not sp[b]:
                continue
            h = np.asarray(sp[b], dtype=np.int64)
            stat['n_span'] += len(h)
            ov = (a[:, None] < h[None, :, 1]) & (h[None, :, 0] < z[:, None])
            hit = real & ov.any(1)
            im[b] = hit
            stat['n_span_covered'] += int((ov & real[:, None]).any(0).sum())
            # a token is PARTIAL when it is marked but its own character span is not wholly
            # inside one identifier occurrence -- subword boundary leakage, measured not assumed
            inside = ((h[None, :, 0] <= a[:, None]) & (z[:, None] <= h[None, :, 1])).any(1)
            stat['n_ident_partial'] += int((hit & ~inside).sum())
        stat['n_ident'] += int(im.sum())
        enc_in = {k: v.to(device) for k, v in enc_in.items()}
        with torch.no_grad():
            hs = enc(**enc_in).last_hidden_state.float().cpu().numpy()
        M = am.astype(np.float32)
        masks = {'all': M}
        mi = (M * im).astype(np.float32)
        mn = (M * (~im)).astype(np.float32)
        ri = np.zeros_like(M)
        rn = np.zeros_like(M)
        for b in range(len(chunk)):
            uni = np.flatnonzero(am[b] > 0)
            g = int(hashlib.md5(('%s|%d|%d' % (side, seed, s + b)).encode()).hexdigest()[:8], 16)
            ri[b, _rand_subset(int(mi[b].sum()), uni, g)] = 1.0
            rn[b, _rand_subset(int(mn[b].sum()), uni, g + 7919)] = 1.0
        stat['n_empty_ident'] += int((mi.sum(1) < 1).sum())
        stat['n_empty_nonident'] += int((mn.sum(1) < 1).sum())
        # an empty pool would emit a zero vector that scores as an arbitrary neighbour; fall back
        # to the full pool, exactly as whyscale.embed does, and count how often it happens
        for m in (mi, mn, ri, rn):
            e = m.sum(1) < 1
            m[e] = M[e]
        masks.update({'ident': mi, 'nonident': mn, 'randi': ri, 'randn': rn})
        for p in POOLS:
            m = masks[p]
            outs[p].append((hs * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1e-9))
        # The non-identifier token ids, COMPRESSED (identifier positions removed, not blanked).
        # Renaming changes how many subword tokens an identifier costs, so every later token
        # shifts; a position-aligned comparison would fail even for a perfect mask. The compressed
        # sequence is what must be invariant under a pure rename, and comparing it to the
        # original's is the strongest available check that the mask caught every renamed token.
        for b in range(len(chunk)):
            real = am[b] > 0
            tokids.append(tuple(ids[b][real & ~im[b]].tolist()))
            identmask.append(bool(am[b].sum() >= W.BLOCK_SIZE))
    res = {p: np.concatenate(outs[p]) for p in POOLS}
    return res, stat, tokids, identmask


# ---------------------------------------------------------------------------
# 3. real6 under matched pooling
# ---------------------------------------------------------------------------

def real6_pool(model_id, device, batch_size, n=W.REAL6_N, seed=564):
    recs, scanned = W._real6_corpus(n)
    views, codes, nvars, audit = W._real6_build(recs, seed)
    src2names = {}
    for r in recs:
        src2names.setdefault(r['src'], set(r['names']))
    names = [src2names[c] for c in codes]

    # --- masks first, on CPU, so a snippet that cannot be masked is dropped from EVERY view
    spans, reasons, diag = {}, {}, {'handler_sites': 0, 'handler_found': 0}
    for m in W.REAL6_MODES:
        col0, col1, rs = [], [], []
        for i, c in enumerate(codes):
            a, b, mp, d = span_pair(c, views[(m, 'a')][i], names[i])
            if a is None:
                col0.append(None); col1.append(None); rs.append(d)
                continue
            diag['handler_sites'] += d['handler_sites']
            diag['handler_found'] += d['handler_found']
            col0.append(a); col1.append(b); rs.append('ok')
        spans[m] = (col0, col1)
        reasons[m] = {k: rs.count(k) for k in set(rs)}
    keep = [i for i in range(len(codes))
            if all(spans[m][0][i] is not None for m in W.REAL6_MODES)]
    print('mask coverage: %d of %d snippets masked in all six views' % (len(keep), len(codes)),
          flush=True)
    for m in W.REAL6_MODES:
        print('   %-10s %s' % (m, reasons[m]), flush=True)

    # --- independent cross-check of the ast spans against the tokenize module
    xcheck = {'n': 0, 'ast_subset_of_tokenize': 0, 'exact_same_set': 0,
              'tokenize_extra_total': 0, 'ast_spans_total': 0}
    for i in keep[:400]:
        tk = tokenize_spans(codes[i], names[i])
        if tk is None:
            continue
        A, T = set(spans['identity'][0][i]), set(tk)
        xcheck['n'] += 1
        xcheck['ast_spans_total'] += len(A)
        xcheck['tokenize_extra_total'] += len(T - A)
        xcheck['ast_subset_of_tokenize'] += int(A <= T)
        xcheck['exact_same_set'] += int(A == T)

    codes_k = [codes[i] for i in keep]
    tok, enc = W.load_encoder(model_id, device)
    C, cstat, ctok, cim = pooled(tok, enc, codes_k, [spans['identity'][0][i] for i in keep],
                                 device, batch_size, side='corpus')
    tag = W._tag(model_id)
    out = {'model': model_id, 'tag': tag, 'n': len(keep), 'n_before_mask_drop': len(codes),
           'seed': seed, 'block_size': W.BLOCK_SIZE, 'scanned': scanned,
           'mean_renameable_vars': float(np.mean([nvars[i] for i in keep])),
           'mask_reasons': reasons, 'handler': diag, 'tokenize_crosscheck': xcheck,
           'corpus_mask': cstat, 'views': {}}
    for m in W.REAL6_MODES:
        texts = [views[(m, 'a')][i] for i in keep]
        Q, qstat, qtok, qim = pooled(tok, enc, texts, [spans[m][1][i] for i in keep],
                                     device, batch_size, side='query_' + m)
        pair = [(a, b) for a, b, ta, tb in zip(ctok, qtok, cim, qim) if not (ta or tb)]
        same = int(sum(1 for a, b in pair if a == b))
        n_untrunc = len(pair)
        row = {'mask': qstat,
               'frac_ident_tokens': qstat['n_ident'] / max(qstat['n_tok'], 1),
               'frac_spans_covered': qstat['n_span_covered'] / max(qstat['n_span'], 1),
               'frac_ident_tokens_partial': qstat['n_ident_partial'] / max(qstat['n_ident'], 1),
               'nonident_tokenseq_identical_frac': same / max(n_untrunc, 1),
               'nonident_tokenseq_n_untruncated': n_untrunc,
               'frac_text_changed': float(np.mean([a != b for a, b in zip(texts, codes_k)]))}
        for p in POOLS:
            hit = W._paired(Q[p], C[p])
            row[p] = {'paired_top1': float(hit.mean()),
                      'self_cos': float((W.unit(Q[p]) * W.unit(C[p])).sum(1).mean()),
                      'bg_cos': W.pair_cos(Q[p])}
        # SYMMETRIC vs ASYMMETRIC. Everything above is symmetric: query and corpus are pooled
        # over the same KIND of subset. The asymmetric variant -- restrict the query's pooling
        # but leave the index pooled normally -- is the one a practitioner would reach for first
        # and it is measured here so it can be reported, not assumed. On the tiny encoder it
        # scores BELOW the all-positions baseline.
        row['asym_query_nonident_corpus_all'] = float(W._paired(Q['nonident'], C['all']).mean())
        row['asym_query_all_corpus_nonident'] = float(W._paired(Q['all'], C['nonident']).mean())
        out['views'][m] = row
        print('  %-10s all %.4f | ident %.4f | nonident %.4f | randi %.4f | randn %.4f '
              '| ASYM %.4f | ident-tok %.3f seq-ident %.3f'
              % (m, row['all']['paired_top1'], row['ident']['paired_top1'],
                 row['nonident']['paired_top1'], row['randi']['paired_top1'],
                 row['randn']['paired_top1'], row['asym_query_nonident_corpus_all'],
                 row['frac_ident_tokens'],
                 row['nonident_tokenseq_identical_frac']), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, 'poolfix_real6__%s.json' % tag)
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', p, flush=True)
    del enc
    return out


# ---------------------------------------------------------------------------
# 4. the cost: POJ-104 MAP@R under the same pooling
# ---------------------------------------------------------------------------

def poj_cost(model_id, device, batch_size, limit=None, n_rename=4,
             schemes=('samelen',)):
    """POJ-104 is C++: no `ast`, so the identifier mask is the STUDENT'S OWN extractor via the
    regex path already in `whyscale.embed`. Stated, not hidden -- it masks declared variables
    only, not function or type names, which makes this a LOWER bound on the cost of dropping
    identifier tokens."""
    recs = W.load_poj(limit)
    labels = np.array([r['label'] for r in recs])
    codes = [r['code'] for r in recs]
    sizes = {int(l): int((labels == l).sum()) for l in np.unique(labels)}
    allvars = [set(W.extract_variables(c, W.LANG)) for c in codes]
    vocab = W.corpus_vocab(recs)
    tok, enc = W.load_encoder(model_id, device)
    tag = W._tag(model_id)
    out = {'model': model_id, 'tag': tag, 'n': len(recs), 'n_rename': n_rename,
           'mean_vars_masked': float(np.mean([len(v) for v in allvars]))}

    Ef, Ec = W.embed(tok, enc, codes, allvars, device, batch_size)
    ok_f, _, _ = W.top1(Ef, Ef, labels)
    ok_c, _, _ = W.top1(Ec, Ec, labels)
    out['identity'] = {
        'all': {'MAP@R': W.map_at_r(Ef, labels), 'top1': float(ok_f.mean()),
                'bg_cos': W.pair_cos(Ef)},
        'nonident': {'MAP@R': W.map_at_r(Ec, labels), 'top1': float(ok_c.mean()),
                     'bg_cos': W.pair_cos(Ec)}}
    print('  POJ identity  MAP@R all %.4f -> nonident %.4f  (top1 %.4f -> %.4f)'
          % (out['identity']['all']['MAP@R'], out['identity']['nonident']['MAP@R'],
             out['identity']['all']['top1'], out['identity']['nonident']['top1']), flush=True)

    for scheme in schemes:
        texts, sp, mean_ren = W.perturb_all(recs, n_rename, scheme, vocab)
        Pf, Pc = W.embed(tok, enc, texts, [b for (_, b) in sp], device, batch_size)
        Of, Oc = W.embed(tok, enc, codes, [a for (a, _) in sp], device, batch_size)
        okp_f, _, _ = W.top1(Pf, Of, labels)
        okp_c, _, _ = W.top1(Pc, Oc, labels)
        ok0_f, _, _ = W.top1(Of, Of, labels)
        ok0_c, _, _ = W.top1(Oc, Oc, labels)
        out[scheme] = {
            'mean_renamed': mean_ren,
            'all': {'MAP@R_cross': W._map_at_r_cross(W.unit(Pf), W.unit(Of), labels, sizes),
                    'top1': float(okp_f.mean()),
                    'retention': W.retention(ok0_f, okp_f)[0]},
            'nonident': {'MAP@R_cross': W._map_at_r_cross(W.unit(Pc), W.unit(Oc), labels, sizes),
                         'top1': float(okp_c.mean()),
                         'retention': W.retention(ok0_c, okp_c)[0]}}
        print('  POJ %-8s MAP@R all %.4f -> nonident %.4f  retention %.4f -> %.4f'
              % (scheme, out[scheme]['all']['MAP@R_cross'],
                 out[scheme]['nonident']['MAP@R_cross'],
                 out[scheme]['all']['retention'], out[scheme]['nonident']['retention']),
              flush=True)
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, 'poolfix_poj__%s.json' % tag)
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', p, flush=True)
    del enc
    return out


# ---------------------------------------------------------------------------
# 5. layer sweep: is a middle layer a second, cheaper fix, and does it compose?
# ---------------------------------------------------------------------------

def _pool_layers(tok, enc, texts, spans, device, batch_size):
    """Mean-pooled hidden state at EVERY layer, under the all- and non-identifier masks.

    One forward pass with output_hidden_states; the pooling happens on the GPU so only the
    (batch, layers, hidden) pooled tensor crosses to the host.
    """
    import torch
    L = None
    acc = {'all': [], 'nonident': []}
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        sp = spans[s:s + batch_size]
        enc_in = tok(list(chunk), truncation=True, max_length=W.BLOCK_SIZE, padding='max_length',
                     return_offsets_mapping=True, return_tensors='pt')
        offs = enc_in.pop('offset_mapping').numpy()
        am = enc_in['attention_mask'].numpy()
        im = np.zeros_like(am, dtype=bool)
        for b in range(len(chunk)):
            a, z = offs[b, :, 0], offs[b, :, 1]
            real = (z > a) & (am[b] > 0)
            if not sp[b]:
                continue
            h = np.asarray(sp[b], dtype=np.int64)
            im[b] = real & ((a[:, None] < h[None, :, 1]) & (h[None, :, 0] < z[:, None])).any(1)
        M = am.astype(np.float32)
        mn = M * (~im)
        e = mn.sum(1) < 1
        mn[e] = M[e]
        enc_in = {k: v.to(device) for k, v in enc_in.items()}
        with torch.no_grad():
            hs = enc(**enc_in, output_hidden_states=True).hidden_states
        L = len(hs)
        last = W.pool_mode() == 'last'
        for key, msk in (('all', M), ('nonident', mn)):
            t = torch.from_numpy(msk).to(device).float()
            if last:
                # decoder readout: last position selected by this mask, not the mean over it
                col = torch.arange(t.shape[1], device=device).float()[None, :]
                li = torch.where(t > 0, col, torch.full_like(col, -1.0)).max(1).values
                li = li.clamp(min=0).long()
                r = torch.arange(t.shape[0], device=device)
                P = torch.stack([h.float()[r, li] for h in hs], 1)
            else:
                den = t.sum(1).clamp(min=1e-9)[:, None]
                P = torch.stack([(h.float() * t[..., None]).sum(1) / den for h in hs], 1)
            acc[key].append(P.cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}, L


def real6_layers(model_id, device, batch_size, n=W.REAL6_N, seed=564,
                 modes=('samelen', 'ph_seen', 'ph_ood', 'permute', 'constants')):
    recs, scanned = W._real6_corpus(n)
    views, codes, nvars, audit = W._real6_build(recs, seed)
    src2names = {}
    for r in recs:
        src2names.setdefault(r['src'], set(r['names']))
    names = [src2names[c] for c in codes]
    spans = {}
    for m in ('identity',) + tuple(modes):
        spans[m] = [span_pair(codes[i], views[(m, 'a')][i], names[i]) for i in range(len(codes))]
    keep = [i for i in range(len(codes))
            if all(spans[m][i][0] is not None for m in spans)]
    codes_k = [codes[i] for i in keep]
    tok, enc = W.load_encoder(model_id, device)
    C, L = _pool_layers(tok, enc, codes_k, [spans['identity'][i][0] for i in keep],
                        device, batch_size)
    tag = W._tag(model_id)
    out = {'model': model_id, 'tag': tag, 'n': len(keep), 'n_layers_incl_embeddings': L,
           'note': 'index 0 = embedding output, index L-1 = final layer; every row is mean '
                   'pooling with query and corpus pooled the SAME way (symmetric) unless the '
                   'key says asym',
           'views': {}}
    print('%s: %d layers, n=%d' % (model_id, L, len(keep)), flush=True)
    for m in modes:
        Q, _ = _pool_layers(tok, enc, [views[(m, 'a')][i] for i in keep],
                            [spans[m][i][1] for i in keep], device, batch_size)
        row = {}
        for key in ('all', 'nonident'):
            row[key] = [float(W._paired(Q[key][:, l], C[key][:, l]).mean()) for l in range(L)]
        row['asym_query_nonident_corpus_all'] = [
            float(W._paired(Q['nonident'][:, l], C['all'][:, l]).mean()) for l in range(L)]
        best_all = int(np.argmax(row['all']))
        best_ni = int(np.argmax(row['nonident']))
        row['best_layer_all'] = best_all
        row['best_layer_nonident'] = best_ni
        row['grid_2x2'] = {
            'all_final': row['all'][L - 1], 'all_best': row['all'][best_all],
            'nonident_final': row['nonident'][L - 1],
            'nonident_best': row['nonident'][best_ni],
            'nonident_at_best_all_layer': row['nonident'][best_all]}
        out['views'][m] = row
        print('  %-10s all: final %.4f best %.4f @L%d | nonident: final %.4f best %.4f @L%d'
              % (m, row['all'][L - 1], row['all'][best_all], best_all,
                 row['nonident'][L - 1], row['nonident'][best_ni], best_ni), flush=True)
        print('             all by layer: %s'
              % ' '.join('%.3f' % x for x in row['all']), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, 'poolfix_layers__%s.json' % tag)
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', p, flush=True)
    del enc
    return out


# ---------------------------------------------------------------------------
# 6. the strict discriminability control: can a fixed readout still tell programs APART?
# ---------------------------------------------------------------------------

def _one_site_mutants(codes, spans, rng_seed=11):
    """Real-scale port of tinyenc._one_site_mutants.

    One occurrence of one renameable local is replaced by a DIFFERENT local of the SAME LENGTH
    from the same program. Same length means every character outside identifier spans keeps its
    offset, so the mutant's non-identifier skeleton is byte-identical to the original's -- a
    readout that ignores identifier positions cannot see the mutation at all, by construction.
    This is semantics-CHANGING, which is the point: a pooling fix that survives renaming is
    worthless if it also cannot distinguish two programs that differ in which variable goes where.
    """
    import random
    rng = random.Random(rng_seed)
    idx, orig, mut, mspans = [], [], [], []
    for i, (c, sp) in enumerate(zip(codes, spans)):
        if not sp:
            continue
        occ_by_name = {}
        for (a, z) in sp:
            occ_by_name.setdefault(c[a:z], []).append((a, z))
        by_len = {}
        for nm in occ_by_name:
            by_len.setdefault(len(nm), []).append(nm)
        cands = [(x, y) for v in by_len.values() if len(v) > 1 for x in v for y in v if x != y]
        if not cands:
            continue
        x, y = rng.choice(cands)
        a, z = rng.choice(occ_by_name[x])
        idx.append(i)
        orig.append(c)
        mut.append(c[:a] + y + c[z:])          # same length -> every later offset unchanged
        mspans.append(list(sp))                # ... so the span list is unchanged too
    return idx, orig, mut, mspans


def discrim(model_id, device, batch_size, n=W.REAL6_N, seed=564):
    recs, scanned = W._real6_corpus(n)
    views, codes, nvars, audit = W._real6_build(recs, seed)
    src2names = {}
    for r in recs:
        src2names.setdefault(r['src'], set(r['names']))
    names = [src2names[c] for c in codes]
    sp0 = [span_pair(codes[i], codes[i], names[i])[0] for i in range(len(codes))]
    sp1 = [span_pair(codes[i], views[('samelen', 'a')][i], names[i])[1]
           for i in range(len(codes))]
    ok = [i for i in range(len(codes)) if sp0[i] is not None and sp1[i] is not None]
    idx, orig, mut, mspans = _one_site_mutants([codes[i] for i in ok],
                                               [sp0[i] for i in ok])
    sel = [ok[j] for j in idx]
    ren = [views[('samelen', 'a')][i] for i in sel]
    rspans = [sp1[i] for i in sel]
    print('%s: %d of %d snippets admit a same-length one-site swap'
          % (model_id, len(sel), len(ok)), flush=True)
    tok, enc = W.load_encoder(model_id, device)
    G = {}
    G['orig'], L = _pool_layers(tok, enc, orig, mspans, device, batch_size)
    G['mut'], _ = _pool_layers(tok, enc, mut, mspans, device, batch_size)
    G['ren'], _ = _pool_layers(tok, enc, ren, rspans, device, batch_size)
    # how byte-identical is "byte-identical" once a subword tokenizer is involved?
    same_len = 0
    for a, b in zip(orig, mut):
        ta = tok(a, truncation=True, max_length=W.BLOCK_SIZE)['input_ids']
        tb = tok(b, truncation=True, max_length=W.BLOCK_SIZE)['input_ids']
        same_len += int(len(ta) == len(tb))
    tag = W._tag(model_id)
    out = {'model': model_id, 'tag': tag, 'n': len(sel), 'n_layers_incl_embeddings': L,
           'frac_mutant_same_token_count': same_len / max(len(sel), 1),
           'note': 'query = the samelen-renamed program; candidate pool = every original PLUS '
                   'every one-site mutant; a win requires the true original to beat both the '
                   'other originals and every mutant STRICTLY, ties scored as losses',
           'grid': {}}
    d = np.arange(len(sel))
    for pool in ('all', 'nonident'):
        for l in range(L):
            O, M, R = (W.unit(G[k][pool][:, l]) for k in ('orig', 'mut', 'ren'))
            so, sm = R @ O.T, R @ M.T
            rival = np.maximum(np.where(np.eye(len(d), dtype=bool), -np.inf, so).max(1),
                               sm.max(1))
            win = so[d, d] > rival
            tie = np.isclose(so[d, d], rival, atol=1e-6)
            out['grid']['%s|L%d' % (pool, l)] = {
                'top1_with_mutants': float(win.mean()),
                'tie_rate': float(tie.mean()),
                'top1_originals_only': float((so.argmax(1) == d).mean()),
                'cos_orig_mutant': float((O * M).sum(1).mean()),
                'cos_orig_renamed': float((O * R).sum(1).mean())}
    g = out['grid']
    for pool in ('all', 'nonident'):
        for lbl, k in (('top1(+mutants)', 'top1_with_mutants'),
                       ('top1(orig only)', 'top1_originals_only'),
                       ('tie rate      ', 'tie_rate')):
            print('  %-9s %s %s' % (pool, lbl,
                  ' '.join('%.3f' % g['%s|L%d' % (pool, l)][k] for l in range(L))), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, 'poolfix_discrim__%s.json' % tag)
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', p, flush=True)
    del enc
    return out


# ---------------------------------------------------------------------------
# 7. the decoder cell: one JSON for "does renaming brittleness reproduce in a real causal
#    decoder, and does identifier diversity fix it"
# ---------------------------------------------------------------------------

DEC_REF = {                      # already on the books, same protocol, for the comparison table
    'codebert-base':        {'samelen': 0.5999, 'cliff': 0.0642},
    'unixcoder-base':       {'samelen': 0.8940, 'cliff': 0.0000},
    'ModernBERT-base':      {'samelen': 0.5846, 'cliff': 0.0000},
    'TinyPy tiny DECODER':  {'samelen': 0.5034, 'cliff': 0.3511},
    'TinyPy tiny ENCODER':  {'samelen': 0.1730, 'cliff': 0.5770},
}


def _r6(tag):
    p = os.path.join(CACHE, 'whyscale_real6__%s.json' % tag)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _layers(tag):
    p = os.path.join(CACHE, 'poolfix_layers__%s.json' % tag)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _cliff(tag, view='samelen'):
    d = _layers(tag)
    if d is None or view not in d['views']:
        return None
    row = d['views'][view]['all']
    L = len(row)
    return {'by_layer': row, 'n_layers_incl_embeddings': L, 'peak': max(row),
            'peak_layer': int(np.argmax(row)), 'final': row[L - 1],
            'cliff_peak_minus_final': max(row) - row[L - 1]}


def _paired_t(deltas):
    d = np.asarray([x for x in deltas if x is not None], dtype=float)
    n = len(d)
    if n < 2:
        return {'n': n, 'mean': float(d.mean()) if n else None, 'sd': None, 't': None}
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else None
    return {'n': int(n), 'mean': float(d.mean()), 'sd': sd, 't': t,
            'all_same_sign': bool(np.all(d > 0) or np.all(d < 0)),
            'deltas': [float(x) for x in d]}


def decoder(seeds=(2, 3, 4, 5)):
    """Aggregate the decoder cell into work/decoder.json. Reads only cache files; computes no
    model forward passes, so it is safe to re-run at any time."""
    out = {'_what': 'bigcode/tiny_starcoder_py (164M GPT-BigCode causal LM) on the renaming '
                    'axis: does the brittleness reproduce, and does identifier diversity fix it',
           '_protocol': {
               'scoring': 'whyscale.py real6, unchanged: 1962 CodeSearchNet Python functions, '
                          'frozen weights, cosine, paired_top1, query never allowed to retrieve '
                          'its own unperturbed self, six AST-certified views',
               'decoder_path': 'AutoModelForCausalLM with output_hidden_states=True; the last '
                               'hidden state (post final layernorm) is the encoder-equivalent '
                               'representation. Padding is on the RIGHT, so with causal masking '
                               'a real token\'s state is bit-identical to its unpadded value '
                               '(verified: max |batch32 - batch1| = 6.9e-6).',
               'pooling': 'PRIMARY = mean over non-pad positions (the readout every encoder row '
                          'in this ledger uses, so the numbers are comparable). SECONDARY = '
                          'last-token (the natural causal readout).',
               'part2': 'repair.py scale with the causal objective auto-selected from the '
                        'checkpoint architecture; arms plain / lowdiv (fixed 8-name vocabulary) '
                        '/ highdiv; 1200 steps x bs 16 over 2000 CodeSearchNet Python functions '
                        '= 9.6 epochs; paired within seed.',
               'epochs': 1200 * 16 / 2000.0},
           '_reference_rows': DEC_REF,
           'part1_frozen': {}, 'part2_diversity': {}, 'regression': {}}

    # --- encoder regression: the causal branch must not have moved an encoder number.
    # Two independent controls: same device (GPU fp16, must be bit-identical) and the CPU
    # fp32 rerun, which also certifies that the decoder rows below -- which may have been
    # produced on CPU because the GPU cap was saturated -- are comparable to the ledger.
    base = _r6('codebert-base')
    for key, tag, expect in (('gpu_fp16_bitidentical', 'codebert-base__regress', True),
                             ('cpu_fp32_device_control', 'codebert-base__cpu', False)):
        r = _r6(tag)
        if not (r and base):
            continue
        out['regression'][key] = {
            'model': 'microsoft/codebert-base', 'tag': tag,
            'ledger_samelen': base['views']['samelen']['paired_top1'],
            'rerun_samelen': r['views']['samelen']['paired_top1'],
            'abs_diff': abs(base['views']['samelen']['paired_top1']
                            - r['views']['samelen']['paired_top1']),
            'bit_identical': base['views']['samelen']['paired_top1']
                             == r['views']['samelen']['paired_top1'],
            'expect_bit_identical': expect,
            'all_views_abs_diff': {m: abs(base['views'][m]['paired_top1']
                                          - r['views'][m]['paired_top1'])
                                   for m in W.REAL6_MODES}}

    # --- part 1: the frozen decoder, both readouts. GPU tag preferred, CPU fallback.
    for label, cands in (('meanpool', ('tiny_starcoder_py', 'tiny_starcoder_py__cpu')),
                         ('lasttoken', ('tiny_starcoder_py__lastpool',
                                        'tiny_starcoder_py__cpulast'))):
        tag = next((t for t in cands if _r6(t) is not None), None)
        if tag is None:
            continue
        d = _r6(tag)
        row = {'tag': tag, 'device': 'cpu_fp32' if '__cpu' in tag else 'cuda_bf16',
               'n': d['n'],
               'views': {m: d['views'][m]['paired_top1'] for m in d['views']},
               'self_cos': {m: d['views'][m]['self_cos'] for m in d['views']},
               'sym_indep': {m: d['views'][m]['sym_indep'] for m in d['views']},
               'frac_text_changed': {m: d['views'][m]['frac_text_changed'] for m in d['views']},
               'mean_tokens': d['views']['identity']['mean_tokens'],
               'frac_truncated': d['views']['identity']['frac_truncated'],
               'binding_fraction': d.get('binding_fraction'),
               'novelty_gap': d.get('novelty_gap'),
               'layer_cliff': _cliff(tag) or _cliff(tag.replace('__lastpool', '__cpulast')
                                                 if '__lastpool' in tag else tag + '__cpu')}
        out['part1_frozen'][label] = row
    ce = _cliff('codebert-base__cpu')
    if ce:
        out['part1_frozen']['_codebert_layer_cliff_same_device'] = ce

    # --- part 2: the three arms x seeds, paired within seed
    arms = ('plain', 'lowdiv', 'highdiv')
    per = {a: {} for a in arms}
    for a in arms:
        for sd in seeds:
            d = _r6('ts_%s_s%d' % (a, sd))
            if d is not None:
                per[a][sd] = {m: d['views'][m]['paired_top1'] for m in d['views']}
    common = sorted(set.intersection(*[set(per[a]) for a in arms]) if all(per[a] for a in arms)
                    else set())
    out['part2_diversity'] = {
        'per_arm_per_seed': {a: {str(k): v for k, v in per[a].items()} for a in arms},
        'seeds_used': common,
        'paired': {}}
    for lo, hi in (('plain', 'lowdiv'), ('plain', 'highdiv'), ('lowdiv', 'highdiv')):
        for view in ('samelen', 'ph_ood', 'permute'):
            k = '%s_minus_%s|%s' % (hi, lo, view)
            out['part2_diversity']['paired'][k] = _paired_t(
                [per[hi][s][view] - per[lo][s][view] for s in common])
    out['part2_diversity']['dose_ordering_samelen'] = {
        'per_seed': {str(s): [per[a][s]['samelen'] for a in arms] for s in common},
        'plain_lt_lowdiv_lt_highdiv_in_n_seeds':
            int(sum(1 for s in common
                    if per['plain'][s]['samelen'] < per['lowdiv'][s]['samelen']
                    < per['highdiv'][s]['samelen'])),
        'n_seeds': len(common)}
    for a in arms:
        vals = [per[a][s]['samelen'] for s in common]
        out['part2_diversity'].setdefault('arm_summary', {})[a] = {
            'mean': float(np.mean(vals)) if vals else None,
            'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            'values': vals}
    # determinism probe: byte-identical command, two destinations
    a1, a2 = _r6('ts_plain_s2'), _r6('ts_plain_s2dup')
    if a1 and a2:
        out['part2_diversity']['determinism_probe'] = {
            'run_a_samelen': a1['views']['samelen']['paired_top1'],
            'run_b_samelen': a2['views']['samelen']['paired_top1'],
            'identical': a1['views']['samelen']['paired_top1']
                         == a2['views']['samelen']['paired_top1'],
            'abs_diff': abs(a1['views']['samelen']['paired_top1']
                            - a2['views']['samelen']['paired_top1'])}
    for a in ('plain', 'highdiv'):
        c = _cliff('ts_%s_s2' % a)
        if c:
            out['part2_diversity'].setdefault('layer_cliff_s2', {})[a] = c
    # --- truncation control: StarCoder's BPE is much more compact on code than codebert's,
    # so at the ledger's fixed 400-token budget the two models do not see the same fraction of
    # each function. Re-scored at 512 (codebert's position table caps this at 511).
    out['truncation_control'] = {
        '_why': 'is the decoder\'s higher samelen just "it sees more of the function"?'}
    for lbl, a, b in (('codebert-base', 'codebert-base__regress', 'codebert-base__b512'),
                      ('tiny_starcoder_py', 'tiny_starcoder_py', 'tiny_starcoder_py__b512')):
        da, db = _r6(a), _r6(b)
        if not (da and db):
            continue
        out['truncation_control'][lbl] = {
            'block400': {'samelen': da['views']['samelen']['paired_top1'],
                         'mean_tokens': da['views']['identity']['mean_tokens'],
                         'frac_truncated': da['views']['identity']['frac_truncated']},
            'block512': {'samelen': db['views']['samelen']['paired_top1'],
                         'mean_tokens': db['views']['identity']['mean_tokens'],
                         'frac_truncated': db['views']['identity']['frac_truncated']},
            'delta_samelen': db['views']['samelen']['paired_top1']
                             - da['views']['samelen']['paired_top1']}
    out['measured_diversity_first2000_train_functions'] = {
        '_note': 'printed by repair.py scale at corpus build. highdiv has about HALF the raw '
                 'distinct-name count of unmodified real code and still beats it, so raw '
                 'vocabulary size is not the active variable; what orders monotonically is how '
                 'arbitrary/decorrelated a name is from its function.',
        'plain': [14361, 14377, 14390, 14411], 'highdiv': 7635, 'lowdiv': 8}
    out['corpus_reject_accounting'] = {
        '_rule': 'rule 5: a parse-based filter must count and print its rejects',
        'kept': 2000, 'rejected_shorter_than_60_chars': 0, 'parse_error': 0,
        'no_extractable_names': 40, 'renamed': 1960, 'rename_left_text_unchanged': 0,
        'functions_differing_from_original': 1960,
        'real6_scoring_corpus': {'n_scored': 1962, 'n_scanned': 8489,
                                 'dropped_by_view_construction': 38}}
    out['prediction_scorecard'] = {
        '_registered': 'work/labnotes.md, 2026-09-17, before any number existed',
        'P1_frozen_meanpool_samelen_in_0.45_0.75': 'WRONG (0.8180)',
        'P2_lasttoken_worse_by_0.10': 'WRONG AND BACKWARDS (last-token better by +0.0658)',
        'P3_meanpool_cliff_in_0.05_0.30': 'CORRECT (0.1193)',
        'P4_highdiv_minus_plain_0.05_0.20_t_gt_3': 'PARTIAL (t=12.64 p=0.0011, but +0.0477)',
        'P5_dose_ordering_in_most_seeds': 'CORRECT (4/4, p=0.00077)',
        'P6_continued_pretraining_degrades': 'WRONG (improved 0.8180 -> 0.9428)'}
    out['caveats'] = [
        'determinism probe FAILED: byte-identical commands gave 0.9444 vs 0.9332 (|diff| '
        '0.0112). repair.py does seed torch, but torch.use_deterministic_algorithms is never '
        'enabled and CUBLAS_WORKSPACE_CONFIG is unset, so GPU kernel nondeterminism leaves '
        '~0.011 of irreducible run-to-run noise inside every paired delta.',
        'ceiling: after continued pretraining all arms sit at 0.93-0.99, so samelen has little '
        'headroom; +0.0477 closes ~83% of the remaining gap to 1.0.',
        'schedule axis (per-batch vs static renaming) deliberately held FIXED; at 9.6 epochs it '
        'is n.s. in the encoder cell (+0.0250, t=1.92) and nothing here speaks to it.',
        'n=4 seeds, ONE budget (9.6 epochs), ONE decoder, ONE size; zero-shot retrieval '
        'geometry only, no generative downstream task.']
    p = os.path.join(W.WORK, 'decoder.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:4000])
    print('wrote', p, flush=True)
    return out


def merge():
    out = {'_what': 'matched-pool retrieval on real code encoders: is renaming brittleness a '
                    'pooling artifact of identifier tokens?',
           '_protocol': {
               'real6': 'whyscale.real6 corpus and views, unchanged (CodeSearchNet Python, '
                        'AST-certified alpha-renamings). query = view, corpus = identity view. '
                        'paired_top1 = fraction whose own original is the nearest neighbour.',
               'matched': 'query and corpus are pooled over the SAME KIND of position subset; '
                          'there is no patching and no subspace fitting anywhere in this file.',
               'ident': 'tokens whose character span overlaps an occurrence of a renameable '
                        'local, from ast spans x tokenizer offset_mapping',
               'randi/randn': 'a uniformly random subset of the real token positions whose '
                              'cardinality matches ident / nonident, drawn per item per side',
               'poj': 'C++: identifier mask is the regex path of whyscale.embed over the '
                      "student's extract_variables, so it masks declared variables only"},
           'real6': {}, 'poj': {}, 'poj_n4000': {}, 'layers': {}, 'discrim': {}}
    for f in sorted(os.listdir(CACHE)):
        if f.startswith('poolfix_real6__'):
            out['real6'][f[len('poolfix_real6__'):-5]] = json.load(open(os.path.join(CACHE, f)))
        elif f.startswith('poolfix_pojN4000__'):
            out['poj_n4000'][f[len('poolfix_pojN4000__'):-5]] = json.load(
                open(os.path.join(CACHE, f)))
        elif f.startswith('poolfix_poj__'):
            out['poj'][f[len('poolfix_poj__'):-5]] = json.load(open(os.path.join(CACHE, f)))
        elif f.startswith('poolfix_discrim__'):
            out['discrim'][f[len('poolfix_discrim__'):-5]] = json.load(
                open(os.path.join(CACHE, f)))
        elif f.startswith('poolfix_layers__'):
            out['layers'][f[len('poolfix_layers__'):-5]] = json.load(
                open(os.path.join(CACHE, f)))
    p = os.path.join(W.WORK, 'poolfix.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote %s  (%d real6, %d poj, %d layers, %d discrim)'
          % (p, len(out['real6']), len(out['poj']), len(out['layers']), len(out['discrim'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['real6', 'poj', 'layers', 'discrim', 'merge',
                                    'decoder'])
    ap.add_argument('--model', default='answerdotai/ModernBERT-base')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--n', type=int, default=W.REAL6_N)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--schemes', default='samelen')
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    tag = W._tag(a.model)
    if a.cmd == 'real6':
        done = os.path.join(CACHE, 'poolfix_real6__%s.json' % tag)
        if os.path.exists(done) and not a.rerun:
            print('skip (exists):', done)
            return
        real6_pool(a.model, a.device, a.batch_size, a.n)
    elif a.cmd == 'layers':
        done = os.path.join(CACHE, 'poolfix_layers__%s.json' % tag)
        if os.path.exists(done) and not a.rerun:
            print('skip (exists):', done)
            return
        real6_layers(a.model, a.device, a.batch_size, a.n)
    elif a.cmd == 'discrim':
        done = os.path.join(CACHE, 'poolfix_discrim__%s.json' % tag)
        if os.path.exists(done) and not a.rerun:
            print('skip (exists):', done)
            return
        discrim(a.model, a.device, a.batch_size, a.n)
    elif a.cmd == 'decoder':
        decoder()
    elif a.cmd == 'poj':
        done = os.path.join(CACHE, 'poolfix_poj__%s.json' % tag)
        if os.path.exists(done) and not a.rerun:
            print('skip (exists):', done)
            return
        poj_cost(a.model, a.device, a.batch_size, a.limit,
                 schemes=tuple(x for x in a.schemes.split(',') if x))
    else:
        merge()


if __name__ == '__main__':
    main()
