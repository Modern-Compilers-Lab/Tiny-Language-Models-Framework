#!/usr/bin/env python3
"""Why renaming brittleness survives at scale, when the tiny-scale mechanism says it should not.

The tiny-scale account is autocompletion at the identifier boundary: meeting a name it has not
memorised, a low-coverage model overwrites what the context says with what its weights prefer.
Two probes on real code models (labnotes 2026-09-16) came out NULL -- `tiny_starcoder_py` copies a
completely unseen identifier out of context with continue-accuracy 0.99 and flat end-accuracy.
Real corpora sit far above the coverage threshold, so that is the predicted result, and it leaves
an open question: the student's ModernBERT figures DO show renaming degradation at scale. If it is
not the boundary decision, what is it?

This file measures the alternative directly, entirely zero-shot on public checkpoints, so nothing
here depends on checkpoints that are not on this machine.

    e1  reproduce the phenomenon and measure its geometry
        POJ-104 test, CodeXGLUE protocol, frozen encoder, meanpool, cosine. Renames produced by
        the STUDENT'S OWN perturb/{extract,rename}.py at PERTURB_SEED=123456, so the perturbation
        is byte-comparable with modernbert_scale. Alongside top-1 retention we record
        self-cosine cos(orig_i, pert_i), the background cosine between unrelated programs, the
        top-1 margin, and the mean pairwise cosine among renamed items -- the last because the
        standard perturbation renames every program's variables to the SAME strings VAR_1..VAR_N,
        which could pull all renamed items toward one another independently of any model property.

    e2  direct vs indirect. Re-pool the SAME forward pass over only those token positions that do
        not belong to a renamed identifier. If the context tokens do not move, the embedding shift
        is just the changed tokens sitting in the average, and context-only pooling restores
        retention. If they do move, names are changing how the model reads the structure.

    e4  rename-scheme ablation at matched N: canonical VAR_k, per-item random length-matched,
        per-item random fixed length, per-item real identifiers from the corpus vocabulary, a
        within-item permutation of the program's own names (misleading, but the identifier
        multiset is unchanged), and an identity no-op control.

    e5  how much of the task is names? Self-retrieval on three views of each program: the full
        source, the identifiers alone, and the source with every identifier collapsed to one
        placeholder.

    e3  runs on the cached embeddings, no GPU: is the displacement a shared nuisance subspace?
        Fit the top-k principal components of d_i = z(pert_i) - z(orig_i) on one half of the test
        set, project them out of the other half, and re-measure retention.

Usage
    python whyscale.py run   --model microsoft/unixcoder-base [--limit N]
    python whyscale.py e3                      # CPU, reads the cached embeddings
    python whyscale.py merge                   # -> work/whyscale.json
"""

import argparse
import json
import hashlib
import os
import random
import re
import sys

os.environ.setdefault('HF_HOME', '/scratch/rb4792/hf_cache')
os.environ.setdefault('XDG_CACHE_HOME', '/scratch/rb4792/xdg_cache')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                 # .../agent_researcher
WORK = os.path.join(os.path.dirname(HERE), 'work')            # tiny_scale/work
MBS_DATA = os.path.join(ROOT, 'modernbert_scale', 'data')     # the student's perturbation code
CACHE = os.environ.get('WHYSCALE_CACHE',
                       '/scratch/rb4792/src/agent_researcher/tiny_scale/work/.whyscale_cache')

sys.path.insert(0, MBS_DATA)
from perturb.extract import extract_variables, RESERVED      # noqa: E402

# Matches modernbert_scale/eval/config.py exactly, so the perturbation and the tokenisation
# budget are the same ones the student's numbers were produced with.
PERTURB_SEED = 123456
# 400 is the student's config and the value every row in the ledger was measured at.
# WHYSCALE_BLOCK exists only for the truncation control (StarCoder's tokenizer is much
# more compact on code than codebert's, so at a FIXED token budget the two models do not
# see the same fraction of each function); default is unchanged.
BLOCK_SIZE = int(os.environ.get('WHYSCALE_BLOCK', 400))
LANG = 'cpp'
IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')


# ---------------------------------------------------------------------------
# Data and perturbation
# ---------------------------------------------------------------------------

def load_poj(limit=None):
    """POJ-104 test split, 12,000 C++ programs in 24 classes (CodeXGLUE)."""
    from datasets import load_dataset
    ds = load_dataset('google/code_x_glue_cc_clone_detection_poj104', split='test')
    recs = [{'code': r['code'], 'label': int(r['label'])} for r in ds]
    if limit and limit < len(recs):
        # Stride, never a prefix: the split is sorted by class and a prefix is one problem.
        stride = len(recs) // limit
        recs = recs[::stride][:limit]
    return recs


def select_vars(code, n, seed, lang=None):
    """The variables the student's rename_variables would have picked, in source order.

    Reproduces its selection exactly (extract_variables is source-ordered and stable, and the
    sample is drawn from a private random.Random) so that every condition below perturbs the SAME
    variables and only the target names differ.
    """
    if n <= 0:
        return []
    cand = extract_variables(code, lang or LANG)
    if not cand:
        return []
    k = min(n, len(cand))
    sel = random.Random(seed).sample(cand, k)
    sel.sort(key=cand.index)
    return sel


def apply_mapping(code, mapping):
    """Word-boundary, SIMULTANEOUS substitution.

    Simultaneous rather than the student's sequential longest-first loop because one of the
    conditions below permutes a program's own names among themselves, and a sequential loop would
    turn a->b, b->a into a->a. For every non-permuting condition the two are identical, since the
    introduced names cannot collide with the source's.
    """
    if not mapping:
        return code
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile(r'\b(' + '|'.join(re.escape(k) for k in keys) + r')\b')
    return pat.sub(lambda m: mapping[m.group(0)], code)


def _rand_name(rng, length):
    return ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(max(1, length)))


def build_mapping(code, sel, scheme, rng, vocab):
    """{old: new} for one program under one renaming scheme."""
    taken = set(IDENT_RE.findall(code))
    out = {}
    if scheme == 'identity':
        return {}
    if scheme == 'permute':
        # A derangement of the selected names among themselves. Semantics-preserving, and the
        # program's identifier multiset -- hence its tokenisation and its bag of names -- is
        # exactly unchanged. Only the assignment of names to roles is destroyed.
        if len(sel) < 2:
            return {}
        order = list(sel)
        for _ in range(64):
            shuf = order[:]
            rng.shuffle(shuf)
            if all(a != b for a, b in zip(order, shuf)):
                break
        else:
            shuf = order[1:] + order[:1]
        return dict(zip(order, shuf))
    for i, old in enumerate(sel):
        if scheme == 'var_k':
            new = f'VAR_{i + 1}'
        elif scheme == 'samelen':
            new = _rand_name(rng, len(old))
        elif scheme == 'rand6':
            new = _rand_name(rng, 6)
        elif scheme == 'realvocab':
            new = rng.choice(vocab)
        else:
            raise ValueError(scheme)
        guard = 0
        while new in taken and guard < 64:
            new = (f'VAR_{i + 1}_{guard}' if scheme == 'var_k'
                   else rng.choice(vocab) if scheme == 'realvocab'
                   else _rand_name(rng, len(old) if scheme == 'samelen' else 6))
            guard += 1
        taken.add(new)
        out[old] = new
    return out


def perturb_all(records, n, scheme, vocab, lang=None, seed_base=None):
    """Return (texts, [renamed-name-set per record], mean number of variables actually renamed)."""
    texts, spans, tot = [], [], 0
    for i, r in enumerate(records):
        seed = (PERTURB_SEED if seed_base is None else seed_base) + n * 1_000_000 + i
        sel = select_vars(r['code'], n, seed, lang)
        m = build_mapping(r['code'], sel, scheme, random.Random(seed ^ 0x5EED), vocab)
        texts.append(apply_mapping(r['code'], m))
        # The names to exclude from context-only pooling: the OLD names in the original text and
        # the NEW names in the perturbed one, so the two pools cover the same source positions.
        spans.append((set(sel), set(m.values()) if m else set(sel)))
        tot += len(sel)
    return texts, spans, tot / max(len(records), 1)


def corpus_vocab(records, cap=4000):
    """Real identifier names from POJ-104 itself, for the 'plausible but wrong name' condition."""
    seen = {}
    for r in records[:2000]:
        for v in extract_variables(r['code'], LANG):
            if len(v) >= 2:
                seen[v] = seen.get(v, 0) + 1
    ranked = sorted(seen, key=lambda v: -seen[v])[:cap]
    return ranked or ['tmp']


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def pool_mode():
    """'mean' (every encoder row in this file) or 'last' (the natural causal-decoder readout).

    Read from the environment at call time rather than baked in, so a single process can be
    told which readout to use without any signature change rippling through poolfix.py.
    Default is 'mean', so encoder behaviour is bit-identical to before this was added.
    """
    return os.environ.get('WHYSCALE_POOL', 'mean').lower()


def _tag(model_id):
    """Cache tag. WHYSCALE_TAG_SUFFIX keeps a second readout of the SAME model from
    overwriting the first one's JSON."""
    return model_id.rstrip('/').split('/')[-1] + os.environ.get('WHYSCALE_TAG_SUFFIX', '')


def is_causal_lm(model_id):
    """True for a decoder-only checkpoint. Deliberately conservative: it fires only on an
    explicit *ForCausalLM / *LMHeadModel architecture or config.is_decoder, so every encoder
    already in this file (RobertaModel, RobertaForMaskedLM, ModernBertForMaskedLM) is
    unaffected."""
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception:
        return False
    if getattr(cfg, 'is_decoder', False):
        return True
    archs = list(getattr(cfg, 'architectures', None) or [])
    return any(('ForCausalLM' in a) or ('LMHeadModel' in a) for a in archs)


class _CausalHidden:
    """Adapter that makes a causal LM quack like the encoders the rest of this file expects.

    `AutoModelForCausalLM(..., output_hidden_states=True).hidden_states` is the tuple
    (embeddings, layer1, ..., layerN) where the LAST element is post-final-layernorm -- exactly
    what an AutoModel encoder calls `last_hidden_state`. So every pooling, metric and layer
    sweep downstream is reused untouched; the only decoder-specific decision is the padding
    side, fixed to RIGHT below, which with causal masking guarantees a real token's hidden
    state is bit-identical to its unpadded value.
    """

    class _Out:
        __slots__ = ('last_hidden_state', 'hidden_states')

    def __init__(self, lm):
        self.lm = lm

    def __call__(self, output_hidden_states=False, **kw):
        o = self.lm(**kw, output_hidden_states=True)
        r = _CausalHidden._Out()
        r.hidden_states = o.hidden_states
        r.last_hidden_state = o.hidden_states[-1]
        return r


def load_encoder(model_id, device):
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    if is_causal_lm(model_id):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = 'right'
        # bfloat16, not float16: StarCoder-family checkpoints were trained in bf16 and a
        # 164M causal LM's residual stream overflows fp16's 65504 range often enough that a
        # silent inf -> nan would poison paired_top1 without raising anything.
        lm = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16 if device == 'cuda' else None)
        lm.config.use_cache = False
        print('causal-LM path: %s (%s), pool=%s, pad=%r right'
              % (model_id, lm.config.model_type, pool_mode(), tok.pad_token), flush=True)
        return tok, _CausalHidden(lm.to(device).eval())
    enc = AutoModel.from_pretrained(model_id, dtype=torch.float16 if device == 'cuda' else None)
    return tok, enc.to(device).eval()


def embed(tok, enc, texts, name_sets, device, batch_size=64, normalize=True):
    """Return (full, ctx): meanpool over all real tokens, and over non-identifier-name tokens.

    Both come out of ONE forward pass, so the only difference between them is which positions
    enter the average. That is what makes the direct/indirect decomposition clean.

    `normalize` collapses all whitespace to single spaces, which is what the POJ-104 path has
    always done and is harmless for C++. It is WRONG for Python, whose indentation is its block
    structure, so the real6 path below passes normalize=False and the model sees real source.
    """
    import torch
    full_out, ctx_out = [], []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        names = name_sets[s:s + batch_size]
        flat = [' '.join(str(t).split()) if normalize else str(t) for t in chunk]
        # Clamp to what the model's position table can actually index. RoBERTa derives
        # position ids as cumsum(mask) + padding_idx, so the largest index is
        # padding_idx + seq_len; a checkpoint trained with a smaller max_position_embeddings
        # than BLOCK_SIZE triggers a device-side assert here. Pretrained encoders have 514+
        # and are unaffected; small from-scratch checkpoints are not.
        # ONLY the RoBERTa family derives position ids as cumsum(mask) + padding_idx, so only
        # it can overflow its table at a length the tokenizer happily produces. ModernBERT has
        # pad_token_id 50283 and absolute positions, where this arithmetic is meaningless and
        # would compute a negative cap.
        _cap = BLOCK_SIZE
        _c = getattr(enc, 'config', None)
        if getattr(_c, 'model_type', None) in ('roberta', 'xlm-roberta', 'camembert'):
            _cap = min(_cap, _c.max_position_embeddings - (_c.pad_token_id or 0) - 2)
        assert _cap > 16, 'position-capacity clamp produced %d' % _cap
        enc_in = tok(flat, truncation=True, max_length=_cap, padding='max_length',
                     return_offsets_mapping=True, return_tensors='pt')
        offs = enc_in.pop('offset_mapping').numpy()
        am = enc_in['attention_mask'].numpy()
        namemask = np.zeros_like(am, dtype=bool)
        for b, (text, nm) in enumerate(zip(flat, names)):
            if not nm:
                continue
            pat = re.compile(r'\b(' + '|'.join(re.escape(x) for x in sorted(nm, key=len,
                                                                            reverse=True)) + r')\b')
            hits = np.array([(m.start(), m.end()) for m in pat.finditer(text)], dtype=np.int64)
            if not len(hits):
                continue
            a, z = offs[b, :, 0], offs[b, :, 1]
            # A token is a name token when its character span overlaps any occurrence span.
            ov = (a[:, None] < hits[None, :, 1]) & (hits[None, :, 0] < z[:, None])
            namemask[b] = (z > a) & ov.any(1)
        enc_in = {k: v.to(device) for k, v in enc_in.items()}
        with torch.no_grad():
            h = enc(**enc_in).last_hidden_state.float().cpu().numpy()
        m = am.astype(np.float32)
        c = (m * (~namemask)).astype(np.float32)
        # If a program is nothing but names after truncation, fall back to the full pool rather
        # than emitting a zero vector that would silently score as an arbitrary neighbour.
        empty = c.sum(1) < 1
        c[empty] = m[empty]
        if pool_mode() == 'last':
            # Decoder readout: the state at the LAST real position. Padding is on the right and
            # attention is causal, so that position never attended to a pad.
            r = np.arange(len(m))
            li = np.maximum(m.sum(1).astype(np.int64) - 1, 0)
            full_out.append(h[r, li])
            col = np.arange(c.shape[1])[None, :]
            ci = np.maximum(np.where(c > 0, col, -1).max(1), 0)
            ctx_out.append(h[r, ci])
            continue
        full_out.append((h * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1e-9))
        ctx_out.append((h * c[..., None]).sum(1) / np.maximum(c.sum(1, keepdims=True), 1e-9))
    F, C = np.concatenate(full_out), np.concatenate(ctx_out)
    if not (np.isfinite(F).all() and np.isfinite(C).all()):
        raise SystemExit('embed(): non-finite pooled embedding (%d/%d rows) -- numeric '
                         'overflow, refusing to emit a metric'
                         % (int((~np.isfinite(F).all(1)).sum()), len(F)))
    return F, C


# ---------------------------------------------------------------------------
# Metrics (the modernbert_scale conventions, kept bit-for-bit)
# ---------------------------------------------------------------------------

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)


def top1(queries, corpus, labels, mask_self=True):
    q, c = unit(queries), unit(corpus)
    best = np.empty(len(q), dtype=np.int64)
    margin = np.empty(len(q), dtype=np.float32)
    selfsim = np.empty(len(q), dtype=np.float32)
    for s in range(0, len(q), 512):
        sl = slice(s, min(s + 512, len(q)))
        sim = q[sl] @ c.T
        idx = np.arange(sl.start, sl.stop)
        selfsim[sl] = sim[np.arange(len(idx)), idx]
        if mask_self:
            sim[np.arange(len(idx)), idx] = -1e9     # never retrieve your own unperturbed self
        o = np.argpartition(-sim, 1, axis=1)[:, :2]
        vals = np.take_along_axis(sim, o, axis=1)
        order = np.argsort(-vals, axis=1)
        o = np.take_along_axis(o, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        best[sl] = o[:, 0]
        margin[sl] = vals[:, 0] - vals[:, 1]
    return labels[best] == labels, selfsim, margin


def map_at_r(vecs, labels):
    v = unit(vecs)
    sizes = {int(l): int((labels == l).sum()) for l in np.unique(labels)}
    aps = []
    for s in range(0, len(v), 256):
        sl = slice(s, min(s + 256, len(v)))
        sim = v[sl] @ v.T
        sim[np.arange(sl.stop - sl.start), np.arange(sl.start, sl.stop)] = -1e9
        rank = np.argsort(-sim, axis=1, kind='stable')
        for j, i in enumerate(range(sl.start, sl.stop)):
            r = sizes[int(labels[i])] - 1
            if r <= 0:
                continue
            hit = labels[rank[j, :r]] == labels[i]
            aps.append(float((np.cumsum(hit) / np.arange(1, r + 1))[hit].sum() / r))
    return float(np.mean(aps)) if aps else 0.0


def retention(before, after, min_support=30):
    sup = int(before.sum())
    if sup < min_support:
        return None, sup
    return float((before & after).sum() / sup), sup


def pair_cos(x, n=2000, seed=0):
    """Mean cosine between unrelated items -- the background against which a shift is read."""
    rng = np.random.default_rng(seed)
    v = unit(x)
    i = rng.integers(0, len(v), n)
    j = rng.integers(0, len(v), n)
    k = i != j
    return float((v[i[k]] * v[j[k]]).sum(1).mean())


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

CONDITIONS = (
    # (name, N, scheme)
    ('var_k_N1', 1, 'var_k'), ('var_k_N2', 2, 'var_k'),
    ('var_k_N4', 4, 'var_k'), ('var_k_N8', 8, 'var_k'),
    ('identity_N4', 4, 'identity'),
    ('samelen_N4', 4, 'samelen'), ('rand6_N4', 4, 'rand6'),
    ('realvocab_N4', 4, 'realvocab'), ('permute_N4', 4, 'permute'),
    ('samelen_N8', 8, 'samelen'), ('rand6_N8', 8, 'rand6'),
    ('realvocab_N8', 8, 'realvocab'), ('permute_N8', 8, 'permute'),
)


def strip_identifiers(code):
    res = RESERVED[LANG]
    return IDENT_RE.sub(lambda m: m.group(0) if m.group(0) in res else 'X', code)


def names_only(code):
    res = RESERVED[LANG]
    return ' '.join(t for t in IDENT_RE.findall(code) if t not in res) or 'X'


def run(model_id, limit, device, batch_size, only=None, do_views=True, shard=''):
    """`only` restricts the condition list so one run can be split over several cheap jobs.

    Every shard recomputes the unperturbed embedding itself -- one extra forward pass, in exchange
    for shards that are completely independent and can be merged in any order.
    """
    os.makedirs(CACHE, exist_ok=True)
    tag = model_id.split('/')[-1]
    recs = load_poj(limit)
    labels = np.array([r['label'] for r in recs])
    vocab = corpus_vocab(recs)
    print(f'{model_id}: {len(recs)} POJ-104 test programs, {len(set(labels))} classes', flush=True)

    tok, enc = load_encoder(model_id, device)
    out = {'model': model_id, 'n_records': len(recs), 'block_size': BLOCK_SIZE,
           'perturb_seed': PERTURB_SEED, 'pooling': 'meanpool', 'conditions': {}}

    base_names = [set() for _ in recs]
    E0f, _ = embed(tok, enc, [r['code'] for r in recs], base_names, device, batch_size)
    np.save(os.path.join(CACHE, f'{tag}__orig_full.npy'), E0f)
    ok0, _, marg0 = top1(E0f, E0f, labels)
    out['top1_N0'] = float(ok0.mean())
    out['MAP@R_N0'] = map_at_r(E0f, labels)
    out['bg_cos_N0'] = pair_cos(E0f)
    out['margin_N0'] = float(marg0.mean())
    print(f"  N=0 top-1 {out['top1_N0']:.4f}  MAP@R {out['MAP@R_N0']:.4f}  "
          f"bg-cos {out['bg_cos_N0']:.4f}", flush=True)

    orig_ctx_cache = {}
    for cname, n, scheme in CONDITIONS:
        if only and cname not in only:
            continue
        texts, spans, mean_renamed = perturb_all(recs, n, scheme, vocab)
        pert_names = [b for (_, b) in spans]
        orig_names = [a for (a, _) in spans]
        Pf, Pc = embed(tok, enc, texts, pert_names, device, batch_size)
        # The original's context pool has to exclude the SAME source positions. Which variables
        # those are depends only on N (the selection seed is shared across schemes), so it is
        # computed once per N rather than once per condition.
        if n not in orig_ctx_cache:
            _, orig_ctx_cache[n] = embed(tok, enc, [r['code'] for r in recs], orig_names,
                                         device, batch_size)
        Oc = orig_ctx_cache[n]

        okf, selff, _ = top1(Pf, E0f, labels)
        rate_f, sup = retention(ok0, okf)
        ok0c, _, _ = top1(Oc, Oc, labels)
        okc, selfc, _ = top1(Pc, Oc, labels)
        rate_c, supc = retention(ok0c, okc)

        d = unit(Pf) - unit(E0f)
        row = {
            'N': n, 'scheme': scheme, 'mean_renamed': mean_renamed,
            'top1': float(okf.mean()), 'retention': rate_f, 'support': sup,
            'self_cos': float(selff.mean()), 'self_cos_p10': float(np.percentile(selff, 10)),
            'bg_cos_pert': pair_cos(Pf),
            'disp_norm': float(np.linalg.norm(d, axis=1).mean()),
            'ctx_top1_N0': float(ok0c.mean()), 'ctx_top1': float(okc.mean()),
            'ctx_retention': rate_c, 'ctx_support': supc,
            'ctx_self_cos': float(selfc.mean()),
        }
        out['conditions'][cname] = row
        np.save(os.path.join(CACHE, f'{tag}__{cname}_full.npy'), Pf)
        print(f"  {cname:14s} ret {row['retention']}  self-cos {row['self_cos']:.4f}  "
              f"bg-cos(pert) {row['bg_cos_pert']:.4f}  ctx-ret {row['ctx_retention']}",
              flush=True)

    # e5: how much of POJ-104 clone retrieval is carried by the identifier names alone?
    views = {} if not do_views else {'names_only': [names_only(r['code']) for r in recs],
                        'struct_only': [strip_identifiers(r['code']) for r in recs]}
    out['views'] = {}
    for vname, texts in views.items():
        V, _ = embed(tok, enc, texts, base_names, device, batch_size)
        okv, _, _ = top1(V, V, labels)
        okx, _, _ = top1(V, E0f, labels)
        out['views'][vname] = {'top1_self': float(okv.mean()),
                               'MAP@R_self': map_at_r(V, labels),
                               'top1_vs_full_corpus': float(okx.mean())}
        print(f"  view {vname:12s} top-1(self) {out['views'][vname]['top1_self']:.4f}  "
              f"MAP@R {out['views'][vname]['MAP@R_self']:.4f}", flush=True)

    path = os.path.join(CACHE, f'whyscale__{tag}{shard}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    return out


# ---------------------------------------------------------------------------
# e3: is the displacement a shared nuisance subspace?
# ---------------------------------------------------------------------------

def e3(limit, conds=None):
    """Fit the displacement's principal directions on half the test set, remove them from the
    other half, and ask how much of the lost retention comes back.

    If a rank-k correction with small k restores most of it, the renamed representation still
    carries the information -- it has been moved, coherently, in a direction shared across
    programs, and the downstream head is simply reading the wrong place. If it does not, the
    information itself is gone.
    """
    conds = conds or tuple(c[0] for c in CONDITIONS)
    recs = load_poj(limit)
    labels = np.array([r['label'] for r in recs])
    res = {}
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith('__orig_full.npy'):
            continue
        tag = f[:-len('__orig_full.npy')]
        E0 = np.load(os.path.join(CACHE, f))
        if len(E0) != len(recs):
            raise SystemExit(
                f'{f} has {len(E0)} rows but --limit gives {len(recs)} records. The cached '
                f'embeddings were produced at a different corpus size; pass the matching '
                f'--limit or point WHYSCALE_CACHE at the right directory.')
        ok0, _, _ = top1(E0, E0, labels)
        res[tag] = {}
        for cond in conds:
            path = os.path.join(CACHE, f'{tag}__{cond}_full.npy')
            if not os.path.exists(path):
                continue
            P = np.load(path)
            d = unit(P) - unit(E0)
            half = len(d) // 2
            A = np.arange(half)                      # fit here
            B = np.arange(half, len(d))              # measure here
            # top1() masks position i of the query block against column i of the corpus, which is
            # only the right self-mask when the query block starts at 0, so do it by hand for B.
            okB = _retrieve(unit(P[B]), unit(E0), labels, B)
            base, sup = retention(ok0[B], okB)
            row = {'retention_uncorrected': base, 'support': sup}

            # The label-free proxy. Does the perturbed program still have ITS OWN original as
            # nearest neighbour, among all the originals? No task labels, no head, no finetuning,
            # two forward passes -- and unlike self-cosine it is measured against the local
            # margin, which matters because CodeBERT and ModernBERT embeddings are so anisotropic
            # (background cosine ~0.95-0.97) that a decisive displacement leaves self-cosine
            # at 0.98 and is invisible in absolute terms.
            qa, ca = unit(P), unit(E0)
            hit = np.empty(len(qa), dtype=bool)
            for t in range(0, len(qa), 512):
                sl = slice(t, min(t + 512, len(qa)))
                hit[sl] = (qa[sl] @ ca.T).argmax(1) == np.arange(sl.start, sl.stop)
            row['paired_top1'] = float(hit.mean())

            # Both sides perturbed. If the renamed corpus retrieves itself as well as the original
            # retrieves itself, the perturbation destroyed no information: the whole loss is a
            # mismatch between a perturbed query and an unperturbed index.
            oks, _, _ = top1(P, P, labels)
            row['symmetric_top1'] = float(oks.mean())
            rsym, _ = retention(ok0[B], oks[B])
            row['symmetric_retention'] = rsym

            # The cheapest possible correction: subtract the mean displacement.
            mu0 = d[A].mean(0)
            q = unit(unit(P[B]) - mu0)
            row['retention_meanshift'], _ = retention(ok0[B], _retrieve(q, unit(E0), labels, B))
            mu = d[A].mean(0)
            row['shared_mean_frac'] = float(
                (mu ** 2).sum() / max((d[A] ** 2).sum(1).mean(), 1e-12))
            _, sv, Vt = np.linalg.svd(d[A] - mu, full_matrices=False)
            var = (sv ** 2) / max((sv ** 2).sum(), 1e-12)
            for k in (1, 2, 4, 8, 16, 32):
                U = Vt[:k]
                q = unit(P[B]); q = unit(q - (q @ U.T) @ U)
                c = unit(E0);   c = unit(c - (c @ U.T) @ U)
                okk = _retrieve(q, c, labels, B)
                r, _ = retention(ok0[B], okk)
                row[f'retention_pc{k}'] = r
                row[f'var_pc{k}'] = float(var[:k].sum())
            res[tag][cond] = row
            print(tag, cond, json.dumps(row), flush=True)
    path = os.path.join(CACHE, 'whyscale__e3.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', path)
    return res


def _retrieve(q, c, labels, qidx):
    """Top-1 class correctness for queries `q` (rows are corpus positions `qidx`) against `c`."""
    ok = np.empty(len(q), dtype=bool)
    for s in range(0, len(q), 512):
        sl = slice(s, min(s + 512, len(q)))
        sim = q[sl] @ c.T
        sim[np.arange(sl.stop - sl.start), qidx[sl]] = -1e9
        ok[sl] = labels[sim.argmax(1)] == labels[qidx[sl]]
    return ok


# ---------------------------------------------------------------------------
# e6: the same perturbation on a task where names are legitimate evidence
# ---------------------------------------------------------------------------

def csn(model_id, n_items, device, batch_size):
    """CodeSearchNet Python, natural-language query -> code retrieval, zero-shot.

    POJ-104 asks which programs solve the same problem, and its C++ identifiers are mostly `i`,
    `n`, `a`. A natural-language search query is the opposite case: the words in a docstring
    routinely ARE the identifier names, so a model that leans on names is doing something correct
    rather than taking a shortcut. Running the identical renaming conditions on both tasks is what
    separates "renaming destroys information the model needs" from "renaming shifts the
    representation away from an unperturbed index".
    """
    from datasets import load_dataset
    tok, enc = load_encoder(model_id, device)
    ds = load_dataset('code_search_net', 'python', split='test', streaming=True)
    recs = []
    for r in ds:
        doc = ' '.join(str(r['func_documentation_string']).split())
        code = r['func_code_string']
        if len(doc.split()) < 5 or not (120 < len(code) < 2000):
            continue
        if len(extract_variables(code, 'python')) < 2:
            continue
        recs.append({'code': code, 'doc': doc.split('.')[0][:300]})
        if len(recs) >= n_items:
            break
    print(f'{model_id}: {len(recs)} CodeSearchNet Python test functions', flush=True)
    labels = np.arange(len(recs))                      # one gold document per query
    blank = [set() for _ in recs]
    vocab = [v for r in recs[:400] for v in extract_variables(r['code'], 'python')][:4000] or ['tmp']

    Q, _ = embed(tok, enc, [r['doc'] for r in recs], blank, device, batch_size)
    C0, _ = embed(tok, enc, [r['code'] for r in recs], blank, device, batch_size)
    out = {'model': model_id, 'n_items': len(recs), 'task': 'codesearchnet_python_nl2code',
           'conditions': {}}

    def score(C):
        sim = unit(Q) @ unit(C).T
        gold = sim[labels, labels]
        ranks = 1 + (sim > gold[:, None]).sum(1)
        return (ranks == 1), float(np.mean(1.0 / ranks))

    ok0, mrr0 = score(C0)
    out['recall@1'] = float(ok0.mean())
    out['MRR'] = mrr0
    print(f"  N=0 recall@1 {out['recall@1']:.4f}  MRR {mrr0:.4f}", flush=True)

    conds = [('var_k_N4', 4, 'var_k'), ('samelen_N4', 4, 'samelen'),
             ('realvocab_N4', 4, 'realvocab'), ('permute_N4', 4, 'permute'),
             ('identity_N4', 4, 'identity')]
    for cname, n, scheme in conds:
        texts, _, mean_renamed = perturb_all(recs, n, scheme, vocab, lang='python')
        C, _ = embed(tok, enc, texts, blank, device, batch_size)
        okn, mrrn = score(C)
        rate, sup = retention(ok0, okn)
        out['conditions'][cname] = {'recall@1': float(okn.mean()), 'MRR': mrrn,
                                    'retention': rate, 'support': sup,
                                    'mean_renamed': mean_renamed,
                                    'self_cos': float((unit(C0) * unit(C)).sum(1).mean())}
        print(f"  {cname:14s} recall@1 {okn.mean():.4f}  MRR {mrrn:.4f}  ret {rate}", flush=True)

    # Strip every identifier, on the code side only. On POJ-104 this is nearly free; here it
    # removes the words the query is matching against.
    for vname, fn in (('struct_only', lambda c: IDENT_RE.sub(
                          lambda m: m.group(0) if m.group(0) in RESERVED['python'] else 'X', c)),
                      ('names_only', lambda c: ' '.join(
                          t for t in IDENT_RE.findall(c) if t not in RESERVED['python']) or 'X')):
        C, _ = embed(tok, enc, [fn(r['code']) for r in recs], blank, device, batch_size)
        okv, mrrv = score(C)
        rate, sup = retention(ok0, okv)
        out['conditions'][vname] = {'recall@1': float(okv.mean()), 'MRR': mrrv,
                                    'retention': rate, 'support': sup}
        print(f"  {vname:14s} recall@1 {okv.mean():.4f}  MRR {mrrv:.4f}  ret {rate}", flush=True)

    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"whyscale_csn__{model_id.split('/')[-1]}.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    return out


def toks(model_id, limit, n_items=1200):
    """The tokenisation control, model-free: does a condition change the token count?

    `samelen` matches the CHARACTER length of the original name (that is what
    contrabert_augment.generate_random_name_same_length does, and what this file copies), which is
    NOT the same as matching its SUBWORD count. A renamed identifier that splits into more pieces
    lengthens the sequence and shifts every later position, and that mechanical effect would
    masquerade as a representational one. So measure it directly, per condition: the mean subword
    length of the introduced names, of the names they replaced, and the change in the whole
    sequence's token count under the same tokenizer and the same 400-token budget the encoder sees.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    recs = load_poj(limit)[:n_items]
    vocab = corpus_vocab(recs)
    base = [len(tok(' '.join(r['code'].split()), truncation=True,
                    max_length=BLOCK_SIZE)['input_ids']) for r in recs]
    out = {'model': model_id, 'n_items': len(recs), 'block_size': BLOCK_SIZE,
           'mean_seq_tokens_orig': float(np.mean(base)), 'conditions': {}}
    print(f'{model_id}: {len(recs)} programs, mean {np.mean(base):.1f} tokens at '
          f'max_length={BLOCK_SIZE}', flush=True)
    for cname, n, scheme in CONDITIONS:
        texts, spans, _ = perturb_all(recs, n, scheme, vocab)
        seq, old_t, new_t = [], [], []
        for (text, (olds, news)) in zip(texts, spans):
            seq.append(len(tok(' '.join(text.split()), truncation=True,
                               max_length=BLOCK_SIZE)['input_ids']))
            for o in olds:
                old_t.append(len(tok(' ' + o, add_special_tokens=False)['input_ids']))
            for w in news:
                new_t.append(len(tok(' ' + w, add_special_tokens=False)['input_ids']))
        row = {'mean_seq_tokens': float(np.mean(seq)),
               'mean_seq_token_delta': float(np.mean(np.array(seq) - np.array(base))),
               'mean_subtokens_old': float(np.mean(old_t)) if old_t else None,
               'mean_subtokens_new': float(np.mean(new_t)) if new_t else None,
               'frac_truncated': float(np.mean(np.array(seq) >= BLOCK_SIZE))}
        out['conditions'][cname] = row
        print(f"  {cname:14s} name subtokens {row['mean_subtokens_old']} -> "
              f"{row['mean_subtokens_new']}   seq {row['mean_seq_tokens']:.1f} "
              f"(delta {row['mean_seq_token_delta']:+.2f})  truncated {row['frac_truncated']:.3f}",
              flush=True)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"whyscale_toks__{model_id.split('/')[-1]}.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    return out


# ---------------------------------------------------------------------------
# Adversarial checks on the two load-bearing results
# ---------------------------------------------------------------------------

SYM_SEED = 987654321          # an independent perturbation draw, unrelated to PERTURB_SEED


def sym(model_id, limit, device, batch_size, conds=None):
    """Is the symmetric-index result an artefact of query and index sharing one rename draw?

    e3's `symmetric_retention` uses the SAME perturbed array as both query and index (with the
    diagonal masked, so nothing retrieves itself). Different programs get different variable
    SELECTIONS there, because the seed is per-record, but every program receives the canonical
    names VAR_1..VAR_N. A reviewer is right to ask whether that makes the two sides trivially
    matched. So: draw the index's perturbation from an INDEPENDENT seed, so query and index
    disagree about which variables were renamed as well, and compare all three numbers.

        asym      perturbed query vs unperturbed index    (the standard protocol)
        sym_same  perturbed query vs the same draw        (what e3 reported)
        sym_indep perturbed query vs an independent draw  (the honest version)
    """
    conds = conds or ('var_k_N4', 'var_k_N8', 'samelen_N8', 'realvocab_N8', 'rand6_N8')
    want = {c: (n, sc) for c, n, sc in CONDITIONS if c in conds}
    recs = load_poj(limit)
    labels = np.array([r['label'] for r in recs])
    vocab = corpus_vocab(recs)
    blank = [set() for _ in recs]
    tok, enc = load_encoder(model_id, device)
    E0, _ = embed(tok, enc, [r['code'] for r in recs], blank, device, batch_size)
    ok0, _, _ = top1(E0, E0, labels)
    out = {'model': model_id, 'n_records': len(recs), 'top1_N0': float(ok0.mean()),
           'sym_seed': SYM_SEED, 'conditions': {}}
    print('%s: %d programs, N=0 top-1 %.4f' % (model_id, len(recs), ok0.mean()), flush=True)
    for cname in conds:
        n, scheme = want[cname]
        tA, spA, _ = perturb_all(recs, n, scheme, vocab)
        tB, spB, _ = perturb_all(recs, n, scheme, vocab, seed_base=SYM_SEED)
        ov = [len(a[0] & b[0]) / max(len(a[0] | b[0]), 1) for a, b in zip(spA, spB)]
        PA, _ = embed(tok, enc, tA, [b for _, b in spA], device, batch_size)
        PB, _ = embed(tok, enc, tB, [b for _, b in spB], device, batch_size)
        r = {}
        r['asym'], _ = retention(ok0, top1(PA, E0, labels)[0])
        r['sym_same'], _ = retention(ok0, top1(PA, PA, labels)[0])
        r['sym_indep'], _ = retention(ok0, top1(PA, PB, labels)[0])
        r['selected_var_jaccard'] = float(np.mean(ov))
        r['identical_texts'] = float(np.mean([a == b for a, b in zip(tA, tB)]))
        out['conditions'][cname] = r
        print('  %-14s asym %.4f  sym_same %.4f  sym_indep %.4f  (draw overlap %.3f, identical %.3f)'
              % (cname, r['asym'], r['sym_same'], r['sym_indep'], r['selected_var_jaccard'],
                 r['identical_texts']), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'whyscale_sym__%s.json' % model_id.split('/')[-1])
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    return out


def _canonical(code, style):
    """Replace every non-reserved identifier with a placeholder, in one of four styles."""
    res = RESERVED[LANG]
    if style in ('X', 'Q'):
        return IDENT_RE.sub(lambda m: m.group(0) if m.group(0) in res else style, code)
    order, seen = [], {}
    for m in IDENT_RE.finditer(code):
        t = m.group(0)
        if t not in res and t not in seen:
            seen[t] = len(order)
            order.append(t)
    fmt = (lambda k: 'VAR_%d' % (k + 1)) if style == 'VAR' else (lambda k: 'v%d' % k)
    return IDENT_RE.sub(lambda m: m.group(0) if m.group(0) in res else fmt(seen[m.group(0)]), code)


def views2(model_id, limit, device, batch_size):
    """Attack the identifiers->X result.

    Three objections, answered in one pass. (1) `X` might itself be a useful structural marker,
    so run a second single placeholder `Q`, and two DISTINCT-per-variable schemes (`v0,v1,..` and
    `VAR_1,VAR_2,..`) which additionally preserve the aliasing structure a single placeholder
    destroys. (2) collapsing identifiers SHORTENS programs, so some that were truncated at the
    400-token budget now fit, which would flatter the view -- report the truncation fraction and
    mean token count for every view. (3) report top-1 on a held-out half as well as the full set.
    """
    recs = load_poj(limit)
    labels = np.array([r['label'] for r in recs])
    blank = [set() for _ in recs]
    tok, enc = load_encoder(model_id, device)
    B = np.arange(len(recs) // 2, len(recs))
    out = {'model': model_id, 'n_records': len(recs), 'views': {}}
    views = [('full', lambda c: c)]
    for st in ('X', 'Q', 'v', 'VAR'):
        views.append(('ident_' + st, (lambda st: lambda c: _canonical(c, st))(st)))
    views.append(('names_only', names_only))
    for vname, fn in views:
        texts = [fn(r['code']) for r in recs]
        ln = np.array([len(tok(' '.join(t.split()), truncation=True,
                              max_length=BLOCK_SIZE)['input_ids']) for t in texts])
        V, _ = embed(tok, enc, texts, blank, device, batch_size)
        ok, _, _ = top1(V, V, labels)
        row = {'top1': float(ok.mean()), 'top1_heldout_half': float(ok[B].mean()),
               'MAP@R': map_at_r(V, labels), 'mean_tokens': float(ln.mean()),
               'frac_truncated': float((ln >= BLOCK_SIZE).mean())}
        out['views'][vname] = row
        print('  %-12s top1 %.4f (half B %.4f)  MAP@R %.4f  tokens %.1f  trunc %.3f'
              % (vname, row['top1'], row['top1_heldout_half'], row['MAP@R'],
                 row['mean_tokens'], row['frac_truncated']), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'whyscale_views2__%s.json' % model_id.split('/')[-1])
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    return out


# ---------------------------------------------------------------------------
# The same analysis at tiny scale, where coverage q is a knob
# ---------------------------------------------------------------------------
#
# The POJ-104 result is representational: renaming translates the pooled embedding into a region
# the unperturbed index does not occupy. Nothing about that is intrinsically large-scale, and at
# tiny scale we own the one axis no real corpus gives us -- identifier coverage q. If the
# representational failure IS the autocompletion failure seen through a pooling layer, it must
# track q and must not be recoverable by perturbing the index. If it is a second component, the
# placeholder views will misbehave at every q including the high ones where autocompletion does
# not exist.
#
# The tiny models are decoders, but a causal decoder has representations: feed the code region
# (everything before the first #STEP) and mean-pool the final-layer residual. Because attention is
# causal, appending the #STEP token cannot change any earlier position, so feeding the code alone
# is exactly equivalent and simpler.
#
# Two things here are BETTER controlled than at large scale, and both matter:
#   * the tokenizer is character-level, so character length IS token count. Every view below is
#     length-preserving, which removes by construction the tokenisation confound that turned out
#     to dominate the POJ-104 `var_k` numbers.
#   * retrieval is paired and needs no labels: does the renamed snippet retrieve ITS OWN original?
#     The diagonal is the TARGET, not a leak, so it must NOT be masked -- the opposite of the
#     POJ-104 protocol. `identity` scoring exactly 1.000 is the check that this is wired right.

TINY_N = 2048
TINY_PLACEHOLDER_SLOTS = 12       # max distinct variables in a snippet is 9


def _tiny_examples(ckpt_dir, n=TINY_N):
    """Code regions of the checkpoint's OWN test corpus.

    interp.corpus_for derives the corpus from the RUN DIRECTORY's name, so a path to a specific
    .pth has to be reduced to its dirname first. Hardcoding data_p1/test.txt is a mistake this
    project has already made once and it produced a result that reached a draft.
    """
    import interp
    run = os.path.dirname(ckpt_dir) if ckpt_dir.endswith('.pth') else ckpt_dir
    path = interp.corpus_for(run)
    ex = open(path).read().split('\n\n')[:-1][:n]
    return [e.split('#STEP', 1)[0].rstrip('\n') for e in ex], path


_TINY_EXACT_CACHE = {}


def _tiny_exact(data_dir):
    """(mask, members) for this corpus, from the CLOSED-FORM reconstruction of the lexicon.

    Replaces the old `lexicon.tsv` scan, which was retracted. That scan walked the token stream,
    and the character tokenizer emits `if` as a single token INSIDE identifiers, so it invented
    fragments: at q=70% it held 16,960 of the 17,576 possible three-character strings and called
    91% of renamed names "seen", all false positives. `lexicon.exact_lexicon` instead replays the
    generator's own LCG for each length and is verified three ways -- it reproduces stats.json's
    `diversity_train` counts exactly, it agrees with an independent rescan, and the two-directional
    test puts original test names 1.000 inside and renamed names 0.000 inside at every coverage.

    Length 1 is every letter at every q: the generator bypasses the coverage rule for one-character
    names, so all 26 are always seen.
    """
    key = os.path.abspath(data_dir)
    if key in _TINY_EXACT_CACHE:
        return _TINY_EXACT_CACHE[key]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lexicon as _lex
    q = _lex.rarity_of(data_dir)
    mask, _ext = _lex.exact_lexicon(q)
    _TINY_EXACT_CACHE[key] = (mask, q)
    return _TINY_EXACT_CACHE[key]


class _ExactSeen:
    """Membership in the training lexicon, as a set-like object.

    `_unseen_name` does `nm not in seen_set` in a redraw loop, so this has to answer membership
    cheaply and cannot be a materialised set: at q=70% the length-5 row alone has 8,035,180 names.
    """

    def __init__(self, mask):
        self.mask = mask

    def __contains__(self, name):
        L = len(name)
        if L == 1:
            return True                      # the length-1 bypass: all 26 letters, every q
        m = self.mask.get(L)
        if m is None:
            return False
        v = 0
        for ch in name:
            if not ('a' <= ch <= 'z'):
                return False
            v = v * 26 + (ord(ch) - 97)
        return bool(m[v])

    def __len__(self):
        """Total distinct identifiers of length 1-5 that training could draw."""
        return int(sum(int(m.sum()) for m in self.mask.values()))

    def per_length(self):
        return {L: int(m.sum()) for L, m in sorted(self.mask.items())}




def _tiny_lexicon(data_dir, per_len=2000):
    """Up to `per_len` identifiers of each length that training really could draw.

    Sampled uniformly from the reconstructed lexicon rather than taken off a frequency-sorted
    head. The LCG visits its covered set once per reset, so within a length every covered name has
    the same expected count and a uniform sample is the frequency head up to noise -- with the
    difference that every name here provably occurred, which the old head could not promise.

    The cap is PER LENGTH, not on lines read: at q=20% a global cap left bank[5] empty.
    """
    import numpy as _np
    mask, _q = _tiny_exact(data_dir)
    by_len = {}
    for L in range(1, 6):
        m = mask.get(L)
        if m is None:
            continue
        idx = _np.flatnonzero(m)
        if len(idx) > per_len:
            rng = _np.random.default_rng(1234567 + L)
            idx = rng.choice(idx, size=per_len, replace=False)
        names = []
        for v in idx.tolist():
            chars = []
            for _ in range(L):
                chars.append(chr(97 + v % 26))
                v //= 26
            names.append(''.join(reversed(chars)))
        by_len[L] = names
    return by_len


def _tiny_seen_set(data_dir):
    """Membership test for "did training ever contain this identifier".

    Needed because "unseen" could not be taken on trust: the thesis renamer enters the same LCG as
    the generator with its counter at 85% of the period, which is meant to make renamed names
    unseen for q <= 0.85, and that is a claim to check rather than assume. With the exact
    reconstruction the check is now exact in both directions.
    """
    mask, _q = _tiny_exact(data_dir)
    return _ExactSeen(mask)


def _tiny_name_banks(data_dir, seed=564, offset=0, seen_set=None):
    """(seen, ood): fixed corpus-wide placeholder names, indexed [length][slot].

    Both banks are length-matched slot for slot, so the only difference between the two
    placeholder views is whether the model has seen the string. There is NO unseen one-character
    name at any q -- the generator bypasses the coverage rule for length 1, so all 26 letters are
    seen everywhere -- so the length-1 row is necessarily identical in both banks. That weakens
    the seen/ood contrast by the ~20% of variables that are one character long, and it is recorded
    rather than hidden.

    `offset` is what makes a genuinely INDEPENDENT second draw possible, and it exists because the
    obvious way of doing it silently did not work. SameLengthRenamer's unseen names come from the
    thesis VariableRenamer's LCG, whose state is hard-coded at 85% of m; its constructor seeds
    Python's `random` but the LCG ignores that. Two renamers built with different seeds therefore
    emit the SAME name stream, and a symmetric control built that way compares a draw with itself
    and scores a meaningless 1.000. Measured before the fix: frac_draws_differ was 0.000 for both
    placeholder views. Offsetting burns `offset` names off the stream first, which does work.
    """
    import interp
    freq = _tiny_lexicon(data_dir)
    seen_set = _tiny_seen_set(data_dir) if seen_set is None else seen_set
    ren = interp.SameLengthRenamer(seed)
    rng = random.Random(seed + 7919 * offset)
    seen, ood = {}, {}
    for L in range(1, 6):
        pool = freq.get(L, [])
        # Sample from the frequent head rather than taking a fixed top-12, so that a second draw
        # is a different set of SEEN names rather than the same one.
        head = pool[:2000] if len(pool) > TINY_PLACEHOLDER_SLOTS else pool
        if len(head) >= TINY_PLACEHOLDER_SLOTS:
            seen[L] = rng.sample(head, TINY_PLACEHOLDER_SLOTS)
        else:
            seen[L] = (head * TINY_PLACEHOLDER_SLOTS)[:TINY_PLACEHOLDER_SLOTS] or ['a']
            seen[L] = seen[L][:TINY_PLACEHOLDER_SLOTS]
        if L == 1:
            ood[L] = list(seen[L])
            continue
        # Burn the stream with an EMPTY reserved set. Reserving what has already been drawn
        # deadlocks here: the length-2 LCG has period 676, so once the burn has seen them all,
        # every further draw is already reserved and `name()` loops forever. Measured by hanging.
        for _ in range(offset):
            ren.name(L, set())
        taken = set()
        ood[L] = []
        guard = set(seen[L])
        for _ in range(TINY_PLACEHOLDER_SLOTS):
            nm = _unseen_name(ren, L, taken | guard, seen_set)  # verified, not assumed, unseen
            taken.add(nm)
            ood[L].append(nm)
    return seen, ood


def _unseen_name(ren, length, reserved, seen_set, tries=64):
    """A same-length name that is neither reserved here nor present in training.

    Written as a redraw loop rather than `ren.name(L, reserved | seen_set)` because the training
    lexicon has up to 1.3M entries and unioning it per variable copied that set ~6,000 times per
    run. Same result, three orders of magnitude cheaper.
    """
    for _ in range(tries):
        nm = ren.name(length, reserved)
        if nm not in seen_set:
            return nm
    return nm                      # give up rather than loop: counted via the audit block


def _stable_seed(*parts):
    """A seed that is the same in every process.

    `hash()` is not: CPython randomises string and tuple hashing per interpreter unless
    PYTHONHASHSEED is set, so the `permute` and `constants` views silently drew a DIFFERENT
    rewrite on every run. Caught by comparing a GPU run (1927 surviving snippets) against a
    later CPU pass (1931).
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode())
        h.update(b'\x00')
    return int.from_bytes(h.digest(), 'big')


def _derangement(names, draw=0):
    """The permutation `permute` applies, as ONE function.

    This logic used to be written out twice -- here and in `_permtrace_map` -- and the two copies
    drifted the moment `_tiny_rename` moved from `hash()` to `_stable_seed`: the stale copy
    reproduced the real derangement on only 39.3% of snippets while still reporting a clean
    substitution check. Anything that needs the mapping calls this.
    """
    if len(names) < 2:
        return None
    perm = list(names)
    rng_local = random.Random(_stable_seed(tuple(names), draw))
    for _ in range(64):                  # reseeded once, outside the loop, or every
        rng_local.shuffle(perm)          # redraw is the same permutation
        if all(a != b for a, b in zip(names, perm)):
            break
    if any(a == b for a, b in zip(names, perm)):
        return None                      # no derangement found; drop from every view
    return dict(zip(names, perm))


def _tiny_rename(code, mode, banks, ren, seen_set=frozenset(), draw=0):
    """Rewrite one snippet's variables. Returns None if the result will not parse back.

    `draw` selects which of the independent rewrites this is. It exists because `permute` and
    `constants` derive their randomness from the CODE rather than from `ren`, so without it draw
    b is literally draw a: measured `frac_draws_differ` 0.0000, and the resulting `sym_indep`
    1.0000 was an identity rather than a measurement. Same trap `_tiny_name_banks` documents for
    the placeholder views.
    """
    import ast
    tree = ast.parse(code)
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
    if mode == 'identity':
        mp = {v: v for v in names}
    elif mode == 'samelen':
        # `seen_set` is unioned into `reserved` so a same-length replacement is verified unseen
        # rather than assumed unseen. Length-1 names are the exception and cannot be unseen.
        mp = {}
        for v in names:
            bad = set(ren.r.reserved_words) | set(names) | set(mp.values())
            mp[v] = (ren.name(1, bad) if len(v) == 1
                     else _unseen_name(ren, len(v), bad, seen_set))
    elif mode == 'permute':
        # A derangement of the program's OWN identifiers. Every name in the file is one the
        # original contained -- the SET of identifiers is identical and no novel string is
        # introduced; only which variable answers to which name changes. (Occurrence counts are
        # permuted, so the token count can shift by a few tokens; `mean_tokens` records it.) This is the exact control
        # that at realistic scale costs 1.6-6.2% of retention, and it is the test of whether a
        # mean-pooled representation is essentially a bag of the names present. A program with
        # fewer than two variables has no derangement and is dropped from every view.
        mp = _derangement(names, draw)
        if mp is None:
            return None                      # <2 names, or no derangement; drop from every view
    elif mode == 'constants':
        # Identifiers untouched; every integer literal replaced by a different integer with the
        # same number of digits. The deflationary null for the whole tiny-encoder probe: if a
        # surface change that touches no identifier moves the pooled representation as far as
        # renaming does, the probe is measuring token overlap and says nothing about names.
        rngc = random.Random(_stable_seed(code, draw))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
                d = len(str(abs(n.value)))
                lo, hi = (0, 9) if d == 1 else (10 ** (d - 1), 10 ** d - 1)
                for _ in range(32):
                    v = rngc.randint(lo, hi)
                    if v != abs(n.value):
                        break
                n.value = -v if n.value < 0 else v
        return ast.unparse(tree).replace('    ', '\t')
    else:
        bank = banks[0] if mode == 'placeholder_seen' else banks[1]
        mp = {v: bank[len(v)][i % TINY_PLACEHOLDER_SLOTS] for i, v in enumerate(names)}
        if len(set(mp.values())) != len(mp):            # two variables collided onto one name
            return None
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            n.id = mp[n.id]
    return ast.unparse(tree).replace('    ', '\t')


def _tiny_build(codes, data_dir, seed, return_keep=False):
    """All four views of the corpus, plus an INDEPENDENT second draw of each for the symmetric
    control. Snippets where any view fails are dropped from ALL views, so every view is scored on
    the same positions. `frac_draws_differ` in the output is the check that draw b really is a
    second draw and not the same one again."""
    import interp
    modes = ('identity', 'samelen', 'placeholder_seen', 'placeholder_ood',
             'permute', 'constants')
    seen_set = _tiny_seen_set(data_dir)
    draws = {}
    for tag, off in (('a', 0), ('b', 997)):        # 997 is coprime with every band size
        banks_d = _tiny_name_banks(data_dir, seed, off, seen_set)
        ren = interp.SameLengthRenamer(seed + off)
        for _ in range(off):                 # offset the LCG so draw b is not draw a again
            for L in range(2, 6):
                ren.name(L, set())
        for m in modes:
            draws[(m, tag)] = [_tiny_rename(c, m, banks_d, ren, seen_set, off) for c in codes]
    keep = [i for i in range(len(codes)) if all(v[i] is not None for v in draws.values())]
    built = {k: [v[i] for i in keep] for k, v in draws.items()}
    if return_keep:
        return built, [codes[i] for i in keep], keep
    return built, [codes[i] for i in keep]


def _tiny_encoder(data_dir):
    """(encode_fn, name) for the corpus. Byte-pair corpora carry their own bpe.json.

    The byte-pair model was trained on the CHARACTER tokenizer's reconstruction of the corpus
    (space-free), not on the raw test.txt, so the text is round-tripped through the character
    vocabulary before the subword encoder sees it. Feeding raw test.txt instead shifts every
    offset by the dropped spaces and gives the two arms of the control different inputs --
    the same trap interp.bpe_boundary documents.
    """
    import interp
    bp = os.path.join(data_dir, 'bpe.json')
    if not os.path.exists(bp):
        return interp.TOK.encode, 'char52'
    from tokenizers import Tokenizer
    bpe = Tokenizer.from_file(bp)
    inv = {i: t for t, i in interp.TOK.encod_map.items()}

    def enc(text):
        return bpe.encode(''.join(inv[i] for i in interp.TOK.encode(text))).ids
    return enc, 'bpe%d' % bpe.get_vocab_size()


def _tiny_embed(model, texts, device, batch_size=64, enc=None):
    """Mean-pool the final-layer residual over the code region."""
    import torch as T
    import interp
    if enc is None:
        enc = interp.TOK.encode
    out = []
    for s in range(0, len(texts), batch_size):
        chunk = [enc(t) for t in texts[s:s + batch_size]]
        if any(t < 0 or t >= model.token_embedding_table.num_embeddings for c in chunk for t in c):
            raise ValueError('tokenizer/checkpoint vocabulary mismatch before GPU embedding')
        L = max(len(c) for c in chunk)
        ids = T.zeros((len(chunk), L), dtype=T.long, device=device)
        msk = T.zeros((len(chunk), L), dtype=T.float32, device=device)
        for b, c in enumerate(chunk):
            ids[b, :len(c)] = T.tensor(c, device=device)
            msk[b, :len(c)] = 1.0
        with T.no_grad():
            _, _, hid = model(ids, return_hidden=True)
        h = hid[-1].float()
        out.append(((h * msk[..., None]).sum(1) / msk.sum(1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def _paired(q, c):
    """Fraction of rows whose OWN counterpart is the nearest neighbour. Diagonal is the target."""
    q, c = unit(q), unit(c)
    hit = np.empty(len(q), dtype=bool)
    for s in range(0, len(q), 512):
        sl = slice(s, min(s + 512, len(q)))
        hit[sl] = (q[sl] @ c.T).argmax(1) == np.arange(sl.start, sl.stop)
    return hit


def tiny(ckpt, device, batch_size, n=TINY_N, seed=564):
    """Run the POJ-104 geometry analysis on one TinyPy checkpoint."""
    import torch as T
    import interp
    codes, corpus_path = _tiny_examples(ckpt, n)
    data_dir = os.path.dirname(corpus_path)
    pth = ckpt if ckpt.endswith('.pth') else glob_pth(ckpt)[-1]
    model = interp.load_model(pth, device)
    enc, tokname = _tiny_encoder(data_dir)
    seen_set = _tiny_seen_set(data_dir)
    _sn, _od = _tiny_name_banks(data_dir, seed, 0, seen_set)
    views, codes = _tiny_build(codes, data_dir, seed)
    E0 = _tiny_embed(model, codes, device, batch_size, enc)
    run = os.path.dirname(ckpt) if ckpt.endswith('.pth') else ckpt
    tag = os.path.basename(os.path.normpath(run))
    frac = re.search(r'checkpoint_([0-9.]+)\.pth', pth).group(1)
    tag = '%s@%s' % (tag, frac)
    # Per length, because the constraint is structural and length-dependent: coverage q is
    # literally the fraction of the name space the model has seen, and for SHORT names it
    # saturates. At q=20% the training lexicon contains 675 of the 676 two-character names, so
    # there is no unseen two-character name to rename into -- exactly as there is no unseen
    # one-character name at any q. The ood bank is therefore genuinely unseen only at the lengths
    # where the space is still mostly unused, and this records which those are per model.
    _flat = lambda b: [x for L in range(2, 6) for x in b[L]]
    _per_len = lambda b: {L: int(sum(n in seen_set for n in b[L])) for L in range(1, 6)}
    out = {'ckpt': pth, 'run': run, 'train_fraction': float(frac), 'corpus': corpus_path,
           'n': len(codes), 'seed': seed, 'tokenizer': tokname,
           'audit': {'train_lexicon_size': len(seen_set),
                     'train_lexicon_by_len': seen_set.per_length(),
                     'seen_bank_in_train': int(sum(n in seen_set for n in _flat(_sn))),
                     'ood_bank_in_train': int(sum(n in seen_set for n in _flat(_od))),
                     'ood_bank_in_train_by_len': _per_len(_od),
                     'slots_per_len': TINY_PLACEHOLDER_SLOTS,
                     'bank_size_len2to5': len(_flat(_sn))},
           'views': {}}
    print('%s: %d snippets from %s' % (tag, len(codes), corpus_path), flush=True)
    half = len(codes) // 2
    A, B = np.arange(half), np.arange(half, len(codes))
    _bases = {}
    for m in ('identity', 'samelen', 'placeholder_seen', 'placeholder_ood',
              'permute', 'constants'):
        PA = _tiny_embed(model, views[(m, 'a')], device, batch_size, enc)
        PB = _tiny_embed(model, views[(m, 'b')], device, batch_size, enc)
        hit = _paired(PA, E0)
        d = unit(PA) - unit(E0)
        mu = d[A].mean(0)
        _, sv, Vt = np.linalg.svd(d[A] - mu, full_matrices=False)
        row = {
            'paired_top1': float(hit.mean()),
            'paired_top1_heldout_half': float(hit[B].mean()),
            'self_cos': float((unit(PA) * unit(E0)).sum(1).mean()),
            'bg_cos': pair_cos(PA), 'bg_cos_orig': pair_cos(E0),
            'sym_indep': float(_paired(PA, PB).mean()),
            'shared_mean_frac': float((mu ** 2).sum() / max((d[A] ** 2).sum(1).mean(), 1e-12)),
            'frac_text_changed': float(np.mean([x != y for x, y in zip(views[(m, 'a')], codes)])),
            'frac_draws_differ': float(np.mean([x != y for x, y in
                                                zip(views[(m, 'a')], views[(m, 'b')])])),
        }
        for k, U in [('mean', None), ('pc4', Vt[:4]), ('pc8', Vt[:8])]:
            if U is None:
                qq, cc = unit(unit(PA) - mu), unit(E0)
            else:
                qq = unit(PA); qq = unit(qq - (qq @ U.T) @ U)
                cc = unit(E0); cc = unit(cc - (cc @ U.T) @ U)
            hh = np.empty(len(B), dtype=bool)
            for s in range(0, len(B), 512):
                sl = slice(s, min(s + 512, len(B)))
                idx = B[sl]
                hh[sl] = (qq[idx] @ cc.T).argmax(1) == idx
            row['corrected_' + k] = float(hh.mean())
        # Keep the basis: whether `samelen` and `permute` move programs in the SAME subspace is
        # the question of whether they are one mechanism or two, and it cannot be read off the
        # per-view scalars. Fit on half A only, so the comparison is between two generalising
        # subspaces rather than between two fits to the same points.
        _bases[m] = Vt[:8]
        row['sv_top16'] = [float(x) for x in sv[:16]]
        row['sv_energy_top8'] = float((sv[:8] ** 2).sum() / max((sv ** 2).sum(), 1e-12))
        out['views'][m] = row
        print('  %-17s paired %.4f (half B %.4f)  sym_indep %.4f  self-cos %.4f  bg %.4f'
              '  mfrac %.3f  mean %.4f pc8 %.4f'
              % (m, row['paired_top1'], row['paired_top1_heldout_half'], row['sym_indep'],
                 row['self_cos'], row['bg_cos'], row['shared_mean_frac'],
                 row['corrected_mean'], row['corrected_pc8']), flush=True)
    # Are the views moving programs in the SAME subspace? Principal angles between the two
    # rank-8 bases, reported as the mean squared cosine (1.0 = identical subspace, 8/384 = 0.021
    # = what two independent random 8-planes in 384 dimensions give). This is the test that
    # decides "one mechanism, two manifestations" against "two mechanisms": if `samelen` and
    # `permute` displace along a shared subspace, the familiarity effect and the binding effect
    # are one geometric object and diversity is removing a coordinate from it.
    _names = [m for m in ('samelen', 'placeholder_seen', 'placeholder_ood', 'permute',
                          'constants') if m in _bases]
    out['subspace_overlap'] = {}
    for i, m1 in enumerate(_names):
        for m2 in _names[i + 1:]:
            U1, U2 = _bases[m1], _bases[m2]
            out['subspace_overlap']['%s|%s' % (m1, m2)] = float(((U1 @ U2.T) ** 2).sum()
                                                                 / U1.shape[0])
    out['subspace_overlap']['_chance'] = float(_bases[_names[0]].shape[0] /
                                               _bases[_names[0]].shape[1])
    print('  subspace overlap (mean cos^2 of principal angles, chance %.4f):'
          % out['subspace_overlap']['_chance'])
    for k, v in out['subspace_overlap'].items():
        if not k.startswith('_'):
            print('    %-38s %.4f' % (k, v), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'whyscale_tiny__%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    return out


# ---------------------------------------------------------------------------
# Part A: is the `permute` cost binding damage, or index collision?
# ---------------------------------------------------------------------------
# paired_top1 asks a renamed program to retrieve its own original out of a pool of ~2048
# originals. TinyPy programs are stereotyped and draw names from a small lexicon, so a permuted
# program may simply look like a DIFFERENT pool member -- crowding, not a damaged representation
# of name-to-role binding. The two accounts make different predictions and this measures them:
#   crowding  -> the true counterpart stays near the top (rank 2-5), top-1 collapses with pool
#                size, the decoy is a structural twin, and deduplicating the pool restores it.
#   damage    -> the true counterpart is pushed deep into the ranking, the rank percentile is
#                pool-size stable, decoys are no more twin-like than random, dedup changes little.

def _canon(code, wipe_consts=False):
    """Canonical form: identifiers renamed v0,v1,... in order of first appearance in the source.

    Two snippets with the same canonical form are the SAME program up to renaming -- exactly the
    equivalence class a bag-of-names representation cannot separate, and (with wipe_consts) the
    class a representation that also ignores literals cannot separate.
    """
    import ast
    tree = ast.parse(code)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    order, seen = [], set()
    for m in re.finditer(r'[A-Za-z_][A-Za-z_0-9]*', code):
        t = m.group(0)
        if t in names and t not in seen:
            seen.add(t); order.append(t)
    mp = {v: 'v%d' % i for i, v in enumerate(order)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            n.id = mp[n.id]
        elif (wipe_consts and isinstance(n, ast.Constant)
              and isinstance(n.value, int) and not isinstance(n.value, bool)):
            n.value = 0
    return ast.unparse(tree)


def _names_of(code):
    import ast
    return frozenset(n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name))


def _tiny_pool_extra(data_dir, k, exclude):
    """Extra pool members: code regions decoded out of val.bin.

    The test set is only 2048 snippets, and the pool-size argument needs a pool bigger than the
    headline one. val.bin holds character ids one per byte for the character corpora, so the text
    comes back without regenerating anything. (Byte-pair corpora store subword ids instead; this
    control runs on the character models.)
    """
    import interp
    path = os.path.join(data_dir, 'val.bin')
    if k <= 0 or not os.path.exists(path) or os.path.exists(os.path.join(data_dir, 'bpe.json')):
        return []
    import ast
    inv = {i: t for t, i in interp.TOK.encod_map.items()}
    # The 52-token vocabulary has no space, so decoding gives `if94>-8:` and `ifsalwn<541:`.
    # Python's own tokenizer then reads `ifsalwn` as ONE identifier and the snippet is a
    # SyntaxError, so a naive parse-filter silently keeps only the programs whose `if`
    # conditions happen to start with a digit or a minus -- a pool biased away from exactly
    # the name-adjacent structure this control is about. Re-space `if`, then round-trip
    # through unparse so the extras carry the same surface form (and TAB indentation, which
    # the tokenizer does encode) as the test snippets.
    # The space goes in only where `if` STARTS a statement. Spacing every `if` token instead
    # splits the identifiers that contain the substring ('hqifd' -> 'hqif d'), which throws
    # away exactly the names whose tokenisation this project is about: 53 of ~5,000 val
    # snippets, all of them name-contains-`if` cases. Measured, not assumed.
    IF = interp.TOK.encod_map['if']
    r = np.memmap(path, dtype=np.uint8, mode='r')
    parts, prev = [], '\n'
    for c in r:
        t = inv[int(c)]
        parts.append('if ' if (int(c) == IF and prev[-1:] in ('\n', '\t')) else t)
        prev = t
    txt = ''.join(parts)
    out, bad = [], 0
    for e in txt.split('\n\n')[:-1]:
        c = e.split('#STEP', 1)[0].rstrip('\n')
        if not c:
            continue
        try:
            c = ast.unparse(ast.parse(c)).replace('    ', '\t')
        except SyntaxError:
            bad += 1
            continue
        if c in exclude:
            continue
        out.append(c)
        if len(out) >= k:
            break
    if bad:
        print('  pool extras: %d val snippets failed to parse (kept %d)' % (bad, len(out)),
              flush=True)
    return out


def _sweep(b, M, sizes):
    """Expected top-1 at pool size S for each query, from its full-pool beat-count b.

    A pool of size S is S-1 distractors drawn without replacement from the M available plus the
    query's own counterpart; the counterpart wins iff none of the b that outscore it is drawn.
    P = C(M-b, S-1)/C(M, S-1), computed exactly rather than sampled.
    """
    from math import lgamma
    res = {}
    for S in sizes:
        d = S - 1
        if d > M:
            continue
        acc = np.empty(len(b))
        for i, bi in enumerate(b):
            if bi > M - d:
                acc[i] = 0.0
            else:
                acc[i] = np.exp(lgamma(M - bi + 1) - lgamma(M - bi - d + 1)
                                - lgamma(M + 1) + lgamma(M - d + 1))
        res[str(S)] = float(acc.mean())
    return res


def _rank_stats(sim, n_q):
    """Rank of the true counterpart (1 = top-1) for each query, plus the argmax decoy."""
    diag = sim[np.arange(n_q), np.arange(n_q)].copy()
    beat = (sim > diag[:, None]).sum(1)           # strictly better scoring pool members
    rank = beat + 1
    top = sim.argmax(1)
    return rank, beat, top


def collide(ckpt, device, batch_size, n=TINY_N, seed=564, extra=10000):
    """Pool-size sweep, decoy forensics, distance geometry and dedup for the permute view."""
    import torch as T
    import interp
    codes, corpus_path = _tiny_examples(ckpt, n)
    data_dir = os.path.dirname(corpus_path)
    pth = ckpt if ckpt.endswith('.pth') else glob_pth(ckpt)[-1]
    model = interp.load_model(pth, device)
    enc, tokname = _tiny_encoder(data_dir)
    views, codes = _tiny_build(codes, data_dir, seed)     # same drop-set as whyscale tiny
    nq = len(codes)
    extras = _tiny_pool_extra(data_dir, extra, set(codes))
    pool_codes = codes + extras
    N = len(pool_codes)
    run = os.path.dirname(ckpt) if ckpt.endswith('.pth') else ckpt
    tag = '%s@%s' % (os.path.basename(os.path.normpath(run)),
                     re.search(r'checkpoint_([0-9.]+)\.pth', pth).group(1))
    print('%s: %d queries, pool %d (%d test + %d val), tok %s'
          % (tag, nq, N, nq, len(extras), tokname), flush=True)

    P = unit(_tiny_embed(model, pool_codes, device, batch_size, enc))
    canon = [_canon(c) for c in pool_codes]
    canon_c = [_canon(c, True) for c in pool_codes]
    names = [_names_of(c) for c in pool_codes]

    # duplicate structure of the pool itself, independent of any model
    import collections as _C
    cc, ccc = _C.Counter(canon), _C.Counter(canon_c)
    dup = {'pool_n': N,
           'distinct_up_to_renaming': len(cc),
           'frac_in_a_renaming_dup_class': float(np.mean([cc[x] > 1 for x in canon])),
           'distinct_up_to_renaming_and_constants': len(ccc),
           'frac_in_a_structure_dup_class': float(np.mean([ccc[x] > 1 for x in canon_c])),
           'mean_class_size_renaming': float(N / max(len(cc), 1)),
           'mean_class_size_structure': float(N / max(len(ccc), 1))}
    print('  pool dup: %d distinct up to renaming (%.4f in a dup class); %d up to renaming+consts'
          % (dup['distinct_up_to_renaming'], dup['frac_in_a_renaming_dup_class'],
             dup['distinct_up_to_renaming_and_constants']), flush=True)

    # --- what a BAG representation could score, computed without the model ------------
    # `permute` leaves the identifier multiset, the character multiset and the token multiset
    # of the program exactly unchanged. So ANY bag-of-surface-form retriever is exactly as
    # accurate on the permuted query as on the original, and its ceiling is simply how often a
    # program's bag is unique in the pool. If that is ~1.0 the retrieval task is still fully
    # solvable after permutation, and a model that nonetheless drops to 0.6 is not the victim
    # of an ambiguous index -- it is reporting something the bag does not contain.
    import ast as _ast
    def _namebag(c):
        return tuple(sorted(_C.Counter(n.id for n in _ast.walk(_ast.parse(c))
                                       if isinstance(n, _ast.Name)).items()))
    def _tokbag(c):
        return tuple(sorted(_C.Counter(interp.TOK.encode(c)).items()))
    kn = [_namebag(c) for c in pool_codes]
    kt = [_tokbag(c) for c in pool_codes]
    cn_, ct_ = _C.Counter(kn), _C.Counter(kt)
    out_or = {
        'frac_name_multiset_unique_in_pool': float(np.mean([cn_[k] == 1 for k in kn[:nq]])),
        'frac_token_multiset_unique_in_pool': float(np.mean([ct_[k] == 1 for k in kt[:nq]])),
        'mean_pool_members_sharing_name_multiset': float(np.mean([cn_[k] for k in kn[:nq]])),
    }
    print('  bag ceiling on permute: name-multiset unique %.4f, token-multiset unique %.4f'
          % (out_or['frac_name_multiset_unique_in_pool'],
             out_or['frac_token_multiset_unique_in_pool']), flush=True)

    rng = np.random.default_rng(seed)
    out = {'ckpt': pth, 'run': run, 'corpus': corpus_path, 'tokenizer': tokname,
           'bag_ceiling_on_permute': out_or,
           'n_queries': nq, 'pool_n': N, 'pool_extra': len(extras), 'seed': seed,
           'pool_duplication': dup, 'views': {}}
    sizes = [64, 256, 1024, 2048, 4096, 8192, N]
    for m in ('identity', 'samelen', 'permute'):
        Q = unit(_tiny_embed(model, views[(m, 'a')], device, batch_size, enc))
        sim = (Q @ P.T).astype(np.float32)
        rank, beat, top = _rank_stats(sim, nq)
        # --- test-only pool (the headline 2048), and the full pool
        sub = sim[:, :nq]
        rank_s, beat_s, top_s = _rank_stats(sub, nq)
        diag = sim[np.arange(nq), np.arange(nq)]
        off = sim.copy(); off[np.arange(nq), np.arange(nq)] = -2.0
        row = {
            'top1_testpool': float((rank_s == 1).mean()),
            'top1_fullpool': float((rank == 1).mean()),
            'mrr_fullpool': float((1.0 / rank).mean()),
            'mrr_testpool': float((1.0 / rank_s).mean()),
            'median_rank_fullpool': float(np.median(rank)),
            'mean_rank_fullpool': float(rank.mean()),
            'rank_pct_fullpool_mean': float((rank / N).mean()),
            'rank_pct_testpool_mean': float((rank_s / nq).mean()),
            'rank_cdf_fullpool': {k: float((rank <= k).mean())
                                  for k in (1, 2, 3, 5, 10, 25, 100, 1000)},
            'rank_cdf_testpool': {k: float((rank_s <= k).mean())
                                  for k in (1, 2, 3, 5, 10, 25, 100, 1000)},
            'rank_quantiles_fullpool': {q: float(np.quantile(rank, q))
                                        for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
            'sweep_random_subpool': _sweep(beat, N - 1, sizes),
            # --- geometry
            'self_cos': float(diag.mean()),
            'offdiag_cos_mean': float(off[off > -1.5].mean()),
            'offdiag_cos_std': float(off[off > -1.5].std()),
            'best_offdiag_cos': float(off.max(1).mean()),
            'margin_self_minus_best_other': float((diag - off.max(1)).mean()),
            'self_z_vs_offdiag': float(((diag - off[off > -1.5].mean())
                                        / max(off[off > -1.5].std(), 1e-9)).mean()),
            'frac_offdiag_above_self': float((beat / (N - 1)).mean()),
        }
        # --- decoy forensics on the test-only pool (what did it fetch instead?)
        fail = np.nonzero(rank_s != 1)[0]
        # How BADLY is the true counterpart beaten? A crowded index loses by a hair to a near
        # twin; a damaged representation loses by a wide margin to something unrelated.
        offs = sub.copy(); offs[np.arange(nq), np.arange(nq)] = -2.0
        gap = offs.max(1) - sub[np.arange(nq), np.arange(nq)]
        row['fail_gap_mean'] = float(gap[fail].mean()) if len(fail) else float('nan')
        row['fail_gap_median'] = float(np.median(gap[fail])) if len(fail) else float('nan')
        row['fail_gap_frac_under_0.005'] = (float((gap[fail] < 0.005).mean())
                                            if len(fail) else float('nan'))
        row['fail_gap_frac_under_0.02'] = (float((gap[fail] < 0.02).mean())
                                           if len(fail) else float('nan'))
        row['self_cos_on_failures'] = (float(sub[fail, fail].mean())
                                       if len(fail) else float('nan'))
        base = rng.integers(0, nq - 1, nq)          # a uniformly random decoy, never the query
        base = base + (base >= np.arange(nq))
        def _prof(qi, di):
            sh = np.array([len(names[a] & names[b]) for a, b in zip(qi, di)], dtype=float)
            nm = np.array([len(names[a]) for a in qi], dtype=float)
            return {'n': int(len(qi)),
                    'mean_shared_names': float(sh.mean()) if len(qi) else float('nan'),
                    'mean_frac_of_query_names_shared':
                        float((sh / np.maximum(nm, 1)).mean()) if len(qi) else float('nan'),
                    'frac_share_ge1': float((sh >= 1).mean()) if len(qi) else float('nan'),
                    'frac_share_ge2': float((sh >= 2).mean()) if len(qi) else float('nan'),
                    'frac_share_ge3': float((sh >= 3).mean()) if len(qi) else float('nan'),
                    'frac_same_nameset': float(np.mean([names[a] == names[b]
                                                        for a, b in zip(qi, di)])) if len(qi) else float('nan'),
                    'frac_same_canon': float(np.mean([canon[a] == canon[b]
                                                      for a, b in zip(qi, di)])) if len(qi) else float('nan'),
                    'frac_same_structure': float(np.mean([canon_c[a] == canon_c[b]
                                                          for a, b in zip(qi, di)])) if len(qi) else float('nan')}
        row['decoys'] = _prof(fail, top_s[fail])
        row['decoys_random_baseline'] = _prof(np.arange(nq), base)
        # a retrieval that lands on a program identical to the query up to renaming is not a
        # mistake any representation could avoid; score it as a hit and see what is left
        row['top1_testpool_tie_tolerant'] = float(np.mean(
            [canon[i] == canon[int(top_s[i])] for i in range(nq)]))
        row['top1_testpool_tie_tolerant_structure'] = float(np.mean(
            [canon_c[i] == canon_c[int(top_s[i])] for i in range(nq)]))
        # --- deduplicated pool: one representative per renaming-equivalence class
        for key, cl in (('dedup_renaming', canon), ('dedup_structure', canon_c)):
            first, keep = {}, []
            for i, c in enumerate(cl):
                if c not in first:
                    first[c] = i
                    if i < nq:
                        keep.append(i)
            reps = sorted(first.values())
            ridx = np.array(reps)
            qidx = np.array(keep)
            if len(qidx) == 0:
                continue
            col = {int(j): k for k, j in enumerate(ridx)}
            sm = sim[np.ix_(qidx, ridx)]
            tgt = np.array([col[int(i)] for i in qidx])
            rk = (sm > sm[np.arange(len(qidx)), tgt][:, None]).sum(1) + 1
            row[key] = {'n_queries': int(len(qidx)), 'pool_n': int(len(ridx)),
                        'top1': float((rk == 1).mean()), 'mrr': float((1.0 / rk).mean()),
                        'median_rank': float(np.median(rk))}
        out['views'][m] = row
        print('  %-9s top1 test %.4f full %.4f | MRR %.4f | med rank %.0f | rank<=5 %.4f '
              '| self %.4f best-other %.4f | dedup(struct) top1 %.4f'
              % (m, row['top1_testpool'], row['top1_fullpool'], row['mrr_fullpool'],
                 row['median_rank_fullpool'], row['rank_cdf_fullpool'][5], row['self_cos'],
                 row['best_offdiag_cos'], row.get('dedup_structure', {}).get('top1', float('nan'))),
              flush=True)
        print('             sweep %s' % ' '.join(
            '%s:%.3f' % (k, v) for k, v in row['sweep_random_subpool'].items()), flush=True)
        print('             decoys n=%d share>=1 %.3f same-nameset %.3f same-struct %.3f '
              '(random baseline %.3f/%.3f/%.3f)'
              % (row['decoys']['n'], row['decoys']['frac_share_ge1'],
                 row['decoys']['frac_same_nameset'], row['decoys']['frac_same_structure'],
                 row['decoys_random_baseline']['frac_share_ge1'],
                 row['decoys_random_baseline']['frac_same_nameset'],
                 row['decoys_random_baseline']['frac_same_structure']), flush=True)
    path = os.path.join(WORK, 'collide.json')
    allo = json.load(open(path)) if os.path.exists(path) else {}
    allo[tag] = out
    with open(path, 'w') as f:
        json.dump(allo, f, indent=1)
    print('wrote', path, flush=True)
    return out


# ---------------------------------------------------------------------------
# real6: the six TINY views, on real Python, under the tiny paired protocol
# ---------------------------------------------------------------------------
#
# The tiny table splits renaming damage into a NOVELTY component (coverage-sensitive, carried by
# one shared displacement direction) and a BINDING component (`permute`: a derangement of the
# program's OWN names, coverage-INSENSITIVE, no shared direction). Both of those views were
# implemented only for TinyPy, where every program is a toy with nothing but local integer
# variables and `ast` can rename every `Name` node blindly. Real code cannot be treated that way:
# a `Name` node may be a builtin, an import, a module global, or a function being called, and
# renaming any of those changes what the program does. So this path
#
#   * restricts itself to the PYTHON subset of what whyscale already loads (CodeSearchNet
#     Python test). POJ-104, the corpus of the `run` path, is C++ and has no `ast`; there is no
#     way to certify a derangement safe there, so it is not used here. STATED, not hidden.
#   * decides renameability with `symtable`, not with a regex or a bare `ast.Name` walk. A
#     uniform bijection on a set of names is semantics-preserving exactly when every one of those
#     names is bound-and-local in every scope that mentions it. `symtable` is the only thing in
#     the stdlib that answers that question, and it is needed: a name can be a local in a nested
#     function and the `list` builtin two lines above it.
#   * scores with the tiny protocol, not the POJ one: paired self-retrieval (does the perturbed
#     program retrieve ITS OWN original?), an independent second draw for `sym_indep`, and
#     `shared_mean_frac`. Label-free, and directly comparable to the tiny rows.
#
# The six views are the tiny six, by name. `ph_seen` / `ph_ood` mean seen / not seen IN THIS
# CORPUS -- we do not know what a public checkpoint was pretrained on, and do not pretend to.

REAL6_N = 2000
REAL6_SLOTS = 12
REAL6_LMAX = 20                   # longest identifier length with a full placeholder bank
REAL6_SKIP = {'self', 'cls'}      # legal to rename, conventionally fixed; left alone, and said so
REAL6_DANGER = {'eval', 'exec', 'locals', 'globals', 'vars', 'compile', '__import__',
                'getattr', 'setattr', 'hasattr', 'delattr'}
REAL6_MODES = ('identity', 'samelen', 'ph_seen', 'ph_ood', 'permute', 'constants')


def _py_renameable(src):
    """The genuine local variables of `src` that a bijection may safely rename. May be empty.

    Refused: anything imported, anything global or declared global/nonlocal, anything bound to a
    def or a class (its binding occurrence is a bare string, not a Name), anything bound at module
    level, and anything that is local in one scope but not local in another scope that mentions it
    (the shadowed-builtin trap). Attribute names, keyword-argument names and dict keys are not
    symbols at all and are therefore untouched by construction.
    """
    import symtable
    try:
        st = symtable.symtable(src, '<real6>', 'exec')
    except SyntaxError:
        return None
    local, bad, allsym = set(), set(), set()
    stack = [st]
    while stack:
        sc = stack.pop()
        stack.extend(sc.get_children())
        module = sc.get_type() == 'module'
        for s in sc.get_symbols():
            n = s.get_name()
            allsym.add(n)
            if s.is_imported() or s.is_declared_global() or s.is_global() or s.is_namespace():
                bad.add(n)
                continue
            if module:                       # module-level binding: visible outside, refuse
                bad.add(n)
                continue
            if s.is_local() or s.is_parameter():
                local.add(n)
            elif s.is_free():
                continue                     # closure over an enclosing local: still safe
            else:
                bad.add(n)
    import builtins as _b, keyword as _k
    # Python 3.12+ synthesises scopes and symbols that are not source identifiers at all -- a
    # function with an annotated signature gets an `__annotate__` scope whose parameter is
    # literally named `.format`. Renaming one of those emits `def f(.format)`, which is not
    # Python. Found by printing the first ten snippets, which is why that check exists.
    ok = {n for n in local - bad - REAL6_SKIP - REAL6_DANGER - set(dir(_b))
          if n.isidentifier() and not _k.iskeyword(n)
          and not (n.startswith('__') and n.endswith('__'))}
    return sorted(ok), {n for n in allsym if n.isidentifier()}


def _real6_parse(code):
    """(canonical source, tree, renameable names) or None if this snippet is not usable.

    The canonical source is `ast.unparse` of the original, and EVERY view including `identity` is
    `ast.unparse` of a tree. That is what makes `identity` an exact no-op and its paired_top1 the
    check that the harness is wired right -- comparing a view against the raw source instead would
    measure the formatter.
    """
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    body = [n for n in tree.body]
    if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    for n in ast.walk(tree):
        # Global/Nonlocal/Match bind through bare strings; an ExceptHandler name is handled
        # explicitly below, these three are not worth the code and are simply dropped.
        if isinstance(n, (ast.Global, ast.Nonlocal)) or type(n).__name__ in ('Match', 'MatchAs'):
            return None
    # Drop docstrings. `ast.unparse` already drops comments, so leaving docstrings in would be
    # inconsistent, and worse: a docstring is natural language that is INVARIANT under all six
    # views, so every view could retrieve on the docstring alone and all six numbers would be
    # compressed toward 1.0. The question here is about identifiers, so the prose goes.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Module)):
            continue
        b = getattr(node, 'body', None)
        if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)):
            node.body = b[1:] or [ast.Pass()]
    try:
        src = ast.unparse(tree)
        tree = ast.parse(src)                # reparse so positions match the canonical text
    except Exception:
        return None
    if len(src) < 100:
        return None
    got = _py_renameable(src)
    if got is None:
        return None
    names, allsym = got
    ints = sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, int)
               and not isinstance(n.value, bool))
    return src, names, sorted(allsym - set(names)), ints


def _real6_apply(src, mapping):
    """Rewrite `src` with {old: new}, touching only real binding/reference sites."""
    import ast
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in mapping:
            n.id = mapping[n.id]
        elif isinstance(n, ast.arg) and n.arg in mapping:
            n.arg = mapping[n.arg]
        elif isinstance(n, ast.ExceptHandler) and n.name in mapping:
            n.name = mapping[n.name]
    return ast.unparse(tree)


def _real6_constants(src, rng):
    import ast
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            d = len(str(abs(n.value)))
            lo, hi = (0, 9) if d == 1 else (10 ** (d - 1), 10 ** d - 1)
            v = abs(n.value)
            for _ in range(32):
                v = rng.randint(lo, hi)
                if v != abs(n.value):
                    break
            n.value = -v if n.value < 0 else v
    return ast.unparse(tree)


def _real6_banks(vocab_by_len, rng, corpus_vocab_set):
    """(seen, ood) placeholder banks, length-matched slot for slot, exactly as the tiny path.

    `seen` are frequent identifiers of this corpus; `ood` are same-length strings that do not
    occur anywhere in it. A shared corpus-wide bank is the point: it is what gives a placeholder
    view one common displacement direction, which is the quantity `shared_mean_frac` reads.
    """
    seen, ood, fell_back = {}, {}, []
    for L in range(1, REAL6_LMAX + 1):
        pool = vocab_by_len.get(L, [])
        head = pool[:2000]
        if len(head) >= REAL6_SLOTS:
            seen[L] = rng.sample(head, REAL6_SLOTS)
        elif head:
            seen[L] = [head[i % len(head)] for i in range(REAL6_SLOTS)]
        else:
            seen[L] = None
        cand, guard = [], 0
        while len(cand) < REAL6_SLOTS and guard < 8192:
            guard += 1
            w = _rand_name(rng, L)
            if w not in corpus_vocab_set and w not in cand:
                cand.append(w)
        if len(cand) < REAL6_SLOTS:          # e.g. L=1: the corpus contains every single letter
            fell_back.append(L)
            cand = list(seen[L]) if seen[L] else [_rand_name(rng, L) for _ in range(REAL6_SLOTS)]
        ood[L] = cand
    return seen, ood, fell_back


def _real6_view(src, names, others, mode, banks, rng):
    """One view of one snippet, or None if it cannot be built (the snippet is then dropped
    from EVERY view, so all views are scored on exactly the same positions)."""
    # Only real SYMBOLS can be collided with. Attribute names, keyword-argument names and dict
    # keys live in other namespaces, so a local that happens to share a spelling with one of them
    # is harmless -- checking against every token in the file instead would reject most snippets.
    taken = set(others)
    if mode == 'identity':
        return src
    if mode == 'constants':
        import ast
        try:
            out = _real6_constants(src, rng)
            ast.parse(out)
        except Exception:
            return None
        return out
    if mode == 'permute':
        if len(names) < 2:
            return None
        perm = list(names)
        for _ in range(64):
            rng.shuffle(perm)
            if all(a != b for a, b in zip(names, perm)):
                break
        if any(a == b for a, b in zip(names, perm)):
            return None
        mp = dict(zip(names, perm))
        out = _real6_apply(src, mp)
        if not _alpha_ok(src, out, mp):
            return None
        # The SET of identifiers must be identical -- that is what "no novel name" means. The
        # multiset of OCCURRENCES is not preserved and cannot be: permuting a name used 3 times
        # with one used 5 times swaps those counts. (An earlier version of this check compared
        # the sorted occurrence list and threw away 370 of 400 snippets. The token-count effect
        # it was worrying about is measured directly instead, as `mean_tokens` per view.)
        if set(IDENT_RE.findall(out)) != set(IDENT_RE.findall(src)):
            return None
        return out
    # A placeholder bank is shared corpus-wide on purpose (that is what creates a common
    # displacement direction), but a frequent real identifier can collide with a global this
    # particular snippet also uses, which would silently shadow it. Rotating the whole snippet
    # through the bank keeps the bank shared while dodging the collision; only if every rotation
    # collides is the snippet dropped.
    for rot in range(REAL6_SLOTS if mode == 'ph_seen' else 1):
        mp = _real6_map(src, names, taken, mode, banks, rng, rot)
        if mp is not None:
            out = _real6_apply(src, mp)
            if _alpha_ok(src, out, mp):
                return out
    return None


def _alpha_ok(src, out, mp):
    """Prove the rename was a pure alpha-renaming: map it back and demand the original.

    This is not a formality. If a new name captured a global the snippet also uses, the inverse
    map rewrites THAT occurrence too and the round trip no longer reproduces the source, so the
    snippet is caught and dropped. It also catches a mapping that missed a binding site.
    """
    import ast
    inv = {v: k for k, v in mp.items()}
    if len(inv) != len(mp):
        return False
    try:
        back = _real6_apply(out, inv)
        ast.parse(out)
    except Exception:
        return False
    return back == src


def _real6_map(src, names, taken, mode, banks, rng, rot):
    mp = {}
    for i, v in enumerate(names):
        if mode == 'samelen':
            new, guard = _rand_name(rng, len(v)), 0
            while (new in taken or new in mp.values()) and guard < 256:
                new = _rand_name(rng, len(v))
                guard += 1
            if new in taken or new in mp.values():
                return None
        else:
            bank = banks[0] if mode == 'ph_seen' else banks[1]
            row = bank.get(min(len(v), REAL6_LMAX))
            if not row:
                return None
            new = row[(i + rot) % REAL6_SLOTS]
        mp[v] = new
    if len(set(mp.values())) != len(mp):     # two variables collided onto one placeholder
        return None
    if set(mp.values()) & taken:
        return None                          # a placeholder would shadow a global of this snippet
    return mp


def _real6_corpus(n_items, min_vars=3, seed=20260916):
    """CodeSearchNet PYTHON test functions that survive every filter, in stream order.

    The filter depends only on the source, never on the model, so every model below is scored on
    a byte-identical snippet set -- which is what makes the rows comparable.
    """
    # Cached to ONE file, because the corpus must be byte-identical across models and across
    # jobs, and because `streaming=True` needs the network, which a compute node may not have.
    cache = os.path.join(CACHE, 'real6_corpus.json')
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        if d['n_items'] == n_items and d['min_vars'] == min_vars and d['lmax'] == REAL6_LMAX:
            print('real6 corpus from cache: %d snippets (%d scanned)'
                  % (len(d['recs']), d['scanned']), flush=True)
            return d['recs'], d['scanned']
        # A SMALLER request must never clobber a larger cached corpus. The records are in stream
        # order and the filter is model-independent, so the first n of a larger cache is exactly
        # what a smaller run would have produced -- return a prefix instead of rebuilding.
        # (Twice in this project a reduced-n smoke test silently truncated the shared cache from
        # 2000 snippets to 40, invalidating every cross-model comparison that reads it.)
        if (d['min_vars'] == min_vars and d['lmax'] == REAL6_LMAX
                and d['n_items'] > n_items):
            print('real6 corpus: serving %d of %d cached snippets (cache left intact)'
                  % (n_items, d['n_items']), flush=True)
            return d['recs'][:n_items], d['scanned']
    from datasets import load_dataset
    ds = load_dataset('code_search_net', 'python', split='test', streaming=True)
    recs, scanned = [], 0
    for r in ds:
        scanned += 1
        code = r['func_code_string']
        if not (120 < len(code) < 2000):
            continue
        p = _real6_parse(code)
        if p is None:
            continue
        src, names, others, ints = p
        if len(names) < min_vars or ints < 1:
            continue
        if max(len(v) for v in names) > REAL6_LMAX:
            continue
        recs.append({'src': src, 'names': names, 'others': others, 'ints': ints})
        if len(recs) >= n_items:
            break
        if scanned > 400000:
            break
    os.makedirs(CACHE, exist_ok=True)
    with open(cache, 'w') as f:
        json.dump({'n_items': n_items, 'min_vars': min_vars, 'lmax': REAL6_LMAX,
                   'scanned': scanned, 'recs': recs}, f)
    print('real6 corpus: %d snippets from %d scanned -> %s' % (len(recs), scanned, cache),
          flush=True)
    return recs, scanned


def _real6_build(recs, seed):
    """All six views, twice (two independent draws), plus the audit of the banks."""
    vocab_by_len, vocab_set = {}, set()
    for r in recs:
        for t in IDENT_RE.findall(r['src']):
            vocab_set.add(t)
    counts = {}
    for r in recs:
        for v in r['names']:
            counts[v] = counts.get(v, 0) + 1
    for v in sorted(counts, key=lambda x: -counts[x]):
        if 1 <= len(v) <= REAL6_LMAX:
            vocab_by_len.setdefault(len(v), []).append(v)
    draws, audit = {}, {}
    for tag, off in (('a', 0), ('b', 997)):
        rng_bank = random.Random(seed + 7919 * off)
        banks = _real6_banks(vocab_by_len, rng_bank, vocab_set)
        audit['bank_fallback_lengths_' + tag] = banks[2]
        for m in REAL6_MODES:
            col = []
            for i, r in enumerate(recs):
                rng = random.Random((seed + off) * 1_000_003 + i * 131
                                    + 7919 * REAL6_MODES.index(m))
                col.append(_real6_view(r['src'], r['names'], r['others'], m, banks[:2], rng))
            draws[(m, tag)] = col
    keep = [i for i in range(len(recs)) if all(v[i] is not None for v in draws.values())]
    audit['n_before_view_drop'] = len(recs)
    audit['n_kept'] = len(keep)
    return ({k: [v[i] for i in keep] for k, v in draws.items()},
            [recs[i]['src'] for i in keep],
            [len(recs[i]['names']) for i in keep], audit)


def real6(model_id, device, batch_size, n=REAL6_N, seed=564, corpus=None):
    """The six views on one real encoder."""
    recs, scanned = (corpus if corpus is not None else _real6_corpus(n))
    views, codes, nvars, audit = _real6_build(recs, seed)
    tok, enc = load_encoder(model_id, device)
    blank = [set() for _ in codes]
    E0, _ = embed(tok, enc, codes, blank, device, batch_size, normalize=False)
    tag = _tag(model_id)
    out = {'model': model_id, 'task': 'codesearchnet_python_paired_selfretrieval',
           'n': len(codes), 'seed': seed, 'block_size': BLOCK_SIZE,
           'mean_renameable_vars': float(np.mean(nvars)), 'scanned': scanned,
           'audit': audit, 'views': {}}
    print('%s: %d CodeSearchNet Python functions, %.2f renameable locals each'
          % (model_id, len(codes), np.mean(nvars)), flush=True)
    half = len(codes) // 2
    A, B = np.arange(half), np.arange(half, len(codes))
    for m in REAL6_MODES:
        ta, tb = views[(m, 'a')], views[(m, 'b')]
        PA, _ = embed(tok, enc, ta, blank, device, batch_size, normalize=False)
        PB, _ = embed(tok, enc, tb, blank, device, batch_size, normalize=False)
        hit = _paired(PA, E0)
        d = unit(PA) - unit(E0)
        mu = d[A].mean(0)
        _, sv, Vt = np.linalg.svd(d[A] - mu, full_matrices=False)
        ln = np.array([len(tok(t, truncation=True, max_length=BLOCK_SIZE)['input_ids'])
                       for t in ta])
        row = {
            'paired_top1': float(hit.mean()),
            'paired_top1_heldout_half': float(hit[B].mean()),
            'self_cos': float((unit(PA) * unit(E0)).sum(1).mean()),
            'bg_cos': pair_cos(PA), 'bg_cos_orig': pair_cos(E0),
            'sym_indep': float(_paired(PA, PB).mean()),
            'shared_mean_frac': float((mu ** 2).sum() / max((d[A] ** 2).sum(1).mean(), 1e-12)),
            'frac_text_changed': float(np.mean([x != y for x, y in zip(ta, codes)])),
            'frac_draws_differ': float(np.mean([x != y for x, y in zip(ta, tb)])),
            'mean_tokens': float(ln.mean()), 'frac_truncated': float((ln >= BLOCK_SIZE).mean()),
        }
        for k, U in [('mean', None), ('pc4', Vt[:4]), ('pc8', Vt[:8])]:
            if U is None:
                qq, cc = unit(unit(PA) - mu), unit(E0)
            else:
                qq = unit(PA); qq = unit(qq - (qq @ U.T) @ U)
                cc = unit(E0); cc = unit(cc - (cc @ U.T) @ U)
            hh = np.empty(len(B), dtype=bool)
            for s in range(0, len(B), 512):
                sl = slice(s, min(s + 512, len(B)))
                idx = B[sl]
                hh[sl] = (qq[idx] @ cc.T).argmax(1) == idx
            row['corrected_' + k] = float(hh.mean())
        out['views'][m] = row
        print('  %-12s paired %.4f (half B %.4f)  sym_indep %.4f  self-cos %.4f  mfrac %.4f'
              '  changed %.3f  draws-differ %.3f  tok %.1f'
              % (m, row['paired_top1'], row['paired_top1_heldout_half'], row['sym_indep'],
                 row['self_cos'], row['shared_mean_frac'], row['frac_text_changed'],
                 row['frac_draws_differ'], row['mean_tokens']), flush=True)
    sl, pm = out['views']['samelen']['paired_top1'], out['views']['permute']['paired_top1']
    out['binding_fraction'] = float((1 - pm) / (1 - sl)) if sl < 1 else None
    out['novelty_gap'] = float(sl - pm)
    print('  binding_fraction %s  novelty_gap %+.4f'
          % (out['binding_fraction'], out['novelty_gap']), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'whyscale_real6__%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path, flush=True)
    del enc
    return out


def glob_pth(d):
    import glob as _g
    fs = _g.glob(os.path.join(d, 'checkpoint_*.pth'))
    if not fs:
        raise SystemExit('no checkpoint_*.pth in ' + d)
    return sorted(fs, key=lambda f: float(re.search(r'checkpoint_([0-9.]+)\.pth', f).group(1)))


def merge():
    """Fold every shard into one record per model. Shards are named <model><shard>.json."""
    out = {}
    for f in sorted(os.listdir(CACHE)):
        if not (f.startswith('whyscale') and f.endswith('.json')):
            continue
        with open(os.path.join(CACHE, f)) as fh:
            d = json.load(fh)
        key = f[10:-5]
        if 'model' in d:
            key = d['model'].split('/')[-1]
            if f.startswith('whyscale_csn__'):
                key += '__csn'
            elif f.startswith('whyscale_toks__'):
                key += '__toks'
            elif f.startswith('whyscale_real6__'):
                key += '__real6'
        if key not in out:
            out[key] = d
        else:
            out[key].setdefault('conditions', {}).update(d.get('conditions', {}))
            out[key].setdefault('views', {}).update(d.get('views', {}))
    path = os.path.join(WORK, 'whyscale.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path, list(out))



# ===========================================================================
# permtrace: is the DECODER TRACING metric blind to the binding component?
# ===========================================================================
# `tiny` above measures retrieval geometry of the tiny decoders' mean-pooled residual and finds
# the damage splits in two: a NOVELTY component (samelen/placeholder_ood, coverage-sensitive,
# with a shared displacement direction) and a BINDING component (permute: a derangement of the
# program's OWN identifiers -- identical character multiset, identical tokenisation, zero novel
# strings -- which is coverage-INSENSITIVE and has no shared direction).
#
# REGISTERED PREDICTION (labnotes 2026-09-16, written before this code ran): decoder tracing
# end-accuracy on `permute`d programs is ~1.000 at every q >= 10%, i.e. indistinguishable from
# tracing on unmodified programs, because a permutation is a bijective renaming into names the
# model has already memorised and the tracing task never has to decide WHICH variable a name
# denotes beyond copying it. If tracing drops on permuted programs the blindness account fails.
#
# The crux is the GROUND TRUTH TRACE. A permuted program has a different correct trace, so the
# original cannot be reused. Traces are regenerated with tinypy_gen.TinyPyCodeTracer, the same
# tracer that made the corpus. Two independent checks are computed and stored per model:
#   identity_trace_matches_corpus  -- regenerating the unmodified code must reproduce the corpus
#                                     example byte for byte (ast.unparse is a normalising round
#                                     trip on this grammar, so it does).
#   permute_trace_matches_subst    -- a bijective rename is semantics-preserving, so the
#                                     regenerated permuted trace must equal the ORIGINAL corpus
#                                     example with the same whole-word name substitution applied
#                                     to the whole text, trace included. This check never calls
#                                     the tracer on the permuted code, so it is independent of it.
# If either drops below 1.0 the regeneration is wrong and the numbers mean nothing.

# REPRODUCIBILITY. `_tiny_rename`'s `permute` and `constants` branches seed a local RNG from
# `hash(...)` of a str/tuple, and CPython randomises string hashing per process, so both views
# differ from run to run unless PYTHONHASHSEED is fixed. permtrace_all refuses to run without it.
# (Two consequences for the retrieval table computed by `tiny`, which uses the same function:
# its permute/constants views were one unrecorded draw, and because the permute branch ignores
# both `banks` and `ren`, draw b IS draw a -- permute's sym_indep == 1.0000 is therefore an
# identity, not evidence that the displacement is reproducible.)
PERMTRACE_MODES = ('identity', 'samelen', 'permute', 'constants')
# ckpt_p10s is excluded: work/data_p10s has no lexicon.tsv, so `_tiny_seen_set` cannot verify
# that a same-length replacement is unseen there, and the view set would not be comparable.
PERMTRACE_RUNS = ('ckpt_p0.1s', 'ckpt_p1s', 'ckpt_p10', 'ckpt_p20', 'ckpt_p40', 'ckpt_p70')


def _permtrace_subst(text, mp):
    """Whole-word identifier substitution over an entire example (code AND trace)."""
    if not mp:
        return text
    pat = re.compile(r'(?<![a-z])(%s)(?![a-z])' % '|'.join(sorted(mp, key=len, reverse=True)))
    return pat.sub(lambda m: mp[m.group(1)], text)


def _permtrace_map(code):
    """The permute derangement `_tiny_rename` uses, recovered for the substitution check."""
    import ast
    names = sorted({n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)})
    return _derangement(names, 0)


def permtrace(ckpt, device='cuda', batch_size=64, n=TINY_N, seed=564):
    """Tracing accuracy on the same four views of the same snippets `tiny` scores by retrieval."""
    import torch as T
    import interp
    import tinypy_gen as G
    codes, corpus_path = _tiny_examples(ckpt, n)
    data_dir = os.path.dirname(corpus_path)
    run = os.path.dirname(ckpt) if ckpt.endswith('.pth') else ckpt
    pth = ckpt if ckpt.endswith('.pth') else glob_pth(ckpt)[-1]
    tag = os.path.basename(os.path.normpath(run))

    # The full corpus examples (code + trace) indexed the same way as `codes`, for the checks.
    raw = open(corpus_path).read().split('\n\n')[:-1][:n]

    # Exactly the view construction `tiny` uses, so the snippet set is the SAME one the retrieval
    # table was computed on (six modes, two draws, drop a snippet if ANY view fails).
    views, kept_codes, keep_idx = _tiny_build(codes, data_dir, seed, return_keep=True)
    raw = [raw[i] for i in keep_idx]

    tracer = G.TinyPyCodeTracer()
    traced, err_counts = {}, {}
    for m in PERMTRACE_MODES:
        outs, bad = [], 0
        for c in views[(m, 'a')]:
            t, e, _ = tracer.trace_snippet(c)
            if e is not None:
                outs.append(None); bad += 1; continue
            try:
                interp.TOK.encode(t + '\n\n')
            except KeyError:
                outs.append(None); bad += 1; continue
            outs.append(t)
        traced[m] = outs
        err_counts[m] = bad
    # Two snippet sets. `ok` is aligned across ALL FOUR views and is the primary table: every
    # view is scored on byte-identical positions. `ok3` drops only identity/samelen/permute
    # failures, so the headline permute-vs-identity contrast is not paid for by the ~15% of
    # snippets whose constants view puts a value outside the tracer's [-999, 999] range.
    ok = [i for i in range(len(kept_codes))
          if all(traced[m][i] is not None for m in PERMTRACE_MODES)]
    ok3 = [i for i in range(len(kept_codes))
           if all(traced[m][i] is not None for m in ('identity', 'samelen', 'permute'))]
    traced3 = {m: [traced[m][i] for i in ok3] for m in ('identity', 'samelen', 'permute')}
    traced = {m: [traced[m][i] for i in ok] for m in PERMTRACE_MODES}
    raw = [raw[i] for i in ok]
    src = [kept_codes[i] for i in ok]

    # --- the two independent correctness checks on the regenerated traces -------------------
    id_match = float(np.mean([a == b for a, b in zip(traced['identity'], raw)]))
    # Did replacing the integer literals flip a branch? If it did, the correct trace has a
    # different number of #STEP blocks and the model has to predict different control flow --
    # which is why `constants` is NOT a null edit for a tracer even though it is nearly one for
    # a mean-pooled representation.
    cf_changed = [t.count('#STEP') != o.count('#STEP')
                  for t, o in zip(traced['constants'], raw)]
    sub_hits, sub_n = 0, 0
    for c, orig, perm in zip(src, raw, traced['permute']):
        mp = _permtrace_map(c)
        if mp is None:
            continue
        sub_n += 1
        sub_hits += int(_permtrace_subst(orig, mp) == perm)
    perm_match = float(sub_hits / max(1, sub_n))

    device = T.device(device) if isinstance(device, str) else device
    model = interp.load_model(pth, device)
    out = {'ckpt': pth, 'run': run, 'corpus': corpus_path, 'n': len(src), 'seed': seed,
           'n_ipm3': len(ok3),
           'n_built': len(kept_codes), 'trace_errors': err_counts,
           'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
           'checks': {'identity_trace_matches_corpus': id_match,
                      'permute_trace_matches_subst': perm_match,
                      'n_permute_checked': sub_n},
           'constants_control_flow_changed': int(sum(cf_changed)),
           'views': {}}
    print('%s: %d snippets (%d built) from %s | identity==corpus %.4f  permute==subst %.4f'
          % (tag, len(src), len(kept_codes), corpus_path, id_match, perm_match), flush=True)
    for m in PERMTRACE_MODES:
        ex = traced[m]
        r = interp.score_tf(model, ex, device, batch=batch_size)
        b = interp.boundary_discrimination(model, ex, device, batch=batch_size)
        row = {'end_acc': float(r['hard'].mean()),
               'nll': float(r['nll'].mean()),
               'boundary_cont': b['cont'], 'boundary_end': b['end'], 'boundary_bal': b['bal'],
               'boundary_exact': b['exact'], 'n_cont': b['n_cont'], 'n_end': b['n_end'],
               'frac_text_changed': float(np.mean([x != y for x, y in zip(views[(m, 'a')], src)])),
               'first_fail_hist': {c: int((r['first_fail'] == i).sum())
                                   for i, c in enumerate(interp.CLASSES)},
               'class_acc': {c: float(r['class_correct'][i] / max(1, r['class_total'][i]))
                             for i, c in enumerate(interp.CLASSES)},
               'class_total': {c: int(r['class_total'][i]) for i, c in enumerate(interp.CLASSES)},
               # token the model SHOULD have emitted at its first mistake, so a failure can be
               # localised offline without another forward pass
               'first_fail_tok': [interp.TOK.decod_map[interp.TOK.encode(e + '\n\n')[
                                      interp.encode_prompt_len(interp.TOK.encode(e + '\n\n')) + int(fp)]]
                                  if fp >= 0 else None
                                  for e, fp in zip(ex, r['first_fail_pos'])],
               'bits': ''.join(map(str, r['hard'].tolist()))}
        if m == 'constants':
            h = r['hard']
            cc = np.array(cf_changed)
            row['end_acc_cf_same'] = float(h[~cc].mean()) if (~cc).any() else float('nan')
            row['end_acc_cf_changed'] = float(h[cc].mean()) if cc.any() else float('nan')
            row['n_cf_changed'] = int(cc.sum())
        if m in traced3:
            r3 = interp.score_tf(model, traced3[m], device, batch=batch_size)
            row['end_acc_ipm3'] = float(r3['hard'].mean())
        out['views'][m] = row
        print('  %-11s end_acc %.4f (ipm3 %s)  nll %.4f  cont %.3f end %.3f bal %.3f'
              % (m, row['end_acc'],
                 ('%.4f' % row['end_acc_ipm3']) if 'end_acc_ipm3' in row else '-',
                 row['nll'], b['cont'], b['end'], b['bal']), flush=True)
    return out


def permtrace_all(runs, device='cuda', batch_size=64, n=TINY_N, seed=564):
    if os.environ.get('PYTHONHASHSEED') is None:
        raise SystemExit('set PYTHONHASHSEED (e.g. 0): permute/constants are hash-seeded')
    path = os.path.join(WORK, 'permtrace.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    for r in runs:
        d = r if os.path.isabs(r) else os.path.join(WORK, r)
        key = os.path.basename(os.path.normpath(d))
        out[key] = permtrace(os.path.join(d, 'checkpoint_1.00.pth'), device, batch_size, n, seed)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('wrote', path, flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'csn', 'toks', 'sym', 'views2', 'tiny',
                                    'real6', 'collide', 'e3', 'merge', 'permtrace'])
    ap.add_argument('--model', default='microsoft/unixcoder-base')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--conditions', default='', help='comma-separated subset of CONDITIONS')
    ap.add_argument('--no-views', action='store_true')
    ap.add_argument('--shard', default='')
    a = ap.parse_args()
    lim = a.limit or None
    if a.cmd == 'run':
        only = set(x for x in a.conditions.split(',') if x) or None
        run(a.model, lim, a.device, a.batch_size, only, not a.no_views, a.shard)
    elif a.cmd == 'csn':
        csn(a.model, lim or 1000, a.device, a.batch_size)
    elif a.cmd == 'toks':
        toks(a.model, lim)
    elif a.cmd == 'sym':
        sym(a.model, lim, a.device, a.batch_size,
            tuple(x for x in a.conditions.split(',') if x) or None)
    elif a.cmd == 'views2':
        views2(a.model, lim, a.device, a.batch_size)
    elif a.cmd == 'tiny':
        tiny(a.model, a.device, a.batch_size, lim or TINY_N)
    elif a.cmd == 'real6':
        # One corpus build, reused for every model, so the rows are strictly comparable.
        corpus = _real6_corpus(lim or REAL6_N)
        for mid in [x for x in a.model.split(',') if x]:
            done = os.path.join(CACHE, 'whyscale_real6__%s.json' % _tag(mid))
            if os.path.exists(done):        # resume: a requeued job keeps what it already has
                print('skip (exists):', done, flush=True)
                continue
            real6(mid, a.device, a.batch_size, lim or REAL6_N, corpus=corpus)
    elif a.cmd == 'collide':
        collide(a.model, a.device, a.batch_size, lim or TINY_N)
    elif a.cmd == 'permtrace':
        permtrace_all([x for x in a.conditions.split(',') if x] or PERMTRACE_RUNS,
                      a.device, a.batch_size, lim or TINY_N)
    elif a.cmd == 'e3':
        e3(lim, tuple(x for x in a.conditions.split(',') if x) or None)
    else:
        merge()


if __name__ == '__main__':
    sys.exit(main())


# ---------------------------------------------------------------------------
# binding_acc: a decoder-side readout that SEES the binding component
# ---------------------------------------------------------------------------
# Whole-trace end-accuracy is ~99% copying, so a name->role binding error is diluted into a
# handful of tokens (previous section: `value` class accuracy 0.99997 -> 0.99894 under permute).
# These functions restrict scoring to exactly the tokens that carry binding: the VALUE field of
# each `name?value` entry of each state dump, conditioned teacher-forced on the name. Copy
# tokens, structural tokens, identifier tokens and the instruction pointer are excluded.
#
# Everything here is additive; nothing above is modified.

BINDING_MODES = ('identity', 'samelen', 'permute', 'constants')
_BINDING_STOP = ('\n', '\n#STEP\n', '\n\n')


def _binding_scan(ids, n_prompt, dec):
    """Parse the state dumps of an encoded trace straight off the token stream.

    -> ordered list of {'step', 'name', 'val', 'vpos', 'end'} -- one per `name?value` entry.
       `vpos` are the token indices of the value characters; `end` is the index of the `;` or
       newline that closes it (-1 if the example ends there).

    Written from the token stream rather than from `interp.token_classes` because the character
    tokenizer emits `if` as a SINGLE token, so a name containing `if` breaks the
    previous-token-is-a-letter rule `token_classes` uses. `_binding_eval` reports how often the
    two agree (`cls_agree`), which is the check that this parser is the same object.
    """
    out, step, i, N = [], 0, n_prompt, len(ids)
    while i < N:
        s = dec[ids[i]]
        if s == '\n#STEP\n':
            step += 1
            i += 1
            continue
        if s != '$':
            i += 1
            continue
        i += 1                                     # '$' opens the state dump
        while i < N and dec[ids[i]] not in _BINDING_STOP:
            nm = []
            while i < N and dec[ids[i]] not in ('?', ';') and dec[ids[i]] not in _BINDING_STOP:
                nm.append(dec[ids[i]])
                i += 1
            if i >= N or dec[ids[i]] != '?':
                break                              # malformed entry: give up on this dump
            i += 1                                 # skip '?'
            vp = []
            while i < N and dec[ids[i]] != ';' and dec[ids[i]] not in _BINDING_STOP:
                vp.append(i)
                i += 1
            if not vp:
                break
            out.append({'step': step, 'name': ''.join(nm),
                        'val': ''.join(dec[ids[t]] for t in vp),
                        'vpos': vp, 'end': i if i < N else -1})
            if i < N and dec[ids[i]] == ';':
                i += 1
    return out


def _binding_map(code_a, code_b):
    """The name mapping that carries `code_a` to `code_b`, read off the two syntax trees.

    NOT `_permtrace_map`. That helper replays `random.Random(hash(tuple(names)))`, which is what
    `_tiny_rename` used when `permtrace` was written; `_tiny_rename` has since been migrated to
    `_stable_seed` (the PYTHONHASHSEED retraction) and the helper was not, so it now reproduces
    the real derangement on only ~39% of snippets. Reading the mapping off the trees instead is
    independent of the generator's RNG and cannot go stale. Returns None if the two trees are not
    the same shape or the mapping is not a bijection.
    """
    import ast
    try:
        na = [n for n in ast.walk(ast.parse(code_a)) if isinstance(n, ast.Name)]
        nb = [n for n in ast.walk(ast.parse(code_b)) if isinstance(n, ast.Name)]
    except SyntaxError:
        return None
    if len(na) != len(nb) or not na:
        return None
    mp = {}
    for x, y in zip(na, nb):
        if mp.setdefault(x.id, y.id) != y.id:
            return None                       # one name went to two different names
    if len(set(mp.values())) != len(mp):
        return None                           # two names collapsed onto one
    return mp


def _binding_eval(model, examples, device, batch=32, mps=None):
    """Teacher-forced binding readout over a list of traces.

    `mps[i]` is the permute derangement of snippet i (original name -> permuted name), or None.
    The name-bound distractor for a slot named `b` is the slot named `mp[b]` in the SAME dump:
      * on the permute view b = mp[a], so mp[b] names the variable that ORIGINALLY owned `b`;
      * on the identity view the same rule picks the SAME pair of variables of the same dump, so
        identity is a matched reference for the margin rather than a trivial zero.
    Not meaningful for `samelen` (names are novel, no variable ever owned them) -- pass mps=None.
    """
    import torch as T
    import interp
    dec = interp.TOK.decod_map
    VAL = interp.CLASSES.index('value')
    enc = [interp.encode(e) for e in examples]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]), reverse=True)
    n_slot = n_first = n_exact = n_digits = 0
    # A state dump repeats EVERY live variable at EVERY step, so most slots are a verbatim copy
    # of the same name's value one dump earlier. Those need no lookup at all. A slot is `new`
    # when the name was absent from the previous dump or its value changed -- i.e. the model has
    # to read the marked line, find its operands' values, and write a value it has not just been
    # shown. Those are the genuine binding events and they are ~1/5 of all slots.
    n_new = n_first_new = n_exact_new = 0
    per_n = np.zeros(len(enc), np.int32)
    per_first = np.zeros(len(enc), np.int32)
    per_exact = np.zeros(len(enc), np.int32)
    per_n_new = np.zeros(len(enc), np.int32)
    per_first_new = np.zeros(len(enc), np.int32)
    mg, win, n_pair_found, n_pair_tied = [], 0, 0, 0
    mgn, winn = [], 0
    # Probability DIFFERENCE saturates: with p(target) ~ 1 - 1e-5 the difference is 1.0000 for
    # any distractor and carries no information. The log-odds log p_target - log p_distractor
    # keeps resolving after the difference has pinned, which is the regime these models are in.
    lg, lgn = [], []
    agree_ok = agree_n = 0
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        logits, lens = interp.tf_logits(model, [enc[i][0] for i in idx], device)
        pred = logits.argmax(-1).cpu().numpy()
        prob = T.softmax(logits, -1)
        for j, i in enumerate(idx):
            ids, npmt = enc[i]
            a = np.asarray(ids)
            slots = _binding_scan(ids, npmt, dec)
            if not slots:
                continue
            cls = interp.token_classes(ids, npmt)
            prev, cur, cstep = {}, {}, slots[0]['step']
            for sl in slots:
                if sl['step'] != cstep:
                    prev, cur, cstep = cur, {}, sl['step']
                sl['new'] = prev.get(sl['name']) != sl['val']
                cur[sl['name']] = sl['val']
            fp = np.array([sl['vpos'][0] for sl in slots], np.int64)
            agree_ok += int((cls[fp] == VAL).sum())
            agree_n += len(fp)
            ok_first = pred[j, fp - 1] == a[fp]
            n_slot += len(fp)
            n_first += int(ok_first.sum())
            per_n[i] = len(fp)
            per_first[i] = int(ok_first.sum())
            isnew = np.array([sl['new'] for sl in slots], bool)
            n_new += int(isnew.sum())
            n_first_new += int((ok_first & isnew).sum())
            per_n_new[i] = int(isnew.sum())
            per_first_new[i] = int((ok_first & isnew).sum())
            ex_ok = dg_ok = 0
            for k, sl in enumerate(slots):
                p = np.array(sl['vpos'], np.int64)
                d_ok = bool((pred[j, p - 1] == a[p]).all())
                dg_ok += int(d_ok)
                if sl['end'] >= 0:
                    e = sl['end']
                    w = int(d_ok and pred[j, e - 1] == a[e])
                else:
                    w = int(d_ok)
                ex_ok += w
                if sl['new']:
                    n_exact_new += w
            n_exact += ex_ok
            n_digits += dg_ok
            per_exact[i] = ex_ok
            if mps is not None and mps[i]:
                mp = mps[i]
                bystep = {}
                for k, sl in enumerate(slots):
                    bystep.setdefault(sl['step'], {})[sl['name']] = k
                pos, tt, dd, nw = [], [], [], []
                for sl in slots:
                    dn = mp.get(sl['name'])
                    if dn is None:
                        continue
                    kj = bystep[sl['step']].get(dn)
                    if kj is None:
                        continue
                    n_pair_found += 1
                    # Compare at the first character where the two values DIFFER, not blindly at
                    # the first character. The shared prefix is byte-identical, so at that offset
                    # the teacher-forced context is exactly the prefix of BOTH values and the two
                    # probabilities are strictly comparable. Values are compared with their
                    # closing token appended, so `6` vs `62` contrasts `;` against `2`. Without
                    # this ~45% of pairs are thrown away for sharing a leading digit.
                    st = sl['val'] + (dec[a[sl['end']]] if sl['end'] >= 0 else '')
                    sd_ = slots[kj]
                    sdv = sd_['val'] + (dec[a[sd_['end']]] if sd_['end'] >= 0 else '')
                    L = next((x for x in range(min(len(st), len(sdv))) if st[x] != sdv[x]), -1)
                    if L < 0 or L >= len(sl['vpos']) + (1 if sl['end'] >= 0 else 0):
                        n_pair_tied += 1          # identical, or differ only past the target
                        continue
                    tpos = (sl['vpos'] + ([sl['end']] if sl['end'] >= 0 else []))[L]
                    t_tok, d_tok = int(a[tpos]), interp.TOK.encod_map[sdv[L]]
                    if t_tok == d_tok:
                        n_pair_tied += 1
                        continue
                    pos.append(tpos - 1)
                    tt.append(t_tok)
                    dd.append(d_tok)
                    nw.append(sl['new'])
                if pos:
                    P = T.tensor(pos, device=prob.device)
                    rows = prob[j].index_select(0, P)
                    pt = rows.gather(1, T.tensor(tt, device=prob.device)[:, None])[:, 0]
                    pd = rows.gather(1, T.tensor(dd, device=prob.device)[:, None])[:, 0]
                    d = (pt - pd).cpu().numpy()
                    lo = (T.log(pt.clamp_min(1e-30)) - T.log(pd.clamp_min(1e-30))).cpu().numpy()
                    mg.append(d)
                    lg.append(lo)
                    win += int((d > 0).sum())
                    nw = np.array(nw, bool)
                    if nw.any():
                        mgn.append(d[nw])
                        lgn.append(lo[nw])
                        winn += int((d[nw] > 0).sum())
    _cat = lambda v: np.concatenate(v) if v else np.zeros(0, np.float32)
    mg, mgn, lg, lgn = _cat(mg), _cat(mgn), _cat(lg), _cat(lgn)
    ok = per_n > 0
    okn = per_n_new > 0
    out = {
        'n_slots': int(n_slot),
        'binding_acc_first': float(n_first / max(1, n_slot)),
        'binding_acc_exact': float(n_exact / max(1, n_slot)),
        'binding_acc_digits': float(n_digits / max(1, n_slot)),
        'binding_acc_first_macro': float((per_first[ok] / per_n[ok]).mean()) if ok.any() else 0.0,
        'binding_acc_exact_macro': float((per_exact[ok] / per_n[ok]).mean()) if ok.any() else 0.0,
        'snippet_all_slots_first': float((per_first[ok] == per_n[ok]).mean()) if ok.any() else 0.0,
        'cls_agree': float(agree_ok / max(1, agree_n)),
        'cls_agree_n': int(agree_n),
        'n_slots_new': int(n_new),
        'binding_acc_first_new': float(n_first_new / max(1, n_new)),
        'binding_acc_exact_new': float(n_exact_new / max(1, n_new)),
        'n_err_first': int(n_slot - n_first),
        'n_err_first_new': int(n_new - n_first_new),
        'n_err_exact_new': int(n_new - n_exact_new),
        'snippet_all_new_slots_first': (float((per_first_new[okn] == per_n_new[okn]).mean())
                                        if okn.any() else 0.0),
    }
    if mps is not None:
        out.update({
            'margin_n': int(len(mg)),
            'margin_pairs_found': int(n_pair_found),
            'margin_pairs_tied': int(n_pair_tied),
            'binding_margin': float(mg.mean()) if len(mg) else float('nan'),
            'binding_margin_median': float(np.median(mg)) if len(mg) else float('nan'),
            'binding_margin_p10': float(np.percentile(mg, 10)) if len(mg) else float('nan'),
            'binding_margin_p90': float(np.percentile(mg, 90)) if len(mg) else float('nan'),
            'binding_win': float(win / max(1, len(mg))),
            'margin_n_new': int(len(mgn)),
            'binding_margin_new': float(mgn.mean()) if len(mgn) else float('nan'),
            'binding_margin_new_median': float(np.median(mgn)) if len(mgn) else float('nan'),
            'binding_margin_new_p10': float(np.percentile(mgn, 10)) if len(mgn) else float('nan'),
            'binding_win_new': float(winn / max(1, len(mgn))),
            'binding_logmargin': float(lg.mean()) if len(lg) else float('nan'),
            'binding_logmargin_median': float(np.median(lg)) if len(lg) else float('nan'),
            'binding_logmargin_p10': float(np.percentile(lg, 10)) if len(lg) else float('nan'),
            'binding_logmargin_new': float(lgn.mean()) if len(lgn) else float('nan'),
            'binding_logmargin_new_median': (float(np.median(lgn)) if len(lgn)
                                             else float('nan')),
            'binding_logmargin_new_p10': (float(np.percentile(lgn, 10)) if len(lgn)
                                          else float('nan')),
            'binding_logmargin_new_frac_neg': (float((lgn < 0).mean()) if len(lgn)
                                               else float('nan')),
        })
    return out


def binding(ckpt, device='cuda', batch_size=32, n=TINY_N, seed=564):
    """binding_acc / binding_margin for one checkpoint, on permtrace's exact snippet set.

    The view construction, the tracer, and the four-view `ok` filter are copied from `permtrace`
    verbatim, so every number here is on the same snippets as the tracing table AND the retrieval
    table (`paired_top1` is recomputed here on those same snippets rather than read from the
    cache, so the 24-cell correlation is not paying for a snippet-set mismatch).
    """
    import torch as T
    import interp
    import tinypy_gen as G
    codes, corpus_path = _tiny_examples(ckpt, n)
    data_dir = os.path.dirname(corpus_path)
    run = os.path.dirname(ckpt) if ckpt.endswith('.pth') else ckpt
    pth = ckpt if ckpt.endswith('.pth') else glob_pth(ckpt)[-1]
    tag = os.path.basename(os.path.normpath(run))
    raw = open(corpus_path).read().split('\n\n')[:-1][:n]

    views, kept_codes, keep_idx = _tiny_build(codes, data_dir, seed, return_keep=True)
    raw = [raw[i] for i in keep_idx]

    tracer = G.TinyPyCodeTracer()
    traced, err_counts = {}, {}
    for m in BINDING_MODES:
        outs, bad = [], 0
        for c in views[(m, 'a')]:
            t, e, _ = tracer.trace_snippet(c)
            if e is not None:
                outs.append(None)
                bad += 1
                continue
            try:
                interp.TOK.encode(t + '\n\n')
            except KeyError:
                outs.append(None)
                bad += 1
                continue
            outs.append(t)
        traced[m] = outs
        err_counts[m] = bad
    ok = [i for i in range(len(kept_codes))
          if all(traced[m][i] is not None for m in BINDING_MODES)]
    traced = {m: [traced[m][i] for i in ok] for m in BINDING_MODES}
    raw = [raw[i] for i in ok]
    src = [kept_codes[i] for i in ok]
    vcode = {m: [views[(m, 'a')][i] for i in ok] for m in BINDING_MODES}
    # Read the derangement off the two trees (see `_binding_map`), then require it to be a
    # true derangement -- a fixed point would make the name-bound distractor the target.
    mps = []
    for a, b in zip(vcode['identity'], vcode['permute']):
        mp = _binding_map(a, b)
        if mp is not None and any(k == v for k, v in mp.items()):
            mp = None
        mps.append(mp)

    id_match = float(np.mean([a == b for a, b in zip(traced['identity'], raw)]))
    sub_hits = sub_n = 0
    for orig, perm, mp in zip(traced['identity'], traced['permute'], mps):
        if mp is None:
            continue
        sub_n += 1
        sub_hits += int(_permtrace_subst(orig, mp) == perm)

    device = T.device(device) if isinstance(device, str) else device
    model = interp.load_model(pth, device)
    # Binding below uses interp.TOK for every trace. An incidental bpe.json in
    # the corpus directory must not silently switch the code-embedding tokenizer.
    if model.token_embedding_table.weight.shape[0] != len(interp.TOK.keywords):
        raise ValueError('binding requires a character-tokenizer checkpoint')
    enc, tokname = interp.TOK.encode, 'char52 (checkpoint-validated)'
    E0 = _tiny_embed(model, src, device, batch_size, enc)

    out = {'ckpt': pth, 'run': run, 'corpus': corpus_path, 'n': len(src), 'seed': seed,
           'n_built': len(kept_codes), 'trace_errors': err_counts,
           'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
           'checks': {'identity_trace_matches_corpus': id_match,
                      'permute_trace_matches_subst': float(sub_hits / max(1, sub_n)),
                      'n_permute_checked': sub_n,
                      'n_with_derangement': int(sum(x is not None for x in mps))},
           'views': {}}
    print('%s: %d snippets from %s | identity==corpus %.4f  permute==subst %.4f'
          % (tag, len(src), corpus_path, id_match, out['checks']['permute_trace_matches_subst']),
          flush=True)
    for m in BINDING_MODES:
        # `samelen` renames into names no variable ever owned, so the name-bound distractor does
        # not exist; the margin is reported as absent rather than as a number.
        row = _binding_eval(model, traced[m], device, batch_size,
                            mps=None if m == 'samelen' else mps)
        PA = _tiny_embed(model, vcode[m], device, batch_size, enc)
        row['paired_top1'] = float(_paired(PA, E0).mean())
        out['views'][m] = row
        print('  %-10s slots %6d (new %5d)  first %.5f  exact %.5f | NEW first %.5f exact %.5f'
              '  logmargin %s win %s | paired %.4f'
              % (m, row['n_slots'], row['n_slots_new'], row['binding_acc_first'],
                 row['binding_acc_exact'], row['binding_acc_first_new'],
                 row['binding_acc_exact_new'],
                 ('%+.2f' % row['binding_logmargin_new']) if 'binding_logmargin_new' in row else '  -  ',
                 ('%.4f' % row['binding_win_new']) if 'binding_win_new' in row else '  -   ',
                 row['paired_top1']), flush=True)
    return out


def binding_all(runs=None, device='cuda', batch_size=32, n=TINY_N, seed=564):
    if os.environ.get('PYTHONHASHSEED') is None:
        raise SystemExit('set PYTHONHASHSEED (e.g. 0): permute/constants are hash-seeded')
    runs = runs or PERMTRACE_RUNS
    # BINDING_JSON lets two shards write to separate files and be merged afterwards; the
    # merged result is always work/binding.json and the shard files are deleted.
    path = os.environ.get('BINDING_JSON') or os.path.join(WORK, 'binding.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    for r in runs:
        d = r if os.path.isabs(r) else os.path.join(WORK, r)
        key = os.path.basename(os.path.normpath(d))
        out[key] = binding(os.path.join(d, 'checkpoint_1.00.pth'), device, batch_size, n, seed)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('wrote', path, flush=True)
    return out


# ---------------------------------------------------------------------------
# shortcut: does the decoder resolve binding FROM THE DUMP or FROM MEMORY?
# ---------------------------------------------------------------------------
# The `binding_acc` section concluded that the trace format hands the decoder the name->role
# binding in context at every step (each `#STEP` block reprints the program and then, after `$`,
# a state dump `name?value;...` of the state BEFORE the marked line), so the decoder can resolve
# an operand by matching its NAME STRING against the previous dump. That was an explanation, not
# a measurement. These functions test it three ways: by DISTANCE to the matchable occurrence, by
# isolating slots where no previous dump contains the name at all, and -- the causal one -- by
# corrupting the preceding dumps of the ORIGINAL program and scoring the untouched last dump.
#
# Everything here is additive. Nothing above is modified.

SHORTCUT_CONDS = ('intact', 'dumpperm', 'dumpvalperm', 'dumpshuf')
_DIST_BINS = (0, 8, 16, 32, 64, 128, 256, 512, 10 ** 9)


def _dump_spans(text):
    """Character spans (lo, hi) of every state-dump BODY in a trace: after `$`, up to EOL.

    `$` occurs exactly once per `#STEP` block, immediately after the marked line, and the dump
    runs to the newline that ends that line (or to the end of the example for the final block).
    """
    spans, i = [], 0
    while True:
        j = text.find('$', i)
        if j < 0:
            return spans
        k = text.find('\n', j + 1)
        if k < 0:
            k = len(text)
        spans.append((j + 1, k))
        i = k + 1


def _dump_entries(body):
    """`a?1;b?-2` -> [('a','1'), ('b','-2')]; '' -> []. None if it does not have that shape."""
    if body == '':
        return []
    out = []
    for piece in body.split(';'):
        if piece.count('?') != 1:
            return None
        nm, v = piece.split('?')
        if not nm or not v:
            return None
        out.append((nm, v))
    return out


def _dump_render(entries):
    return ';'.join('%s?%s' % (n, v) for n, v in entries)


def _corrupt_dumps(text, cond, draw=0, scope='all'):
    """Rewrite every state dump EXCEPT THE LAST one. Code is untouched, and so is the last dump,
    so the scored target string is byte-identical across conditions.

    `dumpperm`     names permuted in place by a derangement of that dump's OWN live names, values
                   left in their positions -> name m now sits beside the value of d^-1(m).
    `dumpvalperm`  the SAME induced name->value map, but names stay in their display positions and
                   the values move instead. Isolates the surface edit from the map corruption.
    `dumpshuf`     entry order shuffled, `name?value` pairs intact -> the map is PRESERVED. The
                   matched out-of-distribution control, and the name-matching vs positional-copy
                   separator.

    `scope` 'all' corrupts every dump before the scored one; 'prev' corrupts ONLY the dump
    immediately preceding it, leaving every earlier dump intact -- the recency version, which
    separates "the model reads the nearest dump" from "the model reads the dump history".

    -> (new_text, n_dumps_corrupted, prev_map) where prev_map[name] is the value the corrupted
       dump immediately preceding the last one attaches to `name` ({} if there is none).
    """
    spans = _dump_spans(text)
    if not spans:
        return None
    parsed = [_dump_entries(text[a:b]) for a, b in spans]
    if any(p is None for p in parsed):
        return None
    out, pos, n_bad, prev_map = [], 0, 0, {}
    # index of the dump immediately preceding the last NON-EMPTY dump
    nonempty = [i for i, p in enumerate(parsed) if p]
    prev_i = nonempty[-2] if len(nonempty) >= 2 else None
    for i, ((a, b), ents) in enumerate(zip(spans, parsed)):
        new = ents
        hit = (i < len(spans) - 1) if scope == 'all' else (i == prev_i)
        if cond != 'intact' and hit and len(ents) >= 2:
            names = [n for n, _ in ents]
            d = _derangement(sorted(names), draw)
            if d is None:
                pass
            elif cond == 'dumpperm':
                new = [(d[n], v) for n, v in ents]
                n_bad += 1
            elif cond == 'dumpvalperm':
                inv = {v: k for k, v in d.items()}
                byname = dict(ents)
                new = [(n, byname[inv[n]]) for n, _ in ents]
                n_bad += 1
            elif cond == 'dumpshuf':
                # a DERANGEMENT of positions, so this is matched to `dumpperm` (which moves
                # every name) rather than leaving half the 2-entry dumps untouched
                order = list(range(len(ents)))
                rg = random.Random(_stable_seed(tuple(names), 'shuf', draw))
                for _ in range(64):
                    rg.shuffle(order)
                    if all(a != b for a, b in enumerate(order)):
                        break
                if any(a == b for a, b in enumerate(order)):
                    pass
                else:
                    n_bad += 1
                    new = [ents[k] for k in order]
        if i == prev_i:
            prev_map = dict(new)
        out.append(text[pos:a] + _dump_render(new))
        pos = b
    out.append(text[pos:])
    return ''.join(out), n_bad, prev_map


def _name_runs(ids, dec):
    """Maximal runs of letter tokens -> [(name, start, end, in_dump)].

    A run is an identifier occurrence (or the `if` keyword, which the character tokenizer emits
    as ONE token -- the reason `_binding_scan` exists at all). `in_dump` is True when the run sits
    between a `$` and the newline that closes that line.
    """
    runs, i, N, in_dump = [], 0, len(ids), False
    while i < N:
        s = dec[ids[i]]
        if s == '$':
            in_dump = True
            i += 1
            continue
        if s in ('\n', '\n#STEP\n', '\n\n'):
            in_dump = False
            i += 1
            continue
        if s.isalpha():
            j = i
            buf = []
            while j < N and dec[ids[j]].isalpha():
                buf.append(dec[ids[j]])
                j += 1
            runs.append((''.join(buf), i, j - 1, in_dump))
            i = j
            continue
        i += 1
    return runs


def _assign_kind(code):
    """name -> 'computed' if the RHS of its (first) assignment mentions an identifier, else
    'literal'. Used to isolate the slots where a first appearance really needs a lookup."""
    import ast
    out = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Assign) and len(nd.targets) == 1 and isinstance(nd.targets[0], ast.Name):
            nm = nd.targets[0].id
            if nm in out:
                continue
            has = any(isinstance(x, ast.Name) for x in ast.walk(nd.value))
            out[nm] = 'computed' if has else 'literal'
    return out


def _shortcut_slots(ids, n_prompt, dec, kinds):
    """`_binding_scan` slots, annotated with the distance and first-appearance fields P1/P2 need.

    Adds per slot: `cls3` in {repeat, changed, first}; `kind` in {computed, literal, ?};
    `d_dump` (tokens back to the most recent occurrence of this name INSIDE a state dump, -1 if
    none) and `d_any` (the same over occurrences anywhere, -1 if none). The slot's own name run,
    which ends two tokens before the value, is excluded from both.
    """
    slots = _binding_scan(ids, n_prompt, dec)
    if not slots:
        return slots
    prev, cur, cstep, seen = {}, {}, slots[0]['step'], set()
    for sl in slots:
        if sl['step'] != cstep:
            seen |= set(prev)
            prev, cur, cstep = cur, {}, sl['step']
        if sl['name'] not in seen and sl['name'] not in prev:
            sl['cls3'] = 'first'
        elif prev.get(sl['name']) == sl['val']:
            sl['cls3'] = 'repeat'
        else:
            sl['cls3'] = 'changed'
        sl['new'] = prev.get(sl['name']) != sl['val']     # the previous agent's binary split
        sl['kind'] = kinds.get(sl['name'], '?')
        cur[sl['name']] = sl['val']
    runs = _name_runs(ids, dec)
    last_dump, last_any = {}, {}
    ri = 0
    for sl in slots:
        p = sl['vpos'][0]
        while ri < len(runs) and runs[ri][2] < p - 2:     # own name run ends at p-2
            nm, _, e, ind = runs[ri]
            last_any[nm] = e
            if ind:
                last_dump[nm] = e
            ri += 1
        sl['d_dump'] = p - last_dump[sl['name']] if sl['name'] in last_dump else -1
        sl['d_any'] = p - last_any[sl['name']] if sl['name'] in last_any else -1
    return slots


def _bin_of(d):
    if d < 0:
        return 'absent'
    for k in range(len(_DIST_BINS) - 1):
        if _DIST_BINS[k] <= d < _DIST_BINS[k + 1]:
            return '%d-%d' % (_DIST_BINS[k], _DIST_BINS[k + 1] - 1)
    return 'absent'


def _shortcut_p12(model, examples, codes, device, batch=32):
    """P1 (distance) and P2 (first appearance) for one view: per-bucket binding_acc_first."""
    import interp
    dec = interp.TOK.decod_map
    enc = [interp.encode(e) for e in examples]
    kinds = [_assign_kind(c) for c in codes]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]), reverse=True)
    dist, dany, cls3, kind2, tot = {}, {}, {}, {}, [0, 0]
    import torch as T
    slotlists = [None] * len(enc)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        logits, _ = interp.tf_logits(model, [enc[i][0] for i in idx], device)
        pred = logits.argmax(-1).cpu().numpy()
        lsm = T.log_softmax(logits, -1).cpu().numpy()
        for j, i in enumerate(idx):
            ids, npmt = enc[i]
            a = np.asarray(ids)
            sl = _shortcut_slots(ids, npmt, dec, kinds[i])
            slotlists[i] = sl
            for k, x in enumerate(sl):
                p = x['vpos'][0]
                ok = int(pred[j, p - 1] == a[p])
                nll = -float(lsm[j, p - 1, a[p]])
                x['ok'], x['nll'] = ok, nll
                pp = np.array(x['vpos'], np.int64)
                ex = int((pred[j, pp - 1] == a[pp]).all() and
                         (x['end'] < 0 or pred[j, x['end'] - 1] == a[x['end']]))
                tot[0] += 1
                tot[1] += ok
                for tab, key in ((dist, _bin_of(x['d_dump'])), (dany, _bin_of(x['d_any'])),
                                 (cls3, x['cls3']),
                                 (kind2, x['cls3'] + '/' + x['kind'])):
                    r = tab.setdefault(key, [0, 0, 0, 0.0])
                    r[0] += 1
                    r[1] += ok
                    r[2] += ex
                    r[3] += nll
    fmt = lambda t: {k: {'n': v[0], 'acc_first': v[1] / max(1, v[0]),
                         'acc_exact': v[2] / max(1, v[0]), 'n_err_first': v[0] - v[1],
                         'nll_first': v[3] / max(1, v[0])}
                     for k, v in sorted(t.items())}
    return {'n_slots': tot[0], 'acc_first': tot[1] / max(1, tot[0]),
            'dist_bins': list(_DIST_BINS),
            'by_d_dump': fmt(dist), 'by_d_any': fmt(dany),
            'by_cls3': fmt(cls3), 'by_cls3_kind': fmt(kind2)}, slotlists


def _shortcut_p3(model, texts, codes, device, batch=32, draw=0):
    """P3: corrupt every dump but the last, score the last dump's slots only.

    For each condition, returns accuracy restricted to (a) all last-dump slots, and (b) the
    CONTESTED subset -- slots whose name appears in the corrupted preceding dump with a DIFFERENT
    value, where the shortcut and the truth disagree and a forced choice is possible.
    """
    import torch as T
    import interp
    dec = interp.TOK.decod_map
    built = {}
    for cond in SHORTCUT_CONDS:
        rows = []
        for t in texts:
            r = _corrupt_dumps(t, cond, draw)
            rows.append(r)
        built[cond] = rows
    keep = [i for i in range(len(texts))
            if all(built[c][i] is not None for c in SHORTCUT_CONDS)
            and built['dumpperm'][i][1] > 0]
    kinds = [_assign_kind(codes[i]) for i in keep]
    out = {'n_examples': len(keep), 'n_dropped': len(texts) - len(keep), 'conds': {}}
    # target strings come from the INTACT text and are identical in every condition
    for cond in SHORTCUT_CONDS:
        ex = [built[cond][i][0] for i in keep]
        pmap = [built[cond][i][2] for i in keep]
        enc = [interp.encode(e) for e in ex]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]), reverse=True)
        n = nf = nx = 0
        cn = cf = cx = 0
        an = af = 0                      # name ABSENT from the corrupted predecessor
        kk = {}                          # (present?, literal|computed) -> [n, ok]
        mg, win = [], 0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            logits, _ = interp.tf_logits(model, [enc[i][0] for i in idx], device)
            pred = logits.argmax(-1).cpu().numpy()
            prob = T.softmax(logits, -1)
            for j, i in enumerate(idx):
                ids, npmt = enc[i]
                a = np.asarray(ids)
                sl = _binding_scan(ids, npmt, dec)
                if not sl:
                    continue
                last = max(x['step'] for x in sl)
                for x in sl:
                    if x['step'] != last:
                        continue
                    p = x['vpos'][0]
                    ok = int(pred[j, p - 1] == a[p])
                    pp = np.array(x['vpos'], np.int64)
                    exact = int((pred[j, pp - 1] == a[pp]).all() and
                                (x['end'] < 0 or pred[j, x['end'] - 1] == a[x['end']]))
                    n += 1
                    nf += ok
                    nx += exact
                    dv = pmap[i].get(x['name'])
                    kr = kk.setdefault(('present' if dv is not None else 'absent') + '/'
                                       + kinds[i].get(x['name'], '?'), [0, 0])
                    kr[0] += 1
                    kr[1] += ok
                    if dv is None:
                        an += 1
                        af += ok
                        continue
                    if dv == x['val']:
                        continue          # predecessor agrees with the truth: not contested
                    cn += 1
                    cf += ok
                    cx += exact
                    # forced choice at the first character where truth and the dump's claim differ
                    st = x['val'] + (dec[a[x['end']]] if x['end'] >= 0 else '')
                    sd = dv + (dec[a[x['end']]] if x['end'] >= 0 else '')
                    L = next((z for z in range(min(len(st), len(sd))) if st[z] != sd[z]), -1)
                    if L < 0 or L >= len(x['vpos']) + (1 if x['end'] >= 0 else 0):
                        continue
                    tpos = (x['vpos'] + ([x['end']] if x['end'] >= 0 else []))[L]
                    t_tok, d_tok = int(a[tpos]), interp.TOK.encod_map[sd[L]]
                    if t_tok == d_tok:
                        continue
                    row = prob[j, tpos - 1]
                    pt = float(row[t_tok])
                    pd = float(row[d_tok])
                    mg.append(float(np.log(max(pt, 1e-30)) - np.log(max(pd, 1e-30))))
                    win += int(pt > pd)
        mg = np.array(mg, np.float64)
        out['conds'][cond] = {
            'n_slots': n, 'acc_first': nf / max(1, n), 'acc_exact': nx / max(1, n),
            'n_err_first': n - nf,
            'n_contested': cn, 'acc_first_contested': cf / max(1, cn),
            'acc_exact_contested': cx / max(1, cn),
            'n_absent': an, 'acc_first_absent': af / max(1, an),
            'by_present_kind': {k: {'n': v[0], 'acc_first': v[1] / max(1, v[0])}
                                for k, v in sorted(kk.items())},
            'margin_n': int(len(mg)),
            'logmargin_mean': float(mg.mean()) if len(mg) else float('nan'),
            'logmargin_median': float(np.median(mg)) if len(mg) else float('nan'),
            'frac_prefers_truth': float(win / max(1, len(mg))),
            'frac_prefers_dump': float((len(mg) - win) / max(1, len(mg))),
        }
    return out


def _shortcut_paired(sa, sb):
    """P1 done PAIRED: the permute penalty at the SAME slot, bucketed by the IDENTITY view's
    distance. Slot lists are 1:1 across the two views (a derangement preserves the program's
    structure and the dump's insertion order), which `n_mismatch` checks rather than assumes."""
    tabs = {'by_d_dump': {}, 'by_d_any': {}, 'by_cls3': {}}
    mism = paired = 0
    for la, lb in zip(sa, sb):
        if la is None or lb is None or len(la) != len(lb):
            mism += 1
            continue
        for xa, xb in zip(la, lb):
            paired += 1
            for key, val in (('by_d_dump', _bin_of(xa['d_dump'])),
                             ('by_d_any', _bin_of(xa['d_any'])),
                             ('by_cls3', xa['cls3'])):
                r = tabs[key].setdefault(val, [0, 0, 0, 0.0, 0.0])
                r[0] += 1
                r[1] += xa['ok']
                r[2] += xb['ok']
                r[3] += xa['nll']
                r[4] += xb['nll']
    out = {'n_paired': paired, 'n_mismatch': mism}
    for key, t in tabs.items():
        out[key] = {k: {'n': v[0],
                        'acc_identity': v[1] / max(1, v[0]),
                        'acc_permute': v[2] / max(1, v[0]),
                        'penalty_acc_pp': 100.0 * (v[1] - v[2]) / max(1, v[0]),
                        'nll_identity': v[3] / max(1, v[0]),
                        'nll_permute': v[4] / max(1, v[0]),
                        'penalty_nll': (v[4] - v[3]) / max(1, v[0])}
                    for k, v in sorted(t.items())}
    return out


def shortcut(ckpt, device='cuda', batch_size=32, n=TINY_N, seed=564, dump_examples=0):
    """P1/P2/P3 for one checkpoint, on `binding`'s exact snippet set."""
    import torch as T
    import interp
    import tinypy_gen as G
    codes, corpus_path = _tiny_examples(ckpt, n)
    data_dir = os.path.dirname(corpus_path)
    run = os.path.dirname(ckpt) if ckpt.endswith('.pth') else ckpt
    pth = ckpt if ckpt.endswith('.pth') else glob_pth(ckpt)[-1]
    tag = os.path.basename(os.path.normpath(run))
    raw = open(corpus_path).read().split('\n\n')[:-1][:n]

    # verbatim from `binding`: same build, same tracer, same four-view `ok` filter
    views, kept_codes, keep_idx = _tiny_build(codes, data_dir, seed, return_keep=True)
    raw = [raw[i] for i in keep_idx]
    tracer = G.TinyPyCodeTracer()
    traced, err_counts = {}, {}
    for m in BINDING_MODES:
        outs, bad = [], 0
        for c in views[(m, 'a')]:
            t, e, _ = tracer.trace_snippet(c)
            if e is not None:
                outs.append(None); bad += 1; continue
            try:
                interp.TOK.encode(t + '\n\n')
            except KeyError:
                outs.append(None); bad += 1; continue
            outs.append(t)
        traced[m] = outs
        err_counts[m] = bad
    ok = [i for i in range(len(kept_codes))
          if all(traced[m][i] is not None for m in BINDING_MODES)]
    traced = {m: [traced[m][i] for i in ok] for m in BINDING_MODES}
    raw = [raw[i] for i in ok]
    vcode = {m: [views[(m, 'a')][i] for i in ok] for m in BINDING_MODES}

    id_match = float(np.mean([a == b for a, b in zip(traced['identity'], raw)]))
    mps = [_binding_map(a, b) for a, b in zip(vcode['identity'], vcode['permute'])]
    sub_hits = sub_n = 0
    for orig, perm, mp in zip(traced['identity'], traced['permute'], mps):
        if mp is None or any(k == v for k, v in mp.items()):
            continue
        sub_n += 1
        sub_hits += int(_permtrace_subst(orig, mp) == perm)

    # P3's corruption is a pure permutation WITHIN each dump: the multiset of names, the multiset
    # of values, the number of entries and the total length must all be preserved, and the last
    # dump must be byte-identical to `intact`. Checked on every snippet, not asserted.
    c_ok = {c: [0, 0, 0] for c in SHORTCUT_CONDS}
    for c in SHORTCUT_CONDS:
        for t in traced['identity']:
            r = _corrupt_dumps(t, c, 0)
            if r is None:
                continue
            nt = r[0]
            sa, sb = _dump_spans(t), _dump_spans(nt)
            ea = [_dump_entries(t[x:y]) for x, y in sa]
            eb = [_dump_entries(nt[x:y]) for x, y in sb]
            good = (len(sa) == len(sb)
                    and t[:sa[0][0]] == nt[:sb[0][0]]
                    and t[sa[-1][0]:] == nt[sb[-1][0]:]          # last dump + tail untouched
                    and all(sorted(p) == sorted(q) for p, q in zip(ea, eb))
                    if c == 'dumpshuf' else
                    len(sa) == len(sb)
                    and t[:sa[0][0]] == nt[:sb[0][0]]
                    and t[sa[-1][0]:] == nt[sb[-1][0]:]
                    and all(sorted(x[0] for x in p) == sorted(x[0] for x in q)
                            and sorted(x[1] for x in p) == sorted(x[1] for x in q)
                            for p, q in zip(ea, eb)))
            c_ok[c][0] += 1
            c_ok[c][1] += int(good)
            c_ok[c][2] += r[1]

    device = T.device(device) if isinstance(device, str) else device
    model = interp.load_model(pth, device)
    out = {'ckpt': pth, 'run': run, 'corpus': corpus_path, 'n': len(traced['identity']),
           'seed': seed, 'trace_errors': err_counts,
           'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
           'checks': {'identity_trace_matches_corpus': id_match,
                      'permute_trace_matches_subst': float(sub_hits / max(1, sub_n)),
                      'n_permute_checked': sub_n,
                      'corrupt_structure_ok': {c: c_ok[c][1] / max(1, c_ok[c][0])
                                               for c in SHORTCUT_CONDS},
                      'corrupt_dumps_per_snippet': {c: c_ok[c][2] / max(1, c_ok[c][0])
                                                    for c in SHORTCUT_CONDS}},
           'p12': {}, 'p3': {}}
    print('%s: n=%d identity==corpus %.4f permute==subst %.4f structure_ok %s'
          % (tag, out['n'], id_match, out['checks']['permute_trace_matches_subst'],
             out['checks']['corrupt_structure_ok']), flush=True)
    if dump_examples:
        out['examples'] = [{'cond': c, 'text': _corrupt_dumps(traced['identity'][k], c, 0)[0]}
                           for k in range(dump_examples) for c in SHORTCUT_CONDS]
        out['examples_intact_code'] = vcode['identity'][:dump_examples]
    sl_keep = {}
    for m in ('identity', 'permute'):
        r, sls = _shortcut_p12(model, traced[m], vcode[m], device, batch_size)
        sl_keep[m] = sls
        out['p12'][m] = r
        print('  p12 %-9s slots %6d acc_first %.5f | cls3 %s'
              % (m, r['n_slots'], r['acc_first'],
                 {k: round(v['acc_first'], 5) for k, v in r['by_cls3'].items()}), flush=True)
    out['p12']['paired'] = _shortcut_paired(sl_keep['identity'], sl_keep['permute'])
    print('  p12 paired    n %6d (mismatch %d) | penalty_acc_pp by d_dump %s'
          % (out['p12']['paired']['n_paired'], out['p12']['paired']['n_mismatch'],
             {k: round(v['penalty_acc_pp'], 4)
              for k, v in out['p12']['paired']['by_d_dump'].items()}), flush=True)
    out['p3'] = _shortcut_p3(model, traced['identity'], vcode['identity'], device, batch_size)
    for c in SHORTCUT_CONDS:
        v = out['p3']['conds'][c]
        print('  p3  %-12s n %5d acc_first %.5f exact %.5f | contested n %5d acc %.5f '
              'logmargin %+.3f prefers_dump %.4f'
              % (c, v['n_slots'], v['acc_first'], v['acc_exact'], v['n_contested'],
                 v['acc_first_contested'], v['logmargin_mean'], v['frac_prefers_dump']),
              flush=True)
    return out


def shortcut_all(runs=None, device='cuda', batch_size=32, n=TINY_N, seed=564):
    if os.environ.get('PYTHONHASHSEED') is None:
        raise SystemExit('set PYTHONHASHSEED (e.g. 0): permute/constants are hash-seeded')
    runs = runs or PERMTRACE_RUNS
    path = os.environ.get('SHORTCUT_JSON') or os.path.join(WORK, 'shortcut.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    for i, r in enumerate(runs):
        d = r if os.path.isabs(r) else os.path.join(WORK, r)
        key = os.path.basename(os.path.normpath(d))
        out[key] = shortcut(os.path.join(d, 'checkpoint_1.00.pth'), device, batch_size, n, seed,
                            dump_examples=3 if i == 0 else 0)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('wrote', path, flush=True)
    return out


# ---------------------------------------------------------------------------
# subspace: is the renaming displacement removable at realistic scale WITHOUT
#           retraining, and does a basis fitted on one view repair another?
# ---------------------------------------------------------------------------
#
# The tiny-scale result (labnotes 2026-09-16 SUBSPACE OVERLAP + e3) is that the
# renaming displacement is LOW-RANK and NAME-SPECIFIC: eight directions fitted on
# half the programs and removed from the disjoint other half lift renamed
# retrieval from 0.2569 to 0.6831, and the rank-8 bases of `samelen` and
# `permute` overlap at 0.52-0.60 mean cos^2 (chance 8/384 = 0.0208) while
# `constants` is at or below chance against both.
#
# If that structure exists in ModernBERT-base on POJ-104, projecting it out is a
# train-free defence that needs none of the missing mbert_cb_* checkpoints.
#
# Protocol, fixed before any number was looked at:
#   * SPLIT BY CLASS, not by program. POJ-104 test is 24 classes x 500. The 12
#     even-indexed classes are the FIT half, the 12 odd-indexed the EVAL half.
#     No program and no class contributes to both, so the basis cannot have seen
#     the eval half's programs OR its problem.
#   * d_i = unit(emb(perturbed_i)) - unit(emb(original_i)) over the FIT half.
#     Top-k right singular vectors of d (uncentered -- the mean displacement is
#     then the k=1 direction, which is the deployable recipe: "remove k
#     directions"). The centered basis is also fitted, because that is what the
#     tiny-scale principal angles used and the comparison has to be like for like.
#   * Project out of BOTH query and corpus on the EVAL half, renormalise, and
#     recompute top-1 retention and MAP@R there.
#   * Perturbed views are REUSED from modernbert_scale/data/datasets/poj104/
#     (built by perturb/generate.py at PERTURB_SEED, AST-offset substitution).
#     `samelen` does not ship, so it is built here with the SAME variable
#     selection draw and the SAME substitute_identifiers, so it differs from the
#     shipped `permute`/`var_k` files only in the target strings.

SUBSPACE_KS = (0, 1, 2, 4, 8, 16, 32, 64)
SUBSPACE_VIEWS = ('samelen', 'permute', 'constants', 'var_k')
SUBSPACE_FILES = {'permute': 'test_permute{n}.jsonl',
                  'constants': 'test_constants{n}.jsonl',
                  'var_k': 'test_edits{n}.jsonl'}


def _poj_dir():
    return os.path.join(ROOT, 'modernbert_scale', 'data', 'datasets', 'poj104')


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def _samelen_texts(records, n, seed_base=PERTURB_SEED):
    """Length-matched rename on the same variables the shipped files renamed."""
    from perturb.rename import substitute_identifiers
    out, tot = [], 0
    for i, r in enumerate(records):
        seed = seed_base + n * 1_000_000 + i
        sel = select_vars(r['code'], n, seed)
        rng = random.Random(seed ^ 0x5EED)
        taken = set(IDENT_RE.findall(r['code']))
        mp = {}
        for old in sel:
            new = _rand_name(rng, len(old))
            guard = 0
            while new in taken and guard < 64:
                new = _rand_name(rng, len(old))
                guard += 1
            taken.add(new)
            mp[old] = new
        out.append(substitute_identifiers(r['code'], mp, LANG))
        tot += len(mp)
    return out, tot / max(len(records), 1)


def _subspace_views(ns=(4, 8), limit=None):
    """{view_name: [text]} for identity and every perturbed view, position-aligned."""
    recs = _read_jsonl(os.path.join(_poj_dir(), 'test.jsonl'))
    keep = None
    if limit and limit < len(recs):
        stride = len(recs) // limit
        keep = list(range(0, len(recs), stride))[:limit]
        recs = [recs[i] for i in keep]
    texts = {'identity': [r['code'] for r in recs]}
    meta = {}
    for n in ns:
        for v, tmpl in SUBSPACE_FILES.items():
            rows = _read_jsonl(os.path.join(_poj_dir(), tmpl.format(n=n)))
            if keep is not None:
                rows = [rows[i] for i in keep]
            if len(rows) != len(recs):
                raise SystemExit('%s misaligned: %d vs %d' % (tmpl, len(rows), len(recs)))
            texts['%s_N%d' % (v, n)] = [r['code'] for r in rows]
            meta['%s_N%d' % (v, n)] = {
                'source': tmpl.format(n=n),
                'mean_edits': float(np.mean([r.get('num_renamed', 0) for r in rows])),
                'frac_text_changed': float(np.mean(
                    [a['code'] != b for a, b in zip(recs, texts['%s_N%d' % (v, n)])])),
            }
        t, m = _samelen_texts(recs, n)
        texts['samelen_N%d' % n] = t
        meta['samelen_N%d' % n] = {
            'source': 'generated (select_vars + substitute_identifiers)',
            'mean_edits': m,
            'frac_text_changed': float(np.mean([a['code'] != b for a, b in zip(recs, t)])),
        }
    labels = np.array([int(r['label']) for r in recs])
    return recs, texts, labels, meta


def _map_at_r_cross(q, c, labels, sizes):
    """MAP@R with perturbed queries against an unperturbed corpus, self masked.

    Row i of `q` and row i of `c` are the same program, so column i is masked for
    query i -- a perturbed program is never allowed to retrieve its own original.
    R is (class size - 1), which makes the identity view's value the ordinary
    CodeXGLUE MAP@R.
    """
    aps = []
    idx = np.arange(len(q))
    for s in range(0, len(q), 512):
        sl = slice(s, min(s + 512, len(q)))
        sim = q[sl] @ c.T
        sim[np.arange(sl.stop - sl.start), idx[sl]] = -1e9
        # one argpartition width for the whole slice: a 512-row slice can span
        # two classes, so take the widest R any row in it needs.
        R = max(1, max(int(sizes[int(l)]) for l in np.unique(labels[sl])) - 1)
        part = np.argpartition(-sim, R, axis=1)[:, :R]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1, kind='stable')
        rank = np.take_along_axis(part, order, axis=1)
        for j, i in enumerate(range(sl.start, sl.stop)):
            r = int(sizes[int(labels[i])]) - 1
            if r <= 0:
                continue
            hit = labels[rank[j, :r]] == labels[i]
            aps.append(float((np.cumsum(hit) / np.arange(1, r + 1))[hit].sum() / r))
    return float(np.mean(aps)) if aps else 0.0


def _proj(x, U):
    x = unit(x)
    if U is None or len(U) == 0:
        return x
    return unit(x - (x @ U.T) @ U)


def _subspace_score(P, E0, labels, idx, U, sizes, ok0=None, want_map=True):
    q = _proj(P[idx], U)
    c = _proj(E0[idx], U)
    lab = labels[idx]
    hit = np.empty(len(q), dtype=bool)
    ar = np.arange(len(q))
    for s in range(0, len(q), 512):
        sl = slice(s, min(s + 512, len(q)))
        sim = q[sl] @ c.T
        sim[np.arange(sl.stop - sl.start), ar[sl]] = -1e9
        hit[sl] = lab[sim.argmax(1)] == lab[sl]
    row = {'top1': float(hit.mean())}
    if ok0 is not None:
        sup = int(ok0.sum())
        row['retention'] = float((ok0 & hit).sum() / max(sup, 1))
        row['support'] = sup
    if want_map:
        row['map_at_r'] = _map_at_r_cross(q, c, lab, sizes)
    return row, hit


def subspace(model_id='answerdotai/ModernBERT-base', device='cuda', batch_size=64,
             limit=None, ns=(4, 8), out_path=None, cross_ks=(4, 8, 16, 32)):
    tag = model_id.rstrip('/').split('/')[-1]
    os.makedirs(CACHE, exist_ok=True)
    recs, texts, labels, meta = _subspace_views(ns, limit)
    npz = os.path.join(CACHE, 'subspace_emb__%s_%d.npz' % (tag, len(recs)))
    if os.path.exists(npz):
        E = dict(np.load(npz))
        print('loaded cached embeddings', npz, flush=True)
    else:
        blank = [set() for _ in recs]
        tok, enc = load_encoder(model_id, device)
        E = {}
        for name, txt in texts.items():
            V, _ = embed(tok, enc, txt, blank, device, batch_size)
            E[name] = V.astype(np.float32)
            print('embedded %-14s %s' % (name, V.shape), flush=True)
        np.savez_compressed(npz, **E)
        print('wrote', npz, flush=True)

    D = E['identity'].shape[1]
    classes = np.unique(labels)
    fit_classes = classes[::2]
    eval_classes = classes[1::2]
    FIT = np.where(np.isin(labels, fit_classes))[0]
    EV = np.where(np.isin(labels, eval_classes))[0]
    sizes = {int(l): int((labels[EV] == l).sum()) for l in eval_classes}
    E0 = E['identity']

    out = {
        'model': model_id, 'n_records': len(recs), 'hidden': int(D),
        'block_size': BLOCK_SIZE, 'pooling': 'meanpool', 'perturb_seed': PERTURB_SEED,
        'split': {'by': 'class', 'fit_classes': [int(x) for x in fit_classes],
                  'eval_classes': [int(x) for x in eval_classes],
                  'n_fit': int(len(FIT)), 'n_eval': int(len(EV))},
        'ks': list(SUBSPACE_KS), 'views': meta, 'chance_cos2_k8': 8.0 / D,
    }

    # identity retrieval on the eval half -- the denominator for every retention
    base, ok0 = _subspace_score(E0, E0, labels, EV, None, sizes, None, want_map=True)
    out['identity_eval_half'] = base
    print('identity eval-half: top1 %.4f MAP@R %.4f  (n=%d)'
          % (base['top1'], base['map_at_r'], len(EV)), flush=True)

    # ---- fit the bases on the FIT half -------------------------------------
    bases_raw, bases_cen, fitinfo = {}, {}, {}
    vnames = [v for v in E if v != 'identity']
    for v in vnames:
        d = unit(E[v][FIT]) - unit(E0[FIT])
        mu = d.mean(0)
        _, sv_r, Vt_r = np.linalg.svd(d, full_matrices=False)
        _, sv_c, Vt_c = np.linalg.svd(d - mu, full_matrices=False)
        bases_raw[v] = Vt_r
        bases_cen[v] = Vt_c
        e_r = (sv_r ** 2) / max((sv_r ** 2).sum(), 1e-12)
        fitinfo[v] = {
            'disp_norm': float(np.linalg.norm(d, axis=1).mean()),
            'shared_mean_frac': float((mu ** 2).sum() / max((d ** 2).sum(1).mean(), 1e-12)),
            'energy_raw': {str(k): float(e_r[:k].sum()) for k in SUBSPACE_KS if k},
            'sv_top8_raw': [float(x) for x in sv_r[:8]],
        }
    out['fit'] = fitinfo

    rng = np.random.default_rng(20260916)
    Q, _ = np.linalg.qr(rng.standard_normal((D, max(SUBSPACE_KS))))
    bases_raw['random'] = Q.T
    bases_cen['random'] = Q.T

    # ---- k-sweep: own-view basis, plus identity under that basis ------------
    sweep = {}
    for v in vnames:
        sweep[v] = {}
        for k in SUBSPACE_KS:
            U = bases_raw[v][:k] if k else None
            r, _ = _subspace_score(E[v], E0, labels, EV, U, sizes, ok0)
            ri, _ = _subspace_score(E0, E0, labels, EV, U, sizes, ok0)
            r['identity_top1'] = ri['top1']
            r['identity_map_at_r'] = ri['map_at_r']
            r['identity_retention'] = ri['retention']
            Uc = bases_cen[v][:k] if k else None
            rc, _ = _subspace_score(E[v], E0, labels, EV, Uc, sizes, ok0, want_map=False)
            r['retention_centered_basis'] = rc['retention']
            sweep[v][str(k)] = r
            print('  %-14s k=%-3d ret %.4f  top1 %.4f  MAP@R %.4f | id top1 %.4f MAP@R %.4f'
                  % (v, k, r['retention'], r['top1'], r['map_at_r'],
                     ri['top1'], ri['map_at_r']), flush=True)
    out['ksweep'] = sweep

    # ---- random-basis control ----------------------------------------------
    ctl = {}
    for v in vnames:
        ctl[v] = {}
        for k in (8, 32):
            r, _ = _subspace_score(E[v], E0, labels, EV, Q.T[:k], sizes, ok0, want_map=False)
            ctl[v][str(k)] = r
    out['random_basis_control'] = ctl

    # ---- cross-view matrix --------------------------------------------------
    cross = {}
    for k in cross_ks:
        cross[str(k)] = {}
        for fv in list(vnames) + ['random']:
            cross[str(k)][fv] = {}
            for ev in ['identity'] + list(vnames):
                U = bases_raw[fv][:k]
                src = E0 if ev == 'identity' else E[ev]
                r, _ = _subspace_score(src, E0, labels, EV, U, sizes, ok0, want_map=False)
                cross[str(k)][fv][ev] = r
            print('  cross k=%d fit=%-14s %s' % (
                k, fv, ' '.join('%s %.4f' % (e, cross[str(k)][fv][e]['retention'])
                                for e in cross[str(k)][fv])), flush=True)
    out['cross'] = cross

    # ---- principal angles ---------------------------------------------------
    # Two references are needed to read a cross-view overlap. The FLOOR is a
    # random orthonormal 8-plane (k/D). The CEILING is the same view fitted on
    # two disjoint halves of the FIT half: that is how much of the basis is
    # reproducible at this sample size at all, so a cross-view overlap can only
    # be read as a fraction of it.
    halfA, halfB = FIT[0::2], FIT[1::2]
    out['angles_splithalf'] = {}
    for v in vnames:
        U = []
        for H in (halfA, halfB):
            d = unit(E[v][H]) - unit(E0[H])
            _, _, Vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
            U.append(Vt[:8])
        out['angles_splithalf'][v] = float(((U[0] @ U[1].T) ** 2).sum() / 8.0)
    print('  split-half ceiling:', json.dumps(out['angles_splithalf']), flush=True)
    for label, B in (('centered', bases_cen), ('raw', bases_raw)):
        ang = {'_chance': 8.0 / D}
        names = [v for v in vnames]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                U1, U2 = B[a][:8], B[b][:8]
                ang['%s|%s' % (a, b)] = float(((U1 @ U2.T) ** 2).sum() / U1.shape[0])
            ang['%s|random' % a] = float(((B[a][:8] @ Q.T[:8].T) ** 2).sum() / 8.0)
        out['angles_' + label] = ang
    print('  angles (centered, chance %.4f):' % (8.0 / D), flush=True)
    for k, v in sorted(out['angles_centered'].items()):
        if not k.startswith('_'):
            print('    %-34s %.4f' % (k, v), flush=True)

    out_path = out_path or os.path.join(WORK, 'subspace.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', out_path, flush=True)
    return out


def _shortcut_p3_scope(model, texts, codes, device, batch, conds, scope):
    """`_shortcut_p3` with an explicit condition list and corruption scope. Kept separate from
    `_shortcut_p3` so the already-computed `p3` block in shortcut.json is not disturbed."""
    import torch as T
    import interp
    dec = interp.TOK.decod_map
    built = {c: [_corrupt_dumps(t, c, 0, scope) for t in texts] for c in conds}
    keep = [i for i in range(len(texts))
            if all(built[c][i] is not None for c in conds)
            and built[conds[-1]][i][1] > 0]
    kinds = [_assign_kind(codes[i]) for i in keep]
    out = {'scope': scope, 'n_examples': len(keep), 'conds': {}}
    for cond in conds:
        ex = [built[cond][i][0] for i in keep]
        pmap = [built[cond][i][2] for i in keep]
        enc = [interp.encode(e) for e in ex]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]), reverse=True)
        n = nf = nx = cn = cf = win = 0
        mg = []
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            logits, _ = interp.tf_logits(model, [enc[i][0] for i in idx], device)
            pred = logits.argmax(-1).cpu().numpy()
            prob = T.softmax(logits, -1)
            for j, i in enumerate(idx):
                ids, npmt = enc[i]
                a = np.asarray(ids)
                sl = _binding_scan(ids, npmt, dec)
                if not sl:
                    continue
                last = max(x['step'] for x in sl)
                for x in sl:
                    if x['step'] != last:
                        continue
                    p = x['vpos'][0]
                    ok = int(pred[j, p - 1] == a[p])
                    pp = np.array(x['vpos'], np.int64)
                    exact = int((pred[j, pp - 1] == a[pp]).all() and
                                (x['end'] < 0 or pred[j, x['end'] - 1] == a[x['end']]))
                    n += 1; nf += ok; nx += exact
                    dv = pmap[i].get(x['name'])
                    if dv is None or dv == x['val']:
                        continue
                    cn += 1; cf += ok
                    st = x['val'] + (dec[a[x['end']]] if x['end'] >= 0 else '')
                    sd = dv + (dec[a[x['end']]] if x['end'] >= 0 else '')
                    L = next((z for z in range(min(len(st), len(sd))) if st[z] != sd[z]), -1)
                    if L < 0 or L >= len(x['vpos']) + (1 if x['end'] >= 0 else 0):
                        continue
                    tpos = (x['vpos'] + ([x['end']] if x['end'] >= 0 else []))[L]
                    t_tok, d_tok = int(a[tpos]), interp.TOK.encod_map[sd[L]]
                    if t_tok == d_tok:
                        continue
                    row = prob[j, tpos - 1]
                    pt, pd = float(row[t_tok]), float(row[d_tok])
                    mg.append(float(np.log(max(pt, 1e-30)) - np.log(max(pd, 1e-30))))
                    win += int(pt > pd)
        mg = np.array(mg, np.float64)
        out['conds'][cond] = {
            'n_slots': n, 'acc_first': nf / max(1, n), 'acc_exact': nx / max(1, n),
            'n_contested': cn, 'acc_first_contested': cf / max(1, cn),
            'margin_n': int(len(mg)),
            'logmargin_mean': float(mg.mean()) if len(mg) else float('nan'),
            'frac_prefers_dump': float((len(mg) - win) / max(1, len(mg)))}
    return out


def shortcut_prev(runs=None, device='cuda', batch_size=32, n=TINY_N, seed=564):
    """Recency follow-up: corrupt ONLY the dump immediately preceding the scored one.

    Merged into the existing work/shortcut.json under `p3_prev`; the `p12` and `p3` blocks the
    main pass wrote are left exactly as they are.
    """
    import torch as T
    import interp
    import tinypy_gen as G
    if os.environ.get('PYTHONHASHSEED') is None:
        raise SystemExit('set PYTHONHASHSEED (e.g. 0)')
    runs = runs or PERMTRACE_RUNS
    path = os.environ.get('SHORTCUT_JSON') or os.path.join(WORK, 'shortcut.json')
    out = json.load(open(path))
    for r in runs:
        d = r if os.path.isabs(r) else os.path.join(WORK, r)
        key = os.path.basename(os.path.normpath(d))
        ckpt = os.path.join(d, 'checkpoint_1.00.pth')
        codes, corpus_path = _tiny_examples(ckpt, n)
        views, kept_codes, _ = _tiny_build(codes, os.path.dirname(corpus_path), seed,
                                           return_keep=True)
        tracer = G.TinyPyCodeTracer()
        traced = {}
        for m in BINDING_MODES:
            outs = []
            for c in views[(m, 'a')]:
                t, e, _ = tracer.trace_snippet(c)
                if e is not None:
                    outs.append(None); continue
                try:
                    interp.TOK.encode(t + '\n\n')
                except KeyError:
                    outs.append(None); continue
                outs.append(t)
            traced[m] = outs
        ok = [i for i in range(len(kept_codes))
              if all(traced[m][i] is not None for m in BINDING_MODES)]
        texts = [traced['identity'][i] for i in ok]
        src = [views[('identity', 'a')][i] for i in ok]
        dev = T.device(device) if isinstance(device, str) else device
        model = interp.load_model(ckpt, dev)
        res = _shortcut_p3_scope(model, texts, src, dev, batch_size,
                                 ('intact', 'dumpperm'), 'prev')
        out[key]['p3_prev'] = res
        v = res['conds']['dumpperm']
        print('%s p3_prev: n %d acc_first %.5f exact %.5f | contested %d acc %.5f '
              'logmargin %+.3f prefers_dump %.4f'
              % (key, v['n_slots'], v['acc_first'], v['acc_exact'], v['n_contested'],
                 v['acc_first_contested'], v['logmargin_mean'], v['frac_prefers_dump']),
              flush=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('wrote', path, flush=True)
    return out


SHORTCUT_Q = {'ckpt_p0.1s': '0.1%', 'ckpt_p1s': '1%', 'ckpt_p10': '10%',
              'ckpt_p20': '20%', 'ckpt_p40': '40%', 'ckpt_p70': '70%'}


def shortcut_tables(path=None):
    """The three tables, printed straight out of work/shortcut.json. No number is retyped."""
    path = path or os.path.join(WORK, 'shortcut.json')
    D = json.load(open(path))
    runs = [r for r in PERMTRACE_RUNS if r in D]
    lab = lambda r: SHORTCUT_Q.get(r, r)

    print('\n== P1 DISTANCE: permute penalty vs distance back to the matchable name ==')
    for field in ('by_d_any', 'by_d_dump'):
        keys = sorted({k for r in runs for k in D[r]['p12']['paired'][field]},
                      key=lambda k: (k == 'absent', int(k.split('-')[0]) if k != 'absent' else 0))
        print('\n-- bucketed by %s (token distance) --' % field[3:])
        print('%-6s %-9s' % ('q', 'stat') + ''.join('%12s' % k for k in keys))
        for r in runs:
            t = D[r]['p12']['paired'][field]
            g = lambda k, f: t[k][f] if k in t else float('nan')
            print('%-6s %-9s' % (lab(r), 'n') + ''.join('%12d' % (t[k]['n'] if k in t else 0)
                                                        for k in keys))
            print('%-6s %-9s' % ('', 'pen_acc_pp') + ''.join('%12.4f' % g(k, 'penalty_acc_pp')
                                                             for k in keys))
            print('%-6s %-9s' % ('', 'pen_nll') + ''.join('%12.5f' % g(k, 'penalty_nll')
                                                          for k in keys))

    print('\n== P2 FIRST APPEARANCE: repeat vs changed vs strictly-first ==')
    keys = ['repeat', 'changed', 'first']
    print('%-6s %-11s' % ('q', 'stat') + ''.join('%12s' % k for k in keys))
    for r in runs:
        t = D[r]['p12']['paired']['by_cls3']
        g = lambda k, f: t[k][f] if k in t else float('nan')
        print('%-6s %-11s' % (lab(r), 'n') + ''.join('%12d' % (t[k]['n'] if k in t else 0)
                                                     for k in keys))
        print('%-6s %-11s' % ('', 'acc_ident') + ''.join('%12.5f' % g(k, 'acc_identity')
                                                         for k in keys))
        print('%-6s %-11s' % ('', 'acc_perm') + ''.join('%12.5f' % g(k, 'acc_permute')
                                                        for k in keys))
        print('%-6s %-11s' % ('', 'pen_acc_pp') + ''.join('%12.4f' % g(k, 'penalty_acc_pp')
                                                          for k in keys))
        print('%-6s %-11s' % ('', 'pen_nll') + ''.join('%12.5f' % g(k, 'penalty_nll')
                                                       for k in keys))
    print('\n-- split by how the variable was assigned (identity / permute acc_first) --')
    kk = ['repeat/literal', 'repeat/computed', 'first/literal', 'first/computed']
    print('%-6s %-9s' % ('q', 'view') + ''.join('%17s' % k for k in kk))
    for r in runs:
        for v in ('identity', 'permute'):
            t = D[r]['p12'][v]['by_cls3_kind']
            print('%-6s %-9s' % (lab(r) if v == 'identity' else '', v)
                  + ''.join('%17s' % ('%.5f(%d)' % (t[k]['acc_first'], t[k]['n'])
                                      if k in t else '-') for k in kk))

    print('\n== P3 CORRUPT THE PRECEDING DUMPS (last dump scored, byte-identical everywhere) ==')
    print('%-6s %-13s %7s %9s %9s %8s %9s %11s %11s'
          % ('q', 'condition', 'n', 'acc_first', 'acc_exact', 'n_cont', 'acc_cont',
             'logmargin', 'pref_dump'))
    for r in runs:
        for c in SHORTCUT_CONDS:
            v = D[r]['p3']['conds'][c]
            print('%-6s %-13s %7d %9.5f %9.5f %8d %9.5f %11.3f %11.4f'
                  % (lab(r) if c == 'intact' else '', c, v['n_slots'], v['acc_first'],
                     v['acc_exact'], v['n_contested'], v['acc_first_contested'],
                     v['logmargin_mean'], v['frac_prefers_dump']))
    print('\n-- P3 by whether the name is in the corrupted predecessor, and how it was assigned --')
    kk2 = ['present/literal', 'present/computed', 'absent/literal', 'absent/computed']
    print('%-6s %-13s' % ('q', 'condition') + ''.join('%18s' % k for k in kk2))
    for r in runs:
        for c in SHORTCUT_CONDS:
            t = D[r]['p3']['conds'][c]['by_present_kind']
            print('%-6s %-13s' % (lab(r) if c == 'intact' else '', c)
                  + ''.join('%18s' % ('%.5f(%d)' % (t[k]['acc_first'], t[k]['n'])
                                      if k in t else '-') for k in kk2))

    print('\n== checks ==')
    for r in runs:
        ch = D[r]['checks']
        print('%-6s n=%4d  id==corpus %.4f  perm==subst %.4f  structure_ok %s  corrupted/snip %s'
              % (lab(r), D[r]['n'], ch['identity_trace_matches_corpus'],
                 ch['permute_trace_matches_subst'],
                 {k: round(v, 4) for k, v in ch['corrupt_structure_ok'].items()},
                 {k: round(v, 2) for k, v in ch['corrupt_dumps_per_snippet'].items()}))


# ---------------------------------------------------------------------------
# lengthrule: does the NAMES-PER-LENGTH rule survive a POOLED readout?
# ---------------------------------------------------------------------------
#
# The rule ("generalisation at identifier length L needs >=~1000 memorised names of that length,
# or q >= 0.05") was measured entirely on TRACING end-accuracy -- 52 (run, length) cells in
# figures.lengthaxis(). Since then `shortcut` showed that tracing lets the decoder copy a value
# out of the preceding state dump by matching the variable's NAME STRING, so the tracing readout
# need not have memorised anything at all. The rule may therefore describe the copying readout
# and say nothing about pooled representations.
#
# This block re-derives the SAME length axis on the pooled readout. For each checkpoint it builds
# per-length views -- rename ONLY the identifiers of length L, leave every other identifier byte
# identical -- and measures paired retrieval (does the renamed snippet retrieve its own original
# out of the full pool of ~2000 unperturbed snippets?). x-axis is unchanged: floor(q * 26^L),
# which `lexicon.exact_lexicon` confirms is exactly the number of length-L names training could
# draw.
#
# Two edit sizes, because the count of length-L variables in a snippet is itself a function of L
# and would otherwise be confounded with it:
#   'all' -- rename every length-L identifier (the tracing protocol's analogue)
#   'one' -- rename exactly ONE length-L identifier, so the edit is the same size at every L
# Only snippets that actually contain a length-L identifier are scored; the retrieval pool is
# always the full corpus, so the task difficulty does not move with L.

LENGTHRULE_RUNS = ('ckpt_p0.1s', 'ckpt_p0.1s_s2', 'ckpt_p1s', 'ckpt_p1s_s2',
                   'ckpt_p10', 'ckpt_p10s', 'ckpt_p20', 'ckpt_p40', 'ckpt_p70')
LENGTHRULE_Q = {'ckpt_p0.1s': 0.001, 'ckpt_p0.1s_s2': 0.001, 'ckpt_p1s': 0.01,
                'ckpt_p1s_s2': 0.01, 'ckpt_p10': 0.10, 'ckpt_p10s': 0.10,
                'ckpt_p20': 0.20, 'ckpt_p40': 0.40, 'ckpt_p70': 0.70}


def _tiny_rename_len(code, L, mode, ren, seen_set):
    """Rename only the identifiers of length exactly `L`. Returns (text, stats) or None.

    stats: n_renamed, n_occ (occurrences rewritten), n_unseen (introduced names verified absent
    from the training lexicon), n_cand (length-L identifiers present).
    `mode` 'all' renames every one of them, 'one' renames a single deterministically chosen one.
    """
    import ast
    tree = ast.parse(code)
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
    cand = [v for v in names if len(v) == L]
    st = {'n_cand': len(cand), 'n_renamed': 0, 'n_occ': 0, 'n_unseen': 0}
    if not cand:
        return ast.unparse(tree).replace('    ', '\t'), st
    tgt = cand if mode == 'all' else [cand[random.Random(_stable_seed(code, L)).randrange(len(cand))]]
    mp = {}
    for v in tgt:
        bad = set(ren.r.reserved_words) | set(names) | set(mp.values())
        nm = ren.name(1, bad) if L == 1 else _unseen_name(ren, L, bad, seen_set)
        mp[v] = nm
        st['n_unseen'] += int(nm not in seen_set)
    st['n_renamed'] = len(mp)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in mp:
            st['n_occ'] += 1
            n.id = mp[n.id]
    return ast.unparse(tree).replace('    ', '\t'), st


def lengthrule_tiny(ckpt, device='cpu', batch_size=64, n=TINY_N, seed=564):
    """Per-length POOLED retrieval for one TinyPy checkpoint.

    The identity row is the wiring check and must read exactly 1.0000, same convention as `tiny`:
    the diagonal is the TARGET here, not a leak, so it is not masked.
    """
    import interp
    codes, corpus_path = _tiny_examples(ckpt, n)
    data_dir = os.path.dirname(corpus_path)
    pth = ckpt if ckpt.endswith('.pth') else glob_pth(ckpt)[-1]
    model = interp.load_model(pth, device)
    enc, tokname = _tiny_encoder(data_dir)
    seen_set = _tiny_seen_set(data_dir)
    run = os.path.dirname(ckpt) if ckpt.endswith('.pth') else ckpt
    tag = os.path.basename(os.path.normpath(run))

    # Baseline: the unperturbed corpus, re-unparsed so that every view differs from it only in
    # the identifiers. ast.unparse round-tripping changes whitespace, and comparing a rewritten
    # view against the RAW text would score that formatting difference as renaming damage.
    import ast
    base = []
    for c in codes:
        try:
            base.append(ast.unparse(ast.parse(c)).replace('    ', '\t'))
        except SyntaxError:
            base.append(None)
    keep = [i for i, b in enumerate(base) if b is not None]
    codes = [codes[i] for i in keep]
    base = [base[i] for i in keep]

    views, stats = {}, {}
    for mode in ('all', 'one'):
        for L in range(1, 6):
            ren = interp.SameLengthRenamer(seed)
            txt, sts = [], []
            for c in codes:
                r = _tiny_rename_len(c, L, mode, ren, seen_set)
                txt.append(r[0]); sts.append(r[1])
            views[(mode, L)] = txt
            stats[(mode, L)] = sts

    E0 = _tiny_embed(model, base, device, batch_size, enc)
    out = {'ckpt': pth, 'run': run, 'corpus': corpus_path, 'n': len(base), 'seed': seed,
           'tokenizer': tokname, 'q': LENGTHRULE_Q.get(tag),
           'train_lexicon_by_len': seen_set.per_length(), 'cells': {}}
    # identity wiring check
    out['identity_paired_top1'] = float(_paired(E0, E0).mean())
    print('%s: n=%d  identity %.4f' % (tag, len(base), out['identity_paired_top1']), flush=True)
    for mode in ('all', 'one'):
        for L in range(1, 6):
            sts = stats[(mode, L)]
            aff = np.array([s['n_renamed'] > 0 for s in sts])
            P = _tiny_embed(model, views[(mode, L)], device, batch_size, enc)
            hit = _paired(P, E0)
            chg = np.array([a != b for a, b in zip(views[(mode, L)], base)])
            row = {
                'mode': mode, 'L': L,
                'n_affected': int(aff.sum()),
                'paired_top1': float(hit[aff].mean()) if aff.any() else None,
                'paired_top1_allrows': float(hit.mean()),
                'frac_text_changed': float(chg.mean()),
                'mean_renamed': float(np.mean([s['n_renamed'] for s in sts])),
                'mean_occ_renamed': float(np.mean([s['n_occ'] for s in sts])),
                'mean_cand': float(np.mean([s['n_cand'] for s in sts])),
                'frac_unseen': (float(sum(s['n_unseen'] for s in sts) /
                                      max(sum(s['n_renamed'] for s in sts), 1))),
                'self_cos': float((unit(P)[aff] * unit(E0)[aff]).sum(1).mean()) if aff.any() else None,
                'names_of_this_length': max(1, int((LENGTHRULE_Q.get(tag) or 0) * 26 ** L)),
            }
            out['cells']['%s_L%d' % (mode, L)] = row
            print('  %-4s L=%d  n_aff %4d  paired %.4f  vars %.2f  occ %.2f  unseen %.3f'
                  % (mode, L, row['n_affected'], row['paired_top1'] or float('nan'),
                     row['mean_renamed'], row['mean_occ_renamed'], row['frac_unseen']), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'lengthrule_tiny__%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path, flush=True)
    return out


def lengthrule_tiny_all(runs=None, device='cpu', batch_size=64, n=TINY_N):
    for r in (runs or LENGTHRULE_RUNS):
        d = os.path.join(WORK, r)
        p = os.path.join(CACHE, 'lengthrule_tiny__%s.json' % r)
        if os.path.exists(p):
            print('skip (exists):', p, flush=True)
            continue
        if not glob_pth(d):
            print('no checkpoint:', d, flush=True)
            continue
        lengthrule_tiny(d, device, batch_size, n)


# --- the same axis on a real model -----------------------------------------

def _poj_len_views(recs, L, mode, seed_base=987654321):
    """Rename only the length-L identifiers of each POJ-104 program.

    'all' takes every length-L variable extract_variables finds, 'one' takes a single one, so the
    edit size is constant across L. Replacements are random same-length lowercase strings, i.e.
    the `samelen` scheme the existing N=4 numbers use.
    """
    texts, has, nren, nocc = [], [], [], []
    for i, r in enumerate(recs):
        code = r['code']
        cand = [v for v in extract_variables(code, LANG) if len(v) == L]
        cand = sorted(set(cand), key=cand.index)
        rng = random.Random(seed_base + 1_000_003 * L + i)
        if not cand:
            texts.append(code); has.append(False); nren.append(0); nocc.append(0)
            continue
        tgt = cand if mode == 'all' else [cand[rng.randrange(len(cand))]]
        taken = set(IDENT_RE.findall(code))
        mp = {}
        for old in tgt:
            new = _rand_name(rng, L)
            g = 0
            while (new in taken or new in RESERVED[LANG]) and g < 64:
                new = _rand_name(rng, L); g += 1
            taken.add(new); mp[old] = new
        new_text = apply_mapping(code, mp)
        occ = sum(len(re.findall(r'\b%s\b' % re.escape(o), code)) for o in mp)
        texts.append(new_text); has.append(True); nren.append(len(mp)); nocc.append(occ)
    return texts, np.array(has), nren, nocc


def lengthrule_real(model_id='answerdotai/ModernBERT-base', limit=4000, device='cuda',
                    batch_size=64, lengths=(1, 2, 3, 4, 5, 6, 7, 8)):
    """POJ-104 top-1 retention under renaming restricted to identifiers of ONE length."""
    os.makedirs(CACHE, exist_ok=True)
    tag = model_id.split('/')[-1]
    recs = load_poj(limit)
    labels = np.array([r['label'] for r in recs])
    tok, enc = load_encoder(model_id, device)
    base_names = [set() for _ in recs]
    E0, _ = embed(tok, enc, [r['code'] for r in recs], base_names, device, batch_size)
    ok0, _, _ = top1(E0, E0, labels)
    ntok0 = np.array([len(tok(' '.join(r['code'].split()), truncation=True,
                             max_length=BLOCK_SIZE)['input_ids']) for r in recs])
    out = {'model': model_id, 'n_records': len(recs), 'block_size': BLOCK_SIZE,
           'top1_N0': float(ok0.mean()), 'mean_tokens_N0': float(ntok0.mean()), 'cells': {}}
    print('%s: %d programs, N=0 top-1 %.4f' % (model_id, len(recs), out['top1_N0']), flush=True)
    for mode in ('all', 'one'):
        for L in lengths:
            texts, has, nren, nocc = _poj_len_views(recs, L, mode)
            if has.sum() < 30:
                print('  %-4s L=%d  only %d programs have one; skipped'
                      % (mode, L, int(has.sum())), flush=True)
                continue
            names = [set() for _ in recs]
            P, _ = embed(tok, enc, texts, names, device, batch_size)
            okf, selff, _ = top1(P, E0, labels)
            before = ok0 & has
            rate, sup = retention(before, okf)
            ntok = np.array([len(tok(' '.join(t.split()), truncation=True,
                                     max_length=BLOCK_SIZE)['input_ids']) for t in texts])
            row = {'mode': mode, 'L': L, 'n_has': int(has.sum()),
                   'retention': rate, 'support': sup,
                   'top1': float(okf[has].mean()),
                   'self_cos': float(selff[has].mean()),
                   'mean_renamed': float(np.mean([x for x, h in zip(nren, has) if h])),
                   'mean_occ_renamed': float(np.mean([x for x, h in zip(nocc, has) if h])),
                   'd_tokens': float((ntok - ntok0)[has].mean()),
                   'disp_norm': float(np.linalg.norm(unit(P) - unit(E0), axis=1)[has].mean())}
            out['cells']['%s_L%d' % (mode, L)] = row
            print('  %-4s L=%d  n %5d  ret %.4f  self-cos %.4f  vars %.2f  occ %.2f  dtok %+.1f'
                  % (mode, L, row['n_has'], rate, row['self_cos'], row['mean_renamed'],
                     row['mean_occ_renamed'], row['d_tokens']), flush=True)
    path = os.path.join(CACHE, 'lengthrule_real__%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path, flush=True)
    return out


def _lengthrule_tracing_cells():
    """The 52 (run, length) cells figures.lengthaxis() plots, rebuilt from the same files."""
    sig = json.load(open(os.path.join(WORK, 'signatures.json')))
    gp = os.path.join(WORK, 'generality.json')
    gen = json.load(open(gp)) if os.path.exists(gp) else {}
    Q = {'ckpt_p0.1s': 0.001, 'ckpt_p0.1s_s2': 0.001, 'ckpt_p1s': 0.01, 'ckpt_p1s_s2': 0.01,
         'ckpt_p10': 0.10, 'ckpt_p20': 0.20, 'ckpt_p40': 0.40, 'ckpt_p70': 0.70}
    cells = []
    for run, q in Q.items():
        v = sig.get(run, {}).get('sets', {}).get('rvs')
        if not v:
            continue
        for L, acc in v['bylen']['acc'].items():
            if L == '1':
                continue
            cells.append({'run': run, 'q': q, 'L': int(L), 'series': 'sweep',
                          'names': max(1, int(q * 26 ** int(L))), 'acc': acc,
                          'n': v['bylen']['n'][L]})
    for k, s in [('g1:ckpt_p0.1seven_szM', 'len7'), ('g1:ckpt_p0.1seven_s2_szM', 'len7'),
                 ('g1:ckpt_p0.1s_szS', 'size'), ('g1:ckpt_p0.1s_szXL', 'size')]:
        v = gen.get(k, {}).get('rvs', {}).get('end_by_len')
        for L, d in (v or {}).items():
            if L == '1':
                continue
            cells.append({'run': k, 'q': 0.001, 'L': int(L), 'series': s,
                          'names': max(1, int(0.001 * 26 ** int(L))), 'acc': d['end'],
                          'n': d.get('n')})
    return cells


def _separation(cells, key='acc', thr=1000, qthr=0.05):
    """The statistic figures.lengthaxis() prints: does the rule separate the cells?"""
    A = [c for c in cells if c['names'] >= thr]
    B = [c for c in cells if c['names'] < thr]
    C = [c for c in cells if c['names'] >= thr or c['q'] >= qthr]
    D = [c for c in cells if not (c['names'] >= thr or c['q'] >= qthr)]
    f = lambda xs: [x[key] for x in xs if x[key] is not None]
    out = {'n': len(cells)}
    for lab, hi, lo in (('count', A, B), ('count_or_q', C, D)):
        h, l = f(hi), f(lo)
        out[lab] = {'n_pass': len(h), 'min_pass': min(h) if h else None,
                    'n_fail': len(l), 'max_fail': max(l) if l else None,
                    'mean_pass': float(np.mean(h)) if h else None,
                    'mean_fail': float(np.mean(l)) if l else None,
                    'separates': bool(h and l and min(h) > max(l)),
                    'gap': (min(h) - max(l)) if (h and l) else None}
    return out


def lengthrule_merge():
    """-> work/lengthrule.json. Nothing here is typed in; every number is read off a cache file."""
    out = {'what': 'does the names-per-length rule survive a pooled readout?',
           'tracing_cells': _lengthrule_tracing_cells()}
    out['tracing_rule'] = _separation(out['tracing_cells'])

    cells = []
    for r in LENGTHRULE_RUNS:
        p = os.path.join(CACHE, 'lengthrule_tiny__%s.json' % r)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for k, c in d['cells'].items():
            cells.append({'run': r, 'q': d['q'], 'L': c['L'], 'mode': c['mode'],
                          'names': c['names_of_this_length'], 'acc': c['paired_top1'],
                          'n': c['n_affected'], 'mean_renamed': c['mean_renamed'],
                          'mean_occ_renamed': c['mean_occ_renamed'],
                          'frac_unseen': c['frac_unseen'], 'self_cos': c['self_cos'],
                          'identity_check': d['identity_paired_top1'],
                          'train_lexicon_by_len': d['train_lexicon_by_len']})
    out['retrieval_cells'] = cells
    for mode in ('all', 'one'):
        sel = [c for c in cells if c['mode'] == mode and c['L'] > 1]
        if sel:
            out['retrieval_rule_%s' % mode] = _separation(sel)
    # the count/coverage dissociation pairs: near-equal name count, very different q
    out['dissociation'] = []
    for mode in ('all', 'one'):
        sel = [c for c in cells if c['mode'] == mode and c['L'] > 1]
        for i, a in enumerate(sel):
            for b in sel[i + 1:]:
                if a['q'] == b['q'] or a['acc'] is None or b['acc'] is None:
                    continue
                lo, hi = sorted((a['names'], b['names']))
                if hi <= 3 * lo:                       # matched to within 3x
                    out['dissociation'].append(
                        {'mode': mode, 'lo_q': min(a, b, key=lambda x: x['q']),
                         'hi_q': max(a, b, key=lambda x: x['q'])})

    # --- the nine real corpora, on both axes ------------------------------
    cd = json.load(open(os.path.join(WORK, 'corpus_div.json')))
    real = {}
    for k, v in cd['corpus'].items():
        s = v['S1_coverage']
        real[k] = {'occurrences': v['occurrences'], 'distinct': v['distinct'],
                   'by_len': {L: {'distinct': s[str(L)]['distinct'], 'q_L': s[str(L)]['q_L'],
                                  'occ': s[str(L)]['occurrences'],
                                  'occ_per_name': s[str(L)]['occurrences'] /
                                  max(s[str(L)]['distinct'], 1)}
                              for L in range(1, 8) if str(L) in s},
                   'by_len_sub500k': {L: {'q_L': v['SUB']['q_L'][str(L)],
                                          'distinct': round(v['SUB']['q_L'][str(L)] * 26 ** L)}
                                      for L in range(1, 8) if str(L) in v['SUB']['q_L']}}
    out['real_corpora'] = real
    out['real_corpora_range'] = {
        L: {'distinct': [min(real[k]['by_len'][L]['distinct'] for k in real),
                         max(real[k]['by_len'][L]['distinct'] for k in real)],
            'q_L': [min(real[k]['by_len'][L]['q_L'] for k in real),
                    max(real[k]['by_len'][L]['q_L'] for k in real)],
            'occ_per_name': [min(real[k]['by_len'][L]['occ_per_name'] for k in real),
                             max(real[k]['by_len'][L]['occ_per_name'] for k in real)]}
        for L in range(1, 8)}
    out['tiny_corpora'] = {k: {'q_requested': v.get('q_requested'),
                               'q_measured_generator': v.get('q_measured_generator'),
                               'by_len': {L: {'distinct_scanned': v['S1_coverage'][str(L)]['distinct'],
                                              'occ_per_name': v['S1_coverage'][str(L)]['occurrences'] /
                                              max(v['S1_coverage'][str(L)]['distinct'], 1)}
                                          for L in range(1, 6) if str(L) in v['S1_coverage']}}
                           for k, v in cd['synth'].items()}

    # --- the real models, per length ---------------------------------------
    ws = json.load(open(os.path.join(WORK, 'whyscale.json')))
    out['real_models'] = {}
    for tag in ('ModernBERT-base', 'r4_plain'):
        p = os.path.join(CACHE, 'lengthrule_real__%s.json' % tag)
        row = {'whole_view_baseline': {k: {'retention': v['retention'], 'N': v['N'],
                                           'scheme': v['scheme']}
                                       for k, v in ws.get(tag, {}).get('conditions', {}).items()
                                       if v['scheme'] in ('samelen', 'identity', 'permute')}}
        if os.path.exists(p):
            d = json.load(open(p))
            row.update({'n_records': d['n_records'], 'top1_N0': d['top1_N0'],
                        'cells': d['cells']})
        out['real_models'][tag] = row
    with open(os.path.join(WORK, 'lengthrule.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', os.path.join(WORK, 'lengthrule.json'))
    return out


def _ols(y, X, names):
    """Least squares with t-statistics. Small and explicit so the fit can be checked by eye."""
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float((resid ** 2).sum() / dof)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    ybar = y.mean()
    r2 = 1.0 - float((resid ** 2).sum() / max(((y - ybar) ** 2).sum(), 1e-12))
    return {'names': ['const'] + list(names), 'beta': [float(b) for b in beta],
            'se': [float(x) for x in se],
            't': [float(b / s) if s > 0 else None for b, s in zip(beta, se)],
            'r2': r2, 'n': int(len(y))}


def lengthrule_fits():
    """Add the count-vs-coverage regressions to work/lengthrule.json.

    names = q * 26^L, so log10(names) = log10(q) + L*log10(26): the count axis is an exact linear
    combination of the two. Regressing accuracy on log10(q) and L separately is therefore the only
    way to ask which of the two the accuracy actually follows, and it is well posed here because
    the design has 6 coverages crossed with 4-5 lengths.
    """
    p = os.path.join(WORK, 'lengthrule.json')
    out = json.load(open(p))
    out['fits'] = {}
    def fit(cells, lab):
        cs = [c for c in cells if c.get('acc') is not None and c['L'] > 1 and c.get('q')]
        if len(cs) < 6:
            return
        y = np.array([c['acc'] for c in cs])
        lq = np.log10(np.array([c['q'] for c in cs]))
        LL = np.array([float(c['L']) for c in cs])
        ln = np.log10(np.maximum(np.array([c['names'] for c in cs]), 1.0))
        out['fits'][lab] = {
            'acc~log10q+L': _ols(y, [lq, LL], ['log10_q', 'L']),
            'acc~log10_names': _ols(y, [ln], ['log10_names']),
            'acc~log10q': _ols(y, [lq], ['log10_q']),
            'acc~L': _ols(y, [LL], ['L']),
        }
    fit(out['tracing_cells'], 'tracing')
    for m in ('all', 'one'):
        fit([c for c in out['retrieval_cells'] if c['mode'] == m], 'retrieval_%s' % m)
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote fits ->', p)
    return out
