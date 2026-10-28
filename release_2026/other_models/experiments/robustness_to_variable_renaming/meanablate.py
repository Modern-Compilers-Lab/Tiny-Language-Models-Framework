"""Figure 7B: remove the k highest-ranked copy heads by MEAN ablation.

    python exp_fig7b_mean_ablation.py --model CHECKPOINT --corpus CORPUS --run RUN

ITEMS (shared by copyscore, meanablate, force and firsterror)
    The evaluation functions are the ~2,000 CodeSearchNet Python test functions of Appendix D.6
    (real6_occs in lib/decmech.py). An ITEM is an occurrence of a local variable that already
    appeared earlier in the same function, so the answer is in the context. Each item is scored
    with teacher forcing twice: in the original function (view "identity", called ID) and after
    EVERY local variable of the function is renamed to a random name of the same length (view
    "samelen", called OOD). An item is kept only if the name aligns with token boundaries in both
    views and lies inside the 320-token window.
    The QUERY is the position just before the name's first token; the DECISION is the model's most
    likely next token there, compared with the name's first token.
    The functions are split at random (numpy default_rng(0)) into halves A and B. Copy heads are
    RANKED on half A of the renamed functions and every head analysis is MEASURED on half B, so
    choosing and measuring never share data (Appendix E.2).

MEAN ABLATION (Figure 7B). Each removed head's output is replaced, at every position, by its mean
output over all tokens of the half-A functions; the mean is computed separately for the original
and the renamed text. The k = 1..15 highest-ranked copy heads are removed, and, as the control,
10 random sets of k heads drawn from the remaining heads (random generator default_rng(1000 + k),
the same draws as the zero-ablation sweep). Measured: first-token accuracy on half B.

OUTPUT: one JSON with base_acc_{ood,id}, ranked_top16 and curve (one row per k: acc_ood, acc_id,
random_ood_mean, random_ood_sd, the draws and the random heads).

PROVENANCE. score_snapshot is the function that produced the paper's numbers (analysis
script add_meanablate45.py). Only the way it loads the library and the model differs.
"""

import sys
import time

import env
from adapter import load_model, select_heads, source_label


def score_snapshot(model_dir, a, device):
    import numpy as np
    import torch
    from types import SimpleNamespace

    DM = env.lib_module("decmech")
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = DM.device_of(None if device == "auto" else device)
    views = ("identity", "samelen")  # ID (original), OOD (renamed)
    ns = SimpleNamespace(n=a.n, seed=getattr(a, "rename_seed", 564), limit=a.limit)
    _, texts, occs, fn_ids, dropped, _ = DM.real6_occs(ns)
    tok, model, layout = load_model(model_dir, dev)
    cfg = layout.config
    L, H, E = cfg.n_layer, cfg.n_head, cfg.n_embd
    hd = d_head = layout.D
    max_len = min(a.max_len, getattr(cfg, "n_positions", a.max_len))
    t0 = time.time()

    TL, lay = {}, {}
    for v in views:
        enc = tok(
            [texts[v][f] for f in fn_ids],
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        TL[v] = {f: enc["input_ids"][j] for j, f in enumerate(fn_ids)}
        lay[v] = {
            f: DM.token_layout(
                texts[v][f], [tuple(o) for o in enc["offset_mapping"][j]], occs[f][v]
            )[1]
            for j, f in enumerate(fn_ids)
        }
    items = {}
    for f in fn_ids:
        seen, it = set(), []
        for oi, (_, _, k) in enumerate(occs[f]["identity"]):
            if k in seen and all(
                lay[v][f][oi]["name_ok"]
                and lay[v][f][oi]["cover"][0] >= 1
                and lay[v][f][oi]["cover"][-1] + 1 < len(TL[v][f])
                for v in views
            ):
                it.append(oi)
            seen.add(k)
        items[f] = it
    perm = np.random.default_rng(getattr(a, "split_seed", 0)).permutation(len(fn_ids))
    split_a = {fn_ids[i] for i in perm[: len(fn_ids) // 2]}
    order = [
        f
        for f in sorted(fn_ids, key=lambda f: len(TL["identity"][f]), reverse=True)
        if items[f]
    ]
    if not order:
        raise ValueError(
            "No eligible repeated-name items; increase n/max_len or check tokenizer/corpus"
        )
    if not any(f in split_a for f in order) or not any(f not in split_a for f in order):
        raise ValueError("Empty selection/evaluation split")
    fa = [f for f in order if f in split_a]
    fb_all = [f for f in order if f not in split_a]

    def batch_inputs(view, fb):
        T = max(len(TL[view][f]) for f in fb)
        inp = torch.full((len(fb), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(fb), T), dtype=torch.long)
        for j, f in enumerate(fb):
            ids = TL[view][f]
            inp[j, : len(ids)] = torch.tensor(ids)
            att[j, : len(ids)] = 1
        return inp.to(dev), att.to(dev)

    state = {"mode": None, "heads": (), "mean": None}
    acc_sum = {}

    def pre(li):
        def hook(mod, args):
            x = args[0]
            if state["mode"] == "mean":  # accumulate over real tokens
                m = state["mask"][..., None].to(x.dtype)
                acc_sum[li] = acc_sum.get(li, 0) + (x * m).sum((0, 1)).double()
                return None
            heads = [h for (l, h) in state["heads"] if l == li]
            if not heads:
                return None
            x = x.clone()
            for h in heads:
                sl = slice(h * hd, (h + 1) * hd)
                x[..., sl] = state["mean"][li][sl].to(x.dtype)
            return (x,) + tuple(args[1:])

        return hook

    handles = [
        proj.register_forward_pre_hook(pre(li))
        for li, proj in enumerate(layout.projections)
    ]

    # ---- copy heads: same ranking as copyscore.py (renamed split A) ---------------------------
    ssum, sq = np.zeros((L, H)), 0
    for s0 in range(0, len(fa), a.batch):
        fb = fa[s0 : s0 + a.batch]
        inp, att = batch_inputs("samelen", fb)
        with torch.no_grad():
            A = torch.stack(
                model(
                    input_ids=inp, attention_mask=att, output_attentions=True
                ).attentions,
                1,
            ).float()
        for j, f in enumerate(fb):
            for oi in items[f]:
                key = occs[f]["samelen"][oi][2]
                q = lay["samelen"][f][oi]["cover"][0] - 1
                kk = [
                    lay["samelen"][f][pj]["cover"][0]
                    for pj in range(oi)
                    if occs[f]["samelen"][pj][2] == key
                    and lay["samelen"][f][pj]["cover"]
                ]
                if kk:
                    ssum += A[j, :, :, q, kk].sum(-1).cpu().numpy()
                    sq += 1
        del A
    sel = ssum / max(sq, 1)
    ranked = sorted(
        ((float(sel[l, h]), l, h) for l in range(L) for h in range(H)), reverse=True
    )
    ranked = select_heads(ranked, a, layout)
    top = [(l, h) for _, l, h in ranked]
    print(
        "copy heads (split A): %s" % [(l, h, round(s, 3)) for s, l, h in ranked[:6]],
        flush=True,
    )

    # ---- the mean output of every head, per view, over split A --------------------------------
    means = {}
    for v in views:
        acc_sum.clear()
        ntok = 0
        for s0 in range(0, len(fa), a.batch):
            fb = fa[s0 : s0 + a.batch]
            inp, att = batch_inputs(v, fb)
            state.update(mode="mean", mask=att)
            with torch.no_grad():
                model(input_ids=inp, attention_mask=att)
            state.update(mode=None, mask=None)
            ntok += int(att.sum())
        means[v] = {li: (acc_sum[li] / ntok).float() for li in range(L)}
    print(
        "head means computed over %d tokens per view  (%.1f min)"
        % (ntok, (time.time() - t0) / 60),
        flush=True,
    )

    def accuracy(view, heads=()):
        state.update(heads=tuple(heads), mean=means[view])
        n = ok = 0
        try:
            for s0 in range(0, len(fb_all), a.batch):
                fb = fb_all[s0 : s0 + a.batch]
                inp, att = batch_inputs(view, fb)
                with torch.no_grad():
                    pred = (
                        model(input_ids=inp, attention_mask=att)
                        .logits.argmax(-1)
                        .cpu()
                        .numpy()
                    )
                for j, f in enumerate(fb):
                    ids = TL[view][f]
                    for oi in items[f]:
                        c0 = lay[view][f][oi]["cover"][0]
                        n += 1
                        ok += int(pred[j, c0 - 1]) == ids[c0]
        finally:
            state.update(heads=(), mean=None)
        return ok / max(n, 1), n

    base_ood, n_items = accuracy("samelen")
    base_id, _ = accuracy("identity")
    print(
        "split B: %d occurrences; first-token accuracy ID %.4f, OOD %.4f"
        % (n_items, base_id, base_ood),
        flush=True,
    )
    rows = []
    for k in range(1, a.kmax + 1):
        rr = np.random.default_rng(1000 + k)  # the same draws as copyscore.py
        pool = top[k:]
        draws = [
            [pool[i] for i in rr.choice(len(pool), size=k, replace=False)]
            for _ in range(a.n_random)
        ]
        r_ood = [accuracy("samelen", d)[0] for d in draws]
        r_id = [accuracy("identity", d)[0] for d in draws]
        rec = {
            "k": k,
            "heads": [list(x) for x in top[:k]],
            "acc_ood": accuracy("samelen", top[:k])[0],
            "acc_id": accuracy("identity", top[:k])[0],
            "random_ood_mean": float(np.mean(r_ood)),
            "random_ood_sd": float(np.std(r_ood)),
            "random_ood_draws": r_ood,
            "random_id_mean": float(np.mean(r_id)),
            "random_id_draws": r_id,
            "random_heads": [[list(x) for x in d] for d in draws],
        }
        rows.append(rec)
        print(
            "  k=%-2d copy heads: OOD %.4f ID %.4f | random: OOD %.4f +- %.4f (min %.4f)  "
            "ID %.4f   (%.1f min)"
            % (
                k,
                rec["acc_ood"],
                rec["acc_id"],
                rec["random_ood_mean"],
                rec["random_ood_sd"],
                min(r_ood),
                rec["random_id_mean"],
                (time.time() - t0) / 60,
            ),
            flush=True,
        )
    for h_ in handles:
        h_.remove()
    return {
        "model": source_label(model_dir),
        "n_layer": L,
        "n_head": H,
        "max_len": max_len,
        "n_functions": len(order),
        "n_occurrences_splitB": n_items,
        "base_acc_ood": base_ood,
        "base_acc_id": base_id,
        "ablation": "mean",
        "ranked_top16": [
            {"layer": l, "head": h, "score": s} for s, l, h in ranked[:16]
        ],
        "curve": rows,
        "kmax": a.kmax,
        "n_random": a.n_random,
        "minutes": (time.time() - t0) / 60,
    }


def main(argv=None):
    from run import main as run_main

    return run_main(argv, experiment="meanablate")


if __name__ == "__main__":
    sys.exit(main())
