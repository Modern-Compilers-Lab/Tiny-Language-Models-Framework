"""Figure 7A (copy score) and the zero-ablation check of Section 4.5.

    python exp_fig7a_copy_score.py --model CHECKPOINT --corpus CORPUS --run RUN
    python exp_zero_ablation.py --model CHECKPOINT --corpus CORPUS --run RUN --kmax 15

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

COPY SCORE (Figure 7A). For head h and query q: the sum of the attention weights at q on the
first-token positions of all earlier occurrences of the same variable. A head's score is the mean
over queries. Heads are ranked by this score on the renamed functions of half A; Figure 7A reports
the largest half-B score among the five highest-ranked heads, separately on ID and OOD.

ZERO ABLATION (the sentence "setting a head's output to zero damages the model even when the heads
are selected randomly"). Remove the k = 1..15 highest-ranked heads by zeroing their slice of the
input of attn.c_proj (GPT-BigCode concatenates the per-head outputs there, so columns
[h*64, (h+1)*64) are head h and nothing else, also with multi-query attention), and compare with
10 random sets of k heads (50 sets at k = 3) drawn from the other heads. Reported: first-token
accuracy on half B, OOD and ID.

ALSO RECORDED ("s42"): the net decomposition of the errors added by renaming (wrong renamed minus
wrong original, by first-error location). It is superseded in the paper by the paired analysis of
firsterror.py and is kept only for comparison.

OUTPUT: one JSON with keys splitB.{id,ood}.highest_copy_score, ranked_top16, curve (zero
ablation, one row per k), s42.

PROVENANCE. score_snapshot is the function that produced the paper's numbers (analysis
script add_mech45.py). Only the way it loads the library and the model differs.
"""

import re
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
    views = ("identity", "samelen")  # ID, OOD
    ns = SimpleNamespace(n=a.n, seed=getattr(a, "rename_seed", 564), limit=a.limit)
    _, texts, occs, fn_ids, dropped, _ = DM.real6_occs(ns)
    tok, model, layout = load_model(model_dir, dev)
    cfg = layout.config
    L, H, E = cfg.n_layer, cfg.n_head, cfg.n_embd
    hd = d_head = layout.D
    max_len = min(a.max_len, getattr(cfg, "n_positions", a.max_len))
    nv = max(len(tok), cfg.vocab_size)
    idch, idstart = re.compile(r"[A-Za-z0-9_]"), re.compile(r"^\s*[A-Za-z_]")
    cont = np.zeros(nv, dtype=bool)  # token would continue an identifier
    idlike = np.zeros(nv, dtype=bool)  # token can begin an identifier
    for i in range(len(tok)):
        s = tok.decode([i])
        cont[i] = bool(s) and bool(idch.match(s[0]))
        idlike[i] = bool(idstart.match(s))
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
    n_occ = sum(len(items[f]) for f in order)
    print(
        "%d functions with repeated names, %d occurrences; %d layers x %d heads; device %s"
        % (len(order), n_occ, L, H, dev),
        flush=True,
    )

    def answer_keys(v, f, oi):
        k = occs[f][v][oi][2]
        return [
            lay[v][f][pj]["cover"][0]
            for pj in range(oi)
            if occs[f][v][pj][2] == k and lay[v][f][pj]["cover"]
        ]

    def hooks_for(heads):
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        hs = []
        for l, hh in by_layer.items():

            def pre(mod, args, hh=hh):
                x = args[0].clone()
                for h in hh:
                    x[..., h * d_head : (h + 1) * d_head] = 0.0
                return (x,) + args[1:]

            hs.append(layout.projections[l].register_forward_pre_hook(pre))
        return hs

    def run(view, part, scores=False, where=False, heads=()):
        n = ok = wrong = offctx = 0
        loc = {"start": 0, "middle": 0, "end": 0}
        ssum, sq = np.zeros((L, H)), 0
        fs = [f for f in order if part == "all" or (f in split_a) == (part == "A")]
        hs = hooks_for(heads)
        try:
            for s0 in range(0, len(fs), a.batch):
                fb = fs[s0 : s0 + a.batch]
                T = max(len(TL[view][f]) for f in fb)
                inp = torch.full((len(fb), T), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(fb), T), dtype=torch.long)
                for j, f in enumerate(fb):
                    ids = TL[view][f]
                    inp[j, : len(ids)] = torch.tensor(ids)
                    att[j, : len(ids)] = 1
                with torch.no_grad():
                    out = model(
                        input_ids=inp.to(dev),
                        attention_mask=att.to(dev),
                        output_attentions=scores,
                    )
                pred = out.logits.argmax(-1).cpu().numpy()
                A = torch.stack(out.attentions, 1).float() if scores else None
                for j, f in enumerate(fb):
                    ids = TL[view][f]
                    for oi in items[f]:
                        cov = lay[view][f][oi]["cover"]
                        q = cov[0] - 1
                        n += 1
                        p = int(pred[j, q])
                        if p == ids[cov[0]]:
                            ok += 1
                        else:
                            wrong += 1
                            if idlike[p] and p not in ids[: q + 1]:
                                offctx += 1
                        if where:
                            first = None
                            for t in cov:
                                if pred[j, t - 1] != ids[t]:
                                    first = "start" if t == cov[0] else "middle"
                                    break
                            if first is None and cont[pred[j, cov[-1]]]:
                                first = "end"
                            if first:
                                loc[first] += 1
                        if scores:
                            kk = answer_keys(view, f, oi)
                            if kk:
                                ssum += A[j, :, :, q, kk].sum(-1).cpu().numpy()
                                sq += 1
        finally:
            for h in hs:
                h.remove()
        r = {
            "first_piece_acc": ok / max(n, 1),
            "n": n,
            "out_of_context_rate": offctx / max(wrong, 1),
            "n_wrong_first_piece": wrong,
        }
        if where:
            r["first_error"] = loc
        if scores:
            r["scores"] = ssum / max(sq, 1)
            r["n_score_queries"] = sq
        return r

    # ---- 4.2 ------------------------------------------------------------------------------
    b = {v: run(v, "all", where=True) for v in views}
    added = {
        w: b["samelen"]["first_error"][w] - b["identity"]["first_error"][w]
        for w in ("start", "middle", "end")
    }
    tot = sum(b["samelen"]["first_error"].values()) - sum(
        b["identity"]["first_error"].values()
    )
    s42 = {
        "n_names": b["samelen"]["n"],
        "failures_original": b["identity"]["first_error"],
        "failures_renamed": b["samelen"]["first_error"],
        "added_by_renaming": added,
        "added_total": tot,
        "share_of_added_at": {w: (added[w] / tot if tot > 0 else None) for w in added},
    }
    print(
        "  4.2  wrong names: original %d, renamed %d, added by renaming %d %s  (%.1f min)"
        % (
            sum(s42["failures_original"].values()),
            sum(s42["failures_renamed"].values()),
            tot,
            added,
            (time.time() - t0) / 60,
        ),
        flush=True,
    )

    # ---- 4.3 correlational ------------------------------------------------------------------
    sel = run("samelen", "A", scores=True)["scores"]
    ranked = sorted(
        ((float(sel[l, h]), l, h) for l in range(L) for h in range(H)), reverse=True
    )
    ranked = select_heads(ranked, a, layout)
    top = [(l, h) for _, l, h in ranked]
    ev = {}
    for v, key in (("identity", "id"), ("samelen", "ood")):
        r = run(v, "B", scores=True)
        sc = r.pop("scores")
        r["top5"] = [
            {"layer": l, "head": h, "score": float(sc[l, h])} for l, h in top[:5]
        ]
        r["highest_copy_score"] = max(float(sc[l, h]) for l, h in top[:5])
        ev[key] = r
    print(
        "  4.3  top-5 heads %s | copy score ID %.3f OOD %.3f | first-token acc ID %.4f OOD %.4f"
        % (
            [(l, h, round(s, 3)) for s, l, h in ranked[:5]],
            ev["id"]["highest_copy_score"],
            ev["ood"]["highest_copy_score"],
            ev["id"]["first_piece_acc"],
            ev["ood"]["first_piece_acc"],
        ),
        flush=True,
    )

    # ---- 4.3 causal -------------------------------------------------------------------------
    rows = []
    for k in range(1, a.kmax + 1):
        rec = {
            "k": k,
            "heads": [list(x) for x in top[:k]],
            "acc_id": run("identity", "B", heads=top[:k])["first_piece_acc"],
            "acc_ood": run("samelen", "B", heads=top[:k])["first_piece_acc"],
        }
        rr = np.random.default_rng(1000 + k)
        pool = top[k:]
        nr = a.n_random_k3 if k == 3 else a.n_random
        r = [
            run(
                "samelen",
                "B",
                heads=[pool[i] for i in rr.choice(len(pool), size=k, replace=False)],
            )["first_piece_acc"]
            for _ in range(nr)
        ]
        rec.update(
            random_ood_mean=float(np.mean(r)),
            random_ood_sd=float(np.std(r)),
            n_random=nr,
        )
        rows.append(rec)
        print(
            "  k=%-2d OOD %.4f  ID %.4f  random OOD %.4f +- %.4f   (%.1f min)"
            % (
                k,
                rec["acc_ood"],
                rec["acc_id"],
                rec["random_ood_mean"],
                rec["random_ood_sd"],
                (time.time() - t0) / 60,
            ),
            flush=True,
        )
    return {
        "model": source_label(model_dir),
        "n_layer": L,
        "n_head": H,
        "d_head": d_head,
        "multi_query": bool(getattr(cfg, "multi_query", False)),
        "max_len": max_len,
        "n_functions": len(order),
        "n_functions_dropped_by_checks": dict(dropped),
        "s42": s42,
        "splitB": ev,
        "ranked_top16": [
            {"layer": l, "head": h, "score": s} for s, l, h in ranked[:16]
        ],
        "curve": rows,
        "kmax": a.kmax,
        "n_random": a.n_random,
        "n_random_k3": a.n_random_k3,
        "minutes": (time.time() - t0) / 60,
    }


def main(argv=None):
    from run import main as run_main

    return run_main(argv, experiment="copyscore")


if __name__ == "__main__":
    sys.exit(main())
