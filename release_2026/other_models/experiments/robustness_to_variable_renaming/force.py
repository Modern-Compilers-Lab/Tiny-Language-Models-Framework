"""Forced attention (Section 4.5, Appendix E.2): redirect the copy heads at a failing position.

    python exp_table6_forced_attention.py --model CHECKPOINT --corpus CORPUS --run RUN --ks 3,12

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

FAILURE: a half-B item whose first token the model predicts correctly with the original names
but incorrectly after renaming.

INTERVENTION. At the query only, each selected head's attention row is replaced with equal weights
on the target positions, and the head's output is recomputed as sum_j a_j v_j with the renamed
function's own value vectors (multi-query attention: all heads of a layer share one value vector
per position). Nothing else changes; the layers above are recomputed. A REPAIR is a failure for
which the intervened model's most likely token is the correct first token.

CONDITIONS, for k = 3 and k = 12 heads:
    copy_this    the k highest-ranked copy heads attend to the first tokens of earlier
                 occurrences of the same (renamed) variable
    random_this  k random heads, one drawn from the same layer as each selected head
                 (numpy default_rng(--seed)); if a layer has no unselected head left, the first
                 free head of the model is used. Same targets.
    copy_other   the copy heads attend to the first tokens of earlier occurrences of ALL other
                 variables of the function; restricted to failures where such occurrences exist.
                 Reported: repairs, and how often the prediction becomes the first token of one
                 of those variables ("writes that identifier"), next to the same rate with no
                 intervention.

SELF-CHECK (recorded under "selfcheck"): giving every head its own attention row reproduces the
model's head outputs to within ~4e-5 and changes no prediction.

OUTPUT: one JSON with n_failures, by_k.{3,12}.{copy_this_rescue, random_this_rescue,
copy_other_rescue, copy_other_writes_other}, no_intervention_writes_other, ranked_top16, selfcheck.

PROVENANCE. score_snapshot is the function that produced the paper's numbers (analysis
script add_force45.py). Only the way it loads the library and the model differs.
"""

import sys
import time

import env
from adapter import load_model, select_heads, source_label


def _rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None}


def score_snapshot(model_dir, a, device):
    import numpy as np
    import torch
    from types import SimpleNamespace

    DM = env.lib_module("decmech")
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = DM.device_of(None if device == "auto" else device)
    views = ("identity", "samelen")  # original, renamed
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

    def firsts(v, f, oi, same):
        k = occs[f][v][oi][2]
        return [
            lay[v][f][pj]["cover"][0]
            for pj in range(oi)
            if (occs[f][v][pj][2] == k) == same and lay[v][f][pj]["cover"]
        ]

    def batch_inputs(view, fb):
        T = max(len(TL[view][f]) for f in fb)
        inp = torch.full((len(fb), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(fb), T), dtype=torch.long)
        for j, f in enumerate(fb):
            ids = TL[view][f]
            inp[j, : len(ids)] = torch.tensor(ids)
            att[j, : len(ids)] = 1
        return inp.to(dev), att.to(dev)

    # ---- copy heads (same ranking as copyscore.py: renamed functions of split A) ------------
    ssum, sq = np.zeros((L, H)), 0
    fa = [f for f in order if f in split_a]
    for s0 in range(0, len(fa), a.batch):
        fb = fa[s0 : s0 + a.batch]
        inp, att = batch_inputs("samelen", fb)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=att, output_attentions=True)
        A = torch.stack(out.attentions, 1).float()
        for j, f in enumerate(fb):
            for oi in items[f]:
                q = lay["samelen"][f][oi]["cover"][0] - 1
                kk = firsts("samelen", f, oi, True)
                if kk:
                    ssum += A[j, :, :, q, kk].sum(-1).cpu().numpy()
                    sq += 1
        del out, A
    sel = ssum / max(sq, 1)
    ranked = sorted(
        ((float(sel[l, h]), l, h) for l in range(L) for h in range(H)), reverse=True
    )
    ranked = select_heads(ranked, a, layout)
    top = [(l, h) for _, l, h in ranked]
    print(
        "copy heads (split A, %d queries): %s"
        % (sq, [(l, h, round(s, 3)) for s, l, h in ranked[:6]]),
        flush=True,
    )
    rng = np.random.default_rng(a.seed)
    rand_sets = {}
    for k in a.ks:
        taken, rs = set(top[:k]), []
        for l, _h in top[:k]:
            free = [h for h in range(H) if (l, h) not in taken and (l, h) in top]
            pick = (
                (l, int(rng.choice(free)))
                if free
                else [
                    (ll, hh)
                    for ll in range(L)
                    for hh in range(H)
                    if (ll, hh) not in taken and (ll, hh) in top
                ][0]
            )
            taken.add(pick)
            rs.append(pick)
        rand_sets[k] = rs

    # ---- hooks: value vectors, and one-position head outputs from a forced attention row ----
    state = {"heads": ()}
    vals, capt = {}, {}

    def c_proj_pre(li):
        def pre(mod, args):
            x = args[0]
            if state.get("mode") == "capture":
                capt[li] = x.detach()
                return None
            heads = [h for (l, h) in state["heads"] if l == li]
            if state.get("mode") != "attn" or not heads:
                return None
            x = x.clone()
            b = torch.arange(x.shape[0], device=x.device)
            for h in heads:
                w = state["rows"][(li, h)]  # [B,T]
                x[b, state["pos"], h * hd : (h + 1) * hd] = torch.einsum(
                    "bt,btd->bd", w, vals[li][:, : w.shape[1], h, :]
                )
            return (x,) + tuple(args[1:])

        return pre

    handles = layout.value_hooks(
        lambda li, v: (
            vals.__setitem__(li, v) if state.get("mode") in ("attn", "self") else None
        )
    )
    for li, proj in enumerate(layout.projections):
        handles.append(proj.register_forward_pre_hook(c_proj_pre(li)))

    def forward(view, fb, pos, mode=None, heads=(), rows=None, attentions=False):
        state.update(mode=mode, heads=tuple(heads), pos=pos, rows=rows)
        inp, att = batch_inputs(view, fb)
        try:
            with torch.no_grad():
                out = model(
                    input_ids=inp, attention_mask=att, output_attentions=attentions
                )
        finally:
            state.update(mode=None, heads=(), rows=None)
        return out

    cnt = {}

    def add(key, hit):
        c = cnt.setdefault(key, [0, 0])
        c[0] += int(hit)
        c[1] += 1

    selfcheck = {"max_abs_diff": None, "pred_unchanged": [0, 0]}
    fb_all = [f for f in order if f not in split_a]
    occ_all = [(f, oi) for f in fb_all for oi in items[f]]
    n_occ, ok_id, ok_ood, n_fail = len(occ_all), 0, 0, 0
    for s0 in range(0, len(occ_all), a.batch):
        chunk = occ_all[s0 : s0 + a.batch]
        fb = [f for f, _ in chunk]
        B = len(chunk)
        bidx = torch.arange(B, device=dev)
        qo = torch.tensor(
            [lay["identity"][f][oi]["cover"][0] - 1 for f, oi in chunk], device=dev
        )
        qr = torch.tensor(
            [lay["samelen"][f][oi]["cover"][0] - 1 for f, oi in chunk], device=dev
        )
        go = [TL["identity"][f][lay["identity"][f][oi]["cover"][0]] for f, oi in chunk]
        gr = [TL["samelen"][f][lay["samelen"][f][oi]["cover"][0]] for f, oi in chunk]
        po = forward("identity", fb, qo).logits[bidx, qo].argmax(-1).tolist()
        base = forward("samelen", fb, qr, attentions=(s0 == 0))
        pr = base.logits[bidx, qr].argmax(-1).tolist()
        ok_id += sum(po[j] == go[j] for j in range(B))
        ok_ood += sum(pr[j] == gr[j] for j in range(B))
        # self-check on the first chunk: own attention reproduces the model exactly
        if s0 == 0:
            own = torch.stack(base.attentions, 1)[
                bidx, :, :, qr, :
            ].float()  # [B,L,H,T]
            state["mode"] = "capture"
            capt.clear()
            with torch.no_grad():
                inp, att = batch_inputs("samelen", fb)
                model(input_ids=inp, attention_mask=att)
            real = torch.stack([capt[l][bidx, qr, :] for l in range(L)], 1)
            state["mode"] = "self"
            vals.clear()
            with torch.no_grad():
                model(input_ids=inp, attention_mask=att)
            state["mode"] = None
            T_ = own.shape[-1]
            recon = torch.stack(
                [
                    torch.cat(
                        [
                            torch.einsum(
                                "bt,btd->bd", own[:, l, h, :], vals[l][:, :T_, h, :]
                            )
                            for h in range(H)
                        ],
                        -1,
                    )
                    for l in range(L)
                ],
                1,
            )
            selfcheck["max_abs_diff"] = float((recon - real).abs().max())
            allh = [(l, h) for l in range(L) for h in range(H)]
            p1 = (
                forward(
                    "samelen",
                    fb,
                    qr,
                    "attn",
                    allh,
                    {(l, h): own[:, l, h, :] for l, h in allh},
                )
                .logits[bidx, qr]
                .argmax(-1)
                .tolist()
            )
            selfcheck["pred_unchanged"] = [sum(x == y for x, y in zip(pr, p1)), B]
            print(
                "self-check: recomputed head outputs differ by at most %.2e; %d/%d predictions "
                "unchanged when every head is given its own attention"
                % (selfcheck["max_abs_diff"], selfcheck["pred_unchanged"][0], B),
                flush=True,
            )
            del own
        del base
        fail = [po[j] == go[j] and pr[j] != gr[j] for j in range(B)]
        if not any(fail):
            continue
        T = int(max(len(TL["samelen"][f]) for f in fb))
        same = [firsts("samelen", f, oi, True) for f, oi in chunk]
        other = [firsts("samelen", f, oi, False) for f, oi in chunk]
        other_first = [
            {TL["samelen"][f][p] for p in other[j]} for j, (f, _) in enumerate(chunk)
        ]

        def forced(targets):
            r = torch.zeros((B, T), device=dev)
            for j in range(B):
                if targets[j]:
                    r[j, targets[j]] = 1.0 / len(targets[j])
            return r

        r_same, r_other = forced(same), forced(other)
        res = {}
        for k in a.ks:
            for name, hs, rows in (
                ("copy_this", top[:k], r_same),
                ("random_this", rand_sets[k], r_same),
                ("copy_other", top[:k], r_other),
            ):
                res[(k, name)] = (
                    forward(
                        "samelen", fb, qr, "attn", hs, {(l, h): rows for l, h in hs}
                    )
                    .logits[bidx, qr]
                    .argmax(-1)
                    .tolist()
                )
        for j in range(B):
            if not fail[j]:
                continue
            n_fail += 1
            if other[j]:
                add(("none", "writes_other"), pr[j] in other_first[j])
            for k in a.ks:
                add((k, "copy_this", "rescue"), res[(k, "copy_this")][j] == gr[j])
                add((k, "random_this", "rescue"), res[(k, "random_this")][j] == gr[j])
                if other[j]:
                    add((k, "copy_other", "rescue"), res[(k, "copy_other")][j] == gr[j])
                    add(
                        (k, "copy_other", "writes_other"),
                        res[(k, "copy_other")][j] in other_first[j],
                    )
        if (s0 // a.batch) % 25 == 0:
            print(
                "  %d/%d occurrences  (%d failures so far, %.1f min)"
                % (min(s0 + a.batch, n_occ), n_occ, n_fail, (time.time() - t0) / 60),
                flush=True,
            )
    for h_ in handles:
        h_.remove()

    def R(*key):
        c = cnt.get(key)
        return _rate(c[0], c[1]) if c else _rate(0, 0)

    by_k = {
        str(k): {
            "heads": [list(x) for x in top[:k]],
            "random_heads": [list(x) for x in rand_sets[k]],
            "copy_this_rescue": R(k, "copy_this", "rescue"),
            "random_this_rescue": R(k, "random_this", "rescue"),
            "copy_other_rescue": R(k, "copy_other", "rescue"),
            "copy_other_writes_other": R(k, "copy_other", "writes_other"),
        }
        for k in a.ks
    }
    none_other = R("none", "writes_other")
    print(
        "split B: %d occurrences; first-token accuracy original %.3f, renamed %.3f; %d failures"
        % (n_occ, ok_id / max(1, n_occ), ok_ood / max(1, n_occ), n_fail),
        flush=True,
    )
    for k in a.ks:
        b_ = by_k[str(k)]
        print(
            "  k=%-2d rescue: copy heads -> this name %.3f | random heads -> this name %.3f | "
            "copy heads -> another name %.3f (writes that name %.3f; no intervention %.3f)"
            % (
                k,
                b_["copy_this_rescue"]["rate"] or 0,
                b_["random_this_rescue"]["rate"] or 0,
                b_["copy_other_rescue"]["rate"] or 0,
                b_["copy_other_writes_other"]["rate"] or 0,
                none_other["rate"] or 0,
            ),
            flush=True,
        )
    return {
        "model": source_label(model_dir),
        "n_layer": L,
        "n_head": H,
        "d_head": hd,
        "max_len": max_len,
        "n_functions": len(order),
        "n_functions_dropped_by_checks": dict(dropped),
        "n_occurrences_splitB": n_occ,
        "acc_original_splitB": ok_id / max(1, n_occ),
        "acc_renamed_splitB": ok_ood / max(1, n_occ),
        "n_failures": n_fail,
        "n_selection_queries": sq,
        "ranked_top16": [
            {"layer": l, "head": h, "score": s} for s, l, h in ranked[:16]
        ],
        "selfcheck": selfcheck,
        "by_k": by_k,
        "no_intervention_writes_other": none_other,
        "ks": list(a.ks),
        "minutes": (time.time() - t0) / 60,
    }


def main(argv=None):
    from run import main as run_main

    return run_main(argv, experiment="force")


if __name__ == "__main__":
    sys.exit(main())
