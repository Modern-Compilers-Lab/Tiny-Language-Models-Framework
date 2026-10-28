"""Where renaming errors fall, paired by name (Section 4.5 "Localising the failure", Appendix E.2).

    python exp_e2_paired_first_error.py --model CHECKPOINT --corpus CORPUS --run RUN

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
    This analysis selects no heads, so it uses ALL functions (no A/B split).

WHAT IS COUNTED. Each name is followed through both versions of its function. A name is
CORRECT -> INCORRECT if every token of it is predicted correctly in the original function and
some token is wrong after renaming. For those names, the FIRST error in the renamed name is
located: its first token ("start"), a later token ("middle"), or the end (every token right but
the next prediction would continue the identifier: "end"). The paper reports the share at the
first token, overall and split by the number of tokens of the renamed name (1, 2, 3, 4+), because
a one-token name can only fail at its first token or at its end.

The same file also recomputes the net decomposition of copyscore.py ("net"),
so the two statistics can be compared on the same model.

OUTPUT: one JSON with paired.{all,1,2,3,4+}.{items, correct_before, c2i, c2i_at.{start,middle,end},
i2c, i2i} and net.

PROVENANCE. score_snapshot is the function that produced the paper's numbers (analysis
script add_pairedloc45.py). Only the way it loads the library and the model differs.
"""

import sys

import env
from adapter import load_model, source_label


BINS = ("1", "2", "3", "4+")


LOCS = ("start", "middle", "end")


def pbin(n):
    return str(n) if n < 4 else "4+"


def score_snapshot(model_dir, a, device):
    import numpy as np
    import torch
    from types import SimpleNamespace

    DM = env.lib_module("decmech")
    torch.manual_seed(0)
    views = ("identity", "samelen")
    ns = SimpleNamespace(n=a.n, seed=getattr(a, "rename_seed", 564), limit=a.limit)
    _, texts, occs, fn_ids, dropped, _ = DM.real6_occs(ns)
    tok, model, layout = load_model(model_dir, device)
    cfg = layout.config
    L, H, E = cfg.n_layer, cfg.n_head, cfg.n_embd
    hd = d_head = layout.D
    max_len = min(a.max_len, getattr(cfg, "n_positions", a.max_len))
    import re

    idch = re.compile(r"[A-Za-z0-9_]")
    nv = max(len(tok), cfg.vocab_size)
    cont = np.zeros(nv, dtype=bool)  # the token would continue an identifier
    for i in range(len(tok)):
        s = tok.decode([i])
        cont[i] = bool(s) and bool(idch.match(s[0]))
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
    for f in fn_ids:  # same item rule as copyscore.py
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
    order = [
        f
        for f in sorted(fn_ids, key=lambda f: len(TL["identity"][f]), reverse=True)
        if items[f]
    ]
    if not order:
        raise ValueError(
            "No eligible repeated-name items; increase n/max_len or check tokenizer/corpus"
        )
    first = {v: {} for v in views}  # (f, oi) -> 'start'|'middle'|'end'|None
    for v in views:
        for s0 in range(0, len(order), a.batch):
            fb = order[s0 : s0 + a.batch]
            T = max(len(TL[v][f]) for f in fb)
            inp = torch.full((len(fb), T), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(fb), T), dtype=torch.long)
            for j, f in enumerate(fb):
                inp[j, : len(TL[v][f])] = torch.tensor(TL[v][f])
                att[j, : len(TL[v][f])] = 1
            with torch.no_grad():
                pred = (
                    model(input_ids=inp.to(device), attention_mask=att.to(device))
                    .logits.argmax(-1)
                    .cpu()
                    .numpy()
                )
            for j, f in enumerate(fb):
                ids = TL[v][f]
                for oi in items[f]:  # same first-error rule as copyscore.py
                    cov = lay[v][f][oi]["cover"]
                    loc = None
                    for t in cov:
                        if pred[j, t - 1] != ids[t]:
                            loc = "start" if t == cov[0] else "middle"
                            break
                    if loc is None and cont[pred[j, cov[-1]]]:
                        loc = "end"
                    first[v][(f, oi)] = loc
    keys = [(f, oi) for f in order for oi in items[f]]
    net = {v: {w: sum(first[v][k] == w for k in keys) for w in LOCS} for v in views}
    added = {w: net["samelen"][w] - net["identity"][w] for w in LOCS}
    tot = sum(added.values())
    cell = {
        b: {
            "items": 0,
            "correct_before": 0,
            "c2i": 0,
            "i2c": 0,
            "i2i": 0,
            "c2i_at": {w: 0 for w in LOCS},
        }
        for b in BINS + ("all",)
    }
    for f, oi in keys:
        b = pbin(len(lay["samelen"][f][oi]["cover"]))
        o, r = first["identity"][(f, oi)], first["samelen"][(f, oi)]
        for c in (cell[b], cell["all"]):
            c["items"] += 1
            c["correct_before"] += o is None
            if o is None and r is not None:
                c["c2i"] += 1
                c["c2i_at"][r] += 1
            elif o is not None and r is None:
                c["i2c"] += 1
            elif o is not None:
                c["i2i"] += 1
    return {
        "model": source_label(model_dir),
        "n_functions": len(order),
        "n_items": len(keys),
        "net": {
            "failures_original": net["identity"],
            "failures_renamed": net["samelen"],
            "added": added,
            "added_total": tot,
            "share_of_added_at": {
                w: (added[w] / tot if tot > 0 else None) for w in LOCS
            },
        },
        "paired": cell,
        "limit": a.limit,
        "decisions": [
            {
                "function": f,
                "occurrence": oi,
                "id": first["identity"][(f, oi)],
                "ood": first["samelen"][(f, oi)],
                "pieces": len(lay["samelen"][f][oi]["cover"]),
            }
            for f, oi in keys
        ],
    }


def main(argv=None):
    from run import main as run_main

    return run_main(argv, experiment="firsterror")


if __name__ == "__main__":
    sys.exit(main())
