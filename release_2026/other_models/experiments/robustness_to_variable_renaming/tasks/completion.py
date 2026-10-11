"""PY150 lexical-token accuracy."""

from ..registry import TASKS


def render(tokens):
    text = ""
    spans = []
    for tok in tokens:
        if tok in ("<s>", "</s>"):
            spans.append(None)
            continue
        if tok == "<EOL>":
            text += "\n"
            spans.append(None)
            continue
        if text and not text.endswith((" ", "\n")):
            text += " "
        a = len(text)
        text += tok
        spans.append((a, len(text)))
    return text, spans


def token_completion(model, rows, options):
    rendered = [render(r["code"]) for r in rows]
    pred = model.token_predictions(
        [x[0] for x in rendered], options.get("max_len", 320)
    )
    out = []
    if len(pred) != len(rows):
        raise ValueError("missing prediction rows")
    for row, (text, spans), (ids, offs, prs) in zip(rows, rendered, pred):
        for j, span in enumerate(spans):
            if span is None:
                continue
            a, b = span
            ks = [
                k
                for k, (x, y) in enumerate(offs)
                if k > 0 and y > a and x < b and text[x:y].strip()
            ]
            if not ks or offs[ks[0]][0] > a or offs[ks[-1]][1] < b:
                continue
            # Exclude pieces crossing a lexical boundary (whitespace prefixes are allowed).
            if any(text[offs[k][0] : a].strip() for k in ks if offs[k][0] < a):
                continue
            if any(text[b : offs[k][1]].strip() for k in ks if offs[k][1] > b):
                continue
            out.append(
                dict(
                    id=f"{row['id']}:{j}",
                    gold=[ids[k] for k in ks],
                    prediction=[prs[k - 1] for k in ks],
                    correct=all(ids[k] == prs[k - 1] for k in ks),
                )
            )
    return out


TASKS.add("py150_token", token_completion)
