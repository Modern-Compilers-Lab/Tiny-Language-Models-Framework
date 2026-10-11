"""Shared lexical rendering and fractional metrics from codelab."""


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


def fraction(values):
    values = list(values)
    k = sum(bool(v) for v in values)
    n = len(values)
    return dict(correct=k, count=n, accuracy=k / n if n else None)
