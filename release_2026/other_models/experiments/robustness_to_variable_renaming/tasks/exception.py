from ..registry import TASKS
from .cloze import cloze

LABELS = [
    "ValueError",
    "KeyError",
    "AttributeError",
    "TypeError",
    "OSError",
    "IOError",
    "ImportError",
    "IndexError",
    "DoesNotExist",
    "KeyboardInterrupt",
    "StopIteration",
    "AssertionError",
    "SystemExit",
    "RuntimeError",
    "HTTPError",
    "UnicodeDecodeError",
    "NotImplementedError",
    "ValidationError",
    "ObjectDoesNotExist",
    "NameError",
]


def exceptions(model, rows, options):
    converted = []
    for r in rows:
        if "__HOLE__" not in r["code"]:
            raise ValueError("missing exception mask")
        converted.append(
            dict(
                id=r["id"],
                prefix=r["code"].split("__HOLE__")[0],
                gold=r["label"],
                choices=LABELS,
            )
        )
    return cloze(model, converted, options)


TASKS.add("cubert_exception", exceptions)
