"""Standard benchmark workflows, independent of the renaming experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    name: str
    trainable: bool
    metrics: tuple[str, ...]
    source: str
    protocol: str


BENCHMARKS = {}
for spec in [
    Benchmark(
        "codexglue_py150",
        True,
        ("token_accuracy",),
        "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token",
        "Lexical token exact match, teacher forced; BOS/EOS/EOL excluded. Sliding causal context.",
    ),
    Benchmark(
        "codexglue_cloze_all",
        False,
        ("accuracy",),
        "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-all",
        "930 official candidates; causal left-context likelihood, all answer pieces. Recovered CSN gold.",
    ),
    Benchmark(
        "codexglue_cloze_maxmin",
        False,
        ("accuracy",),
        "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin",
        "max/min candidates; causal left-context likelihood, all answer pieces. Recovered CSN gold.",
    ),
    Benchmark(
        "bifi",
        True,
        ("repair_accuracy",),
        "https://github.com/michiyasunaga/BIFI",
        "Top-1 critic success AND token Levenshtein distance in [1,4]; heldout chunks 3 and 4.",
    ),
    Benchmark(
        "manytypes4py",
        True,
        (
            "exact_match@1",
            "exact_match@5",
            "exact_match@10",
            "mrr@10",
            "parametric_match@10",
        ),
        "https://github.com/saltudelft/many-types-4-py-dataset",
        "Type4Py exact/outer parametric matches and MRR; train-only candidate vocabulary; causal ranking adaptation.",
    ),
    Benchmark(
        "cruxeval_input",
        False,
        ("pass@1", "pass@5"),
        "https://github.com/facebookresearch/cruxeval",
        "CRUXEval-I: any input giving the reference output; unbiased pass@k estimator.",
    ),
    Benchmark(
        "cruxeval_output",
        False,
        ("pass@1", "pass@5"),
        "https://github.com/facebookresearch/cruxeval",
        "CRUXEval-O: Python value equality; unbiased pass@k estimator.",
    ),
]:
    BENCHMARKS[spec.name] = spec
