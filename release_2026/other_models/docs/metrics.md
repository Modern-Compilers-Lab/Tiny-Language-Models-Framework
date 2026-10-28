# Metrics

Result rates are fractions: `0.25` means 25%. Read them together with their
counts and the `subset` flag. A score of zero has a denominator; `null` means the
requested statistic is undefined or unavailable.

## Standard benchmarks

| Route | Definition |
| --- | --- |
| PY150 | `token_accuracy = correct / count` over complete lexical targets, with every BPE piece required to match |
| Cloze | `accuracy = correct / count` for the highest-scoring official candidate, using causal left context |
| BIFI | `repair_accuracy` is the fraction satisfying the critic and making between one and four token edits |
| ManyTypes4Py | Exact match at 1/5/10, reciprocal rank through 10 and parametric match through 10 |
| CRUXEval | Mean pass@1 and pass@5 across tasks, using each task's number of sampled answers and successes |

PY150's `count` is scored lexical tokens, not input files or BPE pieces.
`evaluated_examples` records input files. Its adapter uses overlapping windows
and excludes partial targets at context boundaries.

Cloze ranks the sum of the answer tokens' log probabilities. If two candidates tie, the
standard route selects the first one. Gold labels are recovered by exact token
matching against CodeSearchNet; ambiguous or missing labels fail preparation.

BIFI uses the released token representation. Its bracket critic counts each
bracket type independently; other cases require restored code to parse. An
unchanged output fails the edit rule, even if it parses. `by_error_type` retains
the same correctness rule within each group.

ManyTypes4Py uses a frozen candidate vocabulary from TRAIN or an external file.
`mrr@10` contributes `1/rank` for a correct answer in the first ten candidates,
and zero otherwise. Parametric matching accepts an exact type or a match to the
outer generic type according to the evaluator's rule. `by_kind` separates
Parameter, Return and Variable targets.

For CRUXEval, with `n` samples, `c` successful samples and requested `k`:

```text
pass@k = 1 - choose(n - c, k) / choose(n, k)
```

It is `null` when fewer than `k` samples were produced. To report pass@5, request
at least five samples and a positive temperature. Invalid answers and timeouts
remain unsuccessful samples in the denominator. Input evaluation accepts any
valid call yielding the expected output; output evaluation parses literals.

## Renaming curves

For the [variable renaming experiment](variable-renaming.md), each rename level
N = 0, 1, 2, 4, 8 has its own accuracy. Retention asks whether items that were
correct before renaming remain correct after renaming the same function.

```text
accuracy(N) = correct items at N / scored items at N
retention(N) = items correct at both 0 and N / items correct at 0
```

Rates are stored as `{k, n, rate}`. If no item was correct at N = 0, retention is
`null`; the misuse-detection scorer also returns `null` below 30 such items.
Misuse localization and ClozeTest count ties as wrong. Non-finite diagnostic
values are exported as `null` and their field paths are recorded.

The mechanistic analyses report first-token accuracy on half B of the evaluation
functions, copy scores as attention mass, and forced-attention repairs as a rate
among the failures (correct in the original, wrong after renaming). Random-head
controls are summarised by their mean and standard deviation over draws; `report`
gives the sample standard deviation over training seeds separately.

The implementations are in [standard metrics](../benchmark_suite/metrics.py),
[renaming summaries](../experiments/robustness_to_variable_renaming/score.py) and
the experiment's [scoring library](../experiments/robustness_to_variable_renaming/lib/).
