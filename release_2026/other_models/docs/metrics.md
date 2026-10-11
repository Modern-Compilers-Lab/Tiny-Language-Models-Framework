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

For the [variable renaming experiment](../experiments/robustness_to_variable_renaming), each rename level has its own accuracy. Retention asks whether targets that were
correct before renaming remain correct after renaming, using shared target IDs.

```text
accuracy(N) = correct targets at N / scored targets at N
retention(N) = still-correct baseline targets / baseline-correct paired targets
```

For the seven non-paper task adapters, curve fields include `count`, `correct`,
`paired_count`, `retained` and `baseline_correct_paired`. The paper adapters keep
their original scoring protocol and raw results. The experiment's multi-token
Cloze adapter treats a tied gold score as incorrect; this differs from the
standard route's first-candidate tie rule.

If no paired baseline target was correct, retention is `null`, including at N=0.
Changed BPE lengths can change how many complete targets fit inside the context,
so accuracy counts need not be identical across levels. Requested rename caps
and achieved counts are separate; view manifests record the latter.

CuBERT detection thresholds are fitted on DEV and held fixed across the renamed
TEST views. TEST labels do not select a threshold. The DEV sample must contain
both classes. Paper diagnostic values that are nonfinite are exported as `null`
with their field paths recorded.

The implementations are in [standard metrics](../benchmark_suite/metrics.py),
[renaming curves](../experiments/robustness_to_variable_renaming/curves.py) and
[experiment task scoring](../experiments/robustness_to_variable_renaming/tasks/scoring.py).
