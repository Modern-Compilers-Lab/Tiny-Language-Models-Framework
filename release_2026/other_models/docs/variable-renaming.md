# Variable-renaming experiment

Decoder-only code models often fail when the local variables of a program are
renamed, even though the program means the same thing. This experiment measures
that brittleness on the framework's models and looks for its mechanism inside the
model: which attention heads copy an earlier variable name, what happens when
they are removed, and whether redirecting them repairs a failure. It reproduces
the protocols of *Understanding Brittleness to Variable Renaming in Decoder-Only
Code Models* (Sections 4.5, Appendices D.5 and E.2) on any checkpoint trained by
`train`.

The implementation is in
[`experiments/robustness_to_variable_renaming/`](../experiments/robustness_to_variable_renaming/).
Every action runs from the repository root through `run.py experiment`. See
[setup](environment.md) for the Python environment and the
[command reference](commands.md#experiment-commands) for the action list.

## Models

The experiment accepts the three model families of
[models and training](models-training.md): Small StarCoder (`gpt_bigcode`,
multi-query attention), Small Llama (`llama`, also with grouped-query attention)
and Small GPT-NeoX (`gpt_neox`). `--model` takes either a training run directory
(the experiment reads its `model/` folder) or a Hugging Face model directory.
Checkpoints load through `models.HFModel` in float32 with eager attention, because
the analyses read attention weights. The tokenizer must be a fast tokenizer with
character offsets, as the framework's default `bigcode/tiny_starcoder_py` is.

Head counts are read from each model config. With the small presets (20 layers,
3 heads) a model has 60 heads, enough for the paper's budgets of up to 15 ablated
heads and 12 forced heads, each with a disjoint random control set.

## Folder layout

| File | Contents |
| --- | --- |
| `prepare_corpus.py` | Builds the evaluation corpus and the ClozeTest cache |
| `transform.py` | Creates the `plain`, `halfvar` and `highdiv` training sets |
| `exp_*.py` | One entry point per paper result (table below) |
| `run.py` | The four mechanistic protocols in one process; `--suite paper` runs every entry point |
| `copyscore.py`, `meanablate.py`, `force.py`, `firsterror.py` | Mechanistic scoring functions |
| `tasks.py`, `pilot.py` | Behavioural tasks and the D.5 pilot measures |
| `adapter.py` | Checkpoint loading and per-head access for the three architectures |
| `env.py` | Process checks and access to `lib/` |
| `score.py`, `report.py` | Per-run summary metrics and the multi-seed report |
| `toy.py`, `tests/`, `fixtures/` | Random-model end-to-end check, unit tests and a 40-function corpus |
| `lib/` | The original scoring library that produced the paper's numbers |
| `paper.json` | The paper's budgets, usable with `--config` |

## 1. Prepare the evaluation corpus

All experiments read one folder passed as `--corpus`:

```powershell
python release_2026/other_models/run.py experiment prepare --out CORPUS --n 2000 --cloze 3000
```

`real6_corpus.json` holds CodeSearchNet Python TEST functions that pass the
paper's filters: 120 to 2,000 characters, parseable, at least three local
variables and one integer literal, names of at most 20 characters. They are kept
in source order and the first `--n` are written. The source is the first complete
local CodeSearchNet download found under `other_models/runs/` with a
`python/test-*.parquet` file, or `--source DIR`; otherwise only the test split is
downloaded. CodeSearchNet TEST functions are already excluded from the training
corpora (see [datasets](datasets.md)).

`--cloze N` also writes `cloze_all_items_n<N>.json`, needed by the bench group of
the tasks. It is described under [shared benchmark data](#shared-benchmark-data).

Pass the experiments an `--n` no larger than the corpus. The library serves the
first `--n` functions of a larger corpus and stops with "cached real6 corpus does
not match the design" when the corpus is too small. `fixtures/real6_corpus.json`
has 40 functions and is meant for tests (`--n 40`).

## 2. Training conditions

The paper compares models trained on the same documents with different variable
names. `transform` applies the paper's treatment (`lib/corpus.py`) to a prepared
training JSONL:

| Mode | Treatment |
| --- | --- |
| `plain` | Original code |
| `halfvar` | About half of the eligible local bindings of each document get fresh names |
| `highdiv` | Every eligible local binding gets a fresh name |

```powershell
python release_2026/other_models/run.py experiment transform --input release_2026/other_models/runs/datasets/data/codesearchnet.jsonl --output release_2026/other_models/runs/halfvar-s2/train.jsonl --mode halfvar --seed 2
python release_2026/other_models/run.py train --arch llama --data release_2026/other_models/runs/halfvar-s2/train.jsonl --seed 2 --run release_2026/other_models/runs/halfvar-s2/model-run
```

Fresh names are unique across the corpus, so the treatment raises name diversity.
Only identifier spans change; comments, strings, attributes and imports are kept,
as are row order, IDs and other fields. The manifest written next to the output
records changed rows, renamed bindings, hashes and the number of distinct variable
names before and after (over the first 2,000 rows). Train every condition with the
same recipe and seeds, and label each analysis with `--arm` and `--training-seed`.

## 3. Behavioural robustness

| Paper item | Action | Output in `RUN/results/` |
| --- | --- | --- |
| Figure 6, Table 4 (identifier reproduction) | `tasks --group idrep` | `tasks_idrep.json` |
| Figure 6, Table 4 (misuse localization, ClozeTest) | `tasks --group bench` | `tasks_bench.json` |
| Appendix D.5 non-identifier accuracy | `nonidentifier` | `pilot_comp.json` |
| Appendix D.5 misuse detection | `misuse-detection` | `pilot_vm.json` |

```powershell
python release_2026/other_models/run.py experiment tasks --group idrep --model CHECKPOINT --corpus CORPUS --run RUN_IDREP --device cuda
python release_2026/other_models/run.py experiment tasks --group bench --model CHECKPOINT --corpus CORPUS --run RUN_BENCH --device cuda --n-sample 3000
```

Each test function is scored as written (N = 0) and after renaming N = 1, 2, 4
and 8 of its variables. Every renamed variable gets a random lowercase name of
the same length that does not occur in the function, with a seed fixed per
function and N, so all models see identical views. Every renaming is verified by
applying its inverse.

- **Identifier reproduction.** Teacher-forced. At every repeated use of a variable,
  the model's most likely next tokens must reproduce every token of the name.
- **Variable-misuse localization** (after CuBERT). One use of a variable is
  replaced by another variable in scope. Every candidate edit is scored by the
  length-normalised log-likelihood of the whole edited function; the best edit must
  be the modified position. Ties count as wrong. At most 150 candidates per item
  and `--n-items` items (1,600 in the paper).
- **ClozeTest-all.** Only items whose answer is not a renamable variable are kept.
  The model ranks the single-token candidates given the code before the mask.
- **D.5 non-identifier accuracy.** Teacher-forced next-token accuracy at tokens
  that do not overlap a variable name, before and after renaming every variable.
- **D.5 misuse detection.** A two-way choice between the correct function and a
  copy with one misused variable (chance 0.5). It is a different task from the
  localization above, so the two accuracies are not comparable.

Two numbers are kept per task and rename level: **accuracy** over all items, and
**retention**, the fraction of the items correct at N = 0 that are still correct
at N, stored as `{k, n, rate}`. Undefined values are `null`, for example a
retention with no correct baseline item. The misuse-detection scorer also reports
`null` retention when fewer than 30 items were correct at N = 0.

## 4. Mechanistic analyses

These analyses share one item definition. An **item** is a use of a local
variable whose name already appeared earlier in the same function. Each item is
scored twice with teacher forcing: in the original function (ID) and after every
local variable is renamed to a random same-length name (OOD). The **decision** is
the most likely token at the position just before the name, compared with the
name's first token. Functions are split at random into halves A and B: heads are
ranked on half A of the renamed functions and every intervention is measured on
half B, so selection and measurement never share data.

| Paper item | Action | Output in `RUN/results/` |
| --- | --- | --- |
| Figure 7A copy score | `copy-score` | `copyscore.json` |
| Section 4.5 zero ablation | `zero-ablation --kmax 15` | `copyscore.json` with the ablation curve |
| Figure 7B mean ablation | `mean-ablation --kmax 15` | `meanablate.json` |
| Table 6, Appendix E.2 forced attention | `forced-attention --ks 3,12` | `force.json` |
| Appendix E.2 paired first error | `first-error` | `firsterror.json` |

```powershell
python release_2026/other_models/run.py experiment copy-score --model CHECKPOINT --corpus CORPUS --run RUN_COPY --device cuda --arm halfvar --training-seed 2
```

- **Copy score.** For each head, the attention mass that the decision position
  puts on the first tokens of earlier occurrences of the same variable, averaged
  over items. Figure 7A reports the largest half-B score among the five heads
  ranked highest on half A. This maximum is biased upwards by selection.
- **Zero ablation.** The k = 1..kmax highest-ranked heads are removed by zeroing
  their slice of the attention output projection input, against 10 random sets of
  k other heads (50 at k = 3). Reported: first-token accuracy on half B.
- **Mean ablation.** As above, but each removed head's output is replaced by its
  mean over the half-A tokens of the same view. This removes what the head reads
  from the context while keeping its average output, so the rest of the model is
  not pushed off its usual input range.
- **Forced attention.** On failures (correct in the original, wrong after
  renaming), the attention row of the k top heads at the decision is replaced by
  equal weights on the earlier occurrences of the same variable, and the head
  outputs are recomputed from the model's own value vectors. A **repair** is a
  failure that becomes correct. Controls: k random heads from the same layers with
  the same targets, and the same heads pointed at the other variables of the
  function. A self-check confirms that giving every head its own attention row
  reproduces the model (difference at most 5e-5, no prediction changed).
- **Paired first error.** For names correct in the original and wrong after
  renaming, the position of the first wrong token in the renamed name: its first
  token, a later token, or the end (the next prediction would continue the
  identifier). Reported overall and by the number of tokens of the name.

`--layers` and `--heads` restrict the heads that can be ranked, which changes the
protocol. A budget that leaves too few heads for disjoint random controls fails
with an explicit message. `mechanistic` runs all four protocols in one process and
also writes `metrics.json`, recomputed from the result files; `--resume` skips
completed protocols after checking that the configuration and hashes match.

## 5. Full suite and reports

`suite` runs all nine entry points, each in its own process and subdirectory
(`fig7a`, `zero`, `mean`, `force`, `firsterror`, `idrep`, `bench`, `comp`, `vm`):

```powershell
python release_2026/other_models/run.py experiment suite --model CHECKPOINT --corpus CORPUS --run RUN_ROOT --device cuda --config release_2026/other_models/experiments/robustness_to_variable_renaming/paper.json
python release_2026/other_models/run.py experiment mechanistic --model CHECKPOINT --corpus CORPUS --run RUN_PLAIN_S2 --device cuda --arm plain --training-seed 2
python release_2026/other_models/run.py experiment report --runs RUN_PLAIN_S2 RUN_PLAIN_S3 RUN_HALFVAR_S2 RUN_HALFVAR_S3 --out REPORT
```

`paper.json` holds the paper's budgets: 2,000 functions, context 320, k = 1..15,
10 random controls (50 at k = 3) and forced attention at k = 3 and 12. Command-line
options override it. `report` takes `mechanistic` runs (all four protocols in one
directory, one run per trained model), groups them by `--arm` and writes
`summary.json`, `summary.csv` and `curves.csv` with the mean and sample standard
deviation over training seeds. It rejects two runs with the same arm and seed.

Every run writes `config.json` with the options and the SHA-256 of the corpus,
checkpoint files and experiment sources, `logs/run.log` and its result JSON. The
mechanistic runs also write `environment.json`.

## 6. Small-scale check

Before a long run, check the installation and the code paths:

```powershell
$env:PYTHONHASHSEED = '0'
python -m pytest release_2026/other_models/experiments/robustness_to_variable_renaming/tests -q
python release_2026/other_models/run.py experiment toy --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/toy
```

The tests check, for the five attention layouts (GPT-BigCode multi-query and
multi-head, Llama multi-head and grouped-query, GPT-NeoX), that per-head outputs
rebuilt from the attention weights and value vectors equal the model's own. They
also cover head selection, rates, the library checksums, and `--help` for every
entry point. `toy` runs the four mechanistic protocols on random two-layer models
of each family. Its numbers only test the code.

A reduced suite on a trained checkpoint takes about ten minutes on a CPU:

```powershell
python release_2026/other_models/run.py experiment prepare --out release_2026/other_models/experiments/robustness_to_variable_renaming/runs/smoke/corpus --n 60 --cloze 40
python release_2026/other_models/run.py experiment suite --model release_2026/other_models/runs/llama --corpus release_2026/other_models/experiments/robustness_to_variable_renaming/runs/smoke/corpus --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/smoke/llama --n 60 --limit 30 --kmax 3 --ks 3 --n-random 2 --n-random-k3 2 --n-items 20 --n-sample 40 --batch 8
```

With so few items and a briefly trained model, most rates are close to zero or
undefined; the run only shows that every protocol completes.

## Scoring library

`lib/` contains the evaluation code used for the paper, byte for byte;
`lib/SHA256SUMS` lists it and the tests check it (line endings are normalised, so
a Windows checkout passes). Its modules import each other by bare name, so `env.py`
adds `lib/` to `sys.path`, and on Windows it provides the `fcntl` locking the
library imports. Wrappers outside `lib/` only replace the library's model loader
with the framework's and remove its progress prints.

| File | Role |
| --- | --- |
| `decmech.py` | Evaluation items (`real6_occs`, `token_layout`), identifier reproduction, misuse localization, ClozeTest and non-identifier accuracy |
| `whyscale.py` | The corpus filters and the renamed views |
| `varmisuse.py` | Misuse items and the two-way detection test |
| `corpus.py` | The training treatments (`FreshNames`, `rename_document_halfvar`) |
| `poolfix.py`, `perturb/` | Helpers and the tree-sitter variable extractor |
| `prep.py`, `train.py` | The original data preparation and trainer, kept for reference; the framework's trainer is used instead |

Two library properties shape the commands:

- Renamed views are seeded with string hashes, so Python must start with
  `PYTHONHASHSEED=0`. `run.py experiment` sets it for every child process; set it
  yourself when calling a script directly.
- Importing `varmisuse.py` sets `WHYSCALE_BLOCK=319` for the whole process. That
  is why identifier reproduction, the bench group and the two D.5 measures run in
  separate processes, as they did for the paper.

`misuse-detection` builds its items from the first 2,000 corpus functions. With a
smaller corpus it uses all of them; otherwise the library would download
CodeSearchNet again and overwrite the prepared corpus.

## Shared benchmark data

The ClozeTest cache reuses the framework's CodeXGLUE preparation
(`benchmark_suite.workflow.ensure_data` for `codexglue_cloze_all`), including its
gold labels recovered from CodeSearchNet, and applies the library's renaming rule
and seeds to its TEST rows. Downloads go to `--benchmark-run` (default
`other_models/runs/benchmarks`); `--benchmark-data` selects an existing prepared
folder. Rows must keep their token boundaries (`pl_tokens`), which current
preparations do; an older preparation without them needs its raw source or a new
`--benchmark-run`.

The paper sampled 3,000 ClozeTest rows with a stride over the whole set. The
shared preparation takes the first `--cloze` rows instead, so the items differ
from the paper's; the scoring is the same.
