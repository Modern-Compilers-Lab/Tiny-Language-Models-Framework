# Codelab experiments

Codelab currently contains one experiment family, `robustness_to_variable_renaming`.
It is integrated here, together with its reusable data preparation, token cache,
diversity measurement, campaign, freezing and reporting utilities. It works with
Llama, StarCoder (`gpt_bigcode`) and GPT-NeoX checkpoints, at either configured size.
The transformations, task adapters, registry, vendor code, tests and this guide are
contained in `experiments/robustness_to_variable_renaming/`. Shared models, training,
runtime and benchmark services remain in their existing framework modules.
No import from a sibling codelab checkout or dependency on ORX is required.

Install this folder's `requirements.txt` in your chosen Python environment. Run
commands from the repository root using `python release_2026/other_models/run.py`.
The absolute path to `run.py` also works from another directory. Each experiment
command accepts `--help`.

## What is included

Training conditions preserve document order, IDs, metadata and non-code fields:

| Condition | Intervention |
| --- | --- |
| `plain` | Original code |
| `halfvar` | Rename about half the eligible local bindings |
| `highdiv` | Rename all eligible local bindings using fresh corpus-wide names |

Training replacements touch identifier spans; comments, literals, attributes and
imports remain unchanged. Unsafe or unparseable documents are recorded by the
original treatment code. Manifests record changed documents, renamed bindings,
source/output hashes and identifier diversity. The `diversity` command checks
whether the treatment reaches the actual truncated training tokens.

Evaluation uses paired views at requested rename caps **0, 1, 2, 4 and 8**.
The achieved number of renamed names can be smaller; view manifests record it.
Code views use the original canonical representation and alpha-equivalence checks.
This evaluation preprocessing is separate from the project's formatter, which
these commands never invoke.

The ten original tasks are:

- `identifier_reproduction`, `variable_misuse_localization`, `cloze_all_paper`
- `py150_token`, `cloze_maxmin`, `cloze_all_multitoken`
- `cubert_exception`, `cubert_variable_misuse`, `cubert_swapped_operands`,
  `cubert_wrong_binary_operator`

The three paper tasks run in fresh processes with the original scorer. The other
seven save raw predictions before pure paired scoring. CuBERT detection thresholds
use DEV labels only, with one fixed threshold across all rename levels. A sampled
DEV set must contain both classes; increase the sample if calibration rejects it.
Accuracy counts may differ across views because changed BPE lengths alter which
complete lexical targets fit in the context. Retention uses the shared target IDs,
and is `null` when no baseline-correct paired targets exist.

The existing seven `benchmark` commands remain available. They evaluate their
standard protocols; renaming views belong to this separate ten-task experiment.

## Prepare and train controlled conditions

```powershell
python release_2026/other_models/run.py experiment list
python release_2026/other_models/run.py experiment prepare-data --arch llama --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/training-data
python release_2026/other_models/run.py experiment prepare-training --source release_2026/other_models/experiments/robustness_to_variable_renaming/runs/training-data/data/codesearchnet.jsonl --tokenizer bigcode/tiny_starcoder_py --arch llama --run-root release_2026/other_models/experiments/robustness_to_variable_renaming/runs/training --token-cache
python release_2026/other_models/run.py experiment campaign --cells release_2026/other_models/experiments/robustness_to_variable_renaming/runs/training/training-cells.json
```

`prepare-data` downloads complete training sources and prepares all eligible
documents by default. Use `--n-docs` for a preparation cap, `--max-files` for a
download shard cap, or `--source-cache`/`--reference-cache` for local source files.
Dataset defaults follow the selected model; `--dataset` overrides them.

`prepare-training` defaults to all three conditions and seeds `2,3,4`, 20,000
updates, global batch 16 and context 320. Optional `--steps`, `--batch-size`,
`--micro-batch`, `--max-len`, `--warmup`, `--device`, `--scale`, `--modes`, `--seeds`
and `--workers` make laptop pilots practical. `--pretrained` starts every condition
from the same existing checkpoint; omit it for training from scratch. Every cell
saves its own **new** `model/` checkpoint and refuses to overwrite trained outputs.
Always choose a fresh run root when changing a recipe.

The local campaign runner executes cells sequentially, saving driver logs and a
`training-cells.status.json` record. It uses the framework's current single-device
trainer. Codelab's historical ORX ledger, scheduler submission, DDP/resume checks
and audits of archived paper runs are not prerequisites for this integration.

For one transformation without a campaign:

```powershell
python release_2026/other_models/run.py experiment transform --input training.jsonl --transform halfvar --seed 2 --run release_2026/other_models/runs/transformed
```

Train `data/transformed.jsonl` using the existing `train --data` command.

## Prepare, evaluate and score

```powershell
python release_2026/other_models/run.py experiment prepare --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/benchmarks
python release_2026/other_models/run.py experiment prepare-views --data release_2026/other_models/experiments/robustness_to_variable_renaming/runs/benchmarks/data --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/views
python release_2026/other_models/run.py experiment evaluate --model CHECKPOINT --data release_2026/other_models/experiments/robustness_to_variable_renaming/runs/benchmarks/data --views release_2026/other_models/experiments/robustness_to_variable_renaming/runs/views --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/evaluation --device cuda
python release_2026/other_models/run.py experiment score --data release_2026/other_models/experiments/robustness_to_variable_renaming/runs/benchmarks/data --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/evaluation
python release_2026/other_models/run.py experiment report --run release_2026/other_models/experiments/robustness_to_variable_renaming/runs/evaluation --out release_2026/other_models/experiments/robustness_to_variable_renaming/runs/report
```

`prepare` defaults to `--n 0`: complete reference data and every available benchmark
row/shard/page, with the original eligibility filters. Positive `--n` explicitly
caps prepared rows and uses bounded CuBERT downloads. `--cache` accepts a local
CodeSearchNet source with its complete Python test split; existing reference sources
are discovered automatically when no path is provided. `--corpus-only` prepares just
the CodeSearchNet reference for the identifier and localization tasks.
`--pinned-eval` explicitly opts into an archived reference corpus for historical
comparisons; it is not the default source.

Both view preparation and evaluation default to `--limit 0` (all prepared rows).
Use `--limit 2` for a laptop pilot. Use the same comma-separated `--tasks` selection
for `prepare-views`, `evaluate` and `score`. Evaluation accepts `--batch`,
`--max-len`, `--dtype` and `--device`. An existing checkpoint can be evaluated
directly; controlled-condition training is optional.

Primary outputs are `results/curves.json`, raw `*.curves.predictions.jsonl` files,
paper scorer JSON/logs, and `results/prediction_manifest.json`. Scoring validates
prediction hashes. Reporting recomputes curves from saved predictions, verifies
their hashes, and exports `report.json` and `curves.csv`. A root containing several
evaluation runs can be passed to `report`; it includes means and sample standard
deviations for defined metrics, with explicit run counts.

Freeze the evaluator and prepared inputs before a larger campaign:

```powershell
python release_2026/other_models/run.py experiment freeze --data BENCHMARK_DATA --out FROZEN_JSON
python release_2026/other_models/run.py experiment freeze --data BENCHMARK_DATA --out FROZEN_JSON --verify
```

The copied scoring library is byte-identical to codelab's `SHA256SUMS`; changes to
it are detected by the integrity test. Project-specific wrappers sit outside it.

## Shared benchmark data

CodeXGLUE preparation delegates to `benchmark_suite.workflow.ensure_data`, the
existing download, preparation and manifest validation services. The experiment
contains no separate CodeXGLUE download URLs or label-recovery pipeline. Its task
adapters retain the shared TEST IDs, gold labels and official candidate vocabulary.
It records the source manifest and TEST hashes in the experiment manifest.
Sampling follows the shared source-order prefix, replacing codelab's historical
strided Cloze sampling; the original renaming intervention and paper scorer remain.

Prepared data and complete raw downloads under `other_models/runs/` are discovered
and verified before any network transfer. `--benchmark-run` selects shared storage
(default `other_models/runs/benchmarks`). A full request never reuses a prepared
subset. Explicit inputs whose hashes changed are rejected. Use these repeatable
options when earlier benchmark runs use different storage directories:

```powershell
python release_2026/other_models/run.py experiment prepare --run EXPERIMENT_RUN --n 2 --tasks py150_token,cloze_maxmin --benchmark-data codexglue_py150=PREPARED_PY150 --benchmark-data codexglue_cloze_maxmin=PREPARED_CLOZE
python release_2026/other_models/run.py experiment prepare --run EXPERIMENT_RUN --benchmark-source codexglue_py150=RAW_PY150 --benchmark-source codexglue_cloze_maxmin=RAW_CLOZE --benchmark-source codexglue_cloze_all=RAW_CLOZE_ALL
```

`--benchmark-data` paths point to the directory with `manifest.json` and TEST JSONL.
`--benchmark-source` paths point to the raw directory created by `benchmark download`.
Raw reuse performs preparation locally, without downloading those sources again.
Missing sources go through the shared benchmark services, with full datasets by
default and explicit positive `--n` subsets. `--tasks` also controls preparation so
small pilots need not fetch unrelated CuBERT tasks or a reference corpus.

New shared Cloze preparations retain `pl_tokens` and `mask_index`, preserving string
literals and comments exactly. Older prepared rows are automatically annotated from
their existing raw source when its manifest is available; this reads the original
tokens locally without downloading or recovering labels again. Other old rows can
be reused when their tokens can be reconstructed unambiguously. Flattened comments
with no raw source lack the original token boundaries; such rows require a raw
benchmark source via `--benchmark-source` (use a fresh `--benchmark-run`). They are
rejected rather than silently renamed with guessed boundaries.

## Verified offline laptop rerun

`runs/laptop-shared-benchmarks-20261010-v2/` in this experiment folder contains the
rerun, source hashes, logs, new checkpoints and verified reports. It uses 16 genuine
CodeSearchNet training documents, two updates for each of three Llama conditions,
two cached PY150 examples, one compatible cached Cloze max/min example and two
CodeSearchNet reference functions. The halfvar checkpoint is evaluated on PY150,
Cloze max/min, identifier reproduction and variable-misuse localization at all five
rename levels. Preparation prohibits HTTP requests; workers use Hugging Face offline
mode. Existing benchmark files and the initial checkpoint remain unchanged.
This is a pipeline check, not evidence of a training-condition effect.

Run both test directories to check shared services and the relocated experiment:

```powershell
python -m pytest release_2026/other_models/tests release_2026/other_models/experiments/robustness_to_variable_renaming/tests -q
```
