# Standard benchmarks

Standard benchmark commands.
See [metric implementation](benchmark_suite/metrics.py) for the definitions behind every result field.

## Available routes

| Name                     | Task                                  | Metrics                                                             | Benchmark fine-tuning |
| ------------------------ | ------------------------------------- | ------------------------------------------------------------------- | --------------------- |
| `codexglue_py150`        | Python lexical token completion       | `token_accuracy`                                                    | Yes                   |
| `codexglue_cloze_all`    | Masked identifier, 930 candidates     | `accuracy`                                                          | No                    |
| `codexglue_cloze_maxmin` | Masked `max` or `min`                 | `accuracy`                                                          | No                    |
| `bifi`                   | Syntax repair                         | `repair_accuracy`, `by_error_type`                                  | Yes                   |
| `manytypes4py`           | Type ranking                          | exact match at 1/5/10, MRR at 10, parametric match at 10, `by_kind` | Yes                   |
| `cruxeval_input`         | Find inputs yielding a given output   | pass@1, pass@5                                                      | No                    |
| `cruxeval_output`        | Predict the output of a function call | pass@1, pass@5                                                      | No                    |

```powershell
python release_2026/other_models/run.py benchmark list
```

## Download and prepare full data

The default downloads complete raw releases and prepares all available splits.
Trainable benchmarks include TRAIN, DEV and TEST; evaluation-only routes include TEST.

```powershell
python release_2026/other_models/run.py benchmark prepare --benchmark codexglue_py150 --run release_2026/other_models/runs/py150-full
```

For a smaller run, supply a positive limit and an explicit transfer budget:

```powershell
python release_2026/other_models/run.py benchmark prepare --benchmark codexglue_py150 --limit 2 --include-train --max-download-mb 64 --run release_2026/other_models/runs/py150-pilot
```

| Preparation option                     | Effect                                          |
| -------------------------------------- | ----------------------------------------------- |
| No row limit, `--full`, or `--limit 0` | Complete releases and uncapped rows             |
| Positive `--limit N`                   | At most N rows per split; TEST only by default  |
| `--include-train`                      | Add TRAIN/DEV to a capped preparation           |
| `--test-only`                          | Prepare TEST only, even in full mode            |
| `--max-download-mb N`                  | Transfer cap per benchmark preparation          |
| `--max-download-mb 0`                  | Unlimited transfer                              |
| `--version complete`                   | ManyTypes4Py complete variant; default is clean |

A positive row cap selects a 64 MiB default transfer budget. Full preparation
has no byte cap unless you set one. BIFI shares its positive TEST row cap across
heldout chunks 3 and 4.

Raw-only downloads are available separately:

```powershell
python release_2026/other_models/run.py benchmark download --benchmark bifi --run release_2026/other_models/runs/bifi-raw
python release_2026/other_models/run.py benchmark prepare --benchmark bifi --source release_2026/other_models/runs/bifi-raw/sources/bifi --run release_2026/other_models/runs/bifi-full
```

See [release downloader](benchmark_suite/sources.py) for inventories, hashes and offline reuse.

## Local data

`--source` accepts:

- A verified complete raw source directory from `benchmark download`.
- A directory containing normalized `train.jsonl`, `dev.jsonl` and `test.jsonl`.
- The original BIFI ZIP or ManyTypes4Py tar.gz archive.

Evaluation-only routes need only `test.jsonl`. A capped trainable route also needs
TRAIN/DEV when `--include-train` is supplied.

```powershell
python release_2026/other_models/run.py benchmark prepare --benchmark manytypes4py --source D:/datasets/ManyTypes4PyDataset-v0.7.tar.gz --run release_2026/other_models/runs/types-full
```

ManyTypes4Py's gzip tar is sequential. Reaching even one processed project can
exceed the sampling byte cap. Use an existing archive or raise the cap explicitly.
The sampled path does not write a full archive; full downloads retain it.

All normalized rows have `id` and `split`, plus these fields:

| Benchmark    | Required task fields                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| PY150        | `code`: lexical token list                                                     |
| Cloze        | `prefix`, `gold`, `choices`; optional audit `suffix`                           |
| BIFI         | `buggy_tokens`; TRAIN/DEV also `fixed_tokens`; TEST also `chunk`, `error_type` |
| ManyTypes4Py | Untyped `code`, `scope`, `target`, `kind`, `gold`                              |
| CRUXEval     | `code`, argument-expression `input`, literal-expression `output`               |

Preparation checks labels, schemas, duplicate IDs and split overlap.
Fine-tuning checks TRAIN against heldout IDs and exact inputs again.
Official splits are retained; the framework does not randomly resplit benchmarks.

## Evaluate one checkpoint

Evaluation scores every prepared TEST example by default, including when
fine-tuning is enabled. Omit `--limit` or use `--limit 0` for complete evaluation.
A positive evaluation limit is useful for an optional pilot and does not change
the downloaded data.

```powershell
python release_2026/other_models/run.py benchmark evaluate --data release_2026/other_models/runs/py150-full/data/codexglue_py150 --model release_2026/other_models/runs/llama/model --batch 4 --max-len 1024 --run release_2026/other_models/runs/py150-direct
```

Candidate scoring uses `--batch`; `--max-len` controls context. For generative
tasks, the prompt and `--max-new-tokens` budget must fit the context. Oversized
prompts fail rather than silently deleting code. Use a longer-context checkpoint
or explicitly prepare suitable shorter inputs.

CRUXEval pass@5 needs at least five sampled answers:

```powershell
python release_2026/other_models/run.py benchmark prepare --benchmark cruxeval_output --run release_2026/other_models/runs/crux-data
python release_2026/other_models/run.py benchmark evaluate --data release_2026/other_models/runs/crux-data/data/cruxeval_output --model release_2026/other_models/runs/llama/model --samples 5 --temperature 0.2 --max-new-tokens 64 --run release_2026/other_models/runs/crux-test
```

## Optional fine-tuning

Use the supplied `configs/finetune-example.json`, or save a recipe such as `release_2026/other_models/configs/my-finetune.json`:

```json
{
  "steps": 20,
  "lr": 5e-05,
  "warmup": 0,
  "batch_size": 2,
  "micro_batch": 1,
  "max_len": 1024,
  "save_every": 10,
  "seed": 2
}
```

These budgets are examples, not tuned benchmark recipes.
To fine-tune and evaluate in one command:

```powershell
python release_2026/other_models/run.py benchmark evaluate --data release_2026/other_models/runs/py150-full/data/codexglue_py150 --model release_2026/other_models/runs/llama/model --finetune-config release_2026/other_models/configs/finetune-example.json --run release_2026/other_models/runs/py150-tuned
```

Or save a tuned checkpoint first and evaluate it separately:

```powershell
python release_2026/other_models/run.py benchmark finetune --data release_2026/other_models/runs/py150-full/data/codexglue_py150 --model release_2026/other_models/runs/llama/model --config release_2026/other_models/configs/finetune-example.json --run release_2026/other_models/runs/py150-ft
python release_2026/other_models/run.py benchmark evaluate --data release_2026/other_models/runs/py150-full/data/codexglue_py150 --model release_2026/other_models/runs/py150-ft/model --run release_2026/other_models/runs/py150-ft-test
```

Only TRAIN supplies updates. DEV and TEST do not select the training schedule.
The original checkpoint remains unchanged. Architecture and tokenizer come from
that checkpoint; the recipe cannot override `arch`, `model`, `data`, `pretrained`,
`tokenizer`, `benchmark`, or `supervised`.

BIFI and ManyTypes4Py mask prompt loss and train on complete answer targets plus
EOS. PY150 uses causal continuation training. Targets too long for the recipe's
context cause an error.

Cloze and CRUXEval reject benchmark fine-tuning because their releases contain
evaluation items only.

## Compare all routes on one model

Repeat preparation and evaluation for each name in the table. Use a separate
data and result run per route. Preparation downloads each full benchmark by
default.

This PowerShell example evaluates all prepared TEST examples (select a real model checkpoint first):

```powershell
$benchmarks = @('codexglue_py150', 'codexglue_cloze_all', 'codexglue_cloze_maxmin', 'bifi', 'manytypes4py', 'cruxeval_input', 'cruxeval_output')
foreach ($benchmark in $benchmarks) {
    python release_2026/other_models/run.py benchmark evaluate --data "release_2026/other_models/runs/bench-data/$benchmark/data/$benchmark" --model release_2026/other_models/runs/llama/model --batch 2 --max-new-tokens 64 --run "release_2026/other_models/runs/bench-results/$benchmark"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $benchmark" }
}
```

Prepare each directory first with
`--benchmark <name> --run release_2026/other_models/runs/bench-data/<name>` and no row cap.
Full ManyTypes4Py preparation includes TRAIN to build a vocabulary. If you prepare
TEST only, pass a frozen `--type-vocabulary`.
Add sampling flags when reporting CRUXEval pass@5.

## Protocol details

| Route        | Details that affect comparisons                                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| PY150        | Official 95k TRAIN, final 5k original train files as DEV, 50k TEST; complete lexical alignment and overlapping windows     |
| Cloze        | Causal left-context likelihood; recovered gold; official candidates; no bidirectional CodeBERT claim                       |
| BIFI         | Released token representation and original critic/edit rule; original round0 training pairs; no iterative breaker training |
| ManyTypes4Py | Released file-level splits; untyped code only; candidate vocabulary from TRAIN or a frozen external file                   |
| CRUXEval     | Compact causal prompts, literal predictions, any correct input accepted; invalid samples stay in the denominator           |

CodeXGLUE cloze answer files contain placeholders rather than usable gold labels.
Recovery must be unambiguous. BIFI restores anonymized literals for AST checks;
its bracket critic counts types independently. Error-group reports are coarse.

CRUXEval input checks run in time-limited worker processes with restricted
imports. They are reliability guards, not an OS security sandbox.
Output predictions are parsed as literals rather than executed.

External pretraining contamination is not established by benchmark split checks.
Keep the source, tokenizer, context, prompt and fine-tuning recipe with any result.

## Outputs

| Path                                              | Contents                                          |
| ------------------------------------------------- | ------------------------------------------------- |
| Preparation `data/<name>/manifest.json`           | Source, protocol, limits, split counts and hashes |
| Preparation `data/<name>/{train,dev,test}.jsonl`  | Available normalized splits                       |
| Evaluation `results/metrics.json`                 | Scores, denominators, checkpoint and data hashes  |
| Evaluation `results/<name>.predictions.jsonl`     | Item-level predictions and correctness            |
| Tuned evaluation `finetune/model/`                | Separate tuned checkpoint                         |
| Tuned evaluation `finetune/results/training.json` | Fine-tuning budget and loss                       |

## Sources and tests

- [CodeXGLUE token completion](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token)
- [ClozeTest-all](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-all) and [ClozeTest-maxmin](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin)
- [BIFI evaluator](https://github.com/michiyasunaga/BIFI/blob/main/src/c005__eval_fixer.py)
- [ManyTypes4Py release](https://zenodo.org/records/5244636) and [Type4Py evaluator](https://github.com/saltudelft/type4py/blob/main/type4py/eval.py)
- [CRUXEval evaluator](https://github.com/facebookresearch/cruxeval/tree/main/evaluation)

```powershell
python -m pytest release_2026/other_models/tests -q
```

These tests use local fixtures and tiny checkpoints. They do not download full releases.

## Integration behavior

Run commands from the framework repository root. All integration code and the
examples' generated outputs live in `release_2026/other_models/`; codelab is not a
runtime dependency. Install this directory's `requirements.txt`, including `ijson`.

Prepare or download every registered route with `--benchmark all`:

```powershell
python release_2026/other_models/run.py benchmark download --benchmark all --run release_2026/other_models/runs/bench-raw
python release_2026/other_models/run.py benchmark prepare --benchmark all --run release_2026/other_models/runs/bench-data
```

Each route keeps its own `sources/<name>/` and `data/<name>/` directories.
The two CRUXEval routes intentionally read the same release with different tasks.
Use separate raw/preparation/evaluation run directories because their recorded
recipes differ. `--source` accepts one benchmark at a time.

Raw-only `download` retains complete release files; `--max-download-mb` is an
optional hard transfer budget. To download and prepare a prefix without fetching
complete releases, use `prepare --limit N`. The subset readers use HTTP ranges
or stop streaming once the requested rows are collected. Some formats still
need supporting files or substantial archive prefixes before the requested
rows are reachable. A byte cap fails explicitly if that prefix is too large.

Every default preparation includes all official splits available for that route.
ManyTypes4Py downloads the entire release archive and prepares all eligible clean
variant rows by default; `--version complete` selects that variant instead.

Fine-tuning always creates a distinct final checkpoint. Standalone `finetune`
saves `<run>/model/`; `evaluate --finetune-config` saves `<run>/finetune/model/`
and evaluates that checkpoint. Writing inside the source checkpoint, reusing
its output location, or replacing an existing final tuned checkpoint is rejected.
The tokenizer is saved with each new checkpoint. TRAIN/heldout overlap checks,
training, and prediction output use disk or bounded buffers for full corpora.
Source-checkpoint hashes are recorded with fine-tuning metadata.

All three integrated architectures can use the trainable routes. Zero-shot
benchmark evaluation performs no gradient updates and creates no tuned model.
Cloze and CRUXEval support evaluation only. Exact optimizer-state resume is not
implemented; continued training from an existing checkpoint starts a new optimizer.

The supplied fine-tuning recipe is an example, not a calibrated benchmark budget.
Use `--max-len` and `--max-new-tokens` appropriate to your checkpoint and prompts.
CRUXEval input workers retain the upstream timeout/import restrictions; they do
not provide an operating-system sandbox for arbitrary code.
