# Quick start

This example trains a small Llama model and evaluates a few PY150 files. The
budgets are for checking the workflow.

Install the [environment](environment.md) first. Then, from the repository root:

```powershell
$cli = 'release_2026/other_models/run.py'
$pilot = 'release_2026/other_models/runs/quickstart'
python $cli list
```

Use a new `$pilot` path if these outputs already contain a trained model.

## Prepare training data

```powershell
python $cli prepare --arch llama --n-docs 32 --max-files 1 --run "$pilot/data"
```

This downloads one Python training shard and keeps the first 32 eligible
documents. Preparation also needs the complete CodeSearchNet Python test split
for exclusion checks; a training-shard cap does not remove that reference
download. Existing sources can be supplied with `--cache` and `--reference-cache`.

The training file is `$pilot/data/data/codesearchnet.jsonl`. Its sidecar manifest
records the source revision, row count and exclusions.

For a complete preparation, omit both limits:

```powershell
python $cli prepare --arch llama --run release_2026/other_models/runs/csn-full
```

## Train and load the checkpoint

```powershell
python $cli train --arch llama --data "$pilot/data/data/codesearchnet.jsonl" --steps 20 --warmup 0 --batch-size 2 --micro-batch 1 --max-len 128 --device auto --run "$pilot/training"
python $cli generate --model "$pilot/training/model" --prompt 'def add(a, b):' --max-new-tokens 24 --device auto
```

The final model and tokenizer are saved together in `$pilot/training/model`.
The loss and update count are in `$pilot/training/results/training.json`.
Initial training obtains the tokenizer from `bigcode/tiny_starcoder_py`; it
builds the small model with random weights.

## Evaluate a small benchmark sample

The example uses `--device auto`, the default, so it runs on CUDA when available
and falls back to CPU.

```powershell
python $cli benchmark prepare --benchmark codexglue_py150 --limit 2 --run "$pilot/py150-data"
python $cli benchmark evaluate --data "$pilot/py150-data/data/codexglue_py150" --model "$pilot/training/model" --limit 2 --device auto --max-len 128 --batch 2 --run "$pilot/py150-evaluation"
```

Read `results/metrics.json` in the evaluation run. Its `count` is the number of
scored lexical tokens; `evaluated_examples` is the number of input files.
Predictions are saved separately in `results/codexglue_py150.predictions.jsonl`.

StarCoder and GPT-NeoX use the same commands with a different `--arch`. See
[models and training](models-training.md) for their defaults,
[benchmarks](benchmarks.md) for the other tasks, and
[variable renaming](variable-renaming.md) for controlled training conditions.
