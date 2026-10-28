# Development

The integration keeps its source under `release_2026/other_models`. Shared model,
data and benchmark services are separate from the variable-renaming experiment.

```text
other_models/
  run.py, __main__.py, cli.py       command entry points
  models.py                       architecture presets and HFModel adapter
  datasets.py                     training corpus readers and filters
  source_downloads.py             downloads and source manifest checks
  training.py                     streaming causal/supervised trainer
  runtime.py                      run snapshots, JSONL I/O and hashes
  configs/                        small recipes and fine-tuning example
  benchmark_suite/                standard readers, protocols, metrics and workers
  experiments/
    workflow.py                   experiment command dispatch
    robustness_to_variable_renaming/
      exp_*.py                    one entry point per paper result
      run.py, tasks.py, pilot.py  mechanistic, behavioural and pilot drivers
      adapter.py, env.py          model access and library setup
      lib/                        original scoring library
      fixtures/, tests/           test corpus and checks
      runs/                       local experiment artifacts, ignored by Git
  tests/                          shared model, data and benchmark tests
  docs/                           user and maintainer guides
  runs/                           shared local artifacts, ignored by Git
```

The module names inside `benchmark_suite` separate source inventories, streamed
data readers, prediction workflows, metric calculations and CRUXEval execution.
The renaming folder owns its corpus preparation, training treatment, analyses and
reporting; it loads models through `models.HFModel`.

## Tests

Install `pytest` alongside the runtime dependencies, then run both directories.
The experiment tests need `PYTHONHASHSEED=0`:

```powershell
$env:PYTHONHASHSEED = '0'
python -m pytest release_2026/other_models/tests release_2026/other_models/experiments/robustness_to_variable_renaming/tests -q
```

Tests use local fixtures and tiny checkpoints. They cover architecture counts,
complete versus sampled sources, split isolation, real optimizer updates,
supervised loss masking, token alignment, metric arithmetic, per-head attention
reconstruction, the scoring-library checksums and reuse with network calls
prohibited.

Formatting is a separate maintenance step. The examples in these docs do not
invoke the project's formatter.

## Add a training corpus or model

For a corpus, add the source inventory to `source_downloads.py`, a reader to
`datasets.source_rows`, and the key to `models.resolve_dataset`. Check the CLI
choices and source validation too. Keep repository/function exclusions and the
JSONL schema unless the new corpus needs a documented alternative.

For a model, add its factory in `models.build_config`, its metadata in
`MODEL_DEFAULTS`, and a recipe under `configs`. Update the CLI recipe filename
mapping. The trainer expects a Hugging Face causal model with a matching tokenizer
and supported context fields. Test save/load and an optimizer update, not only
configuration construction.

## Add a standard benchmark

Add its protocol metadata in `benchmark_suite.__init__`, then update source
inventories, readers, row validation, prediction and metric handling. Registering
a name alone does not supply those implementations. Declare whether an official
training split exists and how DEV/TEST are used.

Keep gold recovery outside prediction. Test the metric on hand-calculated cases,
verify split and label checks, and check both full and explicit subset preparation.
If the data is also used by an experiment, reuse `workflow.ensure_data` and the
prepared manifest rather than adding another downloader.

## Add an experiment

Create `experiments/<name>/` with its scripts and tests. Add an action through
`experiments/workflow.py` only when it has a public CLI. Reuse the shared trainer,
model adapter and benchmark sources.
