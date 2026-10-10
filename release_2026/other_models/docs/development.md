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
      configs/                    small and large experiment recipes
      data_transformations/       training interventions
      tasks/                      experiment prediction and paper adapters
      _vendor/                    original scoring library
      tests/                      renaming and shared-cache checks
      runs/                       local experiment artifacts, ignored by Git
  tests/                          shared model, data and benchmark tests
  docs/                           user and maintainer guides
  runs/                           shared local artifacts, ignored by Git
```

The module names inside `benchmark_suite` separate source inventories, streamed
data readers, prediction workflows, metric calculations and CRUXEval execution.
The renaming folder owns its registry, curve scoring, reference preparation,
campaign runner, token cache and reporting helpers.

## Tests

Install `pytest` alongside the runtime dependencies, then run both directories:

```powershell
python -m pytest release_2026/other_models/tests release_2026/other_models/experiments/robustness_to_variable_renaming/tests -q
```

Tests use local fixtures and tiny checkpoints. They cover architecture counts,
complete versus sampled sources, split isolation, real optimizer updates,
supervised loss masking, token alignment, metric arithmetic, renaming semantics,
cache integrity and reuse with network calls prohibited.

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

Create `experiments/<name>/` with configs, orchestration, task-specific code and
tests. Add an action through `experiments/workflow.py` only when it has a public
CLI. Reuse the shared trainer, model adapter and benchmark sources.

The renaming registry has explicit `TASKS` and `TRANSFORMS` entries loaded by its
`builtins()` function. Its `MODELS`/`DATASETS` maps and `discover()` loader are
retained library helpers, not a general CLI plugin system. The CLI has no
`--plugins` option. A new task needs an input/view builder as well as a predictor;
the current curve preparation assumes the existing task schemas.

The vendor scorer is kept byte-identical to codelab. Put compatibility changes
and model adapters outside `_vendor`. Windows locking, separate paper workers
and diagnostic cleanup already have wrappers for that purpose. Its README and
`SHA256SUMS` describe the retained files.
