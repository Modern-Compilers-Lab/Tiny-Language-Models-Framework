# Other models

This part of the framework trains small code models and evaluates saved
checkpoints.

| Guide | Contents |
| --- | --- |
| [Quick start](quickstart.md) | Prepare a small corpus, train Llama and evaluate PY150 |
| [Environment](environment.md) | Installation, entry points, CUDA and offline use |
| [Datasets](datasets.md) | Complete downloads, subsets, local sources and leakage filters |
| [Models and training](models-training.md) | Architecture presets, JSON recipes, loading and continuation |
| [Benchmarks](benchmarks.md) | The seven benchmarks, data preparation and optional fine-tuning |
| [Metrics](metrics.md) | Score definitions, denominators and undefined results |
| [Variable renaming](variable-renaming.md) | The variable renaming tasks and evaluation methods |
| [Commands](commands.md) | CLI actions and where to find their options |
| [Runs and reproducibility](reproducibility.md) | Output files, hashes, run reuse and recorded results |
| [Development](development.md) | Source layout, tests and extension points |

Examples run from the framework repository root with the selected environment's
`python`. Paths in command examples are repository-relative. Replace uppercase
placeholders such as `CHECKPOINT` with a real path before running them.

Training corpora and benchmarks are complete by default. A positive document or
row limit is an explicit sample. Evaluation also scores every prepared example
unless given a positive limit.
