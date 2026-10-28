# Commands

From the repository root, use:

```powershell
python release_2026/other_models/run.py COMMAND [options]
```

Append `--help` to an action for its current parser options. The tables below
describe the workflows; the linked guides cover recipes and output files.

## Training and inference

| Action | Required selection | Main options |
| --- | --- | --- |
| `list` | None | Lists model defaults, dataset repositories and tokenizer |
| `download` | `--dataset` or `--arch` | `--run`, `--revision`, `--dry-run`, `--max-files`, repeatable `--allow-pattern` |
| `prepare` | `--dataset` or `--arch` | `--run`, `--cache`, `--reference-cache`, `--n-docs`, `--max-files`, repeatable `--exclude-repo` |
| `train` | Architecture through `--arch` or `--config` | `--data`, `--data-run`, `--dataset`, `--tokenizer`, `--pretrained`, training budgets |
| `generate` | `--model`, `--prompt` | `--max-new-tokens`, `--device`, `--local-files-only` |

`download` and `prepare` default to `other_models/runs/datasets`.
`train` defaults to `other_models/runs/<arch>`. Explicit paths make separate
preparation and training runs easier to track. `--dataset all` selects every
training corpus for downloading/preparation.

See [datasets](datasets.md) and [models and training](models-training.md).

## Benchmark commands

| Action | Required arguments | Main options |
| --- | --- | --- |
| `benchmark list` | None | Lists protocols and fine-tuning availability |
| `benchmark download` | `--benchmark`, `--run` | `--max-download-mb`; `--benchmark all` is supported |
| `benchmark prepare` | `--benchmark`, `--run` | `--source`, `--limit` or `--full`, `--include-train` or `--test-only`, `--version`, `--max-download-mb` |
| `benchmark finetune` | `--data`, `--model`, `--config`, `--run` | Writes a new model checkpoint |
| `benchmark evaluate` | `--data`, `--model`, `--run` | `--limit`, `--device`, `--dtype`, `--batch`, `--max-len`, `--max-new-tokens`, `--finetune-config` |

`benchmark prepare --benchmark all` supports automatic downloads; `--source`
requires one benchmark. In `evaluate`, `--data` is the prepared benchmark
directory with `manifest.json`, rather than the parent run.

`benchmark evaluate` defaults to `--device auto`, float32, batch 4 and context 1,024.
It selects CUDA when available and otherwise CPU. Use `--device cpu` or
`--device cuda` to override this selection. Fine-tuning uses the `device` field in its recipe.

Generation options for evaluation are `--samples`, `--temperature`, `--top-p`,
`--seed` and `--timeout`. ManyTypes4Py can use `--type-vocabulary` when a vocabulary
was not built from TRAIN. See [benchmarks](benchmarks.md).

## Experiment commands

Each action below follows `experiment` and runs one script of the
variable-renaming experiment in a new process with `PYTHONHASHSEED=0`. Options
after the action go to the script; `experiment ACTION --help` lists them.

| Action | Required arguments | Purpose |
| --- | --- | --- |
| `list` | None | List the actions and their scripts |
| `prepare` | `--out` | Evaluation corpus (`--n`) and optional ClozeTest cache (`--cloze`) |
| `transform` | `--input`, `--output` | Training set for `--mode plain`, `halfvar` or `highdiv` |
| `tasks` | `--model`, `--corpus`, `--run`, `--group` | Identifier reproduction (`idrep`) or misuse localization and ClozeTest (`bench`) |
| `nonidentifier` | `--model`, `--corpus`, `--run` | Appendix D.5 non-identifier accuracy and retention |
| `misuse-detection` | `--model`, `--corpus`, `--run` | Appendix D.5 two-way misuse detection |
| `copy-score` | `--model`, `--corpus`, `--run` | Figure 7A copy score |
| `zero-ablation` | `--model`, `--corpus`, `--run` | Zero ablation of k = 1..`--kmax` heads |
| `mean-ablation` | `--model`, `--corpus`, `--run` | Figure 7B mean ablation |
| `forced-attention` | `--model`, `--corpus`, `--run` | Table 6 forced attention for `--ks` |
| `first-error` | `--model`, `--corpus`, `--run` | Appendix E.2 paired first-error positions |
| `mechanistic` | `--model`, `--corpus`, `--run` | The four mechanistic protocols in one run |
| `suite` | `--model`, `--corpus`, `--run` | Every entry point above, each in its own process |
| `report` | `--runs`, `--out` | Mean and sample SD over seeds of `mechanistic` runs, by `--arm` |
| `toy` | `--run` | All mechanistic protocols on small random models |

`--model` accepts a training run directory or a Hugging Face model directory.
The analyses default to 2,000 functions, context 320, batch 16 and `--device cpu`;
`--device auto` or `cuda` selects a GPU. `--limit N` keeps the first N functions,
and `--config` reads defaults from a JSON file such as the experiment's
`paper.json`. The tasks default to `--device auto`.

See [variable renaming](variable-renaming.md) for complete command sequences.
