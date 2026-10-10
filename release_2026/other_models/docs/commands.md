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

Each action below follows `experiment`. The current family is variable renaming.

| Action | Required arguments | Purpose |
| --- | --- | --- |
| `list` | None | List treatments, rename levels and ten task names |
| `prepare-data` | `--run`, plus `--arch` or `--dataset` | Shared training-corpus preparation, optionally `--transform` |
| `transform` | `--input`, `--run` | Apply `--transform plain`, `halfvar` or `highdiv`; default halfvar |
| `prepare-training` | `--source`, `--tokenizer`, `--run-root` | Write condition/seed recipes and a campaign cell list |
| `token-cache prepare` | `--config`, `--run`, `--cache` | Tokenize a code-only training corpus to a verified cache |
| `token-cache train` | `--config`, `--run`, `--cache` | Train from cached IDs with the shared trainer |
| `campaign` | `--cells` | Run explicit cells sequentially; optional `--status` output |
| `prepare` | `--run` | Prepare selected experiment tasks, reusing shared benchmark data |
| `prepare-views` | `--data`, `--run` | Build paired inputs; `--tasks` and `--limit` select a subset |
| `evaluate` | `--data`, `--run`, `--model` | Save predictions; non-paper tasks also require `--views` |
| `score` | `--data`, `--run` | Score saved predictions for the selected `--tasks` |
| `diversity` | `--config`, `--run`, `--cache`, `--baseline-cache` | Check identifiers and changed training IDs; optional `--n` |
| `freeze` | `--data`, `--out` | Hash code/inputs; `--verify` checks the saved snapshot |
| `report` | `--run`, `--out` | Verify saved scores and export JSON/CSV |

`prepare --n N` caps source rows. `--benchmark-data NAME=PATH` selects prepared
CodeXGLUE inputs; `--benchmark-source NAME=PATH` selects their existing raw files.
Both are repeatable. `--benchmark-run` chooses shared benchmark storage.
`--corpus-only`, `--cache` and `--pinned-eval` control the reference corpus.

`prepare-training` accepts architecture, dataset, scale, treatments, seeds,
workers and training budgets. It defaults to three treatments, seeds `2,3,4`,
20,000 updates and `--device auto`. `--pretrained` gives each cell the same
starting weights; `--token-cache` creates caches during preparation.

`evaluate` defaults to context 320, batch 16, float32 and CUDA. Set `--device cpu`
or `auto` when needed. Preparation and evaluation are uncapped by default; use
positive `--n` and `--limit` for a pilot. Keep the `--tasks` selection consistent
across view preparation, prediction and scoring.

See [variable renaming](variable-renaming.md) for complete command sequences.
