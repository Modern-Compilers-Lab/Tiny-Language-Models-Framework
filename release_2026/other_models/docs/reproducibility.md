# Runs and reproducibility

Keep each preparation, training and evaluation in a named run directory. The
examples use `other_models/runs/` for shared work and the experiment's own
`runs/` directory for renaming campaigns. Generated runs are ignored by Git.

## Output files

| Workflow | Files to keep |
| --- | --- |
| Corpus download | `sources/<dataset>/.source-manifest.json`, downloaded files, `results/download-<dataset>.json` |
| Corpus preparation | `data/<dataset>.jsonl`, sidecar manifest, `data/heldout_repos.json` |
| Training | `config.json`, `logs/training.jsonl`, `model/`, `results/training.json`, optional `checkpoints/step-N/` |
| Benchmark preparation | `data/<name>/manifest.json` and available split JSONL files; raw files under `sources/<name>/` when downloaded fully |
| Benchmark evaluation | `config.json`, `results/metrics.json`, `results/<name>.predictions.jsonl` |
| Standalone fine-tune | `model/`, training config/report and generated `data/finetune.jsonl` |
| Evaluation with fine-tuning | New checkpoint and training records under `finetune/`, evaluation records under `results/` |
| Renaming corpus | `real6_corpus.json`, optional `cloze_all_items_n<N>.json`, `manifest.json` with hashes |
| Renaming analyses | `config.json` with corpus, checkpoint and source hashes, `logs/run.log`, result JSON under `results/` |
| Renaming report | `summary.json`, `summary.csv` and `curves.csv` |

Temporary files and caches are separate from these records. A cache helps avoid
repeated work; the manifests establish which inputs and outputs were used.

## Reuse rules

Commands using `setup_run` reject a changed recorded config. Standard benchmark
preparation rejects an existing prepared output. Training rejects completed model
outputs, even for an identical recipe. Choose a new directory for another run.

Downloads can reuse verified completed files and retry incomplete transfers.
Training continuation uses `--pretrained` with a new output directory and a new
optimizer. Exact optimizer-state resume is not implemented.

Dataset downloads record pinned Hugging Face revisions. Benchmark source
manifests record file URLs, sizes and hashes. Prepared benchmark manifests record
split counts and hashes; evaluation checks those hashes. Fine-tuning also records
source checkpoint hashes and checks TRAIN against heldout IDs and exact inputs.

The renaming analyses refuse a run directory that already has a different
`config.json`; `--resume` continues only when the options and the corpus,
checkpoint and source hashes all match. Summary metrics are recomputed from the
saved result files. Each entry point runs in its own process so that the scoring
library's process-wide settings do not affect another task.
