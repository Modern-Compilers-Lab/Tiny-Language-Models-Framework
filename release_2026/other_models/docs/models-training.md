# Models and training

The presets build small Hugging Face causal models with random weights. They
are architecture variants of StarCoder, Llama and GPT-NeoX; a model name does not
load pretrained weights. To use saved weights, supply a checkpoint explicitly.

| Model | `--arch` | Parameters | Default dataset | Small context |
| --- | --- | ---: | --- | ---: |
| Small StarCoder | `gpt_bigcode` or `starcoder` | 18,919,552 | CodeSearchNet | 8,192 |
| Small Llama | `llama` | 18,292,416 | CodeSearchNet | 1,024 |
| Small GPT-NeoX | `gpt_neox` or `gpt-neox` | 18,334,848 | The Vault | 1,024 |

These counts use the default 49,152-token vocabulary and small configurations.
All three use width 192, 20 layers, 3 heads and tied embeddings. StarCoder uses
multi-query attention and learned positions. Llama uses rotary positions and
RMSNorm. GPT-NeoX uses partial rotary positions and parallel residuals.

The JSON `scale` field also accepts `large`: width 768, 20 layers and 12 heads.
Its context is 2,048 for Llama/NeoX and 8,192 for StarCoder. A JSON `model` object
can override `hidden`, `layers`, `heads` and `context`. These overrides affect
fresh models; a loaded checkpoint supplies its own architecture and context.

## Training recipes

The defaults live in [`configs/`](../configs/). The CLI starts with the selected
architecture's small recipe, overlays `--config`, then applies explicit command
options. Supply `--data` when selecting a particular prepared file.

```powershell
python release_2026/other_models/run.py train --config release_2026/other_models/configs/llama-small.json --data release_2026/other_models/runs/csn-prepared/data/codesearchnet.jsonl --run release_2026/other_models/runs/llama
```

Without an explicit file, `train` looks in `<data-run>/data/<dataset>.jsonl`.
`--data-run` defaults to `other_models/runs/datasets`; a missing file triggers
complete preparation. An explicit `--dataset` changes the default corpus and
supersedes a recipe's `data` field unless `--data` is also supplied.

| Field | Default | Meaning |
| --- | --- | --- |
| `arch` | Required | Model family |
| `scale` | `small` | Architecture preset |
| `dataset` | Model default | Corpus selection/provenance |
| `data` | Selected by CLI | Training JSONL path |
| `tokenizer` | `bigcode/tiny_starcoder_py` | Hugging Face ID or local tokenizer directory |
| `pretrained` | Unset | Existing checkpoint for weight continuation |
| `steps` | 20,000 | Optimizer updates |
| `batch_size` | 16 | Sequences per update |
| `micro_batch` | 1 | Sequences per forward/backward pass |
| `max_len` | 320 | Token truncation length, capped by model context |
| `lr` | 0.0003 | Learning rate after warmup |
| `warmup` | 2,000 | Linear warmup updates; zero disables warmup |
| `seed` | 2 | Model initialization and data-order seed |
| `save_every` | 2,000 | Intermediate model/tokenizer checkpoint interval |
| `shuffle_buffer` | 1,024 | Maximum tokenized documents buffered for shuffling |
| `device` | `auto` | CPU or CUDA |
| `precision` | `float32` | `float32` or `bfloat16` autocast |
| `threads` | 4 | PyTorch CPU threads |
| `gradient_checkpointing` | `false` | Reduce activation memory during training |

Field names in JSON use underscores; CLI options use hyphens. `scale`, `model`,
`shuffle_buffer` and `gradient_checkpointing` are JSON settings. The rest of the
training options are listed by `train --help`.

`batch_size` must be divisible by `micro_batch`. The trainer accumulates gradients
across that many microbatches, uses AdamW with weight decay 0.01, clips gradients
at 1.0, and keeps the learning rate constant after warmup. It cycles over the
JSONL with a bounded shuffle buffer. This is a single-device trainer.

## Continue from saved weights

```powershell
python release_2026/other_models/run.py train --arch llama --pretrained release_2026/other_models/runs/llama/model --tokenizer release_2026/other_models/runs/llama/model --data release_2026/other_models/runs/csn-prepared/data/codesearchnet.jsonl --steps 100 --warmup 0 --run release_2026/other_models/runs/llama-continued
```

Use the checkpoint's tokenizer. Vocabulary and architecture mismatches are
rejected. This starts a new optimizer and schedule from saved weights. The
trainer does not save optimizer/RNG state for exact interrupted-run resume.

Final weights and tokenizer files go in `model/`. Intermediate checkpoints go in
`checkpoints/step-N/`; the training report is `results/training.json`. A completed
output cannot be overwritten by another training command.

## Load and generate

```powershell
python release_2026/other_models/run.py generate --model release_2026/other_models/runs/llama/model --prompt 'def add(a, b):' --max-new-tokens 32 --local-files-only
```

Generation is greedy. Prompt plus generation budget must fit the model context.
The Python interfaces are `build_config`, `build_model`, `load_model` and
`HFModel` in [`models.py`](../models.py). `load_model` returns the model and
tokenizer; `HFModel` supplies the prediction/scoring methods used by benchmarks.

[Benchmark fine-tuning](benchmarks.md#optional-fine-tuning) uses the same trainer
but obtains architecture and tokenizer from the source checkpoint. BIFI and
ManyTypes4Py mask prompt loss and append EOS to complete targets; PY150 uses
ordinary causal training. Each fine-tune saves a separate checkpoint.
