# Zero-Shot Variable-Renaming Robustness of Variable-Diversified CodeBERT

This experiment asks a single question:

> If you pre-train a CodeBERT model on CodeSearchNet **after scrambling all
> variable names into random character sequences**, do its raw (zero-shot)
> code representations become more robust to variable renaming than the
> original CodeBERT — and how does it compare to ContraBERT_C?

This release evaluates the models with **two separate tests**:

1. **Regular test (in-distribution quality)** — **MAP@R on the untouched
   POJ-104 test set.** Each model is **fine-tuned** on POJ-104 clone detection,
   then scored with MAP@R on the *regular, un-renamed* test set. This says how
   good the model is at the actual task when trained on it. **MAP@R is reported
   only here, on the untouched data.**
2. **Robustness test (renaming sensitivity)** — **zero-shot top-1 retrieval
   accuracy at N = 0 → 8.** The raw pre-trained encoder is used *as-is* (no
   fine-tuning): embed POJ-104, retrieve nearest neighbour by cosine similarity,
   then rename variables in the query (N = 0, 1, 4, 8) and measure how many
   retrievals survive.

Key point: **MAP@R is the regular test only; robustness is the zero-shot
retrieval test only.** MAP@R is never computed on renamed code, and the
robustness test never fine-tunes.

---

## 1. The idea in one picture

```
CodeSearchNet (6 langs)
        │
        │  variable names → RANDOM character sequences   (diversify_variable_names.py)
        │  (re-randomized every epoch)
        ▼
  Pre-train CodeBERT  ───────────────►  checkpoints/exp1_diverse/checkpoint-*
   (train.sh / train_codebert.py)            (one per epoch)
        │
        │   each checkpoint, NO fine-tuning
        ▼
  Each checkpoint + 2 imported baselines:
        │                                • microsoft/codebert-base
        │                                • claudios/ContraBERT_C
        │
        ├───────────────────────────────┬─────────────────────────────────┐
        ▼                                ▼                                  │
  ROBUSTNESS TEST                  REGULAR TEST                             │
  (A) ZERO-SHOT, no fine-tune      (B) FINE-TUNE on POJ-104                 │
   (evaluate_zeroshot.py)           (evaluate_robustness.py)               │
        │                                │                                  │
        │  top-1 retrieval               │  MAP@R on the UNTOUCHED          │
        │  at N = 0,1,4,8                 │  POJ-104 test set                │
        ▼                                ▼                                  │
  build_zeroshot_table.py          build_table.py                          │
        │                                │                                  │
        ▼                                ▼                                  │
  zeroshot_results.{txt,csv}       finetuned_results.{txt,csv}             │
        (both in eval_results_varname/)                                    │
        │                                                                   │
        └─ query programs renamed to var_<number> at N = 1,4,8 via the ◄────┘
           SHARED, fixed-seed sets  (generate_renamed_testsets.py)
```

Two **distinct** renaming stages — do not confuse them:

| Stage | Script | What a renamed variable looks like |
|-------|--------|------------------------------------|
| **Training-data scrambling** | [diversify_variable_names.py](diversify_variable_names.py) | a **random character sequence** (e.g. `xk3f9q`, `q7a_2z`) — *not* `var_…` |
| **POJ-104 test perturbation** | [generate_renamed_testsets.py](generate_renamed_testsets.py) | `var_<random number>` (e.g. `var_48127`) |

---

## 2. Pipeline & scripts

Run the steps in order. All paths are relative to the project root.

### Step 0 — Download data
```bash
python download_data.py            # or: bash download_data.sh
```
- [download_data.py](download_data.py) — downloads **CodeSearchNet** (all 6
  languages: Python, Java, JavaScript, Ruby, Go, PHP; 1,880,853 train examples)
  and **POJ-104** clone-detection (`google/code_x_glue_cc_clone_detection_poj104`)
  into `data_cache/`. POJ-104 is written as `data_cache/poj104/{train,valid,test}.jsonl`.

### Step 1 — Pre-train the variable-diversified CodeBERT
```bash
bash train.sh                      # 8-GPU DDP, ~100k steps (~109 epochs)
```
- [train.sh](train.sh) — launcher. Effective batch 2048, lr 5e-4 (10k warmup,
  linear decay), max seq length 512, FP16, **`--diversify_varnames`**, saves a
  checkpoint **every epoch** to `checkpoints/exp1_diverse/`.
- [train_codebert.py](train_codebert.py) — the trainer. With `--diversify_varnames`
  it calls `diversify_code()` on every example, **re-randomizing the variable
  names each epoch** so the model never sees the same names twice.
- [diversify_variable_names.py](diversify_variable_names.py) — the scrambler.
  Uses tree-sitter ASTs to find real variable/parameter identifiers (never
  keywords, types, or function names) and replaces each with a fresh random
  character sequence (`generate_random_name`). Used as a library by the trainer;
  also runnable standalone to materialize a diversified dataset copy.

### Step 2 — Build the shared renamed POJ-104 test sets
```bash
python generate_renamed_testsets.py \
    --data_dir data_cache/poj104 --rename_counts 1,4,8 --seed 123456
```
- [generate_renamed_testsets.py](generate_renamed_testsets.py) — writes
  `data_cache/poj104/test_renamed_N{1,4,8}.jsonl`. Renames are produced **once**
  with a fixed seed so **every** model (all checkpoints + both baselines) is
  scored on **byte-for-byte identical** perturbations — a fair head-to-head.
  Renamed variables become `var_<number>`.
- [evaluate_robustness.py](evaluate_robustness.py) — **used here only as a
  dependency**: `generate_renamed_testsets.py` imports its `rename_variables` /
  `extract_variables` helpers. (No fine-tuning is involved in this experiment.)

### Step 3 — Zero-shot evaluation (all checkpoints + both baselines)
```bash
bash run_zeroshot_parallel.sh
```
- [run_zeroshot_parallel.sh](run_zeroshot_parallel.sh) — 8-GPU orchestrator
  (one process per GPU, refill on completion). Runs **14 jobs**:
  - `baseline_codebert` — `microsoft/codebert-base` (loaded from a local
    safetensors copy under `models_cache/`)
  - `baseline_contrabert` — `claudios/ContraBERT_C`
  - 12 of your checkpoints: epochs 9, 18, 27, …, 108 (every 9 epochs)

  Then it automatically calls the table builder (Step 4).
- [evaluate_zeroshot.py](evaluate_zeroshot.py) — the actual evaluator, run once
  per model:
  1. Load the **raw pre-trained encoder** (no task head, no fine-tuning).
  2. Embed every POJ-104 test program with **mean-pooled** token hidden states.
  3. Rank all programs by **cosine similarity**; a query is *correct at N=0* if
     its nearest neighbour shares its label (clone class).
  4. For each `N ∈ {1, 4, 8}`, re-embed the **renamed** query (from the shared
     `test_renamed_N{N}.jsonl`) and check whether it still retrieves a correct
     neighbour. **Retention = correct@N / correct@0.**

  Output: `eval_results_varname/zeroshot_<model>.json`.

### Step 4 — Compile the results table
```bash
python build_zeroshot_table.py --eval_dir eval_results_varname
```
- [build_zeroshot_table.py](build_zeroshot_table.py) — reads every
  `zeroshot_*.json` and emits, with baselines listed first then checkpoints in
  epoch order:
  - `eval_results_varname/zeroshot_results.txt`
  - `eval_results_varname/zeroshot_results.csv`

  Columns: `Correct@N=0 (#/%)`, retention at `N=1`, `N=4`, `N=8`, and the
  `N=8 drop` (= `1 − retention@8`).

### Step 5 — Regular test: MAP@R on untouched POJ-104 (fine-tuned)
This is the **in-distribution quality** number. Each model is **fine-tuned on
POJ-104 clone detection**, then scored with **MAP@R on the regular, un-renamed
test set**. This is the *only* place MAP@R is reported.

```bash
bash run_eval_parallel.sh                       # fine-tune + MAP@R, all 14 models
python build_table.py --eval_dir eval_results_varname
```
- [run_eval_parallel.sh](run_eval_parallel.sh) — 8-GPU orchestrator (same job
  layout as the zero-shot run: 2 baselines + epochs 9…108). For each model it
  calls `evaluate_robustness.py` to fine-tune (2 epochs, lr 2e-5, batch 8) and
  evaluate, writing `eval_results_varname/finetuned_<model>.json`.
- [evaluate_robustness.py](evaluate_robustness.py) — the fine-tune + MAP@R
  engine: loads the encoder → **fine-tunes on POJ-104 clone detection**
  (`train_model`) → computes **`MAP@R_N0`** on the **untouched** test set
  (`compute_map_at_r`). (Same file already used as a helper in Step 2.)
- [build_table.py](build_table.py) — writes `finetuned_results.{txt,csv}`.

> **MAP@R is computed only on the untouched POJ-104, only after fine-tuning.**
> The robustness test (Steps 3–4) is the zero-shot top-1 retrieval at N=0→8 and
> never uses MAP@R.
>
> *Note:* `run_eval_parallel.sh` passes `--rename_counts 0,1,4,8`, so
> `evaluate_robustness.py` additionally records the fine-tuned model's accuracy
> on the renamed sets and `build_table.py` shows it as extra `N=1/4/8` columns.
> This is a **byproduct**, not part of the two-test design above — the headline
> robustness measure is the zero-shot retrieval test. To skip it entirely, run
> with `--rename_counts 0`.

---

## 3. How to read the numbers

**Robustness test — zero-shot table (`zeroshot_results.*`):**
- **N = 0 %** — top-1 retrieval accuracy on the *unperturbed* POJ-104 test set
  (out of `num_test`). This is the model's baseline retrieval quality.
- **N > 0 %** — **retention rate** = fraction of the originally-correct queries
  that remain correct after `N` variables are renamed to `var_<number>`.
- **N = 8 drop** — how much retention is lost at the hardest setting; smaller is
  more robust.

**Regular test — fine-tuned table (`finetuned_results.*`):**
- **MAP@R** — Mean Average Precision @ R on the **untouched** POJ-104 test set
  after fine-tuning. This is the **in-distribution** quality number (higher =
  better at the actual clone-detection task). *This is the column that matters
  here.*
- **N = 1 / 4 / 8** — byproduct only (fine-tuned model's accuracy on renamed
  sets); not part of the two-test design.

Difficulty levels are **N = 0, 1, 4, 8** renamed variables per program (not every
integer 1–8). Higher `N` = more variables scrambled = harder.

---

## 4. Configuration notes

- The exact difficulty levels live in `evaluate_zeroshot.py` (`rename_counts`,
  default `(0, 1, 4, 8)`) and must match the `--rename_counts` used in Step 2.
- Embeddings use **mean pooling** by default (`--embedding meanpool`), matching
  the ContraBERT zero-shot methodology; `--embedding cls` is also available.
- All renamings use the fixed seed **123456** for reproducibility.
- `run_zeroshot_parallel.sh` picks the interpreter from `$PYTHON` (defaults to a
  hard-coded conda path); override it for your environment, e.g.
  `PYTHON=$(which python) bash run_zeroshot_parallel.sh`.

## 5. Requirements

See [requirements.txt](requirements.txt) (PyTorch, Transformers, Datasets,
tree-sitter + per-language grammars, NumPy, tqdm). A CUDA GPU is required for
training; evaluation also expects GPUs (the orchestrator assumes 8, but
`evaluate_zeroshot.py` runs fine on a single GPU/CPU for one model at a time).
