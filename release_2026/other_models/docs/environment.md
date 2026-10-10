# Environment

Use Python 3.10 or newer. Install the integration's requirements in the environment used for all
commands. The examples below create an environment inside `other_models`.

```powershell
python -m venv release_2026/other_models/.venv
release_2026/other_models/.venv/Scripts/Activate.ps1
python -m pip install -r release_2026/other_models/requirements.txt
```

On Linux, activate with
`source release_2026/other_models/.venv/bin/activate`.
For tests, also install `pytest` in that environment.
Without activation, call `.venv/Scripts/python.exe` on Windows or `.venv/bin/python`
on Linux using its full path under `other_models`.

The requirement file covers PyTorch, Transformers, dataset and Parquet readers,
Hugging Face downloads, NumPy and the Python tree-sitter parser used for renaming.
There is no installation step for codelab itself.

## Entry points

From the repository root:

```powershell
python release_2026/other_models/run.py --help
python release_2026/other_models/run.py list
```

An absolute path to `run.py` works from another directory. Output and input paths
you pass remain relative to that working directory. Package invocation is also
available from the repository root:

```powershell
python -m release_2026.other_models --help
```

## CPU and CUDA

`train`, `generate` and standard benchmark evaluation default to `--device auto`,
which selects CUDA when available and otherwise CPU. Use `--device cpu` or
`--device cuda` to select a device explicitly. Renaming evaluation defaults to
CUDA. Benchmark fine-tuning uses the device in its training recipe.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For GPU use, install a PyTorch build compatible with the machine's driver in this
same environment. Installing the requirement file alone does not establish that
CUDA is available. Subprocesses use the calling Python interpreter.

## Offline runs

Use local tokenizer/checkpoint directories and existing dataset sources. For
Hugging Face operations in an offline session:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
```

Generation also accepts `--local-files-only`. Those flags do not prevent direct
benchmark HTTP downloads; supply local data or verified raw source directories.
The [dataset guide](datasets.md#local-sources) and
[renaming guide](variable-renaming.md#shared-benchmark-data) cover those paths.
