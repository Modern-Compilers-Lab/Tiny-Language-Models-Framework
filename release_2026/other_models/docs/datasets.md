# Training datasets

| Key | Source repository | Input used by preparation |
| --- | --- | --- |
| `codesearchnet` | `code-search-net/code_search_net` | Python training Parquet shards |
| `thevault` | `Fsoft-AIC/the-vault-function` | All available Python training partitions |
| `codeparrot` | `codeparrot/codeparrot-clean` | Compressed JSON records |

StarCoder and Llama default to CodeSearchNet. GPT-NeoX defaults to The Vault.
`--dataset` overrides that choice for any model.

```powershell
python release_2026/other_models/run.py prepare --arch llama --dataset thevault --run release_2026/other_models/runs/llama-vault-data
```

## Download and prepare

`download` retains the upstream repository. `prepare` writes the eligible Python
training rows as JSONL and downloads sources when no cache path is supplied.

```powershell
python release_2026/other_models/run.py download --dataset codesearchnet --dry-run --run release_2026/other_models/runs/csn-preview
python release_2026/other_models/run.py download --dataset codesearchnet --run release_2026/other_models/runs/csn-source
python release_2026/other_models/run.py prepare --dataset codesearchnet --cache release_2026/other_models/runs/csn-source/sources/codesearchnet --run release_2026/other_models/runs/csn-prepared
```

The default download includes every repository file, including other languages
and splits. Preparation selects Python training records. These are separate
scopes, recorded in the manifests.

Use `--dataset all` to download or prepare the three corpora in one command.
Use `--revision` to select a source ref; the downloader records its resolved
commit and retains it on retry.

## Limits

| Option | Scope |
| --- | --- |
| `prepare --n-docs N` | Stop after N eligible documents; zero means all |
| `--max-files N` | Download at most N matching training files |
| `download --allow-pattern PATTERN` | Select matching repository files; repeat for several patterns |
| `download --dry-run` | Show revision, file count and size without downloading |

A document cap alone does not cap the raw download. Use a file selection when
transfer size matters. `--max-files` applies to a new download and cannot be
combined with `prepare --cache`. A changed file selection needs a new source run.

Preparation always uses all available Vault Python training partitions exposed
by the CLI; model size does not select a smaller data partition.

## Local sources

Use a directory created by `download`, or a directory with the same supported
source layout. Downloaded sources have `.source-manifest.json`, which preparation
checks before reuse. An external directory without that manifest is recorded as
having unknown source revision/completeness.

`--reference-cache` points to a CodeSearchNet source with `python/test-*.parquet`.
If absent, preparation reuses the test split in the corpus cache when possible,
or downloads that split under the preparation run's `reference/` directory.
It is needed for all three training corpora.

## Filters and output

Preparation normalizes repository names and excludes repositories present in
CodeSearchNet test. It also excludes exact heldout function hashes, records with
missing repository provenance, duplicate code, and code outside 60–200,000
characters. `--exclude-repo` adds repository exclusions. Hashes use parsed ASTs
where possible and normalized text otherwise.

These checks cover the supplied corpora against the reference split. They do not
establish that a separately pretrained checkpoint has never seen a benchmark.

Rows look like this:

```json
{
  "id": "content-hash",
  "code": "def add(value, other):\n    result = value + other\n    return result\n",
  "repo": "owner/repository",
  "split": "train",
  "meta": {"source": "codesearchnet", "path": "module.py"}
}
```

Preparation writes `data/<dataset>.jsonl`, its `.manifest.json` sidecar, and
`data/heldout_repos.json`. The manifest distinguishes all rows available in a
provided cache from a complete upstream repository. `full` and `source_full`
should be checked before treating a run as a full-corpus experiment.

The trainer requires a string `code` field for ordinary causal training and
accepts TRAIN rows only. Renaming preserves all non-code fields, row order and
source IDs. Benchmark schemas are described in [benchmarks](benchmarks.md#local-data).
