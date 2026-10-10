# renaming/lib: the code that produced the paper's numbers (do not edit)

These files are byte-identical to the versions used for the paper; `SHA256SUMS` lists them and
`tests/test_lib_integrity.py` checks them. They import each other by bare module name, so this
folder must be on `sys.path`. `renaming/env.py` arranges that, and also sets the environment
variables below before anything here is imported.

| file | role |
|---|---|
| `corpus.py` | training-data treatments: the document subset rule, fresh corpus-unique names (`FreshNames`), `rename_document_halfvar`, and the certification helpers (`identifier_skeleton`, `invert_check`) |
| `prep.py` | builds the baseline and high-diversity token streams in one pass over codeparrot-clean |
| `train.py` | the from-scratch trainer: GPT-BigCode (tiny_starcoder_py config), DDP, fixed global batch, Muon + AdamW, resumable |
| `decmech.py` | evaluation functions and items: `real6_occs` and `token_layout` (the items of every head analysis), `cmd_idcurve` (identifier reproduction), `cmd_vmloc` (variable-misuse localization), `cmd_cloze` (ClozeTest-all), `cmd_comp` (pilot next-token accuracy) |
| `whyscale.py` | the evaluation corpus (`_real6_corpus`: CodeSearchNet test functions and filters) and the renamed views (`_real6_build`) |
| `varmisuse.py` | variable-misuse items, and the pilot's two-way detection test |
| `poolfix.py` | helpers used by `decmech.py` and `varmisuse.py` |
| `perturb/` | the variable extractor (tree-sitter) shared with the rest of the project |

These modules contain many more commands than the paper uses (they were the project's research
code). Only the functions named above are on the paths this package runs.

Environment they expect (set by `renaming/env.py` / `config.sh`):

- `PYTHONHASHSEED=0` before Python starts. Renamed views are seeded with hashes.
- `WHYSCALE_CACHE`: the folder of the evaluation sets (`<work>/eval_corpus`).
- `HF_HOME`: the Hugging Face cache. The modules fall back to paths on the original machine when
  these are unset.
- `WHYSCALE_BLOCK` is set to 319 as a side effect of importing `varmisuse.py`. This is why
  identifier reproduction runs in a separate process from variable-misuse localization and cloze.
