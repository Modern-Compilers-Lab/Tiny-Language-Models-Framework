# Research question 1 : measuring robustness to OOD shifts

## starter notes

- This readme.md file serves as a guide that links each folder name to the actual experiment mentioned in the manuscript.

- It is highly recommend for the user to first get familiar with the framework's structure by visiting the `dry_framework` repo, execute the scripts few times and try to play with the hyperparameters, this will make it much easier for the user to read through the differents scripts in this `research_question_1` repo, which contains many references to the `dry_framework` repo.

- Use this folder as a way to copy existing configurations and technics to reproduce results, rather than a usable framework (to do the latter use `dry_framework`)

- This file clarifies new files that did not exist in the `dry_framework`.

- The 'results' and 'datasets' folders are not shipped with the repository: they are recreated automatically when you run the scripts (everything is reproducible), and are ignored through the root `.gitignore` to keep the project light enough to share on github

## Data generation folder :

- `alpha_0.X` : Generates data in tiny-alpha language 
    - scripts containing the keyword `BPE` : generates and tokenizes data using BPE tokenizer (produces `bpe_tokenizer.json` as well)
    - scripts containing the keyword `UNIGRAM` : generates and tokenizes data using  SP UNIGRAM tokenizer (produces `sp_unigram_newest.model/vocab` as well)
    - the rest of the scripts in the same level generate and tokenizes data with the regular Character level tokenizer (20 or 300 million examples)
    - `OOD` : contains 6 folders, each reflects a different change in the genreation rules of tiny-alpha, every folder contains the regular scripts from `dry_framework` but the hyperparameters are tweaked to break at least one rule from the regular set :
        - deeper_nesting : the indentation level parameter is changed (`max_nesting_depth`)
        - denser_snippets : modified the condition in `HIGH-LEVEL-FILTERING` zone so that at least a single indentation contains 3 if blocks or more
        - hidden_numbers : LEGACY, used to include integers that were not generated in the original script
        - hidden_variables : modified the `arr` structure so that the generator produces variables not generated in the original script
        - mixed_ops : instead of using `main_op` to fix a single operation type per snippet, the script makes sure there are at least one reachable occurence of both + and -
- `alpha_1.X` and `alpha_2.X` : similar to `dry_framework` but the only tweak is in the `max_nesting_depth` (e.g. from 0 to 6 instead of 0 to 2)
- `alpha_D` is also similar to `dry_framework` but the only tweak is in the `max_nesting_depth` (e.g. from 7 to 9 instead of 0 to 2)
- `beta_0.X`, `beta_1.X`, `beta_2.X` and `beta_D` mirror the alpha folders, the only difference is that the generation includes `for loops` as well


## Training folder :

- `exp1_alpha`: does a regular training on 20 million regular tiny-alpha dataset
- `exp2_alpha`: does a regular training on 300 million regular tiny-alpha dataset
- `exp3_alpha1`, `exp3_alpha2`: trains respectively on the `alpha_1.X` and `alpha_2.X` datasets
- `exp4_alpha`: does an ablation study, where it does multiple training runes : 
    - `exp4_alpha_Alibi_BPE` : trains on the regular alibi (Attention with linear biases) positional encoding but on data tokenized using BPE tokenizer (same training script, different data tokenizer)
    - `exp4_alpha_Alibi_SP` : trains on the regular (Attention with linear biases) alibi positional encoding but on data tokenized using SP UNIGRAM tokenizer (same training script, different data tokenizer)
    - `exp4_alpha_APE` : trains on an different positional encoding (Absolute positional encoding) on regular Character level tokenized data (different training script to modify the positional encoding within the model architecture)
    - `exp4_alpha_RoPE` : trains on an different positional encoding (Rotary positional encoding) on regular Character level tokenized data (different training script to modify the positional encoding within the model architecture)
    - `exp4_alpha_RPE` : trains on an different positional encoding (relative positional encoding) on regular Character level tokenized data (different training script to modify the positional encoding within the model architecture)
- the corresponding `beta` folders also mirror the alpha folders, the only difference is that it trains on the beta dataset folders and not the alpha's


## Evaluation folder : 
- `exp{NUM}_alpha` evaluates the checkpoints produces from the `exp{NUM}_alpha` in the training folder, same goes for `beta` folders
- all evaluation folders follow the same structure of (eval.py, scheduler.py, tokenizer.py), with the following differences from the `dry_framework` structure
    - evaluations are not done on a single `test.txt` file, instead `scheduler.py` manages multi-checkpoint and multi-test-set evaluations
    - the multiple test sets usually comes from the genreated OOD datasets in the data generation folder (so along side the regular test set, you would also evaluate on deeper nesting, longer snippets ..etc)
    - some folders contain `semantic_preserving_edits_scripts` folder, which contains scripts that, after the model gets evaluated on regular test set, and produces results, these scripts take the sucessful examples (examples that the model got right) and replaces their variable names with new unseen names, therefore having to re-evaluate the model on these perturbed snippet using `scheduler_semantic_preserving_edits.py` scripts (a more detailed explanation is provided in the README.md file of the `research_project_2` repo
    - in the `exp4` folders (or ablation study folders), tokenizer files must correspond to the same tokenizers used to tokenize the data on which the model was trained
    - also, in those `exp4` folders, `eval.py` files are modifed to match the architecture of the model trained (so a model using APE positional encoding, will have to be evaluated using an eval.py file of a matching configuration)
