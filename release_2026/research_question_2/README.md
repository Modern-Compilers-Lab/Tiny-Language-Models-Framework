# Research question 2 : measuring robustness to variable renaming shift

## starter notes

- This readme.md file serves as a guide that links each folder name to the actual experiment mentioned in the manuscript.

- It is highly recommend for the user to first get familiar with the framework's structure by visiting the `dry_framework` repo, execute the scripts few times and try to play with the hyperparameters, this will make it much easier for the user to read through the differents scripts in this `research_question_2` repo, which contains many references to the `dry_framework` repo.

- Use this folder as a way to copy existing configurations and technics to reproduce results, rather than a usable framework (to do the latter use `dry_framework`)

- This file clarifies new files that did not exist in the `dry_framework`.

- You will see a lot of empty 'results' or 'datasets' folders, they are displayed on purpose to show the architecture of the project, content is deleted so that the project is light enough to be shared on github (everything is reproducible) - "update: you will not be able to see those empty folders in github because git does not track empty repositories, you will find some README.md files talking about repos that you cannot find, this is because they are not tracked, the full version with the empty repos can be found in the .zip file provided in `release_2026`"
## Data generation folder :

- `data_X_p` : any folder following this naming (with X being a value from $0.1$ to $80$) reflects a regular tiny-alpha dataset surrounding 20 million examples, the only difference that separates each folder (which, spoiler alert, is in the name itself) is this X value that varies across the datasets, this value is called 'variable diversity' and this in a nutshell tells how many unique patterns of variable names does the model see, this value is a percentage (so from 0 to 100). more details are provided in the mansucript, but in general, smaller percentages means less variable name patterns and combinations that the models sees, so more repeating names in the dataset, bigger percentages means more unique combinations that 'enrich' the dataset's variable name diversity. a value of 100 means that basically every single possible variable name combination is present in the dataset:
    - all single-character variable names are present no matter the value of variable diversity, longer variables however follow the variable diversity rule
    - variable name length is limited (default one used in the project is 5), so it is possible to cover every single possible combination and contain it in one dataset
    - even though we can go for 100 coverage it's not our goal, our goal is to expose the model on $x$% of all possible combinations, then test the model on code snippets with unseen variables (variables that are not part of the $x$% group). and then see whether a bigger $x$% would yield better performance (hence testing multiple datasets of different variable diversities), having a variable diversity value of 100 would not allow us to (hide) variable combinations for the test part so it's not really useful for our goal here.
    - a part from that, scripts follow the same logic as `dry_framework`, in every script the `rarity` parameter varies from $0.01/100$ up to $80/100$
- `OOD` : in addition to testing on replaced variable names, we can also test on other OOD tasks, although not explicitly cited in the manuscript, the process follows the same logic as `research_question_1`, please check the corresponding `README.md` for more information.

## Training folder :

- follows the same logic as `dry_framework`, given N genreated datasets we do N distinct training runs, explaining why there are many checkpoint folders


## Evaluation folder : 
- ok, in this one we have a lot of evaluations so let's go step by step
- so we have $V$ variable diversity values, each one generated $C$ saved checkpoints, and then for the evalution, we have indistribution testing, the different out-of-distirubtion tasks, and finally, the renamed variable testing, where we take sucessful examples of indistirubiton testing, modify the snippets, and reevalute to see if perfromance holds, lets say we have in total $T$ tasks.
- so the total number of evaluations that we have to do is $V.C.T$, the `results_archive` and `results_extended` folders contain every single evaluation of these combinations, the naming follows this nomeclature : 
    - `eval_jobs1_checkpoint_c.pth_t_p_v` where c is the checkpoint number, t is the task name, and v is the percentage value
- let's break it down step by step to see how we managed to schedule all of these evaluations
- so in `scheduler.py`, we have `checkpoints` where we include all checkpoint names, `percentages` where we include all variable diversity values we generated and `tasks` where we specify all tasks 'in addition to indisturbtion' that we want to evaluate.
- then in in that same `scheduler.py` file, we structure all of these information given by the arrays to generate `jobs1` which is a list of jobs containing, information about: the checkpoint path, checkpoint number, variable diveristy value, .txt test file path, and finally test name, each item in `jobs1` is used to open a screen, and launch the evaluation script named `simple_arch.py` (equivalent to `eval.py` in `dry_framework` folder), and giving it all the necessary arguments based on the info given in the job item.
- all of this could be organised in a single `scheduler.py` file, but because we did not decide to run on all percentage values at once, we resulted in duplicating the same logic to create `scheduler20to50.py` and `scheduler80.py`.
- because we need to have the indistribution results before doing the 'replaced variable names' testing, we had to isolate the evaluation on replaced variable names separately in a `scheduler_semantic_preserving_edits.py` scheduler file, so first you have `semantic_preserving_edits_scripts` folder which contains all necessary scripts to take the results of indistribution evaluation, isolate the correctly predicted examples, then apply the variable renaming logic on them, producing new evaluation test sets saved in `semantic_preserving_edits_files_new`. Second, `scheduler_semantic_preserving_edits.py` launches all testing combinations (percentages X checkpoints), and saves the results. Third, to visualize all scores that we produce, we use the `viz.ipynb` script to autommatically explore all result folders, read the scores, and do the necessary computation to render progress charts.

## large_scale_validation

- Through this framework 'as reported in the mansucript' it has been shown that at a small scale, variable diversity is shown be a determining factor to the model's performance over snippets with replaced variables.
- So to validate this result at a larger scale, the `large_scale_validation` project enables us to do the following (in a nutshell):
    - train codebert architecture from scratch on an ALTERED version of Codesearchnet dataset (all variable names are replaced with a random sequence of characters)
    - evaluate the saved checkpoints + codebert baseline + contrabert model (from the litterature) on POJ-104 dataset (zero-shot) to measure regular performance
    - evaluate the same models but on an edited version of POJ-104 (with variables replaced with placeholders "e.g. var_01") to measure performance over snippets with renamed identifiers
- a very detailed `README.md` file within the project explains the exection process in detail
