# 2026 release

This folder contains the source code that produced the results reported in the manuscript, the organisation is as follows : 
- We start with a simplest version of the framework, located in `dry_framework`, that lets the user get familiar with the different steps of conducting any experiment
- After getting comfortable with the framework, the user can explore `research_question_1` and `research_question_2`, which are applications of the framework on different research questions
- It is recommended to use `dry_framework` as the main "usable" project, then add/modify configurations from scripts present in `research_question_1` and `research_question_2`
- Every folder contains a `README.md` file explaining the scripts at a lower level, `dry_framework` scripts are commented with the most important hyperparameters, most used commands, etc.
- The full explanation behind the framework can be found in the corresponding manuscript
- `release_2026.zip` file is a copy of `release_2026`, but it contains all the empty folders necessary to explain the different dataset generations, model trainings, and model evaluations conducted (git does not track empty folders by default). 
- Excluding the .zip file, The output folders (datasets, checkpoints, inference and evaluation results) are **not** shipped in the repository. They are regenerated on demand at runtime by the scripts (via `os.makedirs(..., exist_ok=True)`), and are ignored through the root `.gitignore`. Everything is reproducible.


Contact: [ia2921@nyu.edu](mailto:ia2921@nyu.edu)
