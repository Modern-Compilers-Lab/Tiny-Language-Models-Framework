# 1. Environment setup
Before starting anything we have to install the dependencies needed to run the project, using one of these two ways

## 1.1 using conda's environement.yml upload

in your project path, create a fresh environement with the dependencies we want using this command : \
``` conda env create -f environment.yml ``` 

## 1.2 using python's venv (the requirements.txt way)
in your project path, create a fresh environement with the dependencies we want using the following commands :

```terminal
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
``` 


**IMPORTANT** : whether you used method 1.1 or 1.2, you must make absolutely sure that the new environement you 
created is set to the default environement, this project relies on 'screens' and launching processes in separate sessions
so when those processes wake up, they must be in the same environement we are preparing right now, so just set it to default.



# 2. TinyTracer project

the project consists of 3 sequential steps

## 2.1 `data_generation`
this step basically hosts all data generation related scripts, we need it mainly for : 
- generating the training dataset (in distribution)
- generating the test dataset (in distribution and out of distribution)
- saving all generated datasets in subfolders under `data_generation`

after running the main 3 scripts of `data_generation` you will have a folder containing the following : 
- `raw_id.txt` file containing all generated snippets
- `stats.txt` file containing general stats about the `raw_id.txt` file (distribution of variable length, how many generated snippets ..etc) (some numbers are not representative of the actual dataset because they have been zeroed out for debugging reasons)
- `determinism_filtered_snippets.txt` file containing snippets that passed the determinsim filtering test
- `oversize_snippets.txt` file containing snippets that did not pass the determinsim filtering test
- `train.bin` a fragment of the dataset mainly used for training (binary file)
- `test.bin`/`test.txt` a fragement of the dataset mainly used for testing (binary and text files)
- `val.bin`/`val.txt` a fragement of the datset mainly used for validation (binary and text files)
- `vocab_size.txt` vocabulary size written as a single number in a text file

Correclty generating the data requires following this order of execution

1. `tinypy_code_tracing_generator.py` generates one big txt file containing all snippets following specific generation rules, produced files : `raw_id.txt`, `stats.txt`
2. `determinism_filtering.py` scans through `raw_id.txt` and picks only snippets that pass the determisim filtering test, produced files : `determinism_filtered_snippets.txt`, `oversize_snippets.txt` 
3. `data_preparation.py` : from `determinism_filtered_snippets.txt`, this script fragments the files into train, validation and test sets, note that all of them follow the same distribution, that is same genreation rules, splittig is done according to rules set in that same .py file, produced files : `train.bin`,`test.bin`/`test.txt`, `val.bin`/`val.txt`, `vocab_size.txt`

**Note** : hyperparameters like dataset size or some generation rules are accessible in each individual script.

## 2.2 `training`

this step will be responsible of training a language model from scratch on a generated dataset, it contains a single script : `optimus_train.py`

this script takes a dataset folder generated in the previous step (2.1) and trains a language model from scratch, so if the name of the dataset is `dataset1`,
the script would use `./data_generation/dataset1/train.bin` as the training binary file and `./data_generation/dataset1/val.bin` as validation file.

produced files : a folder containing all the saved checkpoints during training + a `best-model.pth` file which points to the best checkpoint

**Note** : hyperparameters like seeds, number of checkpoints, gpus used ..etc are accessible in the `optimus_train.py` file

## 2.3 `evaluation`

this step will take the checkpoints produced in `training` and evaluate them on a set of test datasets, it can be used for example : 

- to test on in-distribution data, if the model was trained on `./data_generation/dataset1/train.bin` then we would evaluate on `./data_generation/dataset1/test.txt`
- to test on out-of-distribution data, if the model was trained on `./data_generation/dataset1/train.bin` then we would evaluate on `./data_generation/NOTdataset1/test.txt`

the output is a folder containing a list of evaluations of several checkpoints on a specific test dataset

the folder contains the following scripts : 

- `eval.py` : an atomic evaluation script that evaluates a single checkpoint on a single dataset
- `scheduler.py` : an orchestrator that launches many `eval.py` processes in parallel, to take into account multiple checkpoints or multiple test sets

**Note** : hyperparameters like which test datasets to use, what gpus to use ..etc are accessible in the `scheduler.py` file




**Important** : for the third step of model evaluation, although the terminal might show that it has been executed succesfully, that does not necessarily mean that is done, in fact, screens in the background may be still running, and you might want to use `screen -ls` command to check whether evaluations are still running in the background or not


## 2.4 recap of scripts execution order :

`tinypy_code_tracing_generator.py` -> `determinism_filtering.py` -> `data_preparation.py` -> `optimus_train.py` -> `scheduler.py`