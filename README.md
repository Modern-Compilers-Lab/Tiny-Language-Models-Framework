# Tiny Language Models Framework

This repository contains the implementation and resources for the Tiny Language Models Framework project. In this project, we developed small-scale language models to facilitate detailed research into various aspects of large language models (LLMs), particularly in the domain of code.

<p align="center">
  <img src="https://github.com/Modern-Compilers-Lab/Tiny-Language-Models-Framework/assets/86785811/946011ac-90ca-454f-baeb-d74b09a1721c" width="1000" >
</p>

## Project Structure

- `data/`
  - `meta.pkl` : Metadata for the dataset.
  - `prepare.py` : Script to prepare data for training.
  - `sample_data.txt` : Sample data used for testing and demonstration.
  - `test.bin` : Binary file containing test data.
  - `test.txt` : Text file containing test data.
  - `tinypy_generator.py` : Script to generate TinyPy data.
  - `train.bin` : Binary file containing training data.
  - `train.txt` : Text file containing training data.
  - `val.bin` : Binary file containing validation data.
  - `val.txt` : Text file containing validation data.
 
- `generalization/`
  - `data/` : Contains tokenized data to fine-tune and evaluate Code LLaMa model.
  - `models/` : Stores fine-tuned Code LLaMa models.
  - `results/` : Holds results from the evaluation.
  - `demonstration.ipynb` : Jupyter notebook demonstrating fine-tuned Code LLaMa capabilities.
  - `evaluate.py` : Script to evaluate fine-tuned Code LLaMa.
  - `finetune.py` : Script for fine-tuning Code LLaMa model.
  - `tokenizing.py` : Handles tokenization for Code LLaMa model.

- `models/`
  - `arithmetics_level1_696K.pth` : Pretrained model for arithmetic operations at level 1 with 696K parameters.

- `results/`
  - Directory to store results of model evaluations and tests.

- `demonstration.ipynb` : Jupyter notebook demonstrating the usage of the models and scripts.

- `eval.py` : Script to evaluate the trained models.

- `model.py` : Contains the model architecture and related functions.

- `README.md` : This file.

- `train.py` : Script to train the models.

## Requirements

To install the required packages, you can use the following:

```bash
pip install -r requirements.txt
```

## Usage

### Data Generation
Generate the data using TinyPy Generator by running : 

```bash
cd data/
python tinypy_generator.py --num_programs 1000 --level 1.1 --filename sample_data.txt --deduplicate
```

### Data Preparation
Prepare the data by running:

```bash
python prepare.py
```

This generation command is just an example to get you started. If you want to train your own model, you'll likely need to generate significantly more data. 

### Training
Train the model using the following command:

bash
```bash
cd ..
python train.py --batch_size 64 --max_iters 35000 --learning_rate 0.01 --miles 0.7 0.8 0.9 --eval_interval 10000 --eval_iters 500 --data_dir data
```

### Evaluation
Evaluate the trained model by running:

```bash
python eval.py --dataset_dir data --model_name arithmetics_level1_696K
```

### Demonstration
To see a demonstration of the model's capabilities, open the demonstration.ipynb notebook and follow the instructions within.

### Generalization
This section aims to generalize the results obtained from training tiny language models to large language models. This can be achieved through fine-tuning Code LLaMa.

#### Fine-tuning
Fine-tune Code LLaMa model using the following command:

```bash
cd generalization/
python finetune.py  --train_dataset_path data/tokenized_train --val_dataset_path data/tokenized_val --output_dir models/code-llama-finetuned-demo
```

#### Evaluation
Evaluate the fine-tuned Code LLaMa model by running:

```bash
python evaluate.py --checkpoint_dir models/code-llama-finetuned-level1 --test_file data/test.txt --output_file results/result_llama.txt --csv_file results/results_llama.csv 
```

#### Demonstration
To see a demonstration of the model's capabilities, open the generalization/demonstration.ipynb notebook and follow the instructions within.


# License
This project is licensed under the MIT License.

#  Acknowledgements
This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise.
