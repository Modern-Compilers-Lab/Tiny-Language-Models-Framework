# Line execution counting guide

the task consists of the model predicting the count of how many lines of code will be executed given a random python code snippet


this folder contains .py and .ipynb files achieving the following goals :

## Data generation

[tinypy_generator.py](./tinypy_generator.py) is the main file that generates the dataset in the form of labeled python code snippets, this file is a modified version of the original [tinypy_generator](https://github.com/MarwaNair/TinyPy-Generator) which generates python code snippets and labels it with the output of that code.

for our case, we will keep the code snippet, and just change the label, instead of labeling with the code output, we will label with the line count, hence the modification.

the modification that you would notice after exploring that [file](./tinypy_generator.py) is that a new independent method has been added - given a python code snippet, the method returns its count -

for the sake of experiments, that same method has been written in a separate demonstrative .py file : [lineCounter.py](./lineCounter.py), the method accepts any functionning python code as an input.

a detailed explanation of how that method works is provided in the following [Docs](https://docs.google.com/document/d/1Fz0KGN1wb-6rVqU0BdrTBSaodM-pksPXhfoAGQbU7Dk/edit?usp=sharing) file.

## Data split

before moving on to finetuning, [prepare.py](./prepare.py) makes it possible to format the data generated previously, from a .txt file it splits the code examples into training, evaluation and test examples, saved in 3 separate files, in addition to generating a meta.pkl file that will help in the next stage "retrieve the needed information about the generated data"...

## Finetuning the model

[finetuning.ipynb](./finetuning.ipynb) contains the entire process "explained in the comments" that follows data generation :
- Data preparation "tokenization.. etc"
- Model structure definition "and lora implementation"
- Training "or finetuning if lora is activated"
- Evaluation