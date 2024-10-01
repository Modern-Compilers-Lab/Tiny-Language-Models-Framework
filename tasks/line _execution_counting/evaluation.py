import random
import os
import pickle
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import argparse
from finetuning_model import GPT

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--model_path', type=str, default=None)

args = parser.parse_args()

DATA_DIR = args.data_dir
MODEL_PATH = args.model_path
if not MODEL_PATH:
    print("Please provide a model path to evaluate.")
    exit()
meta_path = os.path.join(DATA_DIR, 'meta.pkl')

dic = torch.load(MODEL_PATH)
model = GPT()
model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
else:
    print("Meta file not found. Please ensure the meta.pkl file is present in the data directory.")

def encode(s):
    return [meta['stoi'][c] for c in s]

def decode(l):
    return ''.join([meta['itos'][i] for i in l])

test_data = np.memmap(os.path.join(DATA_DIR, 'test.bin'), dtype=np.uint16, mode='r')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate example "line execution counting"
def evaluate_example(model, example, max_new_tokens=30):
    
    # Split example and determine maximum new tokens allowed
    splited_example = example.split("# count")
    if not ("for" in splited_example[0]):
        max_new_tokens = 22
    # Encode prompt and prepare for evaluation    
    encoded_example = torch.tensor(encode(splited_example[0] + "# count"), dtype=torch.long).unsqueeze(0).to(device)
    prompt_text = splited_example[0] + "# count"
    result_example = splited_example[-1]
    
    # Extract real results from example
    real_results = [float(match.group()) for match in re.finditer(r"(?<=# )-?\d+(\.\d+)?", result_example.split('\n\n')[0].replace("\n", ""))]
    
    # Generate response from model and extract generated results
    try:
        response = decode(model.generate(encoded_example, max_new_tokens=max_new_tokens)[0].tolist())
        splited_response = response.split("# count")
        result_response = splited_response[-1]
        generated_results = [float(match.group()) for match in re.finditer(r"(?<=# )-?\d+(\.\d+)?", result_response.split('\n\n')[0].replace("\n", ""))]
    except:
        generated_results = "error"
    return prompt_text, real_results, generated_results



# Write results to file
def write_results_to_file(output_file, prompt, real_results, generated_results):
    df = pd.DataFrame({
        'Prompt': prompt,
        'Real_Results': real_results,
        'Generated_Results': generated_results
    })
    df.to_csv(output_file, index=False)

# Evaluation Loop

# Split examples and initialize lists for results
examples = decode(test_data).split("\n\n")
examples = [example for example in examples if example]
# Taking a subset of the examples for short "aimed for verification purposes" evaluations
example_subset = examples[:5000]
# Start evaluation process
prompt = []
real_results = []
generated_results = []

# Iterate through examples and evaluate the model on each one
for example in tqdm(example_subset):
    prompt_text, real_result, result = evaluate_example(model, example)
    prompt.append(prompt_text)
    real_results.append(real_result)
    generated_results.append(result)

# Calculate and print accuracy
correct_count = sum(1 for real, generated in zip(real_results, generated_results) if real == generated)
accuracy = correct_count / len(generated_results)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Store accuracy in a file
with open("accuracy.txt", 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

# Store predictions in a CSV file
    write_results_to_file("predictions.csv", prompt, real_results, generated_results)