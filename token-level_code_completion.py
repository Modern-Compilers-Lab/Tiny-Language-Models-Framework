import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from model import GPT


# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate NanoGPT model on token-level code completion.')
parser.add_argument('--dataset_dir', type=str, default='data', help='Directory where the dataset is stored')
parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model (without .pth extension)')

# Parse the command-line arguments
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1337)


# Constants for dataset and file paths
MODEL_FILE = f"models/{args.model_name}.pth"
ACCURACY_FILE = f"results/{args.model_name}_acc_token-level_code_completion.txt"
RESULTS_FILE = f"results/{args.model_name}_token-level_code_completion.csv"


data_dir = args.dataset_dir
test_data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

model = GPT()  
print("Compiling model...")
model = torch.compile(model) # pytorch 2.0
model.load_state_dict(torch.load(MODEL_FILE))
m = model.to(device)

examples = decode(test_data).split("\n\n")
examples = [example for example in examples if example]

correct_predictions = 0
total_predictions = 0

results = []

for code_snippet in tqdm(examples):
    
    tokens = torch.tensor(encode(code_snippet), dtype=torch.long).unsqueeze(0).to(device)
    
    for i in range(1, tokens.shape[1]):
        
        context = tokens[:, :i]
        actual_next_token = tokens[:, i].item()
        predicted_next_token = m.generate(context, max_new_tokens=1)
        predicted_next_token = predicted_next_token[:, -1].item()
        is_correct = (predicted_next_token == actual_next_token)
        
        if is_correct:
            correct_predictions += 1
        results.append({
            'context': context.cpu(),
            'actual_next_token': actual_next_token,
            'predicted_next_token': predicted_next_token,
            'is_correct': is_correct
        })
        
        total_predictions += 1

df = pd.DataFrame(results)
df.to_csv(RESULTS_FILE, index=False)


accuracy = (correct_predictions / total_predictions) * 100

# Store accuracy in a file
with open(ACCURACY_FILE, 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}%\n")

print(accuracy)