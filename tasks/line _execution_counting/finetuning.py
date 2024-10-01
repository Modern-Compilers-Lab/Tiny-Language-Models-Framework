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
from finetuning_model import GPT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--model_dir', type=str, default='./models')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--lora_rank', type=int, default=12)

args = parser.parse_args()

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed) 
random.seed(seed)
np.random.seed(seed)

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set to {device}.")

# Helper functions to load and save data
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    
# Directory where the data is stored "must contain 4 files : train.txt, val.txt, test.txt and a meta.pkl file"
DATA_DIR = args.data_dir
# Directory where the model is stored
MODEL_DIR = args.model_dir

# Attempt to derive vocab_size from the dataset

meta_path = os.path.join(DATA_DIR, 'meta.pkl')
vocab_size = None

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"found vocab_size = {vocab_size} (inside {meta_path})")
else:
    print("Meta file not found. Please ensure the meta.pkl file is present in the data directory.")

# Encode and decode functions for character-level Tokenzation 
def encode(s):
    return [meta['stoi'][c] for c in s]

def decode(l):
    return ''.join([meta['itos'][i] for i in l])

# Load data
train_data = load_data(os.path.join(DATA_DIR, 'train.txt'))
val_data = load_data(os.path.join(DATA_DIR, 'val.txt'))
test_data = load_data(os.path.join(DATA_DIR, 'test.txt'))

# Encode data
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

# Save encoded data to bin files, make sure to choose "Files only" on the persistence option of the session so that you don't encode data each time
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile( 'train.bin')
val_ids.tofile( 'val.bin')
test_ids.tofile('test.bin')

print("Encoded data saved as binary files.")

del(train_ids)
del(val_ids)
del(test_ids)

# Load encoded data
train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

##### Model #####

# Hyperparameters for the GPT model
block_size = 256  # Maximum context length
n_embd = 372      # Embedding dimension
n_head = 6        # Number of attention heads
n_layer = 6       # Number of transformer blocks
dropout = 0       # Dropout rate
batch_size = 64   # Batch size for training
max_iters = 100_000  # Maximum number of iterations
learning_rate = 1e-3 # Initial Learning rate value
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]  # Milestones for learning rate decay as fractions of max_iters
eval_interval = 10_000 # Evaluation interval
eval_iters = 1000 # Number of iterations for evaluation
vocab_size = 53 # Vocabulary size

# Model to be fine-tuned "set the model name without .pth" (Keep it empty for training from scratch)
model_name = args.model_name

# LoRA Rank - Set it to 0 if you want to train from scratch or perform full fine-tuning
lora_r = args.lora_rank

compile = False

print(f"Data in tokens: {len(train_data)}")
iters4epoch = len(train_data)//(batch_size * block_size)
print(f"Number of iters for one pseudo-epoch : {iters4epoch}")
print(f"Number of pseudo-epochs : {max_iters / iters4epoch:.2f}")

# Get random batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss on train and val splits
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Helper function to make large numbers of parameters human-readable
def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# load the language model
def load_model():
        """
        Load pre-trained model based on the provided model name.
        """
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        model = GPT()
        print("Compiling the model...\n")
        r = -1
        if compile:
            try:
                model = torch.compile(model)  # requires PyTorch 2.0
            except Exception as e:
                pass

            checkpoint = torch.load(model_path, map_location=device)
            if 'lora_rank' in checkpoint.keys():
                r = checkpoint['lora_rank']
                state = checkpoint['state_dict']

                if r > 0:
                    model.activate_lora(r)
                model.load_state_dict(state)
            else:
                model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(model_path, map_location=device)
            if 'lora_rank' in checkpoint.keys():
                r = checkpoint['lora_rank']
                state_dict = checkpoint['state_dict']

                if r > 0:
                    model.activate_lora(r)
            else:
                state_dict = checkpoint
            
            state_dict_keys = map(lambda x: x.replace("_orig_mod.", ""), state_dict.keys())
            state_dict = dict(zip(state_dict_keys, state_dict.values()))
            model.load_state_dict(state_dict)

        m = model.to(device)
        return m, (r > 0)

# Initialize model and move it to the device (GPU)
if len(model_name) > 0:
    print("Loading model...\n")
    model, r_exists = load_model()

else:
    model = GPT()
    m = model.to(device)
    r_exists = False

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

if lora_r > 0 and not r_exists:
    print("Activating LoRA...")
    model.activate_lora(lora_r)
    model = model.to(device)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_parameters_hr = human_readable(num_parameters)
print(f'The model has {num_parameters_hr} trainable parameters')

### Training ###

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)

# Get current date and hour to get track of experiments
now = datetime.datetime.now()
date_hour = now.strftime("%Y-%m-%d_%H-%M")

# Train
# Start training timer
start_time = time.time()

# Training loop
for iter in tqdm(range(max_iters)):

    # evaluate the model on the train and val splits and log the losses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
        
    # train the model for one iteration
    xb, yb = get_batch('train')

    # forward passd
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    #loss.requires_grad = True
    loss.backward()
    optimizer.step()

    # Step the scheduler
    scheduler.step()

# End training timer
end_time = time.time()
print(f'Training time: {(end_time - start_time) / 60}  min')

# Save the trained model
model_path = f"{num_parameters_hr}_{date_hour}.pth"
checkpoint = {
    'lora_rank': model.lora_rank if(hasattr(model, "lora_rank")) else -1,
    'state_dict': model.state_dict()
}

torch.save(checkpoint, model_path)
print(f"Model saved to {model_path}\n")
