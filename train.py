import os
import time
import datetime
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import GPT

# Argument parsing
parser = argparse.ArgumentParser(description='Training script for NanoGPT model.')
# Define command-line arguments
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--max_iters', type=int, default=35000, help='Maximum number of iterations')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--miles', type=float, nargs='+', default=[0.7, 0.8, 0.9], help='Milestones for learning rate decay as fractions of max_iters')
parser.add_argument('--eval_interval', type=int, default=10000, help='Evaluation interval')
parser.add_argument('--eval_iters', type=int, default=500, help='Number of iterations for evaluation')
parser.add_argument('--data_dir', type=str, default='data', help='Directory where the data is stored')

# Parse the command-line arguments
args = parser.parse_args()

# Use the parsed arguments
batch_size = args.batch_size
max_iters = args.max_iters
learning_rate = args.learning_rate
# Calculate milestone iterations based on fractions
miles = [int(max_iters * m) for m in args.miles]
eval_interval = args.eval_interval
eval_iters = args.eval_iters
data_dir = args.data_dir

# Set directories for data and models
DATA_DIR = data_dir
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Other settings
block_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout = 0
compile = True

# wandb settings
wandb_log = True
wandb_project = 'TinyLanguageModel'

# For logging purposes
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# Set random seed for reproducibility
torch.manual_seed(1337)

print(f"Loading training data from {DATA_DIR}...\n")
train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(DATA_DIR, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})\n")

# Function to get random batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Function to calculate loss on train and validation splits
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

# Initialize model and move it to the device (GPU/CPU)
print("Initializing the model...\n")
model = GPT()
m = model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Compile the model if enabled
if compile:
    print("Compiling the model... (takes a ~minute)\n")
    try:
        model = torch.compile(model)  # requires PyTorch 2.0
    except Exception as e:
        pass

# Helper function to make large numbers of parameters human-readable
def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

num_parameters_hr = human_readable(num_parameters)
print(f'The model has {num_parameters_hr} trainable parameters\n')

# Get current date and time
now = datetime.datetime.now()
date_hour = now.strftime("%Y-%m-%d_%H-%M")

# Construct wandb run name
wandb_run_name = f'TLM_RUN_{num_parameters_hr}_{date_hour}'

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)

# Initialize wandb if logging is enabled
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    print(f"wandb logging is enabled. Project: {wandb_project}, Run name: {wandb_run_name}\n")

# Start training timer
start_time = time.time()
print("Starting training...\n")

# Training loop
for iter in range(max_iters):
    # Evaluate the model on the train and val splits and log the losses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
        if wandb_log:
            wandb.log({
                "iter": iter,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": scheduler.get_last_lr()[0],
            })

    # Train the model for one iteration
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Step the scheduler
    scheduler.step()



# End training timer
end_time = time.time()
print(f'Training complete! Total time: {(end_time - start_time) / 60:.2f} minutes\n')

# Save the trained model
model_path = f"{MODELS_DIR}/{num_parameters_hr}_{date_hour}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}\n")
