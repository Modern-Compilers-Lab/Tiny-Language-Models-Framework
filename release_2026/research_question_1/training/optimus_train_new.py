import random
import os
import time
import torch
import datetime
import math
import inspect
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import sys
from pathlib import Path

# torchrun --standalone --nproc-per-node=4 optimus_train_new.py

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Tiny LLM Training')
parser.add_argument('--data-path', type=str, default=1, help='the dataset path')
args = parser.parse_args()
percentage = "80"
data_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_"+percentage+"_p"
script_dir = os.path.dirname(os.path.abspath(__file__))
base_seed = 1

# Device IDs
deviceids = [0,1,2,3]

# Check DDP status
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    print("ddp is set to True")
    from torch.distributed import init_process_group, destroy_process_group
    assert torch.cuda.is_available()
    init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    assert ddp_world_size == len(deviceids)
    device = f'cuda:{deviceids[ddp_rank]}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    print('ddp is set to False')
    assert len(deviceids) == 1
    master_process = True
    ddp_rank = 0
    seed_offset = 0
    ddp_world_size = 1
    device = f'cuda:{deviceids[ddp_rank]}'
    torch.cuda.set_device(device)

# -----------------------------------------------------------------------------
# Logging Utilities (FIXED TO PRINT TO CONSOLE)
# -----------------------------------------------------------------------------

if master_process:
    os.makedirs(f"{script_dir}/checkpoints{base_seed}_{percentage}/", exist_ok=True)
    log_file = open(f"{script_dir}/checkpoints{base_seed}_{percentage}/train.log", "w")
    pbar_recept_string = " " * 200 + "\n"
    log_file.write(pbar_recept_string * 2)
    log_file.flush()

    def log(s:str, p_level=None):
        if p_level == 1:
            # Write to file (top line)
            log_file.seek(0,0)
            log_file.write(pbar_recept_string)
            log_file.seek(0,0)
            log_file.write(s)
            log_file.seek(0,2)
            # Print to console (Updates in place)
            sys.stdout.write("\r" + s)
            sys.stdout.flush()
        elif p_level == 2:
            # Write to file (second line)
            log_file.seek(len(pbar_recept_string), 0)
            log_file.write(pbar_recept_string)
            log_file.seek(len(pbar_recept_string), 0)
            log_file.write(s)
            log_file.seek(0,2)
            # Print to console (Updates in place)
            sys.stdout.write("\r" + s)
            sys.stdout.flush()
        else:
            # Standard log
            log_file.write('\n' + s + '\n')
            # Clear the current line before printing new block
            sys.stdout.write("\r" + " " * 200 + "\r") 
            print(s)
        log_file.flush()

    def convert_seconds(seconds:float):
        seconds = int(seconds)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return (days, hours, minutes, seconds)

    batch_log = open(f"{script_dir}/batch_log.log", 'w')
else:
    def log(s, p_level=None): pass

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

DDIR = data_path + '/'
seed = base_seed + seed_offset
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Hyperparams
block_size = 512
n_embd     = 384   
n_head     = 16    
n_layer    = 12    
dropout    = 0.0     

# Vocab Size
with open(DDIR+"vocab_size.txt", "rb") as f:
    vocab_size = int(f.read())
if master_process: log(f"vocab_size: {vocab_size}")

# Load Data
if master_process: log("Loading train.bin")
before = time.time()
train_data_mm = np.memmap(DDIR+"train.bin", dtype=np.uint8, mode="r")
train_data = np.array(train_data_mm) # Loading into RAM
after = time.time()
if master_process: log(f"took {convert_seconds(after - before)}")

if master_process: log("Loading val.bin")
before = time.time()
val_data = np.memmap(DDIR+"val.bin", dtype=np.uint8, mode="r")
#val_data = np.array(val_data_mm) # Loading into RAM
after = time.time()
if master_process: log(f"took {convert_seconds(after - before)}")

# Training Hyperparams
batch_size = 512
batch_nb_tokens = batch_size * block_size
assert batch_size % len(deviceids) == 0 
microbatch_size = batch_size // len(deviceids) 
microbatch_nb_tokens = microbatch_size * block_size 
acum_steps = 1
assert microbatch_size % acum_steps == 0

max_pseudo_epochs = 1 
max_iters = int((max_pseudo_epochs * len(train_data)) / (batch_size * block_size))
epoch_iters = int(len(train_data) / (batch_size * block_size))

learning_rate = 1e-3    
decay_lr = True
warmup_iters = int(0.1 * max_iters)
lr_decay_iters = max_iters
assert lr_decay_iters <= max_iters
assert warmup_iters <= lr_decay_iters
min_lr = learning_rate * 0.1
beta1 = 0.9 
beta2 = 0.95 
weight_decay = 1e-1 
grad_clip = 1.0 
cuda_sync = False
compile = True
compile_mode = 'default'

eval_interval = int(max_iters / 24)
eval_iters = 100 
eval_batch_size = 128
eval_batch_nb_tokens = eval_batch_size * block_size

# -----------------------------------------------------------------------------
# Data Utilities
# -----------------------------------------------------------------------------

def get_batch(split):
	data = train_data if split == 'train' else val_data
	# This is no curriculum so we are just going to sample a random index
	idx = random.randint(0, len(data) - microbatch_nb_tokens - 1)
	
	x = torch.from_numpy((data[idx:idx+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
	y = torch.from_numpy((data[idx+1:idx+1+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
	x, y = x.to(device), y.to(device)

	return x, y

def load_eval_batch(split):
	data = train_data if split == 'train' else val_data
	idx = random.randint(0, len(data) - eval_batch_nb_tokens - 1)
	x = torch.from_numpy((data[idx:idx+eval_batch_nb_tokens]).astype(np.int64)).view(eval_batch_size, block_size)
	y = torch.from_numpy((data[idx+1:idx+1+eval_batch_nb_tokens]).astype(np.int64)).view(eval_batch_size, block_size)
	x, y = x.to(device), y.to(device)
	return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_start_time = time.time()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            past = time.time()
            X, Y = load_eval_batch(split)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            present = time.time()
            if master_process:
                log(f"{split}>|ITERS: {k+1}/{eval_iters}|RATE: {1/(present-past):.2f} it/s", p_level=2)
        out[split] = losses.mean()
    model.train()
    return out

# Define function to make large numbers of parameters human-readable
def human_readable(num):
	magnitude = 0
	while abs(num) >= 1000:
		magnitude += 1
		num /= 1000.0
	return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Optimized ALiBi Utils
# -----------------------------------------------------------------------------

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])

# -----------------------------------------------------------------------------
# Optimized Model Architecture
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.dropout_p = dropout
        
        # Batched QKV
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi setup
        self.register_buffer("m", get_alibi_slope(n_head).view(1, n_head, 1, 1))
        
        # Flash check
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.c_attn(x)
        q, k, v = qkv.view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        
        if self.flash:
            x_pos = torch.arange(T, device=x.device)
            y_pos = torch.arange(T, device=x.device)
            relative_pos = x_pos[None, :] - y_pos[:, None] 
            dist = -torch.abs(relative_pos).view(1, 1, T, T)
            alibi_bias = self.m * dist
            
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            causal_mask = causal_mask.view(1, 1, T, T)
            attn_mask = alibi_bias + causal_mask

            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0,
                is_causal=False 
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            x_pos = torch.arange(T, device=x.device)
            y_pos = torch.arange(T, device=x.device)
            relative_pos = x_pos[None, :] - y_pos[:, None]
            dist = -torch.abs(relative_pos).view(1, 1, T, T)
            alibi_bias = self.m * dist
            
            att = att + alibi_bias
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.RMSNorm(normalized_shape=n_embd, eps=1e-5)
        self.ln2 = nn.RMSNorm(normalized_shape=n_embd, eps=1e-5)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # No position embedding (ALiBi)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        x = tok_emb 
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# -----------------------------------------------------------------------------
# Init Models & Optimizer
# -----------------------------------------------------------------------------

model = GPT()
model.to(device)


# --- INSERT START ---
# Define your checkpoint path here
load_checkpoint_path = "None"

if os.path.exists(load_checkpoint_path):
    if master_process:
        print(f"Loading model weights from {load_checkpoint_path}")
    # Load weights (map_location is crucial for DDP/GPU handling)
    state_dict = torch.load(load_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    if master_process:
        print(f"No checkpoint found at {load_checkpoint_path}, training from scratch.")
# --- INSERT END ---

if master_process:
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_parameters_hr = human_readable(num_parameters)
    print(f'The model has {num_parameters_hr} trainable parameters')
    log(f'The model has {num_parameters_hr} trainable parameters')

train_model = model

if ddp:
    train_model = DDP(train_model, device_ids=[deviceids[ddp_rank]])

if compile:
    if master_process: log(f"Compiling model with mode {compile_mode}...")
    train_model = torch.compile(train_model, mode=compile_mode)

# Optimizer
param_dict = {pn: p for pn, p in train_model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and ('cuda' in device)
extra_args = dict(fused=True) if use_fused else dict()
if master_process: print(f"using fused AdamW: {use_fused}")

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)

# --- INSERT OPTIMIZER LOAD ---
load_optim_path = "None"
if os.path.exists(load_optim_path):
    if master_process: print(f"Loading optimizer state from {load_optim_path}")
    optimizer.load_state_dict(torch.load(load_optim_path, map_location=device))
# -----------------------------

torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

train_start_time = time.time()

if master_process:
    log("Evaluating initial loss")
    last_losses = estimate_loss()
    log(f'INITIAL LOSS: train {last_losses["train"]:.4f} | val {last_losses["val"]:.4f}')
    best_val_loss = last_losses["val"]

print('Current Device:', torch.cuda.current_device())

nb_time_samples = 0
deltatimes = 0
tl_warmup_iters = 0
tl_max_warmup_iters = 20

for iter in range(0, max_iters):
    past = time.time()
    
    optimizer.zero_grad(set_to_none=True)
    
    xb, yb = get_batch('train')

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = train_model(xb, yb)

    loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), grad_clip)
    
    micro_loss = loss.detach()
    if ddp:
        dist.all_reduce(micro_loss, op=dist.ReduceOp.AVG)

    # Scheduler
    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()

    if cuda_sync:
        torch.cuda.synchronize(device)

    present = time.time()

    # Logging
    if master_process:
        dt = present - past
        if tl_warmup_iters <= tl_max_warmup_iters:
            tl_warmup_iters += 1
            log_dt = dt
        else:
            deltatimes += dt
            nb_time_samples += 1
            log_dt = deltatimes / nb_time_samples
        
        msg = (f"|LOSS: {micro_loss:.4f} | TPS: {int(batch_size*block_size/dt):8} "
               f"| ITERS: {iter+1}/{max_iters} | COMP: {(iter+1)/max_iters:.1%} "
               f"| LR: {lr:.2e} | SPD: {dt*1000:.1f}ms | ET: {convert_seconds((max_iters-iter)*log_dt)}")
        
        log(msg, p_level=1)
        batch_log.write(f"{micro_loss:.5f}\n")

    # Evaluation
    if ((iter+1) % eval_interval == 0 or (iter == max_iters - 1)) and master_process:
        log("\nCheckpointing & Evaluating...\n")
        epoch = (block_size * batch_size * (iter+1))/len(train_data)
        
        save_path = f"{script_dir}/checkpoints{base_seed}_{percentage}"
        torch.save(model.state_dict(), f"{save_path}/checkpoint_{epoch:.2f}.pth")
        torch.save(optimizer.state_dict(), f"{save_path}/optimizer_{epoch:.2f}.pth")

        losses = estimate_loss()
        
        log(f'ITER {iter+1} | train {losses["train"]:.4f} | val {losses["val"]:.4f}')

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), f"{save_path}/best-model.pth")

if ddp:
    destroy_process_group()

if master_process:
    batch_log.close()
    log_file.close()