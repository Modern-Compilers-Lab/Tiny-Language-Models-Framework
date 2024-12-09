# Model training
XP = 'XP-3-5'
DDIR = "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-11/datapreps-11/dataprep-11-1/data-dp-11-1/"
deviceid = 5

## Logging boilerplate
log_file = open("model-training-atmpt-1.log", "w")
# progress bar reception string
pbar_recept_string = " " * 200 + "\n"
log_file.write(pbar_recept_string)
log_file.write(pbar_recept_string)
log_file.flush()
def log(s:str, p_level=None):
	if p_level == 1:
		log_file.seek(0,0)
		log_file.write(pbar_recept_string)
		log_file.seek(0,0)
		log_file.write(s)
		log_file.seek(0,2)
	elif p_level == 2:
		log_file.seek(len(pbar_recept_string), 0)
		log_file.write(pbar_recept_string)
		log_file.seek(len(pbar_recept_string), 0)
		log_file.write(s)
		log_file.seek(0,2)
	else:
		if s[0].upper() == s[0]:
			start = "\n"
			end = ":"
		else:
			start = "	--> "
			end = ""
		log_file.write(start + s + end + "\n")
	log_file.flush()

## Convert seconds to days, hours, minutes, seconds
def convert_seconds(seconds:float):
	# ignoring the sub seconds 
	seconds = int(seconds)
	days, seconds = divmod(seconds, 86400)
	hours, seconds = divmod(seconds, 3600)
	minutes, seconds = divmod(seconds, 60)
	return (days, hours, minutes, seconds)

log("Importing ...")
import time

before = time.time()
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
after = time.time()
log(f"took {convert_seconds(after - before)}")

## Starting the netpune logging
log("Starting the netpune logging")
log("neptune init")
import neptune
run = neptune.init_run(
	project="younes-boukacem-workspace/Tiny-Language-Models-Framework",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZGFmNDg2Yy01MzRkLTQwNGMtYmZmMy1hYzM0Y2NkY2QyNmMifQ==",
	capture_hardware_metrics = False,
	custom_run_id = XP
)
# First attempt so we log the runid
log("saving the runid")
runid = run["sys/id"].fetch()
with open("runid.txt", "w") as f:
	f.write(runid)


## Set the random seed for reproducibility
log("Set the random seed for reproducibility")
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


## Set the device to GPU if available, otherwise CPU
log("Set the device to GPU if available, otherwise CPU")
device = torch.device(f'cuda:{deviceid}' if torch.cuda.is_available() else 'cpu')
log(f"device set to {device}.")


## Setting arch-hyperparams for the GPT model
log("Setting arch-hyperparams for the GPT model")
run["arch-hyperparams/block_size"] = block_size = 256  # Maximum context length
run["arch-hyperparams/n_embd"] = n_embd = 464	  # Embedding dimension
run["arch-hyperparams/n_head"] = n_head = 16		# Number of attention heads
run["arch-hyperparams/n_layer"] = n_layer = 24	   # Number of transformer blocks

# Loading the training and evaluation data
log("Loading the training and evaluation data")
log("train.bin")
before = time.time()
train_data = np.memmap(DDIR+"train.bin", dtype = np.uint16, mode="r")
train_data = np.array(train_data)
after = time.time()
log(f"took {convert_seconds(after - before)}")

log("val.bin")
before = time.time()
val_data = np.memmap(DDIR+"val.bin", dtype = np.uint16, mode="r")
val_data = np.array(val_data)
after = time.time()
log(f"took {convert_seconds(after - before)}")

# Setting the train-hyperparams and util variables
log("Setting train-hyperparams and util variables")
run["train-hyperparams/batch_size"] = batch_size = 64   # Batch size for training
run["train-hyperparams/dropout"] = dropout = 0	   # Dropout rate
run["train-hyperparams/max_pseudo_epochs"] = max_pseudo_epochs = 1.7
run["train-hyperparams/learning_rate"] = learning_rate = 1e-3 # Initial Learning rate value
run["train-hyperparams/max_degradations"] = max_degradations = -1 # number of consecutive degradations on val loss before stoping the training
eval_interval = 5000 # Evaluation interval
eval_iters = 500  # Number of iterations for evaluation
max_iters = int( ( max_pseudo_epochs * len(train_data) ) / ( batch_size * block_size ) )
log(f"max_iters = {max_iters}")
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]  # Milestones for learning rate decay as fractions of max_iters
run["train-hyperparams/miles"] = str(miles)



compile = False # requires PyTorch 2.0

## Defining the model and utilities
log("Defining the model and utilities")
log("The model")

class Head(nn.Module):
	"""One head of self-attention."""

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B,T,C = x.shape
		k = self.key(x)   # (B, T, 16)
		q = self.query(x) # (B, T, 16)
		v = self.value(x)
		
		out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0, is_causal=True)
			
		return out

class MultiHeadAttention(nn.Module):
	"""multiple heads of self-attention in parallel."""

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out
	
class FeedForward(nn.Module):
	""" a simple linear layer followed by a non-linearity."""

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd, bias=False),
			nn.SiLU(),
			nn.Linear( 4 * n_embd, n_embd, bias=False),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)
	
class Block(nn.Module):
	""" Transformer block: communication followed by feedforward."""

	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
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
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd, bias=False) 
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def forward(self, idx, targets=None):
		B, T = idx.shape

		# idx and targets are both (B,T) tensor of integers
		tok_emb = self.token_embedding_table(idx) # (B,T,C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
		x = tok_emb + pos_emb # (B,T,C)
		x = self.blocks(x) # (B,T,C)
		x = self.ln_f(x) # (B,T,C)
		logits = self.lm_head(x) # (B,T,vocab_size)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss
	
	def generate(self, idx, max_new_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop idx to the last block_size tokens
			idx_cond = idx[:, -block_size:] # (B, T)
			# get the predictions
			logits, loss = self(idx_cond)
			# focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # (B, C)
			# sample from the distribution
			_, idx_next = torch.max(probs, dim=1, keepdim=True) # (B, 1)
			#idx_next = torch.multinomial(probs, num_samples=1)
			# append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx

# get random batch of data
log("def get random batch of data")
def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
	y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

# estimate loss on train and val splits
log("def estimate loss")
@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			past = time.time() 
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
			present = time.time()
			log(f"{split}>|ITERS: {k+1} / {eval_iters} | COMP: {(k+1)/eval_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((eval_iters-k-1) * (present-past))} |", p_level = 2)
		out[split] = losses.mean()
	model.train()
	return out

# helper function to make large numbers of parameters human-readable
log("def human readable")
def human_readable(num):
	magnitude = 0
	while abs(num) >= 1000:
		magnitude += 1
		num /= 1000.0
	return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


## Loading the meta object
log("loading vocab_size")
with open(DDIR+"vocab_size.txt", "rb") as f:
	vocab_size = int(f.read())


## Creating a new model
log("Creating the model")
model = GPT()
m = model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
if compile:
	print("compiling the model... (takes a ~minute)")
	model = torch.compile(model) 
num_parameters_hr = human_readable(num_parameters)
log(f'The model has {num_parameters_hr} trainable parameters')


## Preparing for the training loop
log("Preparing for the training loop")

# initializing the optimizer
log("initialiazing the optimizer")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# initializing the learning rate scheduler
log("initializing the learing rate scheduler")
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)


# computing the initial loss
log("computing the initial loss")
losses = estimate_loss()

# saving the last_losses for early stopping
log("saving the last loss for early stopping")
last_losses = losses
best_val_loss = losses["val"]

# neptune logging the initial loss
log("neptune logging the initial loss")
run["losses_trace/train_loss"].append(losses["train"], step = 0)
run["losses_trace/val_loss"].append(losses["val"], step = 0)


## Training loop
log("Training loop")

import os
os.makedirs(os.path.dirname("checkpoints/"), exist_ok=True)

log("==========================================================================================")
early_stopping = {"state": False, "iter": None, "epoch": None}
now = datetime.datetime.now()
date_hour = now.strftime("%Y-%m-%d_%H-%M")
log(f'{date_hour} : iter {0:5d} <=> epoch 0 | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
nb_degradations = 0


log("training ...")
for iter in range(max_iters):
	past = time.time()
	# train the model for one iteration
	xb, yb = get_batch('train')
	# forward pass
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
	# Step the scheduler
	scheduler.step()

	present = time.time()
	log(f"|ITERS: {iter+1} / {max_iters} | COMP: {(iter+1)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((max_iters-iter-1) * (present-past))}", p_level = 1)
	
	# evaluate the model on the train and val splits and log the losses
	if (iter+1) % eval_interval == 0:
		log("checkpointing ...")
		epoch = (block_size * batch_size * (iter+1))/len(train_data)
		losses = estimate_loss()
		now = datetime.datetime.now()
		date_hour = now.strftime("%Y-%m-%d_%H-%M")
		log(f'{date_hour} : iter {iter+1:10d} <=> epoch {epoch} | train loss {losses["train"]:.10f} | val loss {losses["val"]:.10f}')
		if losses["val"] > last_losses["val"]:
			nb_degradations += 1
			if nb_degradations == max_degradations:
				log(f"EARLY STOPPING at iter {iter+1} == epoch {epoch}")
				early_stopping = {"state": True, "iter": iter+1, "epoch": epoch}
				break
		else:
			nb_degradations = 0
			
		# Logging the losses trace
		run["losses_trace/train_loss"].append(losses["train"], step = epoch)
		run["losses_trace/val_loss"].append(losses["val"], step = epoch)
		
		# Saving the last_losses
		last_losses = losses
		
		# Saving the model
		now = datetime.datetime.now()
		date_hour = now.strftime("%Y-%m-%d_%H-%M")
		torch.save(model.state_dict(), f"checkpoints/checkpoint.pth")
		torch.save(optimizer.state_dict(), "checkpoints/optimizer.pth")
		torch.save(scheduler.state_dict(), "checkpoints/scheduler.pth")
		with open(f"checkpoint.info", "w") as f:
			f.write(f"iter : {iter+1}\n")
			f.write(f"epoch : {epoch}\n")
			f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
			f.write(f"date-hour : {date_hour}\n")

		if losses["val"] < best_val_loss:
			best_val_loss = losses["val"]
			torch.save(model.state_dict(), f"checkpoints/best-model.pth")
			with open(f"best-model.info", "w") as f:
				f.write(f"iter : {iter+1}\n")
				f.write(f"epoch : {epoch}\n")
				f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
				f.write(f"date-hour : {date_hour}\n")
		log("training ...")

run["early_stopping"] = early_stopping
log_file.close()