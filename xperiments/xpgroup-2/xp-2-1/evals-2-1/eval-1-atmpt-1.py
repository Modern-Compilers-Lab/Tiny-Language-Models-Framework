#@ EVAL-2-ATMPT-1
#@ We evaluate the 20M model on hodhaifa generated test.txt for the 200M code snippets dataset

## On Greene
DDIR  = "../../../../datasets/dataset-5/datapreps-5/dataprep-5-1/data-dp-5-1/"
deviceid = 7


## Logging boilerplate
log_file = open("eval-1-atmpt-1.log","w")
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
	

## Imports
log("Imports")
import torch
import torch.nn as nn
import torch.nn.functional as F


## Set the device to GPU if available, otherwise CPU
log("Set the device to GPU if available, otherwise CPU")
device = torch.device(f'cuda:{deviceid}')
print(f"Device set to {device}.")


## Loading the encode and decode functions and setting the vocab_size from the meta object
log("Loading the encode and decode functions and setting the vocab_size from the meta object")
import pickle

with open(DDIR+"meta.pkl", "rb") as f:
	meta = pickle.load(f)
stoi = meta["stoi"]
itos = meta["itos"]

def encode(s):
	return [stoi[c] for c in s] # encoder: take a string, output a list of integers

def decode(l):
	return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

vocab_size = meta["vocab_size"]


## Redefining the model of the training
log("Redefining the model of the training")
block_size = 256  # Maximum context length
n_embd = 642	  # Embedding dimension for 120M
n_head = 6		# Number of attention heads
n_layer = 6	   # Number of transformer blocks
dropout = 0	   # Dropout rate
batch_size = 64   # Batch size for training


## Loading the model
log("Loading the model")
class LayerNorm(nn.Module):
	""" LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

	def __init__(self, ndim, bias):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(ndim))
		self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

	def forward(self, input):
		return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
			nn.GELU(),
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
		self.ln1 = nn.LayerNorm(n_embd, bias=False)
		self.ln2 = nn.LayerNorm(n_embd, bias=False)

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
	
	@torch.no_grad()
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
	

## Creating and loading the model
log("Creating and loading the model")
model = GPT()
model.eval()
model.load_state_dict(torch.load("/data/hb3020/github-repo/Tiny-Language-Models-Framework/xperiments/xpgroup-2/xp-2-1/train-2-1/checkpoints/best-model.pth"))
model.to(device)


## Reading the test data
log("Reading the test data")
with open(DDIR+"test.txt", "r") as f:
	test_data = f.read()


## Splitting the test data into examples
log("Splitting the test data into examples")
examples = test_data.split("\n\n")
examples = examples[:]


## Sequential Evaluation loop
log("Sequential Evaluation loop")
hard_match_counter = 0
soft_match_counter = 0
failures_counter = 0

hard_match_successes = {"example":[], "all-generated-output":[]} # correct generated output + correct stopping (no hallucination) i.e. fully correct
soft_match_successes = {"example":[], "all-generated-output":[]} # correct initial generated output BUT uncorrect stopping (hallucination)
failures = {"example":[], "all-generated-output":[]} # completely uncorrect answer

import time
import pandas as pd
import os

os.makedirs(os.path.dirname("infers/"), exist_ok = True)

checkpoint_interval = 500

hard_match_base = 0
soft_match_base = 0
failures_base = 0

for i, example in enumerate(examples):
	
	past = time.time()
	
	# generating the output of the model
	example_match = example.split("# output\n")
	example_prompt = example_match[0] + "# output\n"
	example_output = example_match[1]
	example_prompt_tensor = torch.tensor(encode(example_prompt), dtype=torch.long).unsqueeze(0).to(device)
	generated_example = decode(model.generate(example_prompt_tensor, max_new_tokens = len(example_output) + 20)[0].tolist())
	
	generated_output  = generated_example[len(example_prompt):]
	example_code_generated_output = generated_output.split("\n\n")[0]
	# if hard match
	if example_code_generated_output == example_output:
		hard_match_counter += 1
		hard_match_successes["example"].append(example)
		hard_match_successes["all-generated-output"].append(generated_output+"@")
	# elif soft checking
	elif example_code_generated_output[:len(example_output)] == example_output:
		soft_match_counter += 1
		soft_match_successes["example"].append(example)
		soft_match_successes["all-generated-output"].append(generated_output+"@")
	# else complete failure
	else:
		failures_counter += 1
		failures["example"].append(example)
		failures["all-generated-output"].append(generated_output+"@")
	
	present = time.time()
	
	log(f"|ITERS: {i+1} / {len(examples)} | COMP: {(i+1)/len(examples) * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((len(examples)-i-1) * (present-past))} |", p_level = 1)
	log(f"|hard-accuracy: {(hard_match_counter/(i+1))*100:.2f}% | soft-accuracy: {(soft_match_counter/(i+1))*100:.2f}% |", p_level = 2)

	if (i+1) % checkpoint_interval == 0:
		
		mode, header = ("w",True) if (i+1) == checkpoint_interval else ("a", False)
		
		hard_match_successes_df = pd.DataFrame(hard_match_successes)
		soft_match_successes_df = pd.DataFrame(soft_match_successes)
		failures_df = pd.DataFrame(failures)

		hard_match_successes_df.index = hard_match_successes_df.index + hard_match_base
		soft_match_successes_df.index = soft_match_successes_df.index + soft_match_base
		failures_df.index = failures_df.index + failures_base
		
		hard_match_base = hard_match_counter
		soft_match_base = soft_match_counter
		failures_base = failures_counter

		hard_match_successes_df.to_csv("infers/hard-match-successes.csv", mode = mode, header = header)
		soft_match_successes_df.to_csv("infers/soft-match-successes.csv", mode = mode, header = header)
		failures_df.to_csv("infers/failures.csv", mode = mode, header = header)
		
		hard_match_successes = {"example":[], "all-generated-output":[]}
		soft_match_successes = {"example":[], "all-generated-output":[]}
		failures = {"example":[], "all-generated-output":[]}


## Logging the metrics
log("Logging the metrics")
import neptune

run = neptune.init_run(
	project="younes-boukacem-workspace/tiny-lm-full-random-mode",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZGFmNDg2Yy01MzRkLTQwNGMtYmZmMy1hYzM0Y2NkY2QyNmMifQ==",
	with_id = "IMG1-78",
	capture_hardware_metrics = False
)
run["eval-1/hard-accuracy-percentage"] = (hard_match_counter/len(examples))*100
run["eval-1/soft-accuracy-percentage"] = (soft_match_counter/len(examples))*100

log_file.close()