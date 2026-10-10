from tqdm import tqdm

## On Greene
DDIR  = "../../../../../../datasets/dataset-31/datapreps-31/dataprep-31-1/data-dp-31-1/"
deviceid = 7
amp = True


## Logging boilerplate
log_file = open("model-eval.log","w")
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
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

## Set the device to GPU if available, otherwise CPU
log("Set the device to GPU if available, otherwise CPU")
device = f'cuda:{deviceid}'
print(f"Device set to {device}.")


## Loading the encode and decode functions and setting the vocab_size from the meta object
log("Loading the vocab_size")
with open (DDIR+'vocab_size.txt') as f:
	vocab_size = int(f.read())

## Redefining the model of the training
## Redefining the model of the training
log("Redefining the model of the training")
block_size = 512  # Maximum context length
n_embd = 464	  # Embedding dimension for 120M
n_head = 16	# Number of attention heads
n_layer = 24    # Number of transformer blocks
dropout = 0	   # Dropout rate
batch_size = 64   # Batch size for training


## Loading the model
log("Loading the model")
pos_embeds_indices = []
for i in range(block_size):
	line = []
	for j in range(block_size):
		line.append(max(i-j, 0))
	pos_embeds_indices.append(line)
pos_embeds_indices = torch.tensor(pos_embeds_indices, dtype=torch.int64, device=device) # (block_size, block_size)

causal_mask = torch.tril(torch.ones((block_size, block_size), dtype=torch.bool, device=device)) == False

class Head(nn.Module):
	"""One head of self-attention."""

	def __init__(self, head_size):
		super().__init__()
		self.head_size = head_size
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.pos_embeds = nn.Embedding(block_size, 1)
		self.attn_dropout = nn.Dropout(dropout)

	def forward(self, x):
		B,T,C = x.shape
		k = self.key(x)   # (B, T, C)
		q = self.query(x) # (B, T, C)
		v = self.value(x) # (B, T, C)
		pos_bias_matrix = self.pos_embeds(pos_embeds_indices[:T]).squeeze() # (T, T)
		attn = ((q @ k.transpose(-2, -1)) + pos_bias_matrix) * (self.head_size ** -0.5) # (B, T, T)
		attn = attn.masked_fill(causal_mask[:T, :T], float('-inf')) # (B, T, T)
		attn = F.softmax(attn, dim=-1) # (B, T, T)
		attn = self.attn_dropout(attn) # (B, T, T)
		out = attn @ v # (B, T, C)
		
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
	"""Transformer block: communication followed by feedforward."""

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
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd, bias=False)
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def forward(self, idx, targets=None):
		B, T = idx.shape

		# idx and targets are both (B,T) tensor of integers
		x = self.token_embedding_table(idx) # (B,T,C)
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
def generate(model, compiled_model, idx, prompt_lengths, max_new_tokens):
	# idx is (B, T) array of indices in the current context
	idx_cond = torch.stack(idx).to(device) # (B, T)
	for _ in tqdm(range(max_new_tokens)):
		# crop idx to the last block_size tokens
		# get the predictions
		with ctx:
			loc_tens = torch.ones((batch_size, 512), dtype=torch.long, device=device).copy_(idx_cond)
			logits, loss = compiled_model(loc_tens)

		# focus only on the last time step
		# logits = logits[:, -1, :] # becomes (B, C)
		logits = torch.stack([logits[b, prompt_lengths[b]-1, :] for b in range(batch_size)])
		# apply softmax to get probabilities
		probs = F.softmax(logits, dim=-1) # (B, C)
		# sample from the distribution
		_, idx_next = torch.max(probs, dim=1, keepdim=True) # (B, 1)
		#idx_next = torch.multinomial(probs, num_samples=1)
		# append sampled index to the running sequence
		#idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

		for b in range(batch_size):
			if prompt_lengths[b] < block_size:
				idx_cond[b][prompt_lengths[b]] = idx_next[b][0]
				idx[b][prompt_lengths[b]] = idx_next[b][0]
			else:
				idx_cond[b] = torch.cat((idx_cond[b][1:], idx_next[b]))
				idx[b] = torch.cat((idx[b], idx_next[b]))

			prompt_lengths[b] = prompt_lengths[b] + 1 if prompt_lengths[b] < block_size else block_size
	
	return idx


## Creating and loading the model
log("Creating and loading the model")
model = GPT()
model.eval()
model.load_state_dict(torch.load("/data/yb2618/Tiny-Language-Models-Framework/xperiments/xpgroup-10/xp-10-2/train-10-2/atmpt-3/checkpoints/best-model.pth", map_location = device))
model.to(device)
compiled_model = torch.compile(model, mode='default')

torch.set_float32_matmul_precision('high')
ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if amp else nullcontext()

## Reading the test data
log("Reading the test data")
with open("/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-41/more-while-iterations/more-while-iterations-data/final_snippets.txt", "r") as f:
	test_data = f.read()
# with open("./test_traced.txt", "r") as f:
# 	test_data = f.read()
## Splitting the test data into examples
log("Splitting the test data into examples")
examples = test_data.split("\n\n")
examples = examples[:128]

## Sequential Evaluation loop
log("Sequential Evaluation loop")
hard_match_counter = 0
soft_match_counter = 0
failures_counter = 0

hard_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} # correct generated output + correct stopping (no hallucination) i.e. fully correct
soft_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} # correct initial generated output BUT uncorrect stopping (hallucination)
failures = {"example_input":[], "example_output":[], "all-generated-output":[]} # completely uncorrect answer

import time
import pandas as pd
import os
from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer
import re

regex = re.compile("((?:.|\n)*?#STEP\n)((?:.|\n)*)")
tpt = TinypyTokenizer()

# print(f"Sorting ... {len(examples)} examples")
# examples = sorted(examples, key = lambda x: len(tpt.encode(regex.match(x).group(2))), reverse=True)
# print("Done !")

os.makedirs(os.path.dirname("infers/"), exist_ok = True)

checkpoint_interval = 1

hard_match_base = 0
soft_match_base = 0
failures_base = 0
start_time = time.time()

batch_idx = 0
batch_size = 128

# Inference loop
while batch_idx < len(examples):
	
	past = time.time()
	
	batch_examples = examples[batch_idx:batch_idx + batch_size]
	# TODO
	# if len(batch_examples) < batch_size => padd with dummy examples
	prompts_list = []
	tensors_list = [] # list of tensors
	outputs_list = [] # list of strings

	prompt_lengths = []
	max_output_len = 0

	for example in batch_examples:
		match = regex.match(example)
		
		prompt = match.group(1)
		prompts_list.append(prompt)
		encoded_prompt = tpt.encode(prompt)
		prompt_lengths.append(len(encoded_prompt))
		if prompt_lengths[-1] < block_size:
			padding = [0] * (block_size - prompt_lengths[-1])
			encoded_prompt = encoded_prompt + padding
		else:
			encoded_prompt = encoded_prompt[-block_size:]
		input_tens = torch.tensor(encoded_prompt, dtype=torch.long).to(device)
		tensors_list.append(input_tens)
		
		output = match.group(2)
		tokenized_output = tpt.tokenize(output)
		if len(tokenized_output) > max_output_len:
			max_output_len = len(tokenized_output)
		outputs_list.append(" ".join(tokenized_output))


	log(" generating ...")
	generated_outputs = generate(model, compiled_model, tensors_list, list(prompt_lengths), max_new_tokens = max_output_len + 5 )
	generated_outputs = [output.tolist() for output in generated_outputs]

	for prompt, generated_output, example_output, prompt_length in zip(prompts_list, generated_outputs, outputs_list, prompt_lengths):
		generated_output = " ".join(tpt.decode(generated_output)[prompt_length:])
		generated_output = generated_output.split(" \n\n")[0]
		# if hard match
		if generated_output == example_output:
			hard_match_counter += 1
			hard_match_successes["example_input"].append(prompt)
			hard_match_successes["example_output"].append(example_output)
			hard_match_successes["all-generated-output"].append(generated_output+"@")
		# elif soft checking
		elif generated_output.split('\n')[-1] == example_output.split('\n')[-1]:
			soft_match_counter += 1
			soft_match_successes["example_input"].append(prompt)
			soft_match_successes["example_output"].append(example_output)
			soft_match_successes["all-generated-output"].append(generated_output+"@")
		# else complete failure
		else:
			failures_counter += 1
			failures["example_input"].append(prompt)
			failures["example_output"].append(example_output)
			failures["all-generated-output"].append(generated_output+"@")

	present = time.time()
		
	log(f"|ITERS: {batch_idx+batch_size} / {len(examples)} | COMP: {(batch_idx+batch_size)/len(examples) * 100:.2f}% | RATE: {(batch_size)/(present-past):.2f} ex./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds(((len(examples)-(batch_idx+batch_size))/batch_size) * (present-past))} | ET: {convert_seconds(time.time()-start_time)}", p_level = 1)
	log(f"|hard-accuracy: {hard_match_counter} = {(hard_match_counter/(batch_idx+batch_size))*100:.2f}% | soft-accuracy: {soft_match_counter} = {(soft_match_counter/(batch_idx+batch_size))*100:.2f}% |", p_level = 2)
		
	mode, header = ("w",True) if batch_idx == 0 else ("a", False)
	
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
	
	hard_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} # correct generated output + correct stopping (no hallucination) i.e. fully correct
	soft_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} # correct initial generated output BUT uncorrect stopping (hallucination)
	failures = {"example_input":[], "example_output":[], "all-generated-output":[]} # completely uncorrect answer

	batch_idx += batch_size

log_file.close()