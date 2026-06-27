#@ EVAL-2-ATMPT-1
#@ We evaluate the 20M model on hodhaifa generated test.txt for the 200M code snippets dataset

from tqdm import tqdm
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cptpth", required=True) # checkpoint path (path to the model checkpoint)
parser.add_argument("--datpth", required=True) # data path (path to the training data folder "must contain vocab.txt")
parser.add_argument("--tstpth", required=True) # test path (path to the test dataset)
parser.add_argument("--outpth", required=True) # output path (path to the output folder where evaluation results will be written)
args = parser.parse_args()

## On Greene
# model_name = "best-model.pth"
# DDIR  = "/data/ia2921/Tiny_language_model_framework/push_to_the_limit/more_data/data_gen/data/"
DDIR = args.datpth
deviceid = 0
amp = True


## Logging boilerplate


os.makedirs(args.outpth, exist_ok=True)
log_file = open(os.path.join(args.outpth,"model-eval.log"),"w")
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
# torch.device(device)

## Loading the encode and decode functions and setting the vocab_size from the meta object
log("Loading the vocab_size")
with open (DDIR+'vocab_size.txt') as f:
	vocab_size = int(f.read())

vocab_size = (vocab_size + 63) // 64 * 64

## Redefining the model of the training
log("Redefining the model of the training")
block_size = 512  # Maximum context length
n_embd = 384	  # Embedding dimension for 120M
n_head = 16		  # Number of attention heads
n_layer = 12      # Number of transformer blocks
dropout = 0	      # Dropout rate
batch_size = 64   # Batch size for training

def build_rope_cache(dim, device, seq_len=block_size): # precomputing the SIN(Theta) and COS(Theta) for all possible Theta(i,m) combinations out there

	theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
	pos = torch.arange(seq_len, dtype=torch.float, device=device)
	freqs = torch.einsum('i,j->ij', pos, theta)  # (seq_len, dim/2)
	sin = torch.sin(freqs)
	cos = torch.cos(freqs)
	return sin, cos

def apply_rope(x, sin, cos):
    # x is now shape: (Batch, Heads, Seq_Len, Head_Size)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # Extract the sequence length from the correct dimension (dim 2)
    seq_len = x.shape[2]

    # Trim to current seq length and reshape to (1, 1, Seq_Len, Head_Size/2) for broadcasting
    sin = sin[:seq_len, :].view(1, 1, seq_len, -1)
    cos = cos[:seq_len, :].view(1, 1, seq_len, -1)

    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rotated.flatten(-2)

# def apply_rope(x, sin, cos):
#     x1 = x[..., ::2]
#     x2 = x[..., 1::2]

#     # trim sin/cos to match seq length
#     sin = sin[:x.shape[1], :]
#     cos = cos[:x.shape[1], :]

#     x_rotated = torch.stack([
#         x1 * cos - x2 * sin,
#         x1 * sin + x2 * cos
#     ], dim=-1)

#     return x_rotated.flatten(-2)
# ---------------------------------------------------

# --------------------- MODEL -----------------------

# class Head(nn.Module):
#     """One head of self-attention."""

#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embd, head_size, bias=False)
#         self.query = nn.Linear(n_embd, head_size, bias=False)
#         self.value = nn.Linear(n_embd, head_size, bias=False)
#         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
#         self.dropout = nn.Dropout(dropout)
#         self.head_size = head_size


#     def forward(self, x, rope_sin, rope_cos):
#         B,T,C = x.shape

#         k = self.key(x)   # (B, T, 16)
#         q = self.query(x) # (B, T, 16)
#         v = self.value(x)

#         q = apply_rope(q, rope_sin, rope_cos)
#         k = apply_rope(k, rope_sin, rope_cos)

#         out = torch.nn.functional.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=None,
#             dropout_p=dropout if self.training else 0,
#             is_causal=True
#         )
        
#         return out

# class MultiHeadAttention(nn.Module):
#     """multiple heads of self-attention in parallel."""

#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, rope_sin, rope_cos):
#         out = torch.cat([h(x, rope_sin, rope_cos) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention computed in parallel."""

    def __init__(self, n_head, head_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        
        # 1 matrix for Queries, Keys, and Values for ALL heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, rope_sin, rope_cos):
        B, T, C = x.size()
        
        # Calculate Q, K, V all at once
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape to (Batch, Heads, Sequence_Length, Head_Size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, rope_sin, rope_cos)
        k = apply_rope(k, rope_sin, rope_cos)

        # FlashAttention
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout if self.training else 0,
            is_causal=True
        )
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.dropout(self.c_proj(y))
    
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

    def forward(self, x, rope_sin=None, rope_cos=None):
        x = x + self.sa(self.ln1(x), rope_sin, rope_cos)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        head_size = n_embd // n_head
        sin, cos = build_rope_cache(head_size, 'cpu', seq_len=block_size)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        
        for block in self.blocks:
            x = block(x, self.rope_sin, self.rope_cos)
        
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
    

@torch.no_grad()
def generate(model, compiled_model, idx, prompt_lengths, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    idx_cond = torch.stack(idx) # (B, T)
    
    # Use dynamic batch size so the final remainder batch doesn't crash!
    current_bs = idx_cond.shape[0] 
    
    for _ in tqdm(range(max_new_tokens)):
        with ctx:
            # Replaced global 'batch_size' with 'current_bs'
            loc_tens = torch.ones((current_bs, 512), dtype=torch.long, device=device).copy_(idx_cond)
            logits, loss = compiled_model(loc_tens)

        # Replaced global 'batch_size' with 'current_bs'
        logits = torch.stack([logits[b, prompt_lengths[b]-1, :] for b in range(current_bs)])
        probs = F.softmax(logits, dim=-1) 
        _, idx_next = torch.max(probs, dim=1, keepdim=True) 

        # Replaced global 'batch_size' with 'current_bs'
        for b in range(current_bs):
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
model.load_state_dict(torch.load(args.cptpth, map_location = device))
model.to(device)
compiled_model =  torch.compile(model, mode='default')

torch.set_float32_matmul_precision('high')
ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if amp else nullcontext()

## Reading the test data
log("Reading the test data")
with open(args.tstpth, "r") as f:
	test_data = f.read()
# with open("./test_traced.txt", "r") as f:
# 	test_data = f.read()
## Splitting the test data into examples
log("Splitting the test data into examples")
examples = test_data.split("\n\n")[:-1]
examples = examples[:1024]

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
from tinypy_code_tracer_tokenizer import TinypyTokenizer
import re

regex = re.compile("((?:.|\n)*?#STEP\n)((?:.|\n)*)")
tpt = TinypyTokenizer()

print(f"Sorting ... {len(examples)} examples")
examples = sorted(examples, key = lambda x: len(tpt.encode(regex.match(x).group(2))), reverse=True)
print("Done !")

abs_infers_path = os.path.join(args.outpth, "infers")
os.makedirs(abs_infers_path, exist_ok=True)

checkpoint_interval = 1

hard_match_base = 0
soft_match_base = 0
failures_base = 0
start_time = time.time()

batch_idx = 0
batch_size = 512

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
		input_tens = torch.tensor(encoded_prompt, dtype=torch.long, device=device)
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
	hard_match_successes_df.to_csv(os.path.join(abs_infers_path,"hard-match-successes.csv"), mode = mode, header = header)
	soft_match_successes_df.to_csv(os.path.join(abs_infers_path,"soft-match-successes.csv"), mode = mode, header = header)
	failures_df.to_csv(os.path.join(abs_infers_path,"failures.csv"), mode = mode, header = header)
	
	hard_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} # correct generated output + correct stopping (no hallucination) i.e. fully correct
	soft_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} # correct initial generated output BUT uncorrect stopping (hallucination)
	failures = {"example_input":[], "example_output":[], "all-generated-output":[]} # completely uncorrect answer

	batch_idx += batch_size

log_file.close()