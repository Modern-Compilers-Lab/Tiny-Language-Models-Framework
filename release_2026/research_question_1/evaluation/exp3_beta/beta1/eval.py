#@ EVAL-FINAL-CORRECTED-KV
#@ We evaluate the 20M model on hodhaifa generated test.txt for the 200M code snippets dataset
import os
from tqdm import tqdm
import argparse
import glob
import re
import math
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# Configuration & Arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--cptpth", required=True) # checkpoint path
parser.add_argument("--datpth", required=True) # data path 
parser.add_argument("--tstpth", required=True) # test path 
parser.add_argument("--outpth", required=True) # output path 
args = parser.parse_args()

DDIR  = args.datpth
amp = True

# -----------------------------------------------------------------------------
# Logging Utilities
# -----------------------------------------------------------------------------

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
        if len(s) > 0 and s[0].upper() == s[0]:
            start = "\n"
            end = ":"
        else:
            start = "   --> "
            end = ""
        log_file.write(start + s + end + "\n")
    log_file.flush()

def convert_seconds(seconds:float):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (days, hours, minutes, seconds)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

log("Imports")
device = f'cuda:{0}'
print(f"Device set to {device}.")

log("Loading the vocab_size")
try:
    with open (DDIR+'vocab_size.txt') as f:
        vocab_size = int(f.read())
except FileNotFoundError:
    print("Warning: vocab_size.txt not found, defaulting to 50304 or similar if needed.")
    vocab_size = 50304 

log("Redefining the model of the training")
block_size = 512 
n_embd = 384     
n_head = 16      
n_layer = 12     
dropout = 0.0      
batch_size = 512   

# -----------------------------------------------------------------------------
# Model Architecture (KV ENABLED + PADDING MASK FIX)
# -----------------------------------------------------------------------------

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.dropout_p = dropout
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("m", get_alibi_slope(n_head).view(1, n_head, 1, 1))
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, past_kv=None, attention_mask=None):
        B, T, C = x.shape
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_size).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_head, self.head_size).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_head, self.head_size).permute(0, 2, 1, 3)

        # KV Cache: Append to past keys if they exist
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        present_kv = (k, v)

        # ALiBi Logic
        total_len = k.size(2)
        if past_kv is not None:
            # Inference Mode: Generating NEXT token
            relative_pos = torch.arange(total_len, device=x.device) - (total_len - 1)
            relative_pos = relative_pos.view(1, 1, 1, total_len)
            dist = -torch.abs(relative_pos)
            alibi_bias = self.m * dist
            attn_mask = alibi_bias
        else:
            # Training/Prefill Mode
            x_pos = torch.arange(T, device=x.device)
            y_pos = torch.arange(T, device=x.device)
            relative_pos = x_pos[None, :] - y_pos[:, None] 
            dist = -torch.abs(relative_pos).view(1, 1, T, T)
            alibi_bias = self.m * dist
            
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            causal_mask = causal_mask.view(1, 1, T, T)
            attn_mask = alibi_bias + causal_mask

        # --- FIX: Apply Padding Mask ---
        # attention_mask comes in as (B, Total_Len) where 0 is PAD, 1 is KEEP
        if attention_mask is not None:
            # Broadcast to (B, 1, 1, Total_Len)
            # We want to add -inf where mask is 0
            # Ensure mask aligns with the current Key length (total_len)
            mask_slice = attention_mask[:, :total_len].view(B, 1, 1, total_len)
            
            # Create a large negative number tensor
            pad_bias = (1.0 - mask_slice) * -1e9
            attn_mask = attn_mask + pad_bias

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0, 
                is_causal=False 
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attn_mask # Apply ALiBi + Causal + Padding mask
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, present_kv

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

    def forward(self, x, past_kv=None, attention_mask=None):
        sa_out, present_kv = self.sa(self.ln1(x), past_kv, attention_mask)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, present_kv

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, past_kvs=None, attention_mask=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) 
        
        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            p_kv = past_kvs[i] if past_kvs is not None else None
            # Pass attention mask down
            x, present_kv = block(x, p_kv, attention_mask)
            new_past_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_past_kvs

# -----------------------------------------------------------------------------
# Optimized Generation Logic (With Masking)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate(model, compiled_model, idx_list, max_new_tokens):
    """
    idx_list: List of 1D tensors (varying lengths).
    max_new_tokens: How many tokens to generate.
    """
    batch_size = len(idx_list)
    
    # 1. Left Padding Strategy
    max_len = max(len(t) for t in idx_list)
    input_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    
    # Create Attention Mask (0 for pad, 1 for real)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    
    for i, seq in enumerate(idx_list):
        l = len(seq)
        input_tensor[i, -l:] = seq
        attention_mask[i, -l:] = 1.0 # Mark real tokens as 1

    # 2. Prefill Step
    with ctx:
        # Pass the mask to ignore padding in the KV cache
        logits, past_kvs = compiled_model(input_tensor, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :]

    generated_tokens = []

    # 3. Decoding Loop
    for _ in tqdm(range(max_new_tokens), desc="Decoding Batch", leave=False):
        probs = F.softmax(next_token_logits, dim=-1)
        _, next_token = torch.max(probs, dim=1, keepdim=True) 
        
        generated_tokens.append(next_token)
        
        # Extend the attention mask for the new token (it is real, so 1.0)
        new_mask_col = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        attention_mask = torch.cat((attention_mask, new_mask_col), dim=1)
        
        with ctx:
            logits, past_kvs = compiled_model(next_token, past_kvs=past_kvs, attention_mask=attention_mask)
        
        next_token_logits = logits[:, -1, :]

    # 4. Reconstruct Output
    gen_tensor = torch.cat(generated_tokens, dim=1) 
    
    final_results = []
    for i in range(batch_size):
        orig = idx_list[i]
        gen = gen_tensor[i]
        final_results.append(torch.cat([orig, gen]))
        
    return final_results

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

log("Creating and loading the model")
model = GPT()
model.eval()

print(f"Loading checkpoint from {args.cptpth}")
model.load_state_dict(torch.load(args.cptpth, map_location=device, weights_only=True))
model.to(device)

# Important: torch.compile can sometimes struggle with dynamic masks + lists
# If this errors, comment out the compile line.
#compiled_model = torch.compile(model, mode='default')
compiled_model = model
# compiled_model = model # Fallback if compile fails with mask

torch.set_float32_matmul_precision('high')
ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else nullcontext()

log("Reading the test data")
from tinypy_code_tracer_tokenizer import TinypyTokenizer
tpt = TinypyTokenizer()

with open(args.tstpth, "r") as f:
    test_data = f.read()

log("Splitting the test data into examples")
examples = test_data.split("\n\n")[:-1]
examples = examples[:1024]

log("Sequential Evaluation loop")
hard_match_counter = 0
soft_match_counter = 0
failures_counter = 0

hard_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} 
soft_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} 
failures = {"example_input":[], "example_output":[], "all-generated-output":[]} 

regex = re.compile("((?:.|\n)*?#STEP\n)((?:.|\n)*)")

print(f"Sorting ... {len(examples)} examples")
examples = sorted(examples, key = lambda x: len(tpt.encode(regex.match(x).group(2))), reverse=True)
print("Done !")

abs_infers_path = os.path.join(args.outpth, "infers")
os.makedirs(abs_infers_path, exist_ok=True)

hard_match_base = 0
soft_match_base = 0
failures_base = 0
start_time = time.time()

batch_idx = 0
batch_size_infer = 64

# Inference loop
while batch_idx < len(examples):
    
    past = time.time()
    
    batch_examples = examples[batch_idx:batch_idx + batch_size_infer]
    prompts_list = []
    tensors_list = [] 
    outputs_list = [] 
    prompt_lengths = []
    max_output_len = 0

    for example in batch_examples:
        match = regex.match(example)
        
        prompt = match.group(1)
        prompts_list.append(prompt)
        
        # Encode
        encoded_prompt = tpt.encode(prompt)
        prompt_lengths.append(len(encoded_prompt))
        
        # Append raw tensor to list
        input_tens = torch.tensor(encoded_prompt, dtype=torch.long, device=device)
        tensors_list.append(input_tens)
        
        output = match.group(2)
        tokenized_output = tpt.tokenize(output)
        if len(tokenized_output) > max_output_len:
            max_output_len = len(tokenized_output)
        outputs_list.append(" ".join(tokenized_output))

    max_output_len = min(max_output_len, block_size)

    log(f"Batch {batch_idx//batch_size_infer}: Generating...", p_level=0)
    
    # Generate (Handles KV Cache and Padding)
    generated_outputs = generate(model, compiled_model, tensors_list, max_new_tokens = max_output_len + 5 )
    generated_outputs = [output.tolist() for output in generated_outputs]

    for prompt, generated_output, example_output, prompt_length in zip(prompts_list, generated_outputs, outputs_list, prompt_lengths):
        # Decode and slice
        generated_output_str = " ".join(tpt.decode(generated_output)[prompt_length:])
        generated_output_clean = generated_output_str.split(" \n\n")[0]
        
        # Match Logic
        is_hard = generated_output_clean == example_output
        gen_last = generated_output_clean.strip().split('\n')[-1]
        ref_last = example_output.strip().split('\n')[-1]
        is_soft = gen_last == ref_last

        if is_hard:
            hard_match_counter += 1
            hard_match_successes["example_input"].append(prompt)
            hard_match_successes["example_output"].append(example_output)
            hard_match_successes["all-generated-output"].append(generated_output_clean+"@")
        elif is_soft:
            soft_match_counter += 1
            soft_match_successes["example_input"].append(prompt)
            soft_match_successes["example_output"].append(example_output)
            soft_match_successes["all-generated-output"].append(generated_output_clean+"@")
        else:
            failures_counter += 1
            failures["example_input"].append(prompt)
            failures["example_output"].append(example_output)
            failures["all-generated-output"].append(generated_output_clean+"@")

    present = time.time()
    current_batch_len = len(batch_examples)
    total_processed = batch_idx + current_batch_len
    
    # Status strings
    stats_msg = f"|ITERS: {total_processed} / {len(examples)} | COMP: {total_processed/len(examples) * 100:.2f}% | RATE: {current_batch_len/(present-past):.2f} ex./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds(((len(examples)-total_processed)/current_batch_len) * (present-past))} | ET: {convert_seconds(time.time()-start_time)}"
    acc_msg = f"|hard-accuracy: {hard_match_counter} = {(hard_match_counter/total_processed)*100:.2f}% | soft-accuracy: {soft_match_counter} = {(soft_match_counter/total_processed)*100:.2f}% |"
    
    # Print to Terminal AND Log
    print(stats_msg)
    print(acc_msg)
    log(stats_msg, p_level = 1)
    log(acc_msg, p_level = 2)
        
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

    hard_match_successes_df.to_csv(os.path.join(abs_infers_path, "hard.csv"), mode = mode, header = header)
    soft_match_successes_df.to_csv(os.path.join(abs_infers_path, "soft.csv"), mode = mode, header = header)
    failures_df.to_csv(os.path.join(abs_infers_path, "fail.csv"), mode = mode, header = header)

    # Reset lists
    hard_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} 
    soft_match_successes = {"example_input":[], "example_output":[], "all-generated-output":[]} 
    failures = {"example_input":[], "example_output":[], "all-generated-output":[]} 

    batch_idx += batch_size_infer

log_file.close()