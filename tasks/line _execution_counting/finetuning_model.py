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

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters for the GPT model
block_size = 256  # Maximum context length
n_embd = 912      # Embedding dimension
n_head = 6        # Number of attention heads
n_layer = 6       # Number of transformer blocks
dropout = 0       # Dropout rate
batch_size = 64   # Batch size for training
max_iters = 100_000  # Maximum number of iterations
learning_rate = 1e-3 # Initial Learning rate value
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]  # Milestones for learning rate decay as fractions of max_iters
eval_interval = 10_000 # Evaluation interval
eval_iters = 1000 # Number of iterations for evaluation
vocab_size = 54 # Vocabulary size

# defining the entire structure of the model, and in parallel implementing lora
class LayerNorm(nn.Module):
    """ LayerNorm with an optional bias. PyTorch's LayerNorm doesn't support simply bias=False """

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
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Apply scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0, is_causal=True
        )
        
        return out
    

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Concatenate the outputs from each head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class LinearLoRA(nn.Module):
    def __init__(self, original_layer, rank=8):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        self.rank = rank
        
        self.lora_a = nn.Parameter(torch.randn((original_layer.in_features, rank)))
        self.lora_b = nn.Parameter(torch.randn((rank, original_layer.out_features)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)
        
    def forward(self, x):
        lora_output = x @ self.lora_a @ self.lora_b
        return self.original_layer(x) + lora_output
    
class Block(nn.Module):
    """Transformer block: communication followed by feedforward."""

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
    """GPT language model."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate new tokens given an initial context `idx`."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Crop to the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # Focus on the last time step
            probs = F.softmax(logits, dim=-1) # Convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # Sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # Append sampled index to the sequence
        return idx
    
    def activate_lora(self, r=8, heads_only=False, freeze_others=True):
        self.lora_rank = r
        self.replace_multihead_attention_recursion(heads_only)
        if freeze_others:
            self.freeze_parameters_except_lora_and_bias()
    
    def replace_multihead_attention_recursion(self, heads_only=False, model=None):
        children = self.named_children() if model is None else model.named_children()
        for name, module in children:
            if heads_only and name in {"query", "key", "value"}:
                # Replace with Lora SelfAttention
                new_layer = LinearLoRA(module, rank=self.lora_rank)

                if model == None:
                    self.__setattr__(name, new_layer)
                else:
                    setattr(model, name, new_layer)
            
            elif isinstance(module, nn.Linear) and not heads_only:
                new_layer = LinearLoRA(module, rank=self.lora_rank)
                
                if model == None:
                    self.__setattr__(name, new_layer)
                else:
                    setattr(model, name, new_layer)
            
            else:
                # Recursive call for child modules
                self.replace_multihead_attention_recursion(heads_only, model=module)
                
                
    def freeze_parameters_except_lora_and_bias(self):
        for name, param in self.named_parameters():
            is_trainable = (
                "lora_" in name
                #(self.train_layer_norms and "LayerNorm" in name)
            )

            param.requires_grad = is_trainable