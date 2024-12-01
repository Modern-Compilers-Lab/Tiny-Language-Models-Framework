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
import time
import argparse
from tqdm.auto import tqdm
import pandas as pd
import argparse

class GPTTrainer:
    def __init__(self, device, batch_size, block_size, eval_iters, compile=True):
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_iters = eval_iters
        self.compile = compile
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.optimizer = None
        self.scheduler = None
        self.meta = None
        
     # Helper functions to load and save data
    def save_data(self,data, file_path):
        with open(file_path, 'w') as f:
            f.write(data)

    def read_data(self,file_path):
        with open(file_path, 'r') as f:
            return f.read()

    def load_data(self, data_directory):
        print(f"Loading data from {data_directory}...\n")
        # Load meta information
        with open(os.path.join(data_directory,"meta.pkl"), 'rb') as f:
            self.meta = pickle.load(f)

        args.vocab_size = self.meta['vocab_size']
        itos = self.meta['itos']
        stoi = self.meta['stoi']

        print(f"vocab size: {args.vocab_size:,}")
        print("all the unique characters:", ''.join(sorted(stoi.keys())))

        # Load data
        train_data = self.read_data(os.path.join(data_directory, 'train.txt'))
        val_data = self.read_data(os.path.join(data_directory, 'val.txt'))
        test_data = self.read_data(os.path.join(data_directory, 'test.txt'))

        # Encode data
        train_ids = self.encode(train_data)
        val_ids = self.encode(val_data)
        test_ids = self.encode(test_data)

        # Save encoded data to bin files, make sure to choose "Files only" on the persistence option of the session so that you don't encode data each time
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        test_ids = np.array(test_ids, dtype=np.uint16)

        train_ids.tofile( 'train.bin')
        val_ids.tofile( 'val.bin')
        test_ids.tofile('test.bin')
        
        print(f"train has {len(train_data):,} tokens")
        print(f"val has {len(val_data):,} tokens")
        print(f"test has {len(test_data):,} tokens")
        print("Encoded data saved as binary files.")
        
        print(f"Data in tokens: {len(train_data)}")
        iters4epoch = len(train_data)//(args.batch_size * args.block_size)
        print(f"Number of iters for one pseudo-epoch : {iters4epoch}")
        print(f"Number of pseudo-epochs : {args.max_iters / iters4epoch:.2f}")
        
        
        del(train_ids)
        del(val_ids)
        del(test_ids)
        

        self.train_data = np.memmap("train.bin", dtype=np.uint16, mode='r')
        self.val_data = np.memmap("val.bin", dtype=np.uint16, mode='r')
        self.test_data = np.memmap("test.bin", dtype=np.uint16, mode='r')

    def encode(self, s):
        return np.array([self.meta['stoi'][c] for c in s], dtype=np.int32)

    def decode(self, l):
        return ''.join([self.meta['itos'][i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    @staticmethod
    def human_readable(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        self.model = GPT()  # Now correctly using the imported GPT class
        print("Compiling the model...\n")
        r = -1
        
        if self.compile:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Compilation failed: {e}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'lora_rank' in checkpoint:
            r = checkpoint['lora_rank']
            state = checkpoint['state_dict']
            if r > 0:
                self.model.activate_lora(r)
            self.model.load_state_dict(state)
        else:
            state_dict = checkpoint
            state_dict_keys = map(lambda x: x.replace("_orig_mod.", ""), state_dict.keys())
            state_dict = dict(zip(state_dict_keys, state_dict.values()))
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        return r

    def initialize_model(self, model_path, lora_r):
        if model_path:
            print("Loading model...\n")
            r = self.load_model(model_path)
            
            if r > 0:
                lora_r = r
        else:
            self.model = GPT().to(self.device)  # Using the imported GPT class
            r = -1
            if self.compile:
                print("Compiling the model... (takes a ~minute)")
                self.model = torch.compile(self.model)
        
        if lora_r > 0 and r < 0:
            print("Activating LoRA...")
            self.model.activate_lora(lora_r)
            self.model = self.model.to(self.device)
        
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_parameters_hr = self.human_readable(num_parameters)
        print(f'The model has {num_parameters_hr} trainable parameters')

    def setup_optimizer(self, learning_rate, milestones):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def train(self, max_iters, eval_interval):
        now = datetime.datetime.now()
        date_hour = now.strftime("%Y-%m-%d_%H-%M")
        start_time = time.time()

        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = self.estimate_loss()
                print(f'iter {iter:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
            
            xb, yb = self.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        end_time = time.time()
        print(f'Training time: {(end_time - start_time) / 60:.2f} min')

        return date_hour

    def save_model(self, date_hour):
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_parameters_hr = self.human_readable(num_parameters)
        model_path = f"./output/{num_parameters_hr}_{date_hour}.pth"
        checkpoint = {
            'lora_rank': self.model.lora_rank if hasattr(self.model, "lora_rank") else -1,
            'state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}\n")
    # Evaluate example

    def evaluate_example(self,example):
            """
            Evaluate an example using the loaded model.
            """

            # Encode prompt and prepare for evaluation
            prompt_text = example[:-1]
            encoded_example = torch.tensor(self.encode(prompt_text), dtype=torch.long).unsqueeze(0).to(self.device)
            result_example = example[-1]

            # Generate response from model and extract generated results
            with torch.no_grad():
                response = self.decode(self.model.generate(encoded_example, max_new_tokens=1)[0].tolist())
            generated_results = response[-1]

            return prompt_text, result_example, generated_results

    # Write results to file
    def write_results_to_file(self,output_file, prompt, real_results, generated_results):
        df = pd.DataFrame({
            'Prompt': prompt,
            'Real_Results': real_results,
            'Generated_Results': generated_results
        })
        df.to_csv(output_file, index=False)

    def evaluate_model(self,examples):
        # Start evaluation process
        prompt = []
        real_results = []
        generated_results = []
        
        examples = self.decode(self.test_data).split("\n\n")

        # Iterate through examples and evaluate the model on each one
        for example in tqdm(examples):
            prompt_text, real_result, result = self.evaluate_example(example)
            prompt.append(prompt_text)
            real_results.append(real_result)
            generated_results.append(result)

        # Calculate and print accuracy
        correct_count = sum(1 for real, generated in zip(real_results, generated_results) if real == generated)
        accuracy = correct_count / len(generated_results)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Store accuracy in a file
        with open("./output/accuracy.txt", 'w') as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

            # Store predictions in a CSV file
            self.write_results_to_file("./output/predictions.csv", prompt, real_results, generated_results)


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
        self.key = nn.Linear(args.n_embd, head_size, bias=False)
        self.query = nn.Linear(args.n_embd, head_size, bias=False)
        self.value = nn.Linear(args.n_embd, head_size, bias=False)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Apply scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=args.dropout if self.training else 0, is_causal=True
        )
        
        return out
    

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)
        
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
            nn.Dropout(args.dropout),
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
        self.token_embedding_table = nn.Embedding(args.vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.Sequential(*[Block(args.n_embd, n_head=args.n_head) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd, bias=False) 
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=args.device)) # (T, n_embd)
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
            idx_cond = idx[:, -args.block_size:] # Crop to the last block_size tokens
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





parser = argparse.ArgumentParser(description="Fine-tune a GPT model")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the model on (cuda or cpu)")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for training")
parser.add_argument("--block_size", type=int, default=256,
                    help="Block size for input sequences")
parser.add_argument("--eval_iters", type=int, default=10,
                    help="Number of iterations for evaluation")
parser.add_argument("--model_path", type=str, default="./model/model.pth",
                    help="Name of the pre-trained model to load")
parser.add_argument("--lora_r", type=int, default=20,
                    help="LoRA rank (0 to disable)")
parser.add_argument("--learning-rate", type=float, default=1e-3,
                    help="Learning rate for the optimizer")
parser.add_argument("--max_iters", type=int, default=10,
                    help="Maximum number of training iterations")
parser.add_argument("--eval_interval", type=int, default=2000,
                    help="Interval between evaluations")
parser.add_argument("--input_path", type=str, required=True, default="./data/",
                    help="Path to the input data file")
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--n_embd", type=int, default=372)
parser.add_argument("--n_head", type=int, default=6)
parser.add_argument("--n_layer", type=int, default=6)



args = parser.parse_args()

args.milestones = [int(args.max_iters * m) for m in [0.7, 0.8, 0.9]]
args.compile = False


# Create trainer instance
trainer = GPTTrainer(
    device=torch.device(args.device),
    batch_size=args.batch_size,
    block_size=args.block_size,
    eval_iters=args.eval_iters,
    compile=args.compile
)

# Load and process data
trainer.load_data(args.input_path)

# Initialize model
trainer.initialize_model(args.model_path, args.lora_r)

# Setup optimizer and scheduler
trainer.setup_optimizer(args.learning_rate, args.milestones)

# Train the model
date_hour = trainer.train(args.max_iters, args.eval_interval)

# Save the trained model
trainer.save_model(date_hour)

trainer.evaluate_model(trainer.test_data)