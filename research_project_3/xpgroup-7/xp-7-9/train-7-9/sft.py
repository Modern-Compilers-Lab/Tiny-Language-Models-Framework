import 	random
import 	os
import 	time
import 	torch
import 	datetime
import 	math
import inspect
import 	torch.nn 			as nn
import 	torch.nn.functional as F
import 	numpy 				as np
import 	torch.distributed 	as dist
from 	torch.nn.parallel 	import DistributedDataParallel as DDP
from 	SFTSampler 			import SFTRandomSampler
from 	tinypy_code_tracer_m2_tokenizer import TinypyTokenizer
tpt = TinypyTokenizer()


# print('hello')
# import sys; sys.exit(0)
# __Setup the multi-GPU training__

# Set the device ids
deviceids = [1, 2, 3, 4, 5, 6, 7] # Set the device ids to use for training

# Check if this is a ddp run
ddp = int(os.environ.get('RANK', -1)) != -1

# If ddp than setup per process control variables
if ddp:
	print("ddp is set to True")
	from torch.distributed import init_process_group, destroy_process_group
	assert torch.cuda.is_available()
	init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
	ddp_rank = int(os.environ['RANK'])
	ddp_local_rank = int(os.environ['LOCAL_RANK'])
	ddp_world_size = int(os.environ['WORLD_SIZE'])
	assert ddp_world_size == len(deviceids) # for now just to make sure
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

if master_process:

	# Prepare the checkpoints folder
	os.makedirs(os.path.dirname("checkpoints/"), exist_ok=True)
	
	# Open the log file
	log_file = open("train.log", "w")

	# Create the log boilerplate for progress bars
	pbar_recept_string = " " * 200 + "\n"
	log_file.write(pbar_recept_string)
	log_file.write(pbar_recept_string)
	log_file.flush()

	# Define the log function
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
			log_file.write('\n' + s + '\n')
			print(s)
		log_file.flush()

	# Define function to convert seconds to days, hours, minutes, seconds
	def convert_seconds(seconds:float):
		# ignoring the sub seconds
		seconds = int(seconds)
		days, seconds = divmod(seconds, 86400)
		hours, seconds = divmod(seconds, 3600)
		minutes, seconds = divmod(seconds, 60)
		return (days, hours, minutes, seconds)

	batch_log = open('batch_log.log', 'w')

# Set the data directory
DDIR = "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-31/datapreps-31/dataprep-31-1/data-dp-31-1/"

# Set the random seed for reproducibility
seed = 142+seed_offset
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Set arch-hyperparams for the GPT model
block_size 	= 512 	# Maximum context length
n_embd 		= 464	# Embedding dimension
n_head 		= 16	# Number of attention heads
n_layer 	= 24	# Number of transformer blocks
with open(DDIR+"vocab_size.txt", "rb") as f:
	vocab_size = int(f.read())
if master_process: log(f"vocab_size: {vocab_size}")

# Initialize the sft sampler. Here we use the random sampler


# Set the train-hyperparams
batch_size = 448  # Batch size for training
batch_nb_tokens = batch_size * block_size
assert batch_size % len(deviceids) == 0 # for now just to make sure
microbatch_size = batch_size // len(deviceids) # micro batch size == batch size per device
microbatch_nb_tokens = microbatch_size * block_size # Number of tokens in a micro-batch
acum_steps = 1
assert microbatch_size % acum_steps == 0
nano_batch_size = microbatch_size // acum_steps
dropout = 0 # Dropout rate
nb_train_examples = None
train_sftrs = SFTRandomSampler("/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-51/data-ds-51/filtered_traced_snippets.txt", nb_train_examples, microbatch_size, block_size)
nb_batches = int(train_sftrs.nb_tokens / batch_nb_tokens)
nb_epochs = 1
max_iters = nb_batches * nb_epochs
learning_rate = 1e-3 	# Initial Learning rate value
decay_lr = True
warmup_iters = int(0.1 * max_iters)
lr_decay_iters = max_iters
assert lr_decay_iters <= max_iters
assert warmup_iters <= lr_decay_iters
min_lr = learning_rate * 0.1
beta1 = 0.9 # Adam beta1
beta2 = 0.95 # Adam beta2
weight_decay = 1e-1 # Weight decay
grad_clip = 1.0 # Gradient clipping value
cuda_sync = False
compile = True
compile_mode = 'default'

# Set the evaluation-hyperparams
if master_process:
	eval_batch_size = 128
	eval_interval = int(max_iters / 40)
	print("[*]eval_interval = ", eval_interval, "iters")
	eval_iters = 100  # Number of iterations for evaluation
	nb_eval_examples = None
	val_sftrs = SFTRandomSampler("/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-35/data-ds-35/train.txt", nb_eval_examples, eval_batch_size, block_size)

# Define function to estimate loss on train and val splits
@torch.no_grad()
def estimate_loss():
	out = {"train":-1, "val": -1}
	model.eval()
	eval_start_time = time.time()
	for sftrs in [val_sftrs]:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			past = time.time()
			X, Y, EOI_tokens_indices = sftrs.__next__()
			for i in range(len(X)):
				X[i] = tpt.encode_tokens_list(X[i])
				Y[i] = tpt.encode_tokens_list(Y[i])
			X, Y = torch.tensor(data=X, dtype=torch.int64, device=device), torch.tensor(data=Y, dtype=torch.int64, device=device)
			with torch.autocast(device_type=device, dtype=torch.bfloat16):
				logits = model(X, Y, EOI_tokens_indices)
			B, T, C = logits.shape
			unmasked_logits = []
			unmasked_targets = []
			for i , EOI_token_index in enumerate(EOI_tokens_indices):
				unmasked_logits.append(logits[i, EOI_token_index:, :])
				unmasked_targets.append(Y[i, EOI_token_index:])
			logits = torch.cat(unmasked_logits, dim=0)
			targets = torch.cat(unmasked_targets, dim=0)
			# logits = logits.view(B*T, C)
			# targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)
			losses[k] = loss.item()
			present = time.time()
			log(f"val>|ITERS: {k+1} / {eval_iters} | COMP: {(k+1)/eval_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((eval_iters-k-1) * (present-past))} | ET: {convert_seconds(time.time()-eval_start_time)}", p_level = 2)
		out["val"] = losses.mean()
	model.train()
	return out # For now let's only compute the loss on the validation split

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

# __Define the model__

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

	def forward(self, idx, targets=None, EOI_tokens_indices=None):
		B, T = idx.shape

		# idx and targets are both (B,T) tensor of integers
		tok_emb = self.token_embedding_table(idx) # (B,T,C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
		x = tok_emb + pos_emb # (B,T,C)
		x = self.blocks(x) # (B,T,C)
		x = self.ln_f(x) # (B,T,C)
		logits = self.lm_head(x) # (B,T,vocab_size)

		# if targets is None:
		# 	loss = None
		# else:
			# B, T, C = logits.shape
			# unmasked_logits = []
			# unmasked_targets = []
			# for i , EOI_token_index in enumerate(EOI_tokens_indices):
			# 	unmasked_logits.append(logits[i, EOI_token_index:, :])
			# 	unmasked_targets.append(targets[i, EOI_token_index:])
			# logits = torch.cat(unmasked_logits, dim=0)
			# targets = torch.cat(unmasked_targets, dim=0)
			# # logits = logits.view(B*T, C)
			# # targets = targets.view(B*T)
			# loss = self.compute_loss(logits, targets)

		return logits#, loss
	
	# @torch.compiler.disable()
	# def compute_loss(self, logits, targets):
	# 	return F.cross_entropy(logits, targets)
	
	# TODO: try to optimize the generate function by only computing the next token PD ...
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

# Create the model
model = GPT()
model.load_state_dict(torch.load("/data/yb2618/Tiny-Language-Models-Framework/xperiments/xpgroup-5/xp-5-18/train-5-18/atmpt-2/checkpoints/best-model.pth", map_location=device))
# Loading the model from the last checkpoint
model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_parameters_hr = human_readable(num_parameters)
if master_process:
	print(f'The model has {num_parameters_hr} trainable parameters')

# Wrap the model in DDP and/or compilation
train_model = model

# If ddp then wrap the model in DDP
if ddp:
	train_model = DDP(train_model, device_ids=[deviceids[ddp_rank]])

# If compile == True compile the model
if compile:
	if master_process: log("Compiling the model... (takes a ~minute)")
	train_model = torch.compile(train_model, mode=compile_mode)


# Initialize the optimizer
param_dict = {pn: p for pn, p in train_model.named_parameters()}
# filter out those that do not require grad
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
	{'params': decay_params, 'weight_decay': weight_decay},
	{'params': nodecay_params, 'weight_decay': 0.0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
# Create AdamW optimizer and use the fused version if it is available
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and ('cuda' in device)
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
print(f"using fused AdamW: {use_fused}")

# Set the torch precision to tf32
torch.set_float32_matmul_precision('high')

train_start_time = time.time()

if master_process:
	log("Evaluating initial loss")
	last_losses = estimate_loss()
	log(f'INITIAL LOSS: train loss {last_losses["train"]} | val loss {last_losses["val"]}')
	best_val_loss = last_losses["val"]

print('Current Device:', torch.cuda.current_device())



# Training Loop
start_iter = 0
nb_time_samples = 0
deltatimes = 0
tl_max_warmup_iters = 10
tl_warmup_iters = 0
for iter in range(start_iter, max_iters):

	# Do one forward backward pass
	past = time.time()
	
	# Clear the gradients
	optimizer.zero_grad(set_to_none=True)
	
	xb, yb, EOI_tokens_indices = train_sftrs.__next__()
	
	for i in range(len(xb)):
		xb[i] = tpt.encode_tokens_list(xb[i])
		yb[i] = tpt.encode_tokens_list(yb[i])

	xb, yb = torch.tensor(data=xb, dtype=torch.int64, device=device), torch.tensor(data=yb, dtype=torch.int64, device=device)

	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits = train_model(xb, yb, EOI_tokens_indices)
	
	B, T, C = logits.shape
	unmasked_logits = []
	unmasked_targets = []
	for i , EOI_token_index in enumerate(EOI_tokens_indices):
		unmasked_logits.append(logits[i, EOI_token_index:, :])
		unmasked_targets.append(yb[i, EOI_token_index:])
	logits = torch.cat(unmasked_logits, dim=0)
	targets = torch.cat(unmasked_targets, dim=0)
	# logits = logits.view(B*T, C)
	# targets = targets.view(B*T)
	loss = F.cross_entropy(logits, targets)

	loss.backward()

	if grad_clip != 0.0:
		torch.nn.utils.clip_grad_norm_(train_model.parameters(), grad_clip)
	
	micro_loss = loss.detach()

	if ddp:
		dist.all_reduce(micro_loss, op=dist.ReduceOp.AVG)

	lr = get_lr(iter) if decay_lr else learning_rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	optimizer.step()

	if cuda_sync:
		torch.cuda.synchronize(device)

	present = time.time()

	if master_process:
		
		if tl_warmup_iters <= tl_max_warmup_iters:
			log(f"INIT ITER TIME: {present-past}")
			log(f"|BATCH_LOSS: {micro_loss:.5f} | TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec. | ITERS: {iter+1} / {max_iters} | EPOCHS: {(iter+1)/nb_batches:.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | LR: {lr:.4f} | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {convert_seconds((max_iters-iter-1) * (present-past))} | ET: {convert_seconds(present-train_start_time)}", p_level = 1)
			print(f"|BATCH_LOSS: {micro_loss:.5f} | TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec. | ITERS: {iter+1} / {max_iters} | EPOCHS: {(iter+1)/nb_batches:.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | LR: {lr:.4f} | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {convert_seconds((max_iters-iter-1) * (present-past))} | ET: {convert_seconds(present-train_start_time)}")
			tl_warmup_iters += 1
		else:
			deltatimes += present-past
			nb_time_samples += 1
			mean_deltatime = deltatimes / nb_time_samples # mean deltatime
			log(f"|BATCH_LOSS: {micro_loss:.5f} | TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec. | ITERS: {iter+1} / {max_iters} | EPOCHS: {(iter+1)/nb_batches:.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | LR: {lr:.4f} it./s | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {convert_seconds((max_iters-iter-1) * mean_deltatime)} | ET: {convert_seconds(present-train_start_time)}", p_level = 1)
			print(f"|BATCH_LOSS: {micro_loss:.5f}|TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec.|ITERS: {iter+1} / {max_iters}|EPOCHS: {(iter+1)/nb_batches:.2f}|COMP: {(iter+1)/max_iters * 100:.2f}%|LR: {lr:.4f} it./s|SPD: {(present - past) * 1000 :.4f} ms/it|ERT: {convert_seconds((max_iters-iter-1) * mean_deltatime)}|ET: {convert_seconds(present-train_start_time)}")
		
		batch_log.write(f"BATCH_LOSS: {micro_loss:.5f}\n")
		batch_log.flush()
	
	# If we reach the evaluation interval, evaluate the model on train.bin and val.bin and checkpoint it
	if master_process and ((iter+1) % eval_interval == 0 or (iter == max_iters - 1)):
		
		log("\nEvaluation and checkpointing ...\n")
		epoch = (iter+1)/nb_batches
		
		torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch:.2f}.pth")
		torch.save(optimizer.state_dict(), f"checkpoints/optimizer_{epoch:.2f}.pth")
		
		losses = estimate_loss()

		date_hour = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
		log(f'\ncheckpoint_{date_hour} : iter {iter+1:10d} | epoch {epoch:.2f} | train loss {losses["train"]:.10f} | val loss {losses["val"]:.10f}\n')
		
		# Save the last checkpoint
		with open(f"checkpoints/checkpoint_{epoch:.2f}.info", "w") as f:
			f.write(f"iter : {iter+1}\n")
			f.write(f"epoch : {epoch:.2f}\n")
			f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
			f.write(f"date-hour : {date_hour}\n")
		
		# If this is the best model, save it
		if losses["val"] < best_val_loss:
			best_val_loss = losses["val"]
			torch.save(model.state_dict(), f"checkpoints/best-model.pth")
			with open(f"best-model.info", "w") as f:
				f.write(f"iter : {iter+1}\n")
				f.write(f"epoch : {epoch:.2f}\n")
				f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
				f.write(f"date-hour : {date_hour}\n")

if ddp:
	destroy_process_group()

if master_process:
	batch_log.close()
	log_file.close()