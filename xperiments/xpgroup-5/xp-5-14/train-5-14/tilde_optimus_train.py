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


# print('hello')
# import sys; sys.exit(0)
# __Setup the multi-GPU training__

# Set the device ids
deviceids 	= [4]

# Check if this is a ddp run
ddp = int(os.environ.get('RANK', -1)) != -1

# If ddp than setup per process control variables
if ddp:
	print("ddp is set to True")
	from torch.distributed import init_process_group, destroy_process_group
	assert torch.cuda.is_available()
	init_process_group(backend='nccl')
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
	pbar_recept_string = " " * 500 + "\n"
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
DDIR = "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-25/datapreps-25/dataprep-25-1/data-dp-25-1/"

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Set arch-hyperparams for the GPT model
block_size 	= 512 	# Maximum context length
n_embd 		= 320	# Embedding dimension
n_head 		= 16	# Number of attention heads
n_layer 	= 24	# Number of transformer blocks
with open(DDIR+"vocab_size.txt", "rb") as f:
	vocab_size = int(f.read())
# Set the vocab size manually to 128 to round up to the nearest power of two for optimized training
# vocab_size = 128
if master_process: log(f"vocab_size: {vocab_size}")

# Load train.bin
if master_process: log("Loading train.bin")
before 		= time.time()
train_data 	= np.memmap(DDIR+"train.bin", dtype = np.uint16, mode="r")
# train_data 	= np.array(train_data)
after 		= time.time()
if master_process: log(f"took {convert_seconds(after - before)}")
# import sys;sys.exit(0)
# Load val.bin
if master_process: log("Loading val.bin")
before 		= time.time()
val_data 	= np.memmap(DDIR+"val.bin", dtype = np.uint16, mode="r")
# val_data 	= np.array(val_data[:10_000_000])
after 		= time.time()
if master_process: log(f"took {convert_seconds(after - before)}")

# Set the train-hyperparams
batch_size = 128   	# Batch size for training
batch_nb_tokens = batch_size * block_size
assert batch_size % len(deviceids) == 0 # for now just to make sure
microbatch_size = batch_size // len(deviceids) # micro batch size == batch size per device
microbatch_nb_tokens = microbatch_size * block_size # Number of tokens in a micro-batch
acum_steps = 1
assert microbatch_size % acum_steps == 0
nano_batch_size = microbatch_size // acum_steps
dropout = 0 # Dropout rate
max_pseudo_epochs = 2	# Number of pseudo-epochs to train for
max_iters = int( ( max_pseudo_epochs * len(train_data) ) / ( batch_size * block_size ) )
log(f"max_iters: {max_iters}")
epoch_iters = int(len(train_data) / (batch_size * block_size))
log(f"epoch_iters: {epoch_iters}")
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
deepening_intervals = [max_iters//n_layer for _ in range(n_layer)]
deepening_intervals = np.cumsum(deepening_intervals)[:-1]
if master_process: log(f'deepening_intervals: {deepening_intervals}')
reset_lmhead = False
cuda_sync = True
compile = True
compile_mode = 'reduce-overhead'

# Set the evaluation-hyperparams
eval_interval = int(max_iters * 0.05) # Evaluation interval
eval_iters = 250  # Number of iterations for evaluation

# __Define some utils__

# Define function to get batch of token chunks
def get_batch(split):
	data = train_data if split == 'train' else val_data
	loc_iter = iter % epoch_iters
	idx = loc_iter * batch_nb_tokens + ddp_rank * microbatch_nb_tokens
	# In case the process batch is out of the data range, reset the index
	x = torch.from_numpy((data[idx:idx+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
	y = torch.from_numpy((data[idx+1:idx+1+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
	x, y = x.to(device), y.to(device)
	return x, y

# Define the evaluation batch size
eval_batch_size = 128
eval_batch_nb_tokens = eval_batch_size * block_size

# Define the evaluation batch loader
def load_eval_batch(split):
	data = train_data if split == 'train' else val_data
	idx = random.randint(0, len(data) - eval_batch_nb_tokens - 1)
	x = torch.from_numpy((data[idx:idx+eval_batch_nb_tokens]).astype(np.int64)).view(eval_batch_size, block_size)
	y = torch.from_numpy((data[idx+1:idx+1+eval_batch_nb_tokens]).astype(np.int64)).view(eval_batch_size, block_size)
	x, y = x.to(device), y.to(device)
	return x, y

# Define function to estimate loss on train and val splits
@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	eval_start_time = time.time()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			t0 = time.time()
			X, Y = load_eval_batch(split)
			with torch.autocast(device_type=device, dtype=torch.bfloat16):
				logits, loss = model(X, Y)
			losses[k] = loss.item()
			t1 = time.time()
			log(f"{split}>|ITERS: {k+1} / {eval_iters} | COMP: {(k+1)/eval_iters * 100:.2f}% | RATE: {1/(t1-t0):.2f} it./s | SPD: {t1 - t0 :.4f} s/it.| ERT: {convert_seconds((eval_iters-k-1) * (t1-t0))} | ET: {convert_seconds(time.time()-eval_start_time)}", p_level = 2)
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

	def __init__(self, init_nb_blocks=1):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(init_nb_blocks)])
		self.ln_f = nn.LayerNorm(n_embd, bias=False)
		self.lm_head = nn.Linear(n_embd, vocab_size)
	
	def add_block(self, reset_lmhead=False):
		new_block = Block(n_embd, n_head=n_head)
		new_block.to(device)
		self.blocks.append(new_block)
		if reset_lmhead:
			self.lm_head = nn.Linear(n_embd, vocab_size, device=device)
		
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
# optimizer = torch.optim.AdamW(train_model.parameters(), lr=learning_rate)
def init_optimizer(train_model:nn.Module, lr):
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
	optimizer = torch.optim.AdamW(optim_groups, lr=lr , betas=(beta1, beta2), **extra_args)
	print(f"using fused AdamW: {use_fused}")
	return optimizer

optimizer = init_optimizer(train_model, lr=learning_rate)

# Initialize the learning rate scheduler
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)

# Set the torch precision to tf32
torch.set_float32_matmul_precision('high')

# Set the number of initial number of iterations during which the ERT is estimated based on last batch time only
init_iters = 20

# Set the deltatimes to 0 to compute the mean deltatime
deltatimes = 0

train_start_time = time.time()

if master_process:
	log("Evaluating initial loss")
	last_losses = {'train':10, 'val':10}# estimate_loss()
	best_val_loss = last_losses["val"]
	log(f'INITIAL LOSS: train loss {last_losses["train"]} | val loss {last_losses["val"]}')

print('Current Device:', torch.cuda.current_device())


# Launch the training loop
time_samples = 0
tl_warmups = 0
max_tl_warmups = 20
for iter in range(max_iters):

	# If we reach the deepening interval, add a block to the model
	if iter in deepening_intervals:
		if master_process:
			log('------------------------------------------------------------------------------------------------------------------------')
			log(f"Deepening the model from {len(model.blocks)} to {len(model.blocks)+1} blocks")
			log(f'Checkpointing the model first')
			losses = estimate_loss()
			log(f'checkpoint : iter: {iter:10d} | epoch: {epochs:.2f} | depth: {len(model.blocks)} | train loss: {losses["train"]:.10f} | val loss: {losses["val"]:.10f}')
			# Save the last checkpoint
			torch.save(model.state_dict(), f"checkpoints/checkpoint_depth_{len(model.blocks)}.pth")
			torch.save(optimizer.state_dict(), f"checkpoints/optimizer_depth_{len(model.blocks)}.pth")
			with open(f"checkpoints/checkpoint_depth_{len(model.blocks)}.info", "w") as f:
				f.write(f"depth: {len(model.blocks)}\n")
				f.write(f"iter : {iter}\n")
				f.write(f"epoch : {epochs}\n")
				f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
			log(f'Adding the new block, ddp = {ddp}, compile = {compile}')
		model.add_block(reset_lmhead=reset_lmhead)
		torch._dynamo.reset()
		train_model = model
		if ddp:
			if master_process: log("Re-wrapping the model in DDP...")
			train_model = DDP(train_model, device_ids=[deviceids[ddp_rank]])
		if compile:
			if master_process: log("Re-Compiling the model... (takes a ~minute)")
			train_model = torch.compile(train_model, mode=compile_mode)
		optimizer = init_optimizer(train_model, lr=learning_rate)
		if master_process:
			log(f"Model deepened to {len(train_model.blocks)} blocks")
			log('------------------------------------------------------------------------------------------------------------------------')
	
	# Do one forward backward pass
	t0 = time.time()
	
	# Clear the gradients
	optimizer.zero_grad(set_to_none=True)
	
	xb, yb = get_batch('train')

	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits, loss = train_model(xb, yb)

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

	t1 = time.time()

	# Logging, evaluating, and checkpointing
	if master_process:
		
		iter_time = t1 - t0
		if tl_warmups < max_tl_warmups:
			tl_warmups += 1
		else:
			time_samples += 1
			deltatimes += iter_time
			iter_time = deltatimes / time_samples
		
		itnd = deepening_intervals[len(model.blocks)] - iter - 1 # iterations to next deepening
		ttnd = convert_seconds((deepening_intervals[len(model.blocks)] - iter - 1) * (iter_time)) # time to next deepening
		itne = eval_interval - (iter % eval_interval) - 1 # iterations to next evaluation
		ttne = convert_seconds((eval_interval - (iter % eval_interval) - 1) * (iter_time)) # time to next evaluation
		ert = convert_seconds((max_iters-iter-1) * (iter_time)) # estimated remaining time
		et = convert_seconds(t1-train_start_time) # elapsed time
		tps = int(batch_size*block_size/(t1-t0)) # tokens per second
		epochs = (block_size * batch_size * (iter+1))/len(train_data) # epochs
		comp = (iter+1)/max_iters * 100 # completion
		rate = 1/(t1-t0) # iteration rate
		spd = (t1 - t0) * 1000 # seconds per iteration (speed)
		
		info_string = f"|BATCH LOSS: {micro_loss:.5f} | DEPTH: {len(model.blocks)} | TPS: {tps:8} tokens/sec. | LR: {lr:.6f} | ITERS: {iter+1} / {max_iters} | EPOCHS: {epochs:.2f} | COMP: {comp:.2f}% | RATE: {rate:.2f} it./s | SPD: {spd:.4f} ms/it.| ITND: {itnd} | TTND: {ttnd} | ITNE: {itne} | TTNE: {ttne} | ERT: {ert} | ET: {et}"
		log(info_string, p_level = 1)
		batch_log.write(info_string+'\n')
		batch_log.flush()
	
		# If we reach the evaluation interval, evaluate the model on train.bin and val.bin and checkpoint it
		if (iter+1) % eval_interval == 0 or (iter == max_iters - 1):
			
			log("Evaluating and checkpointing ...")
			
			losses = estimate_loss()
			log(f'checkpoint : iter: {iter:10d} | epoch: {epochs:.2f} | depth: {len(model.blocks)} | train loss: {losses["train"]:.10f} | val loss: {losses["val"]:.10f}')
			# Save the last checkpoint
			torch.save(model.state_dict(), f"checkpoints/checkpoint.pth")
			torch.save(optimizer.state_dict(), "checkpoints/optimizer.pth")
			with open(f"checkpoint.info", "w") as f:
				f.write(f"depth: {len(model.blocks)}\n")
				f.write(f"iter : {iter}\n")
				f.write(f"epoch : {epochs}\n")
				f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
				# f.write(f"date-hour : {date_hour}\n")
			
			# If this is the best model, save it
			if losses["val"] < best_val_loss:
				best_val_loss = losses["val"]
				torch.save(model.state_dict(), f"checkpoints/best-model.pth")
				with open(f"best-model.info", "w") as f:
					f.write(f"depth: {len(model.blocks)}\n")
					f.write(f"iter : {iter}\n")
					f.write(f"epoch : {epochs}\n")
					f.write(f"losses:\n	train {losses['train']}\n	val {losses['val']}")
					# f.write(f"date-hour : {date_hour}\n")

if ddp:
	destroy_process_group()

if master_process:
	batch_log.close()
	log_file.close()