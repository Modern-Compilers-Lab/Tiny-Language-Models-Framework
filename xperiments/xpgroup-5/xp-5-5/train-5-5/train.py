import 	random
import 	os
import 	time
import 	torch
import 	torch.nn 			as nn
import 	torch.nn.functional as F
import 	numpy 				as np
import 	torch.distributed 	as dist
from 	torch.nn.parallel 	import DistributedDataParallel as DDP

# Set some general constants
DDIR 		= "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-20/datapreps-20/dataprep-20-1/data-dp-20-1/"
deviceids 	= [6,7]

# Open the log file
# log_file = open("train.log", "w")

# Create the log boilerplate for progress bars
# pbar_recept_string = " " * 200 + "\n"
# log_file.write(pbar_recept_string)
# log_file.write(pbar_recept_string)
# log_file.flush()

# Def. the log function
# def log(s:str, p_level=None):
# 	if p_level == 1:
# 		log_file.seek(0,0)
# 		log_file.write(pbar_recept_string)
# 		log_file.seek(0,0)
# 		log_file.write(s)
# 		log_file.seek(0,2)
# 	elif p_level == 2:
# 		log_file.seek(len(pbar_recept_string), 0)
# 		log_file.write(pbar_recept_string)
# 		log_file.seek(len(pbar_recept_string), 0)
# 		log_file.write(s)
# 		log_file.seek(0,2)
# 	else:
# 		if s[0].upper() == s[0]:
# 			start = "\n"
# 			end = ":"
# 		else:
# 			start = "	--> "
# 			end = ""
# 		log_file.write(start + s + end + "\n")
# 	log_file.flush()

# Def. function to convert seconds to days, hours, minutes, seconds
def convert_seconds(seconds:float):
	# ignoring the sub seconds 
	seconds = int(seconds)
	days, seconds = divmod(seconds, 86400)
	hours, seconds = divmod(seconds, 3600)
	minutes, seconds = divmod(seconds, 60)
	return (days, hours, minutes, seconds)

# Set the random seed for reproducibility
# log("Set the random seed for reproducibility")
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Set arch-hyperparams for the GPT model
# log("Setting arch-hyperparams for the GPT model")
block_size 	= 512 	# Maximum context length
n_embd 		= 192	# Embedding dimension
n_head 		= 16	# Number of attention heads
n_layer 	= 24	# Number of transformer blocks
with open(DDIR+"vocab_size.txt", "rb") as f:
	vocab_size = int(f.read())
# Set the vocab size manually to 128 to round up to the nearest power of two for optimized training
# vocab_size = 128
# log(f"vocab_size: {vocab_size}")

# Load the training and evaluation data
# log("Loading the training and evaluation data")
# log("loading train.bin")
before 		= time.time()
train_data 	= np.memmap(DDIR+"train.bin", dtype = np.uint16, mode="r")
train_data 	= np.array(train_data)
after 		= time.time()
# log(f"took {convert_seconds(after - before)}")

# log("loading val.bin")
before 		= time.time()
val_data 	= np.memmap(DDIR+"val.bin", dtype = np.uint16, mode="r")
val_data 	= np.array(val_data)
after 		= time.time()
# log(f"took {convert_seconds(after - before)}")

# Set the train-hyperparams
# log("Setting train-hyperparams and util variables")
batch_size = 128   	# Batch size for training
batch_nb_tokens = batch_size * block_size
assert batch_size % len(deviceids) == 0 # for now just to make sure
microbatch_size = batch_size // len(deviceids) # micro batch size == batch size per device
microbatch_nb_tokens = microbatch_size * block_size # Number of tokens in a micro-batch
dropout = 0	   	# Dropout rate
max_pseudo_epochs = 1.7	# Number of pseudo-epochs to train for
learning_rate = 1e-3 	# Initial Learning rate value
max_degradations = -1 	# number of consecutive degradations on val loss before stoping the training
max_iters = int( ( max_pseudo_epochs * len(train_data) ) / ( batch_size * block_size ) )
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]  # Milestones for learning rate decay as fractions of max_iters
compile = False

# Set the evaluation-hyperparams
eval_interval = 5000 # Evaluation interval
eval_iters = 500  # Number of iterations for evaluation

# Define the model
# log("Defining the model and utilities")

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

# Set the multi-GPU training setup
from torch.distributed import init_process_group, destroy_process_group

# Check if this is a ddp run
ddp = int(os.environ.get('RANK', -1)) != -1
# if ddp than setup per process control variables
if ddp:
	assert torch.cuda.is_available()
	init_process_group(backend='nccl')
	ddp_rank = int(os.environ['RANK'])
	ddp_local_rank = int(os.environ['LOCAL_RANK'])
	ddp_world_size = int(os.environ['WORLD_SIZE'])
	assert ddp_world_size == len(deviceids) # for now just to make sure
	device = f'cuda:{deviceids[ddp_rank]}'
	torch.cuda.set_device(device)
	master_process = ddp_rank == 0
else:
	master_process = True
	ddp_rank = 0
	ddp_world_size = 1
	device = f'cuda:{deviceids[ddp_rank]}'

# Test the ddp training schema
# print(f"ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, has taken {device}, ddp: {ddp}, master_process: {master_process}, ddp_world_size: {ddp_world_size}")
# destroy_process_group()
# import sys; sys.exit(0)

# Def. get random batch of data
def get_batch(split):
	data = train_data if split == 'train' else val_data
	idx = iter * batch_nb_tokens + ddp_rank * microbatch_nb_tokens
	# In case the process batch is out of the data range, reset the index
	if idx + microbatch_nb_tokens + 1 > len(data):
		idx = 0
	x = torch.from_numpy((data[idx:idx+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
	y = torch.from_numpy((data[idx+1:idx+1+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
	x, y = x.to(device), y.to(device)
	return x, y

# Def. estimate loss on train and val splits
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
			# log(f"{split}>|ITERS: {k+1} / {eval_iters} | COMP: {(k+1)/eval_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((eval_iters-k-1) * (present-past))} |", p_level = 2)
		out[split] = losses.mean()
	model.train()
	return out

# Def. helper function to make large numbers of parameters human-readable
def human_readable(num):
	magnitude = 0
	while abs(num) >= 1000:
		magnitude += 1
		num /= 1000.0
	return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# Create the model
# log("Creating the model")
model = GPT()
model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_parameters_hr = human_readable(num_parameters)
# log(f'The model has {num_parameters_hr} trainable parameters')

# If compile == True compile the model
if compile:
	print("compiling the model... (takes a ~minute)")
	# train_model = torch.compile(model, mode='max-autotune')
	train_model = torch.compile(model)
else:
	train_model = model

# If ddp then wrap the model in DDP
if ddp:
	train_model = DDP(model, device_ids=[deviceids[ddp_rank]])

# Initialize the optimizer
# log("initialiazing the optimizer")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize the learning rate scheduler
# log("initializing the learing rate scheduler")
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)


# Create the checkpoints directory
# log("training ...")

# Set the torch precision to tf32
torch.set_float32_matmul_precision('high')

# Create the train_start_time
train_start_time = time.time()

eval_interval = 100

# Launch the training loop
for iter in range(max_iters):
	
	# Do one forward backward pass
	past = time.time()
	xb, yb = get_batch('train')
	# with torch.autocast(device_type=device, dtype=torch.bfloat16):
	logits, loss = train_model(xb, yb)
	local_loss = loss.detach()
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	if ddp:
		dist.all_reduce(local_loss, op=dist.ReduceOp.AVG)
	optimizer.step()
	torch.cuda.synchronize()
	# scheduler.step()
	present = time.time()

	# Log the training progess bar
	if master_process:
		# log(f"|ITERS: {iter+1} / {max_iters} | EPOCHS: {(block_size * batch_size * (iter+1))/len(train_data):.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {convert_seconds((max_iters-iter-1) * (present-past))} | ET: {convert_seconds(present-train_start_time)}", p_level = 1)
		print(f"|TRAIN_LOSS: {local_loss:.5f} | ITERS: {iter+1:{len(str(max_iters))}} / {max_iters} | EPOCHS: {(block_size * batch_size * (iter+1))/len(train_data):.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {str(convert_seconds((max_iters-iter-1) * (present-past))):20} | ET: {str(convert_seconds(present-train_start_time)):20}")

if ddp:
	destroy_process_group()

# log_file.close()