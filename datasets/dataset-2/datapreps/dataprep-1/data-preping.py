# Data preping (on Greene)
DIR = "/scratch/yb2618/Tiny-Language-Models-Framework/datasets/dataset-2/datapreps/dataprep-1"

import pickle
import numpy as np
import gc

# Logging boilerplate
log_file = open(DIR+"data-preping-log.txt", "w")
def log(s:str):
	if s[0].upper() == s[0]:
		start = "\n"
		end = ":"
	else:
		start = "	--> "
		end = ""
	log_file.write(start + s + end + "\n")
	log_file.flush()

# Saving the numpy random state
log("Saving the numpy random state")
log("saving it")
np_random_state = np.random.get_state()
with open(DIR+"data/np-random-state.bin", "wb") as f:
	pickle.dump(np_random_state, f)
log("freeing its memory")
del np_random_state
gc.collect()

# Loading the dataset
log("Loading the dataset")
with open(DIR+"datasets/dataset-2/data/data.txt", "r") as f:
	data = f.read()

# Get all the unique characters that occur in this text
log("Get all the unique characters that occur in this text")
chars = sorted(list(set(data)))
vocab_size = len(chars)
log("all the unique characters: " + repr(''.join(chars)))
log(f"vocab size: {vocab_size:,}")

# Create a mapping from characters to integers
log("Create a mapping from characters to integers")
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Save the meta information as well, to help us encode/decode later
log("Save the meta information as well, to help us encode/decode later")
meta = {
	'vocab_size': vocab_size,
	'itos': itos,
	'stoi': stoi,
}
with open(DIR+'data/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
log("freeing its memory")
del meta
gc.collect()

# Split by examples using \n\n
log("Split by examples using \\n\\n")
log("splitting")
examples = data.split("\n\n")[:-1]
log("freeing data memory")
del data
gc.collect()
n = len(examples)
log(f"total number of examples: {n:,}\n")

# Shuffle the examples adn split into train, test and val
log("Shuffle the examples adn split into train, test and val")
log("shuffling")
np.random.shuffle(examples)
train_examples = examples[:int(n*0.8)]
val_examples = examples[int(n*0.8):int(n*0.9)]
test_examples = examples[int(n*0.9):]
log("freeing examples memory")
del examples
gc.collect()

# Join the examples back into strings
log("Join the examples back into strings")
train_data = "\n\n".join(train_examples)
train_examples_len = len(train_examples)
del train_examples
gc.collect()
val_data = "\n\n".join(val_examples)
val_examples_len = len(val_examples)
del val_examples
gc.collect()
test_data = "\n\n".join(test_examples)
test_examples_len = len(test_examples)
del test_examples
gc.collect()

# Save train.txt, val.txt, and test.txt sets to separate files
log("Save train, val, and test sets to separate files")
with open(DIR+"data/train.txt", 'w') as f:
	f.write(train_data)
with open(DIR+"data/val.txt", 'w') as f:
	f.write(val_data)
with open(DIR+"data/test.txt", 'w') as f:
	f.write(test_data)

# We define the encoding function
log("We define the encoding function")
def encode(s:str)->str:
	return [stoi[c] for c in s]
     
# Encode both to integers
log("Encode both to integers")
log("encoding train_data")
train_ids = encode(train_data)
del train_data
gc.collect()
log("encoding val_data")
val_ids = encode(val_data)
del val_data
gc.collect()

log(f"train has {len(train_ids):,} tokens for {train_examples_len:,} examples")
log(f"val has {len(val_ids):,} tokens for {val_examples_len:,} examples")

# Export to bin files
log("Export to bin files\n")
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(DIR+"data/train.bin")
del train_ids
gc.collect()
val_ids = np.array(val_ids, dtype=np.uint16)
val_ids.tofile(DIR+"data/val.bin")
del val_ids
gc.collect()

log_file.close()