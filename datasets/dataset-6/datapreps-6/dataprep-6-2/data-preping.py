## Data preping (on Kindi)

## Imports
from transformers import AutoTokenizer
import pickle
import numpy as np
import gc
import struct
import time
import math

## Logging boilerplate
log_file = open("data-preping.log", "w")
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


## Initialize CodeLlama tokenizer
log("Initializing CodeLlama tokenizer")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
vocab_size = tokenizer.vocab_size
log(f"Vocabulary size: {vocab_size:,}")

## Saving the numpy random state
log("Saving the data-preping-numpy-random state")
log("saving it")
np_random_state = np.random.get_state()
with open("data-preping-np-random-state.bin", "wb") as f:
	pickle.dump(np_random_state, f)
log("freeing its memory")
del np_random_state
gc.collect()


## Loading the dataset
log("Loading the dataset")
with open("../../data-ds-6/data.txt", "r") as f:
	data = f.read()


## Save the meta information
log("Save the meta information")
meta = {
    'vocab_size': vocab_size,
    'tokenizer_name': "codellama/CodeLlama-7b-hf",
}

with open('data-dp-6-2/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
log("freeing its memory")
del meta
gc.collect()


## Split by examples using \n\n
log("Split by examples using \\n\\n")
log("splitting")
examples = data.split("\n\n")[:-1]
log("freeing data memory")
del data
gc.collect()
n = len(examples)
log(f"total number of examples: {n:,}\n")


## Creating the train.txt, val.txt and test.txt
log("Creating the train.txt, val.txt and test.txt")
log("shuffling examples")
np.random.shuffle(examples)

log("creating the train_examples")
train_examples = examples[:int(n*0.8)]
log(f"train_examples has {len(train_examples)} examples")
log("creating the train_data")
train_data = "\n\n".join(train_examples)
del train_examples

log("writing the train_data to train.txt")
with open("data-dp-6-2/train.txt", 'w') as f:
	f.write(train_data)

# Tokenize and save train data to binary
log("Tokenizing and saving train data to binary")
tokens = tokenizer.encode(train_data)
del train_data
log(f"Encoded train data has {len(tokens)} tokens")
with open("data-dp-6-2/train.bin", "wb") as f:
    for token in tokens:
        f.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
del tokens

# Process validation split
log("Processing validation split")
val_examples = examples[int(n*0.8):int(n*0.9)]
log(f"val_examples has {len(val_examples)} examples")
val_data = "\n\n".join(val_examples)
del val_examples
log(f"val_data has {(val_tokens := len(val_data))} characters")
log("writing the val_data to val.txt")
with open("data-dp-6-2/val.txt", 'w') as f:
    f.write(val_data)

# Tokenize and save validation data to binary
log("Tokenizing and saving validation data to binary")
tokens = tokenizer.encode(val_data)
del val_data
log(f"Encoded validation data has {len(tokens)} tokens")
with open("data-dp-6-2/val.bin", "wb") as f:
    for token in tokens:
        f.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
del tokens

# Process test split
log("Processing test split")
test_examples = examples[int(n*0.9):]
log(f"test_examples has {len(test_examples)} examples")
test_data = "\n\n".join(test_examples)
del test_examples
log(f"test_data has {len(test_data)} characters")
log("writing the test_data to test.txt")
with open("data-dp-6-2/test.txt", 'w') as f:
    f.write(test_data)

# Tokenize and save test data to binary
log("Tokenizing and saving test data to binary")
tokens = tokenizer.encode(test_data)
del test_data
log(f"Encoded test data has {len(tokens)} tokens")
with open("data-dp-6-2/test.bin", "wb") as f:
    for token in tokens:
        f.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
del tokens

log("freeing examples memory")
del examples
gc.collect()

log("Data preparation complete!")
log_file.close()