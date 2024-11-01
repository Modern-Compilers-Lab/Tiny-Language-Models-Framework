## Data preping (on Kindi)

## Imports
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
with open("../../data-ds-8/data.txt", "r") as f:
	data = f.read()


## Get all the unique characters that occur in this text
log("Get all the unique characters that occur in this text")
chars = sorted(list(set(data)))
vocab_size = len(chars)
log("all the unique characters: " + repr(''.join(chars)))
log(f"vocab size: {vocab_size:,}")


## Create a mapping from characters to integers
log("Create a mapping from characters to integers")
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


## Save the meta information as well, to help us encode/decode later
log("Save the meta information as well, to help us encode/decode later")
meta = {
	'vocab_size': vocab_size,
	'itos': itos,
	'stoi': stoi,
}
with open('data-dp-8-1/meta.pkl', 'wb') as f:
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
log(f"train_data has {(train_tokens := len(train_data))} tokens")
log("writing the train_data to train.txt")
with open("data-dp-8-1/train.txt", 'w') as f:
	f.write(train_data)
del train_data

log("creating the val_examples")
val_examples = examples[int(n*0.8):int(n*0.9)]
log(f"val_examples has {len(val_examples)} examples")
log("creating the val_data")
val_data = "\n\n".join(val_examples)
del val_examples
log(f"val_data has {(val_tokens := len(val_data))} tokens")
log("writing the val_data to val.txt")
with open("data-dp-8-1/val.txt", 'w') as f:
	f.write(val_data)
del val_data

log("creating the test_examples")
test_examples = examples[int(n*0.9):]
log(f"test_examples has {len(test_examples)} examples")
log("creating the test_data")
test_data = "\n\n".join(test_examples)
del test_examples
log(f"test_data has {len(test_data)} tokens")
log("writing the test_data to test.txt")
with open("data-dp-8-1/test.txt", 'w') as f:
	f.write(test_data)
del test_data

log("freeing examples memory")
del examples
gc.collect()

## We read the stoi from the meta.pkl
with open("data-dp-8-1/meta.pkl", "rb") as f:
	meta = pickle.load(f)
stoi = meta["stoi"]
del meta

## We define the encoding function
log("We define the encoding function")
def encode_generator(s:str):
	for c in s:
		yield stoi[c]

log("Reading and encoding train.txt directly to binary")
with open("data-dp-8-1/train.txt", "r") as f, open("data-dp-8-1/train.bin", "wb") as bin_file:
	chunk_size = 1024 * 1024 * 1000  # 1 GB
	max_iters = math.ceil(train_tokens/chunk_size)
	i = 0
	while True:
		past = time.time()
		chunk = f.read(chunk_size)
		if not chunk:
			break
		for token in encode_generator(chunk):
			bin_file.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
		i = i+1
		present = time.time()
		log(f"|ITERS: {i} / {max_iters} | COMP: {(i)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((max_iters-i) * (present-past))}", p_level = 1)

log("Reading and encoding val.txt directly to binary")
with open("data-dp-8-1/val.txt", "r") as f, open("data-dp-8-1/val.bin", "wb") as bin_file:
	chunk_size = 1024 * 1024 * 1000  # 1 GB
	max_iters = math.ceil(val_tokens/chunk_size)
	i = 0
	while True:
		past = time.time()
		chunk = f.read(chunk_size)
		if not chunk:
			break
		for token in encode_generator(chunk):
			bin_file.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
		i = i+1
		present = time.time()
		log(f"|ITERS: {i} / {max_iters} | COMP: {(i)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((max_iters-i) * (present-past))}", p_level = 2)

log_file.close()