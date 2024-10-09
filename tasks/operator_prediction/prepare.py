import os
import pickle
import requests
import numpy as np

from io import StringIO
from contextlib import redirect_stdout

input_file_path = os.path.join(os.path.dirname(__file__),  'output_data.txt' )

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}\n")


# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(f'meta.pkl', 'wb') as f:
    pickle.dump(meta, f)


# split by examples using "\n\n"
examples = data.split("\n\n")[:-1]
n = len(examples)
print(f"total number of examples: {n:,}\n")
# shuffle the examples
np.random.shuffle(examples)

# split into train, val, and test sets
train_examples = examples[:int(n*0.8)]
val_examples = examples[int(n*0.8):int(n*0.9)]
test_examples = examples[int(n*0.9):]

# join the examples back into strings
train_data = "\n\n".join(train_examples)
val_data = "\n\n".join(val_examples)
test_data = "\n\n".join(test_examples)



# Save train, val, and test sets to separate files
with open(os.path.join(os.path.dirname(__file__), 'train.txt'), 'w') as f:
    f.write(train_data)
with open(os.path.join(os.path.dirname(__file__), 'val.txt'), 'w') as f:
    f.write(val_data)
with open(os.path.join(os.path.dirname(__file__), 'test.txt'), 'w') as f:
    f.write(test_data)




# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens for {len(train_examples):,} examples")
print(f"val has {len(val_ids):,} tokens for {len(val_examples):,} examples")
print(f"test has {len(test_ids):,} tokens for {len(test_examples):,} examples\n")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__),   'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

