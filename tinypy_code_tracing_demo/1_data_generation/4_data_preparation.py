import random
import numpy as np
import gc
from tinypy_code_tracer_tokenizer import TinypyTokenizer

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# Initialize the tokenizer
print("[*] Initializing the tokenizer ...")
tpt = TinypyTokenizer()

# Load the dataset of traced snippets
print("[*] Loading the dataset of traced snippets ...")
with open("./data/determinism_filtered_snippets.txt", "r") as f:
	data = f.read()

# Split dataset by examples
print("[*] Splitting dataset by examples ...")
examples = data.split("\n\n")[:-1]

# Free memory from the loaded dataset
print("[*] Freeing memory from loaded dataset ...")
del data
gc.collect()

# Print the number of examples
print(f"[*] Total number of examples: {len(examples):,}")

# Set the proportions for train, validation, and test splits
print("[*] Setting the proportions for train, validation, and test splits ...")
train_number = int(len(examples) * 0.8)
val_number = int(len(examples) * 0.1)
test_number = len(examples) - train_number - val_number

# Creating the train dataset
print("[*] Creating the train dataset ...")
train_out_path = "./data/train.txt"
train_bin_out_path = "./data/train.bin"
train_examples = examples[:train_number]
print(f"[*] There are {len(train_examples)} train examples.")
train_data = "\n\n".join(train_examples) + "\n\n"
del train_examples
print("[*] Writing the train dataset to train.txt")
with open(train_out_path, 'w') as f:
	f.write(train_data)
del train_data

# We generate the tokenized file of train.txt in train.bin
print("[*] We generate the tokenized file of train.txt in val.bin")
print(tpt.encode_to_file(train_out_path, train_bin_out_path))


# Creating the validation dataset
print("[*] Creating the validation dataset ...")
val_out_path = "./data/val.txt"
val_bin_out_path = "./data/val.bin"
val_examples = examples[train_number:train_number+val_number]
print(f"[*] There are {len(val_examples)} validation examples.")
val_data = "\n\n".join(val_examples) + "\n\n"
del val_examples
print("[*] writing the validation dataset to val.txt")
with open(val_out_path, 'w') as f:
	f.write(val_data)
del val_data
	
# We generate the tokenized file of val.txt in val.bin
print("[*] We generate the tokenized file of val.txt in val.bin")
print(tpt.encode_to_file(val_out_path, val_bin_out_path))

# Creating the test dataset
print("[*] Creating the test dataset ...")
test_out_path = "./data/test.txt"
test_examples = examples[-test_number:]
print(f"There are {len(test_examples)} test examples")
test_data = "\n\n".join(test_examples) + "\n\n"
del test_examples
print("[*] Writing the test dataset to test.txt")
with open(test_out_path, 'w') as f:
	f.write(test_data)
del test_data

print("[*] Freeing examples memory ...")
del examples
gc.collect()

# Create the vocab_size.txt file
print("[*] Creating the vocab_size.txt file ...")
voc_size_path = "./data/vocab_size.txt"
with open(voc_size_path, "w") as f:
	f.write(str(len(tpt.keywords)))