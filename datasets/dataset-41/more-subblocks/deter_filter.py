from tqdm import tqdm
from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer

tpt = TinypyTokenizer()

src_path ="./more-subblocks-data/traced_snippets.txt"
out_path = "./more-subblocks-data/traced_snippets_filtered.txt"
block_size = 512

print('Reading data ...')
with open(src_path, 'r') as f:
	data = f.read()

print("Splitting examples ...")
examples = data.split('\n\n')[:-1]
print("Number of examples:", len(examples))

print("Purging examples that have too long step transitions ...")
valid_examples = []
for example in tqdm(examples):
	orig_example = example + "\n\n"
	example = tpt.tokenize(orig_example)
	valid = True
	boundaries_indices = []
	for i, token in enumerate(example):
		if token == "# code\n": 
			boundaries_indices.append(i)
	boundaries_indices.append(len(example)) # Necessary to ensure that the last step transition is taken into account
	for boundary_i in range(2, len(boundaries_indices)):
		pair_length = boundaries_indices[boundary_i] - boundaries_indices[boundary_i-2]
		if pair_length > block_size:
			valid = False
			break
	if valid:
		valid_examples.append(orig_example)

print("Number of valid examples:", len(valid_examples))
with open(out_path, 'w') as f:
	f.write("".join(valid_examples))