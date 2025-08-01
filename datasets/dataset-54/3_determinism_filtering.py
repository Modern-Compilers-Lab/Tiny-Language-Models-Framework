from tqdm import tqdm
from tinypy_code_tracer_tokenizer import TinypyTokenizer

# Read the data from the source path
print('Reading data ...')
with open("./data/traced_snippets.txt", 'r') as f:
	data = f.read()

# Split the data into examples using the double newline separator
print("Splitting examples ...")
examples = data.split('\n\n')[:-1]
print("Number of examples:", len(examples))

# Set the output path for the filtered snippets
output_file = open("./data/determinism_filtered_snippets.txt", "w")

# Open the file to write oversize snippets
oversize_snippets_file = open("./data/oversize_snippets.txt", "w")

# Initialize the tokenizer
tpt = TinypyTokenizer()

# Set the block size (i.e. context window size) for filtering
block_size = 100

# Intialize a progress bar for filtered examples
pbar = tqdm(examples, total=len(examples), desc="Filtering examples | Oversize (0):", unit=" Snippet")

# Initialize a counter for the number of oversize snippets
nb_oversize = 0

# Iterate over the examples and filter out those with too oversize step transitions
for example in pbar:

	# Recreate the original example with double newlines which play the role of trace terminator
	orig_example = example + "\n\n"

	# Tokenize the example
	example = tpt.tokenize(orig_example)
	
	# Intialize the validity flag for the example
	valid = True

	# Find the indices of the boundaries (i.e. where "# code\n" appears)
	boundaries_indices = []
	for i, token in enumerate(example):
		if token == "# code\n":
			boundaries_indices.append(i)
	boundaries_indices.append(len(example)) # Necessary to ensure that the last step transition is taken into account
	
	# Check the length of each pair of boundaries
	for boundary_i in range(2, len(boundaries_indices)):
		pair_length = boundaries_indices[boundary_i] - boundaries_indices[boundary_i-2]
		
		# If the pair length exceeds the block size, we consider it oversize
		if pair_length > block_size:
			valid = False
			nb_oversize += 1
			pbar.set_description(f"Filtering examples | Oversize ({nb_oversize}):")
			oversize_snippets_file.write("Pair length too long: " + str(pair_length) + "\n")
			oversize_snippets_file.write(orig_example)
			break
	
	if valid:
		output_file.write(orig_example)