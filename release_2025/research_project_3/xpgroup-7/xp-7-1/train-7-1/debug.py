import random
from tqdm import tqdm
from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer
import pickle
import time

tpt = TinypyTokenizer()
block_size = 512
batch_size = 512
random.seed(42)
t0 = time.time()
print("Reading Data ...")
with open("valid_examples.bin", "rb") as f:
	valid_examples = pickle.load(f)
t1 = time.time()
print(f"Data read in {t1 - t0:.2f} seconds")

def set_boundaries_indices(example_nb):
	boundaries_indices = []
	for i, token in enumerate(valid_examples[example_nb]):
		if token == '# code\n':
			boundaries_indices.append(i)
	return boundaries_indices

nb_batches = 0
current_example_nb = 0
boundaries_indices = set_boundaries_indices(current_example_nb)
current_boundary_nb = 0
pbar = tqdm(total=len(valid_examples))
# desc=f'Last Example Number: {current_example_nb} | Last Boundary Number: {current_boundary_nb}'
try:
	while True:

		for _ in range(batch_size):
			start_example_nb = current_example_nb
			current_example = valid_examples[current_example_nb]
			stop_index = boundaries_indices[current_boundary_nb] + block_size
			x_batch_sample = current_example[boundaries_indices[current_boundary_nb]:stop_index]
			y_batch_sample = current_example[boundaries_indices[current_boundary_nb]+1:stop_index+1]

			if stop_index >= len(current_example) - 1:
				current_example_nb += 1
				if current_example_nb >= len(valid_examples):
					nb_batches += 1
					end_example_nb = current_example_nb
					pbar.update(end_example_nb - start_example_nb)
					raise StopIteration
				boundaries_indices = set_boundaries_indices(current_example_nb)
				current_boundary_nb = 0
			# Else len(x_batch_sample) == block_size and len(y_batch_sample) == block_size
			else:
				# Get the new_boundary_nb == boundary that is closest to the stop_index
				# In case the stop_index is right at the last boundary_index or beyond, we set the new_boundary_nb to the one before last boundary_nb
				new_boundary_nb = current_boundary_nb
				while boundaries_indices[new_boundary_nb] < stop_index:
					if new_boundary_nb + 1 < len(boundaries_indices):
						new_boundary_nb += 1
					else: break
				if (boundaries_indices[new_boundary_nb] > stop_index) or (new_boundary_nb == len(boundaries_indices) - 1):
					new_boundary_nb -= 1
				# Set the new current_boundary_nb within the example
				current_boundary_nb = new_boundary_nb
				
			end_example_nb = current_example_nb
			# pbar.set_description(f'Last Example Number: {current_example_nb:7d} | Last Boundary Number: {current_boundary_nb:4d}')
			pbar.update(end_example_nb - start_example_nb)

		# Increment nb_batches
		nb_batches += 1

except StopIteration:
	pass

except KeyboardInterrupt:
	print("Current example nb:", current_example_nb)
	print("Len Boundaries Indices: ", len(boundaries_indices))
	print("Current boundary nb:", current_boundary_nb)

print(nb_batches)