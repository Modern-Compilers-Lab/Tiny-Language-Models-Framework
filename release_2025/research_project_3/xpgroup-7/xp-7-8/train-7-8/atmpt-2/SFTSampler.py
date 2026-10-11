from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer
from tqdm import tqdm
import random

tpt = TinypyTokenizer()

class SFTRandomSampler:
	def __init__(self, data_path, nb_examples, batch_size, block_size, purge = False):
		"""
		optimode: One of 'compute' or 'memory'. Not implemented yet.
		"""
		print('[*]Initializing SFTRandomSampler ...')
		self.data_path = data_path
		self.batch_size = batch_size
		self.block_size = block_size

		print('[*]Reading data ...')
		with open(data_path, 'r') as f:
			data = f.read()
		
		print('[*]Preparing examples ...')
		examples = data.split('\n\n')[:-1]
		if nb_examples:
			examples = random.sample(examples, nb_examples)
		preped_examples = []
		if purge:
			# Purging examples that have too long step transitions
			print("[*]Purging examples that have too long step transitions ...")
			for example in tqdm(examples):
				example = example + "\n\n"
				example = tpt.tokenize(example)
				valid = True
				boundaries_indices = []
				for i, token in enumerate(example):
					if token == "# code\n": 
						boundaries_indices.append(i)
				boundaries_indices.append(len(example)) # Necessary to ensure that the last step transition is taken into account
				for boundary_i in range(2, len(boundaries_indices)):
					pair_length = boundaries_indices[boundary_i] - boundaries_indices[boundary_i-2]
					if pair_length > self.block_size:
						valid = False
						break
				if valid:
					preped_examples.append(example)
		else:
			print("[*]Tokenizing examples ...")
			for example in tqdm(examples):
				example = example + "\n\n"
				example = tpt.tokenize(example)
				preped_examples.append(example)
		print(f"[*]Number of examples = {len(preped_examples)}")
		self.examples = preped_examples
		self.nb_tokens = sum([len(example) for example in self.examples])
	
	def __iter__(self):
		return self

	def __next__(self):
		# debugging stuff ...
		# selected_examples = []
		# counters = []
		# start_indices = []
		x = []
		y = []
		EOI_tokens_indices = [] # End-Of-Instruction tokens indices
		for _ in range(self.batch_size):
			
			# Randomly select an example (the -1 is here because randint is inclusive)
			current_example_nb = random.randint(0, len(self.examples) - 1)
			current_example = self.examples[current_example_nb]
			# selected_examples.append(current_example)

			boundaries_indices = []
			for i, token in enumerate(current_example):
				if token == '# code\n':
					boundaries_indices.append(i)

			# Randomly select an initial boundary in the current example (we exclude the last two boundaries thus :-2)
			init_boundary_nb = random.randint(0, len(boundaries_indices[:-2]) - 1)

			# Randomly select a start index for the sample. The start index is located between the initial boundary, and the boundary that follows it i.e init_boundary+1
			start_index = random.randint(boundaries_indices[init_boundary_nb], boundaries_indices[init_boundary_nb + 1]-1)
			#start_indices.append(start_index)
			# Set the EOI_token_index for this sample. This is the index of the "\n#STEP\n" token relatively to the start_index, thus it is just before the second boundary index that follows the initial boundary i.e. init_boundary+2
			if start_index == boundaries_indices[init_boundary_nb]:
				b = 1
			#	counters.append(1)
			else:
				b = 2
			#	counters.append(0)
			EOI_token_index = boundaries_indices[init_boundary_nb + b] - 1 - start_index
			
			# Prepare the samples
			stop_index = start_index + self.block_size
			x_batch_sample = current_example[start_index:stop_index]
			y_batch_sample = current_example[start_index+1:stop_index+1]
			filler_x = ["\n\n"] * (self.block_size - len(x_batch_sample))
			x_batch_sample = x_batch_sample + filler_x
			filler_y = ["\n\n"] * (self.block_size - len(y_batch_sample))
			y_batch_sample = y_batch_sample + filler_y
			x.append(x_batch_sample)
			y.append(y_batch_sample)
			EOI_tokens_indices.append(EOI_token_index)
		
		return x, y, EOI_tokens_indices#, selected_examples, counters, start_indices


if __name__ == "__main__":
	sftcs = SFTRandomSampler(
	data_path = "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-31/datapreps-31/dataprep-31-1/data-dp-31-1/val.txt",
	batch_size = 512,
	block_size = 512
)	
	i = 0
	for x, y in sftcs:
		i += 1
		print(f"Iteration = {i}")
		pass