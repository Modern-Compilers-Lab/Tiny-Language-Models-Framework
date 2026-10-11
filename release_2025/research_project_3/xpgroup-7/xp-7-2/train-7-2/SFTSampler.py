from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer
from tqdm import tqdm
import random

tpt = TinypyTokenizer()

# I could add the notion of epochs and shuffle
# BEWARE: in case of automatic stop, the last batch returned by the sampler will not be a full batch

# Contiguous in the sense that each time the sampler selects an example
# (either one after the other in the dataset or randomly), it will exhaust it
# before moving to the next one
class SFTContiguousSampler:
	def __init__(self, data_path, nb_examples, batch_size, block_size, automatic_stop):
		print('Initializing SFTContiguousSampler ...')
		self.data_path = data_path
		self.batch_size = batch_size
		self.block_size = block_size
		self.automatic_stop = automatic_stop
		print('Reading data ...')
		with open(data_path, 'r') as f:
			data = f.read()
		examples = data.split('\n\n')[:-1]
		if nb_examples:
			examples = random.sample(examples, nb_examples)
		# Purging examples that have too long step transitions
		print("Purging examples that have too long step transitions ...")
		valid_examples = []
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
				valid_examples.append(example)
		self.examples = valid_examples
		self.current_example_nb = 0
		self.set_boundaries_indices()
		self.current_boundary_nb = 0


	def compute_nb_batches(self):
		nb_batches = 0
		pbar = tqdm(desc='Processed examples', total=len(self.examples))
		try:
			while True:
				for _ in range(self.batch_size):
					current_example = self.examples[self.current_example_nb]
					stop_index = self.boundaries_indices[self.current_boundary_nb] + self.block_size

					# If the current example has been exhausted
					if stop_index >= len(current_example) - 1:
						pbar.update(1)
						if self.current_example_nb + 1 == len(self.examples):
							nb_batches += 1
							raise StopIteration
						self.current_example_nb += 1
						self.set_boundaries_indices()
						self.current_boundary_nb = 0
					# Else, there are still other batches to extract from the current example
					else:
						# Get the new_boundary_nb == boundary that is closest to the stop_index
						new_boundary_nb = self.current_boundary_nb
						while self.boundaries_indices[new_boundary_nb] < stop_index:
							if new_boundary_nb + 1 < len(self.boundaries_indices):
								new_boundary_nb += 1
							else: break
						# In case the stop_index is right at the last boundary_index or beyond, we set the new_boundary_nb to the one before last boundary_nb
						if (self.boundaries_indices[new_boundary_nb] > stop_index) or (new_boundary_nb == len(self.boundaries_indices) - 1):
							new_boundary_nb -= 1
						# Set the new current_boundary_nb within the example
						self.current_boundary_nb = new_boundary_nb

				# Increment nb_batches
				nb_batches += 1
		
		except StopIteration:
			pass
		
		# Restore the contiguous sampler state
		self.current_example_nb = 0
		self.set_boundaries_indices()
		self.current_boundary_nb = 0
		self.nb_batches = nb_batches
		return nb_batches
	

	def set_boundaries_indices(self):
		self.boundaries_indices = []
		for i, token in enumerate(self.examples[self.current_example_nb]):
			if token == '# code\n':
				self.boundaries_indices.append(i)
	

	def __iter__(self):
		return self


	def __next__(self):
		x = []
		y = []
		EOI_tokens_indices = [] # End-Of-Instruction tokens indices
		for _ in range(self.batch_size):
			current_example = self.examples[self.current_example_nb]
			stop_index = self.boundaries_indices[self.current_boundary_nb] + self.block_size
			x_batch_sample = current_example[self.boundaries_indices[self.current_boundary_nb]:stop_index]
			y_batch_sample = current_example[self.boundaries_indices[self.current_boundary_nb]+1:stop_index+1]
			EOI_tokens_indices.append(x_batch_sample.index("\n#STEP\n"))
			
			# If the current example has been exhausted
			if stop_index >= len(current_example) - 1:
				filler_x = ["\n\n"] * (self.block_size - len(x_batch_sample))
				x_batch_sample = x_batch_sample + filler_x
				filler_y = ["\n\n"] * (self.block_size - len(y_batch_sample))
				y_batch_sample = y_batch_sample + filler_y

				if self.current_example_nb + 1 == len(self.examples):
					if self.automatic_stop:
						raise StopIteration
					else:
						self.current_example_nb = 0
				else: self.current_example_nb += 1

				self.set_boundaries_indices()
				self.current_boundary_nb = 0
			# Else, there are still other batches to extract from the current example
			else:
				# Get the new_boundary_nb == boundary that is closest to the stop_index
				new_boundary_nb = self.current_boundary_nb
				while self.boundaries_indices[new_boundary_nb] < stop_index:
					if new_boundary_nb + 1 < len(self.boundaries_indices):
						new_boundary_nb += 1
					else: break
				# In case the stop_index is right at the last boundary_index or beyond, we set the new_boundary_nb to the one before last boundary_nb
				if (self.boundaries_indices[new_boundary_nb] > stop_index) or (new_boundary_nb == len(self.boundaries_indices) - 1):
					new_boundary_nb -= 1
				# Set the new current_boundary_nb within the example
				self.current_boundary_nb = new_boundary_nb

			x.append(x_batch_sample)
			y.append(y_batch_sample)
			
		return x, y, EOI_tokens_indices


class SFTRandomSampler:
	def __init__(self, data_path, nb_examples, batch_size, block_size):
		print('Initializing SFTContiguousSampler ...')
		self.data_path = data_path
		self.batch_size = batch_size
		self.block_size = block_size

		print('Reading data ...')
		with open(data_path, 'r') as f:
			data = f.read()
		examples = data.split('\n\n')[:-1]
		if nb_examples:
			examples = random.sample(examples, nb_examples)
		# Purging examples that have too long step transitions
		print("Purging examples that have too long step transitions ...")
		valid_examples = []
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
				valid_examples.append(example)
		self.examples = valid_examples
		self.current_example_nb = random.randint(0, len(self.examples) - 1)
		self.set_boundaries_indices()
		self.current_boundary_nb = random.randint(0, len(self.boundaries_indices) - 1)
		self.nb_processed_batches = 0

	def compute_nb_batches(self):
		nb_batches = 0
		pbar = tqdm(desc='Processed examples', total=len(self.examples))
		try:
			while True:
				for _ in range(self.batch_size):
					current_example = self.examples[self.current_example_nb]
					stop_index = self.boundaries_indices[self.current_boundary_nb] + self.block_size

					# If the current example has been exhausted
					if stop_index >= len(current_example) - 1:
						pbar.update(1)
						if self.current_example_nb + 1 == len(self.examples):
							nb_batches += 1
							raise StopIteration
						self.current_example_nb += 1
						self.set_boundaries_indices()
						self.current_boundary_nb = 0
					# Else, there are still other batches to extract from the current example
					else:
						# Get the new_boundary_nb == boundary that is closest to the stop_index
						new_boundary_nb = self.current_boundary_nb
						while self.boundaries_indices[new_boundary_nb] < stop_index:
							if new_boundary_nb + 1 < len(self.boundaries_indices):
								new_boundary_nb += 1
							else: break
						# In case the stop_index is right at the last boundary_index or beyond, we set the new_boundary_nb to the one before last boundary_nb
						if (self.boundaries_indices[new_boundary_nb] > stop_index) or (new_boundary_nb == len(self.boundaries_indices) - 1):
							new_boundary_nb -= 1
						# Set the new current_boundary_nb within the example
						self.current_boundary_nb = new_boundary_nb

				# Increment nb_batches
				nb_batches += 1
		
		except StopIteration:
			pass
		
		# Restore the contiguous sampler state
		self.current_example_nb = 0
		self.set_boundaries_indices()
		self.current_boundary_nb = 0
		self.nb_batches = nb_batches
		return nb_batches
	

	def set_boundaries_indices(self):
		self.boundaries_indices = []
		for i, token in enumerate(self.examples[self.current_example_nb]):
			if token == '# code\n':
				self.boundaries_indices.append(i)
	
	def __iter__(self):
		return self


	def __next__(self):

		x = []
		y = []
		EOI_tokens_indices = [] # End-Of-Instruction tokens indices
		for _ in range(self.batch_size):
			self.current_example_nb = random.randint(0, len(self.examples) - 1)
			self.set_boundaries_indices()
			self.current_boundary_nb = random.randint(0, len(self.boundaries_indices) - 1)
			current_example = self.examples[self.current_example_nb]
			stop_index = self.boundaries_indices[self.current_boundary_nb] + self.block_size
			x_batch_sample = current_example[self.boundaries_indices[self.current_boundary_nb]:stop_index]
			y_batch_sample = current_example[self.boundaries_indices[self.current_boundary_nb]+1:stop_index+1]
			EOI_tokens_indices.append(x_batch_sample.index("\n#STEP\n"))

			x.append(x_batch_sample)
			y.append(y_batch_sample)
		
			self.nb_processed_batches += 1
		
		return x, y, EOI_tokens_indices


if __name__ == "__main__":
	sftcs = SFTContiguousSampler(
	data_path = "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-31/datapreps-31/dataprep-31-1/data-dp-31-1/val.txt",
	batch_size = 512,
	block_size = 512
)	
	i = 0
	for x, y in sftcs:
		i += 1
		print(f"Iteration = {i}")
		pass