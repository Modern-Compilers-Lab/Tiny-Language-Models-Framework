from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer

tpt = TinypyTokenizer()
# I could add the notion of epochs and shuffle
class SFTContiguousSampler:
	def __init__(self, data_path, batch_size, block_size):
		print('Initializing CodeTracesIterator ...')
		self.data_path = data_path
		self.batch_size = batch_size
		self.block_size = block_size

		print('Reading data ...')
		with open(data_path, 'r') as f:
			data = f.read()
		
		examples = data.split('\n\n')[:-1]
		examples[-1] = examples[-1] + '\n\n'
		
		for i, example in enumerate(examples):
			example = tpt.tokenize(example)
			examples[i] = example
		
		self.examples = examples
		
		self.current_example_nb = 0
		
		self.set_boundaries_indices()

		self.current_boundary_nb = 0

		# # Tokenize the data
		# print('Tokenizing data ...')
		# self.data_tokens = tpt.tokenize(data)
		
		# # Get the indices of all boundary tokens : '# code\n' and '\n#STEP\n'
		# print('Getting boundary tokens indices ...')

		# self.boundaries_indices = []
		# for i, token in enumerate(self.data_tokens):
		# 	if token == '# code\n':
		# 		self.boundaries_indices.append(i)
		# self.current_boundary_nb = 0
	
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
		for _ in self.batch_size:

			current_example = self.examples[self.current_example_nb]
			stop_index = min(self.boundaries_indices[self.current_boundary_nb] + self.block_size, len(current_example))
			x_batch_sample = current_example[self.boundaries_indices[self.current_boundary_nb]:stop_index]
			y_batch_sample = current_example[self.boundaries_indices[self.current_boundary_nb]+1:stop_index+1]
			
			if len(x_batch_sample) < self.block_size or len(y_batch_sample) < self.block_size:
				filler = ["\n\n"] * (self.block_size - len(x_batch_sample))
				x_batch_sample = x_batch_sample + filler
				y_batch_sample = y_batch_sample + filler + ["\n\n"]
				self.current_example_nb += 1
				self.set_boundaries_indices()
				self.current_boundary_nb = 0
			# Else len(x_batch_sample) == self.block_size and len(y_batch_sample) == self.block_size
			else:
				# Get the boundary that is closest to the stop_index
				new_boundary_nb = self.current_boundary_nb
				while self.boundaries_indices[new_boundary_nb] < stop_index:
					new_boundary_nb += 1
					if new_boundary_nb >= len(self.boundaries_indices):
						new_boundary_nb -= 1
						break
				if self.boundaries_indices[new_boundary_nb] > stop_index:
					new_boundary_nb -= 1
				
				# Set the new current_boundary_nb within the example
				self.current_boundary_nb = new_boundary_nb
			
			x.append(x_batch_sample)
			y.append(y_batch_sample)
		
		return x, y