import torch
import random
from tqdm import tqdm
from tinypy_code_tracer_m2_tokenizer import TinypyTokenizer

tpt = TinypyTokenizer()

class CodeTracesIterator:
	def __init__(self, data_path, block_size, shuffle):
		print('Initializing CodeTracesIterator ...')
		self.data_path = data_path
		self.block_size = block_size
		self.index = 0
		print('Reading data ...')
		with open(data_path, 'r') as f:
			data = f.read()
		
		# Tokenize the data
		print('Tokenizing data ...')
		self.data_tokens = tpt.tokenize(data)
		
		# Get the indices of all '# code\n' tokens in the data
		print('Getting boundary tokens indices ...')
		
		
		if shuffle:
			print('Shuffling ...')
			random.shuffle(self.examples_indices)

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.index < len(self.examples_indices):
			chunk = self.data_tokens[self.examples_indices[self.index] : self.examples_indices[self.index] + self.block_size + 1]
			
			output_idx = None
			for i, token in enumerate(chunk):
				if token == '\n#STEP\n' or token == '\n\n':
					output_idx = i
					break
			
			if '\n\n' in chunk:
				nn_idx = chunk.index('\n\n')
				x = chunk[:nn_idx]
				y = chunk[output_idx:nn_idx+1]
			else:
				x = chunk[:-1]
				y = chunk[output_idx:]

			x = torch.tensor(tpt.encode_tokens_list(x), dtype=torch.int64).view(1,-1)
			y = torch.tensor(tpt.encode_tokens_list(y), dtype=torch.int64).view(1,-1)
			
			self.index += 1
			
			return x, y
		else:
			raise StopIteration


# Create an iterator class over the steps pairs
class StepsPairsIterator:

	def __init__(self, data_path, shuffle):
		self.index = 0

		with open(data_path, 'r') as f:
			data = f.read()
		code_trace_examples = data.split('\n\n')
		
		steps_pairs = []
		for code_trace_example in code_trace_examples:
			steps = code_trace_example.split('\n#STEP\n')
			for i in range(len(steps)-1):
				input = steps[i]
				output = '\n#STEP\n' + steps[i+1]
				if i != len(steps)-2:
					output += '\n#STEP\n'
				else:
					output += '\n\n'
				steps_pairs.append((input, output))
		self.steps_pairs = steps_pairs
		self.tpt = TinypyTokenizer()

	def __iter__(self):
		return self

	def __next__(self):

		if self.index < len(self.steps_pairs):
			x = self.tpt.encode(self.steps_pairs[self.index][0])
			y = self.tpt.encode(self.steps_pairs[self.index][1])
			x = x + y[:-1]
			x = torch.tensor(x, dtype=torch.int64).view(1,-1)
			y = torch.tensor(y, dtype=torch.int64).view(1,-1)
			self.index += 1
			return x, y
		else:
			raise StopIteration