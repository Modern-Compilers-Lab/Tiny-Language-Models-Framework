import re
import struct
from tqdm import tqdm

class TinypyTokenizer():
	def __init__(self):
		# defining the keywords list
		self.keywords = sorted([
			# ' ',
			'# ', '&', ';', '?', '|', '@', '^', '$',
			'# output\n', '# code\n', '\n#STEP\n',
			'\n', '\t', '\n\n',
			'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
			'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'for', 'in', 'range(', 'while',
			'=', '+', '-', '/', '*', '%', '//',
			'(', ')', ':', ',', '.',
			'if', 'elif', 'else',
			'==', '<', '<=', '>', '>=', 'not', '!=',
			'print(',
		], key = len, reverse = True)
		# creating the tokenizing regex
		pattern = "|".join([re.escape(kw) for kw in self.keywords]) + '|[^ ]+?'
		self.regex = re.compile(pattern)
		# creating the encoding_map and decoding_map
		self.encod_map = { kw : i for i, kw in enumerate(self.keywords)}
		self.decod_map = { i : kw for i, kw in enumerate(self.keywords)}
	
	def tokenize(self, input_string):
		return self.regex.findall(input_string)
	
	def encode(self, input_string):
		tokens  = self.tokenize(input_string)
		return [self.encod_map[token] for token in tokens]
	
	def encode_tokens_list(self, tokens_list):
		return [self.encod_map[token] for token in tokens_list]

	def encode_test(self, input_string):
		tokens = self.tokenize(input_string)
		try:
			for token in tokens:
				self.encod_map[token]
		except Exception:
			print(f'{token}')
			return -1
		return 0

	def decode(self, tokens_ids):
		return [self.decod_map[id] for id in tokens_ids]

	def encode_to_file(self, input_file_path:str, output_file_path:str):
		print('loading', input_file_path, '...')
		with open(input_file_path, 'r') as f:
			examples_string  = f.read()
		print('splitting the examples ...')
		examples = examples_string.split('\n\n')[:-1] # We remove the last element of the list because it is an emtpy string
		output_file = open(output_file_path, 'wb')
		for example in tqdm(examples):
			# tokenizing
			example = example + '\n\n'
			tokenized_example = self.tokenize(example)
			
			# Encoding and writing the token_ids of the example
			# We put it inside a try catch block in case there are keywords that
			# we are not considering in the tokens_list so we can identify them
			try:
				for token in tokenized_example:
					token_id = self.encod_map[token]
					output_file.write(struct.pack('B', token_id))
			except Exception:
				return(f"error token:{token}")
		output_file.close()
		return None