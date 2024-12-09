import re
import struct
from tqdm import tqdm

class TinypyTokenizer():
	def __init__(self):
		# defining the keywords list
		self.keywords = sorted([
			'# output', '# code', '#',# ' ',
			'\n', '\t',
			'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
			'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'for', 'in', 'range(', 'while',
			'=', '+', '-', '/', '*', '%',
			')', ':', ',', '.',
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
		with open(input_file_path, 'r') as f:
			examples_string  = f.read()
		examples = examples_string.split('# code')
		examples = examples[1:] # we remove the first entry in examples which is an empty string
		output_file = open(output_file_path, 'wb')
		for example in tqdm(examples):
			# tokenizing
			tokenized_example = self.tokenize(example)
			# writing the token_id of the '# code' keyword as it has been removed after the split
			output_file.write(struct.pack('H', self.encod_map["# code"]))
			# Encoding and writing the token_ids of the example
			# We put it inside a try catch block in case there are keywords that
			# we are not considering in the tokens_list so we can identify them
			try:
				for token in tokenized_example:
					token_id = self.encod_map[token]
					output_file.write(struct.pack('H', token_id))
			except Exception:
				return(f"error token:{token}")
			
		return None

		# for now cannot use the following method as loading in chunks doesn't guarentee that
		# we will get an exact number of tokens (i.e the last token might be truncated => error ...)
		# chunk_size = 2**30  # 1 GiB
		# input_file = open(input_file_path, 'r')
		# output_file = open(output_file_path, 'wb')
		# while True:
		# 	input_chunk = input_file.read(chunk_size)
		# 	if not input_chunk:
		# 		break
		# 	# generating the tokens list from the input chunk
		# 	tokens = self.tokenize(input_chunk)
		# 	for token in tokens:
		# 		token_id = self.encod_map[token]
		# 		output_file.write(struct.pack('H', token_id))  # 'H' stands for unsigned short (2 bytes)



# testing script
# if __name__ == '__main__':
# 	tpt = TinypyTokenizer()
# 	code = """# codecle
# w = 4   6
# o = 147
# b = 187
# while b < 188 :
# 	b = b + 2
# 	u = 155
# v = o / 172
# print(w)
# print(v)
# o = w + 23
# if o < 69 :
# 	v = b - o
# 	print(v)
# 	b = 84 / v
# 	v = v - v
# f = 141
# print(w)
# # output
# # 46
# # 0.8546511627906976
# # 46
# """
# 	print(tpt.tokenize(code))
	# print(tpt.encode_to_file("code.txt", "data.bin"))
	# print('\n\n==> encoding from code.txt to data.bin ...')
	# with open('code.txt', 'r') as f:
	# 	code = f.read()
# 	print(tpt.encode_test(code))
	# tpt.encode_to_file('code.txt', 'data.bin')
	# print('loading code.txt and data.bin ...')
	# import numpy as np
	# data = np.memmap('data.bin', mode='r', dtype=np.uint16)
	# print('checking if tpt.encode(code) == data')
	# eq = True
	# code_ids = tpt.encode(code)
	# for code_id, data_id in zip(code_ids, data):
	# 	eq = code_id == data_id
	# 	if not eq: break
	# print(eq)
	# print('\n\n==> checking if the encod and decode functions are reciprocal')
	# print('decoding')
	# tokens = tpt.decode(code_ids)
	# print('comparing')
	# print(tpt.tokenize(code) == tokens)
	
	# encod = tpt.encode(code)
	# tpt.encode_to_file('code.txt', 'data.bin')
	# tpt.encode(code)
	# tpt.tokenize()
	# print(tpt.encode(code))
	# import numpy as np
	# data = np.memmap('data.bin', mode='r', dtype=np.uint16)
	# print(len(encod) - len(data))
	# eq = True
	# for d, e in zip(encod, data):
	# 	eq = d == e
	# 	if not eq: break
	# print(eq)