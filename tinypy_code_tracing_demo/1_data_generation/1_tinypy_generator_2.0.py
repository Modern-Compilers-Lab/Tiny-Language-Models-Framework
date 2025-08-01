from collections import deque
from tqdm import tqdm
import random
import hashlib

class Context:
	def __init__(self, init_dict:dict, init_queue:deque):
		self.GID = init_dict
		self.keywords_queue = init_queue


	def enqueue(self, keywords):
		for keyword in keywords:
			self.keywords_queue.append(keyword)
	

	def dequeue(self):
		if not self.keywords_queue:
			raise IndexError("dequeue from empty queue")
		return self.keywords_queue.popleft()
	

	def __iter__(self):
		return self


	def __next__(self):
		if not self.keywords_queue:
			raise StopIteration("no more keywords")
		return self.dequeue()


class ContextStack:
	def __init__(self):
		self.context_stack = list()
		self.push()


	def push(self, init_dict=None, init_queue=None):
		
		if init_dict is None:
			init_dict = dict()
		if init_queue is None:
			init_queue = deque()
		
		if not isinstance(init_dict, dict):
			raise TypeError("init_dict must be a dictionary")
		if not isinstance(init_queue, deque):
			raise TypeError("init_queue must be a deque")
		self.context_stack.append(Context(init_dict, init_queue))
	

	def pop(self):
		if not self.context_stack:
			raise IndexError("pop from empty context stack")
		return self.context_stack.pop()
	

	def top(self):
		if not self.context_stack:
			raise IndexError("top from empty context stack")
		return self.context_stack[-1]
	

	def get(self, index):
		if not self.context_stack:
			raise IndexError("get from empty context stack")
		if index < 0 or index >= len(self.context_stack):
			raise IndexError("index out of range")
		return self.context_stack[index]
	

	def __len__(self):
		return len(self.context_stack)


	def __repr__(self):
		repr_str = "ContextStack(["
		for context in self.context_stack:
			repr_str += f"\n\tContext(\n\t\tGID={context.GID},\n\t\tCKQ={list(context.keywords_queue)},\n\t),"
		repr_str += "\n])"
		return repr_str


class TinyPyGenerator2:
	def __init__(self):
		pass

	# CODE INSTANTIATION ALGORITHM
	def instantiate_code(self):

		#=========================================
		# Code instantiation loop
		#=========================================
		while self.context_stack.top().keywords_queue:
			self.keyword = self.context_stack.top().dequeue()

			#|===================================================|
			#|[*] [// User-defined low-level generation rules //]|
			#|===================================================|
			if self.keyword == "ASSIGNMENT":
				identifier = random.choice(['a', 'b', 'c'])
				value = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}{identifier} = {value}\n'
				self.line_counter += 1

			elif self.keyword == "IF_STATEMENT":
				number_1 = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
				number_2 = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}if {number_1} < {number_2}:\n'
				self.line_counter += 1
				# self.context_stack.push()
				self.context_stack.push({
					"block_type": "if_block"
				})
			
			elif self.keyword == "WHILE_LOOP":

				control_identifier = random.choice(['a', 'b', 'c'])
				iterations = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}{control_identifier} = 0\n'
				self.code_snippet = self.code_snippet + f'{tabs}while {control_identifier} < {iterations}:\n'
				self.code_snippet = self.code_snippet + f'{tabs}	{control_identifier} = {control_identifier} + 1\n'
				self.line_counter += 3
				self.context_stack.push({
					"block_type": "while_block",
				})

			elif self.keyword == "UNINDENT":
				self.context_stack.pop()
			
			elif self.keyword == 'END':
				pass

			else:
				raise Exception(f'No match for keyword {self.keyword}')			

	
	# SKELETON CONSTRUCTION ALGORITHM
	def construct_skeleton(self):

		#=========================================
		# Keywords sequence initialization
		#=========================================
		keywords_sequence = []
		
		#|====================================================|
		#|[*] [// User-defined high-level generation rules //]|
		#|====================================================|
		if self.line_counter == 0:
			keywords_sequence.extend([
				"IF_STATEMENT",
				"ASSIGNMENT",
				"END"
				])
		
		else:

			if self.context_stack.top().GID["block_type"] == "if_block":
				r = random.random()

				if r < 0.3:
					keywords_sequence.append("WHILE_LOOP")
				
				else:
					keywords_sequence.append("ASSIGNMENT")
				
				keywords_sequence.extend([
					"ASSIGNMENT",
					"UNINDENT"
				])

			else:
				keywords_sequence.append("UNINDENT")
		
		#=========================================
		# Top Context Queing
		#=========================================
		self.context_stack.top().enqueue(keywords_sequence)


	# MAIN ALGORITHM
	def generate_code_snippet(self):

		#=========================================
		# Main global variables
		#=========================================
		self.code_snippet = ""
		self.keyword = "[START]"
		self.context_stack = ContextStack()
		
		#|=========================================================================|
		#|[*] [// User-defined global dynamic variables for snippet generation //] |
		#|=========================================================================|
		self.context_stack.top().GID = {
			'block_type': None
		}

		self.line_counter = 0

		#=========================================
		# Snippet generation loop
		#=========================================
		while self.keyword != "END":
			self.construct_skeleton()
			self.instantiate_code()

		return self.code_snippet
	

	# CORPUS CREATION
	def create_corpus(self):

		#|===========================================================|
		#|[*] // User-defined global algorithm for corpus creation //|
		#|===========================================================|

		# Set the random seed for reproducibility
		random.seed(40)

		# Initialize the number of programs to generate and the output file
		nb_programs = 10000

		# Open the output file for writing
		output_file = open("./data/snippets.txt", "w")

		# Initialize the progress bar
		pbar = tqdm(range(nb_programs), total=nb_programs, desc="Generating code snippets | Deduplicates (0)", unit=" Snippet")
		
		# Intialize the counter for the number of generated programs
		nb_generated_programs = 0
		
		# Initialize the set of hashes to keep track of duplicates
		hashes = set()

		# Set the maximyum number of deduplication trials
		max_deduplication_trials = 100

		# Initialize the counters for deduplication trials
		nb_deduplication_trials = 0

		# Initialize the number of occured deduplicates
		nb_deduplicates = 0

		# Initialize the deduplication problem flag
		deduplication_problem = False

		# Corpus generation loop
		while (nb_generated_programs < nb_programs) and (not deduplication_problem):
			
			# Generate a code snippet
			code_snippet = self.generate_code_snippet()
			
			# Remove the trailing newline character
			code_snippet = code_snippet.strip("\n")
 
			# Compute the has of the code snippet
			code_hash = hashlib.sha256(code_snippet.encode('utf-8')).hexdigest()
			
			# If the snippet is a duplicate
			if code_hash in hashes:

				# Increment the deduplication trials
				nb_deduplication_trials += 1

				# Increment the number of occured deduplicates
				nb_deduplicates += 1

				# Update the progress bar description to account for the new number of duplicates
				pbar.set_description(f"Generating code snippets | Deduplicates ({nb_deduplicates})")
				
				# If the number of deduplication trials reaches the maximum, we stop the generation
				if nb_deduplication_trials == max_deduplication_trials:
					print("DEDUPLICATE PROBLEM ")
					deduplication_problem = True
					continue

			# Else, the snippet is not a duplicate
			else:
				
				# Add the new snippet hash to the set of hashes
				hashes.add(code_hash)

				# Reset the deduplication trials counter
				nb_deduplication_trials = 0
				
				# Update the progress bar description to account for the new number of generted snippets
				pbar.update(1)

				# Increment the number of generated programs
				nb_generated_programs += 1

				# Write the code snippet to the output file. We prepend the code snippet with a comment indicating that it is a code snippet.
				# And we add a double newline character at the end of the code snippet to act as a separator between code snippets in the output file.
				output_file.write("# code\n" + code_snippet + "\n\n")

		# Close the output file
		output_file.close()


if __name__ == "__main__":
	tpg2 = TinyPyGenerator2()
	tpg2.create_corpus()