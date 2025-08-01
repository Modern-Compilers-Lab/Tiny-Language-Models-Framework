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
	


	def depth(self):
		return len(self) - 1



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
				lhs = random.choice(self.variable_identifiers)
				u = random.random()
				if u < 0.15:
					rhs = random.randint(-99, 99)
				
				elif u < 0.3:
					if self.generated_variables:
						rhs = random.choice(self.generated_variables)
					else:
						rhs = random.randint(-99, 99)
				
				else:
					if self.generated_variables:
						if random.random() < 0.5:
							op1 = random.choice(self.generated_variables)
						else:
							op1 = random.randint(-99, 99)
						
						if random.random() < 0.5:
							op2 = random.choice(self.generated_variables)						
						else:
							op2 = random.randint(0, 99)
					else:
						op1 = random.randint(-99, 99)
						op2 = random.randint(0, 99)
					
					op = random.choice(['+', '-'])
					rhs = f'{op1} {op} {op2}'
				
				tabs = '\t' * (self.context_stack.depth())
				self.code_snippet = self.code_snippet + f'{tabs}{lhs} = {rhs}\n'
				self.generated_variables.append(lhs)
				self.context_stack.top().GID["block_line_count"] += 1
				self.line_count += 1


			elif self.keyword == "IF_STATEMENT":
				if self.generated_variables:
					if random.random() < 0.5:
						op1 = random.choice(self.generated_variables)
					else:
						op1 = random.randint(-99, 99)
					
					if random.random() < 0.5:
						op2 = random.choice(self.generated_variables)
					else:
						op2 = random.randint(-99, 99)
				else:
					op1 = random.randint(-99, 99)
					op2 = random.randint(-99, 99)
				
				op = random.choice(['<', '>'])
				conditional = f'{op1} {op} {op2}'
				tabs = '\t' * (self.context_stack.depth())
				self.code_snippet = self.code_snippet + f'{tabs}if {conditional}:\n'
				self.context_stack.top().GID["block_line_count"] += 1
				self.line_count += 1
				self.context_stack.push(init_dict={"block_line_count": 0}, init_queue=deque())


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
		if self.line_count < self.min_line_count:
			if self.context_stack.depth() == 0:
				keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT"]))
			
			elif self.context_stack.depth() < self.max_nesting_depth:
				if self.context_stack.top().GID.get("block_line_count") != 0:
					keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT", "UNINDENT"]))
				
				else:
					keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT"]))
			
			else:
				if self.context_stack.top().GID.get("block_line_count") != 0:
					keywords_sequence.append(random.choice(["ASSIGNMENT", "UNINDENT"]))
				
				else:
					keywords_sequence.append("ASSIGNMENT")
		
		elif self.line_count < self.max_line_count:
			if self.context_stack.depth() == 0:
				keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT", "END"]))
			
			elif self.context_stack.depth() < self.max_nesting_depth:
				if self.context_stack.top().GID.get("block_line_count") != 0:
					keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT", "UNINDENT", "END"]))
				
				else:
					new_keyword = random.choice(["ASSIGNMENT", "IF_STATEMENT", "END"])
					if new_keyword == "END":
						keywords_sequence.extend(["ASSIGNMENT", "END"])
					
					else:
						keywords_sequence.append(new_keyword)
			
			else:
				if self.context_stack.top().GID.get("block_line_count") != 0:
					keywords_sequence.append(random.choice(["ASSIGNMENT", "UNINDENT", "END"]))
				
				else:
					new_keyword = random.choice(["ASSIGNMENT", "END"])
					if new_keyword == "END":
						keywords_sequence.extend(["ASSIGNMENT", "END"])
		
					else:
						keywords_sequence.append(new_keyword)
		
		else:
			if self.context_stack.depth() != 0 and self.context_stack.top().GID.get("block_line_count") == 0:
				keywords_sequence.extend(["ASSIGNMENT", "END"])
			
			else:
				keywords_sequence.append("END")
		
		#=========================================
		# Top Context Queuing
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
		
		#|==============================================================|
		#|[*] [// User-defined global setup for snippet generation //]  |
		#|==============================================================|
		self.context_stack.top().GID["block_line_count"] = 0
		self.variable_identifiers = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
									 "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", 
									 "u", "v", "w", "x", "y", "z"]
		self.line_count = 0
		self.generated_variables = list()
		self.min_line_count = 5
		self.max_line_count = 10
		self.max_nesting_depth = 2


		#=========================================
		# Snippet generation loop
		#=========================================
		while self.keyword != "END":
			self.construct_skeleton()
			self.instantiate_code()

		return self.code_snippet





class TinyPyCodeTracer:
	def __init__(self):
		pass


	def trace_snippet(self, snippet):

		#|=======================================================================|
		#|[*] [// User-defined logic for snippet tracing with error handling //] |
		#|=======================================================================|

		# Create the tracing environment
		tracing_env = """
from sys import settrace

def line_tracer(frame, event, arg):
	current_step = code.split("\\n")

	# Check runtime value range
	for key, value in frame.f_locals.items():
		if isinstance(value, (int, float)):
			if value < -99 or value > 99:
				raise ValueError(f"Value out of range: {key} = {value}")

	state_fill = ";".join([f"{key}?{value:}" for key, value in frame.f_locals.items()])
	if event == "line":
		current_step[frame.f_lineno - 2] = "@" + current_step[frame.f_lineno - 2] + "$" + state_fill
		trace.append("#STEP\\n" + "\\n".join(current_step))
	
	elif event == 'return':
		current_step.append("@^" + "$" + state_fill)
		trace.append("#STEP\\n" + "\\n".join(current_step))
	
	
	return line_tracer


def global_tracer(frame, event, arg):
	return line_tracer


settrace(global_tracer)
try:
	func()
finally:
	settrace(None)
"""
		
		# Insert the code snippet into the tracing environment: put it inside a function named func, and append the tracing environment
		indented_snippet = "\n".join([f"	{line}" for line in snippet.split("\n")])
		snippet_in_tracing_env = "def func():\n" + indented_snippet + tracing_env
		
		# Launch snippet tracing with runtime exceptions handling
		trace = []
		try:
			exec(snippet_in_tracing_env, {
				"__builtins__":__builtins__,
				"code": snippet,
				"trace": trace,
				}
			)
		except Exception as e:
			return snippet, type(e).__name__, str(e)

		# Write the succesfully traced code snippet to the output file
		return snippet+"\n"+"\n".join(trace), None, None


class TinyPyCodeTracingGenerator:
	def __init__(self):
		self.tpg2 = TinyPyGenerator2()
		self.tpct = TinyPyCodeTracer()



	def create_corpus(self):

		#|===========================================================|
		#|[*] // User-defined global algorithm for corpus creation //|
		#|===========================================================|

		# Set the random seed for reproducibility
		random.seed(40)

		# Initialize the number of programs to generate and the output file
		nb_programs = 3_000_000

		# Open the output file for writing
		output_file = open("./data/traced_snippets.txt", "w")

		# Initialize the progress bar
		pbar = tqdm(desc="Generating snippets | Deduplicates (0) | ValueErrors (0)", total=nb_programs, unit=" Snippet")
		
		# Intialize the counter for the number of generated programs
		nb_generated_programs = 0
		
		# Initialize the set of hashes to keep track of duplicates
		hashes = set()

		# Set the maximyum number of deduplication trials
		max_deduplication_trials = 100

		# Initialize the counters for deduplication trials
		nb_deduplication_trials = 0

		# Initialize the number of occured deduplicates
		count_deduplicates = 0

		# Initialize the deduplication problem flag
		deduplication_problem = False

		# Initialize the errors counters
		count_value_errors = 0
		count_unbound_local_errors = 0

		# Corpus generation loop
		while (nb_generated_programs < nb_programs) and (not deduplication_problem):
			
			# Generate a code snippet
			code_snippet = self.tpg2.generate_code_snippet()
			
			# Remove the trailing newline character
			code_snippet = code_snippet.strip("\n")
 
			# Compute the has of the code snippet
			code_hash = hashlib.sha256(code_snippet.encode('utf-8')).hexdigest()
			
			# If the snippet is a duplicate
			if code_hash in hashes:

				# Increment the deduplication trials
				nb_deduplication_trials += 1

				# Increment the number of occured deduplicates
				count_deduplicates += 1

				# Update the progress bar description to account for the new number of duplicates
				pbar.set_description(f"Generating code snippets | Deduplicates ({count_deduplicates}) | ValueErrors ({count_value_errors}) | UnboundLocalErrors ({count_unbound_local_errors})")
				
				# If the number of deduplication trials reaches the maximum, we stop the generation
				if nb_deduplication_trials == max_deduplication_trials:
					print("[*] DEDUPLICATE PROBLEM")
					deduplication_problem = True
					continue

			# Else, the snippet is not a duplicate
			else:

				# Trace the code snippet	
				exec_trace, error_type, error_msg = self.tpct.trace_snippet(code_snippet)
				
				# If there is an error, identify it and handle it
				if error_type is not None:

					# If the error is a ValueError, we increment the counter
					if error_type == "ValueError":
						count_value_errors += 1
					
					elif error_type == "UnboundLocalError":
						count_unbound_local_errors += 1

					# Else, the error corresponds to no expected error, so we write it to the error file for checking and exit
					else:
						with open("./execution_error.txt", "w") as f_exec_error:
							f_exec_error.write(f"Error type: {error_type}\nError message: {error_msg}\nCode snippet:\n{code_snippet}\n\n")
						import sys; sys.exit(0)
					
					# Update the progress bar description to account for the new number of errors
					pbar.set_description(f"Generating code snippets | Deduplicates ({count_deduplicates}) | ValueErrors ({count_value_errors}) | UnboundLocalErrors ({count_unbound_local_errors})")
				
				# Else, there are no errors and we can proceede
				else:
					
					# Add the new snippet hash to the set of hashes
					hashes.add(code_hash)

					# Reset the deduplication trials counter
					nb_deduplication_trials = 0
					
					# Update the progress bar description to account for the new number of generted snippets
					pbar.update(1)

					# Increment the number of generated programs
					nb_generated_programs += 1

					# Write the code snippet to the output file.
					# We prepend the code snippet with a comment indicating that it is a code snippet.
					# And we add a double newline character at the end of the code snippet to act as a separator between code snippets in the output file.
					output_file.write(exec_trace + "\n\n")

		# Close the output file
		output_file.close()
	


if __name__ == "__main__":
	tpctg = TinyPyCodeTracingGenerator()
	tpctg.create_corpus()