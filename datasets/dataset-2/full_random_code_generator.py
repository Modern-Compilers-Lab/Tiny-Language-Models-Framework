import random
import re
# from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout
import pickle
import argparse
import datetime
import multiprocessing as mp
from time import sleep
import signal
import hashlib
from pathlib import Path

cfg_rules = {
    # Variables and digits
    "VARIABLE": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ],
    "DIGIT": [str(i) for i in range(256)],

    # Operators
    "ARITHMETIC_OPERATOR": ["+", "-", "/", "*", "%"],
    "RELATIONAL_OPERATOR": ["<", ">", "<=", ">=", "!=", "=="],
    "LOGICAL_OPERATOR_INFIX": ["and", "or"],
    "LOGICAL_OPERATOR_PREFIX": ["not"],
    "LOGICAL_OPERATOR": ["LOGICAL_OPERATOR_INFIX", "LOGICAL_OPERATOR_PREFIX"],
    "OPERATOR": ["ARITHMETIC_OPERATOR"],

    # Formatting
    "NEW_LINE": ["\n"],
    "TAB_INDENT": ["\t"],
    "BRACKET_OPEN": ['('],
    "BRACKET_CLOSE": [')'],
    "EQUALS": ["="],
    "COLON": [":"],
    "COMMA": [","],

    # Keywords
    "IF": ["if"],
    "ELIF": ["elif"],
    "ELSE": ["else"],
    "FOR": ["for"],
    "IN": ["in"],
    "RANGE": ["range"],
    "WHILE": ["while"],
    "PRINT": ["print"],

    # Initializations and assignments
    "IDENTIFIER_INITIALIZATION": ["IDENTIFIER_INITIALIZATION INITIALIZATION",
                                  "INITIALIZATION"],

    "INITIALIZATION": ["VARIABLE SPACE EQUALS SPACE DIGIT NEW_LINE"],
    
	"SIMPLE_ASSIGNMENT": ["A_VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE"],
    
	"ADVANCED_ASSIGNMENT": ["A_VARIABLE SPACE EQUALS SPACE SIMPLE_ARITHMETIC_EVALUATION NEW_LINE"],
    
	"SIMPLE_ARITHMETIC_EVALUATION": ["SIMPLE_ARITHMETIC_EVALUATION ARITHMETIC_OPERATOR ENCLOSED_EXPRESSION", 
                                     "ENCLOSED_EXPRESSION",
                                    ],
	
	# Terms and expressions
    "TERM": ["EXPRESSION_IDENTIFIER", "DIGIT"],
    "EXPRESSION": ["TERM SPACE OPERATOR SPACE TERM"],
    "ENCLOSED_EXPRESSION": ["BRACKET_OPEN EXPRESSION BRACKET_CLOSE"],
    "DISPLAY_EXPRESSION": ["EXPRESSION_IDENTIFIER SPACE OPERATOR SPACE EXPRESSION_IDENTIFIER",
                            "EXPRESSION_IDENTIFIER SPACE OPERATOR SPACE DIGIT"],

    # Conditions
    "SIMPLE_IF_STATEMENT": ["IF SPACE CONDITION SPACE COLON NEW_LINE"],
    "ADVANCED_IF_STATEMENT": ["IF SPACE CHAIN_CONDITION SPACE COLON NEW_LINE"],
    "SIMPLE_ELIF_STATEMENT": ["ELIF SPACE CONDITION SPACE COLON NEW_LINE"],
    "ADVANCED_ELIF_STATEMENT": ["ELIF SPACE CHAIN_CONDITION SPACE COLON NEW_LINE"],
    "ELSE_STATEMENT": ["ELSE SPACE COLON NEW_LINE"],

    "CHAIN_CONDITION": ["CHAIN_CONDITION SPACE LOGICAL_OPERATOR_INFIX SPACE ENCLOSED_CONDITION", 
                        "LOGICAL_OPERATOR_PREFIX SPACE ENCLOSED_CONDITION", 
                        "ENCLOSED_CONDITION"],
    "ENCLOSED_CONDITION": ["BRACKET_OPEN CONDITION BRACKET_CLOSE"],
    "CONDITION": ["OPTIONAL_NOT CONDITION_EXPRESSION", "CONDITION_EXPRESSION"],
    "CONDITION_EXPRESSION": ["EXPRESSION_IDENTIFIER SPACE RELATIONAL_OPERATOR SPACE EXPRESSION_IDENTIFIER", 
                                "EXPRESSION_IDENTIFIER SPACE RELATIONAL_OPERATOR SPACE DIGIT"],
    "OPTIONAL_NOT": ["LOGICAL_OPERATOR_PREFIX SPACE", "SPACE"], 

    # For loops
    "FOR_HEADER": ["FOR SPACE VARIABLE SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL COMMA SPACE STEP BRACKET_CLOSE SPACE COLON NEW_LINE", 
                    "FOR SPACE VARIABLE SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL BRACKET_CLOSE SPACE COLON NEW_LINE"],
    "INITIAL": ["DIGIT"],

    "FOR_LOOP": ["FOR_HEADER NEW_LINE TAB_INDENT DISPLAY"],
    "ADVANCED_FOR_LOOP": ["FOR_LOOP",
						  "FOR_HEADER NEW_LINE TAB_INDENT ADVANCED_DISPLAY"],
	
	# While 
	"WHILE_LOOP_LESS": ["WHILE_HEADER_LESS TAB_INDENT UPDATE_LESS"],
	"WHILE_HEADER_LESS": ["WHILE_CONTROL_INITIALIZATION WHILE SPACE CONDITION_EXPRESSION_LESS SPACE COLON NEW_LINE"],
	"CONDITION_EXPRESSION_LESS": ["EXPRESSION_IDENTIFIER_WHILE SPACE RELATIONAL_OPERATOR_LESS SPACE FINAL_LESS"],
	"UPDATE_LESS": ["WHILE_IDENTIFIER SPACE EQUALS SPACE WHILE_IDENTIFIER SPACE + SPACE STEP NEW_LINE"],
	"RELATIONAL_OPERATOR_LESS": [ "<", "<="],

	"WHILE_LOOP_GREATER": ["WHILE_HEADER_GREATER TAB_INDENT UPDATE_GREATER"],
	"WHILE_HEADER_GREATER": ["WHILE_CONTROL_INITIALIZATION WHILE SPACE CONDITION_EXPRESSION_GREATER SPACE COLON NEW_LINE"],
	"CONDITION_EXPRESSION_GREATER": ["EXPRESSION_IDENTIFIER_WHILE SPACE RELATIONAL_OPERATOR_GREATER SPACE FINAL_GREATER"],
	"UPDATE_GREATER": ["WHILE_IDENTIFIER SPACE EQUALS SPACE WHILE_IDENTIFIER SPACE - SPACE STEP NEW_LINE"],
	"RELATIONAL_OPERATOR_GREATER": [">", ">="],

	"WHILE_CONTROL_INITIALIZATION": ["VARIABLE SPACE EQUALS SPACE DIGIT NEW_LINE"],
	
	# Displaying 
	"DISPLAY" : ["PRINT BRACKET_OPEN DISPLAY_IDENTIFIER BRACKET_CLOSE NEW_LINE"],
	"ADVANCED_DISPLAY" : ["DISPLAY",
					   	  "PRINT BRACKET_OPEN DISPLAY_EXPRESSION BRACKET_CLOSE NEW_LINE"],
	# Temporary ...						 
	"END" : [""]
}
pattern_vocabulary = {
	"INITIALIZATION",
    "SIMPLE_ASSIGNMENT",
    "ADVANCED_ASSIGNMENT",
    "SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
    "ELSE_STATEMENT",
    "WHILE_LOOP_LESS",
	"WHILE_LOOP_GREATER",
    "FOR_HEADER",
	"DISPLAY",
	"ADVANCED_DISPLAY"
}

loop_statements = {
    "WHILE_LOOP_LESS",
	"WHILE_LOOP_GREATER",
    "FOR_HEADER",
}

conditional_statements = {
	"SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
}

indentation_statements = {
    "WHILE_LOOP_LESS",
	"WHILE_LOOP_GREATER",
    "FOR_HEADER",
	"SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
	"ELSE_STATEMENT"
}

non_indentation_statements = pattern_vocabulary - indentation_statements

variable_creation_statements = {
	"INITIALIZATION",
    "SIMPLE_ASSIGNMENT",
    "ADVANCED_ASSIGNMENT",
	"WHILE_LOOP_LESS",
	"WHILE_LOOP_GREATER",
    "FOR_HEADER",
}

pattern_vocab_for_regex = "|".join(pattern_vocabulary)

def generate_code(symbol, assigned_identifiers:list, x:float, for_init_step)->str:
	"""
	Generate code recursively based on the context-free grammar rules.

	Parameters:
	- symbol (str): The symbol to generate code for.
	- assigned_identifiers (dict): Dictionary of assigned identifiers and their values.
	- last_variable (set): Set of the last used variables.
	- parent (Node): Parent node in the syntax tree.

	Returns:
	- str: The generated code.
	"""
	#node = Node(symbol, parent=parent)

	# If the symbol is a non-terminal <--> it's a production rule (PR)
	if symbol in cfg_rules:
		# We develop the PR
		rule = random.choice(cfg_rules[symbol])
		symbols = rule.split(" ")
		# We call the generate code function to get the string associated with this PR
		generated_symbols = [generate_code(s, assigned_identifiers, x, for_init_step) for s in symbols]
		res_string = ''.join(generated_symbols)
		# If it's an INITIAL=>DIGIT PR , we record the DIGIT=>0..255 value in the for_init_step dictionary (will be used when calculating the FINAL of the for loop)
		if symbol == "INITIAL":
			init = generated_symbols[0]
			for_init_step["initial_value"] = init
		# Elif it's an INITIALIZATION PR, we record the generated VARIABLE and it's DIGIT value in the assigned_identifiers dictionary
		elif symbol in variable_creation_statements:
			if symbol == "FOR_HEADER":
				variable_name = generated_symbols[2]
			else:
				variable_name = res_string[0]  
			assigned_identifiers.append(variable_name)
		elif symbol == "WHILE_CONTROL_INITIALIZATION":
			for_init_step["initial_var"] = generated_symbols[0]
			for_init_step["initial_value"] = generated_symbols[4]
		# Concatenate the generated_sub_codes and return the resulting sub_code
		return res_string

	# Else the symbol is a (meta-)terminal, a terminal being one that is returned as is (the simplest case), and a meta-terminal must be generated based on past generations   
	# If EXPRESSION_IDENTIFIER (like we find in ASSIGNEMENTS, DISPLAYS, and FOR loops), we choose randomly among one of the previously initialized variables
	# NOTE: FOR loops don't require the control variable to be initialized -> this could be a point of generalization
	if symbol == "EXPRESSION_IDENTIFIER":
		identifier = random.choice(assigned_identifiers if assigned_identifiers else random.choice(cfg_rules["DIGIT"]))
		return identifier
	# If EXPRESSION_IDENTIFIER_WHILE (i.e. "the declaration" of the control variable of the while loop)
	# NOTE: this one contrary to for loop ... must be one of the existing initialized variables
	if symbol == "EXPRESSION_IDENTIFIER_WHILE":
		return for_init_step["initial_var"]    
	# If WHILE_IDENTIFIER (i.e. the "update" of the control variable of the while loop), get it from the for_init_step dictionary (filled by the EXPRESSION_IDENTIFIER_WHILE meta-terminal)
	if symbol == "WHILE_IDENTIFIER":
		return for_init_step.get("initial_var", "*")
	# If the symbol is a FINAL (for the for loop) or FINAL_LESS (for the while <= loop), choose a step and number of executions, compute the FINAL/_LESS using the for_init_step dict, and record the setp for the for loop as it will be needed later to fill the STEP meta-terminal
	if (symbol == "FINAL") or (symbol == "FINAL_LESS"):    
		initial_value = for_init_step.get("initial_value", "0")
		# Generate valid step_value and execution_count
		valid_values = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
		step_value, execution_count = random.choice(valid_values)
		for_init_step["step"] = str(step_value)
		final_value = step_value * execution_count + int(initial_value) - 1
		return str(final_value)
	# Same thing as for the one before but this one is only meant for the while loop
	if symbol == "FINAL_GREATER":
		initial_value = for_init_step.get("initial_value", "0")
		# Generate valid step_value and execution_count
		valid_values = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
		step_value, execution_count = random.choice(valid_values)
		for_init_step["step"] = str(step_value)
		final_value = int(initial_value) - step_value * execution_count + 1
		return str(final_value)
	# If the STEP meta variable, fill it with the for_init_step dict  
	if symbol == "STEP":
		return for_init_step.get("step", "0")

	# If the symbol is an assigned variable, we try to write to an existing variable instead of creating new ones with a probability "x" times greater
	if symbol == "A_VARIABLE":
		# In case there are available readable and writable identifiers
		if (read_write_vars := list(set(assigned_identifiers) & set(cfg_rules["VARIABLE"]))):
			alpha = len(assigned_identifiers) / len(cfg_rules["VARIABLE"])
			p = ((1-alpha)*x - alpha)/((1-alpha)*(1+x))
			# We return an existing read_write_var with the appropriate probability
			if random.random() < p:
				return random.choice(read_write_vars)
		# In case there is no read_write_var or the probability failed			
		return random.choice(cfg_rules["VARIABLE"])
	
	# If DISPLAY_IDENTIFIER, fill it with either the last variable (if there was an ASSIGNEMENTS), or any randomly chosen variable 
	if symbol == "DISPLAY_IDENTIFIER":
		try:
			return f"{random.choice(assigned_identifiers)}"
		except Exception:
			return random.choice(cfg_rules["DIGIT"])
	# If non of the above i.e. its a terminal (not a meta-terminal)
	return symbol
# Regular expressions
re_pattern_line_parser = re.compile("(\t*)("+pattern_vocab_for_regex+")(:[^,=]+=[^,=]+(?:,[^,=]+=[^,=]+)*$|$)")
re_general_line_finder = re.compile(".+(?:\n|$)")
re_while_identifier = re.compile(".*\nwhile ([a-z])")

max_depth = 3
max_sub_blocks = 3

def distribution_controller(min_init,
							min_length,
							max_length,
							line_counter,
							context_stack)->dict:
	
	# If the line_counter is less the min_init we return an INITIALIZATION
	if line_counter <= min_init:
		return {"INITIALIZATION": 1.0}
	
	# Elif it's above max_length
	if line_counter > max_length:
		# If we can end the code here i.e. we aren't at the begining of an indentation block (for now the while loop is not considered ...)
		if context_stack[-1]["nb_lines_in_block"] != 0:
				return {"END":1.0}
		# Else we return a distribution over the statements which do not require an indentation
		uniproba = 1/len(non_indentation_statements)
		return {keyword : uniproba for keyword in non_indentation_statements} 
	
	## In other cases i.e. min_init < line_counter <= max_length
	
	# We set the potential keywords
	potential_keywords = set(pattern_vocabulary)

	# In case we achieved max_depth or max_sub_blocks inside the current context we remove the indentation statements
	if len(context_stack) - 1 >=  max_depth or context_stack[-1]["nb_sub_blocks"] >= max_sub_blocks:
		potential_keywords.difference_update(indentation_statements)

	# In case we are not in an If statement we remove the elif + else
	elif not context_stack[-1]["if_statement"]:
		potential_keywords.difference_update({"SIMPLE_ELIF_STATEMENT", "ELSE_STATEMENT"})
	
	# We add the END keyword if we are not at the begining of an indentation block
	if context_stack[-1]["nb_lines_in_block"] != 0:
		potential_keywords.add("END")

	# We return a uniform distribution over the remaining keywords
	uniproba = 1/len(potential_keywords)
	return {potential_keyword: uniproba for potential_keyword in potential_keywords}


def generate_random_code(min_init = 0,
						 min_length = 2,
						 max_length = 15,
						 max_init_count = 3,
						 decay_factor = 0.5,
						 x = 2
						 ):
	
	# We create the code_lines list, the context_stack and initialize it
	code_lines = list()
	context_stack = list()
	context_stack.append(
		{
			"nb_sub_blocks": 0,
			"if_statement": False,
			"readable_variables": list(),
			"writable_variables": list(cfg_rules["VARIABLE"]),
			"nb_lines_in_block": 0,
		}
	)

	# We set the line_counter to 0 and the new_pattern_line to empty string
	line_counter = 1
	new_pattern_line = ""

	# While we didn't reach the END keyword
	while new_pattern_line != "END":

		# We get the distribution from the distribution controller
		new_distribution = distribution_controller(min_init, min_length, max_length, line_counter, context_stack)
		
		# We uniformly randomly choose a random keyword from the distribution 
		new_pattern_line = random.choices(list(new_distribution.keys()), list(new_distribution.values()))[0]
		
		# We set the "VARIABLES" PR to the current context
		cfg_rules["VARIABLE"] = context_stack[-1]["writable_variables"]
		
		# We generate the code using the grammar
		new_code_line = generate_code(new_pattern_line, context_stack[-1]["readable_variables"], x, dict()).replace("SPACE", " ")
		
		# We append the new_code_line to the code_lines (think about replacing this one with the random expression)
		code_lines.append("\n".join([(len(context_stack)-1) * "\t" + new_code_line for new_code_line in new_code_line.split("\n")[:-1]])+"\n")
		
		## Update the context
		
		# Update the if statement state of the context
		if new_pattern_line in conditional_statements:
			context_stack[-1]["if_statement"] = True
		else:
			context_stack[-1]["if_statement"] = False
		
		# Update the number of sub loops in the context
		if new_pattern_line in indentation_statements:
			context_stack[-1]["nb_sub_blocks"] += 1
		
		# Update the number of code lines in the context
		lines_to_add = 3 if new_pattern_line in ("WHILE_LOOP_LESS", "WHILE_LOOP_GREATER") else 1
		context_stack[-1]["nb_lines_in_block"] += lines_to_add
		line_counter += lines_to_add

		# In case where we have to indent like for the for loop, while loop and conditionals
		if new_pattern_line in indentation_statements:
			new_writable_variables = context_stack[-1]["writable_variables"]
			
			# If the indentation statement is a while loop, we remove the control variable from the writable variables
			if new_pattern_line in ("WHILE_LOOP_LESS", "WHILE_LOOP_GREATER"):
				while_control_variable = re_while_identifier.match(new_code_line).group(1)
				new_writable_variables = list(new_writable_variables)
				new_writable_variables.remove(while_control_variable)
			
			# We stack the new indentation level
			context_stack.append({
				"nb_sub_blocks": 0,
				"if_statement": False,
				"readable_variables": list(context_stack[-1]["readable_variables"]),
				"writable_variables": new_writable_variables,
				"nb_lines_in_block": 0,
			})
		
		# Else in case where we might un-indent or stay
		else:
			# In case we don't stay i.e. we un-indent, we pop the stack and update the number of lines for the just-before context
			while len(context_stack)>1 and random.random() > decay_factor ** context_stack[-1]["nb_lines_in_block"]:
				last_context = context_stack.pop()
				context_stack[-1]["nb_lines_in_block"] += last_context["nb_lines_in_block"]
			
			# We compute the geometrically decreasing staying probability
			
	#>> END OF WHILE LOOP: while new_pattern_line != "END"
	
	# We append to the code_lines a display/advanced_display statement
	code_lines[-1] = generate_code(
			symbol = random.choice(("DISPLAY", "ADVANCED_DISPLAY")),
			assigned_identifiers = context_stack[0]["readable_variables"],
			x = x,
			for_init_step = None
		).replace("SPACE", " ")
	
	# We join the code_lines to obtain the final code	
	code = "".join(code_lines)
	
	# We set the VARIABLE PR back to its original state
	cfg_rules["VARIABLE"] = context_stack[0]["writable_variables"]
	
	return code
	

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description = "Full Random TinyPy Generator")
	
	parser.add_argument("--random_state", help = "Path to python random state to be loaded if any")
	parser.add_argument("--nb_programs", default = 100000, help = "Number of programs to be generated")
	parser.add_argument("--output_file", default = "./data.txt", help = "Number of programs to be generated")
	parser.add_argument("--timeout", default = 2, help = "Number of seconds to wait for a process to terminate")
	parser.add_argument("--log_file", default = "./log.txt", help = "The path to the logging file for monitoring progress")
	parser.add_argument("--log_interval", default = 10000, help = "The number of code snippets generations before logging to the --log_file for monitoring progress")
	parser.add_argument("--deduplicate", help = "Whether to perform deduplication of generated programs (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_deduplication_trials", default = 50, help = "The maximum number of consecutive trials when deduplication occurs")
	parser.add_argument("--programs_separator", default = "", help = "String to put at the top of each code example (Defaults to empty string)")
	parser.add_argument("--use_tqdm", help = "Whether or not to use tqdm for monitoring progress (set to True for true, False for anything else), defaults to True)")
	
	args = parser.parse_args()
	random_state =  args.random_state
	nb_programs = int(args.nb_programs)
	output_file = args.output_file
	timeout = int(args.timeout)
	log_file = args.log_file
	log_interval = int(args.log_interval)
	deduplicate = True if args.deduplicate in ("true", None) else False
	max_deduplication_trials = int(args.max_deduplication_trials)
	programs_separator = args.programs_separator + '\n' if args.programs_separator else ""
	use_tqdm = True if args.use_tqdm  in ("true", None) else False 
	
	# Saving or setting the random state
	if args.random_state is None:
		random_state = random.getstate()
		now = datetime.datetime.now()
		date_hour = now.strftime("%Y-%m-%d_%H-%M")
		Path("./frcg-random-states").mkdir(parents = True, exist_ok = True)
		with open(f"./frcg-random-states/random_state_{date_hour}.bin", "wb") as f:
			pickle.dump(random_state, f)
	else:
		try:
			with open(args.random_state, "rb") as f:
				random_state = pickle.load(f)
				random.setstate(random_state)
		except Exception:
			random_state = random.getstate()
			now = datetime.datetime.now()
			date_hour = now.strftime("%Y-%m-%d_%H-%M")
			Path(args.random_state).mkdir(parents = True, exist_ok = True)
			with open(f"{args.random_state}/random_state_{date_hour}.bin", "wb") as f:
				pickle.dump(random_state, f)
	
	## Launching the generation
	class TimeoutException(Exception):
		pass

	def timeout_handler(signum, frame):
		raise TimeoutException()

	signal.signal(signal.SIGALRM, timeout_handler)
	

	nb_timeouts = 0
	nb_zero_divisions = 0
	nb_generated_programs = 0
	hashes = set()
	nb_deduplication_trials = 0
	
	# Setting the starting_time and first checkpoint time
	start_time = datetime.datetime.now()
	checkpoint_time = start_time

	# Opening the logging file and the data output file
	f_log_file = open(log_file, "w")
	f = open(output_file, "w")

	# Checking if we use tqdm
	if use_tqdm:
		from tqdm import tqdm
		pbar = tqdm(desc="Generation", total=nb_programs)

	# Launching the loop
	while nb_generated_programs < nb_programs:
		
		# Checking if it's log interval
		if nb_generated_programs % log_interval == 0:
			now = datetime.datetime.now()
			f_log_file.write(f"Generated {nb_generated_programs:<{len(str(nb_programs))}} programs,  absolute time: {now - start_time},  relative time: {now - checkpoint_time}\n")
			f_log_file.flush()
			checkpoint_time = now
		
		# Generating the code
		code = generate_random_code()
		
		# In case of deduplicate
		if deduplicate:
			code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
			if code_hash in hashes:
				nb_deduplication_trials += 1
				if nb_deduplication_trials == max_deduplication_trials:
					print("DEDUPLICATE PROBLEM ")
					break
				else:
					continue
			else:
				nb_deduplication_trials = 0
				hashes.add(code_hash)
		
		# Trying the execute the generated code
		sio = StringIO()
		try:
			with redirect_stdout(sio):
				signal.alarm(timeout)
				exec(code, dict())
			
			# Setting the alarm to 0 just in case it's not enough for the remaining code of the try block to finish execution if no exception occured during exec
			signal.alarm(0)

			# Saving the code example with its output
			output = sio.getvalue()
			result = programs_separator + code + "# output\n# " + "\n# ".join(output.split("\n")[:-1])
			f.write(result + "\n\n")
			nb_generated_programs += 1
			
			# If using tqdm ...
			if use_tqdm:
				pbar.update(1) 

		except ZeroDivisionError:
			output = "ZeroDivisionError"
			nb_zero_divisions += 1
		except ValueError:
			nb_timeouts += 1
			output = "ValueError"
		except OverflowError:
			nb_timeouts += 1
			output = "OverflowError"
		except TimeoutException as e:
			nb_timeouts += 1
			output = "TimeoutError"
		finally:
			signal.alarm(0)
	
	print(f"percentage of timeouts: {nb_timeouts/nb_programs * 100:.2f}%")
	print(f"percentage of zero divisions: {nb_zero_divisions/nb_programs * 100:.2f}%")

	# Closing the logging and data output files
	f_log_file.close()
	f.close()