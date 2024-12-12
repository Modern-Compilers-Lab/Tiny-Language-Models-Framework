import random
import re
# from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout
import pickle
import argparse
import datetime
import hashlib
from pathlib import Path
from collections import deque

# Some algorithm parameters
min_init 			= 0
max_depth 			= 2
max_sub_blocks 		= 1
min_length 			= 5
max_length 			= 10
decay_factor 		= 0.5
x 					= 2
unindentation_speed = 2

# Some global variables
context_stack 		= list()
line_counter 		= 1
code 				= ''

VARIABLES				= ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]
DIGIT 					= [str(i) for i in range(256)]
ARITHMETIC_OPERATORS 	= ["+", "-", "/", "*", "%"]
RELATIONAL_OPERATORS 	= ["<", ">", "<=", ">=", "!=", "=="]

def execute_gen_action (gen_action:str):
	
	match gen_action:
		
		case 'INTIALIZATION':
			return f'{random.choice(context_stack['writable_variables'])} = {random.choice(DIGIT)}\n'
		
		case 'SIMPLE_ASSIGNMENT':
			operand1 = random.choice((
				random.choice(context_stack['readable_variables']),
				random.choice(DIGIT)
				))
			operand2 = random.choice((
				random.choice(context_stack['readable_variables']),
				random.choice(DIGIT)
				))
			operator = random.choice(ARITHMETIC_OPERATORS)
			return f'{random.choice(context_stack['writable_variables'])} = {operand1} {operator} {operand2}\n'
		
		case 'SIMPLE_IF_STATEMENT':
			operand1 = random.choice((
				random.choice(context_stack['readable_variables']),
				random.choice(DIGIT)
				))
			operand2 = random.choice((
				random.choice(context_stack['readable_variables']),
				random.choice(DIGIT)
				))
			operator = random.choice(RELATIONAL_OPERATORS)
			return f'if {operand1} {operator} {operand2}:\n'
		
		case 'SIMPLE_ELIF_STATEMENT':
			operand1 = random.choice((
				random.choice(context_stack['readable_variables']),
				random.choice(DIGIT)
				))
			operand2 = random.choice((
				random.choice(context_stack['readable_variables']),
				random.choice(DIGIT)
				))
			operator = random.choice(RELATIONAL_OPERATORS)
			return f'elif {operand1} {operator} {operand2}:\n'
		
		case 'ELSE_STATEMENT':
			# __Update the context_stack__

			# Update the current context
			context_stack[-1]['nb_lines_in_block'] += 1
			context_stack[-1]['if_state'] = False

			# Stack the new context
			context_stack.append({
				'nb_if_blocks': 0,
				'nb_while_loops': 0,
				'nb_for_loops': 0,
				'nb_blocks': 0,
				'if_state': False,
				'while_state': None,
				'readable_variables': list(context_stack[-1]['readable_variables']),
				'writable_variables': list(context_stack['writable_variables']),
				'nb_lines_in_block': 0,
				'actions_queue': deque(),
			})
			
			# Append the code
			code.append(
				'else:\n'
			)

			# Update the line_counter
			line_counter += 1

		case 'WHILE_LOOP':

			# __Creating the control variable__

			# Choose the initiali value of the control variable
			control_variable_initial_value = random.choice(DIGIT)
			# Choose an identifier for the control variable
			control_variable_identifier = random.choice(context_stack['writable_variables'])
			# Add the identifier to the list of new readable_variables
			new_readable_variables = [control_variable_identifier]
			# Update the current context readable_variables with the control_variable_identifier
			context_stack[-1]['readable_variables'].append(control_variable_identifier)
			# Create the initialization expression of the control variable
			control_variable_initialization_expression = f'{control_variable_identifier} = {control_variable_initial_value}\n'
			# Initializing nb_mew_lines (to update the current context of the stack afterwards)
			nb_new_lines = 1

			# Choosing the number of iterations
			nb_iters = random.randint(a=1, b=20)
			
			# Choosing the update step
			delta = random.choice((-1, 1)) * random.randint(a=1, b=5)
			
			# Choosing the update operator based on the sign of delta
			update_operator = '+' if delta > 0 else '-'
			update_expression = f'{control_variable_identifier} = {control_variable_identifier} {update_operator} {abs(delta)}\n'			
			
			# Choosing a relational operator
			relational_operator = random.choice(["<", ">", "<=", ">="])
			
			# We compute the border_value
			if '=' in relational_operator:
				border_value = random.randint(control_variable_initial_value + (nb_iters-1) * delta, control_variable_initial_value + nb_iters * delta - 1)
			else:
				border_value = random.randint(control_variable_initial_value + (nb_iters-1) * delta + 1, control_variable_initial_value + nb_iters * delta)
			
			# __Choose if we create a border variable to hold the border value__

			if random.random() < 0.5:
				border_variable_defined = True
				# Create the border_variable_identifier
				border_variable_identifier = random.choice(context_stack['writable_variables'])
				new_readable_variables.append(border_variable_identifier)
				# Update the current context stack with the border_variable_identifier
				context_stack[-1]['readable_variables'].append(border_variable_identifier)
				# Create the border_variable_initialization_expression
				border_variable_initialization_expression = f'{border_variable_identifier} = {border_value}\n'
				# Increment nb_new_lines
				nb_new_lines += 1
				# Set the border term to be used in the while statement, to be the border_variable_identifier
				border_term_for_while_statement = border_variable_identifier
			
			# Else, we directly use the border value in the while statement
			else:
				border_variable_defined = False
				# Create an empty border_variable_initialization_expression
				border_variable_initialization_expression = ''
				# Set the border term to be used in the while statement, to be the border_value
				border_term_for_while_statement = border_value
				
			# __Create the while expression__

			if delta > 0:
				if '<' in relational_operator:
					while_expression = f'while {control_variable_identifier} {relational_operator} {border_term_for_while_statement}:\n'
				else:
					while_expression = f'while {border_term_for_while_statement} {relational_operator} {control_variable_identifier}:\n'
			else:
				if '<' in relational_operator:
					while_expression = f'while {border_term_for_while_statement} {relational_operator} {control_variable_identifier}:\n'
				else:
					while_expression = f'while {control_variable_identifier} {relational_operator} {border_term_for_while_statement}:\n'
			# Increment nb_new_lines
			nb_new_lines += 1

			# __Create the expressions before the while loop (11 different possible scenarios implemented)__
			
			# In the first case we precede with the control_variable_initialization_expression and then the border_variable_initialization_expression
			if not border_variable_defined or random.random() < 0.5:
				first_expression, second_expression = (control_variable_initialization_expression, border_variable_initialization_expression)
				
				# Remove the control variable from the new_writable_variables
				new_writable_variables = list(context_stack['writable_variables'])
				new_writable_variables.remove(control_variable_identifier)
				
				# Choose if we create an intermediate expression between the first and the second expressions
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					new_readable_variables.append(identifier)
					intermediate_expression_one = f'{identifier} = {operand1} {operator} {operand2}\n'
					
					# Increment nb_new_lines
					nb_new_lines += 1

				# Remove the border_variable_identifier from the new_writable_variables If it exists
				if border_variable_defined:
					new_writable_variables.remove(border_variable_identifier)
				
				# Choose if we create and intermediate expression between the second expression and the while expression
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					new_readable_variables.append(identifier)
					intermediate_expression_two = f'{identifier} = {operand1} {operator} {operand2}\n'
					
					# Increment nb_new_lines
					nb_new_lines += 1

			# In the second case we precede with the border_variable_initialization_expression and then the control_variable_initialization_expression
			else:
				first_expression, second_expression = (border_variable_initialization_expression, control_variable_initialization_expression)
				
				# Remove the control variable from the new_writable_variables
				new_writable_variables = list(context_stack['writable_variables'])
				new_writable_variables.remove(border_variable_identifier)
				
				# Choose if we create an intermediate expression between the first and the second expressions
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					new_readable_variables.append(identifier)
					intermediate_expression_one = f'{identifier} = {operand1} {operator} {operand2}\n'
					
					# Increment nb_new_lines
					nb_new_lines += 1
				
				# Remove the control_variable_identifier from the new_writable_variables
				new_writable_variables.remove(control_variable_identifier)

				# Choose if we create and intermediate expression between the second expression and the while expression
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					new_readable_variables.append(identifier)
					intermediate_expression_two = f'{identifier} = {operand1} {operator} {operand2}\n'		
					# Increment nb_new_lines
					nb_new_lines += 1

			# __Updating the context_stack__

			# Update the current context
			context_stack[-1]['readable_variables'] = context_stack[-1]['readable_variables'] + [identifier for identifier in new_readable_variables if identifier not in context_stack[-1]['readable_variables']]
			context_stack[-1]['nb_lines_in_block'] += nb_new_lines
			context_stack[-1]['nb_while_loops'] += 1
			context_stack[-1]['nb_blocks'] += 1
			context_stack[-1]['if_state'] = False

			# Stack the new context
			context_stack.append({
				'nb_if_blocks': 0,
				'nb_while_loops': 0,
				'nb_for_loops': 0,
				'nb_blocks': 0,
				'if_state': False,
				'while_state': update_expression,
				'readable_variables': list(context_stack[-1]['readable_variables']),
				'writable_variables': new_writable_variables,
				'nb_lines_in_block': 0,
				'actions_queue': deque(),
			})

			# Append the code
			code.append(
				f'{first_expression}{intermediate_expression_one}{second_expression}{intermediate_expression_two}{while_expression}'
			)

			# Updating the line_counter
			line_counter += nb_new_lines

		case 'WHILE_UPDATE':

			# Retrieve the while_update_expression
			while_update_expression = context_stack[-1]['while_state']

			# Update the current context
			context_stack[-1]['while_state'] = None
			context_stack[-1]['nb_lines_in_block'] += 1

			# Append the code
			code.append(
				while_update_expression
			)

			# Updating the line_counter
			line_counter += 1

		case 'FOR_LOOP':
			control_variable_identifier = random.choice(context_stack['writable_variables'])
			initial_value = random.choice(DIGIT)
			step = random.randint(a=1, b=5)
			nb_iters = random.randint(a=1, b=20)
			border_value = random.randint(control_variable_initial_value + (nb_iters-1) * delta + 1, control_variable_initial_value + nb_iters * delta)
			if step == 1 :
				if random.random() < 0.5:
					step_expression = ''
				else:
					step_expression = ', 1'
			else:
				step_expression = f', {step}'
			
			# __Updating the context_stack__
			
			# Update the current context
			context_stack[-1]['nb_for_loops'] += 1
			context_stack[-1]['nb_lines_in_blocks'] += 1
			context_stack[-1]['if_state'] = False
			if control_variable_identifier not in context_stack[-1]['readable_variables']:
				context_stack[-1]['readable_variables'].append(control_variable_identifier)
			
			# Stack the new context
			context_stack.append({
				'nb_if_blocks': 0,
				'nb_while_loops': 0,
				'nb_for_loops': 0,
				'nb_blocks': 0,
				'if_state': False,
				'while_state': None,
				'readable_variables': list(context_stack[-1]['readable_variables']),
				'writable_variables': list(context_stack[-1]['writable_variables']),
				'nb_lines_in_block': 0,
				'actions_queue': deque(),
			})

			# Append the code
			code.append(
				f'for {control_variable_identifier} in ({initial_value}, {border_value}{step_expression}):\n'
			)

			# Updating the line_counter
			line_counter += 1
		
		case 'DISPLAY':
			
			# Update the current context
			context_stack[-1]['nb_lines_in_blocks'] += 1
			context_stack[-1]['if_state'] = False
			
			# Append the code
			code.append(
				f'print({random.choice(context_stack["readable_variables"])})\n'
			)

			# Updating the line_counter
			line_counter += 1
		
		case 'END':
			# Do nothing
			pass

pattern_vocabulary = [
	"INITIALIZATION",
    "SIMPLE_ASSIGNMENT",
    "ADVANCED_ASSIGNMENT",
    "SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
    "ELSE_STATEMENT",
	'WHILE_LOOP',
    "FOR_LOOP",
	"DISPLAY",
	"ADVANCED_DISPLAY"
]

loop_statements = [
	'WHILE_LOOP',
    "FOR_LOOP",
]

conditional_statements = [
	"SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
]

indentation_statements = [
	'WHILE_LOOP',
    "FOR_LOOP",
	"SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
	"ELSE_STATEMENT"
]

non_indentation_statements = [stm for stm in pattern_vocabulary if stm not in indentation_statements]

variable_creation_statements = [
	"INITIALIZATION",
    "SIMPLE_ASSIGNMENT",
    "ADVANCED_ASSIGNMENT",
	'WHILE_LOOP',
    "FOR_LOOP",
]

pattern_vocab_for_regex = "|".join(pattern_vocabulary)

## __Regular expressions__

re_pattern_line_parser = re.compile("(\t*)("+pattern_vocab_for_regex+")(:[^,=]+=[^,=]+(?:,[^,=]+=[^,=]+)*$|$)")
re_general_line_finder = re.compile(".+(?:\n|$)")
re_while_identifier = re.compile(".*\nwhile ([a-z])")


def queue_gen_actions():
	
	# If the line_counter is less the min_init we return an INITIALIZATION
	if line_counter <= min_init:
		context_stack[-1]['actions_queue'].append("INITIALIZATION")
		
		# Exit
		return
	
	# Elif it's above max_length
	if line_counter > max_length:

		# If we can end the code here i.e. we aren't at the begining of an indentation block (for now the while loop is not considered ...)
		if context_stack[-1]["nb_lines_in_block"] != 0:
				context_stack[-1]['actions_queue'].append("END")
		
		# Else we return a distribution over the statements which do not require an indentation
		keyword = random.choice([non_indentation_statements])
		context_stack[-1]['actions_queue'].append(keyword)
		
		# Exit
		return
	
	# Choose if we unindent. This can happen only if we are at some indentation level > 0 and there is at least one code line
	# in the current block, with higher probability of unindenting the higher nb_lines_in_block
	if len(context_stack) > 1 and random.random() > (1/unindentation_speed) ** context_stack[-1]['nb_lines_in_block']:
		
		# In case we are currently in a while loop and the update statement hasn't been generated yet
		if context_stack[-1]['while_state']:
			context_stack[-1]['actions_queue'].append('WHILE_UPDATE')
		
		# Queue the UNINDENT action
		context_stack[-1]['actions_queue'].append('UNINDENT')

		# Exit
		return
	
	# __In other cases__
	
	# We set the potential keywords
	potential_keywords = list(pattern_vocabulary)

	# Check for while_state
	if context_stack[-1]['while_state']:
		potential_keywords = 'WHILE_UPDATE'
	
	# In case we achieved max_depth or max_sub_blocks inside the current context we remove the indentation statements
	# remove the indentation_statements from potential_keywords
	if len(context_stack) - 1 >=  max_depth or context_stack[-1]["nb_sub_blocks"] >= max_sub_blocks:
		potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in indentation_statements]

	# Else If we are not in an If statement we remove the elif + else
	elif not context_stack[-1]["if_statement"]:
		potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in {"SIMPLE_ELIF_STATEMENT", "ELSE_STATEMENT"}]

	# We add the END keyword if we are not at the begining of an indentation block
	if context_stack[-1]["nb_lines_in_block"] != 0 and line_counter > min_length:
		potential_keywords.append("END")

	# We return a uniform distribution over the remaining keywords
	keyword = random.choice(potential_keywords)
	context_stack[-1]['actions_queue'].append(keyword)
	
	# Exit
	return


# GENERATE_RANDOM_CODE
def generate_random_code():
	"""
	Generates a random code snippet by orchestrating the use of the 'distribution_conroller' and 'develop_code_line' functions
	"""
	
	global context_stack 
	global line_counter
	
	# Initialize the context_stack
	context_stack = list()
	context_stack.append(
		{
			'nb_if_blocks': 0,
			'nb_while_loops': 0,
			'nb_for_loops': 0,
			'nb_blocks': 0,
			'if_state': False,
			# If None, indicates that either we are not at a while loop level, or we are but the update expression of the while loop has already been generated
			# If not None, the value must be the update expression of the while loop
			'while_state': None,
			'readable_variables': list(),
			'writable_variables': list(VARIABLES),
			'nb_lines_in_block': 0,
			'actions_queue': deque(),
		}
	)

	# Initialize the line_counter (might as well rename this to program_counter ...)
	line_counter = 1
	
	code = ''
	gen_action = 'START'
	while gen_action != 'END':

		# Call the queue_actions function
		queue_gen_actions()

		# Loop over the created actions for the current context
		while gen_action := context_stack[-1]['actions_queue'].popleft() != 'END':
			execute_gen_action(gen_action)
			

	# 	# We get the distribution from the distribution controller
	# 	new_distribution = distribution_controller(min_init, min_length, max_length, line_counter, max_depth, max_sub_blocks, context_stack)
		
	# 	# We uniformly randomly choose a random keyword from the distribution
	# 	new_pattern_line = random.choices(list(new_distribution.keys()), list(new_distribution.values()))[0]

	# 	# We generate the new code line(s) using the develop_code_line function
	# 	new_code_line = develop_code_line(keyword=new_pattern_line, context_stack=context_stack)
		
	# 	# We append the new_code_line to the code_lines (think about replacing this one with the random expression)
	# 	code_lines.append("\n".join([(len(context_stack)-1) * "\t" + new_code_line for new_code_line in new_code_line.split("\n")[:-1]])+"\n")
		
	# 	## __Update the context__
		
	# 	# Update the if statement state of the context
	# 	if new_pattern_line in conditional_statements:
	# 		context_stack[-1]["if_statement"] = True
	# 	else:
	# 		context_stack[-1]["if_statement"] = False
		
	# 	# Update the number of sub_blocks in the context
	# 	if new_pattern_line in indentation_statements:
	# 		context_stack[-1]["nb_sub_blocks"] += 1
		
	# 	# Update the number of code lines in the context
	# 	lines_to_add = 3 if new_pattern_line in ("WHILE_LOOP_LESS", "WHILE_LOOP_GREATER") else 1
	# 	context_stack[-1]["nb_lines_in_block"] += lines_to_add
	# 	line_counter += lines_to_add

	# 	# If we have to indent like for the for loop, while loop and conditionals
	# 	if new_pattern_line in indentation_statements:

	# 		# Set the new_writable_variables to the writable_variables of the current stack by default ...
	# 		new_writable_variables = context_stack[-1]["writable_variables"]
			
	# 		# If the indentation statement is a while loop, we remove the control variable from the writable variables
	# 		if new_pattern_line == 'WHILE_LOOP':
	# 			while_control_variable = re_while_identifier.match(new_code_line).group(1)
	# 			new_writable_variables = list(new_writable_variables)
	# 			new_writable_variables.remove(while_control_variable)
			
	# 		# We stack the new indentation level
	# 		context_stack.append({
	# 			"nb_sub_blocks": 0,
	# 			"if_statement": False,
	# 			"readable_variables": list(context_stack[-1]["readable_variables"]),
	# 			"writable_variables": new_writable_variables,
	# 			"nb_lines_in_block": 0,
	# 		})
		
	# 	# Else in case where we may either un-indent or stay
	# 	else:
	# 		# In case we don't stay i.e. we un-indent, we pop the stack and update the number of lines for the just-before context
	# 		while len(context_stack)>1 and random.random() > decay_factor ** context_stack[-1]["nb_lines_in_block"]:
	# 			last_context = context_stack.pop()
	# 			context_stack[-1]["nb_lines_in_block"] += last_context["nb_lines_in_block"]
			
	# #>> END OF WHILE LOOP: while new_pattern_line != "END"
	
	# # We append to the code_lines a display/advanced_display statement
	# code_lines[-1] = generate_code(
	# 		symbol = random.choice(("DISPLAY", "ADVANCED_DISPLAY")),
	# 		assigned_identifiers = context_stack[0]["readable_variables"],
	# 		x = x,
	# 		for_init_step = None
	# 	).replace("SPACE", " ")
	
	# # We join the code_lines to obtain the final code	
	# code = "".join(code_lines)
	
	# # We set the VARIABLE PR back to its original state
	# cfg_rules["VARIABLE"] = context_stack[0]["writable_variables"]
	
	return code


## Custom exception, raised when a variable's absolute value gets higher than max_val_value  
class VariableValueOverflowError(Exception):
	def __init__(self, message):
		super().__init__(message)

# the execution environment boilerplate for controlling the overflow of computed variables


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "Full Random TinyPy Generator")
	
	parser.add_argument("--random_state", help = "Path to python random state to be loaded if any")
	parser.add_argument("--nb_programs", default = 10000, help = "Number of programs to be generated")
	parser.add_argument("--output_file", default = "./prg_testing/data.txt", help = "Number of programs to be generated")
	parser.add_argument("--timeout", default = 2, help = "Number of seconds to wait for a process to terminate")
	parser.add_argument("--log_file", default = "./log.txt", help = "The path to the logging file for monitoring progress")
	parser.add_argument("--log_interval", default = 10000, help = "The number of code snippets generations before logging to the --log_file for monitoring progress")
	parser.add_argument("--deduplicate", help = "Whether to perform deduplication of generated programs (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_deduplication_trials", default = 50, help = "The maximum number of consecutive trials when deduplication occurs")
	parser.add_argument("--programs_separator", default = "", help = "String to put at the top of each code example (Defaults to empty string)")
	parser.add_argument("--use_tqdm", help = "Whether or not to use tqdm for monitoring progress (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_var_value", default = 10000, help = "The maximum value above which the absolute value of created variables must not go")

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
	max_var_value = args.max_var_value

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
	nb_zero_divisions = 0
	nb_var_value_overflows = 0
	nb_generated_programs = 0
	hashes = set()
	nb_deduplication_trials = 0
	exec_env_boilerplate = f"""
from sys import settrace

def line_tracer(frame, event, arg):
	if event == "exception" :
		raise arg[0]
	for var_value in frame.f_locals.values():
		# print(type(var_value))
		if var_value > {max_var_value} or var_value < -{max_var_value}:
			raise VariableValueOverflowError("Variable Value Overflow")
	return line_tracer

def global_tracer(frame, event, arg):
	return line_tracer

settrace(global_tracer)
try:
	func()
finally:
	settrace(None)"""
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
		
		# preparing the execution environment
		indented = "\n".join([f"	{line}" for line in code.split("\n")])
		func = "def func():\n" + indented
		exec_env = func + exec_env_boilerplate

		# Trying the execute the generated code
		sio = StringIO()
		try:
			with redirect_stdout(sio):
				# We execute the code in a controlled environment
				exec(exec_env, {
					"VariableValueOverflowError" : VariableValueOverflowError
				})

			# Saving the code example with its output
			output = sio.getvalue()
			result = programs_separator + code + "# output\n# " + "\n# ".join(output.split("\n")[:-1])
			f.write(result + "\n\n")
			nb_generated_programs += 1
			
			# If using tqdm ...
			if use_tqdm:
				pbar.update(1) 

		except ZeroDivisionError:
			nb_zero_divisions += 1
		except VariableValueOverflowError as e:
			nb_var_value_overflows += 1


		if use_tqdm:
			pbar.set_description(f"ZeroDiv : {nb_zero_divisions:,} | Overflows : {nb_var_value_overflows:,} |")
	
	print(f"percentage of zero divisions: {nb_zero_divisions/(nb_programs + nb_zero_divisions + nb_var_value_overflows) * 100:.2f}%")
	print(f"percentage of overflows: {nb_var_value_overflows/(nb_programs + nb_zero_divisions + nb_var_value_overflows) * 100:.2f}%")

	# Closing the logging and data output files
	f_log_file.close()
	f.close()