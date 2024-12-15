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
min_init 			= 3
max_depth 			= 2
max_sub_blocks 		= 2
min_length 			= 5
max_length 			= 20
x 					= 2
# Must be >= 0
unindentation_speed = 0.5

# Some global variables
context_stack 		= list()
line_counter 		= 1
code 				= ''

VARIABLES				= ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]
DIGIT 					= [i for i in range(256)]
ARITHMETIC_OPERATORS 	= ["+", "-", "/", "*", "%"]
RELATIONAL_OPERATORS 	= ["<", ">", "<=", ">=", "!=", "=="]

pattern_vocabulary = [
	"INITIALIZATION",
    "SIMPLE_ASSIGNMENT",
    # "ADVANCED_ASSIGNMENT",
    "SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
    "ELSE_STATEMENT",
	'WHILE_LOOP',
    "FOR_LOOP",
	"DISPLAY",
	# "ADVANCED_DISPLAY"
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
    # "ADVANCED_ASSIGNMENT",
	'WHILE_LOOP',
    "FOR_LOOP",
]


# __FUNCTION__: EXECUTE_GEN_ACTION
def execute_gen_action(gen_action:str):
	
	# Setting the global context of the function
	global code
	global line_counter
	
	# Matching gen_action
	match gen_action:

		case 'UNINDENT':
			# Pop the previous context
			previous_context = context_stack.pop()
			# Updat the nb_lines_in_block of the current context
			context_stack[-1]['nb_lines_in_block'] += previous_context['nb_lines_in_block']
		
		case 'INITIALIZATION':
			
			# Choose the identifier
			identifier = random.choice(context_stack[-1]['writable_variables'])
			
			# Update the current context
			context_stack[-1]['if_state'] = False
			context_stack[-1]['nb_lines_in_block'] += 1
			if identifier not in context_stack[-1]['readable_variables']:
				context_stack[-1]['readable_variables'].append(identifier)

			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}{identifier} = {random.choice(DIGIT)}\n'

			# Update the line_counter
			line_counter += 1

		case 'SIMPLE_ASSIGNMENT':
			# __Generate the expression elements__

			# Choose the 'assigend to' identifier
			identifier = random.choice(context_stack[-1]['writable_variables'])
			
			# Choose the first operand
			operand1 = random.choice((
				random.choice(context_stack[-1]['readable_variables']),
				random.choice(DIGIT)
				))
			
			# Choose the second operand
			operand2 = random.choice((
				random.choice(context_stack[-1]['readable_variables']),
				random.choice(DIGIT)
				))

			# Choose the operator
			operator = random.choice(ARITHMETIC_OPERATORS)

			# Update the current context
			context_stack[-1]['if_state'] = False
			context_stack[-1]['nb_lines_in_block'] += 1
			if identifier not in context_stack[-1]['readable_variables']:
				context_stack[-1]['readable_variables'].append(identifier) 

			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'

			# Update the line_counter
			line_counter += 1

		case 'SIMPLE_IF_STATEMENT':

			# Choose operand1 (either a variable or a digit)
			operand1 = random.choice((
				random.choice(context_stack[-1]['readable_variables']),
				random.choice(DIGIT)
				))

			# Choose operand2 (either a variable or a digit)
			operand2 = random.choice((
				random.choice(context_stack[-1]['readable_variables']),
				random.choice(DIGIT)
				))

			# Choose relational operator
			operator = random.choice(RELATIONAL_OPERATORS)
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}if {operand1} {operator} {operand2}:\n'
			
			# Update the line_counter
			line_counter += 1
			
			# Update the current context
			context_stack[-1]['if_state'] = True
			context_stack[-1]['nb_lines_in_block'] += 1
			context_stack[-1]['nb_blocks'] += 1
			context_stack[-1]['nb_if_blocks'] += 1

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
			
		case 'SIMPLE_ELIF_STATEMENT':
			operand1 = random.choice((
				random.choice(context_stack[-1]['readable_variables']),
				random.choice(DIGIT)
				))
			operand2 = random.choice((
				random.choice(context_stack[-1]['readable_variables']),
				random.choice(DIGIT)
				))
			operator = random.choice(RELATIONAL_OPERATORS)

			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}elif {operand1} {operator} {operand2}:\n'

			# Update the current context
			context_stack[-1]['nb_lines_in_block'] += 1

			# Update the line_counter
			line_counter += 1

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
		
		case 'ELSE_STATEMENT':
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}else:\n'
			
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
				'writable_variables': list(context_stack[-1]['writable_variables']),
				'nb_lines_in_block': 0,
				'actions_queue': deque(),
			})
			
			# Update the line_counter
			line_counter += 1

		case 'WHILE_LOOP':

			# __Creating the control variable__

			# Choose the initial value of the control variable
			control_variable_initial_value = random.choice(DIGIT)
			# Choose an identifier for the control variable
			control_variable_identifier = random.choice(context_stack[-1]['writable_variables'])
			# Create the initialization expression of the control variable
			tabs = '	' * (len(context_stack)-1)
			control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
			# Initializing nb_mew_lines (to update the current context of the stack afterwards and the line_counter)
			nb_new_lines = 1

			# Choosing the number of iterations between a and b (both included)
			nb_iters = random.randint(a=1, b=20)
			
			# Choosing the update step : [-b, -a] U [a, b]
			delta = (sign := random.choice((-1, 1))) * random.randint(a=1, b=5)
			
			# Choosing the update operator based on the sign of delta
			update_operator = '+' if delta > 0 else '-'
			update_expression = f'{control_variable_identifier} = {control_variable_identifier} {update_operator} {abs(delta)}\n'
			
			# Choosing a relational operator
			relational_operator = random.choice(["<", ">", "<=", ">="])
			
			# We compute the border_value
			if '=' in relational_operator:
				border_range_one = control_variable_initial_value + (nb_iters-1) * delta
				border_range_two = control_variable_initial_value + nb_iters * delta + sign
				border_value = random.randint(min(border_range_one, border_range_two), max(border_range_one, border_range_two))
			else:
				border_range_one = control_variable_initial_value + (nb_iters-1) * delta + sign
				border_range_two = control_variable_initial_value + nb_iters * delta
				border_value = random.randint(min(border_range_one, border_range_two), max(border_range_one, border_range_two))
			
			# __Choose if we create a border variable to hold the border value__

			if random.random() < 0.5:
				border_variable_defined = True
				# Create the border_variable_identifier, make sure it is not the same as the control_variable_identifier
				tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var != control_variable_identifier]
				border_variable_identifier = random.choice(tmp_writable_variables)
				# Create the border_variable_initialization_expression
				border_variable_initialization_expression = f'{tabs}{border_variable_identifier} = {border_value}\n'
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
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term_for_while_statement}:\n'
				else:
					while_expression = f'{tabs}while {border_term_for_while_statement} {relational_operator} {control_variable_identifier}:\n'
			else:
				if '<' in relational_operator:
					while_expression = f'{tabs}while {border_term_for_while_statement} {relational_operator} {control_variable_identifier}:\n'
				else:
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term_for_while_statement}:\n'
			
			# Increment nb_new_lines
			nb_new_lines += 1

			# __Create the expressions before the while loop__
			# 11 different possible scenarios implemented considering:
			# 1. The order of the control_variable_initialization_expression and the border_variable_initialization_expression
			# 2. The presence of intermediate expressions between the control_variable_initialization_expression, the border_variable_initialization_expression, and the while_expression
			
			# Default value for the intermediate expressions
			intermediate_expression_one = ''
			intermediate_expression_two = ''
			
			# Create a list for the new_writable_variables
			new_writable_variables = list(context_stack[-1]['writable_variables'])
			
			# In the first case, either there is no border_variable_defined, or we start with the control_variable_initialization_expression
			if not border_variable_defined or random.random() < 0.5:
				first_expression, second_expression = (control_variable_initialization_expression, border_variable_initialization_expression)
				
				# Remove the control variable from the new_writable_variables.
				# No need to try-catch it since the control identifier was picked from the writable_variables
				# of the current context. 
				new_writable_variables.remove(control_variable_identifier)
				
				# Add the control variable to the readable variables if it is not already
				if control_variable_identifier not in context_stack[-1]['readable_variables']:
					context_stack[-1]['readable_variables'].append(control_variable_identifier)
				
				# Choose if we create an intermediate expression between the first and the second expressions
				if random.random() < 0.5:
					
					# Choose operand 1
					operand1 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					
					# Choose operand 2
					operand2 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					
					# Choose operator
					operator = random.choice(ARITHMETIC_OPERATORS)
					
					# Choose identifier among the new_writable_variables
					identifier = random.choice(new_writable_variables)
					
					# Add identifier to the readable variables of the current context if not already there
					if identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(identifier)
					
					# Create the expression
					intermediate_expression_one = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
					
					# Increment nb_new_lines
					nb_new_lines += 1

				# If the border variable is defined
				if border_variable_defined:
					
					# Remove the border_variable_identifier from the new_writable_variables
					new_writable_variables.remove(border_variable_identifier)
					
					# Add it to the readable_variables of the current context if not already there
					if border_variable_identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(border_variable_identifier)
				
				# Choose if we create and intermediate expression between the second expression and the while expression
				# follows the same structure as the previous case so no need to comment it
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					if identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(identifier)
					intermediate_expression_two = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
					
					# Increment nb_new_lines
					nb_new_lines += 1

			# In the second case, we start with the border_variable_initialization_expression and then the control_variable_initialization_expression
			# In this case we are sure that the border variable is defined
			else:
				first_expression, second_expression = (border_variable_initialization_expression, control_variable_initialization_expression)
				
				# Remove the border variable from the new_writable_variables
				new_writable_variables.remove(border_variable_identifier)

				# Add the border variable to the readable variables of the current context
				if border_variable_identifier not in context_stack[-1]['readable_variables']:
					context_stack[-1]['readable_variables'].append(border_variable_identifier)
				
				# Choose if we create an intermediate expression between the first and the second expressions
				# Again follows the same structure as the previous cases so no need to comment it
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					if identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(identifier)
					intermediate_expression_one = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
					
					# Increment nb_new_lines
					nb_new_lines += 1
				
				# Remove the control_variable_identifier from the new_writable_variables
				new_writable_variables.remove(control_variable_identifier)
				
				# Add the control_variable_identifier to the readable variables of the current context
				if control_variable_identifier not in context_stack[-1]['readable_variables']:
					context_stack[-1]['readable_variables'].append(control_variable_identifier)
				
				# Choose if we create and intermediate expression between the second expression and the while expression
				# Again follows the same structure as the previous cases so no need to comment it
				if random.random() < 0.5:
					operand1 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					operand2 = random.choice((
						random.choice(context_stack[-1]['readable_variables']),
						random.choice(DIGIT)
						))
					operator = random.choice(ARITHMETIC_OPERATORS)
					identifier = random.choice(new_writable_variables)
					if identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(identifier)
					intermediate_expression_two = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'		
					
					# Increment nb_new_lines
					nb_new_lines += 1

			# __Updating the context_stack__

			# Update the current context
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
			code = code + f'{first_expression}{intermediate_expression_one}{second_expression}{intermediate_expression_two}{while_expression}'

			# Updating the line_counter
			line_counter += nb_new_lines

		case 'WHILE_UPDATE':

			# Retrieve the while_update_expression
			while_update_expression = context_stack[-1]['while_state']

			# Update the current context
			context_stack[-1]['if_state'] = False
			context_stack[-1]['while_state'] = None
			context_stack[-1]['nb_lines_in_block'] += 1

			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}{while_update_expression}'

			# Updating the line_counter
			line_counter += 1

		case 'FOR_LOOP':
			control_variable_identifier = random.choice(context_stack[-1]['writable_variables'])
			initial_value = random.choice(DIGIT)
			step = random.randint(a=1, b=5)
			nb_iters = random.randint(a=1, b=20)
			border_value = random.randint(initial_value + (nb_iters-1) * step + 1, initial_value + nb_iters * step)
			if step == 1 :
				if random.random() < 0.5:
					step_expression = ''
				else:
					step_expression = ', 1'
			else:
				step_expression = f', {step}'
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}for {control_variable_identifier} in ({initial_value}, {border_value}{step_expression}):\n'
			
			# __Updating the context_stack__
			
			# Update the current context
			context_stack[-1]['nb_for_loops'] += 1
			context_stack[-1]['nb_lines_in_block'] += 1
			context_stack[-1]['if_state'] = False
			context_stack[-1]['nb_blocks'] += 1
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

			# Updating the line_counter
			line_counter += 1
		
		case 'DISPLAY':
			
			# Update the current context
			context_stack[-1]['nb_lines_in_block'] += 1
			context_stack[-1]['if_state'] = False
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}print({random.choice(context_stack[-1]["readable_variables"])})\n'

			# Updating the line_counter
			line_counter += 1
		
		case 'END':
			# Do nothing
			pass

		case _:
			raise Exception(f'No match for gen_action {gen_action}')


# __FUNCTION__: QUEUE_GEN_ACTIONS
def queue_gen_actions():
	
	# If the line_counter is less the min_init we return an INITIALIZATION
	if line_counter <= min_init:
		context_stack[-1]['actions_queue'].append("INITIALIZATION")
		
		# Exit
		return
	
	# Elif it's above max_length
	if line_counter > max_length:
		
		# If we are at the indentation level 0 we can directly END the code
		if len(context_stack) == 1:
			context_stack[-1]['actions_queue'].append('END')
		
		# Else If we are in a while_state we generate the while update expression and unindent right after
		elif context_stack[-1]['while_state']:
			context_stack[-1]['actions_queue'].append('WHILE_UPDATE')
			context_stack[-1]['actions_queue'].append('UNINDENT')
		
		# Else if we are in a block with at least one line of code, we directly unindent
		elif context_stack[-1]["nb_lines_in_block"] != 0:
				context_stack[-1]['actions_queue'].append("UNINDENT")
		
		# Else => we are in a block with no lines of code in it => we queue one none indentation statement
		else:
			keyword = random.choice(non_indentation_statements)
			context_stack[-1]['actions_queue'].append(keyword)
		
		# Exit
		return
	
	# Choose if we unindent. This can happen only if we are at some indentation level > 0 and there is at least one code line
	# in the current block, with higher probability of unindenting the higher nb_lines_in_block
	if len(context_stack) > 1 and random.random() > (1/(1+unindentation_speed)) ** context_stack[-1]['nb_lines_in_block']:
		
		# In case we are currently in a while loop and the update statement hasn't been generated yet
		if context_stack[-1]['while_state']:
			context_stack[-1]['actions_queue'].append('WHILE_UPDATE')
		
		# Queue the UNINDENT action
		context_stack[-1]['actions_queue'].append('UNINDENT')

		# Exit
		return
	
	# __In other cases__
	if True:
		# We set the potential keywords
		potential_keywords = list(pattern_vocabulary)

		# Check for while_state
		if context_stack[-1]['while_state']:
			potential_keywords.append('WHILE_UPDATE')
		
		# In case we achieved max_depth or max_sub_blocks inside the current context we remove the indentation statements
		# remove the indentation_statements from potential_keywords
		if len(context_stack) - 1 >=  max_depth or context_stack[-1]["nb_blocks"] >= max_sub_blocks:
			potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in indentation_statements]
		
		# Else If we are not in an If statement we remove the elif + else
		elif not context_stack[-1]["if_state"]:
			potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in {"SIMPLE_ELIF_STATEMENT", "ELSE_STATEMENT"}]

		# We add the END keyword if we are at indentation level 0 and the line_counter is above min_length
		if len(context_stack) == 1 and line_counter > min_length:
			potential_keywords.append("END")

		# We choose a keyword randomly and queue it
		keyword = random.choice(potential_keywords)
		context_stack[-1]['actions_queue'].append(keyword)
		
		# Exit
		return


# __FUNCTION__: GENERATE_RANDOM_CODE
def generate_random_code():
	"""
	Generates a random code snippet by orchestrating the use of the 'distribution_conroller' and 'develop_code_line' functions
	"""
	
	# Initialize the context_stack
	global context_stack 
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
	global line_counter
	line_counter = 1
	
	# Initialize the code
	global code
	code = ''
	
	# Enter the gen_loop
	gen_action = 'START'
	while gen_action != 'END':

		# Call the queue_actions function
		queue_gen_actions()

		# Loop over the created actions for the current context
		while (len(context_stack[-1]['actions_queue']) != 0) and (gen_action := context_stack[-1]['actions_queue'].popleft()) != 'END':
			execute_gen_action(gen_action)
	
	# Return the code
	return code


# Custom exception, raised when a variable's absolute value gets higher than max_val_value
class VariableValueOverflowError(Exception):
	def __init__(self, message):
		super().__init__(message)


# If the script is run as a standalone script
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "Full Random TinyPy Generator")
	
	parser.add_argument("--random_state", default = '/data/yb2618/Tiny-Language-Models-Framework/frcg-random-states/random_state_2024-12-12_08-08.bin', help = "Path to python random state to be loaded if any")
	parser.add_argument("--nb_programs", default = 100000, help = "Number of programs to be generated")
	parser.add_argument("--output_file", default = "./prg_testing/data.txt", help = "Number of programs to be generated")
	parser.add_argument("--timeout", default = 2, help = "Number of seconds to wait for a process to terminate")
	parser.add_argument("--log_file", default = "./log.txt", help = "The path to the logging file for monitoring progress")
	parser.add_argument("--log_interval", default = 10000, help = "The number of code snippets generations before logging to the --log_file for monitoring progress")
	parser.add_argument("--deduplicate", help = "Whether to perform deduplication of generated programs (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_deduplication_trials", default = 50, help = "The maximum number of consecutive trials when deduplication occurs")
	parser.add_argument("--programs_separator", default = "", help = "String to put at the top of each code example (Defaults to empty string)")
	parser.add_argument("--use_tqdm", help = "Whether or not to use tqdm for monitoring progress (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_var_value", default = 100000, help = "The maximum value above which the absolute value of created variables must not go")

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
		with open(args.random_state, "rb") as f:
			random_state = pickle.load(f)
			random.setstate(random_state)
			
	# Launching the generation
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
		code = code.strip('\n')
		
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
			result = programs_separator + code + "\n# output\n# " + "\n# ".join(output.split("\n")[:-1])
			# result = f'PROGRAM #{nb_generated_programs}\n' + code + "\n# output\n# " + "\n# ".join(output.split("\n")[:-1])
			f.write(result + "\n\n")

			# Update the number of generated programs
			nb_generated_programs += 1

			# Update tqdm if used
			if use_tqdm:
				pbar.update(1) 

		except ZeroDivisionError:
			nb_zero_divisions += 1
		except VariableValueOverflowError as e:
			nb_var_value_overflows += 1
		except Exception as e:
			print('Code Snippet Execution Error:', e)
			with open('error_code.txt', 'w') as f:
				f.write(f'PROGRAM PROBLEM#{nb_generated_programs}\n'+code)
			break

		if use_tqdm:
			pbar.set_description(f"ZeroDiv : {nb_zero_divisions:,} | Overflows : {nb_var_value_overflows:,} |")
	
	print(f"percentage of zero divisions: {nb_zero_divisions/(nb_programs + nb_zero_divisions + nb_var_value_overflows) * 100:.2f}%")
	print(f"percentage of overflows: {nb_var_value_overflows/(nb_programs + nb_zero_divisions + nb_var_value_overflows) * 100:.2f}%")

	# Closing the logging and data output files
	f_log_file.close()
	f.close()