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
ARITHMETIC_OPERATORS 	= ["+", "-", "/", "*", "%", "//"]
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

			# Choose the update operator
			update_operator = random.choice(ARITHMETIC_OPERATORS)
			
			# Choose the relational operator
			relational_operator = random.choice(RELATIONAL_OPERATORS)
			
			# Choose the control_variable_identifier
			control_variable_identifier = random.choice(context_stack[-1]['writable_variables'])
			
			# Initializing nb_new_lines to 2 since there is always the control_variable_initialization_expression and the while_expression
			nb_new_lines = 2

			# new_non_writable_variables
			while_loop_variables_identifiers = [control_variable_identifier]
			
			# Create the reauired number of tabs for the current context
			tabs = '	' * (len(context_stack)-1)
			while_prologue_critical_expressions = []
			while_prologue_critical_identifiers = []

			if update_operator == '+':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choice(DIGIT)
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Create the update_operand_value
				update_operand_value = random.randint(a=1, b=5)

				# Choose if we store the update_operand in a variable
				if random.random() < 0.5:
					
					# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					update_operand_identifier = random.choice(tmp_writable_variables)
					
					# Create the update operand initialization expression
					update_operand_initialization_expression = f'{tabs}{update_operand_identifier} = {update_operand_value}\n'
					
					# Update the while_prologue_critical_identifiers
					while_prologue_critical_identifiers.append(update_operand_identifier)
					
					# Update the while_prologue_critical_expressions
					while_prologue_critical_expressions.append({
						'inti_exp': update_operand_initialization_expression,
						'identifier': update_operand_identifier,
					})
					
					# Update the nb_new_lines
					nb_new_lines += 1
					
					# Set the update operand term to the update operand identifier for the control variable update expression
					update_operand_term = update_operand_identifier
				
				else:
					
					# Set the update operand term to the update operand value
					update_operand_term = update_operand_value
				
				# Create the control_variable_update_expression
				control_variable_update_expression = f'{tabs}{control_variable_identifier} = {control_variable_identifier} + {update_operand_term}\n'
				
				# __Create the border__

				# Choose a value for the border
				border_value = random.randint(a=control_variable_initial_value, b=control_variable_initial_value + 20)
				
				# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
				if random.random() < 0.5:
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					border_identifier = random.choice(tmp_writable_variables)
					border_initialization_expression = f'{tabs}{border_identifier} = {border_value}\n'
					while_prologue_critical_identifiers.append(border_identifier)
					while_prologue_critical_expressions.append({
						'inti_exp': border_initialization_expression,
						'identifier': border_identifier,
					})
					nb_new_lines += 1
					border_term = border_identifier
				else:
					border_term = border_value
				
				# Create the while_expression
				if operator in ['<', '<=']:
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
				else:
					while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'

				# __Create the while_prologue__
				
				# Shuffle the while_prologue_critical_expressions
				random.shuffle(while_prologue_critical_expressions)

				# Initialize while_prologue to empty string
				while_prologue = ''
				
				# Set the maximum number of intermediate expressions
				nb_max_intermediate_expressions = 4
				
				# Set the new_writable_variables
				new_writable_variables = list(context_stack[-1]['writable_variables'])
				
				# Iterate over the while_prologue_critical_expressions
				for el in while_prologue_critical_expressions:
					
					# Append the critical expression to the while_prologue
					while_prologue += el['inti_exp']
					
					# Remove the identifier from the new_writable_variables
					new_writable_variables.remove(el['identifier'])
					
					# Add the identifier to the readable_variables of the current context if not already there
					if el['identifier'] not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(el['identifier'])
					
					# Choose the number of intermediate expressions to put after this critical expression
					nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions)
					
					# Make sure to decrease the number of possible intermediate expressions for next time
					nb_max_intermediate_expressions -= nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						
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
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(new_writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in context_stack[-1]['readable_variables']:
							context_stack[-1]['readable_variables'].append(identifier)
						
						# Create the intermediate expression
						intermediate_expression = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
						
						# Append it to while_prologue
						while_prologue += intermediate_expression
					
				# Append the while_prologue and the while_expression to the code
				code = code + while_prologue + while_expression

				# Update the line_counter
				line_counter += nb_new_lines

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
					'while_state': control_variable_update_expression,
					'readable_variables': list(context_stack[-1]['readable_variables']),
					'writable_variables': new_writable_variables,
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})

			elif update_operator == '-':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choice(DIGIT)
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'

				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__

				# Create the update_operand_value
				update_operand_value = random.randint(a=1, b=5)

				# Choose if we store the update_operand in a variable
				if random.random() < 0.5:
					
					# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					update_operand_identifier = random.choice(tmp_writable_variables)
					
					# Create the update operand initialization expression
					update_operand_initialization_expression = f'{tabs}{update_operand_identifier} = {update_operand_value}\n'
					
					# Update the while_prologue_critical_identifiers
					while_prologue_critical_identifiers.append(update_operand_identifier)
					
					# Update the while_prologue_critical_expressions
					while_prologue_critical_expressions.append({
						'inti_exp': update_operand_initialization_expression,
						'identifier': update_operand_identifier,
					})
					
					# Update the nb_new_lines
					nb_new_lines += 1
					
					# Set the update operand term to the update operand identifier for the control variable update expression
					update_operand_term = update_operand_identifier
				
				else:
					
					# Set the update operand term to the update operand value
					update_operand_term = update_operand_value

				# Create the control_variable_update_expression
				control_variable_update_expression = f'{tabs}{control_variable_identifier} = {control_variable_identifier} - {update_operand_term}\n'

				# __Create the border__

				# Choose a value for the border
				border_value = random.randint(a=control_variable_initial_value, b=control_variable_initial_value - 20)

				# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
				if random.random() < 0.5:
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					border_identifier = random.choice(tmp_writable_variables)
					border_initialization_expression = f'{tabs}{border_identifier} = {border_value}\n'
					while_prologue_critical_identifiers.append(border_identifier)
					while_prologue_critical_expressions.append({
						'inti_exp': border_initialization_expression,
						'identifier': border_identifier,
					})
					nb_new_lines += 1
					border_term = border_identifier
				else:
					border_term = border_value

				# Create the while_expression
				if operator in ['<', '<=']:
					while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
				else:
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'

				# __Create the while_prologue__
				
				# Shuffle the while_prologue_critical_expressions
				random.shuffle(while_prologue_critical_expressions)

				# Initialize while_prologue to empty string
				while_prologue = ''
				
				# Set the maximum number of intermediate expressions
				nb_max_intermediate_expressions = 4
				
				# Set the new_writable_variables
				new_writable_variables = list(context_stack[-1]['writable_variables'])
				
				# Iterate over the while_prologue_critical_expressions
				for el in while_prologue_critical_expressions:
					
					# Append the critical expression to the while_prologue
					while_prologue += el['inti_exp']
					
					# Remove the identifier from the new_writable_variables
					new_writable_variables.remove(el['identifier'])
					
					# Add the identifier to the readable_variables of the current context if not already there
					if el['identifier'] not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(el['identifier'])
					
					# Choose the number of intermediate expressions to put after this critical expression
					nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions)
					
					# Make sure to decrease the number of possible intermediate expressions for next time
					nb_max_intermediate_expressions -= nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						
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
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(new_writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in context_stack[-1]['readable_variables']:
							context_stack[-1]['readable_variables'].append(identifier)
						
						# Create the intermediate expression
						intermediate_expression = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
						
						# Append it to while_prologue
						while_prologue += intermediate_expression
					
				# Append the while_prologue and the while_expression to the code
				code = code + while_prologue + while_expression

				# Update the line_counter
				line_counter += nb_new_lines

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
					'while_state': control_variable_update_expression,
					'readable_variables': list(context_stack[-1]['readable_variables']),
					'writable_variables': new_writable_variables,
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})
			
			elif update_operator == '/':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choice(DIGIT)
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Create the possible values
				max_operand_value = 20
				min_operand_value = 2
				possible_values = [i for i in range(min_operand_value, max_operand_value+1)]
				
				# Create the weights for random choice
				# giving 5 times more weight to 10
				weights = [1 if i!= 10 else 5 for i in range(max_operand_value-min_operand_value+1)]
				
				# Choosing the update operand value
				update_operand_value = random.choices(population=possible_values, weights=weights, k=1)[0]

				# Choose if we store the update_operand in a variable
				if random.random() < 0.5:
					
					# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					update_operand_identifier = random.choice(tmp_writable_variables)
					
					# Create the update operand initialization expression
					update_operand_initialization_expression = f'{tabs}{update_operand_identifier} = {update_operand_value}\n'
					
					# Update the while_prologue_critical_identifiers
					while_prologue_critical_identifiers.append(update_operand_identifier)
					
					# Update the while_prologue_critical_expressions
					while_prologue_critical_expressions.append({
						'inti_exp': update_operand_initialization_expression,
						'identifier': update_operand_identifier,
					})
					
					# Update the nb_new_lines
					nb_new_lines += 1
					
					# Set the update operand term to the update operand identifier for the control variable update expression
					update_operand_term = update_operand_identifier
				
				else:
					
					# Set the update operand term to the update operand value
					update_operand_term = update_operand_value
				
				# Create the control_variable_update_expression
				control_variable_update_expression = f'{tabs}{control_variable_identifier} = {control_variable_identifier} / {update_operand_term}\n'
				
				# __Create the border__

				# Choose a value for the border
				border_value = random.randint(a=1, b=control_variable_initial_value)
				
				# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
				if random.random() < 0.5:
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					border_identifier = random.choice(tmp_writable_variables)
					border_initialization_expression = f'{tabs}{border_identifier} = {border_value}\n'
					while_prologue_critical_identifiers.append(border_identifier)
					while_prologue_critical_expressions.append({
						'inti_exp': border_initialization_expression,
						'identifier': border_identifier,
					})
					nb_new_lines += 1
					border_term = border_identifier
				else:
					border_term = border_value
				
				# Create the while_expression
				if operator in ['<', '<=']:
					while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
				else:
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'

				# __Create the while_prologue__
				
				# Shuffle the while_prologue_critical_expressions
				random.shuffle(while_prologue_critical_expressions)

				# Initialize while_prologue to empty string
				while_prologue = ''
				
				# Set the maximum number of intermediate expressions
				nb_max_intermediate_expressions = 4
				
				# Set the new_writable_variables
				new_writable_variables = list(context_stack[-1]['writable_variables'])
				
				# Iterate over the while_prologue_critical_expressions
				for el in while_prologue_critical_expressions:
					
					# Append the critical expression to the while_prologue
					while_prologue += el['inti_exp']
					
					# Remove the identifier from the new_writable_variables
					new_writable_variables.remove(el['identifier'])
					
					# Add the identifier to the readable_variables of the current context if not already there
					if el['identifier'] not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(el['identifier'])
					
					# Choose the number of intermediate expressions to put after this critical expression
					nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions)
					
					# Make sure to decrease the number of possible intermediate expressions for next time
					nb_max_intermediate_expressions -= nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						
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
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(new_writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in context_stack[-1]['readable_variables']:
							context_stack[-1]['readable_variables'].append(identifier)
						
						# Create the intermediate expression
						intermediate_expression = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
						
						# Append it to while_prologue
						while_prologue += intermediate_expression
					
				# Append the while_prologue and the while_expression to the code
				code = code + while_prologue + while_expression

				# Update the line_counter
				line_counter += nb_new_lines

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
					'while_state': control_variable_update_expression,
					'readable_variables': list(context_stack[-1]['readable_variables']),
					'writable_variables': new_writable_variables,
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})

			elif update_operator == '//':

				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choice(DIGIT)
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Create the possible values
				max_operand_value = 20
				min_operand_value = 2
				possible_values = [i for i in range(min_operand_value, max_operand_value+1)]
				
				# Create the weights for random choice
				# giving 5 times more weight to 10
				weights = [1 if i!= 10 else 5 for i in range(max_operand_value-min_operand_value+1)]
				
				# Choosing the update operand value
				update_operand_value = random.choices(population=possible_values, weights=weights, k=1)[0]

				# Choose if we store the update_operand in a variable
				if random.random() < 0.5:
					
					# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					update_operand_identifier = random.choice(tmp_writable_variables)
					
					# Create the update operand initialization expression
					update_operand_initialization_expression = f'{tabs}{update_operand_identifier} = {update_operand_value}\n'
					
					# Update the while_prologue_critical_identifiers
					while_prologue_critical_identifiers.append(update_operand_identifier)
					
					# Update the while_prologue_critical_expressions
					while_prologue_critical_expressions.append({
						'inti_exp': update_operand_initialization_expression,
						'identifier': update_operand_identifier,
					})
					
					# Update the nb_new_lines
					nb_new_lines += 1
					
					# Set the update operand term to the update operand identifier for the control variable update expression
					update_operand_term = update_operand_identifier
				
				else:
					
					# Set the update operand term to the update operand value
					update_operand_term = update_operand_value
				
				# Create the control_variable_update_expression
				control_variable_update_expression = f'{tabs}{control_variable_identifier} = {control_variable_identifier} // {update_operand_term}\n'
				
				# __Create the border__

				# Choose a value for the border
				border_value = random.randint(a=0, b=control_variable_initial_value)

				# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
				if random.random() < 0.5:
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					border_identifier = random.choice(tmp_writable_variables)
					border_initialization_expression = f'{tabs}{border_identifier} = {border_value}\n'
					while_prologue_critical_identifiers.append(border_identifier)
					while_prologue_critical_expressions.append({
						'inti_exp': border_initialization_expression,
						'identifier': border_identifier,
					})
					nb_new_lines += 1
					border_term = border_identifier
				else:
					border_term = border_value
				
				# Create the while_expression
				if operator in ['<', '<=']:
					while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
				else:
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'

				# __Create the while_prologue__
				
				# Shuffle the while_prologue_critical_expressions
				random.shuffle(while_prologue_critical_expressions)

				# Initialize while_prologue to empty string
				while_prologue = ''
				
				# Set the maximum number of intermediate expressions
				nb_max_intermediate_expressions = 4
				
				# Set the new_writable_variables
				new_writable_variables = list(context_stack[-1]['writable_variables'])
				
				# Iterate over the while_prologue_critical_expressions
				for el in while_prologue_critical_expressions:
					
					# Append the critical expression to the while_prologue
					while_prologue += el['inti_exp']
					
					# Remove the identifier from the new_writable_variables
					new_writable_variables.remove(el['identifier'])
					
					# Add the identifier to the readable_variables of the current context if not already there
					if el['identifier'] not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(el['identifier'])
					
					# Choose the number of intermediate expressions to put after this critical expression
					nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions)
					
					# Make sure to decrease the number of possible intermediate expressions for next time
					nb_max_intermediate_expressions -= nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						
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
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(new_writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in context_stack[-1]['readable_variables']:
							context_stack[-1]['readable_variables'].append(identifier)
						
						# Create the intermediate expression
						intermediate_expression = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
						
						# Append it to while_prologue
						while_prologue += intermediate_expression
					
				# Append the while_prologue and the while_expression to the code
				code = code + while_prologue + while_expression

				# Update the line_counter
				line_counter += nb_new_lines

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
					'while_state': control_variable_update_expression,
					'readable_variables': list(context_stack[-1]['readable_variables']),
					'writable_variables': new_writable_variables,
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})

			elif update_operator == '*':
								
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choice(DIGIT)
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Create the possible values
				max_operand_value = 20
				min_operand_value = 2
				possible_values = [i for i in range(min_operand_value, max_operand_value+1)]
				
				# Create the weights for random choice
				# giving 5 times more weight to 10
				weights = [1 if i!= 10 else 5 for i in range(max_operand_value-min_operand_value+1)]
				
				# Choosing the update operand value
				update_operand_value = random.choices(population=possible_values, weights=weights, k=1)[0]

				# Choose if we store the update_operand in a variable
				if random.random() < 0.5:
					
					# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					update_operand_identifier = random.choice(tmp_writable_variables)
					
					# Create the update operand initialization expression
					update_operand_initialization_expression = f'{tabs}{update_operand_identifier} = {update_operand_value}\n'
					
					# Update the while_prologue_critical_identifiers
					while_prologue_critical_identifiers.append(update_operand_identifier)
					
					# Update the while_prologue_critical_expressions
					while_prologue_critical_expressions.append({
						'inti_exp': update_operand_initialization_expression,
						'identifier': update_operand_identifier,
					})
					
					# Update the nb_new_lines
					nb_new_lines += 1
					
					# Set the update operand term to the update operand identifier for the control variable update expression
					update_operand_term = update_operand_identifier
				
				else:
					
					# Set the update operand term to the update operand value
					update_operand_term = update_operand_value
				
				# Create the control_variable_update_expression
				control_variable_update_expression = f'{tabs}{control_variable_identifier} = {control_variable_identifier} * {update_operand_term}\n'
				
				# __Create the border__

				# Choose a value for the border
				border_value = random.randint(a=control_variable_initial_value, b=control_variable_initial_value * 20)
				
				# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
				if random.random() < 0.5:
					tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var not in while_prologue_critical_identifiers]
					border_identifier = random.choice(tmp_writable_variables)
					border_initialization_expression = f'{tabs}{border_identifier} = {border_value}\n'
					while_prologue_critical_identifiers.append(border_identifier)
					while_prologue_critical_expressions.append({
						'inti_exp': border_initialization_expression,
						'identifier': border_identifier,
					})
					nb_new_lines += 1
					border_term = border_identifier
				else:
					border_term = border_value
				
				# Create the while_expression
				if operator in ['<', '<=']:
					while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
				else:
					while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
				
				# __Create the while_prologue__
				
				# Shuffle the while_prologue_critical_expressions
				random.shuffle(while_prologue_critical_expressions)

				# Initialize while_prologue to empty string
				while_prologue = ''
				
				# Set the maximum number of intermediate expressions
				nb_max_intermediate_expressions = 4
				
				# Set the new_writable_variables
				new_writable_variables = list(context_stack[-1]['writable_variables'])
				
				# Iterate over the while_prologue_critical_expressions
				for el in while_prologue_critical_expressions:
					
					# Append the critical expression to the while_prologue
					while_prologue += el['inti_exp']
					
					# Remove the identifier from the new_writable_variables
					new_writable_variables.remove(el['identifier'])
					
					# Add the identifier to the readable_variables of the current context if not already there
					if el['identifier'] not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(el['identifier'])
					
					# Choose the number of intermediate expressions to put after this critical expression
					nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions)
					
					# Make sure to decrease the number of possible intermediate expressions for next time
					nb_max_intermediate_expressions -= nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						
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
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(new_writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in context_stack[-1]['readable_variables']:
							context_stack[-1]['readable_variables'].append(identifier)
						
						# Create the intermediate expression
						intermediate_expression = f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'
						
						# Append it to while_prologue
						while_prologue += intermediate_expression
					
				# Append the while_prologue and the while_expression to the code
				code = code + while_prologue + while_expression

				# Update the line_counter
				line_counter += nb_new_lines

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
					'while_state': control_variable_update_expression,
					'readable_variables': list(context_stack[-1]['readable_variables']),
					'writable_variables': new_writable_variables,
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})
		
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