import 	random
import 	pickle
import 	argparse
import 	datetime
import 	hashlib
import 	signal
import 	math
from 	io 			import StringIO
from 	contextlib 	import redirect_stdout
from 	pathlib 	import Path
from 	collections import deque

# __PARAMETERS DASHBOARD__

# GENERAL PARAMETERS
MIN_INIT_MAX			= 3
MAX_DEPTH 				= 2
MAX_SUB_BLOCKS 			= 2
MIN_LENGTH 				= 5
MAX_LENGTH 				= 10
UNINDENTATION_SPEED 	= 0.09	# if <= 0, will never unindent after the first indentation encountered
# PRINT_NB_DECIMALS 		= 2		# Disabled in this code

# PRINT PARAMETERS
FORCE_PRINT							= True
PRINT_WEIGHTS_CONTROL_COEFFICIENT 	= 0

# READ AND WRITE SECURITIES
READ_SECURITY 				= False
WRITE_SECURITY 				= True
TIMEOUT						= 3

# WHILE LOOP PARAMETERS

ALWAYS_EXECUTE_WHILE_LOOP = False

# WHILE_LOOP GENERAL PARAMETERS
WHILE_LOOP_UPDATE_OPERATORS 	= ['+', '-', '//', '*']
WHILE_LOOP_RELATIONAL_OPERATORS = ['>', '<', '>=', '<=']
NB_MAX_WHILE_LOOP_PROLOGUE_INTERMEDIATE_EXPRESSIONS = 2
NB_MAX_WHILE_LOOP_UPDATE_INTERMEDIATE_EXPRESSIONS = 2

# WHILE LOOP PARAMETERS FOR ADD UPDATE OPERATOR
WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(255+1)]
WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(255+1)]
WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(1, 20+1)]
WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 1 else 4 for i in range(1, 20+1)]
WHILE_LOOP_ADD_UO_NB_ITERS									= [i for i in range(1, 3+1)]

# WHILE LOOP PARAMETERS FOR SUB UPDATE OPERATOR
WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(255+1)]
WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(255+1)]
WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(1, 20+1)]
WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 1 else 4 for i in range(1, 20+1)]
WHILE_LOOP_SUB_UO_NB_ITERS									= [i for i in range(1, 3+1)]

# WHILE LOOP PARAMETERS FOR DIV UPDATE OPERATOR
WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(1, 255+1)]
WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(1, 255+1)]
WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(2, 20+1)]
WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 10 else 3 for i in range(2, 20+1)]
WHILE_LOOP_DIV_UO_NB_ITERS									= [i for i in range(1, 3+1)]
WHILE_LOOP_DIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT		= 0.1

# WHILE LOOP PARAMETER FOR FLOORDIV UPDATE OPERATOR
WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(1, 255+1)]
WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(1, 255+1)]
WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(2, 20+1)]
WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 10 else 4 for i in range(2, 20+1)]
WHILE_LOOP_FDIV_UO_NB_ITERS									= [i for i in range(1, 3+1)]
WHILE_LOOP_FDIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT		= 0.1

# WHILE LOOP PARAMETERS FOR MUL UPDATE OPERATOR
WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(1, 255+1)]
WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS	= [1 for i in range(1, 255+1)]
WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(2, 20+1)]
WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES_WEIGHTS				= [1 if i!= 10 else 4 for i in range(2, 20+1)]
WHILE_LOOP_MUL_UO_NB_ITERS									= [i for i in range(1, 3+1)]
WHILE_LOOP_MUL_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT		= 0.1

# GLOBAL OPERATIONAL VARIABLES
context_stack				= list()
all_assigned_variables	 	= list()
line_counter 				= 1
code 						= ''

VARIABLES						= ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]
DIGIT 							= [i for i in range(10+1)]
ARITHMETIC_OPERATORS 			= ["+", "-", "*", "//", "%"]
ARITHMETIC_OPERATORS_WEIGHTS 	= [3, 3, 5, 7, 14]
RELATIONAL_OPERATORS 			= ["<", ">", "<=", ">=", "!=", "=="]

pattern_vocabulary = [
	"INITIALIZATION",
	'DIRECT_ASSIGNMENT',
	"SIMPLE_ASSIGNMENT",
	# "SIMPLE_IF_STATEMENT",
	# "SIMPLE_ELIF_STATEMENT",
    # "ELSE_STATEMENT",
	# 'WHILE_LOOP',
	"DISPLAY",
]

loop_statements = [
	'WHILE_LOOP',
]

conditional_statements = [
	"SIMPLE_IF_STATEMENT",
    "SIMPLE_ELIF_STATEMENT",
]

indentation_statements = [
	'WHILE_LOOP',
	"SIMPLE_IF_STATEMENT",
	"SIMPLE_ELIF_STATEMENT",
	"ELSE_STATEMENT"
]

non_indentation_statements = [stm for stm in pattern_vocabulary if stm not in indentation_statements]

variable_creation_statements = [
	"INITIALIZATION",
	'DIRECT_ASSIGNMENT',
    "SIMPLE_ASSIGNMENT",
	'WHILE_LOOP',
]

# CUSTOM_PRINT
# import builtins
# original_print = builtins.print

# def custom_print(*args, **kwargs):
# 	formatted_args = [f"{x:.{PRINT_NB_DECIMALS}f}" if isinstance(x, float) else x for x in args]
# 	original_print(*formatted_args, **kwargs)

# builtins.print = custom_print

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

			sign = ''
			if random.random() < 0.15:
				sign = '-'
				
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}{identifier} = {sign}{random.choice(DIGIT)}\n'

			# Update the line_counter
			line_counter += 1
		
		case 'DIRECT_ASSIGNMENT':

			# Select the readable_variables to choose from
			readable_variables = context_stack[-1]['readable_variables']
			if not READ_SECURITY:
				readable_variables = all_assigned_variables

			# Select the writable variables to choose from
			writable_variables = context_stack[-1]['writable_variables']
			if not WRITE_SECURITY:
				writable_variables = VARIABLES

			# Choose the asigned identifier i.e. the name of the variable to assign
			if len(readable_variables) != 0:
				asgnd_id = random.choice(readable_variables)
			else:
				asgnd_id = random.choice(DIGIT)

			# Choose the assignee identifier i.e. the name of the variable to assign to
			assignee = random.choice(writable_variables)

			# Add the assignee to the readable_variables of current context if not already present
			if assignee not in context_stack[-1]['readable_variables']:
				context_stack[-1]['readable_variables'].append(assignee)
			
			# Add the assignee to the all_assigned_variables if not already present
			if assignee not in all_assigned_variables:
				all_assigned_variables.append(assignee)
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}{assignee} = {asgnd_id}\n'

			# Update the current context
			context_stack[-1]['if_state'] = False
			context_stack[-1]['nb_lines_in_block'] += 1

			# Update the line_counter
			line_counter += 1

		case 'SIMPLE_ASSIGNMENT':
			
			# Select the readable_variables to choose from
			readable_variables = context_stack[-1]['readable_variables']
			if not READ_SECURITY:
				readable_variables = all_assigned_variables

			# Select the writable variables to choose from
			writable_variables = context_stack[-1]['writable_variables']
			if not WRITE_SECURITY:
				writable_variables = VARIABLES

			# Operand 1 & 2
			if len(readable_variables) != 0:
				operand1 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
				operand2 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
			else:
				operand1 = random.choice(DIGIT)
				operand2 = random.choice(DIGIT)

			# Operator
			operator = random.choices(ARITHMETIC_OPERATORS, ARITHMETIC_OPERATORS_WEIGHTS)[0]

			# Identifier
			identifier = random.choice(writable_variables)

			# Add the identifier to the readable_variables of current context if not already present
			if identifier not in context_stack[-1]['readable_variables']:
				context_stack[-1]['readable_variables'].append(identifier)
			
			# Add the identifier to the all_assigned_variables if not already present
			if identifier not in all_assigned_variables:
				all_assigned_variables.append(identifier)
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'

			# Update the current context
			context_stack[-1]['if_state'] = False
			context_stack[-1]['nb_lines_in_block'] += 1

			# Update the line_counter
			line_counter += 1

		case 'SIMPLE_IF_STATEMENT':
			
			# Select the readable variables to choose from
			readable_variables = context_stack[-1]['readable_variables']
			if not READ_SECURITY:
				readable_variables = all_assigned_variables
			
			# Choose operand1 (either a variable or a digit)
			if len(readable_variables) != 0:
				operand1 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
				operand2 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
			else:
				operand1 = random.choice(DIGIT)
				operand2 = random.choice(DIGIT)

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
				'current_block': 'IF_BLOCK',
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
			
			# Select the readable variables to choose from
			readable_variables = context_stack[-1]['readable_variables']
			if not READ_SECURITY:
				readable_variables = all_assigned_variables
			
			# Choose operand1 (either a variable or a digit)
			if len(readable_variables) != 0:
				operand1 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
				operand2 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
			else:
				operand1 = random.choice(DIGIT)
				operand2 = random.choice(DIGIT)
			
			# Choose the operator
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
				'current_block': 'IF_BLOCK',
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
				'current_block': 'IF_BLOCK',
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
			update_operator = random.choice(WHILE_LOOP_UPDATE_OPERATORS)
			
			# Choose the relational operator
			relational_operator = random.choice(WHILE_LOOP_RELATIONAL_OPERATORS)
			
			# Choose the control_variable_identifier
			control_variable_identifier = random.choice(context_stack[-1]['writable_variables'])
			
			# Initializing nb_new_lines to 2 since there is always the control_variable_initialization_expression and the while_expression
			nb_new_lines = 2
			
			# Create the required number of tabs for the current context
			tabs = '	' * (len(context_stack)-1)

			# Create the while_prologue_critical_expressions and while_prologue_critical_identifiers
			while_prologue_critical_expressions = []
			while_prologue_critical_identifiers = [control_variable_identifier]

			# ADD UPDATE OPERATOR
			if update_operator == '+':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choices(
					population = WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES,
					weights = WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
					k = 1
				)[0]
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Create the update_operand_value
				update_operand_value = random.choices(
					population=WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES,
					weights=WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
					k=1
				)[0]

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
				control_variable_update_expression = f'{control_variable_identifier} = {control_variable_identifier} + {update_operand_term}\n'
				
				# Choose the number of iterations
				nb_iters = random.choice(WHILE_LOOP_ADD_UO_NB_ITERS)

				# __Create the border__

				# Choose a value for the border corresponding to the number of iterations (give or take 1 iteration actually ...)
				lower_bound = control_variable_initial_value + ((nb_iters-1) * update_operand_value)
				upper_bound = control_variable_initial_value + (nb_iters * update_operand_value)
				border_value = random.randint(a=lower_bound, b=upper_bound)
				
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
				
				if ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
					# Create the while_expression
					if relational_operator in ['<', '<=']:
						while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
					else:
						while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
				else:
					tmp_list = [control_variable_identifier, border_term]
					random.shuffle(tmp_list)
					term1, term2 = tmp_list
					while_expression = f'{tabs}while {term1} {relational_operator} {term2}:\n'
					
			# SUB UPDATE OPERATOR
			elif update_operator == '-':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choices(
					population = WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES,
					weights = WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
					k = 1,
				)[0]
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'

				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__

				# Choose the update_operand_value
				update_operand_value = random.choices(
					population=WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES,
					weights=WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
					k=1
				)[0]

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
				control_variable_update_expression = f'{control_variable_identifier} = {control_variable_identifier} - {update_operand_term}\n'

				# Choose the number of iterations
				nb_iters = random.choice(WHILE_LOOP_SUB_UO_NB_ITERS)

				# __Create the border__

				# Choose a value for the border corresponding to nb_iters (give or take 1 iteration actually ...)
				lower_bound = control_variable_initial_value - ((nb_iters) * update_operand_value)
				upper_bound = control_variable_initial_value - ((nb_iters-1) * update_operand_value)
				border_value = random.randint(a=lower_bound, b=upper_bound)

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

				if ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
					# Create the while_expression
					if relational_operator in ['<', '<=']:
						while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
					else:
						while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
				else:
					tmp_list = [control_variable_identifier, border_term]
					random.shuffle(tmp_list)
					term1, term2 = tmp_list
					while_expression = f'{tabs}while {term1} {relational_operator} {term2}:\n'
			
			# DIV UPDATE OPERATOR
			elif update_operator == '/':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choices(
					population=WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES,
					weights=WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
					k=1
				)[0]
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Choosing the update operand value
				update_operand_value = random.choices(
					population = WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES,
					weights = WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
					k = 1
				)[0]

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
					
					# Set the update_operand_term to the update_operand_identifier for the control_variable_update_expression
					update_operand_term = update_operand_identifier
				
				else:
					
					# Set the update operand term to the update operand value
					update_operand_term = update_operand_value
				
				# Create the control_variable_update_expression
				control_variable_update_expression = f'{control_variable_identifier} = {control_variable_identifier} / {update_operand_term}\n'
				
				# __Create the border__

				# Choosing the number of iterations
				nb_iters = random.choices(
					population = WHILE_LOOP_DIV_UO_NB_ITERS,
					weights = [math.exp(- WHILE_LOOP_DIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT * (1/control_variable_initial_value) * update_operand_value * i) for i in WHILE_LOOP_DIV_UO_NB_ITERS],
					k = 1,
				)[0]

				# Choose a value for the border
				lower_bound = max(int(control_variable_initial_value / (update_operand_value ** nb_iters)), 1)
				upper_bound = max(int(control_variable_initial_value / (update_operand_value ** (nb_iters-1))), 1)
				border_value = random.randint(a=lower_bound, b=upper_bound)
				
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

				if ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
					# Create the while_expression
					if relational_operator in ['<', '<=']:
						while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
					else:
						while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
				else:
					tmp_list = [control_variable_identifier, border_term]
					random.shuffle(tmp_list)
					term1, term2 = tmp_list
					while_expression = f'{tabs}while {term1} {relational_operator} {term2}:\n'

			# FDIV OPERATOR
			elif update_operator == '//':

				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choices(
					population = WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES,
					weights = WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
					k=1,
				)[0]
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__
				
				# Choosing the update operand value
				update_operand_value = random.choices(
					population = WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES,
					weights = WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
					k=1,
				)[0]

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
				control_variable_update_expression = f'{control_variable_identifier} = {control_variable_identifier} // {update_operand_term}\n'
				
				# __Create the border__

				# Choose the number of iterations
				nb_iters = random.choices(
					population = WHILE_LOOP_FDIV_UO_NB_ITERS,
					weights = [math.exp(- WHILE_LOOP_FDIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT * (1/control_variable_initial_value) * update_operand_value * i) for i in WHILE_LOOP_FDIV_UO_NB_ITERS],
					k = 1,
				)[0]
				
				# Choose a value for the border
				lower_bound = control_variable_initial_value // (update_operand_value ** nb_iters)
				upper_bound = control_variable_initial_value // (update_operand_value ** (nb_iters-1))
				border_value = random.randint(a=lower_bound, b=upper_bound)
				
				# If the border_value is 0, we make sure that the relational operator is strict to avoid infinit loops
				if border_value == 0:
					relational_operator = '<' if '<' in relational_operator else '>'
				
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
				
				if ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
					# Create the while_expression
					if relational_operator in ['<', '<=']:
						while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
					else:
						while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
				else:
					tmp_list = [control_variable_identifier, border_term]
					random.shuffle(tmp_list)
					term1, term2 = tmp_list
					while_expression = f'{tabs}while {term1} {relational_operator} {term2}:\n'
			
			# MUL OPERATOR
			elif update_operator == '*':
				
				# __Creating the control_variable__

				# Choosing the control variable initial value
				control_variable_initial_value = random.choices(
					population = WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES,
					weights = WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
					k = 1,
				)[0]
				
				# Create the control variable initialization expression
				control_variable_initialization_expression = f'{tabs}{control_variable_identifier} = {control_variable_initial_value}\n'
				
				# Create the corresponding entry in the while prologue critical expressions
				while_prologue_critical_expressions.append({
					'inti_exp': control_variable_initialization_expression,
					'identifier': control_variable_identifier,
				})

				# __Create the update_operand__

				# Choosing the update operand value
				update_operand_value = random.choices(
					population = WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES,
					weights = WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
					k = 1,
				)[0]

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
				control_variable_update_expression = f'{control_variable_identifier} = {control_variable_identifier} * {update_operand_term}\n'
				
				# __Create the border__

				# Choose the number if iterations
				nb_iters = random.choices(
					population = WHILE_LOOP_MUL_UO_NB_ITERS,
					weights = [math.exp(- WHILE_LOOP_MUL_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT * control_variable_initial_value * update_operand_value * i) for i in WHILE_LOOP_MUL_UO_NB_ITERS],
					k = 1,
				)[0]

				# Choose a value for the border corresponding to the number of iterations (give or take 1 iteration actually ...)
				lower_bound = control_variable_initial_value * (update_operand_value ** (nb_iters-1))
				upper_bound = control_variable_initial_value * (update_operand_value ** nb_iters)
				border_value = random.randint(a=lower_bound, b=upper_bound)
				
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
				
				if ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
					# Create the while_expression
					if relational_operator in ['<', '<=']:
						while_expression = f'{tabs}while {control_variable_identifier} {relational_operator} {border_term}:\n'
					else:
						while_expression = f'{tabs}while {border_term} {relational_operator} {control_variable_identifier}:\n'
				else:
					tmp_list = [control_variable_identifier, border_term]
					random.shuffle(tmp_list)
					term1, term2 = tmp_list
					while_expression = f'{tabs}while {term1} {relational_operator} {term2}:\n'
			# __Create the while_prologue__
			
			# Shuffle the while_prologue_critical_expressions
			random.shuffle(while_prologue_critical_expressions)

			# Initialize while_prologue to empty string
			while_prologue = ''
			
			# Choose the number of intermediate expression to put in the prologue
			nb_max_intermediate_expressions = random.randint(a=0, b=NB_MAX_WHILE_LOOP_PROLOGUE_INTERMEDIATE_EXPRESSIONS)
			
			# Set the new_writable_variables
			new_writable_variables = list(context_stack[-1]['writable_variables'])
			
			# Iterate over the while_prologue_critical_expressions
			for i, el in enumerate(while_prologue_critical_expressions):
				
				# Append the critical expression to the while_prologue
				while_prologue += el['inti_exp']
				
				# Remove the identifier from the new_writable_variables
				new_writable_variables.remove(el['identifier'])
				
				# Add the identifier to the readable_variables of the current context if not already there
				if el['identifier'] not in context_stack[-1]['readable_variables']:
					context_stack[-1]['readable_variables'].append(el['identifier'])
				
				# Add the identifier to all_assigned_variables if not already there
				if el['identifier'] not in all_assigned_variables:
					all_assigned_variables.append(el['identifier'])

				# Choose the number of intermediate expressions to put after this critical expression, if we are at the last critical expression, we finish all the remaining intermediate expressions
				nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions) if i != len(while_prologue_critical_expressions)-1 else nb_max_intermediate_expressions
				
				# Make sure to decrease the number of possible intermediate expressions for next time
				nb_max_intermediate_expressions -= nb_intermediate_expressions

				# Iterate over the number of intermediate expressions
				for _ in range(nb_intermediate_expressions):
					nb_new_lines += 1
					
					# Get the readable variables to choose from
					readable_variables = context_stack[-1]['readable_variables']
					if not READ_SECURITY:
						readable_variables = all_assigned_variables
					
					# Get the writable_variables to choose from
					writable_variables = new_writable_variables
					if not WRITE_SECURITY:
						writable_variables = VARIABLES

					# Choose default operand (either a variable or a digit)
					if len(readable_variables) != 0:
						operand1 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
					else:
						operand1 = random.choice(DIGIT)
					
					# Choose if we add a second operand
					if random.random() < 0.5:
						if len(readable_variables) != 0:
							operand2 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
						else:
							operand2 = random.choice(DIGIT)
						operator = random.choice(ARITHMETIC_OPERATORS)
						additional_expr = f' {operator} {operand2}'
					else:
						additional_expr = ''
					
					# Choose identifier from the new_writable_variables
					identifier = random.choice(writable_variables)
					
					# Add identifier to readable_variables of current context if not already there
					if identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(identifier)
					
					# Add identifier to all assigned variables if not already there
					if identifier not in all_assigned_variables:
						all_assigned_variables.append(identifier)
					
					# Create the intermediate expression
					intermediate_expression = f'{tabs}{identifier} = {operand1}{additional_expr}\n'
					
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
				'current_block': 'WHILE_BLOCK',
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
			
			# Initialize the nb_new_lines to 1 since there is always the while_update_expression
			nb_new_lines = 1

			# Calculate the number of tabs
			tabs = '	' * (len(context_stack)-1)
			
			# Retrieve the default while_update_expression
			default_while_update_expression = context_stack[-1]['while_state']
			
			# __Create the while_update_code__
			
			# Initialize the final_while_update_expression to an empty string by default
			while_update_code = ''

			# Choose if we update the control variable through an intermediate variable
			if random.random() < 0.5:
				
				# Increment nb_new_lines
				nb_new_lines += 1

				# Choose the identifier for the intermediate variable
				intermediate_control_variable_identifier = random.choice(context_stack[-1]['writable_variables'])
				
				# Create temporary writable variables which prevent writing to the intermediate_control_variable_identifier
				tmp_writable_variables = [var for var in context_stack[-1]['writable_variables'] if var != intermediate_control_variable_identifier]
				
				# Create the expression for the intermediate update expression
				default_while_update_expression_right_hand_side = default_while_update_expression.split('=')[-1].strip()
				intermediate_update_expression = f'{tabs}{intermediate_control_variable_identifier} = {default_while_update_expression_right_hand_side}\n'

				# Add the intermediate_update_expression to the while_update_code
				while_update_code += intermediate_update_expression

				# Create the new_while_update_expression
				while_control_variable_identifier = default_while_update_expression.split('=')[0].strip()
				new_while_update_expression = f'{tabs}{while_control_variable_identifier} = {intermediate_control_variable_identifier}\n'
				
				# Add the while_control_variable_identifier to the tmp_writable_variables since its end value is stored in the intermediate_control_variable_identifier
				tmp_writable_variables.append(while_control_variable_identifier)
				
				# Choose a number of intermediate expressions
				nb_intermediate_expressions = random.randint(a=1, b=NB_MAX_WHILE_LOOP_UPDATE_INTERMEDIATE_EXPRESSIONS)
				
				# Increment nb_new_lines with nb_intermediate_expressions
				nb_new_lines += nb_intermediate_expressions

				# Iterate over the number of intermediate expressions
				for _ in range(nb_intermediate_expressions):
					
					# Select the readable variables to choose from
					readable_variables = context_stack[-1]['readable_variables']
					if not READ_SECURITY:
						readable_variables = all_assigned_variables
					
					# Select the writable variables to choose from
					writable_variables = tmp_writable_variables
					if not WRITE_SECURITY:
						writable_variables = VARIABLES
			
					# Choose default operand (either a variable or a digit)
					if len(readable_variables) != 0:
						operand1 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
					else:
						operand1 = random.choice(DIGIT)
					
					# Choose if we add a second operand
					if random.random() < 0.5:
						if len(readable_variables) != 0:
							operand2 = random.choice((random.choice(readable_variables), random.choice(DIGIT)))
						else:
							operand2 = random.choice(DIGIT)
						operator = random.choice(ARITHMETIC_OPERATORS)
						additional_expr = f' {operator} {operand2}'
					else:
						additional_expr = ''
					
					# Choose identifier from the new_writable_variables
					identifier = random.choice(writable_variables)
					
					# Add identifier to readable_variables of current context if not already there
					if identifier not in context_stack[-1]['readable_variables']:
						context_stack[-1]['readable_variables'].append(identifier)
					
					# Add identifier to all assigned variables if not already there
					if identifier not in all_assigned_variables:
						all_assigned_variables.append(identifier)
					
					# Create the intermediate expression
					intermediate_expression = f'{tabs}{identifier} = {operand1}{additional_expr}\n'

					# Add it the the while_update_code
					while_update_code = while_update_code + intermediate_expression
				
				# Add the new_while_update_expression to the while_update_code
				while_update_code += new_while_update_expression
			
			# Else we just use the default_while_update_expression
			else:
				while_update_code = f'{tabs}{default_while_update_expression}'

			# Update the current context
			context_stack[-1]['if_state'] = False
			context_stack[-1]['while_state'] = None
			context_stack[-1]['nb_lines_in_block'] += nb_new_lines

			# Append the code
			code = code + while_update_code

			# Updating the line_counter
			line_counter += nb_new_lines

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
				'current_block': 'FOR_BLOCK',
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
			
			# Select the readable_variables to choose from
			readable_variables = context_stack[-1]['readable_variables']
			if not READ_SECURITY:
				readable_variables = all_assigned_variables
			
			# Choose the printed_term
			if len(readable_variables) != 0:
				printed_term = random.choices(population = (random.choice(readable_variables), random.choice(DIGIT)), weights = (0.95, 0.05), k = 1)[0]
			else:
				printed_term = random.choice(DIGIT)
			
			# Append the code
			tabs = '	' * (len(context_stack)-1)
			code = code + f'{tabs}print({printed_term})\n'

			# Update the current context
			context_stack[-1]['nb_lines_in_block'] += 1
			context_stack[-1]['if_state'] = False

			# Updating the line_counter
			line_counter += 1
		
		case 'END':
			# Do nothing
			pass

		case _:
			raise Exception(f'No match for gen_action {gen_action}')


# __FUNCTION__: QUEUE_GEN_ACTIONS
def queue_gen_actions():
	
	# If the line_counter is less the MIN_INIT we return an INITIALIZATION
	if line_counter <= MIN_INIT:
		context_stack[-1]['actions_queue'].append("INITIALIZATION")
		
		# Exit
		return
	
	# Elif it's above MAX_LENGTH
	if line_counter > MAX_LENGTH:
		
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
	if len(context_stack) > 1 and random.random() > (1/(1+UNINDENTATION_SPEED)) ** context_stack[-1]['nb_lines_in_block']:
		
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

		# Remove the print statements in the middle of the code
		potential_keywords.remove('DISPLAY')
		
		# If we are in a while loop we remove the possibility to have an inner while loop
		if context_stack[-1]['current_block'] == 'WHILE_BLOCK':
			potential_keywords.remove('WHILE_LOOP')
		
		# If we are in an if block we remove all inner if statements and while loops
		if context_stack[-1]['current_block'] == 'IF_BLOCK':
			potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in indentation_statements]
		
		# Check for while_state
		if context_stack[-1]['while_state']:
			potential_keywords.append('WHILE_UPDATE')
		
		# In case we achieved MAX_DEPTH or MAX_SUB_BLOCKS inside the current context we remove the indentation statements
		# remove the indentation_statements from potential_keywords
		if len(context_stack) - 1 >=  MAX_DEPTH or context_stack[-1]["nb_blocks"] >= MAX_SUB_BLOCKS:
			potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in indentation_statements]
		
		# Else If we are not in an If statement we remove the elif + else
		elif not context_stack[-1]["if_state"]:
			potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in {"SIMPLE_ELIF_STATEMENT", "ELSE_STATEMENT"}]

		# We add the END keyword if we are at indentation level 0 and the line_counter is above MIN_LENGTH
		if len(context_stack) == 1 and line_counter > MIN_LENGTH:
			potential_keywords.append("END")

		# We choose a keyword randomly and queue it
		pk_weight = {
			'INITIALIZATION': 1,
			'SIMPLE_ASSIGNMENT': 10,
			'DIRECT_ASSIGNMENT': 1,
			'DISPLAY': 1,
			'END': 1,
		}
		potential_keywords = [(potential_keyword,pk_weight[potential_keyword]) for potential_keyword in potential_keywords]
		random.shuffle(potential_keywords)
		keyword = random.choices([tup[0] for tup in potential_keywords], [tup[1] for tup in potential_keywords])[0]
		
		# If we end the code snippet, add a DISPLAY statement right before ending
		if keyword == 'END':
			context_stack[-1]['actions_queue'].append('DISPLAY')
		
		context_stack[-1]['actions_queue'].append(keyword)
		
		# Exit
		return


# __FUNCTION__: GENERATE_RANDOM_CODE
def generate_random_code():
	"""
	Generates a random code snippet by orchestrating the use of the 'queue_gen_actions' and 'execute_gen_actions' functions
	"""
	
	# Initialize the context_stack
	global context_stack 
	context_stack = list()
	context_stack.append(
		{
			'current_block': 'ROOT',
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

	# Initialize the line_counter
	global line_counter
	line_counter = 1
	
	# Initialize the code
	global code
	code = ''

	# Initialize the all_assigned_variables
	global all_assigned_variables
	all_assigned_variables = list()

	# Initiliaze MIN_INIT
	global MIN_INIT
	MIN_INIT = random.randint(0, MIN_INIT_MAX)
	
	# Enter the gen_loop
	gen_action = 'START'
	while gen_action != 'END':

		# Call the queue_actions function
		queue_gen_actions()

		# Loop over the queued actions for the current context
		while (len(context_stack[-1]['actions_queue']) != 0) and (gen_action := context_stack[-1]['actions_queue'].popleft()) != 'END':
			execute_gen_action(gen_action)
	
	# Return the code
	return code


# Custom exception, raised when a variable's absolute value gets higher than max_val_value
class VariableValueOverflowError(Exception):
	def __init__(self, message):
		super().__init__(message)

# '/data/yb2618/Tiny-Language-Models-Framework/frcg-random-states/random_state_2024-12-12_08-08.bin'

# If the script is run as a standalone script
if __name__ == "__main__":

	# __Parse the command line arguments__

	parser = argparse.ArgumentParser(description = "Full Random TinyPy Generator")
	
	parser.add_argument("--random_state"			, default = None, help = "Path to python random state to be loaded if any")
	parser.add_argument("--nb_programs"				, default = 250_00, help = "Number of programs to be generated")
	parser.add_argument("--output_file"				, default = "./data-ds-51-stage-1/aronly_direct_output_snippets.txt", help = "Number of programs to be generated")
	parser.add_argument("--timeout"					, default = 2, help = "Number of seconds to wait for a process to terminate")
	parser.add_argument("--log_file"				, default = "./log.txt", help = "The path to the logging file for monitoring progress")
	parser.add_argument("--log_interval"			, default = 10000, help = "The number of code snippets generations before logging to the --log_file for monitoring progress")
	parser.add_argument("--deduplicate"				, help = "Whether to perform deduplication of generated programs (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_deduplication_trials", default = 50, help = "The maximum number of consecutive trials when deduplication occurs")
	parser.add_argument("--programs_separator"		, default = "# code", help = "String to put at the top of each code example (Defaults to empty string)")
	parser.add_argument("--use_tqdm"				, default='true', help = "Whether or not to use tqdm for monitoring progress (set to True for true, False for anything else), defaults to True)")
	parser.add_argument("--max_var_value"			, default = 999, help = "The maximum value above which the absolute value of created variables must not go")

	args = parser.parse_args()

	random_state 				= args.random_state
	nb_programs 				= int(args.nb_programs)
	output_file 				= args.output_file
	timeout 					= int(args.timeout)
	log_file 					= args.log_file
	log_interval 				= int(args.log_interval)
	deduplicate 				= True if args.deduplicate in ("true", None) else False
	max_deduplication_trials 	= int(args.max_deduplication_trials)
	programs_separator 			= args.programs_separator + '\n' if args.programs_separator else ""
	use_tqdm 					= True if args.use_tqdm  in ("true", None) else False 
	max_var_value 				= args.max_var_value

	# Save or set the random state
	if args.random_state is None:
		random_state = random.getstate()
		now = datetime.datetime.now()
		date_hour = now.strftime("%Y-%m-%d_%H-%M-%S")
		Path("./frcg-random-states").mkdir(parents = True, exist_ok = True)
		with open(f"./frcg-random-states/random_state_{date_hour}.bin", "wb") as f:
			pickle.dump(random_state, f)
	else:
		with open(args.random_state, "rb") as f:
			random_state = pickle.load(f)
			random.setstate(random_state)
			
	# Initialize nb_generated_programs
	nb_generated_programs 	= 0

	# Initialize the error counters
	nb_zero_divisions 		= 0
	nb_var_value_overflows 	= 0
	nb_name_errors 			= 0
	nb_timeouts 			= 0
	
	# Initialize the hashes set and nb_deduplication_trials
	hashes = set()
	nb_deduplication_trials = 0
	
	# Set up the execution environment boilerplate
	exec_env_boilerplate = f"""
from sys import settrace

def line_tracer(frame, event, arg):
	if event == "exception" :
		raise arg[0]
	for var_value in frame.f_locals.values():
		# if not(isinstance(var_value, (int, float))): continue
		if var_value > {max_var_value} or var_value < -{max_var_value}:
			raise VariableValueOverflowError("Variable Value Overflow")
	return line_tracer

def global_tracer(frame, event, arg):
	func_name = frame.f_code.co_name
	if func_name != 'func':
		return None
	return line_tracer

settrace(global_tracer)
try:
	func()
finally:
	settrace(None)"""
	
	# Set the starting_time and the first checkpoint_time (for logging)
	start_time = datetime.datetime.now()
	checkpoint_time = start_time

	# Open the logging file
	f_log_file = open(log_file, "w")
	
	# Open the data output file
	f = open(output_file, "w")

	# Check if we use tqdm
	if use_tqdm:
		from tqdm import tqdm
		pbar = tqdm(desc="Generation", total=nb_programs)


	# Define the TiemoutException to be raised at timeouts
	class TimeoutException(Exception):
		pass

	if not WRITE_SECURITY:
		# Define the function called by the signal, it will raise TimeoutException
		def timeout_handler(signum, frame):
			raise TimeoutException()

		# Set the signal SIGALARM to call timeout_handler
		signal.signal(signal.SIGALRM, timeout_handler)
	
	# Launch the generation loop
	while nb_generated_programs < nb_programs:
		
		# Check if it's log interval
		if nb_generated_programs % log_interval == 0:
			now = datetime.datetime.now()
			f_log_file.write(f"Generated {nb_generated_programs:<{len(str(nb_programs))}} programs,  absolute time: {now - start_time},  relative time: {now - checkpoint_time}\n")
			f_log_file.flush()
			checkpoint_time = now
		
		# Generate the code
		code = generate_random_code()
		code = code.strip('\n')

		# If we check for deduplicates
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
		
		# Prepare the execution environment
		indented = "\n".join([f"	{line}" for line in code.split("\n")])
		func = "def func():\n" + indented
		exec_env = func + exec_env_boilerplate
		
		# # Try to execute the generated code
		sio = StringIO()
		try:
			with redirect_stdout(sio):

				# Check if we are not using WRITE_SECURITY
				if not WRITE_SECURITY:
					signal.alarm(TIMEOUT)
				
				# Execute the code in a controlled environment
				exec(exec_env, {
					"VariableValueOverflowError" : VariableValueOverflowError
				})
			
			# __If we are here, it means that the code has been executed successfully__

			# Resetting the alarm to 0 if not using the WRITE_SECURITY
			if not WRITE_SECURITY:
				signal.alarm(0)
			
			# Getting the output
			output = sio.getvalue()

			# If we are force printing and the sio output is empty
			if FORCE_PRINT and output == '':
				
				# Select the readable_variables to choose from
				readable_variables = context_stack[-1]['readable_variables']
				if not READ_SECURITY:
					readable_variables = all_assigned_variables
				
				# Choose the printed_term
				if len(readable_variables) != 0:
					printed_term = random.choices(population = (random.choice(readable_variables), random.choice(DIGIT)), weights = (0.95, 0.05), k = 1)[0]
				else:
					printed_term = random.choice(DIGIT)
				
				# Adding a print at the end of the code
				final_print_expression = f'print({printed_term})'
				code += f'\n{final_print_expression}'
				
				# Re-Create the execution environment
				func += f'\n\t{final_print_expression}'
				exec_env = func + exec_env_boilerplate
				sio = StringIO()
				with redirect_stdout(sio):
					exec(exec_env, {
						"VariableValueOverflowError" : VariableValueOverflowError
					})
				output = sio.getvalue()
			
			# Create the final results = code + formatted output
			result = programs_separator + code + "\n# output\n# " + "\n# ".join(output.split("\n")[:-1])
			
			# Write the result to the destination file
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
		except NameError as e:
			nb_name_errors += 1
		except (Exception, KeyboardInterrupt) as e:
			print(f'Code Snippet Execution Error at {nb_generated_programs}:', e)
			with open('error_code.txt', 'w') as f:
				f.write(f'PROGRAM PROBLEM#{nb_generated_programs}\n'+exec_env)
		# except TimeoutException:
		# 	nb_timeouts += 1
			break

		if use_tqdm:
			pbar.set_description(f"ZeroDiv: {nb_zero_divisions:,} |Overflows: {nb_var_value_overflows:,} |NameErrors: {nb_name_errors:,} |Timeouts: {nb_timeouts:,}")
	
	nb_all_attempted_programs = nb_programs + nb_zero_divisions + nb_var_value_overflows + nb_name_errors + nb_timeouts
	print(f"percentage of zero divisions: {nb_zero_divisions/nb_all_attempted_programs * 100:.2f}%")
	print(f"percentage of overflows: {nb_var_value_overflows/nb_all_attempted_programs * 100:.2f}%")
	print(f"percentage of name errors: {nb_name_errors/nb_all_attempted_programs * 100:.2f}%")
	print(f"percentage of timeouts: {nb_timeouts/nb_all_attempted_programs * 100:.2f}%")
	
	# Closing the logging and data output files
	f_log_file.close()
	f.close()