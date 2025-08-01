from collections import deque
from io import StringIO
from contextlib import redirect_stdout
import hashlib
import random
import math
import signal
import tqdm
import datetime

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


	def push(self, init_dict=dict(), init_queue=deque()):
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


	def instantiate_code(self):
		for self.keyword in self.context_stack.top():
			#|===================================================|
			#|[*] [// User-defined low-level generation rules //]|
			#|===================================================|

			if self.keyword == 'UNINDENT':
				# Pop the previous context
				previous_context = self.context_stack.pop()
				# Updat the nb_lines_in_block of the current context
				self.context_stack.top().GID['nb_lines_in_block'] += previous_context.GID['nb_lines_in_block']
			
			elif self.keyword == 'INITIALIZATION':
				
				# Choose the identifier

				identifier = random.choice(self.context_stack.top().GID['writable_variables'])
				
				# Update the current context
				self.context_stack.top().GID['if_state'] = False
				self.context_stack.top().GID['nb_lines_in_block'] += 1
				if identifier not in self.context_stack.top().GID['readable_variables']:
					self.context_stack.top().GID['readable_variables'].append(identifier)
				
				sign = ''
				if random.random() < 0.15:
					sign = '-'
					
				# Append the code
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}{identifier} = {sign}{random.choice(self.DIGIT)}\n'

				# Update the self.line_counter
				self.line_counter += 1
			
			elif self.keyword == 'DIRECT_ASSIGNMENT':

				# Select the readable_variables to choose from
				readable_variables = self.context_stack.top().GID['readable_variables']
				if not self.READ_SECURITY:
					readable_variables = self.all_assigned_variables

				# Select the writable variables to choose from
				writable_variables = self.context_stack.top().GID['writable_variables']
				if not self.WRITE_SECURITY:
					writable_variables = self.VARIABLES

				# Choose the asigned identifier i.e. the name of the variable to assign
				if len(readable_variables) != 0:
					asgnd_id = random.choice(readable_variables)
				else:
					asgnd_id = random.choice(self.DIGIT)

				# Choose the assignee identifier i.e. the name of the variable to assign to
				assignee = random.choice(writable_variables)

				# Add the assignee to the readable_variables of current context if not already present
				if assignee not in self.context_stack.top().GID['readable_variables']:
					self.context_stack.top().GID['readable_variables'].append(assignee)
				
				# Add the assignee to the self.all_assigned_variables if not already present
				if assignee not in self.all_assigned_variables:
					self.all_assigned_variables.append(assignee)
				
				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}{assignee} = {asgnd_id}\n'

				# Update the current context
				self.context_stack.top().GID['if_state'] = False
				self.context_stack.top().GID['nb_lines_in_block'] += 1

				# Update the self.line_counter
				self.line_counter += 1

			elif self.keyword == 'SIMPLE_ASSIGNMENT':
				
				# Select the readable_variables to choose from
				readable_variables = self.context_stack.top().GID['readable_variables']
				if not self.READ_SECURITY:
					readable_variables = self.all_assigned_variables

				# Select the writable variables to choose from
				writable_variables = self.context_stack.top().GID['writable_variables']
				if not self.WRITE_SECURITY:
					writable_variables = self.VARIABLES

				# Operand 1 & 2
				if len(readable_variables) != 0:
					operand1 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
					operand2 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
				else:
					operand1 = random.choice(self.DIGIT)
					operand2 = random.choice(self.DIGIT)

				# Operator
				operator = random.choices(self.ARITHMETIC_OPERATORS, self.ARITHMETIC_OPERATORS_WEIGHTS)[0]

				# Identifier
				identifier = random.choice(writable_variables)

				# Add the identifier to the readable_variables of current context if not already present
				if identifier not in self.context_stack.top().GID['readable_variables']:
					self.context_stack.top().GID['readable_variables'].append(identifier)
				
				# Add the identifier to the self.all_assigned_variables if not already present
				if identifier not in self.all_assigned_variables:
					self.all_assigned_variables.append(identifier)
				
				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}{identifier} = {operand1} {operator} {operand2}\n'

				# Update the current context
				self.context_stack.top().GID['if_state'] = False
				self.context_stack.top().GID['nb_lines_in_block'] += 1

				# Update the self.line_counter
				self.line_counter += 1

			elif self.keyword == 'SIMPLE_IF_STATEMENT':
				
				# Select the readable variables to choose from
				readable_variables = self.context_stack.top().GID['readable_variables']
				if not self.READ_SECURITY:
					readable_variables = self.all_assigned_variables
				
				# Choose operand1 (either a variable or a digit)
				if len(readable_variables) != 0:
					operand1 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
					operand2 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
				else:
					operand1 = random.choice(self.DIGIT)
					operand2 = random.choice(self.DIGIT)

				# Choose relational operator
				operator = random.choice(self.RELATIONAL_OPERATORS)
				
				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}if {operand1} {operator} {operand2}:\n'
				
				# Update the self.line_counter
				self.line_counter += 1
				
				# Update the current context
				self.context_stack.top().GID['if_state'] = True
				self.context_stack.top().GID['nb_lines_in_block'] += 1
				self.context_stack.top().GID['nb_blocks'] += 1
				self.context_stack.top().GID['nb_if_blocks'] += 1

				# Stack the new context
				self.context_stack.push({
					'current_block': 'IF_BLOCK',
					'nb_if_blocks': 0,
					'nb_while_loops': 0,
					'nb_for_loops': 0,
					'nb_blocks': 0,
					'if_state': False,
					'while_state': None,
					'readable_variables': list(self.context_stack.top().GID['readable_variables']),
					'writable_variables': list(self.context_stack.top().GID['writable_variables']),
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})

			elif self.keyword == 'SIMPLE_ELIF_STATEMENT':
				
				# Select the readable variables to choose from
				readable_variables = self.context_stack.top().GID['readable_variables']
				if not self.READ_SECURITY:
					readable_variables = self.all_assigned_variables
				
				# Choose operand1 (either a variable or a digit)
				if len(readable_variables) != 0:
					operand1 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
					operand2 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
				else:
					operand1 = random.choice(self.DIGIT)
					operand2 = random.choice(self.DIGIT)
				
				# Choose the operator
				operator = random.choice(self.RELATIONAL_OPERATORS)

				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}elif {operand1} {operator} {operand2}:\n'

				# Update the current context
				self.context_stack.top().GID['nb_lines_in_block'] += 1

				# Update the self.line_counter
				self.line_counter += 1

				# Stack the new context
				self.context_stack.push({
					'current_block': 'IF_BLOCK',
					'nb_if_blocks': 0,
					'nb_while_loops': 0,
					'nb_for_loops': 0,
					'nb_blocks': 0,
					'if_state': False,
					'while_state': None,
					'readable_variables': list(self.context_stack.top().GID['readable_variables']),
					'writable_variables': list(self.context_stack.top().GID['writable_variables']),
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})
			
			elif self.keyword == 'ELSE_STATEMENT':
				
				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}else:\n'
				
				# __Update the self.context_stack__

				# Update the current context
				self.context_stack.top().GID['nb_lines_in_block'] += 1
				self.context_stack.top().GID['if_state'] = False

				# Stack the new context
				self.context_stack.push({
					'current_block': 'IF_BLOCK',
					'nb_if_blocks': 0,
					'nb_while_loops': 0,
					'nb_for_loops': 0,
					'nb_blocks': 0,
					'if_state': False,
					'while_state': None,
					'readable_variables': list(self.context_stack.top().GID['readable_variables']),
					'writable_variables': list(self.context_stack.top().GID['writable_variables']),
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})
				
				# Update the self.line_counter
				self.line_counter += 1

			elif self.keyword == 'WHILE_LOOP':
				
				# Choose the update operator
				update_operator = random.choice(self.WHILE_LOOP_UPDATE_OPERATORS)
				
				# Choose the relational operator
				relational_operator = random.choice(self.WHILE_LOOP_RELATIONAL_OPERATORS)
				
				# Choose the control_variable_identifier
				control_variable_identifier = random.choice(self.context_stack.top().GID['writable_variables'])
				
				# Initializing nb_new_lines to 2 since there is always the control_variable_initialization_expression and the while_expression
				nb_new_lines = 2
				
				# Create the required number of tabs for the current context
				tabs = '	' * (len(self.context_stack)-1)

				# Create the while_prologue_critical_expressions and while_prologue_critical_identifiers
				while_prologue_critical_expressions = []
				while_prologue_critical_identifiers = [control_variable_identifier]

				# ADD UPDATE OPERATOR
				if update_operator == '+':
					
					# __Creating the control_variable__

					# Choosing the control variable initial value
					control_variable_initial_value = random.choices(
						population = self.WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES,
						weights = self.WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
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
						population=self.WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES,
						weights=self.WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
						k=1
					)[0]

					# Choose if we store the update_operand in a variable
					if random.random() < 0.5:
						
						# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
					nb_iters = random.choice(self.WHILE_LOOP_ADD_UO_NB_ITERS)

					# __Create the border__

					# Choose a value for the border corresponding to the number of iterations (give or take 1 iteration actually ...)
					lower_bound = control_variable_initial_value + ((nb_iters-1) * update_operand_value)
					upper_bound = control_variable_initial_value + (nb_iters * update_operand_value)
					border_value = random.randint(a=lower_bound, b=upper_bound)
					
					# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
					if random.random() < 0.5:
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
					
					if self.ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
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
						population = self.WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES,
						weights = self.WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
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
						population=self.WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES,
						weights=self.WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
						k=1
					)[0]

					# Choose if we store the update_operand in a variable
					if random.random() < 0.5:
						
						# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
					nb_iters = random.choice(self.WHILE_LOOP_SUB_UO_NB_ITERS)

					# __Create the border__

					# Choose a value for the border corresponding to nb_iters (give or take 1 iteration actually ...)
					lower_bound = control_variable_initial_value - ((nb_iters) * update_operand_value)
					upper_bound = control_variable_initial_value - ((nb_iters-1) * update_operand_value)
					border_value = random.randint(a=lower_bound, b=upper_bound)

					# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
					if random.random() < 0.5:
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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

					if self.ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
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
						population=self.WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES,
						weights=self.WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
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
						population = self.WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES,
						weights = self.WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
						k = 1
					)[0]

					# Choose if we store the update_operand in a variable
					if random.random() < 0.5:
						
						# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
						population = self.WHILE_LOOP_DIV_UO_NB_ITERS,
						weights = [math.exp(- self.WHILE_LOOP_DIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT * (1/control_variable_initial_value) * update_operand_value * i) for i in self.WHILE_LOOP_DIV_UO_NB_ITERS],
						k = 1,
					)[0]

					# Choose a value for the border
					lower_bound = max(int(control_variable_initial_value / (update_operand_value ** nb_iters)), 1)
					upper_bound = max(int(control_variable_initial_value / (update_operand_value ** (nb_iters-1))), 1)
					border_value = random.randint(a=lower_bound, b=upper_bound)
					
					# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
					if random.random() < 0.5:
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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

					if self.ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
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
						population = self.WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES,
						weights = self.WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
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
						population = self.WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES,
						weights = self.WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
						k=1,
					)[0]

					# Choose if we store the update_operand in a variable
					if random.random() < 0.5:
						
						# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
						population = self.WHILE_LOOP_FDIV_UO_NB_ITERS,
						weights = [math.exp(- self.WHILE_LOOP_FDIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT * (1/control_variable_initial_value) * update_operand_value * i) for i in self.WHILE_LOOP_FDIV_UO_NB_ITERS],
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
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
					
					if self.ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
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
						population = self.WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES,
						weights = self.WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS,
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
						population = self.WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES,
						weights = self.WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES_WEIGHTS,
						k = 1,
					)[0]

					# Choose if we store the update_operand in a variable
					if random.random() < 0.5:
						
						# Choose the identifier for the update operand from the writable variables except the control_variable_identifier
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
						population = self.WHILE_LOOP_MUL_UO_NB_ITERS,
						weights = [math.exp(- self.WHILE_LOOP_MUL_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT * control_variable_initial_value * update_operand_value * i) for i in self.WHILE_LOOP_MUL_UO_NB_ITERS],
						k = 1,
					)[0]

					# Choose a value for the border corresponding to the number of iterations (give or take 1 iteration actually ...)
					lower_bound = control_variable_initial_value * (update_operand_value ** (nb_iters-1))
					upper_bound = control_variable_initial_value * (update_operand_value ** nb_iters)
					border_value = random.randint(a=lower_bound, b=upper_bound)
					
					# Choose if we store the border in a variable, same structure as the update operand so no need to comment it
					if random.random() < 0.5:
						tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var not in while_prologue_critical_identifiers]
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
					
					if self.ALWAYS_EXECUTE_WHILE_LOOP or ('=' in relational_operator and control_variable_initial_value == border_value):
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
				nb_max_intermediate_expressions = random.randint(a=0, b=self.NB_MAX_WHILE_LOOP_PROLOGUE_INTERMEDIATE_EXPRESSIONS)
				
				# Set the new_writable_variables
				new_writable_variables = list(self.context_stack.top().GID['writable_variables'])
				
				# Iterate over the while_prologue_critical_expressions
				for i, el in enumerate(while_prologue_critical_expressions):
					
					# Append the critical expression to the while_prologue
					while_prologue += el['inti_exp']
					
					# Remove the identifier from the new_writable_variables
					new_writable_variables.remove(el['identifier'])
					
					# Add the identifier to the readable_variables of the current context if not already there
					if el['identifier'] not in self.context_stack.top().GID['readable_variables']:
						self.context_stack.top().GID['readable_variables'].append(el['identifier'])
					
					# Add the identifier to self.all_assigned_variables if not already there
					if el['identifier'] not in self.all_assigned_variables:
						self.all_assigned_variables.append(el['identifier'])

					# Choose the number of intermediate expressions to put after this critical expression, if we are at the last critical expression, we finish all the remaining intermediate expressions
					nb_intermediate_expressions = random.randint(0, nb_max_intermediate_expressions) if i != len(while_prologue_critical_expressions)-1 else nb_max_intermediate_expressions
					
					# Make sure to decrease the number of possible intermediate expressions for next time
					nb_max_intermediate_expressions -= nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						nb_new_lines += 1
						
						# Get the readable variables to choose from
						readable_variables = self.context_stack.top().GID['readable_variables']
						if not self.READ_SECURITY:
							readable_variables = self.all_assigned_variables
						
						# Get the writable_variables to choose from
						writable_variables = new_writable_variables
						if not self.WRITE_SECURITY:
							writable_variables = self.VARIABLES

						# Choose default operand (either a variable or a digit)
						if len(readable_variables) != 0:
							operand1 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
						else:
							operand1 = random.choice(self.DIGIT)
						
						# Choose if we add a second operand
						if random.random() < 0.5:
							if len(readable_variables) != 0:
								operand2 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
							else:
								operand2 = random.choice(self.DIGIT)
							operator = random.choice(self.ARITHMETIC_OPERATORS)
							additional_expr = f' {operator} {operand2}'
						else:
							additional_expr = ''
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in self.context_stack.top().GID['readable_variables']:
							self.context_stack.top().GID['readable_variables'].append(identifier)
						
						# Add identifier to all assigned variables if not already there
						if identifier not in self.all_assigned_variables:
							self.all_assigned_variables.append(identifier)
						
						# Create the intermediate expression
						intermediate_expression = f'{tabs}{identifier} = {operand1}{additional_expr}\n'
						
						# Append it to while_prologue
						while_prologue += intermediate_expression

				# Append the while_prologue and the while_expression to the self.code_snippet
				self.code_snippet = self.code_snippet + while_prologue + while_expression

				# Update the self.line_counter
				self.line_counter += nb_new_lines

				# Update the current context
				self.context_stack.top().GID['nb_lines_in_block'] += nb_new_lines
				self.context_stack.top().GID['nb_while_loops'] += 1
				self.context_stack.top().GID['nb_blocks'] += 1
				self.context_stack.top().GID['if_state'] = False

				# Stack the new context
				self.context_stack.push({
					'current_block': 'WHILE_BLOCK',
					'nb_if_blocks': 0,
					'nb_while_loops': 0,
					'nb_for_loops': 0,
					'nb_blocks': 0,
					'if_state': False,
					'while_state': control_variable_update_expression,
					'readable_variables': list(self.context_stack.top().GID['readable_variables']),
					'writable_variables': new_writable_variables,
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})
			
			elif self.keyword == 'WHILE_UPDATE':
				
				# Initialize the nb_new_lines to 1 since there is always the while_update_expression
				nb_new_lines = 1

				# Calculate the number of tabs
				tabs = '	' * (len(self.context_stack)-1)
				
				# Retrieve the default while_update_expression
				default_while_update_expression = self.context_stack.top().GID['while_state']
				
				# __Create the while_update_code__
				
				# Initialize the final_while_update_expression to an empty string by default
				while_update_code = ''

				# Choose if we update the control variable through an intermediate variable
				if random.random() < 0.5:
					
					# Increment nb_new_lines
					nb_new_lines += 1

					# Choose the identifier for the intermediate variable
					intermediate_control_variable_identifier = random.choice(self.context_stack.top().GID['writable_variables'])
					
					# Create temporary writable variables which prevent writing to the intermediate_control_variable_identifier
					tmp_writable_variables = [var for var in self.context_stack.top().GID['writable_variables'] if var != intermediate_control_variable_identifier]
					
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
					nb_intermediate_expressions = random.randint(a=1, b=self.NB_MAX_WHILE_LOOP_UPDATE_INTERMEDIATE_EXPRESSIONS)
					
					# Increment nb_new_lines with nb_intermediate_expressions
					nb_new_lines += nb_intermediate_expressions

					# Iterate over the number of intermediate expressions
					for _ in range(nb_intermediate_expressions):
						
						# Select the readable variables to choose from
						readable_variables = self.context_stack.top().GID['readable_variables']
						if not self.READ_SECURITY:
							readable_variables = self.all_assigned_variables
						
						# Select the writable variables to choose from
						writable_variables = tmp_writable_variables
						if not self.WRITE_SECURITY:
							writable_variables = self.VARIABLES
				
						# Choose default operand (either a variable or a digit)
						if len(readable_variables) != 0:
							operand1 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
						else:
							operand1 = random.choice(self.DIGIT)
						
						# Choose if we add a second operand
						if random.random() < 0.5:
							if len(readable_variables) != 0:
								operand2 = random.choice((random.choice(readable_variables), random.choice(self.DIGIT)))
							else:
								operand2 = random.choice(self.DIGIT)
							operator = random.choice(self.ARITHMETIC_OPERATORS)
							additional_expr = f' {operator} {operand2}'
						else:
							additional_expr = ''
						
						# Choose identifier from the new_writable_variables
						identifier = random.choice(writable_variables)
						
						# Add identifier to readable_variables of current context if not already there
						if identifier not in self.context_stack.top().GID['readable_variables']:
							self.context_stack.top().GID['readable_variables'].append(identifier)
						
						# Add identifier to all assigned variables if not already there
						if identifier not in self.all_assigned_variables:
							self.all_assigned_variables.append(identifier)
						
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
				self.context_stack.top().GID['if_state'] = False
				self.context_stack.top().GID['while_state'] = None
				self.context_stack.top().GID['nb_lines_in_block'] += nb_new_lines

				# Append the self.code_snippet
				self.code_snippet = self.code_snippet + while_update_code

				# Updating the self.line_counter
				self.line_counter += nb_new_lines

			elif self.keyword == 'FOR_LOOP':
				control_variable_identifier = random.choice(self.context_stack.top().GID['writable_variables'])
				initial_value = random.choice(self.DIGIT)
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
				
				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}for {control_variable_identifier} in ({initial_value}, {border_value}{step_expression}):\n'
				
				# __Updating the self.context_stack__
				
				# Update the current context
				self.context_stack.top().GID['nb_for_loops'] += 1
				self.context_stack.top().GID['nb_lines_in_block'] += 1
				self.context_stack.top().GID['if_state'] = False
				self.context_stack.top().GID['nb_blocks'] += 1
				if control_variable_identifier not in self.context_stack.top().GID['readable_variables']:
					self.context_stack.top().GID['readable_variables'].append(control_variable_identifier)
				
				# Stack the new context
				self.context_stack.push({
					'current_block': 'FOR_BLOCK',
					'nb_if_blocks': 0,
					'nb_while_loops': 0,
					'nb_for_loops': 0,
					'nb_blocks': 0,
					'if_state': False,
					'while_state': None,
					'readable_variables': list(self.context_stack.top().GID['readable_variables']),
					'writable_variables': list(self.context_stack.top().GID['writable_variables']),
					'nb_lines_in_block': 0,
					'actions_queue': deque(),
				})

				# Updating the self.line_counter
				self.line_counter += 1
			
			elif self.keyword == 'DISPLAY':
				
				# Select the readable_variables to choose from
				readable_variables = self.context_stack.top().GID['readable_variables']
				if not self.READ_SECURITY:
					readable_variables = self.all_assigned_variables
				
				# Choose the printed_term
				if len(readable_variables) != 0:
					printed_term = random.choices(population = (random.choice(readable_variables), random.choice(self.DIGIT)), weights = (0.95, 0.05), k = 1)[0]
				else:
					printed_term = random.choice(self.DIGIT)
				
				# Append the self.code_snippet
				tabs = '	' * (len(self.context_stack)-1)
				self.code_snippet = self.code_snippet + f'{tabs}print({printed_term})\n'

				# Update the current context
				self.context_stack.top().GID['nb_lines_in_block'] += 1
				self.context_stack.top().GID['if_state'] = False

				# Updating the self.line_counter
				self.line_counter += 1
			
			elif self.keyword == 'END':
				# Do nothing
				pass

			else:
				raise Exception(f'No match for keyword {self.keyword}')			

	
	def construct_skeleton(self):
		keywords_sequence = []

		#|====================================================|
		#|[*] [// User-defined high-level generation rules //]|
		#|====================================================|

		# If the self.line_counter is less the MIN_INIT we return an INITIALIZATION
		if self.line_counter <= self.MIN_INIT:
			keywords_sequence.append("INITIALIZATION")
		
		# Elif it's above MAX_LENGTH
		elif self.line_counter > self.MAX_LENGTH:
			
			# If we are at the indentation level 0 we can directly END the self.code_snippet
			if len(self.context_stack) == 1:
				keywords_sequence.append("END")
			
			# Else If we are in a while_state we generate the while update expression and unindent right after
			elif self.context_stack.top().GID['while_state']:
				keywords_sequence.append('WHILE_UPDATE')
				keywords_sequence.append('UNINDENT')
				
			# Else if we are in a block with at least one line of self.code_snippet, we directly unindent
			elif self.context_stack.top().GID["nb_lines_in_block"] != 0:
					keywords_sequence.append('UNINDENT')
			
			# Else => we are in a block with no lines of self.code_snippet in it => we queue one none indentation statement
			else:
				keyword = random.choice(self.non_indentation_statements)
				keywords_sequence.append(keyword)
		
		# Choose if we unindent. This can happen only if we are at some indentation level > 0 and there is at least one code line
		# in the current block, with higher probability of unindenting the higher nb_lines_in_block
		elif len(self.context_stack) > 1 and random.random() > (1/(1+self.UNINDENTATION_SPEED)) ** self.context_stack.top().GID['nb_lines_in_block']:
			
			# I		if self.keyword == we are currently in a while loop and the update statement hasn't been generated yet
			if self.context_stack.top().GID['while_state']:
				keywords_sequence.append("WHILE_UPDATE")
			
			# Queue the UNINDENT action
			keywords_sequence.append('UNINDENT')
		
		# __In othe		if self.keyword ==s__
		else:
			
			# We set the potential keywords
			potential_keywords = list(self.pattern_vocabulary)

			# Remove the print statements in the middle of the code
			potential_keywords.remove('DISPLAY')
			
			# If we are in a while loop we remove the possibility to have an inner while loop
			if self.context_stack.top().GID['current_block'] == 'WHILE_BLOCK':
				potential_keywords.remove('WHILE_LOOP')
			
			# If we are in an if block we remove all inner if statements and while loops
			if self.context_stack.top().GID['current_block'] == 'IF_BLOCK':
				potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in self.indentation_statements]
			
			# Check for while_state
			if self.context_stack.top().GID['while_state']:
				potential_keywords.append('WHILE_UPDATE')
			
			# I		if self.keyword == we achieved MAX_DEPTH or MAX_SUB_BLOCKS inside the current context we remove the indentation statements
			# remove the indentation_statements from potential_keywords
			if len(self.context_stack) - 1 >=  self.MAX_DEPTH or self.context_stack.top().GID["nb_blocks"] >= self.MAX_SUB_BLOCKS:
				potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in self.indentation_statements]
			
			# Else If we are not in an If statement we remove the elif + else
			elif not self.context_stack.top().GID["if_state"]:
				potential_keywords = [potential_keyword for potential_keyword in potential_keywords if potential_keyword not in {"SIMPLE_ELIF_STATEMENT", "ELSE_STATEMENT"}]

			# We add the END keyword if we are at indentation level 0 and the self.line_counter is above MIN_LENGTH
			if len(self.context_stack) == 1 and self.line_counter > self.MIN_LENGTH:
				potential_keywords.append("END")

			# We choose a keyword randomly and queue it
			pk_weight = {
				'INITIALIZATION': 1,
				'SIMPLE_ASSIGNMENT': 1,
				'DIRECT_ASSIGNMENT': 1,
				'SIMPLE_IF_STATEMENT': 1,
				'SIMPLE_ELIF_STATEMENT': 1,
				'ELSE_STATEMENT': 1,
				'WHILE_UPDATE': 1,
				'WHILE_LOOP': 1,
				'DISPLAY': 1,
				'END': 1,
			}
			potential_keywords = [(potential_keyword,pk_weight[potential_keyword]) for potential_keyword in potential_keywords]
			random.shuffle(potential_keywords)
			keyword = random.choices([tup[0] for tup in potential_keywords], [tup[1] for tup in potential_keywords])[0]
			
			# If we end the code snippet, add a DISPLAY statement right before ending
			if keyword == 'END':
				keywords_sequence.append('DISPLAY')
			
			keywords_sequence.append(keyword)

		#=========================================
		#-----------------------------------------
		#=========================================
		self.context_stack.top().enqueue(keywords_sequence)


	def generate_code_snippet(self):
		self.code_snippet = ""
		self.keyword = "[START]"
		self.context_stack = ContextStack()

		#|=========================================================================|
		#|[*] [// User-defined global dynamic variables for snippet generation //] |
		#|=========================================================================|
		self.context_stack.top().GID = {
			'current_block': 'ROOT',
			'nb_if_blocks': 0,
			'nb_while_loops': 0,
			'nb_for_loops': 0,
			'nb_blocks': 0,
			'if_state': False,
			'while_state': None, # If None, indicates that either we are not at a while loop level, or we are but the update expression of the while loop has already been generated. If not None, the value must be the update expression of the while loop
			'readable_variables': list(),
			'writable_variables': list(self.VARIABLES),
			'nb_lines_in_block': 0,
		}

		self.all_assigned_variables = list()

		self.MIN_INIT = random.randint(0, self.MIN_INIT_MAX)

		self.line_counter = 1

		#=========================================
		#-----------------------------------------
		#=========================================
		while self.keyword != "END":
			self.construct_skeleton()
			self.instantiate_code()

		return self.code_snippet
	

	def create_corpus(self):

		#|===========================================================|
		#|[*] // User-defined global algorithm for corpus creation //|
		#|===========================================================|
		
		# GENERAL PARAMETERS
		self.MIN_INIT_MAX = 3
		self.MAX_DEPTH 	= 2
		self.MAX_SUB_BLOCKS = 2
		self.MIN_LENGTH = 5
		self.MAX_LENGTH = 15
		self.UNINDENTATION_SPEED = 0.09	# if <= 0, will never unindent after the first indentation encountered

		# PRINT PARAMETERS
		self.FORCE_PRINT = True
		self.PRINT_WEIGHTS_CONTROL_COEFFICIENT = 0

		# READ AND WRITE SECURITIES
		self.READ_SECURITY = False
		self.WRITE_SECURITY = True
		self.TIMEOUT = 3

		# WHILE LOOP PARAMETERS

		self.ALWAYS_EXECUTE_WHILE_LOOP = False

		# WHILE_LOOP GENERAL PARAMETERS
		self.WHILE_LOOP_UPDATE_OPERATORS = ['+', '-', '//', '*']
		self.WHILE_LOOP_RELATIONAL_OPERATORS = ['>', '<', '>=', '<=']
		self.NB_MAX_WHILE_LOOP_PROLOGUE_INTERMEDIATE_EXPRESSIONS = 2
		self.NB_MAX_WHILE_LOOP_UPDATE_INTERMEDIATE_EXPRESSIONS = 2

		# WHILE LOOP PARAMETERS FOR ADD UPDATE OPERATOR
		self.WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(999+1)]
		self.WHILE_LOOP_ADD_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(999+1)]
		self.WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(1, 20+1)]
		self.WHILE_LOOP_ADD_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 1 else 4 for i in range(1, 20+1)]
		self.WHILE_LOOP_ADD_UO_NB_ITERS									= [i for i in range(1, 10+1)]

		# WHILE LOOP PARAMETERS FOR SUB UPDATE OPERATOR
		self.WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(999+1)]
		self.WHILE_LOOP_SUB_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(999+1)]
		self.WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(1, 20+1)]
		self.WHILE_LOOP_SUB_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 1 else 4 for i in range(1, 20+1)]
		self.WHILE_LOOP_SUB_UO_NB_ITERS									= [i for i in range(1, 10+1)]

		# WHILE LOOP PARAMETERS FOR DIV UPDATE OPERATOR
		self.WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(1, 10+1)]
		self.WHILE_LOOP_DIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS 	= [1 for i in range(1, 10+1)]
		self.WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(2, 10+1)]
		self.WHILE_LOOP_DIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 10 else 3 for i in range(2, 10+1)]
		self.WHILE_LOOP_DIV_UO_NB_ITERS									= [i for i in range(1, 10+1)]
		self.WHILE_LOOP_DIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT		= 0.1

		# WHILE LOOP PARAMETER FOR FLOORDIV UPDATE OPERATOR
		self.WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES 		= [i for i in range(1, 999+1)]
		self.WHILE_LOOP_FDIV_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS = [1 for i in range(1, 999+1)]
		self.WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(2, 20+1)]
		self.WHILE_LOOP_FDIV_UO_UPDATE_OPERAND_VALUES_WEIGHTS 			= [1 if i!= 10 else 4 for i in range(2, 20+1)]
		self.WHILE_LOOP_FDIV_UO_NB_ITERS								= [i for i in range(1, 10+1)]
		self.WHILE_LOOP_FDIV_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT	= 0.1

		# WHILE LOOP PARAMETERS FOR MUL UPDATE OPERATOR
		self.WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES 			= [i for i in range(1, 999+1)]
		self.WHILE_LOOP_MUL_UO_CONTROL_VARIABLE_INITIAL_VALUES_WEIGHTS	= [1 for i in range(1, 999+1)]
		self.WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES 					= [i for i in range(2, 20+1)]
		self.WHILE_LOOP_MUL_UO_UPDATE_OPERAND_VALUES_WEIGHTS			= [1 if i!= 10 else 4 for i in range(2, 20+1)]
		self.WHILE_LOOP_MUL_UO_NB_ITERS									= [i for i in range(1, 5+1)]
		self.WHILE_LOOP_MUL_UO_NB_ITERS_WEIGHTS_CONTROL_COEFFICIENT		= 0.01

		self.VARIABLES						= ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]
		self.DIGIT 							= [i for i in range(999+1)]
		self.ARITHMETIC_OPERATORS 			= ["+", "-", "*", "//", "%"]
		self.ARITHMETIC_OPERATORS_WEIGHTS 	= [1,1,1,1,1]
		self.RELATIONAL_OPERATORS 			= ["<", ">", "<=", ">=", "!=", "=="]

		self.pattern_vocabulary = [
			"INITIALIZATION",
			'DIRECT_ASSIGNMENT',
			"SIMPLE_ASSIGNMENT",
			"SIMPLE_IF_STATEMENT",
			"SIMPLE_ELIF_STATEMENT",
			"ELSE_STATEMENT",
			'WHILE_LOOP',
			"DISPLAY",
		]

		self.loop_statements = [
			'WHILE_LOOP',
		]

		self.conditional_statements = [
			"SIMPLE_IF_STATEMENT",
			"SIMPLE_ELIF_STATEMENT",
		]

		self.indentation_statements = [
			'WHILE_LOOP',
			"SIMPLE_IF_STATEMENT",
			"SIMPLE_ELIF_STATEMENT",
			"ELSE_STATEMENT"
		]

		self.non_indentation_statements = [stm for stm in self.pattern_vocabulary if stm not in self.indentation_statements]

		self.variable_creation_statements = [
			"INITIALIZATION",
			'DIRECT_ASSIGNMENT',
			"SIMPLE_ASSIGNMENT",
			'WHILE_LOOP',
		]

		# with open("/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-28/frcg-random-states/random_state_2025-01-19_14-40-49.bin", "rb") as f:
		# 	random_state = pickle.load(f)
		# 	random.setstate(random_state)
		random.seed(40)
		nb_programs = 10
		nb_generated_programs = 0
		output_file = open("./data-ds-50/data.txt", "w")
		nb_zero_divisions = 0
		nb_var_value_overflows = 0
		nb_name_errors = 0
		nb_timeouts = 0
		max_var_value = 999
		pbar = tqdm.tqdm(total=nb_programs)
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

		# Custom exception, raised when a variable's absolute value gets higher than max_val_value
		class VariableValueOverflowError(Exception):
			def __init__(self, message):
				super().__init__(message)
		
		# Define the TiemoutException to be raised at timeouts
		class TimeoutException(Exception):
			pass

		if not self.WRITE_SECURITY:
			# Define the function called by the signal, it will raise TimeoutException
			def timeout_handler(signum, frame):
				raise TimeoutException()

			# Set the signal SIGALARM to call timeout_handler
			signal.signal(signal.SIGALRM, timeout_handler)
			
		# Set the starting_time and the first checkpoint_time (for logging)
		start_time = datetime.datetime.now()
		checkpoint_time = start_time
		f_log_file = open("log.txt", "w")
		log_interval = 100

		hashes = set()
		nb_deduplication_trials = 0
		max_deduplication_trials = 100
		deduplicate = True


		while nb_generated_programs < nb_programs:

			if nb_generated_programs % log_interval == 0:
				now = datetime.datetime.now()
				f_log_file.write(f"Generated {nb_generated_programs:<{len(str(nb_programs))}} programs,  absolute time: {now - start_time},  relative time: {now - checkpoint_time}\n")
				f_log_file.flush()
				checkpoint_time = now
		
			code_snippet = self.generate_code_snippet()
			code_snippet = code_snippet.strip("\n")
			
			# output_file.write(code_snippet+"\n\n")
			# output_file.flush()
			# nb_generated_programs += 1
			# continue
			# import sys; sys.exit(0)

			if deduplicate:
				code_hash = hashlib.sha256(code_snippet.encode('utf-8')).hexdigest()
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
			indented = "\n".join([f"	{line}" for line in code_snippet.split("\n")])
			func = "def func():\n" + indented
			exec_env = func + exec_env_boilerplate

			# Try to execute the generated code_snippet
			sio = StringIO()
			try:
				with redirect_stdout(sio):

					# Check if we are not using WRITE_SECURITY
					if not self.WRITE_SECURITY:
						signal.alarm(self.TIMEOUT)
					
					# Execute the code_snippet in a controlled environment
					exec(exec_env, {
						"VariableValueOverflowError" : VariableValueOverflowError
					})
				
				# __If we are here, it means that the code_snippet has been executed successfully__

				# Resetting the alarm to 0 if not using the WRITE_SECURITY
				if not self.WRITE_SECURITY:
					signal.alarm(0)
				
				# Getting the output
				output = sio.getvalue()

				# If we are force printing and the sio output is empty
				if self.FORCE_PRINT and output == '':
					
					# Select the readable_variables to choose from
					readable_variables = self.context_stack.top().GID['readable_variables']
					if not self.READ_SECURITY:
						readable_variables = self.all_assigned_variables
					
					# Choose the printed_term
					if len(readable_variables) != 0:
						printed_term = random.choices(population = (random.choice(readable_variables), random.choice(self.DIGIT)), weights = (0.95, 0.05), k = 1)[0]
					else:
						printed_term = random.choice(self.DIGIT)
					
					# Adding a print at the end of the code_snippet
					final_print_expression = f'print({printed_term})'
					code_snippet += f'\n{final_print_expression}'
					
					# Re-Create the execution environment
					func += f'\n\t{final_print_expression}'
					exec_env = func + exec_env_boilerplate
					sio = StringIO()
					with redirect_stdout(sio):
						exec(exec_env, {
							"VariableValueOverflowError" : VariableValueOverflowError
						})
					output = sio.getvalue()
				
				# Create the final results = code_snippet + formatted output
				result = "# code\n" + code_snippet + "\n# output\n# " + "\n# ".join(output.split("\n")[:-1])
				
				# Write the result to the destination file
				output_file.write(result + "\n\n")

				# Update the number of generated programs
				nb_generated_programs += 1

				pbar.update(1) 

			except ZeroDivisionError:
				nb_zero_divisions += 1
			except VariableValueOverflowError as e:
				nb_var_value_overflows += 1
			except NameError as e:
				nb_name_errors += 1
			except TimeoutException:
				nb_timeouts += 1
			except (Exception, KeyboardInterrupt) as e:
				print(f'Code Snippet Execution Error at {nb_generated_programs}:', e)
				with open('error_code.txt', 'w') as f:
					f.write(f'PROGRAM PROBLEM#{nb_generated_programs}\n'+exec_env)
				break
			
			pbar.set_description(f"ZeroDiv: {nb_zero_divisions:,} |Overflows: {nb_var_value_overflows:,} |NameErrors: {nb_name_errors:,} |Timeouts: {nb_timeouts:,}")
		
		output_file.close()
		f_log_file.close()


if __name__ == "__main__":
	tpg2 = TinyPyGenerator2()
	tpg2.create_corpus()