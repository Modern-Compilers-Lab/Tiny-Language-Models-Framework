import random
import re
from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout


cfg_rules = {
    # Variables and digits
    "VARIABLE": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ],
    "DIGIT": [str(i) for i in range(256)],

    # Operators
    "ARITHMETIC_OPERATOR": ["+", "-", "*", "/"],
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


re_pattern_line_parser = re.compile("(\t*)("+pattern_vocab_for_regex+")(:[^,=]+=[^,=]+(?:,[^,=]+=[^,=]+)*$|$)")
re_general_line_finder = re.compile(".+(?:\n|$)")
re_while_identifier = re.compile(".*\nwhile ([a-z])")

def pattern_checker(pattern):

	# We initialize the printable_identifiers_stack with the level 0 set
	# Each level of the stack is a set containing the available identifiers for printing
	printable_identifiers_stack = [set()]

	# We split the pattern into its lines
	pattern_lines = pattern.strip().split("\n")
	
	# Loop through the pattern lines and generate their codes
	to_indent = False
	for line_number, pattern_line in enumerate(pattern_lines):
		line_number += 1
		# We parse the pattern line with the corresponding regex
		pattern_line_match = re_pattern_line_parser.match(pattern_line)
		
		# We check if the pattern line is syntactically correct
		if pattern_line_match == None:
			raise Exception(f"check_pattern exception: syntax error at line {line_number}")
		
		# We get the indent level of the pattern line to compare it with the current indent level (represented by the length of the stack - 1)
		new_indent_level = len(pattern_line_match.group(1))
		current_indent_level = len(printable_identifiers_stack)-1

		# The new_indent_level must always be <= current_indent_level and == in case of a to_indent
		if (new_indent_level > current_indent_level) or (to_indent and new_indent_level != current_indent_level):
			raise Exception(f"check_pattern exception: indentation error at line {line_number}")
		
		# In case it is a un-indentation, we pop the stacks as many times as necessary
		for _ in range(current_indent_level - new_indent_level):
			printable_identifiers_stack.pop()
		
		# We get the pattern line keyword
		keyword = pattern_line_match.group(2)
		
		# We get the params_dict of this pattern line
		if pattern_line_match.group(3):
			pattern_line_params = pattern_line_match.group(3)[1:].split(",")
			params_dict = dict()
			for param in pattern_line_params:
				param_id, param_value = param.split("=")
				param_id = param_id.strip().lower()
				param_value = param_value.strip().lower()
				params_dict[param_id] = param_value
			
			if keyword in variable_creation_statements and "id" in params_dict:
				if params_dict["id"] in printable_identifiers_stack[-1]:
					raise Exception(f"check_pattern exception: deduplicated id {params_dict['id']} in the same indent context at line {line_number}")
				printable_identifiers_stack[-1].add(params_dict["id"])
			
			elif keyword  == "DISPLAY" and "tar" in params_dict:
				if params_dict["tar"] not in printable_identifiers_stack[-1]:
					raise Exception(f"check_pattern exception: display of unavailable id for the current indent context at line {line_number}")
		
		to_indent = False
		# If the keyword creates an indentation, we stack a new indentation context
		if keyword in indentation_statements:
			to_indent = True
			printable_identifiers_stack.append(set(printable_identifiers_stack[-1]))


# random_state_save = random.getstate()
def develop_pattern(pattern):
	# Split the pattern into its lines
	pattern_lines = pattern.strip().split("\n")
	# random.setstate(random_state_save)
	init_count = 0
	code = ""

	# I should check what to do with these two ...
	last_variable = set()
	for_init_step = dict()

	# We initialize the readable_identifiers_stack with the level 0 dict
	# Each level of the stack is a dictionary containing the readable identifiers and their values
	readable_identifiers_stack = [dict()]

	# This is the stack for writable identifiers (i.e. those that haven't been taken by the while loops)
	writable_identifiers_stack = [cfg_rules["VARIABLE"]]
	# This is the stack that maps the generated python identifiers to the pypat identifiers
	id_identifier_map_stack = [dict()]

	# Loop through the pattern lines and generate their codes
	for pattern_line in pattern_lines:
		
		# We parse the pattern line
		pattern_line_match = re_pattern_line_parser.match(pattern_line)
		new_indent_level = len(pattern_line_match.group(1))
		current_indent_level = len(readable_identifiers_stack)-1

		# In case it is a un-indentation, we pop the stacks as many times as necessary
		for _ in range(current_indent_level - new_indent_level):
			readable_identifiers_stack.pop()
			writable_identifiers_stack.pop()
		
		# We set the indentation context for the code line
		readable_identifiers = readable_identifiers_stack[-1]
		cfg_rules["VARIABLE"] = writable_identifiers_stack[-1]
		
		# We get the keyword
		keyword = pattern_line_match.group(2)

		# We generate the code line
		keyword_gen_code = generate_code(keyword, readable_identifiers, last_variable, for_init_step).replace("SPACE", " ")
		
		# We get the params_dict of this pattern line
		if pattern_line_match.group(3):
			pattern_line_params = pattern_line_match.group(3)[1:].split(",")
			params_dict = dict()
			for param in pattern_line_params:
				param_id, param_value = param.split("=")
				param_id = param_id.strip().lower()
				param_value = param_value.strip().lower()
				params_dict[param_id] = param_value
			
			if keyword in variable_creation_statements and "id" in params_dict:
				if keyword == "FOR_HEADER": # the control variable is in the 4th position : for i...
					id_identifier_map_stack[-1][params_dict["id"]] = keyword_gen_code[4]
				else: # i.e. it's INITIALIZATION, WHILE_LOOPs, or _ASSIGNMENTS, the control is in the first position: i = ...
					id_identifier_map_stack[-1][params_dict["id"]] = keyword_gen_code[0]
				
			elif keyword  == "DISPLAY" and "tar" in params_dict:
				# The displayed variable is in the 6th position: print(i)
				keyword_gen_code = keyword_gen_code[:6] + id_identifier_map_stack[-1][params_dict["tar"]] + keyword_gen_code[7:]
		
		# If the keyword creates an indentation, we stack a new indentation context
		if keyword in indentation_statements:
			readable_identifiers_stack.append(dict(readable_identifiers_stack[-1]))
			id_identifier_map_stack.append(dict(id_identifier_map_stack[-1]))

			# Even though we should only stack when necessary i.e. when a while loop is about to be created, let's just do it this way now for simplicity
			writable_identifiers_stack.append(list(writable_identifiers_stack[-1]))
		
			# If it is a while loop, we eliminate the control variable from the writable identifiers from the new indentation level
			if keyword in ("WHILE_LOOP_LESS", "WHILE_LOOP_GREATER"):
				while_identifier = re_while_identifier.match(keyword_gen_code).group(1)
				writable_identifiers_stack[-1].remove(while_identifier)
		
		# We create the code_line(s) corresponding to this keyword
		code_line = pattern_line_match.group(1) + pattern_line_match.group(1).join(re_general_line_finder.findall(keyword_gen_code))
		
		# We concatenate it to the rest of the code
		code += code_line
	try:
		exec(code,{})
	except ZeroDivisionError:
		pass
	# file.write(code+"\n\n")
	print("===================")
	print(code)
	# file.close()

max_init = 3

pattern = """
INITIALIZATION: id=2
INITIALIZATION: id=1
INITIALIZATION
DISPLAY: tar=1
SIMPLE_IF_STATEMENT
	SIMPLE_ASSIGNMENT: id=3
	WHILE_LOOP_GREATER: id=4
		ADVANCED_ASSIGNMENT
		DISPLAY: tar=4
SIMPLE_ELIF_STATEMENT
	SIMPLE_IF_STATEMENT
		SIMPLE_ASSIGNMENT:id=kkk
		DISPLAY: tar=kkk
	ELSE_STATEMENT
		FOR_HEADER: id=3
			DISPLAY: tar=3
"""

pattern_checker(pattern)
develop_pattern(pattern)