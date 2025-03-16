from anytree import Node, RenderTree
import random
from io import StringIO
from contextlib import redirect_stdout
import argparse  
import time
from tqdm.auto import tqdm
import hashlib
import os
import psutil


class CodeGenerator:
    def __init__(self):
        """
        Initialize the CodeGenerator object with the given context-free grammar rules.

        """
        
        self.init_count = 0
        self.max_init = 0
        
        # Dictionary containing context-free grammar rules.
        self.cfg_rules = {
                # Variables and digits
                "VARIABLE": ["a", "b", "c", "d", "e"],
                "DIGIT": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],

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

                # Terms and expressions
                "TERM": ["EXPRESSION_IDENTIFIER", "DIGIT"],
                "EXPRESSION": ["TERM SPACE OPERATOR SPACE TERM"],
                "ENCLOSED_EXPRESSION": ["BRACKET_OPEN EXPRESSION BRACKET_CLOSE"],
                "DISPLAY_EXPRESSION": ["EXPRESSION_IDENTIFIER SPACE OPERATOR SPACE EXPRESSION_IDENTIFIER" ,
                                        "EXPRESSION_IDENTIFIER SPACE OPERATOR SPACE DIGIT"],

                # Initializations and assignments
                "IDENTIFIER_INITIALIZATION": ["IDENTIFIER_INITIALIZATION INITIALIZATION", 
                                              "INITIALIZATION"],

                "INITIALIZATION": ["VARIABLE SPACE EQUALS SPACE DIGIT NEW_LINE"],

                "SIMPLE_ASSIGNMENTS": ["VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE" , ""],
                "ADVANCED_ASSIGNMENTS": ["VARIABLE SPACE EQUALS SPACE SIMPLE_ARITHMETIC_EVALUATION NEW_LINE", 
                                         "VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE" , 
                                         ""],

                "SIMPLE_ARITHMETIC_EVALUATION": ["SIMPLE_ARITHMETIC_EVALUATION ARITHMETIC_OPERATOR ENCLOSED_EXPRESSION", 
                                                 "ENCLOSED_EXPRESSION",
                                                ], 

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

                # Loops
                "FOR_HEADER": ["FOR SPACE EXPRESSION_IDENTIFIER SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL COMMA SPACE STEP BRACKET_CLOSE SPACE COLON", 
                               "FOR SPACE EXPRESSION_IDENTIFIER SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL BRACKET_CLOSE SPACE COLON"],
                "INITIAL": ["DIGIT"],
                "FOR_LOOP": ["FOR_HEADER NEW_LINE TAB_INDENT DISPLAY"],
                "ADVANCED_FOR_LOOP": ["FOR_LOOP",
                                      "FOR_HEADER NEW_LINE TAB_INDENT ADVANCED_DISPLAY"],


                # Displaying 
                "DISPLAY" : ["PRINT BRACKET_OPEN DISPLAY_IDENTIFIER BRACKET_CLOSE"],
                "ADVANCED_DISPLAY" : ["DISPLAY",
                                      "PRINT BRACKET_OPEN DISPLAY_EXPRESSION BRACKET_CLOSE"],


                "LEVEL1.1": ["IDENTIFIER_INITIALIZATION SIMPLE_ASSIGNMENTS ADVANCED_DISPLAY"],
                "LEVEL1.2": ["IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_DISPLAY"],
                "LEVEL2.1": ["IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY", 
                            "IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY NEW_LINE SIMPLE_ELIF_STATEMENT TAB_INDENT DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT DISPLAY", 
                            "IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT DISPLAY"],
                "LEVEL2.2": ["IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY", 
                            "IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ADVANCED_ELIF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT ADVANCED_DISPLAY", 
                            "IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT ADVANCED_DISPLAY"],
                "LEVEL3.1": ["IDENTIFIER_INITIALIZATION FOR_LOOP"],
                "LEVEL3.2": ["IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_FOR_LOOP"],
            
                "ALL": ["LEVEL1.1", "LEVEL1.2", "LEVEL2.1", "LEVEL2.2", "LEVEL3.1", "LEVEL3.2"],
            
                    }
        


    def generate_code(self, symbol, assigned_identifiers, last_variable, for_init_step, parent=None ):
        """
        Generate code recursively based on the context-free grammar rules.

        Parameters:
        - symbol (str): The symbol to generate code for.
        - assigned_identifiers (set): Set of assigned identifiers.
        - last_variable (set): Set of the last used variables.
        - parent (Node): Parent node in the syntax tree.

        Returns:
        - str: The generated code.
        """
        node = Node(symbol, parent=parent)

        if symbol in self.cfg_rules:
            if symbol == "IDENTIFIER_INITIALIZATION":
                if self.init_count < self.max_init:
                    self.init_count += 1
                else:
                    symbol = "INITIALIZATION"

            rule = random.choice(self.cfg_rules[symbol])
            symbols = rule.split(" ")

            generated_symbols = [self.generate_code(s, assigned_identifiers, last_variable, for_init_step, node) for s in symbols]

            if symbol == "INITIAL":
                init = generated_symbols[0]
                for_init_step["initial_value"] = init
            

            if symbol == "INITIALIZATION":
                assigned_identifiers.add(generated_symbols[0])

            if (symbol == "SIMPLE_ASSIGNMENTS") or (symbol == "ADVANCED_ASSIGNMENTS"):
                if generated_symbols[0]:
                    last_variable.add(generated_symbols[0])

            return ''.join(generated_symbols)

        elif symbol == "FINAL":
            
            initial_value = for_init_step.get("initial_value", "0")
        
            # Generate valid step_value and execution_count
            valid_values = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            step_value, execution_count = random.choice(valid_values)
            for_init_step["step"] = str(step_value)
            
            final_value = step_value * execution_count + int(initial_value) - 1
            return str(final_value)

            
                
        elif symbol == "STEP":
            
            return for_init_step.get("step", "0")
            
        elif symbol == "EXPRESSION_IDENTIFIER":
            identifier = random.choice(tuple(assigned_identifiers)) if assigned_identifiers else random.choice(self.cfg_rules["DIGIT"])
            return identifier

        elif symbol == "DISPLAY_IDENTIFIER":
            try:
                return f"{tuple(last_variable)[0]}"
            except:
                return f"{random.choice(tuple(assigned_identifiers))}"
        else:
            return symbol

    

    def print_tree(self, root):
        """
        Print the syntax tree using the RenderTree utility from the anytree module.

        Parameters:
        - root (Node): The root node of the syntax tree.
        """
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")

    def generate_program(self, level):
        """
        Generate a program based on the specified level.

        Parameters:
        - level (str): The level of the program.

        Returns:
        - Tuple[Node, str]: The syntax tree root node and the generated program.
        """
        assigned = set()
        last_variable = set()
        for_init_step = {}
        root = Node("ROOT")

        self.init_count = 0
        if level == "1.1":
            self.max_init = 2
        elif level == "1.2":
            self.max_init = 3
        elif level == "2.1":
            self.max_init = 2
        elif level == "3.1":
            self.max_init = 2
        elif level == "3.2":
            self.max_init = 4
        else:
            self.max_init = 5
            
        if level == "ALL" :
            level_passed = level
        else :
            level_passed = "LEVEL" + level
            
        program = self.generate_code(level_passed, assigned, last_variable, for_init_step, root)
        
        return root, program.replace("SPACE", " ")
    
    def memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    def generate_and_write_programs(self, num_programs, level, filename='data.txt', deduplicate=True):
        """
        Generate and write a specified number of programs to a file.

        Parameters:
        - num_programs (int): Number of programs to generate and write.
        - level (str): The level of the programs.
        - filename (str): Name of the file to write the programs (default is 'data.txt').
        - deduplicate (bool, optional): Whether to perform deduplication of generated programs (default is True).
        """
        start_time = time.time()  
        start_mem = self.memory_usage()
        max_tries = 1000
        num_tries = 0
        
        with open(filename, 'w') as file:
            
            generated_programs = 0
            hashes = set() 
            pbar = tqdm(desc="Generation", total=num_programs)
            while generated_programs < num_programs:
                try:
                    root, program = self.generate_program(level)
                    code = program + "\n# output"

                    SIO = StringIO()
                    with redirect_stdout(SIO):
                        exec(code)
                    output = SIO.getvalue().strip()

                    output = '\n'.join([f'# {line}' if line else f'# ' for line in output.split('\n')])
                    result = f"""{code}\n{output}"""
                    
                    program_hash = hashlib.sha256(result.encode('utf-8')).hexdigest()

                    if deduplicate:
                        if program_hash not in hashes:
                            hashes.add(program_hash)
                            file.write(result + '\n\n')
                            generated_programs += 1  
                            pbar.update(1)
                            num_tries = 0
                        else:
                            num_tries += 1
                            if num_tries >= max_tries:
                                print("Hit max tries in deduplication, stopping generation.")
                                break
                    else:
                        file.write(result + '\n\n')
                        
                        generated_programs += 1  
                        pbar.update(1)

                except Exception as e:
                    continue

        pbar.close()
        end_time = time.time()
        end_mem = self.memory_usage()
        deduplication_info = "with deduplication" if deduplicate else "without deduplication"
        print(f"Code generation completed in {end_time - start_time:.2f} seconds.")
        print(f"Memory used during code generation: {end_mem - start_mem} bytes")
        print(f"Generated {generated_programs} {'unique ' if deduplicate else ''}programs {deduplication_info}.")
        print(f"Programs are saved to {filename}.")


def main():
    parser = argparse.ArgumentParser(description='Generate and write programs based on a specified level. ')
    parser.add_argument('--num_programs', type=int, default=1000, help='Number of programs to generate and write (default is 1000)')
    parser.add_argument('--level', default="ALL", help='The level of the programs (1.1, 1.2, 2.1, 2.2, 3.1, 3.2, ALL)')
    parser.add_argument('--filename', default='data.txt', help='Name of the file to write the programs (default is data.txt)')
    parser.add_argument('--deduplicate', action='store_true', default=True, help='Perform deduplication of generated programs (default is True)a')

    args = parser.parse_args()

    code_generator = CodeGenerator()
    code_generator.generate_and_write_programs(num_programs=args.num_programs, level=args.level, filename=args.filename,  deduplicate=args.deduplicate)

if __name__ == "__main__":
    main()
