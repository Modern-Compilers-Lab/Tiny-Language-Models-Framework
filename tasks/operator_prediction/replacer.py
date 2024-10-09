import random
import re
import signal
from io import StringIO
from contextlib import redirect_stdout
import os
import pickle
import numpy as np
from tqdm.auto import tqdm
import argparse

def run_code(code):
        SIO = StringIO()
        with redirect_stdout(SIO):
            exec(code)
        return SIO.getvalue().strip()

def rand_operator(s):
    indices = [m.start() for m in re.finditer(r'[+\-/*]', s)]

    if len(indices) > 0:
        return random.choice(indices)
    
    return -1

def timeout_handler(signum, frame):
    raise Exception("Code execution exceeded the time limit")

def test_outputs(code, op_index, output, timeout=1e-1):
    operators = {'+', '-', '*', '/'} - {code[op_index]}
    
    for operator in operators:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)  # Timeout in seconds
            
            updated_code = code[:op_index] + operator + code[op_index+1:]
            new_output = run_code(updated_code)
            if new_output == output:
                return False
            
        except Exception:
            continue

        finally:
            # Disable the timer after execution
            signal.setitimer(signal.ITIMER_REAL, 0)
    
    return True

def update_code(code, op_index, output_pure):
    output = '\n'.join([f'# {line}' if line else f'# ' for line in output_pure.split('\n')])

    operator = code[op_index]
    code = f"""{code}\n# output\n{output}"""

    code = list(code)
    code[op_index] = '#'
    
    updated_code = "".join(code) + f"""\n# operator\n# {operator}"""
    return updated_code

def replace_operator(code, output):
    if(len(output) <= 1):
        return code, False
    
    op_index = rand_operator(code)
    if op_index < 0:
        return code, False

    isSucess = test_outputs(code, op_index, output)
    if isSucess:
        updated_code = update_code(code, op_index, output)
        return updated_code, True
    else:
        return code, False


parser = argparse.ArgumentParser(description='Create an operator prediction dataset from an output prediction one')
parser.add_argument('--input_file_name', default='input_data.txt', help='Name of input file')
parser.add_argument('--output_file_name', default='output_data.txt', help='Name of output file')

args = parser.parse_args()
input_file_path = args.input_file_name
output_file_path = args.output_file_name

input_file_path = os.path.join(os.path.dirname(__file__),  input_file_path)
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"Length of original dataset in characters: {len(data):,}")

examples = data.split("\n\n")[:-1]
print(f"Length of original dataset: {len(examples):,}\n")

new_examples = []
for i in tqdm(range(len(examples))):
    example = examples[i]
    code, output = example.split("\n# output\n")
    output = output.replace("# ", '')

    n_tries = 0
    while n_tries < 3:
        result, isSuccess = replace_operator(code, output)
        if isSuccess:
            break
        n_tries += 1
    
    if n_tries < 3:
        new_examples.append(result)

print(f"Length of new dataset: {len(new_examples):,}")
print(f"Lengths ratio: {100*len(new_examples)/len(examples):,}%\n") # Experimental ratio: ~73%

new_data = "\n\n".join(new_examples)
with open(os.path.join(os.path.dirname(__file__), output_file_path), 'w') as f:
    f.write(new_data)
print(f"Length of new dataset in characters: {len(new_data):,}\n")