import random
import io
import os
import contextlib
import json
from tqdm import tqdm
from used_variables_detector import get_printed_and_condition_variables
from comparison_check import has_diff_var_comparison
import argparse

def load_simplifications(file_path):
    simplifications = []
    with open(file_path, 'r') as file:
        lines = file.read()
    simplification = lines.split('# Simplification')[1:]
    for i in range(len(simplification)):
        simplifications.append(simplification[i].split('\n\n\n#')[0])
    return simplifications

def initialize_random_values(n):
    return [random.randint(0, n) for _ in range(150)]

def execute_code_with_random_initialization(snippet, variables, num, random_initializations):
    result_true = has_diff_var_comparison(snippet)
    if result_true:
        # Code block when there's a comparison between different variables

        variables = snippet_initialization_code(variables, random_initializations, num, mode='different')
        code = f"""{variables}
        {snippet}
        """
    
    else:
        # Code block when there's no comparison between different variables
        variables = snippet_initialization_code(variables, random_initializations, num, mode='same')
        code = f"""{variables}
        {snippet}
        """
    # Capture the output of the code
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code)
    except Exception as e:
        return str(e)

    return output.getvalue().strip()

def snippet_initialization_code(variables, random_initializations, num, mode='different'):
    variables_init = ""
    if mode == 'different':
        for i, var in enumerate(variables):

            variables_init += f"{var} = {random_initializations[i][num]}\n"
    else:
        for var in variables:
            variables_init += f"{var} = {random_initializations[0][num]}\n"
    return variables_init

def main():
    parser = argparse.ArgumentParser(description='Execute code snippets with random initializations and capture output.')
    parser.add_argument('--level', required=True, help='Specify the level (e.g: 1, 2, 3)')
    parser.add_argument('--dataset-file', required=True, help='Path to the dataset file to execute.')
    parser.add_argument('--output-file', required=True, help='Path to save the JSON output file.')
    
    args = parser.parse_args()

    simplifications = load_simplifications(args.dataset_file)
    printed_vars = {}
    for i, snippet in enumerate(simplifications):
        printed_vars[i] = get_printed_and_condition_variables(snippet)
    
    random.seed(42)

    random_initializations = [initialize_random_values(22) for _ in range(10)]

    outputs = {}
    # Check if the file exists
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            outputs = json.load(f)
    
    for i in tqdm(range(len(simplifications))):
        output = []
        for num in range(150): # Each code will be executed 150 time
            output.append(execute_code_with_random_initialization(simplifications[i], list(printed_vars[i]), num, random_initializations))
        outputs[i] = output
    

    with open(args.output_file, "w") as write_file:
        json.dump(outputs, write_file)

    print(f'The outputs of the simplified programs are stored in {args.output_file}')

if __name__ == "__main__":
    main()
