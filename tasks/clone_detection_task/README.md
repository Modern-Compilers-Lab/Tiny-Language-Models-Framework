## Task Overview

This task tests the ability of the tiny-lm on detecting code clones and non-clones, so first it simplifies the code snippets by removing variable initializations and replacing variables in print statements with their corresponding formulas. The simplified code is then executed 150 time and analyzed to identify code clones, which are later used to form pairs of clone and non-clone examples for further usage (FineTuning the Tiny-LM).

## How It Works

1. **Code Generation**:
   - The project generates code snippets that contain variable initializations, logic, and print statements.
   - The initial code looks like:
     ```python
     b = 6
     g = 4
     n = 3
     o = (g + 1) - (n * g)
     if (not b < 4):
         print(o)
     else:
         print(b * n)
     ```

2. **Simplification**:
   - Using the `simplify_code_levelx.py` script (where `x` refers to the specific level), variable initializations are removed, and the print statements are modified to include the variable's formula instead of the variable itself.
   - The simplified version of the above code would be:
     ```python
     if not b < 4:
         print(g + 1 - n * g)
     else:
         print(b * n)
     ```

3. **Execution**:
   - Each simplified code is executed 150 times with random variable initializations using the `code_execution.py` file.
   - The output of each execution is stored in the `outputs.json` file.

4. **Clone Identification**:
   - Based on the generated outputs, the `identify_clones.py` script is used to detect clone and non-clone code snippets. The clone pairs are stored in the `clones.json` file.
   
5. **Pair Formation**:
   - Each code snippet is paired with other snippets to form clone and non-clone pairs. Each snippet is repeated 4 times—2 times as a clone and 2 times as a non-clone example.

## Files

- `simplify_code_levelx.py`: Script for simplifying code at different levels by removing variable initializations.
- `code_execution.py`: Executes simplified code snippets and stores the results.
- `identify_clones.py`: Identifies clones based on execution results and stores them in `clones.json`.
- `outputs.json`: Stores the outputs from executing the code snippets.
- `clones.json`: Stores the identified clones for further processing.

## Usage

1. **Generating Code**:
   Run the `automate.py` script to generate code snippets.
   ```bash
   python ./tasks/clone_detection_task/automate.py --num_programs <number_of_programs>