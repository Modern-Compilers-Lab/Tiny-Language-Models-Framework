import ast
import random
import string
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from math import gcd
from functools import reduce

class VariableRenamer:
    def __init__(self, p=0.5, mode="all"):
        self.max_string_length = 5
        self.arr = [[0, 0, 0, 0, 0], [315, 574, 53, 1, 676], [7920, 14939, 53, 1, 17576], [196862, 388429, 53, 1, 456976], [5566330, 10099169, 53, 1, 11881376]]
        self.cte = 123456789
        self.rarity = 6/10
        self.mode = mode
        self.variable_identifiers = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
                                        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                                        "u", "v", "w", "x", "y", "z"]
        self.reserved_words =  [
                        "False", "await", "else", "import", "pass",
                        "None", "break", "except", "in", "raise",
                        "True", "class", "finally", "is", "return",
                        "and", "continue", "for", "lambda", "try",
                        "as", "def", "from", "nonlocal", "while",
                        "assert", "del", "global", "not", "with",
                        "async", "elif", "if", "or", "yield"
                    ]
    def resetRand(self,index, arr, cte,worker_id=0):
        template = [[0, 0, 0, 0, 0], [315, 574, 53, 1, 676], [7920, 14939, 53, 1, 17576], [196862, 388429, 53, 1, 456976], [5566330, 10099169, 53, 1, 11881376]]
        arr[index][0] = template[index][0]
        arr[index][1] = template[index][1]

        x = arr[index][0]
        a = arr[index][2]
        c = arr[index][3]
        m = arr[index][4]
        for _ in range(worker_id):
            x = (a * x + c) % m
        arr[index][1] +=worker_id
        arr[index][0] = x

    def next_rand(self,index, arr, rarity = 1,step=1,cte=None):
        template = [[0, 0, 0, 0, 0], [315, 574, 53, 1, 676], [7920, 14939, 53, 1, 17576], [196862, 388429, 53, 1, 456976], [5566330, 10099169, 53, 1, 11881376]]
        x = arr[index][0]
        a = arr[index][2]
        c = arr[index][3]
        m = arr[index][4]
        for _ in range(step):
            x = (a * x + c) % m
        arr[index][1] +=step
        arr[index][0] = x
        if arr[index][1] >= m:
            self.resetRand(index,arr,cte,worker_id=0)
            return arr[index][0] 
        return x
    def number_to_varname(self,n):
        """
        Convert integer n to a variable name string (a-z), length 1-10.
        Numbers are mapped to variable names as base-26.
        """
        # Determine length of the variable name
        total = 0
        for length in range(1, 11):
            count = 26 ** length
            if n < total + count:
                n -= total
                break
            total += count
        else:
            raise ValueError("Number too large for 10-character variable names")

        # Convert n to base-26 digits
        digits = []
        for _ in range(length):
            digits.append(n % 26)
            n //= 26
        digits.reverse()

        # Map digits 0-25 → 'a'-'z'
        name = ''.join(chr(d + ord('a')) for d in digits)
        return name
    def int_to_string(self,x, length):
        s = []
        for _ in range(length):
            s.append(chr(ord('a') + x % 26))
            x //= 26
        return ''.join(reversed(s))
    def generate_string(self,reserved):
        index = random.randint(1, self.max_string_length-1) 
        string = self.int_to_string(self.next_rand(index,self.arr,self.rarity,step=1,cte=self.cte),index+1)
        while string in reserved :
            string = self.int_to_string(self.next_rand(index,self.arr,self.rarity,step=1,cte=self.cte),index+1)
        print(all((self.arr[i][1] > self.arr[i][4] * 0.55 and self.arr[i][1] < self.arr[i][4]) for i in range(2, 5)))
        return string
    
    def rename(self, tree):
        all_vars = set()
        var_positions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                all_vars.add(var_name)
                var_positions.setdefault(var_name, []).append(node)

        if not all_vars:
            return False

        if self.mode == "all":
            rename_candidates = list(all_vars)
        # elif self.mode == "one":
        #     rename_candidates = [random.choice(list(all_vars))]
        # else:  # "proba"
        #     rename_candidates = [v for v in all_vars if random.random() < self.p]

        used_names = all_vars.copy()
        var_map = {}
        for var in rename_candidates:
            new_name = self.generate_string(self.reserved_words)
            var_map[var] = new_name
            used_names.add(new_name)

        for old_name, new_name in var_map.items():
            for node in var_positions[old_name]:
                node.id = new_name

        return tree
    def prime_factors(self,n):
        factors = set()
        
        # factor out 2 first
        while n % 2 == 0:
            factors.add(2)
            n //= 2
        
        # check odd numbers from 3 upwards
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.add(i)
                n //= i
            i += 2
        
        # if remaining n > 1, it's prime
        if n > 1:
            factors.add(n)
        
        return sorted(factors)



    def lcm(self,x, y):
        return x * y // gcd(x, y)

# --- Transformation dispatcher ---

def transform_snippet_with_args(args):
    snippet, perturbation, perturbation_mode, proba = args
    try:
        tree = ast.parse(snippet)
    except Exception as e:
        return f"Could not parse snippet:\n{snippet}\nError: {e}\n", snippet

    if perturbation == 1:
        tree = VariableRenamer(p=proba,mode=perturbation_mode).rename(tree)
    elif perturbation == 2:
        tree = NeutralOpAdder(p=proba,mode=perturbation_mode).transform(tree)
        if tree is False:  # No eligible nodes found
            return False
    elif perturbation == 3:
        tree = ArithmeticCommutativityTransformer(p=proba,mode=perturbation_mode).transform(tree)
        if tree is False:  # No eligible nodes found
            return False
    elif perturbation == 4:
        tree = ComparisonSymmetryTransformer(p=proba,mode=perturbation_mode).transform(tree)
        if tree is False:  # No eligible nodes found
            return False
    elif perturbation == 5:
        tree = NeutralAssignmentInserter().insert_assignment(tree)
        if tree is False:  # No eligible nodes found
            return False
    elif perturbation == 6:
        tree = NeutralExpressionInserter().insert_expression(tree)
        if tree is False:  # No eligible nodes found
            return False
    else:
        return f"Unknown mode: {perturbation_mode}", snippet

    try:
        new_code = ast.unparse(tree)
    except Exception as e:
        new_code = f"Could not unparse transformed AST: {e}"
    return new_code

# --- Batch processor ---
def process_batch(batch_args):
    return [transform_snippet_with_args(args) for args in batch_args]

# --- Batching logic ---
def batch_snippets(snippets, batch_size):
    for i in range(0, len(snippets), batch_size):
        yield snippets[i:i+batch_size]


def main():
    perturbation_names = {
        1: "variable_renaming",
        2: "neutral_operator",
        3: "arithmetic_commutativity",
        4: "comparison_commutativity",
        5: "neutral_assignment",
        6: "neutral_expression"
    }

    perturbation_modes = {
        1: "all",
        2: "one",
        3: "proba"
    }

    print("Choose a transformation mode:")
    print(" 1 = Variable Renaming")
    print(" 2 = Neutral Operator (+/- 0)")
    print(" 3 = Arithmetic Commutativity")
    print(" 4 = Comparison Commutativity")
    print(" 5 = Neutral Assignment Insertion")
    print(" 6 = Neutral Expression Insertion")
    while True:
        try:
            perturbation = int(input("Enter perturbation number (1-6): ").strip())
            if perturbation not in range(1, 7):
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid number between 1 and 6.")
    if perturbation not in (5, 6):
        print("Choose a perturbation mode:")
        print(" 1 = All Variables")
        print(" 2 = One Variable")
        print(" 3 = Probability")
        while True:
            try:
                perturbation_mode_number = int(input("Enter perturbation mode (1-3): ").strip())
                if perturbation_mode_number not in range(1, 4):
                    raise ValueError
                perturbation_mode = perturbation_modes[perturbation_mode_number]
                break
            except ValueError:
                print("Please enter a valid number between 1 and 3.")

        # Perturbations 1-4 may require a probability if the mode is "proba"
        if perturbation_mode_number == 3:
            if perturbation in (1, 2, 3, 4):
                while True:
                    try:
                        proba_input = input("Enter probability 0~1 [Default is 1.0] :").strip()
                        if proba_input == "":
                            proba = 1.0
                            break
                        proba = float(proba_input)
                        if not (0.0 <= proba <= 1.0):
                            raise ValueError
                        break
                    except ValueError:
                        print("Please enter a valid probability between 0.0 and 1.0.")
        else:
            proba = 1.0  # Not used, but passed for signature
    else:
        proba = 1.0  # Not used, but passed for signature
        perturbation_mode = "one"



    def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
        percent = f"{100 * (iteration / float(total)):.1f}"
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        if iteration == total:
            print()

    # Loading snippets
    print('Reading the csv file ...')
    df = pd.read_csv("/data/yb2618/Tiny-Language-Models-Framework/research_project_4/robustness_evolution_study/evals/eval-5.1/infers/hard-match-successes.csv")
    total = len(df)
    snippets = []
    for i, input_prompt in enumerate(df["example_input"]):
        snippets.append(input_prompt.split("\n#STEP")[0])
        print_progress_bar(i + 1, total, prefix='Loading snippets', suffix='Done')

    print(f"Loaded {len(snippets)} snippets.\n")

    if not (perturbation in (3, 4) and perturbation_mode == "all"):
        duplication_factor = 128
        snippets = [snippet for snippet in snippets for _ in range(duplication_factor)]
        print(f"Duplicated each snippet {duplication_factor} times. Total snippets: {len(snippets)}\n")

    args_list = [(snippet, perturbation, perturbation_mode, proba) for snippet in snippets]
    batch_size = 128 * 6 
    batches = list(batch_snippets(args_list, batch_size))

    print(f"\nProcessing {len(snippets)} snippets in {len(batches)} batches...\n")

    results = []
    total_batches = len(batches)
    with ProcessPoolExecutor(max_workers=32) as executor:
        batch_results = executor.map(process_batch, batches)
        for idx, batch in enumerate(batch_results, 1):
            results.extend(batch)
            print_progress_bar(idx, total_batches, prefix='Processing', suffix='Done')


    output_path = f"data/{perturbation_names[perturbation]}_{perturbation_mode}_snippets.txt"
    with open(output_path, "w") as out_f:
        valid_results = [trans for trans in results if trans is not False]
        for trans in valid_results:
            out_f.write(trans + "\n\n")

    print(f"Transformed snippets written to {output_path}")


if __name__ == "__main__":
    main()