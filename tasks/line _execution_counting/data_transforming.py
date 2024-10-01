from line_counter import line_counter
import argparse
import os

parser = argparse.ArgumentParser(description="Transform the data to the format needed for fine-tuning")
parser.add_argument("--data_file_path", type=str, default="data.txt", help="The path to the data file")
parser.add_argument("--output_dir", type=str, default="./data", help="The path to the output dir")
parser.add_argument("--output_file_name", type=str, default="finetuning_data.txt", help="The name of the output file")
args = parser.parse_args()


data_file_path = args.data_file_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

output_file_name = args.output_file_name
output_file_path = os.path.join(output_dir, output_file_name)

data = open(data_file_path, "r").read()
code_snippets = data.split("\n\n")[:-1]
nbr_programs = len(code_snippets)
nbr_exceptions = 0
with open(output_file_path, "w") as f:
    for program in code_snippets:
        try:
            count = line_counter(program)
            result = f"""{program}\n# count\n# {count}"""
            f.write(result + "\n\n")
        except ValueError:
            nbr_exceptions += 1
            pass

print(f"Percentage of programs that raised an exception: {nbr_exceptions/nbr_programs*100}%")
f.close()