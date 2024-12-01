import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Identify clones in the output JSON files")
    parser.add_argument('--input-file', required=True, help='Path to the input JSON file with outputs')
    parser.add_argument('--output-file', required=True, help='Path to the output JSON file to store clones')
    args = parser.parse_args()

    # Load the outputs from the specified file
    with open(args.input_file) as f:
        outputs = json.load(f)

    value_to_keys = defaultdict(list)
    for key, value in outputs.items():
        value_to_keys[tuple(value)].append(int(key))

    clones = {}
    function = 0
    for keys in value_to_keys.values():
        if len(keys) > 1:
            clones[function] = keys
            function += 1

    def trim_map_values(input_map, max_values):
        trimmed_map = {}
        
        for key, values in input_map.items():
            if len(values) > max_values:
                trimmed_map[key] = values[:max_values] 
            else:
                trimmed_map[key] = values  # Keep the original list if it's already <= 200 values
        
        return trimmed_map

    clones = trim_map_values(clones, 200) # Keep only the first 200 values

    # Save the clones to the specified output file
    with open(args.output_file, "w") as write_file:
        json.dump(clones, write_file)

    print(f'The clones are stored in {args.output_file}')

if __name__ == "__main__":
    main()
