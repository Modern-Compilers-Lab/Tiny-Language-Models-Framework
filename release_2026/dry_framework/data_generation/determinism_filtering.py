import multiprocessing
from tqdm import tqdm
import os



# _________________________________SCRIPT_________________________________________________

# cd /Path/to/folder/where/this/script/lives
# python determinism_filtering.py

#_____________________________IMPORTANT_HYPERPARAMETERS___________________________________

#use ctrl+f for quick access:
# NUM_CORES : number of workers used (the more the better)
# BATCH_SIZE : number of snippets processed by each worker at once " ideally : total_nb_snippets mod (num_cores*batch_size) = 0 "


#_____________________________________CODE_______________________________________________



# Import your custom tokenizer
# Ensure this file is in the same directory or python path
from tinypy_code_tracer_tokenizer import TinypyTokenizer

# Global tokenizer variable for worker processes
# This prevents pickling the tokenizer repeatedly
tokenizer_instance = None

def init_worker():
    """Initializes the tokenizer in each worker process."""
    global tokenizer_instance
    tokenizer_instance = TinypyTokenizer()

def process_batch(examples_batch):
    """
    Processes a batch of examples.
    Returns a tuple: (list of valid strings, list of oversize strings)
    """
    block_size = 512
    valid_snippets = []
    oversize_snippets = []
    
    global tokenizer_instance
    
    for orig_example in examples_batch:
        # Recreate the trace terminator if needed, mimicking original logic
        # Note: We append \n\n because the split removed it
        example_text = orig_example + "\n\n"
        
        # Tokenize
        tokens = tokenizer_instance.tokenize(example_text)
        
        valid = True
        
        # Find boundary indices
        boundaries_indices = [i for i, token in enumerate(tokens) if token == "\n#STEP\n"]
        boundaries_indices.append(len(tokens))
        
        # Check constraints
        # Logic: check distance between boundaries starting from index 2
        for i in range(2, len(boundaries_indices)):
            pair_length = boundaries_indices[i] - boundaries_indices[i-2]
            
            if pair_length > block_size:
                valid = False
                oversize_snippets.append(f"Pair length too long: {pair_length}\n{example_text}")
                break
        
        if valid:
            valid_snippets.append(example_text)
            
    return valid_snippets, oversize_snippets

def example_generator(file_path, batch_size=1000):
    """
    Reads the huge file line-by-line and yields batches of examples.
    This avoids loading 180GB into RAM at once.
    """
    current_batch = []
    current_example_lines = []
    
    print(f"Opening large file stream: {file_path} ...")
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Assuming examples are separated by blank lines (double newline)
            if line == '\n': 
                if current_example_lines:
                    # Join lines to form one example
                    full_example = "".join(current_example_lines).strip()
                    if full_example:
                        current_batch.append(full_example)
                    
                    current_example_lines = []
                    
                    # Yield if batch is full
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
            else:
                current_example_lines.append(line)
        
        # Yield remaining data
        if current_example_lines:
            full_example = "".join(current_example_lines).strip()
            if full_example:
                current_batch.append(full_example)
        if current_batch:
            yield current_batch

def main():
    folder_path = "./data/"
    input_file = folder_path + "raw_ID.txt"
    output_valid = folder_path + "determinism_filtered_snippets.txt"
    output_oversize = folder_path + "oversize_snippets.txt"

    # Configuration
    NUM_CORES = 5 # Leave a few cores free for system/IO
    BATCH_SIZE = 100_000 # Higher batch size reduces Inter-Process Communication overhead

    # Prepare output files (open in write mode to clear them, then append mode in loop)
    # We will append to these files as results come in
    with open(output_valid, "w") as f, open(output_oversize, "w") as fo:
        pass 

    print(f"Starting processing with {NUM_CORES} cores...")

    # Initialize Pool
    # imap_unordered is faster as it yields results as soon as any worker finishes
    with multiprocessing.Pool(processes=NUM_CORES, initializer=init_worker) as pool:
        
        # Create the generator
        batch_gen = example_generator(input_file, BATCH_SIZE)
        
        # Counters for progress
        total_valid = 0
        total_oversize = 0
        
        # Use tqdm to track processed batches (unit is Batches, not single examples)
        pbar = tqdm(pool.imap_unordered(process_batch, batch_gen), desc="Processing Batches")
        
        # Open files in append mode to write as we go
        with open(output_valid, "a", encoding='utf-8') as f_valid, \
             open(output_oversize, "a", encoding='utf-8') as f_over:
            
            for valid_batch, oversize_batch in pbar:
                
                # Write results immediately to disk
                if valid_batch:
                    f_valid.writelines(valid_batch)
                    total_valid += len(valid_batch)
                
                if oversize_batch:
                    f_over.writelines(oversize_batch)
                    total_oversize += len(oversize_batch)
                
                # Update description occasionally
                pbar.set_description(f"Valid: {total_valid} | Oversize: {total_oversize}")

    print("Processing complete.")

if __name__ == '__main__':
    main()