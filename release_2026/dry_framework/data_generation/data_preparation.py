import multiprocessing
import os
import glob
from tqdm import tqdm
from tinypy_code_tracer_tokenizer import TinypyTokenizer



# _________________________________SCRIPT_________________________________________________

# cd /Path/to/folder/where/this/script/lives
# python determinism_filtering.py

#_____________________________IMPORTANT_HYPERPARAMETERS___________________________________

#use ctrl+f for quick access:

# VAL_COUNT : number of validation snippets
# TEST_COUNT : number of testing snippets
# TRAIN_CAP : number of training snippets
# NUM_CORES : how many workers to execute the script
# EXAMPLES_PER_SHARD : kinda like batch size .. kind of "" ideally : total_nb_snippets mod (num_cores*EXAMPLES_PER_SHARD) = 0 ""

#_____________________________________CODE_______________________________________________


# --- Configuration ---
root = "./data/"
INPUT_FILE = root+"determinism_filtered_snippets.txt"
OUTPUT_FOLDER = root
TEMP_FOLDER = os.path.join(OUTPUT_FOLDER, "temp_shards")

# Data Splits
VAL_COUNT = 10_000
TEST_COUNT = 2048
TRAIN_CAP = 1_000_000 - VAL_COUNT - TEST_COUNT 

# System
NUM_CORES = 10  # Leave some breathing room
EXAMPLES_PER_SHARD = 50_000 # 

def tokenize_shard(args):
    """
    Worker function to tokenize a single text file into a binary file.
    """
    input_txt, output_bin = args
    try:
        # Re-init tokenizer in worker to avoid pickling issues
        tpt = TinypyTokenizer()
        tpt.encode_to_file(input_txt, output_bin)
        return None
    except Exception as e:
        return f"Error processing {input_txt}: {str(e)}"

def merge_binaries(shard_pattern, final_output):
    """
    Concatenates binary shards into one final file.
    """
    shards = sorted(glob.glob(shard_pattern))
    print(f"[*] Merging {len(shards)} shards into {final_output}...")
    
    with open(final_output, 'wb') as outfile:
        for shard in tqdm(shards, desc="Merging"):
            with open(shard, 'rb') as infile:
                # Copy in chunks to avoid memory issues
                while True:
                    chunk = infile.read(1024 * 1024 * 64) # 64MB chunks
                    if not chunk:
                        break
                    outfile.write(chunk)

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    
    # ---------------------------------------------------------
    # STEP 1: Stream Split (Single Threaded I/O)
    # ---------------------------------------------------------
    print("[*] Phase 1: Streaming and Sharding Text Data...")
    
    # Paths for non-sharded splits
    val_txt_path = os.path.join(OUTPUT_FOLDER, "val.txt")
    test_txt_path = os.path.join(OUTPUT_FOLDER, "test.txt")
    
    # Clear existing
    open(val_txt_path, 'w').close()
    open(test_txt_path, 'w').close()

    current_example_idx = 0
    current_shard_idx = 0
    current_shard_count = 0
    
    # Open the first train shard
    current_train_shard_path = os.path.join(TEMP_FOLDER, f"train_shard_{current_shard_idx:04d}.txt")
    f_train_shard = open(current_train_shard_path, 'w')
    
    f_val = open(val_txt_path, 'a')
    f_test = open(test_txt_path, 'a')
    
    buffer = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f_in:
        for line in tqdm(f_in, desc="Splitting Text"):
            if line == '\n':
                if buffer:
                    # Example complete
                    full_example = "".join(buffer).strip() + "\n\n"
                    
                    # Logic: Where does this example go?
                    if current_example_idx < TEST_COUNT:
                        # Write to TEST
                        f_test.write(full_example)                            
                    elif current_example_idx < (TEST_COUNT + VAL_COUNT):
                        # Write to VAL
                        f_val.write(full_example)
                    elif current_example_idx < (TRAIN_CAP + VAL_COUNT + TEST_COUNT):
                        # Write to TRAIN
                        f_train_shard.write(full_example)
                        current_shard_count += 1
                        
                        # Rotate shard if full
                        if current_shard_count >= EXAMPLES_PER_SHARD:
                            f_train_shard.close()
                            current_shard_idx += 1
                            current_shard_count = 0
                            current_train_shard_path = os.path.join(TEMP_FOLDER, f"train_shard_{current_shard_idx:04d}.txt")
                            f_train_shard = open(current_train_shard_path, 'w')
                        
                    
                    current_example_idx += 1
                    buffer = []
            else:
                buffer.append(line)
                
    # Close all handles
    f_train_shard.close()
    f_val.close()
    f_test.close()
    
    print(f"[*] Total examples processed: {current_example_idx}")
    
    # ---------------------------------------------------------
    # STEP 2: Parallel Tokenization
    # ---------------------------------------------------------
    print(f"[*] Phase 2: Parallel Tokenization on {NUM_CORES} cores...")
    
    tasks = []
    
    # 1. Add Train Shards
    train_shards = sorted(glob.glob(os.path.join(TEMP_FOLDER, "train_shard_*.txt")))
    for shard in train_shards:
        bin_out = shard.replace(".txt", ".bin")
        tasks.append((shard, bin_out))
        
    # 2. Add Val and Test (processed as single chunks usually fine, or could be sharded too)
    tasks.append((val_txt_path, os.path.join(OUTPUT_FOLDER, "val.bin")))
    tasks.append((test_txt_path, os.path.join(OUTPUT_FOLDER, "test.bin"))) # This is a "dummy" task here, actually we want separate final bins
    
    # Remove test/val from the shard merge list later
    
    # Run Pool
    with multiprocessing.Pool(NUM_CORES) as pool:
        results = list(tqdm(pool.imap_unordered(tokenize_shard, tasks), total=len(tasks), desc="Tokenizing"))
    
    # Check errors
    errors = [r for r in results if r is not None]
    if errors:
        print("ERRORS found during tokenization:")
        for e in errors: print(e)
        return

    # ---------------------------------------------------------
    # STEP 3: Merge Train Binaries
    # ---------------------------------------------------------
    print("[*] Phase 3: Merging Train Binaries...")
    
    # We only merge the train shards. Val and Test were tokenized directly to their final destination above.
    # Note: We need to exclude val.bin and test.bin if they ended up in the temp folder, 
    # but here we set their output to OUTPUT_FOLDER, so safe.
    
    merge_binaries(os.path.join(TEMP_FOLDER, "train_shard_*.bin"), os.path.join(OUTPUT_FOLDER, "train.bin"))

    # ---------------------------------------------------------
    # STEP 4: Cleanup & Vocab
    # ---------------------------------------------------------
    print("[*] Creating vocab_size.txt...")
    tpt = TinypyTokenizer()
    with open(os.path.join(OUTPUT_FOLDER, "vocab_size.txt"), "w") as f:
        f.write(str(len(tpt.keywords)))

    print("[*] Cleaning up temp files...")
    # Uncomment below to actually delete temp files
    # import shutil
    # shutil.rmtree(TEMP_FOLDER)
    
    print("[*] Done!")

if __name__ == "__main__":
    main()