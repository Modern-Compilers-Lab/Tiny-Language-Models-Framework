import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

import transformers

transformers.logging.set_verbosity_error()

def tokenize_data(train_file="../data/train.txt",
                  test_file="../data/val.txt",
                  tokenizer_name="codellama/CodeLlama-7b-hf",
                  train_output_dir="data/tokenized_train",
                  val_output_dir="data/tokenized_val"):
    print("Tokenizing data...")

    # Read the training and test data
    with open(train_file) as f:
        train_data = f.read()

    with open(test_file) as f:
        test_data = f.read()

    # Split the snippets into individual examples
    train_snippets = train_data.split('\n\n')
    test_snippets = test_data.split('\n\n')

    # Create datasets from the snippets
    train_dataset = Dataset.from_pandas(pd.DataFrame({'snippets': train_snippets}))
    eval_dataset = Dataset.from_pandas(pd.DataFrame({'snippets': test_snippets}))

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Function to tokenize a prompt
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )

        # For self-supervised learning, labels are also the inputs
        result["labels"] = result["input_ids"].copy()

        return result

    # Function to generate and tokenize a prompt
    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["snippets"]

        return tokenize(full_prompt)

    # Tokenize the training and validation datasets
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    # Save the tokenized datasets to disk
    print(f"Saving tokenized datasets to {train_output_dir} and {val_output_dir}...")
    tokenized_train_dataset.save_to_disk(train_output_dir)
    tokenized_val_dataset.save_to_disk(val_output_dir)
    
    print("Tokenization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize data for language model training.")
    parser.add_argument("--train_file", type=str, default="../data/train.txt", help="Path to the training file")
    parser.add_argument("--test_file", type=str, default="../data/val.txt", help="Path to the test file")
    parser.add_argument("--tokenizer_name", type=str, default="codellama/CodeLlama-7b-hf", help="Name or path of the tokenizer")
    parser.add_argument("--train_output_dir", type=str, default="data/tokenized_train", help="Path to save the tokenized training dataset")
    parser.add_argument("--val_output_dir", type=str, default="data/tokenized_val", help="Path to save the tokenized validation dataset")
    
    args = parser.parse_args()

    tokenize_data(args.train_file, args.test_file, args.tokenizer_name, args.train_output_dir, args.val_output_dir)
