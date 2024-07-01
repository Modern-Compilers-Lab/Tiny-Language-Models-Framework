import os
import re
import torch
import warnings
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

import transformers

transformers.logging.set_verbosity_error()

# Ignore all warnings
warnings.filterwarnings("ignore")

def evaluate_model(base_model="codellama/CodeLlama-7b-hf",
                   checkpoint_dir="models/code-llama-finetuned-level1",
                   test_file="../data/test.txt",
                   output_file='results/result_llama.txt',
                   csv_file='results/results_llama.csv',
                   max_new_tokens=30):
    
    print("Evaluating model...")
    print()

    # Load the pretrained model
    print("Loading the pretrained model...")
    print()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load the tokenizer
    print("Loading the tokenizer...")
    print()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load the fine-tuned model
    print("Loading the fine-tuned model...")
    print()
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    
    # Load and preprocess the test data
    print("Loading the test data...")
    print()
    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()

    examples = [example for example in text.split("\n\n") if example]

    data = []

    print("Generating predictions...")
    print()
    for example in tqdm(examples):
        splited_example = example.split("# output\n")
        prompt_text = splited_example[0] + "# output\n"
        real_response = splited_example[1]
        
        real_number_response = re.search(r"\d+\.\d+|\d+|-\d+|-\d+\.\d+", real_response.replace("\n", ""))
        real_result = float(real_number_response.group()) if real_number_response else 0.0
        
        model_input = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        response = tokenizer.decode(model.generate(**model_input, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
        
        splited_response = response.split("# output")
        number_response = re.search(r"\d+\.\d+|\d+|-\d+|-\d+\.\d+", splited_response[1].replace("\n", ""))
        generated_result = float(number_response.group()) if number_response else 0.0

        data.append({'Prompt': prompt_text, 'Real_Results': real_result, 'Generated_Results': generated_result})

    # Calculate accuracy
    accuracy = sum(1 for d in data if d['Real_Results'] == d['Generated_Results']) / len(data)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Store accuracy in a file
    print("Storing accuracy in a file...")
    print()
    with open(output_file, 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Store results in a CSV file using pandas
    print("Storing results in a CSV file...")
    print()
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with specified parameters.")
    parser.add_argument('--base_model', type=str, default="codellama/CodeLlama-7b-hf", help="Base model name or path")
    parser.add_argument('--checkpoint_dir', type=str, default="models/code-llama-finetuned-level1", help="Directory containing the model checkpoint")
    parser.add_argument('--test_file', type=str, default="../data/test.txt", help="Path to the test file")
    parser.add_argument('--output_file', type=str, default='results/result_llama.txt', help="Path to the output file where the accuracy will be stored")
    parser.add_argument('--csv_file', type=str, default='results/results_llama.csv', help="Path to the CSV file where the results will be stored")
    parser.add_argument('--max_new_tokens', type=int, default=30, help="Maximum number of new tokens to generate")

    args = parser.parse_args()

    evaluate_model(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        test_file=args.test_file,
        output_file=args.output_file,
        csv_file=args.csv_file,
        max_new_tokens=args.max_new_tokens
    )
