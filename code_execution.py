import os
import pickle
import torch
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import argparse
from model import GPT

class ScriptEvaluator:
    """
    Class to evaluate a GPT model on a dataset and save the results.
    """

    def __init__(self, dataset_dir, model_name):
        """
        Initialize ScriptEvaluator with dataset directory and model name.
        
        Args:
        - dataset_dir (str): Directory where the dataset is stored.
        - model_name (str): Name of the pre-trained model (without .pth extension).
        """
        self.dataset = dataset_dir
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(1337)
        self.test_data, self.meta = self.load_dataset()
        self.m = self.load_model()

    def load_dataset(self):
        """
        Load test dataset and metadata.
        """
        test_data = np.memmap(os.path.join(self.dataset, 'test.bin'), dtype=np.uint16, mode='r')
        meta_path = os.path.join(self.dataset, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

        return test_data, meta

    def load_model(self):
        """
        Load pre-trained model based on the provided model name.
        """
        model_path = os.path.join('models', f"{self.model_name}.pth") 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        model = GPT()
        print("Compiling the model...\n")
        try:
            model = torch.compile(model)  # requires PyTorch 2.0
        except Exception as e:
            pass
        model.load_state_dict(torch.load(model_path))
        m = model.to(self.device)
        return m

    def encode(self, s):
        """
        Encode string `s` into token IDs.
        """
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """
        Decode token IDs `l` into a string.
        """
        return ''.join([self.itos[i] for i in l])

    def evaluate_example(self, example, max_new_tokens=30):
        """
        Evaluate an example using the loaded model.
        """
        # Split example and determine maximum new tokens allowed
        splited_example = example.split("# output")
        if not ("for" in splited_example[0]):
            max_new_tokens = 22
        
        # Encode prompt and prepare for evaluation
        encoded_example = torch.tensor(self.encode(splited_example[0] + "# output"), dtype=torch.long).unsqueeze(0).to(self.device)
        prompt_text = splited_example[0] + "# output"
        result_example = splited_example[-1]

        # Extract real results from example
        real_results = [float(match.group()) for match in re.finditer(r"(?<=# )-?\d+(\.\d+)?", result_example.split('\n\n')[0].replace("\n", ""))]

        # Generate response from model and extract generated results
        response = self.decode(self.m.generate(encoded_example, max_new_tokens=max_new_tokens)[0].tolist())
        splited_response = response.split("# output")
        result_response = splited_response[-1]
        generated_results = [float(match.group()) for match in re.finditer(r"(?<=# )-?\d+(\.\d+)?", result_response.split('\n\n')[0].replace("\n", ""))]

        return prompt_text, real_results, generated_results
    
    def write_results_to_file(self, output_file, prompt, real_results, generated_results):
        """
        Write evaluation results to a CSV file.
        """
        df = pd.DataFrame({
            'Prompt': prompt,
            'Real_Results': real_results,
            'Generated_Results': generated_results
        })
        df.to_csv(output_file, index=False)

    def main(self):
        """
        Main evaluation function.
        """
        # Extracting stoi and itos from meta
        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']

        # Split examples and initialize lists for results
        examples = self.decode(self.test_data).split("\n\n")
        examples = [example for example in examples if example]

        # Start evaluation process
        print(f"Starting evaluation for model '{self.model_name}' on dataset '{self.dataset}'...")
        prompt = []
        real_results = []
        generated_results = []

        # Iterate through examples and evaluate each one
        for example in tqdm(examples):
            prompt_text, real_result, result = self.evaluate_example(example)
            prompt.append(prompt_text)
            real_results.append(real_result)
            generated_results.append(result)

        # Calculate and print accuracy
        correct_count = sum(1 for real, generated in zip(real_results, generated_results) if real == generated)
        accuracy = correct_count / len(generated_results)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Store accuracy in a file
        accuracy_file = os.path.join('results', f"{self.model_name}_accuracy.txt")  # Saving in 'results' folder
        with open(accuracy_file, 'w') as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        print(f"Accuracy saved to {accuracy_file}")

        # Store results in a CSV file
        results_file = os.path.join('results', f"{self.model_name}_results.csv")  # Saving in 'results' folder
        self.write_results_to_file(results_file, prompt, real_results, generated_results)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Evaluate NanoGPT model on a dataset.')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Directory where the dataset is stored')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model (without .pth extension)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create ScriptEvaluator instance and run main function
    evaluator = ScriptEvaluator(args.dataset_dir, args.model_name)
    evaluator.main()
