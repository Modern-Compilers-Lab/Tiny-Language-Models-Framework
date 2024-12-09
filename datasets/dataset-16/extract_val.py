with open("../dataset-11/data-ds-11/data.txt", "r") as f:
	data = f.read()

from tinypy_tokenizer import TinypyTokenizer
from tqdm import tqdm

tpt = TinypyTokenizer()

examples = data.split("\n\n")
# i take 1/8 of the number of tokens in train as usual
nb_tokens_max = int(1/8 * 3012762192) # the number of tokens in the train split of DP-10-1 after being tokenized by the keyword based tokenizer for CoT "tinypy_tokenizer.py"
nb_tokens = 0
kept_examples = []

# i reverse the order of reading from examples so that val and train are disjoint
for example in tqdm(reversed(examples)):
	nb_tokens += len(tpt.tokenize(example))
	kept_examples.append(example)
	if nb_tokens >= nb_tokens_max:
		break

result = '\n\n'.join(kept_examples)

with open('data-ds-16/val.txt', 'w') as f:
	f.write(result)
