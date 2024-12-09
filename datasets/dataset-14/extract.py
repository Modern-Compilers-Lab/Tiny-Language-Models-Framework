# Script to create a new dataset by extracting a number of snippets from another dataset

NB_SNIPPETS = 30_000_000

with open('../dataset-11/data-ds-11/data.txt', 'r') as f:
	data = f.read()

examples = data.split('\n\n')[:NB_SNIPPETS]
examples = '\n\n'.join(examples)

with open('data-ds-14/data.txt', 'w') as f:
	f.write(examples)

