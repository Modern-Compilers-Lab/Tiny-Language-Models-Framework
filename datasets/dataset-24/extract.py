with open('../dataset-23/data-ds-23/direct_output_snippets.txt', 'r') as f:
	data = f.read()

examples = data.split('\n\n')[:500_000]
with open('./data-ds-24/direct_output_snippets.txt', 'w') as f:
	f.write('\n\n'.join(examples)+'\n\n')