with open('30PS_chatGPT_normalized.txt', 'r') as f:
	data = f.read()

examples = data.split('\n\n')
for i, example in enumerate(examples):
	example = example[example.index('\n')+1:]
	example = '# code\n' + example
	examples[i] = example

with open('30PS_chatGPT_norm_wo_desc.txt', 'w') as f:
	f.write(('\n\n').join(examples))