with open('30PS_chatGPT.txt', 'r') as f:
	data = f.read()

examples = data.split('\n\n')
for i, example in enumerate(examples):
	example = example[example.index('\n')+1:]
	example = '# code\n' + example
	examples[i] = example

with open('test.txt', 'w') as f:
	f.write(('\n\n').join(examples))