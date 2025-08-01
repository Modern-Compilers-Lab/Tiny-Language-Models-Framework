with open("./data-ds-35/filtered_arccp_traced_snippets.txt", "r") as f:
	data = f.read()
examples = data.split("\n\n")[:-1]

val_examples = examples[-10_000:]
val_examples = "\n\n".join(val_examples) + "\n\n"
with open("./data-ds-35/val.txt", "w") as f:
	f.write(val_examples)

train_examples = examples[:-10_000]
train_examples = "\n\n".join(train_examples) + "\n\n"
with open("./data-ds-35/train.txt", "w") as f:
	f.write(train_examples)