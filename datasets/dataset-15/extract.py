with open("../dataset-7/data-ds-7/data.txt", "r") as f:
	data = f.read()
examples = data.split("\n\n")[:2_500_000]
examples = "\n\n".join(examples)
with open("./data-ds-15/data.txt", "w") as f:
	f.write(examples)