with open("./wider-while-steps-data/traced_snippets_filtered.txt", "r") as f:
	data = f.read()
examples = data.split("\n\n")[:-1]
print("Nb_examples:", len(examples))
examples = examples[:500]
print("Nb_examples:", len(examples))
with open("./wider-while-steps-data/final_snippets.txt", "w") as f:
	f.write("\n\n".join(examples)+"\n\n")