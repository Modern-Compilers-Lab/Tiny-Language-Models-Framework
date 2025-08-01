with open("/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-31/data-ds-31/arccp_direct_output_snippets.txt", "r") as f:
	data = f.read()
examples = data.split("\n\n")[:-1]
examples = examples[:int(len(examples) * 0.0515)]
examples = "\n\n".join(examples)+"\n\n"
with open("./data-ds-stage-4/arccp_direct_output_snippets.txt", "w") as f:
	f.write(examples)
