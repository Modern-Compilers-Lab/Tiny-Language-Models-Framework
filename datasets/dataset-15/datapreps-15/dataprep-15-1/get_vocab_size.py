from tinypy_tokenizer import TinypyTokenizer

with open("data-dp-15-1/vocab_size.txt", "w") as f:
	f.write(str(len(TinypyTokenizer().keywords)))