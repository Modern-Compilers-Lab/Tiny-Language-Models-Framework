from tinypy_tokenizer import TinypyTokenizer

with open("data-dp-16-1/vocab_size.txt", "w") as f:
	f.write(str(len(TinypyTokenizer().keywords)))