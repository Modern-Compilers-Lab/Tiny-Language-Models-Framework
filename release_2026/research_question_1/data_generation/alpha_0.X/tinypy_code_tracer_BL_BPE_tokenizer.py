from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import struct

class BPEByteTokenizer:
    def __init__(self, data_path, vocab_size=512):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = None
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True
        )
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train(files=[data_path], trainer=self.trainer)
        self.tokenizer.save("bpe_tokenizer.json")
        print(f"[INFO] Loaded BPE tokenizer with vocab size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def encode_to_file(self, input_file_path: str, output_file_path: str):
        print(f"[INFO] Preparing to encode {input_file_path} to {output_file_path} ...")
        print(f"[INFO] Loading {input_file_path} ...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        examples = text.split('\n\n')[:-1]  

        with open(output_file_path, 'wb') as out_file:
            for example in tqdm(examples, desc="Encoding examples"):
                example += '\n\n'  # keep original separation
                token_ids = self.encode(example)
                for token_id in token_ids:
                    out_file.write(struct.pack('>H', token_id))  # 2 bytes per token

        print(f"[INFO] Finished writing tokenized data to {output_file_path}")

if __name__ == "__main__":
    data_path = "data/test.txt"  # Path to your training data

    bpe_tokenizer = BPEByteTokenizer(data_path)
    bpe_tokenizer.tokenizer.save("bpe_tokenizer.json")