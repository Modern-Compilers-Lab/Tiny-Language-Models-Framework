import sentencepiece as spm
from tqdm import tqdm
import struct

class SPUnigramEvalTokenizer:
    def __init__(self, vocab_path):
        # Training the SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_path)
        print(f"[INFO] Loaded SP Unigram tokenizer with vocab size: {self.sp.get_piece_size()}")

    def encode(self, text: str):
        """Encode text to token IDs"""
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids):
        """Decode token IDs back to text"""
        return self.sp.decode(token_ids)

    def encode_to_file(self, input_file_path: str, output_file_path: str):
        """
        Encode each paragraph or example in a file and save token IDs in binary.
        """
        print(f"[INFO] Preparing to encode {input_file_path} to {output_file_path} ...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        examples = text.split('\n\n')[:-1]  # split by paragraph/example

        with open(output_file_path, 'wb') as out_file:
            for example in tqdm(examples, desc="Encoding examples"):
                example += '\n\n'
                token_ids = self.encode(example)
                for token_id in token_ids:
                    # For SP vocab > 256, use 2 bytes per token
                    out_file.write(struct.pack('>H', token_id))

        print(f"[INFO] Finished writing tokenized data to {output_file_path}")
