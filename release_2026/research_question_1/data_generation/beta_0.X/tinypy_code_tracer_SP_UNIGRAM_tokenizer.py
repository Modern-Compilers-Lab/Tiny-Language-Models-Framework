import sentencepiece as spm
from tqdm import tqdm
import struct

class SPUnigramTokenizer:
    def __init__(self, data_path, model_prefix="sp_unigram_newest", vocab_size=512):
        # Training the SentencePiece model
        spm.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='unigram',
            normalization_rule_name='identity',
            remove_extra_whitespaces=False,
            split_by_whitespace=False,
            byte_fallback=True,              # Changed: handles unknown characters
            allow_whitespace_only_pieces=True,
            input_sentence_size=10000000,
            shuffle_input_sentence=True,
            character_coverage=1.0,          # Added: ensure all chars covered
            unk_surface=' ',                 # Added: explicit unknown token
        )
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_prefix}.model")
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


