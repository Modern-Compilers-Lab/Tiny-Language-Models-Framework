import os

file_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_10_p/test.bin"
token_count = os.path.getsize(file_path)

print(f"Total tokens in train.bin: {token_count:,}")