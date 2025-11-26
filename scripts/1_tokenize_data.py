import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(sys.path)
from tokenizer.tokenizer import *
import numpy as np
import time

tokenizer = Tokenizer.from_files(
    vocab_path="data/TinyStories—train_vocab.json", merges_path="data/TinyStories-train_merges.txt", special_tokens=["<|endoftext|>"]
)

start_time = time.time()

# 直接写入二进制文件
with open("data/TinyStories_sample.bin", "wb") as out_f:
    with open("data/TinyStories_sample.txt", "r", encoding="utf-8") as in_f:
        for _ids in tokenizer.encode_iterable(in_f):
            np.array(_ids, dtype=np.int32).tofile(out_f)

end_time = time.time()
print(f"\nTokenization and saving took {end_time - start_time:.2f} seconds")

# 加载方式 int32
tokens = np.fromfile("data/TinyStories_sample.bin", dtype=np.int32)
print(f"\nLoaded {len(tokens)} tokens from binary file.")
print("\nFirst 10 tokens:", tokens)
decoded_text = tokenizer.decode(tokens.tolist())
print("\nDecoded text of first 10 tokens:", decoded_text)