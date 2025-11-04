from tokenizer.tokenizer import *
import numpy as np
import time

tokenizer = Tokenizer.from_files(
    vocab_path="data/gpt2_vocab.json", merges_path="data/gpt2_merges.txt", special_tokens=["<|endoftext|>"]
)

start_time = time.time()

# 直接写入二进制文件
with open("data/TinyStories_single_special.bin", "wb") as out_f:
    with open("data/TinyStories_single.txt", "r", encoding="utf-8") as in_f:
        for _ids in tokenizer.encode_iterable(in_f):
            np.array(_ids, dtype=np.int32).tofile(out_f)

end_time = time.time()
print(f"Tokenization and saving took {end_time - start_time:.2f} seconds")

# 加载方式 int32
tokens = np.fromfile("data/TinyStories_single_special.bin", dtype=np.int32)
print(f"Loaded {len(tokens)} tokens from binary file.")