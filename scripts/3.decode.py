# verify the tokenizer's encode and decode works correctly
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding project root to sys.path: {project_root}")
sys.path.insert(0, project_root)

import numpy as np
from tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files("data/TinyStories-train_vocab.json", "data/TinyStories-train_merges.txt")

chunk_size = 1_000_000  # 每次读取100万个tokens

with open("test.txt", "w", encoding="utf-8") as f:
    # 使用 memmap 避免全部加载到内存
    tokens = np.memmap("data/TinyStories_sample.bin", dtype=np.int32, mode='r')
    
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        decoded_text = tokenizer.decode(chunk.tolist())
        f.write(decoded_text)

print("Decoding completed!")