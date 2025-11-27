#这段代码的后期比较有问题，需要修改

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tokenizer.fast_tokenizer import FastTokenizer
import numpy as np
import time
from tqdm import tqdm

def count_lines(filename):
    """快速统计文件行数"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# 初始化 Fast Tokenizer
print("Initializing Fast Tokenizer...")
tokenizer = FastTokenizer(
    vocab_path="data/TinyStories-train_vocab.json",
    merges_path="data/TinyStories-train_merges.txt",
    special_tokens=["<|endoftext|>"]
)

input_file = "data/TinyStoriesV2-GPT4-valid.txt"
output_file = "data/TinyStoriesV2-GPT4-valid.bin"

# 统计总行数（可选，用于显示进度条）
print("Counting lines...")
total_lines = count_lines(input_file)
print(f"Total lines: {total_lines:,}")

print("\nStarting tokenization...")
start_time = time.time()

# 使用 tqdm 显示进度条
with open(output_file, "wb") as out_f:
    with open(input_file, "r", encoding="utf-8") as in_f:
        total_tokens = 0
        
        # 使用 tqdm 包装文件迭代器
        for line in tqdm(in_f, total=total_lines, desc="Tokenizing", unit="lines"):
            # 编码每一行
            token_ids = tokenizer.encode(line)
            # 写入二进制文件
            np.array(token_ids, dtype=np.uint16).tofile(out_f)
            total_tokens += len(token_ids)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n{'='*60}")
print(f"Tokenization completed!")
print(f"Total lines:  {total_lines:,}")
print(f"Total tokens: {total_tokens:,}")
print(f"Time taken:   {elapsed_time:.2f} seconds")
print(f"Speed:        {total_tokens / elapsed_time:,.0f} tokens/sec")
print(f"Output file:  {output_file}")
print(f"File size:    {os.path.getsize(output_file) / (1024**2):.2f} MB")
print(f"{'='*60}")

# 验证编码结果
print("\nVerifying the encoded file...")
tokens = np.fromfile(output_file, dtype=np.uint16)
print(f"Loaded {len(tokens):,} tokens from binary file")

if len(tokens) == total_tokens:
    print("✓ Token count matches!")
else:
    print(f"✗ Token count mismatch: expected {total_tokens:,}, got {len(tokens):,}")

# 显示前 200 个字符
print("\nFirst 200 characters of decoded text:")
decoded_sample = tokenizer.decode(tokens[:100].tolist())
print(decoded_sample[:200])
print("\n" + "="*60)

# 这AI写的也太抽象了，你长度都不一样那肯定不等啊 🤣
# # 比较与原文件的一致性
# print("\nComparing with original text...")
# with open(input_file, "r", encoding="utf-8") as f:
#     original_text = f.read(1000)  # 读取前 1000 字符

# # 找出对应的 token 数量
# test_tokens = []
# char_count = 0
# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         line_tokens = tokenizer.encode(line)
#         test_tokens.extend(line_tokens)
#         char_count += len(line)
#         if char_count >= 1000:
#             break

# decoded_text = tokenizer.decode(test_tokens)
# if decoded_text == original_text:
#     print("✓ Decoded full text matches original!")
# else:
#     print("⚠ Decoded text may differ slightly (due to tokenization boundaries)")
#     print("length of original:", len(original_text))
#     print("length of decoded:", len(decoded_text))
#     print(f"\nFirst 200 chars of original:\n{original_text[:200]}")
#     print(f"\nFirst 200 chars of decoded:\n{decoded_text[:200]}")