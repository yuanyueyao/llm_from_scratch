import time
from tokenizer.tokenizer import Tokenizer as SlowTokenizer
from tokenizer.fast_tokenizer import FastTokenizer
import numpy as np  

TXT_PATH = "decoded_output.txt"
# TXT_PATH = "data/TinyStoriesV2-GPT4-train-Part1.txt"


def benchmark(name, tokenizer, encode_func):
    print(f"\n==== Running benchmark: {name} ====")

    total_lines = 0
    total_tokens = 0
    start = time.time()
    total_ids = []
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue

            ids = encode_func(tokenizer, line)
            total_tokens += len(ids)
            total_ids.extend(ids)
            total_lines += 1

    end = time.time()
    cost = end - start

    print(f"Lines          : {total_lines:,}")
    print(f"Tokens         : {total_tokens:,}")
    print(f"Time cost      : {cost:.2f} s")
    print(f"Lines/sec      : {total_lines / cost:.2f}")
    print(f"Tokens/sec     : {total_tokens / cost:.2f}")

    return total_ids
    
    


def encode_slow(tokenizer, text):
    return tokenizer.encode(text)


def encode_fast(tokenizer, text):
    return tokenizer.encode(text)  # 现在 Fast tokenizer 也有 encode 方法了


if __name__ == "__main__":
    print("Loading vocab and merges...")

    # 初始化 tokenizers
    slow = SlowTokenizer.from_files(
        vocab_path="data/TinyStories-train_vocab.json",
        merges_path="data/TinyStories-train_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    
    # Fast tokenizer 直接从文件加载
    fast = FastTokenizer(
        vocab_path="data/TinyStories-train_vocab.json",
        merges_path="data/TinyStories-train_merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    # 测试一下结果是否一致
    test_text = "Hello, world! This is a test.<|endoftext|>"
    # slow_result = encode_slow(slow, test_text)
    fast_result = encode_fast(fast, test_text)
    
    print(f"\nTest encoding comparison:")
    # print(f"Slow: {slow_result}")
    print(f"Fast: {fast_result}")
    # print(f"Match: {slow_result == fast_result}")
    tokens = np.fromfile("data/TinyStoriesV2-GPT4-train-Part1.bin", dtype=np.int32)
    print(f"Loaded {len(tokens)} tokens from binary file.")
    # 运行 benchmark
    # slow_result = benchmark("Slow naive BPE", slow, encode_slow)
    fast_result = benchmark("Fast PQ+LinkedList BPE", fast, encode_fast)
    print(f"\nFinal comparison of benchmark results:")
    # print(f"Match: {slow_result == fast_result}")