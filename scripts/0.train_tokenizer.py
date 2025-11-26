import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tokenizer.tokenizer import Tokenizer

special_tokens = ["your_special_tokens"] # Define your special tokens if any, for example: ["<s>", "</s>", "<pad>", "<|endoftext|>"]
tokenizer = Tokenizer(None, None, special_tokens = special_tokens) # None vocab and merges, your special tokens here
input_path = "path/to/your/textfile.txt"  # Replace with your text file path
tokenizer.train(input_path, vocab_size=10_000, special_tokens=special_tokens)
tokenizer.save("../data/custom_vocab.json", "../data/custom_merges.txt")
