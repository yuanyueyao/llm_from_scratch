
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding project root to sys.path: {project_root}")
sys.path.insert(0, project_root)

from tokenizer.tokenizer import Tokenizer
import numpy as np

import time
import torch
import numpy.typing as npt

from tqdm import tqdm
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
·
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Calculate the maximum valid starting index
    # We can start from index 0 up to len(dataset) - context_length
    # because we need context_length tokens for input and 1 more for the last target
    max_start_idx = len(dataset) - context_length
    
    # Randomly sample starting indices for each sequence in the batch
    # Use max_start_idx as the upper bound (exclusive)
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Initialize arrays to store the input sequences and targets
    input_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    target_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    
    # For each starting index, extract the input sequence and corresponding targets
    for i, start_idx in enumerate(start_indices):
        # Input sequence: tokens from start_idx to start_idx + context_length - 1
        input_sequences[i] = dataset[start_idx:start_idx + context_length]
        # Target sequence: tokens from start_idx + 1 to start_idx + context_length
        target_sequences[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and move to the specified device
    x = torch.from_numpy(input_sequences).long().to(device)
    y = torch.from_numpy(target_sequences).long().to(device)
    
    return x, y


if __name__ == "__main__":
    token_ids = np.memmap("data/TinyStories_yyy_small_tokens.bin", dtype=np.int32, mode="r")

    # 统计token_ids的一些基本信息
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"First 10 Token IDs: {token_ids[:10]}")

    batch = get_batch(token_ids, batch_size=4, context_length=28, device="cpu")

    tokenizer = Tokenizer.from_files(
        vocab_path="data/gpt2_vocab.json", merges_path="data/gpt2_merges.txt"
    )

    print(tokenizer.decode(batch[0][0].tolist()))