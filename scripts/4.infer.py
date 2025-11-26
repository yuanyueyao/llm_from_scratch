import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding project root to sys.path: {project_root}")
sys.path.insert(0, project_root)

from llm.model import Transformer_lm, cross_entropy
from optimizer.optimizer import AdamW, save_checkpoint, load_checkpoint, gradient_clipping, get_lr_cosine_schedule
from tokenizer.tokenizer import Tokenizer
from data.dataLoader import get_batch, load_data

import hydra
from omegaconf import DictConfig

import torch
from torch import nn
from torch import Tensor
import einops
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch.nn import init
import numpy as np
import wandb

from tqdm import tqdm

@hydra.main(config_path="configs/", config_name="train_lm_single", version_base=None)
def main(cfg: DictConfig):
    model_cfg: DictConfig = cfg.model
    optimizer_cfg: DictConfig = cfg.optimizer
    data_cfg: DictConfig = cfg.data
    training_cfg: DictConfig = cfg.training


    # for key, value in model_cfg.items():
    #     print(f"  {key}: {value}")

    # print("Optimizer Configuration:")
    # for key, value in optimizer_cfg.items():
    #     print(f"  {key}: {value}")

    # print("Training Configuration:")
    # for key, value in data_cfg.items():
    #     print(f"  {key}: {value}")

    lm: Transformer_lm = Transformer_lm(**model_cfg)
    checkpoint = torch.load(training_cfg.checkpoint_path+".step700.pt", weights_only=False)
    lm.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = Tokenizer.from_files("data/TinyStories-train_vocab.json", "data/TinyStories-train_merges.txt", special_tokens=["<|endoftext|>"])
    input_ids = torch.tensor([tokenizer.encode("Once upon a time")], device=data_cfg.device)
    print(input_ids.shape)
    out_ids = lm.generate(input_ids, max_new_tokens=300, eos_token_id=50256)

    print(tokenizer.decode(out_ids[0].tolist()))
if __name__ == "__main__":
    main()