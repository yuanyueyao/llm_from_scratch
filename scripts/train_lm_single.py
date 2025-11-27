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

@hydra.main(config_path="configs/", config_name="train_lm", version_base=None)
def main(cfg: DictConfig):
    model_cfg: DictConfig = cfg.model
    optimizer_cfg: DictConfig = cfg.optimizer
    data_cfg: DictConfig = cfg.data
    training_cfg: DictConfig = cfg.training

    wandb.init(project="train-llm", name="train_no_validation")
    print(isinstance(model_cfg, DictConfig))
    print("Model Configuration:")
    print("model_cfg", model_cfg)
    print("optimizer_cfg", optimizer_cfg)
    print("data_cfg", data_cfg)
    # for key, value in model_cfg.items():
    #     print(f"  {key}: {value}")

    # print("Optimizer Configuration:")
    # for key, value in optimizer_cfg.items():
    #     print(f"  {key}: {value}")

    # print("Training Configuration:")
    # for key, value in data_cfg.items():
    #     print(f"  {key}: {value}")

    lm: Transformer_lm = Transformer_lm(**model_cfg).to(data_cfg.device)
    optimizer: AdamW = AdamW(lm.parameters(), **optimizer_cfg)

    tokenizer = Tokenizer.from_files("data/TinyStories-train_vocab.json", "data/TinyStories-train_merges.txt", special_tokens=["<|endoftext|>"])
    data = load_data(data_cfg.data_path)
    # scripts/train_lm_single.py
    data = load_data(data_cfg.data_path)
    print(f"Data dtype: {data.dtype}, Data shape: {data.shape}")
    # 打印前 10 个 token
    print(f"First 10 tokens: {data[:10].tolist()}") 
    # 打印最大值
    print(f"Max token in whole dataset (sample): {data[:20000].max()}")
    pbar = tqdm(range(0, training_cfg.train_steps), desc="Training", leave=False)
    
    save_interval = training_cfg.save_interval if 'save_interval' in training_cfg else 2000


    for step in pbar:
        input, target = get_batch(data, data_cfg.batch_size, data_cfg.context_length, data_cfg.device)
        
        # --- 添加以下代码进行调试 ---
        max_token_id = input.max().item()
        if max_token_id >= model_cfg.vocab_size:
            print(f"!!! Error Check: Max token ID in batch is {max_token_id}, but vocab_size is {model_cfg.vocab_size}.")
            # 此时应该直接退出或报错，因为已经确认索引越界
        # --- 调试代码结束 ---

        logits: Tensor = lm(input)
        loss: Tensor = cross_entropy(logits, target)

        
        # if "<" in tokenizer.decode(target[0].tolist()):
        #     print("input:", tokenizer.decode(input[0].tolist()))
        #     print("target:", tokenizer.decode(target[0].tolist()))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(lm.parameters(), training_cfg.grad_clip_norm)
        lr = get_lr_cosine_schedule(
            step, training_cfg.lr_max, training_cfg.lr_min, training_cfg.warmup_steps, training_cfg.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
        pbar.set_postfix({"loss": loss.item(), "lr": lr})

        if step % save_interval == 0 and step != 0:
            save_checkpoint(lm, optimizer, step, training_cfg.checkpoint_path+f".step{step}.pt")
            print(f"Checkpoint saved at step {step} to {training_cfg.checkpoint_path+f'.step{step}.pt'}")

    wandb.finish()

if __name__ == "__main__":
    main()