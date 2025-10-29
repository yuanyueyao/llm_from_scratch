import torch
from torch import nn
from torch import Tensor
import einops
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from typing import Optional, Callable, Iterable
from torch.nn import init
from torch.optim import Optimizer
import math

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {'lr':lr}
        super().__init__(params, defaults)

    def step(self, closure:None = None)-> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                if 't' not in state:
                    state['t']=0
                t = state['t']
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state['t'] = t + 1
        return loss


class AdamW(Optimizer):
    def __init__(self, params, lr, weight_decay=0.01, betas=(0.9,0.95),  eps=1e-8):
        defaults = {'lr':lr, 'betas':betas, 'weight_decay':weight_decay, 'eps':eps}
        super().__init__(params, defaults)

    def step(self, closure:None = None)->None:
        loss =None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'm' not in state:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['t'] = 0
    
                m, v, t = state['m'], state['v'], state['t']
                t += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

                p.data -= lr * weight_decay * p.data    # 注意要先进行过 weight decay
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state['t'] = t


def get_lr_cosine_schedule(t, lr_max, lr_min, t_w, t_c):
    if t < t_w:
        lr = lr_max * t / t_w
    elif t <= t_c:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w)))
    else:
        lr = lr_min

    return lr   


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= clip_coef