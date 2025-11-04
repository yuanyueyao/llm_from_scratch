import torch
from torch import nn
from torch import Tensor
import einops
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch.nn import init
class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype))
        
        weight_std = 2.0 / (in_features + out_features) ** 0.5
        init.trunc_normal_(self.weight, mean=0.0, std=weight_std, a=-3*weight_std, b=3*weight_std)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias=False)"
    



class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int,     # 词汇表大小
        embedding_dim: int,      # 向量维度 d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 初始化 embedding 矩阵，形状为 [vocab_size, embedding_dim]
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # 初始化方式：N(0, 1) 截断到 [-3, 3]
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq embedding_dim"]:
        """
        查找每个 token 的向量。
        token_ids: [batch_size, seq_len]
        返回: [batch_size, seq_len, embedding_dim]
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #scale
    def forward(self, x: Float[Tensor, "... d_model"])->Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True)+self.eps).sqrt() ### 这里一定要注意保持dim，否则会因为无法广播报错（广播是从最后一个维度对齐的）
        return (self.weight * (x / rms)).to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    def forward(self, x: Float[Tensor, "... d_model"])->Float[Tensor, "... d_ff"]:
        silu_in = self.w1(x)
        silu = silu_in * torch.sigmoid(silu_in)
        return self.w2(silu * self.w3(x))
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 构造频率因子 (每两个维度共享一个频率)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))  # [d_k/2]
        positions = torch.arange(max_seq_len).float()  # [max_seq_len]
        angles = positions[:, None] * freqs[None, :]   # [max_seq_len, d_k/2]

        sin_v = torch.sin(angles)
        cos_v = torch.cos(angles)
        self.register_buffer("cos", cos_v, persistent=False)
        self.cos: torch.Tensor
        self.register_buffer("sin", sin_v, persistent=False)
        self.sin: torch.Tensor

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: [..., seq_len, d_k]
        token_positions: [..., seq_len]
        return: same shape as x
        """
        # 取出对应位置的 sin / cos
        sin = self.sin[token_positions]  # [..., seq_len, d_k/2]
        cos = self.cos[token_positions]  # 同上

        # 拆分最后一个维度 (奇偶维度)
        x1 = x[..., ::2]  # [..., seq_len, d_k/2]
        x2 = x[..., 1::2] # [..., seq_len, d_k/2]

        # 旋转：二维坐标旋转变换.  分组理解此操作。 就会理解他切片的意义
        # print("debug shape", x1.shape, token_positions.shape, cos.shape, sin.shape)
        # result debug shape torch.Size([32, 2, 64, 64]) torch.Size([32, 64, 64]) torch.Size([32, 64, 64])
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        # 合并回原始形状
        return rearrange(x_rotated, "... seq d_pair two -> ... seq (d_pair two)")

def softmax(in_features: Float[Tensor, " ..."], dim: int)-> Float[Tensor,"..."]:
    x_max = torch.max(in_features, dim=dim, keepdim=True).values
    subtracted_x = in_features - x_max
    sum_exp_x = subtracted_x.exp().sum(dim=dim, keepdim=True)
    return subtracted_x.exp() / sum_exp_x


def sdpa(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Scaled Dot Product Attention.
    Q, K, V: [batch_size, num_heads, seq_len, d_k]
    mask: [batch_size, num_heads, seq_len, seq_len] or None
    """
    d_k = Q.shape[-1]
    
    # 计算注意力分数
    scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)

    # 应用掩码
    if mask is not None:
        scores.masked_fill_(mask == 0, float("-inf"))

    # 计算注意力权重
    attn_weights = softmax(scores, dim=-1)

    # 应用注意力权重到值向量
    return einsum(attn_weights, V, " ... queries keys, ... keys d_v -> ... queries d_v")


class mutihead_self_attention(nn.Module):
    "causal multi-head self-attention"
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_projection = Linear(d_model, d_model)
        self.k_projection = Linear(d_model, d_model)
        self.v_projection = Linear(d_model, d_model)
        self.o_projection = Linear(d_model, d_model)

    def forward(
        self, 
        in_features: Float[Tensor, "... sequence_length d_model"]
    ) -> Float[Tensor, "... sequence_length d_model"]:
        Q = self.q_projection(in_features)
        K = self.k_projection(in_features)
        V = self.v_projection(in_features)

        Q = rearrange(Q, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=self.num_heads)
        K = rearrange(K, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=self.num_heads)
        V = rearrange(V, "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v", num_heads=self.num_heads)

        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2]), diagonal=0).bool()

        # sqpa & o_projection
        out = sdpa(Q, K, V, mask=mask)
        out = rearrange(out, "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")
        return self.o_projection(out)
    

class mhsa_rope(nn.Module):
    "causal multi-head self-attention with RoPE"
    def __init__(self, d_model, num_heads, max_seq_len, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_projection = Linear(d_model, d_model)
        self.k_projection = Linear(d_model, d_model)
        self.v_projection = Linear(d_model, d_model)
        self.o_projection = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, max_seq_len)

    def forward(
        self, 
        in_features: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length"]
    ) -> Float[Tensor, "... sequence_length d_model"]:
        Q = self.q_projection(in_features)
        K = self.k_projection(in_features)
        V = self.v_projection(in_features)

        Q = rearrange(Q, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=self.num_heads)
        K = rearrange(K, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=self.num_heads)
        V = rearrange(V, "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v", num_heads=self.num_heads)
        
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2]), diagonal=0).bool()

        out = sdpa(Q, K, V, mask=mask)

        out = rearrange(out, "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")
        return self.o_projection(out)

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.attn = mhsa_rope(d_model, num_heads, max_seq_len, rope_theta=theta)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff) 

    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length"]
    ) -> Float[Tensor, "... sequence_length d_model"]:
        in_features = in_features + self.attn(self.ln1(in_features), token_positions)
        in_features = in_features + self.ffn(self.ln2(in_features))
        return in_features
    

class Transformer_lm(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.context_length = context_length
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        in_indices: Int[Tensor, "batch sequence_length"]
    ) -> Float[Tensor, "batch sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(in_indices.shape[1], device=in_indices.device) # .unsqueeze(0).expand(in_indices.shape[0], -1)
        for layer in self.layers:
            x = layer(x, token_positions)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad
    def generate(
        self,
        in_indices: Int[Tensor, "batch sequence_length"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        
    ) -> Int[Tensor, "batch sequence_length + new_tokens"]:
        """
        自回归生成新 token，支持温度缩放和 top-p 采样。
        
        Args:
            in_indices: [batch, seq_len] - 输入的 token 序列（prompt）
            max_new_tokens: 最大生成的 token 数量
            temperature: 温度参数 τ
                - τ → 0: 接近贪婪解码（选择概率最大的）
                - τ = 1: 标准采样
                - τ > 1: 更随机的采样
            top_p: nucleus sampling 的阈值（例如 0.9）
                - None: 不使用 top-p 采样
                - 0 < p < 1: 只从累积概率达到 p 的最小词汇集中采样
            eos_token_id: 结束符的 token id，遇到时停止生成
                - None: 生成满 max_new_tokens 个 token
        
        Returns:
            生成的完整序列 [batch, original_seq_len + actual_new_tokens]
        """
        finished = torch.zeros(in_indices.shape[0], dtype=torch.bool, device=in_indices.device)  # 记录每个序列是否已生成结束符
        for _ in range(max_new_tokens):
            logits = self.forward(in_indices)  # [batch, seq_len, vocab_size]

            scaled_logits = logits[:,-1,:] / temperature  # [batch, vocab_size, hidden_size]

            probs = softmax(scaled_logits, dim=-1)  # [batch, vocab_size]

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # 创建掩码，保留累积概率 <= top_p 的 token
                top_p_mask = cumulative_probs <= top_p
                # 确保至少保留一个 token
                top_p_mask[..., 0] = True

                # 创建一个与 probs 相同形状的掩码
                full_mask = torch.zeros_like(probs, dtype=torch.bool)
                full_mask.scatter_(dim=-1, index=sorted_indices, src=top_p_mask)

                # 将不在 top-p 范围内的 token 概率设为 0
                filtered_probs = probs.masked_fill(~full_mask, 0.0)
                # 重新归一化概率
                probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            if eos_token_id is not None:
                next_tokens = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_tokens, eos_token_id),
                    next_tokens
                )
                # 更新 finished 标记
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)

            in_indices = torch.cat([in_indices, next_tokens], dim=1)

            if eos_token_id is not None and finished.all():
                break

        return in_indices


def cross_entropy(
    inputs: Float[Tensor, "batch vocab_size"],
    targets: Int[Tensor, "batch"]
) -> Float[Tensor, ""]:
    
    """
    计算交叉熵损失。
    inputs: [batch_size, vocab_size]  模型输出的 logits
    targets: [batch_size]              真实的 token 索引
    返回: 标量平均损失
    """

    # 数值稳定性：减去最大值
    max_logits = inputs.max(dim=-1, keepdim=True).values  # [batch, 1]
    shifted_logits = inputs - max_logits                  # [batch, vocab_size]
    
    # 关键：直接计算 log_softmax，避免先exp再log的数值问题
    # log_softmax = shifted_logits - log(sum(exp(shifted_logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))  # [batch, 1]
    log_softmax = shifted_logits - log_sum_exp  # [batch, vocab_size]
    
    # 选择目标token的log概率
    target_log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [batch]
    # 交叉熵损失 = -log_prob 的平均值
    ce_loss = -target_log_probs.mean()  # 标量
    
    return ce_loss

# def cross_entropy(
#     inputs: Float[Tensor, "batch vocab_size"],
#     targets: Int[Tensor, "batch"]
# ) -> Float[Tensor, ""]:
#     """
#     计算交叉熵损失。
#     inputs: [batch_size, vocab_size]  模型输出的 logits
#     targets: [batch_size]              真实的 token 索引
#     返回: [1]                 avg交叉熵损失
#     """

#     # 计算 softmax 概率
#     print("inputs", inputs)

#     max_logits = inputs.max(dim=-1, keepdim=True).values # [batch, 1]
#     shifted_logits = inputs - max_logits                  # [batch, vocab_size]
#     print("shifted_logits", shifted_logits)
#     exp_logits = shifted_logits.exp() # [batch, vocab_size]
#     print("exp_logits",exp_logits) # 发生下溢
#     sum_exp = exp_logits.sum(dim=-1,keepdim=True) # [batch]
#     print("sum_exp", sum_exp)
#     ce_loss = -torch.log(torch.gather(exp_logits, dim=-1, index=targets.unsqueeze(-1)) / sum_exp).mean()
#     print("ratio", torch.gather(exp_logits, dim=-1, index=targets.unsqueeze(-1)) / sum_exp)
#     return ce_loss