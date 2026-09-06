from __future__ import annotations

import io
import math
from hashlib import sha256

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nccl_guard.config.core import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: Tensor) -> Tensor:
    """Apply rotary positions to [batch, heads, sequence, head_dim]."""
    _, _, seq, dim = x.shape
    if dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    pos = torch.arange(seq, device=x.device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = torch.outer(pos, inv)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, ffn: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, hidden = x.shape
        shape = (batch, seq, self.heads, self.head_dim)
        q = apply_rope(self.q_proj(x).view(shape).transpose(1, 2))
        k = apply_rope(self.k_proj(x).view(shape).transpose(1, 2))
        v = self.v_proj(x).view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).contiguous().view(batch, seq, hidden))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden)
        self.attn = Attention(config.hidden, config.heads)
        self.mlp_norm = RMSNorm(config.hidden)
        self.mlp = SwiGLU(config.hidden, config.ffn)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.final_norm = RMSNorm(config.hidden)
        self.lm_head = nn.Linear(config.hidden, config.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def canonical_model(config: ModelConfig, seed: int = 17) -> DecoderLM:
    torch.manual_seed(seed)
    return DecoderLM(config)


def state_fingerprint(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def estimate_parameter_bytes(config: ModelConfig, bytes_per_parameter: int = 2) -> int:
    # Embedding+head, attention, SwiGLU, and norms. Empirical peak is authoritative.
    params = 2 * config.vocab_size * config.hidden
    params += config.layers * (
        4 * config.hidden**2 + 3 * config.hidden * config.ffn + 2 * config.hidden
    )
    return math.ceil(params * bytes_per_parameter)


def estimate_activation_bytes(
    config: ModelConfig, batch_size: int, bytes_per_element: int = 2
) -> int:
    if batch_size <= 0 or bytes_per_element <= 0:
        raise ValueError("batch and element size must be positive")
    # Conservative planning estimate for residuals, Q/K/V, attention output, and MLP intermediates.
    elements_per_layer = batch_size * config.sequence_length * (6 * config.hidden + 2 * config.ffn)
    return math.ceil(config.layers * elements_per_layer * bytes_per_element)


def fits_memory_budget(
    config: ModelConfig, batch_size: int, free_bytes: int, headroom: float = 0.80
) -> bool:
    if free_bytes <= 0 or not 0 < headroom <= 1:
        raise ValueError("invalid memory budget")
    # Parameters, gradients, and AdamW master/moment state plus activations.
    estimate = 8 * estimate_parameter_bytes(config, 2) + estimate_activation_bytes(
        config, batch_size
    )
    return estimate <= free_bytes * headroom
