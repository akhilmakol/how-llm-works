from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, weights = self.attention(self.ln_1(x))
        x = x + attention_output
        x = x + self.feed_forward(self.ln_2(x))
        return x, weights
