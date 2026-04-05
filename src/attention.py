from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention for autoregressive decoding."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.shape

        q = self.query(x).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        return self.output(context), weights
