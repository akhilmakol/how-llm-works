from __future__ import annotations

import torch
from torch import nn


class TokenPositionalEmbedding(nn.Module):
    """Combines token embeddings with learned positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int, block_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = x.shape
        positions = torch.arange(sequence_length, device=x.device)
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        return token_vectors + position_vectors
