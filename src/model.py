from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.embeddings import TokenPositionalEmbedding
from src.transformer import TransformerBlock


@dataclass
class MiniGPTConfig:
    vocab_size: int
    block_size: int = 16
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenPositionalEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            block_size=config.block_size,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        hidden = self.embedding(x)
        attention_maps: List[torch.Tensor] = []

        for block in self.blocks:
            hidden, weights = block(hidden)
            if return_attn:
                attention_maps.append(weights)

        logits = self.lm_head(self.final_norm(hidden))

        if targets is None:
            return logits, attention_maps if return_attn else None

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss
