from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WordTokenizer:
    """A tiny word-level tokenizer with special tokens."""

    special_tokens: List[str] = field(default_factory=lambda: ["<pad>", "<unk>"])

    def __post_init__(self) -> None:
        self.token_to_id: Dict[str, int] = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_token: Dict[int, str] = {idx: token for idx, token in enumerate(self.special_tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def fit(self, text: str) -> "WordTokenizer":
        for token in self.tokenize(text):
            if token not in self.token_to_id:
                index = len(self.token_to_id)
                self.token_to_id[token] = index
                self.id_to_token[index] = token
        return self

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        text = " ".join(tokens)
        for punctuation in [".", ",", "!", "?", ";", ":"]:
            text = text.replace(f" {punctuation}", punctuation)
        return text

    def to_state(self) -> Dict[str, object]:
        return {
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "WordTokenizer":
        tokenizer = cls(special_tokens=list(state["special_tokens"]))
        tokenizer.token_to_id = {str(token): int(idx) for token, idx in dict(state["token_to_id"]).items()}
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
        return tokenizer
