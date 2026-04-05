from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniGPT, MiniGPTConfig
from src.tokenizer import WordTokenizer


def build_sequences(token_ids: List[int], block_size: int) -> List[Tuple[List[int], List[int]]]:
    sequences: List[Tuple[List[int], List[int]]] = []
    if len(token_ids) <= block_size:
        raise ValueError("Training data is too short for the selected block size.")
    for start in range(len(token_ids) - block_size):
        x = token_ids[start : start + block_size]
        y = token_ids[start + 1 : start + block_size + 1]
        sequences.append((x, y))
    return sequences


def train_model(
    data_path: Path | str | None = None,
    output_path: Path | str | None = None,
    epochs: int = 150,
    learning_rate: float = 3e-3,
    batch_size: int = 8,
) -> Path:
    data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "sample.txt"
    output_path = Path(output_path) if output_path else PROJECT_ROOT / "model.pth"

    text = data_path.read_text(encoding="utf-8")
    tokenizer = WordTokenizer().fit(text)
    token_ids = tokenizer.encode(text)

    block_size = min(16, max(4, len(token_ids) - 2))
    sequences = build_sequences(token_ids, block_size=block_size)

    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    model = MiniGPT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    random.seed(7)
    torch.manual_seed(7)

    for epoch in range(epochs):
        random.shuffle(sequences)
        total_loss = 0.0

        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            x = torch.tensor([item[0] for item in batch], dtype=torch.long)
            y = torch.tensor([item[1] for item in batch], dtype=torch.long)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            average_loss = total_loss / max(1, len(sequences) / batch_size)
            print(f"epoch={epoch + 1} loss={average_loss:.4f}")

    checkpoint = {
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_state(),
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mini GPT model.")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "sample.txt"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "model.pth"))
    parser.add_argument("--epochs", type=int, default=150)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_path = train_model(data_path=args.data, output_path=args.output, epochs=args.epochs)
    print(f"Saved model to {saved_path}")
