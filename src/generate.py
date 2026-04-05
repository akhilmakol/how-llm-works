from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniGPT, MiniGPTConfig
from src.tokenizer import WordTokenizer


def load_artifacts(model_path: Path | str | None = None) -> Tuple[MiniGPT, WordTokenizer, dict]:
    model_path = Path(model_path) if model_path else PROJECT_ROOT / "model.pth"
    checkpoint = torch.load(model_path, map_location="cpu")
    tokenizer = WordTokenizer.from_state(checkpoint["tokenizer"])
    config = MiniGPTConfig(**checkpoint["config"])
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, checkpoint


def generate_text(prompt: str, model_path: Path | str | None = None, max_new_tokens: int = 20) -> str:
    model, tokenizer, _ = load_artifacts(model_path)
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [tokenizer.unk_id]

    for _ in range(max_new_tokens):
        context = encoded[-model.config.block_size :]
        x = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        logits, _ = model(x)
        next_token_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        encoded.append(next_token_id)

    return tokenizer.decode(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained mini GPT model.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "model.pth"))
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(generate_text(args.prompt, model_path=args.model, max_new_tokens=args.max_new_tokens))
