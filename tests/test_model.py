import unittest

import torch

from src.model import MiniGPT, MiniGPTConfig


class ModelTests(unittest.TestCase):
    def test_forward_pass_shape(self) -> None:
        config = MiniGPTConfig(vocab_size=30, block_size=8, d_model=16, n_heads=4, n_layers=2, d_ff=32)
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (3, config.block_size))
        logits, _ = model(x)
        self.assertEqual(tuple(logits.shape), (3, config.block_size, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
