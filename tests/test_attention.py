import unittest

import torch

from src.attention import CausalSelfAttention


class AttentionTests(unittest.TestCase):
    def test_attention_output_shape(self) -> None:
        attention = CausalSelfAttention(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        context, weights = attention(x)
        self.assertEqual(tuple(context.shape), (2, 5, 16))
        self.assertEqual(tuple(weights.shape), (2, 4, 5, 5))


if __name__ == "__main__":
    unittest.main()
