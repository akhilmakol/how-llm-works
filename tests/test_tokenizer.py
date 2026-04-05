import unittest

from src.tokenizer import WordTokenizer


class TokenizerTests(unittest.TestCase):
    def test_encode_returns_expected_ids(self) -> None:
        tokenizer = WordTokenizer().fit("Hello world.")
        encoded = tokenizer.encode("Hello world.")
        self.assertEqual(encoded, [2, 3, 4])
        self.assertEqual(tokenizer.decode(encoded), "hello world.")


if __name__ == "__main__":
    unittest.main()
