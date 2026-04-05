# How LLM Works

`how-llm-works` is a visual and interactive educational repository that explains how a GPT-style large language model works from first principles.

The project combines:

- a clean mini GPT implementation in PyTorch
- beginner-friendly documentation with simple examples
- a Streamlit app for text generation, token exploration, and attention visualization
- a tiny end-to-end training and inference pipeline that runs on a sample corpus

## Repository Overview

```text
how-llm-works/
├── app/         # Streamlit interface
├── src/         # Core tokenizer, attention, transformer, training, generation
├── docs/        # Learning notes from intuition to real-world LLMs
├── visuals/     # Text placeholders describing diagrams to design later
├── data/        # Sample training corpus
├── notebooks/   # Small exploratory notebooks
├── tests/       # Unit tests using unittest
└── assets/      # Portfolio and publishing assets
```

## Architecture

This project implements a minimal GPT-style language model:

1. Text is split into word-level tokens.
2. Tokens are mapped to vectors with token embeddings.
3. Positional embeddings tell the model where each token appears.
4. Causal self-attention lets each token look back at previous tokens.
5. Transformer blocks refine representations using attention, feed-forward layers, residual connections, and layer normalization.
6. A final linear layer predicts the next token.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training

```bash
python src/train.py
```

## Run Generation

```bash
python src/generate.py --prompt "large language models"
```

## Run The Streamlit Demo

```bash
streamlit run app/ui.py
```

The app includes:

- `Text Generation`
- `Tokenization Explorer`
- `Attention Visualization`

If `model.pth` does not exist, the app automatically trains a fresh model before loading the interface.

## Visual References

The `visuals/` directory contains text placeholders that describe the diagrams to create later:

- `cover.png`
- `pipeline.png`
- `tokenization.png`
- `embeddings.png`
- `attention.png`
- `transformer.png`
- `training.png`
- `inference.png`

## Testing

```bash
python -m unittest discover -s tests -v
```

## Learning Path

1. [docs/01_intro.md](/C:/dev/how-llm-works/docs/01_intro.md)
2. [docs/02_tokens_embeddings.md](/C:/dev/how-llm-works/docs/02_tokens_embeddings.md)
3. [docs/03_attention.md](/C:/dev/how-llm-works/docs/03_attention.md)
4. [docs/04_transformer.md](/C:/dev/how-llm-works/docs/04_transformer.md)
5. [docs/05_training.md](/C:/dev/how-llm-works/docs/05_training.md)
6. [docs/06_inference.md](/C:/dev/how-llm-works/docs/06_inference.md)
7. [docs/07_limitations.md](/C:/dev/how-llm-works/docs/07_limitations.md)
8. [docs/08_real_world_llms.md](/C:/dev/how-llm-works/docs/08_real_world_llms.md)

## License

This repository is released under the MIT License.
