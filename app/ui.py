from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import generate_text, load_artifacts
from src.tokenizer import WordTokenizer
from src.train import train_model

MODEL_PATH = PROJECT_ROOT / "model.pth"
DATA_PATH = PROJECT_ROOT / "data" / "sample.txt"


@st.cache_resource(show_spinner=False)
def ensure_model():
    if not MODEL_PATH.exists():
        train_model(data_path=DATA_PATH, output_path=MODEL_PATH, epochs=120, learning_rate=3e-3)
    return load_artifacts(MODEL_PATH)


def render_attention_heatmap(model, tokenizer: WordTokenizer, text: str) -> None:
    tokens = tokenizer.tokenize(text)
    if not tokens:
        st.info("Enter some text to visualize attention.")
        return

    encoded = tokenizer.encode(text)
    encoded = encoded[: model.config.block_size]
    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    _, attention_maps = model(x, return_attn=True)
    attention = attention_maps[-1][0, 0].detach().cpu().numpy()
    token_labels = tokenizer.decode(encoded).split()

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(attention, cmap="viridis")
    ax.set_title("Last Layer, Head 1 Attention")
    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_yticklabels(token_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="How LLM Works", page_icon="🧠", layout="wide")
    st.title("How LLM Works")
    st.caption("A visual mini-GPT playground for learning tokenization, attention, training, and generation.")

    with st.spinner("Loading model artifacts..."):
        model, tokenizer, checkpoint = ensure_model()

    st.sidebar.header("Model Snapshot")
    st.sidebar.write(f"Vocabulary size: {checkpoint['config']['vocab_size']}")
    st.sidebar.write(f"Context window: {checkpoint['config']['block_size']}")
    st.sidebar.write(f"Layers: {checkpoint['config']['n_layers']}")
    st.sidebar.write(f"Heads: {checkpoint['config']['n_heads']}")

    tab_generate, tab_tokens, tab_attention = st.tabs(
        ["Text Generation", "Tokenization Explorer", "Attention Visualization"]
    )

    with tab_generate:
        st.subheader("Generate text")
        prompt = st.text_input("Prompt", value="Large language models")
        max_new_tokens = st.slider("Tokens to generate", min_value=5, max_value=40, value=20)
        if st.button("Generate", type="primary"):
            generated = generate_text(
                prompt=prompt,
                model_path=MODEL_PATH,
                max_new_tokens=max_new_tokens,
            )
            st.text_area("Generated output", value=generated, height=180)

    with tab_tokens:
        st.subheader("Explore word-level tokenization")
        raw_text = st.text_area(
            "Input text",
            value="Transformers read text as tokens, not as raw characters.",
            height=120,
        )
        tokens = tokenizer.tokenize(raw_text)
        token_ids = tokenizer.encode(raw_text)
        st.write("Tokens")
        st.code(str(tokens))
        st.write("Token IDs")
        st.code(str(token_ids))
        st.write("Decoded text")
        st.code(tokenizer.decode(token_ids))

    with tab_attention:
        st.subheader("Inspect causal self-attention")
        attention_text = st.text_input(
            "Text for attention map",
            value="attention lets each token look at earlier tokens",
        )
        st.caption("This heatmap uses real model attention from the final transformer layer.")
        render_attention_heatmap(model, tokenizer, attention_text)


if __name__ == "__main__":
    main()
