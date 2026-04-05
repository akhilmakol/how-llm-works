# How LLMs Work From First Principles

Large Language Models can feel mysterious because most explanations either stay too abstract or jump straight into massive production systems. This project takes a different route: build a tiny GPT-style model that still keeps the important architectural ideas intact.

The repository covers the entire pipeline:

- tokenization
- embeddings
- scaled dot-product self-attention
- transformer blocks
- next-token training
- autoregressive generation

It also includes a Streamlit app so readers can move from code to intuition. You can type a prompt, inspect tokens, and see a visual attention map from the final transformer layer.
