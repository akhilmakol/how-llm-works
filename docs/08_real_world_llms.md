# 8. Real-World LLMs

![Pipeline diagram](../visuals/pipeline.png)

Modern production LLMs build on the same basic transformer ideas, but at much larger scale.

## What Changes In Real Systems

- much larger vocabularies
- subword tokenization instead of simple word splitting
- many more layers and heads
- massive datasets
- distributed training across many GPUs
- improved optimization and safety systems

## Extra Techniques

Real-world systems often include:

- instruction tuning
- reinforcement learning from human feedback
- retrieval augmentation
- quantization for deployment
- system prompts and tool use

## Example Comparison

This project:

- one tiny sample text
- a small CPU-friendly GPT
- greedy decoding

A production LLM:

- internet-scale or curated large corpora
- billions of parameters
- sophisticated decoding and alignment methods

The core transformer intuition still carries over.
