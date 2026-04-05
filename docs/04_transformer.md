# 4. Transformer Blocks

![Transformer block diagram](../visuals/transformer.png)

A transformer block combines attention with a small neural network called a feed-forward network.

## Main Parts

Each block contains:

- layer normalization
- self-attention
- residual connection
- feed-forward network
- another residual connection

## Why Residual Connections Help

Residual connections let the model keep older information while adding new transformed information.

This makes deep networks easier to train.

## Flow

1. Normalize the input
2. Run causal self-attention
3. Add the attention output back to the input
4. Normalize again
5. Run the feed-forward network
6. Add that output back too

## Example

If the input token representation captures a word identity, the attention layer can add context and the feed-forward layer can refine it further.
