# 5. Training

Training teaches the model to predict the next token.

## Training Data

The sample text is tokenized into token IDs. We then build many short input-target pairs.

Example:

Input:

`["large", "language", "models"]`

Target:

`["language", "models", "learn"]`

Each target token is just the next token shifted by one position.

## Loss Function

We use cross-entropy loss.

This loss becomes smaller when the model assigns higher probability to the correct next token.

## Optimization

The optimizer adjusts model weights a little after each batch to reduce loss.

Over time, the model becomes better at predicting likely continuations.

## Example

If the model predicts `birds` when the correct token is `models`, the loss increases and backpropagation nudges the weights toward better predictions.
