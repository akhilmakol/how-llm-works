# 5. Training

![Training diagram](../visuals/training.png)

Training teaches the model to predict the next token.

## Training Data

The sample text is tokenized into token IDs. We then build many short input-target pairs.

Example:

Input:

`["net", "interest", "margin"]`

Target:

`["interest", "margin", "improved"]`

Each target token is just the next token shifted by one position.

## Loss Function

We use cross-entropy loss.

This loss becomes smaller when the model assigns higher probability to the correct next token.

## Optimization

The optimizer adjusts model weights a little after each batch to reduce loss.

Over time, the model becomes better at predicting likely continuations.

## Example

If the model predicts `capital` when the correct token is `margin`, the loss increases and backpropagation nudges the weights toward better predictions.
