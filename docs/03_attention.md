# 3. Attention

Attention is the mechanism that made transformers successful.

## Intuition

When predicting the next token, not every earlier token matters equally.

In the phrase:

`the animal did not cross the road because it was tired`

the word `it` should probably pay attention to `animal`.

## Query, Key, Value

Each token is transformed into three vectors:

- Query: what this token is looking for
- Key: what this token offers
- Value: the information this token carries

The model compares queries and keys to decide how much attention to place on each earlier token.

## Scaled Dot-Product Attention

The score is roughly:

`Q x K^T / sqrt(d)`

Then we apply softmax to convert scores into probabilities.

## Causal Masking

GPT-style models must not look into the future while training.

So token 4 can attend to tokens 1 to 4, but not token 5 or later.

## Example

For:

`the cat sat`

the token `sat` may attend strongly to `cat` and `the`.
