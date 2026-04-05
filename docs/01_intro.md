# 1. Introduction

![How LLM Works overview](../visuals/cover.png)

Large Language Models, or LLMs, are systems that learn to predict the next token in a sequence.

That sounds simple, but it becomes powerful when the model sees lots of text and learns patterns such as:

- grammar
- common facts
- formatting styles
- reasoning-like token patterns

## The Core Idea

If the model sees:

`a bank earns`

it tries to predict a likely next token such as:

`interest`

During training, the model makes millions of these guesses and improves when it is wrong.

## Why This Repository Exists

Real LLMs are large and complex. This project uses a tiny GPT-style model so you can understand:

- how text becomes tokens
- how tokens become vectors
- how attention works
- how the transformer predicts the next token

## Example

Input:

`banks collect deposits`

Possible next token:

`and`

The model does not understand language like a human. It learns statistical patterns from data.
