# How LLMs Work From First Principles

Large Language Models often feel difficult to approach because explanations tend to fall into two extremes. Some stay too abstract and never show the mechanics. Others jump straight into giant production-scale systems that are hard to follow if you are still building intuition.

`how-llm-works` was created to take a more practical middle path.

The idea behind the project is simple: build a small GPT-style model that is architecturally correct enough to teach the real concepts, while still being compact enough to read, run, and experiment with.

The repository walks through the full pipeline:

- tokenization
- embeddings
- causal self-attention
- transformer blocks
- next-token training
- autoregressive generation

It also includes visual diagrams, step-by-step documentation, exploratory notebooks, and a Streamlit interface so readers can move from theory to interaction.

## Why This Project Exists

Many people want to understand LLMs, but the learning path can feel fragmented. One tutorial explains embeddings. Another explains attention. A third shows a code implementation, but without the intuition behind it.

This project was designed to bring those layers together in one place:

- code for implementation
- visuals for intuition
- docs for explanation
- UI for interaction

That combination makes it especially useful for self-learners, educators, and technical teams building AI literacy.

## What Makes It Different

Instead of using random example text everywhere, the project uses a banking fundamentals scenario across the learning flow. Concepts like deposits, loans, liquidity, credit risk, capital ratios, and interest income make the examples more concrete and realistic.

That matters because domain-grounded examples are often easier to reason about than generic placeholder text.

## What You Can Learn From It

By working through the repository, a learner can understand:

- how text becomes token IDs
- how embeddings create usable vector representations
- how attention decides which earlier tokens matter
- how transformer blocks repeatedly refine context
- how training turns prediction errors into parameter updates
- how inference generates text one token at a time

Even though the model is intentionally small, the conceptual structure carries over to much larger systems.

## Why Open Source Educational AI Matters

Projects like this are valuable beyond code alone. They help make AI systems more understandable, auditable, and teachable. That is important not only for developers, but also for students, domain experts, policy stakeholders, and organizations trying to build responsible AI capability.

In that sense, educational AI tooling is not just a learning exercise. It is part of building a healthier AI ecosystem.

## Explore The Repository

Repository:
https://github.com/akhilmakol/how-llm-works/tree/master

If you are learning transformers, teaching LLM fundamentals, or building practical AI education experiences, this repository is meant to be a useful starting point.
