"""Prompt template for initial idea proposal (Appendix F.7)."""

from __future__ import annotations


def format_prompt(*, model_arch_desc: str, previous_ideas: str, context: str) -> str:
    return f"""\
==== Model Architectures ====
{model_arch_desc}

==== Previous Ideas ====
{previous_ideas}

==== Task ====
Your task is to propose a highly efficient **baseline approach** to solve \
the problem.

# Requirements
- Novelty: The proposed solution must remain strictly distinct from the \
approaches listed in Previous Ideas.
- Model Design: Synthesize a simple and lightweight architecture using \
the provided Model Architectures as a conceptual foundation. Ensure \
the design is unique and has not been suggested in the Previous Ideas.
- Philosophy: Prioritize speed and simplicity over maximum accuracy. \
Exclude resource-intensive techniques, such as heavy augmentations or \
ensembles, to establish a reliable performance baseline.

# Response Format
Your solution must be outlined in natural language without using code or \
specific implementation details. Your response should cover the \
following aspects:
- Model: Describe the model architecture's design and key components.
- Data: Describe the necessary steps to preprocess data for both training \
and evaluation.
- Training: Outline the training procedure, including key techniques (e.g\
., loss functions, optimizers, or training strategies).
- Evaluation: Describe the process for generating predictions on the test \
data.

{context}
"""
