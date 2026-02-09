"""Prompt template for idea improvement (Appendix F.8)."""

from __future__ import annotations


def format_prompt(*, previous_ideas: str, lessons: str, context: str) -> str:
    return f"""\
==== Previous Ideas ====
{previous_ideas}

==== Lessons ====
{lessons}

==== Task ====
Using the insights from the lessons learned during solution development, \
your task is to propose an optimized strategy to solve the problem \
more effectively. You must synthesize the provided "Lessons" to \
propose a structural evolution of the "Previous Ideas".

# Requirements
- Structural Innovation (Exploration): Do not propose trivial \
hyperparameter tuning. You must introduce a fundamental change (e.g., \
a new backbone architecture, a multi-stage pipeline, or a distinct \
feature engineering paradigm) to address identified weaknesses.
- Strategic Retention (Exploitation): Explicitly preserve components \
identified as successful in the "Lessons". Do not discard what is \
already working.
- Computational Budget: The solution is allowed to be moderately heavier \
than previous ideas (e.g., using a stronger backbone), but it must \
remain feasible for standard training environments.
- Citation: Whenever you apply a specific concept or solution from these \
lessons, you must immediately reference it by appending "Cite \
{{lesson_id}}" to the relevant statement.

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
