"""Prompt template for module testing (Appendix F.11)."""

from __future__ import annotations


def format_prompt(*, library_files: str) -> str:
    return f"""\
==== Python Files ====
The following Python files are already provided. Do not modify them.
{library_files}

==== Task ====
Your task is to write code examples demonstrating how to instantiate and \
utilize the classes or functions from the provided Python files.

# Requirements
- Optimize for Speed: Limit hyperparameters (e.g., reduce the number of \
epochs/steps, use a smaller dataset subset) to ensure the \
demonstration executes quickly.
- Verify Logic: Include assertions or validation steps to confirm the \
correctness of complex classes and functions. \
You may skip verification for trivial components, such as configuration \
classes.
"""
