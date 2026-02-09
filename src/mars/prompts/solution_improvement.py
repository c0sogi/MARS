"""Prompt template for solution improvement (Appendix F.13)."""

from __future__ import annotations


def format_prompt(*, lessons: str, previous_solution: str) -> str:
    return f"""\
==== Lessons ====
{lessons}

==== Previous Solution ====
{previous_solution}

==== Task ====
Your task is to modify the Python files from the previous solution to \
optimize performance.

# Requirements
- Modifications must be targeted and specific (ablation-style). Do not \
rewrite the entire solution; focus on isolating and improving specific \
aspects.
- You should apply the relevant knowledge provided in the Lessons section \
to support your optimization strategy. Whenever you apply a specific \
concept or solution from these lessons, you must immediately reference \
it by appending "Cite {{lesson_id}}" to the relevant statement.
- Optimize hyperparameter settings (e.g., training steps, learning rate, \
batch size) to strike the best balance between predictive performance \
and execution speed.
- **Do not remove** the following core logic from the original `runfile.\
py` script:
    - Print the final validation metric computed on the entire hold-out \
validation set.
    - Perform failure analysis on the trained model.
    - Generate predictions for the entire test set and create the \
submission file{{submission_cond}}.

# Response Format
For each file you want to modify, provide the changes in the following \
diff format:

[target file: filename.py]
<<<< SEARCH
old code block to find
====
new code block to replace with
>>>> REPLACE

You can provide multiple diff blocks for different files or multiple \
changes to the same file.
"""
