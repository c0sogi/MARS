"""Prompt template for execution result review (Appendix F.16)."""

from __future__ import annotations


def format_prompt(*, library_files: str, code: str, term_out: str) -> str:
    return f"""\
==== Python Files ====
The following Python files are already provided.
{library_files}

==== Task ====
Your task is to evaluate the output of the code execution for the \
provided code and report the empirical findings. The review must be \
submitted in a specific JSON format with the following fields:
- summary (string): In this field, provide a brief summary describing \
the empirical findings. This must include:
    - The training loss trend (e.g., did it converge/minimize?).
    - Failure analysis.
    - The final validation metric.
    - The reasoning for your `valid_metric` assessment (e.g., "The final \
validation metric is valid," or "The final validation metric is \
invalid due to validation data leakage...").
- metric (number or null): Report the value of the validation metric here.\
 You must convert percentages to decimals (e.g., 95% -> 0.95). This \
should be null if the metric cannot be found or determined.
- valid_metric (boolean): Set to `true` if the final validation metric is\
 valid. Set to `false` if any of the following conditions are met:
    - The computed final validation metric does not match the one defined \
in the Task Description.
    - The final validation metric is calculated incorrectly.
    - The final validation metric is not computed on the entire hold-out \
validation set.
    - There are signs of validation data leakage (e.g., the validation \
set was used in training).

Code:
```
{code}
```

Execution Output:
{term_out}

# Response Format
The review must be submitted in the following JSON format in a single \
markdown code block (wrapped in ```):
```json
{{
    "summary": "The code trains a model to solve the task... The final \
validation metric is ...",
    "metric": 0.99,
    "valid_metric": true
}}
```
"""
