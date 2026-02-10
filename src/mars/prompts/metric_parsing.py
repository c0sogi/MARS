"""Prompt template for metric parsing (Appendix F.1)."""

from __future__ import annotations


def format_prompt(*, task_description: str) -> str:
    return f"""\
==== Task ====
Your task is to analyze the provided problem description to identify the \
primary evaluation metric and determine if a lower score indicates \
better performance. Your response must be in a specific JSON format \
with the following fields:
- metric_name (string): This field specifies the name of the primary \
evaluation metric.
- lower_is_better (boolean): This field indicates whether the metric \
should be minimized. If a lower value of the metric represents better \
performance (e.g., for Mean Squared Error), set this to true. If a \
higher value represents better performance (e.g., for accuracy), set \
this to false.

# Response Format
Your response should be in the following JSON format in a single markdown \
code block (wrapped in ```):
```json
{{
    "metric_name": "accuracy",
    "lower_is_better": false
}}
```

# Problem Description
{task_description}
"""
