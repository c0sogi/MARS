"""Prompt template for bug analysis (Appendix F.14)."""

from __future__ import annotations


def format_prompt(*, lessons: str, files: str, exec_result: str) -> str:
    return f"""\
==== Debug Lessons ====
{lessons}

==== Python Files ====
The following Python files are already provided. Do not modify them.
{files}

==== Task ====
You are tasked with debugging a script failure. You should summarize the \
execution traceback and explain the root cause of the errors. You \
should apply the relevant knowledge provided in the Debug Lessons \
section to support your diagnosis. Whenever you apply a specific \
concept or solution from these lessons, you must immediately reference \
it by appending "Cite {{lesson_id}}" to the relevant statement. You \
can use Google Search as needed for debugging.

Execution Traceback (`python runfile.py`):
{exec_result}
"""
