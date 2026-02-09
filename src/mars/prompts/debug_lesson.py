"""Prompt template for debugging lesson distillation (Appendix F.18)."""

from __future__ import annotations


def format_prompt(
    *,
    source_files: str,
    source_exec_result: str,
    source_error_analysis: str,
    diff: str,
    final_exec_result: str,
) -> str:
    return f"""\
You are an expert Python debugger and instructor. Your task is to analyze \
a debugging attempt and distill a high-value "Lesson Learned".

# Input
Initial State:
{source_files}

Initial Execution Traceback:
{source_exec_result}

Initial Error analysis:
{source_error_analysis}

Attempted Fix (Diff):
{diff}

Execution Traceback after applying the fix:
{final_exec_result}

# Guidelines
- Determine if the Attempted Fix resolved the Initial Error based on the \
Result of Fix.
- If the fix SUCCEEDED: Explain the root cause of the initial error and \
why this specific fix was the correct solution.
- If the fix FAILED: Explain why the attempted fix was insufficient, \
incorrect, or introduced new issues. The lesson must focus on avoiding \
this specific pitfall.

# Response Format
- Title: A concise, imperative, and memorable summary of the lesson.
- Explanation: A clear paragraph synthesizing the error context. Describe \
the specific mechanism of the failure and the logic required to fix \
it.
- Detection: How to identify this issue in the future. List specific \
signals, such as particular Exception types, stack trace patterns, or \
code smells.
"""
