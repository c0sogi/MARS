"""Prompt template for debugging (Appendix F.15)."""

from __future__ import annotations


def format_prompt(
    *,
    lessons: str,
    files: str,
    exec_result: str,
    error_analysis: str,
) -> str:
    return f"""\
==== Debug Lessons ====
{lessons}

==== Python Files ====
The following Python files are already provided. Do not modify them.
{files}

==== Task ====
We ran this command (`python runfile.py`) and got some errors.

Execution Traceback:
{exec_result}

Error analysis:
{error_analysis}

Your task is to revise the given Python files to fix the errors based on \
the provided error analysis. You can use Google Search as needed for \
debugging.

# Requirements
- You should write a brief natural language description of what the issue \
in the previous implementation is and how the issue can be fixed.
- The fix must be targeted. Do not change the core logic or intended \
functionality of the original code; only correct the specific \
implementation error shown in the Execution Traceback.
- You should apply the relevant knowledge provided in the Debug Lessons \
section to guide your fixes. Whenever you apply a specific concept or \
solution from these lessons, you must immediately reference it by \
appending "Cite {{lesson_id}}" to the relevant statement.
- Do not use `try...except` blocks to catch, suppress, or ignore the \
original error. The fix must address the root cause of the problem.

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
