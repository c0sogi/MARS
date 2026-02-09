"""Prompt template for lesson deduplication (Appendix F.19)."""

from __future__ import annotations


def format_prompt(*, existing_lessons: str, new_lesson: str) -> str:
    return f"""\
You are a Machine Learning Engineer responsible for maintaining a \
knowledge base of technical lessons.

==== Existing Lessons ====
{existing_lessons}

==== New Lesson ====
{new_lesson}

==== Task ====
Your task is to determine if the **New Lesson** is semantically \
equivalent to any of the **Existing Lessons**.

### Guidelines
- **Semantic Overlap:** A lesson is a duplicate if the core insight, \
principle, or actionable advice is effectively the same, even if the \
wording differs.
- **Subsets:** If the **New Lesson** is fully covered by a broader \
existing lesson, count it as a duplicate.
- **Novelty:** If the **New Lesson** provides a specific nuance, edge \
case, or context not covered by existing lessons, it is **not** a \
duplicate.

# Response Format
Provide your analysis in a single valid JSON object inside a single \
markdown code block.

**Fields:**
- `reasoning` (string): Briefly explain your decision. If a duplicate \
exists, explicitly quote or summarize the specific existing lesson \
that overlaps.
- `duplicate` (boolean): Use `true` if it is a duplicate, `false` \
otherwise.

**Example Output:**
```json
{{
    "reasoning": "The lesson is the same as one of the existing lessons.",

    "duplicate": true
}}
```
"""
