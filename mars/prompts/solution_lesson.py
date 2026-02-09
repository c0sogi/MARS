"""Prompt template for solution lesson distillation (Appendix F.17)."""

from __future__ import annotations


def format_prompt(*, best_solution: str, new_solution: str) -> str:
    return f"""\
==== Current Best Solution ====
{best_solution}

==== New Solution ====
{new_solution}

==== Task ====
Your task is to analyze the provided solutions to distill a high-value "\
Lesson Learned".

# Guidelines
- **Check Context:**
    - If *Current Best Solution* exists: Comparative Analysis. Contrast \
the algorithmic approach of the New vs. Current. Explain precisely *\
why* the New Solution improves or degrades performance based on the \
execution results.
    - If *Current Best Solution* is missing: Empirical Analysis. Summarize\
 the findings and effectiveness of the New Solution based on its \
execution results.
- **Logic over Syntax:** Focus on algorithmic choices, data structures, \
and architectural decisions. Ignore minor syntactic sugar unless it \
affects performance.
- **Causal Chain:** Trace the logic to prove exactly how the new approach \
resolves the specific bottleneck.
- **Generalizability:** The final lesson must be abstract enough to apply \
to similar problems in the future, not just this specific snippet.

# Response Format
- Title: A clear, memorable title for the lesson.
- Summary: A brief, high-level overview of the methods or algorithmic \
changes applied in the New Solution.
- Empirical Findings: Analysis of the execution results. If comparing, \
highlight the delta in performance (validation metric and execution \
time) and the specific trade-offs observed.
- Key Lesson: A standalone, actionable principle. Write this as a \
heuristic or rule of thumb (e.g., "When handling sparse matrices, \
prefer X over Y because..."). If a developer reads *only* this \
paragraph, they should learn a technique to apply in their own work.
"""
