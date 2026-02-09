"""Prompt template for modular decomposition (Appendix F.9)."""

from __future__ import annotations


def format_prompt(*, idea: str, context: str) -> str:
    return f"""\
==== Idea ====
{idea}

==== Task ====
Your task is to design a modular repository structure to implement the \
given idea. Do not generate the full code yet; focus on the natural \
description of the **architectural logic**.

# Requirements
- **Modularity:** Break the solution into logical modules based on \
functionality (e.g., data handling, core training and evaluation logic\
, utilities).
- **Entry Point:** You must include a `main` module that acts as the \
entry point to execute the end-to-end pipeline.
- **Detail:** For each module, the description must include:
    - The purpose of the module.
    - The names of specific classes or functions to be implemented.
    - A brief description of the implementation logic.
    - A brief explanation of how this module interacts with others.
- **Ordering:** The JSON output must be ordered topologically (\
dependencies first, dependent modules last).

# Response Format
Provide the output strictly as a JSON object in a single markdown code \
block. The keys should be the module names and the values should be \
the detailed descriptions. The module name must not include the `.py` \
extension.

Example Format:
```json
{{
    "module_name": "Implements [Specific Class] to handle [Specific Task].\
 Includes functions like [func_a] and [func_b].",
    "main": "Orchestrates the workflow. Imports DataLoader from the data \
module and Model from the model module to run the pipeline."
}}
```

{context}
"""
