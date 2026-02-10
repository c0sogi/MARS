"""Prompt template for validation dataset verification (Appendix F.3)."""

from __future__ import annotations


def format_prompt(*, code: str, term_out: str) -> str:
    return f"""\
==== Task ====
Analyze the provided Python script and its execution output to verify if \
the validation dataset was handled or created successfully.

# Python Script
{code}

# Execution Output
{term_out}

# Requirements
You must review the script and output based on the criteria below. Your \
entire response must be a single JSON code block.

- Success Criteria: The success field must be set to True if one of the \
following two conditions is met. Otherwise, set it to False.
  1. Existing Validation Set: The script correctly identifies that a \
separate validation dataset is already available in the raw data (i.e\
., no new split is required).
  2. Created Validation Set: The script correctly creates a new \
validation set by splitting the training data. \
Your analysis must confirm that the script's logic properly attempts to \
create a representative split (e.g., by using stratified or group \
sampling).
- JSON Response Format: Provide your review in the following JSON format.
    - analysis (string): A concise rationale for your decision.
        - If successful: Explain which of the two success criteria was \
met.
        - If failed: Briefly explain why the script failed to meet either \
criterion (e.g., "The script split the data randomly instead of using \
stratification.").
    - success (bool): True if the validation dataset was handled or \
created successfully, False otherwise.

# Response Format
The review must be submitted in the following JSON format in a single \
markdown code block (wrapped in ```):
```json
{{
    "analysis": "The validation dataset was not created successfully. The \
script split the training data but did not use stratified sampling, \
failing to create a representative sample.",
    "success": false
}}
```
"""
