"""Prompt template for metadata documentation (Appendix F.4)."""

from __future__ import annotations


def format_prompt(*, code: str, term_out: str) -> str:
    return f"""\
==== Task ====
Your task is to analyze the provided Python script and its execution \
output to create clear documentation for each file generated in the \
`./metadata` directory.

# Python Script
{code}

# Execution Output
{term_out}

# Requirements
For each file generated in the `./metadata` directory, provide a detailed \
breakdown covering:
- Content and Purpose:
    - Describe the information or data contained within the file (e.g., "\
Contains image_id, file_path, and label for the training set.").
    - Explain its primary purpose (e.g., "This file is used by the data \
loader to find image files and match them with their correct labels.").

- Schema / Structure: Detail the structure, such as column names, data \
types, and an example row if applicable.
- Loading Method: Explain the standard method or library function \
required to load this file (e.g., "Load with pandas.read_csv()" or "\
Load with joblib.load()").
"""
