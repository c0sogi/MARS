"""Prompt template for metadata generation (Appendix F.2)."""

from __future__ import annotations


def format_prompt(*, task_description: str, exec_timeout: int) -> str:
    return f"""\
==== Task ====
Your task is to write a Python script that generates three metadata files \
for the training, validation, and test datasets respectively. This \
metadata (e.g., sample IDs, file paths, labels) will be used by other \
scripts to load data efficiently.

# Requirements
- The script's only responsibility is to generate metadata. It should not \
perform any model training or inference.
- The script must read raw data from the `./input` directory. This \
directory should be treated as read-only.
- All generated metadata files (e.g., .csv, .parquet, .json) must be \
saved directly to the `./metadata` directory.
- You must not copy or move the original raw data. The `./metadata` \
directory should only contain the newly generated metadata files.
- All file paths stored within the metadata must be relative to the `./\
input` directory. Review the Dataset Information section above to \
identify the correct file paths and structure.
- The metadata for the training and validation datasets must include the \
ground-truth labels.
- Create a validation set by splitting the training data only if a \
separate validation dataset is not already available.
    - Use a fixed 80:20 ratio (80% training and 20% validation). This \
ratio should not be a user-configurable argument.
    - Randomly shuffle the training data before splitting. To ensure the \
split is reproducible, use a fixed random state (`RANDOM_STATE = 42`).
    - Apply stratified sampling or group sampling to ensure the \
validation dataset's distribution properly represents the original \
data.
        - Stratified Sampling: Use this if it's a classification task, \
stratifying by the target label.
        - Group Sampling: Use this if the data has inherent groups (e.g., \
patient IDs, user IDs) that must not be split across the train and \
validation sets.
- After generating the metadata, the script must immediately load the \
datasets using the new metadata and perform the following checks:
    - Print summary statistics for the final training, validation, and \
test datasets (e.g., total number of samples, class distributions, \
data shapes, number of unique users, etc.).
    - If the metadata contains file paths, programmatically check 1000 \
relative file paths randomly selected from each of the metadata files. \
Calculate the ratio of paths that do not resolve to an existing file \
in `./input`. If this "missing file ratio" is greater than 0.5, the \
script must raise an error. Before raising the error, print a sample (\
e.g., the first five) of the non-resolving file paths to the console \
for debugging purposes.
    - If a new validation set was created, you must programmatically \
verify that it satisfies the requirements:
        - Assert that the stratification or group split was successful.
        - Raise an error (e.g., AssertionError) if these verification \
checks fail.

# Implementation Guideline
- The code should be a single-file python program that is self-contained \
and can be executed as-is.
- The script must be complete and able to run from start to finish \
without premature termination or skipped sections.
- Your response should only contain a single code block.
- All validation checks must fail explicitly, either by raising an \
Exception or triggering an AssertionError.
- Do not use `try ... except` blocks to suppress or ignore errors in your \
code examples.
- Be aware of the running time of the code, it should complete within \
{exec_timeout}.
- All the provided input data is stored in "./input" directory. There is \
no need to unzip any files.

# Task Description
{task_description}
"""
