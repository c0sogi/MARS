"""Prompt template for module implementation (Appendix F.10)."""

from __future__ import annotations


def format_prompt(
    *,
    idea: str,
    library_files: str,
    file_name: str,
    file_description: str,
    context: str,
) -> str:
    return f"""\
==== Idea ====
{idea}

==== Python Files ====
The following Python files are already provided. Do not modify them.
{library_files}

==== Target File Description ({file_name}) ====
{file_description}

==== Task ====
Your task is to implement the `{file_name}` module based on the \
description above.

# Requirements
- Import the functions or classes from the given Python files instead of \
re-implementing them.
- Only implement the module class/functions. DO NOT include an if \
`__name__ == "__main__":` block. DO NOT implement the end-to-end \
pipeline.
- Ensure functions accept arguments for flexibility. You must include \
hyperparameters to control dataset size (for debugging) and training \
steps/epochs.
- When printing validation metrics, please print the full precision \
without any rounding or formatting.
- If loading raw data, use the metadata in `./metadata` to identify the \
correct train/val/test splits.
- If this module performs deterministic data processing, you must \
implement a caching mechanism strictly following this logic:
    - **Function Signature:** The processing function must accept a `\
load_cached_data: bool` argument.
    - **Directory Safety:** Ensure the directory `./working/{{dir_name}}/`\
 exists (use `os.makedirs(..., exist_ok=True)`).
    - **Prohibited:** Do NOT use `pickle`. Use `parquet` (via pandas) or \
`npy` (via numpy).
    - **Logic Flow:**
        1. IF `load_cached_data` is True: Try to load the file.
        2. IF loading fails (file missing or corrupt) OR `\
load_cached_data` is False:
            - Compute/process the data from scratch.
            - Save the result to the cache directory `./working/\
{{dir_name}}/` for future runs.
        3. Return the data.
- If this module handles model training:
    - **Metrics:** Print key training and validation metrics during \
training process.
    - **Optimization:** Implement Early Stopping to prevent overfitting \
and reduce runtime.
- If this module handles submission generation:
    - Generate predictions for the entire test set. Save the final \
predictions to `./submission/submission.csv`.
    - Refer to the sample submission file (e.g., `./input/\
sample_submission.csv` or `./input/sampleSubmission.csv`) for the \
correct formatting required by the competition.

{context}
"""
