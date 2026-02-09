"""Prompt template for solution drafting / main script (Appendix F.12)."""

from __future__ import annotations


def format_prompt(
    *,
    idea: str,
    library_files: str,
    file_description: str,
    context: str,
) -> str:
    return f"""\
==== Idea ====
{idea}

==== Python Files ====
The following Python files are already provided. Do not modify them.
{library_files}

==== Target File Description (`runfile.py`) ====
{file_description}

==== Task ====
Your task is to implement the end-to-end orchestration script `runfile.py\
`. This script serves as a **fast baseline** to verify the idea. It \
must train the model, validate performance, perform failure analysis, \
and generate a submission.

# Requirements
- Import the functions or classes from the given Python files instead of \
re-implementing them.
- Make the model training fast.
    - Limit maximum number of training samples and training steps/epochs \
to ensure a quick baseline execution.
    - Set appropriate batch sizes to prevent CUDA out-of-memory errors.
- After training is complete, you must execute validation assessment, \
failure analysis, and submission generation.
    - You must load the hold-out validation dataset using the metadata \
located in the `./metadata` directory.
    - You must print the final validation metric computed on the entire \
hold-out validation set in this format `Final Validation Metric: <\
value>`. Without this metric, the solution cannot be evaluated, \
rendering the entire code invalid. You must use the validation metric \
defined in the Task Description. Please print the full precision of \
the validation metric without any rounding or formatting.
    - You must perform failure analysis on the trained model. You must \
perform failure analysis on the validation set to identify systematic \
error patterns. Calculate and print the correlation between the model'\
s error magnitude and the input features to reveal which variables are \
most associated with poor performance.
    - You must generate predictions for the entire test set and create \
the submission file{{submission_cond}}. Save the final predictions to \
`./submission/submission.csv`. Refer to the sample submission file (e.\
g., `./input/sample_submission.csv` or `./input/sampleSubmission.csv`)\
 for the correct formatting required by the competition.
- Optimize the validation inference speed.
    - Ensure the model is in evaluation mode for this inference.
    - Your code must automatically detect and utilize an available GPU \
for inference. Ensure the model and all data batches are moved to the \
correct device (GPU or CPU).
    - During inference, you don't need to compute gradients. Disabling \
this process reduces memory consumption and speeds up computation.
- Call data loading functions with `load_cached_data=True` (if applicable\
) to utilize preprocessed data in the `./working` directory.

{context}
"""
