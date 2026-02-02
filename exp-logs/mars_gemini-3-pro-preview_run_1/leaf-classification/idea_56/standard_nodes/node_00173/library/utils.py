import os
import random
import numpy as np
import pandas as pd
from library.config import (
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSION_DIR,
)


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and basic environment variables.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_paths():
    """
    Validates that all critical data files and directories exist as defined in the configuration.
    Raises a FileNotFoundError if any path is missing.
    """
    paths_to_check = [
        TRAIN_METADATA_PATH,
        VAL_METADATA_PATH,
        TEST_METADATA_PATH,
        SAMPLE_SUBMISSION_PATH,
        SUBMISSION_DIR,
    ]

    for path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical file or directory missing: {path}")

    # print("All critical paths validated successfully.")


def save_submission(ids, classes, probs, output_path=None):
    """
    Saves the prediction results to a CSV file in the required submission format.

    Args:
        ids (array-like): 1D array or list of image IDs.
        classes (array-like): 1D array or list of class names (strings), corresponding to the columns of probs.
        probs (array-like): 2D array of probabilities (shape: [n_samples, n_classes]).
        output_path (str, optional): Full path to save the CSV. If None, defaults to submission/submission.csv.
    """
    if output_path is None:
        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    # The submission format requires 'id' as the first column, followed by species columns.
    df = pd.DataFrame(probs, columns=classes)
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
