import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import PRECISION_TYPE, SUBMISSION_DIR


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_precision(data, name="Data"):
    """
    Validates that the input numpy array uses the required float64 precision.
    Raises a ValueError if the precision does not match the configuration.

    Args:
        data (np.ndarray): The data array to check.
        name (str): The name of the data variable for the error message.

    Raises:
        ValueError: If data.dtype is not np.float64.
    """
    if not isinstance(data, np.ndarray):
        return

    if data.dtype != PRECISION_TYPE:
        raise ValueError(
            f"Precision mismatch for {name}. "
            f"Expected {PRECISION_TYPE}, but got {data.dtype}. "
            "Strict double precision (float64) is required to avoid metric floors."
        )


def save_submission(ids, probabilities, class_names, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (array-like): The image IDs.
        probabilities (array-like): The predicted probabilities (n_samples, n_classes).
        class_names (list): The list of class names corresponding to the probability columns.
        filename (str): The name of the output file.
    """
    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Validate inputs
    if len(ids) != len(probabilities):
        raise ValueError(
            f"Length mismatch: {len(ids)} IDs vs {len(probabilities)} probability rows."
        )

    if probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Class count mismatch: {probabilities.shape[1]} probability columns vs {len(class_names)} class names."
        )

    # Create DataFrame
    submission_df = pd.DataFrame(probabilities, columns=class_names)

    # Ensure IDs are the first column
    submission_df.insert(0, "id", ids)

    # Construct full path
    output_path = os.path.join(SUBMISSION_DIR, filename)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
