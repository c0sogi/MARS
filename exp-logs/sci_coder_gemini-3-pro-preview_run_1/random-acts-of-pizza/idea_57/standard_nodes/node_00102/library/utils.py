import os
import random
import pickle
import numpy as np
import torch
import pandas as pd
from library import config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed} for random, numpy, and torch.")


def save_pickle(obj, path):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {path}")


def load_pickle(path):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Object loaded from {path}")
    return obj


def verify_submission_format(submission_df):
    """
    Verifies that the submission DataFrame matches the required format
    specified in the sample submission file.

    Args:
        submission_df (pd.DataFrame): The DataFrame containing predictions.

    Raises:
        ValueError: If the format is incorrect.
    """
    # Locate sample submission
    sample_path = os.path.join(config.INPUT_DIR, "sampleSubmission.csv")
    if not os.path.exists(sample_path):
        # Fallback for different naming conventions if necessary,
        # though description specifies sampleSubmission.csv
        sample_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(sample_path):
        print(
            "Warning: Sample submission file not found. Skipping strict format verification against file."
        )
        # Perform basic checks based on known requirements
        expected_cols = ["request_id", "requester_received_pizza"]
    else:
        sample_df = pd.read_csv(sample_path)
        expected_cols = sample_df.columns.tolist()

        # Check shape
        if len(submission_df) != len(sample_df):
            raise ValueError(
                f"Submission has {len(submission_df)} rows, expected {len(sample_df)}."
            )

        # Check IDs match (assuming order might differ, we check set equality)
        if set(submission_df["request_id"]) != set(sample_df["request_id"]):
            raise ValueError(
                "Submission request_ids do not match the sample submission."
            )

    # Check columns
    if list(submission_df.columns) != expected_cols:
        raise ValueError(
            f"Submission columns {list(submission_df.columns)} do not match expected {expected_cols}."
        )

    # Check types
    if not pd.api.types.is_numeric_dtype(submission_df["requester_received_pizza"]):
        raise ValueError(
            "Column 'requester_received_pizza' must be numeric (probabilities)."
        )

    print("Submission format verified successfully.")
