import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import WORKING_DIR, SUBMISSION_DIR


def seed_everything(seed: int = 42):
    """
    Sets seeds for random, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def arcsinh_transform(x):
    """
    Applies the inverse hyperbolic sine transformation to the input data.
    Useful for normalizing skewed distributions including zero and negative values.
    """
    return np.arcsinh(x)


def get_feature_intersection(
    train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols: list = None
) -> list:
    """
    Identifies the intersection of columns between train and test dataframes,
    removing any columns specified in exclude_cols.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        exclude_cols: List of column names to exclude (e.g., target, ID).

    Returns:
        Sorted list of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Intersection of columns
    common_cols = train_cols.intersection(test_cols)

    # Remove excluded columns
    final_cols = [c for c in common_cols if c not in exclude_cols]

    return sorted(list(final_cols))


def save_to_cache(data, filename: str, directory: str = WORKING_DIR):
    """
    Saves data to the cache directory using parquet or numpy formats.
    Strictly avoids pickle.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    if filename.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame to save as .parquet")

    elif filename.endswith(".npy"):
        np.save(filepath, data)

    elif filename.endswith(".npz"):
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            raise ValueError("Data must be a dictionary to save as .npz")

    else:
        raise ValueError("Filename must end with .parquet, .npy, or .npz")


def load_from_cache(filename: str, directory: str = WORKING_DIR):
    """
    Loads data from the cache directory. Returns None if file does not exist.
    """
    filepath = os.path.join(directory, filename)

    if not os.path.exists(filepath):
        return None

    if filename.endswith(".parquet"):
        return pd.read_parquet(filepath)
    elif filename.endswith(".npy"):
        return np.load(filepath)
    elif filename.endswith(".npz"):
        return np.load(filepath)
    else:
        raise ValueError("Filename must end with .parquet, .npy, or .npz")


def save_submission(request_ids, predictions, filename: str = "submission.csv"):
    """
    Formats and saves the submission file to the SUBMISSION_DIR.
    """
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Ensure inputs are 1D arrays
    request_ids = np.array(request_ids).flatten()
    predictions = np.array(predictions).flatten()

    if len(request_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: IDs ({len(request_ids)}) vs Predictions ({len(predictions)})"
        )

    df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    filepath = os.path.join(SUBMISSION_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")
