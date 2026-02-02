import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and environment variables.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clip_probabilities(preds: np.ndarray) -> np.ndarray:
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to avoid log(0) extremes,
    strictly matching the competition metric definition.

    Args:
        preds (np.ndarray): The array of predicted probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(preds, epsilon, 1 - epsilon)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific normalization and clipping.

    Implements the specific scoring mechanism described in the task:
    1. Rescales rows to sum to 1 (row-wise normalization).
    2. Clips probabilities to [1e-15, 1-1e-15].
    3. Computes log loss.

    Args:
        y_true: Array-like of true labels (class indices or one-hot encoded).
        y_pred: Array-like of predicted probabilities (shape: [n_samples, n_classes]).

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array and float64 precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale rows to sum to 1 (as per metric description)
    # Calculate row sums
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid division by zero (replace with 1, result stays 0)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # 3. Compute Log Loss
    # sklearn log_loss handles the label format (indices vs one-hot) automatically
    # We pass the pre-processed probabilities
    return log_loss(y_true, y_pred_clipped)


def load_dataset(split: str):
    """
    Loads the dataset metadata for a specific split from the metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    path = os.path.join("./metadata", f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)
