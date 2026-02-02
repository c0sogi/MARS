import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch is mentioned in the environment, so we seed it if available
    # to ensure full reproducibility if DL components are added later.
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_float64(arr):
    """
    Enforces double precision (float64) on a numpy array or list.
    Crucial for minimizing numerical noise in the 'Metric Floor' regime.

    Args:
        arr (array-like): Input data.

    Returns:
        np.ndarray: Data converted to float64.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr.astype(np.float64)


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the Multi-class Log Loss with specific normalization and clipping
    rules as defined in the task description.

    Rules:
    1. Rescale: Each row is divided by the row sum.
    2. Clip: Probabilities are clipped to [1e-15, 1 - 1e-15].
    3. Score: Standard multi-class log loss.

    Args:
        y_true (array-like): Ground truth labels (1D array of class labels or 2D one-hot).
        y_pred (array-like): Predicted probabilities (2D array).

    Returns:
        float: The calculated log loss.
    """
    y_pred = ensure_float64(y_pred)

    # 1. Rescale (Normalize rows to sum to 1)
    # Add a small epsilon to denominator to prevent division by zero if a row is all zeros
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the label encoding and calculation
    # We pass the pre-processed probabilities
    return log_loss(y_true, y_pred_clipped)


def save_submission(ids, classes, probs, output_path):
    """
    Formats and saves the submission file in the required CSV format.

    Format:
    id,Class_1,Class_2,...
    2,0.1,0.5,...

    Args:
        ids (array-like): 1D array of image IDs.
        classes (list): List of class names corresponding to the columns of probs.
        probs (array-like): 2D array of predicted probabilities.
        output_path (str): Path to save the CSV file.
    """
    # Ensure consistency
    probs = ensure_float64(probs)
    ids = np.array(ids).flatten()

    if probs.shape[0] != len(ids):
        raise ValueError(
            f"Length mismatch: {len(ids)} IDs vs {probs.shape[0]} probability rows."
        )

    if probs.shape[1] != len(classes):
        raise ValueError(
            f"Class count mismatch: {len(classes)} names vs {probs.shape[1]} probability columns."
        )

    # Create DataFrame
    df = pd.DataFrame(probs, columns=classes)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
