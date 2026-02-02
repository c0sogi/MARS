import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    as defined in the competition metric.

    The metric requires:
    1. Rescaling rows to sum to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].
    3. Computing log loss.

    Args:
        y_true: Array-like of shape (n_samples,). Can be class indices or class labels.
                If labels, they must correspond to the column order of y_pred.
        y_pred: Array-like of shape (n_samples, n_classes). Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays with high precision
    y_pred = np.array(y_pred, dtype=Config.NP_DTYPE)
    y_true = np.array(y_true)

    # 1. Rescale rows to sum to 1
    # Compute row sums
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle rows that sum to 0 to avoid NaN division (though unlikely in valid predictions)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    # Metric rule: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Calculate Log Loss
    # We explicitly provide labels to ensure log_loss knows the dimension if y_true are indices.
    labels = None
    if np.issubdtype(y_true.dtype, np.number):
        labels = np.arange(y_pred.shape[1])

    score = log_loss(y_true, y_pred_clipped, labels=labels)
    return score


def save_submission(ids, classes, probs, filename=Config.SUBMISSION_FILE_PATH):
    """
    Saves the submission file in the correct format.

    Args:
        ids (array-like): Image IDs.
        classes (list): List of class names corresponding to probs columns.
        probs (array-like): Probability matrix (n_samples, n_classes).
        filename (str): Output path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    submission = pd.DataFrame(probs, columns=classes)
    submission.insert(0, "id", ids)

    # Save
    submission.to_csv(filename, index=False)
