import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def validate_precision(data, name="Data", expected_dtype=np.float32):
    """
    Validates that the input array has the expected data type (default: float32).
    If not, it casts the data to the expected type and prints a message.

    Args:
        data (np.ndarray): The data array to check.
        name (str): Name of the data for logging purposes.
        expected_dtype (type): The expected numpy data type.

    Returns:
        np.ndarray: The data cast to the expected dtype.
    """
    if data.dtype != expected_dtype:
        print(
            f"[{name}] dtype mismatch: found {data.dtype}, casting to {expected_dtype} for precision consistency."
        )
        return data.astype(expected_dtype)
    return data


def log_loss_metric(y_true, y_pred):
    """
    Computes the Multi-class Log Loss with specific rescaling and clipping
    as defined in the competition metric.

    The probabilities are first rescaled so that each row sums to 1.
    Then they are clipped to the range [1e-15, 1 - 1e-15].

    Args:
        y_true (array-like): True class labels (indices) or one-hot encoded vectors.
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # 1. Rescale rows to sum to 1
    # Add a small epsilon to row_sums to avoid division by zero if a row is all zeros
    row_sums = y_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums

    # 2. Clip probabilities to avoid extremes of the log function
    # Range: [10^-15, 1 - 10^-15]
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_rescaled, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # We provide the labels explicitly to ensure log_loss handles the shape correctly
    # assuming y_pred columns cover all classes 0 to N-1
    n_classes = y_pred.shape[1]
    labels = np.arange(n_classes)

    return log_loss(y_true, y_pred_clipped, labels=labels)


def format_submission(ids, preds, class_names, output_path):
    """
    Formats the predictions into a CSV file suitable for submission.

    Args:
        ids (array-like): 1D array of image IDs.
        preds (array-like): 2D array of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of preds.
        output_path (str): Path to save the submission CSV.
    """
    # Create DataFrame
    submission_df = pd.DataFrame(preds, columns=class_names)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", ids)

    # Ensure ID is integer
    submission_df["id"] = submission_df["id"].astype(int)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
