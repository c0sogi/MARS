import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch
import library.config as config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_dataset(path, nrows=None):
    """
    Loads a dataset from the specified file path.

    Args:
        path (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read. Useful for debugging.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, nrows=nrows)


def save_submission(ids, probabilities, output_path=None):
    """
    Formats and saves the submission file.

    Args:
        ids (list or np.array): List of sample IDs.
        probabilities (np.array): Predicted probabilities for the classes.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = config.SUBMISSION_PATH

    # Ensure probabilities are a numpy array
    probabilities = np.array(probabilities)

    # Create DataFrame
    submission = pd.DataFrame(probabilities, columns=config.CLASSES)
    submission.insert(0, "id", ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class logarithmic loss with rescaling and clipping
    as defined in the competition metric.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.
        labels (list, optional): List of class labels. Defaults to config.CLASSES.

    Returns:
        float: The calculated log loss.
    """
    if labels is None:
        labels = config.CLASSES

    y_pred = np.array(y_pred)

    # Rescale rows to sum to 1 (as per metric definition)
    row_sums = y_pred.sum(axis=1)
    # Handle zero sums to avoid division by zero (though unlikely with softmax)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to avoid log(0) extremes
    # Metric definition: max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = np.clip(
        y_pred_rescaled, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX
    )

    # Calculate log loss
    return log_loss(y_true, y_pred_clipped, labels=labels)
