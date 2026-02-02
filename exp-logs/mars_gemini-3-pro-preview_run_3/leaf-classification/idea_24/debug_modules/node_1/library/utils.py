import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
import library.config as cfg


def seed_everything(seed: int = cfg.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to the value in config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_and_normalize(probs: np.ndarray) -> np.ndarray:
    """
    Rescales probabilities to sum to 1 per row and clips them to avoid
    extremes of the log function, mimicking the competition metric logic.

    Args:
        probs (np.ndarray): Raw probability matrix (N_samples, N_classes).

    Returns:
        np.ndarray: Processed probability matrix.
    """
    # Work on a copy to avoid side effects
    probs = probs.copy()

    # 1. Rescale: Divide each row by the row sum
    row_sums = probs.sum(axis=1, keepdims=True)

    # Avoid division by zero if any row sums to 0 (unlikely with softmax/LDA but possible)
    # Assign uniform probability to zero-sum rows to ensure valid distribution
    zero_mask = (row_sums == 0).flatten()
    if np.any(zero_mask):
        probs[zero_mask] = 1.0 / probs.shape[1]
        row_sums[zero_mask] = 1.0

    probs_norm = probs / row_sums

    # 2. Clip: Replace probabilities with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    probs_clipped = np.clip(probs_norm, epsilon, 1 - epsilon)

    return probs_clipped


def calculate_log_loss(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list = None
) -> float:
    """
    Calculates the multi-class log loss after applying the specific
    normalization and clipping rules of the task.

    Args:
        y_true (np.ndarray): Ground truth labels (1D array of class indices or names).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).
        labels (list, optional): List of class labels to index the matrix.

    Returns:
        float: The calculated log loss.
    """
    # Apply the competition-specific preprocessing
    y_pred_processed = clip_and_normalize(y_pred)

    # Calculate log loss
    # Note: sklearn log_loss handles normalization internally, but we do it explicitly
    # to match the clipping logic described in the task.
    score = log_loss(y_true, y_pred_processed, labels=labels)
    return score


def save_submission(
    ids: np.ndarray,
    probs: np.ndarray,
    class_names: list,
    output_path: str = cfg.SUBMISSION_PATH,
):
    """
    Formats and saves the submission file.

    Args:
        ids (np.ndarray): Array of image IDs.
        probs (np.ndarray): Predicted probability matrix (N_samples, N_classes).
        class_names (list): List of class names corresponding to the columns of probs.
        output_path (str): Path to save the CSV file.
    """
    # Ensure probabilities are normalized and clipped before saving
    # While the metric does this, it's good practice to submit clean probabilities.
    probs_processed = clip_and_normalize(probs)

    # Create DataFrame
    df = pd.DataFrame(probs_processed, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", ids.astype(int))

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
