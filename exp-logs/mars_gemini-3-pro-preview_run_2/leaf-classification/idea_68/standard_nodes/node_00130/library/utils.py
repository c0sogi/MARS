import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, SUBMISSION_PATH


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(y_pred):
    """
    Rescales and clips probabilities according to the competition metric definition.

    1. Rescales rows to sum to 1.
    2. Clips values to [1e-15, 1 - 1e-15].

    Args:
        y_pred (np.ndarray): The raw predicted probabilities.

    Returns:
        np.ndarray: The processed probabilities.
    """
    # Rescale prior to scoring (each row is divided by the row sum)
    # Adding epsilon to denominator to prevent division by zero if sum is 0
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred_rescaled = y_pred / (row_sums + 1e-15)

    # Replace with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_rescaled, epsilon, 1 - epsilon)

    return y_pred_clipped


def compute_log_loss(y_true, y_pred, classes=None):
    """
    Computes the multi-class log loss using the competition's specific clipping logic.

    Args:
        y_true (np.ndarray or list): True class labels (strings or integers) or one-hot encoded matrix.
        y_pred (np.ndarray): Predicted probabilities.
        classes (list, optional): List of class labels to index y_pred if y_true are labels.
                                  Passed to sklearn.metrics.log_loss.

    Returns:
        float: The calculated log loss.
    """
    # Apply competition-specific rescaling and clipping
    y_pred_processed = clip_probabilities(y_pred)

    # Calculate log loss
    # Note: sklearn log_loss also applies clipping internally, but we do it explicitly
    # to ensure the rescaling step is respected as per task description.
    return log_loss(y_true, y_pred_processed, labels=classes)


def save_submission(ids, classes, probs, output_path=SUBMISSION_PATH):
    """
    Formats and saves the submission file.

    Args:
        ids (np.ndarray or list): The image IDs.
        classes (list): The list of class names (column headers).
        probs (np.ndarray): The matrix of predicted probabilities (N_samples x N_classes).
        output_path (str): Path to save the CSV file.
    """
    # Ensure probabilities are rescaled/clipped for final submission
    # (though the scoring system does it, it's good practice to submit clean probs)
    probs_clean = clip_probabilities(probs)

    # Create DataFrame
    df = pd.DataFrame(probs_clean, columns=classes)
    df.insert(0, "id", ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
