import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
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


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log loss extremes.

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    return np.clip(probs, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)


def calculate_metric(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss.

    Args:
        y_true (array-like): True class labels (can be label encoded or one-hot).
        y_pred (array-like): Predicted probabilities.
        labels (list, optional): List of class labels to index the matrix if y_true are strings.

    Returns:
        float: The calculated log loss.
    """
    # Ensure probabilities are clipped before calculation to match evaluation protocol locally
    y_pred_clipped = clip_probabilities(y_pred)

    # Calculate log loss
    # Note: sklearn log_loss handles normalization internally
    loss = log_loss(y_true, y_pred_clipped, labels=labels)
    return loss


def save_submission(ids, classes, probs, output_path=Config.SUBMISSION_FILE):
    """
    Formats and saves the submission file.

    Args:
        ids (array-like): List or array of image IDs.
        classes (list): List of class names corresponding to the columns of probs.
        probs (np.ndarray): Matrix of predicted probabilities (N_samples x N_classes).
        output_path (str): Path to save the CSV file. Defaults to Config.SUBMISSION_FILE.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    # The submission format requires 'id' as the first column, followed by species names
    df = pd.DataFrame(probs, columns=classes)
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
