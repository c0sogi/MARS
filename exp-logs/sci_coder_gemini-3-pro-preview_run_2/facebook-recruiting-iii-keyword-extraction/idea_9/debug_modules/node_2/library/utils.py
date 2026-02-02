import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(y_true, y_pred, average="samples"):
    """
    Calculates the Mean F1 Score.

    Args:
        y_true: Ground truth binary labels (n_samples, n_classes).
        y_pred: Predicted binary labels (n_samples, n_classes).
        average (str): Averaging method. Defaults to 'samples' for multi-label tasks.

    Returns:
        float: The computed F1 score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def get_dynamic_threshold(
    y_true, y_proba, percentile_low=95, percentile_high=99.9, steps=50
):
    """
    Determines the optimal classification threshold by searching within a dynamic range
    defined by the percentiles of the predicted probabilities.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities.
        percentile_low (float): The lower percentile bound for the search range (0-100).
        percentile_high (float): The upper percentile bound for the search range (0-100).
        steps (int): Number of thresholds to evaluate within the range.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Ensure y_proba is a numpy array for percentile calculation
    if not isinstance(y_proba, np.ndarray):
        y_proba = np.array(y_proba)

    # Calculate the search range based on probability distribution
    low_val = np.percentile(y_proba, percentile_low)
    high_val = np.percentile(y_proba, percentile_high)

    # Generate candidate thresholds
    thresholds = np.linspace(low_val, high_val, steps)

    best_score = -1.0
    best_threshold = 0.5

    # Iterate through candidates to find the optimum
    for thresh in thresholds:
        # Binarize predictions
        y_pred = (y_proba >= thresh).astype(int)

        # Calculate score
        score = calculate_f1_score(y_true, y_pred, average="samples")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
