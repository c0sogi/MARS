import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import FLOAT_PRECISION


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
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


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific rescaling and clipping
    as defined in the competition metric.

    Steps:
    1. Rescale: Each row is divided by the row sum.
    2. Clip: Probabilities are clipped to [1e-15, 1 - 1e-15].
    3. Score: Multi-class log loss.

    Args:
        y_true (array-like): Ground truth labels (1D indices/labels or 2D indicator matrix).
        y_pred (array-like): Predicted probabilities (2D array).

    Returns:
        float: The calculated log loss.
    """
    # Ensure high precision for metric calculation
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # 1. Rescale rows to sum to 1
    row_sums = np.sum(y_pred, axis=1, keepdims=True)
    # Handle edge case of zero sum (though unlikely with valid model output)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1.0 - epsilon)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the summation and logarithmic penalty.
    # We pass the pre-processed probabilities.
    return log_loss(y_true, y_pred_clipped)


def save_submission(ids, classes, probs, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (array-like): List or array of image IDs.
        classes (list): List of species names corresponding to probability columns.
        probs (array-like): 2D array of predicted probabilities.
        output_path (str): Path where the submission CSV should be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame with high precision
    df = pd.DataFrame(probs, columns=classes, dtype=FLOAT_PRECISION)

    # Insert the 'id' column at the beginning
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
