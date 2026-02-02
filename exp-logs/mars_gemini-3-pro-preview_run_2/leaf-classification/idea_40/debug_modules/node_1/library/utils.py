import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import PROB_CLIP, FLOAT_PRECISION, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def clip_probabilities(preds):
    """
    Clips probabilities to the range [PROB_CLIP, 1 - PROB_CLIP].
    Ensures input is cast to FLOAT_PRECISION.
    """
    preds = np.array(preds, dtype=FLOAT_PRECISION)
    # Clip to avoid log(0) and log(1) extremes as per metric definition
    return np.clip(preds, PROB_CLIP, 1.0 - PROB_CLIP)


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss with specific rescaling and clipping
    as defined in the competition metric.

    Args:
        y_true: Ground truth labels (1D array).
        y_pred: Predicted probabilities (2D array).
        labels: Optional list of class labels to index the y_pred columns correctly.

    Returns:
        float: The log loss score.
    """
    # Ensure precision
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # 1. Rescale: Each row is divided by the row sum
    # "The submitted probabilities... are rescaled prior to being scored"
    row_sums = y_pred.sum(axis=1)
    # Handle zero sums to avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: Replace with max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # 3. Calculate Log Loss
    score = log_loss(y_true, y_pred_clipped, labels=labels)

    return score


def save_submission(ids, classes, probs, output_path):
    """
    Saves the submission file in the required CSV format.

    Args:
        ids: Array-like of image IDs.
        classes: List of class names corresponding to the columns of probs.
        probs: 2D array of predicted probabilities.
        output_path: File path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    # Ensure probs are float64
    probs = np.array(probs, dtype=FLOAT_PRECISION)
    submission_df = pd.DataFrame(probs, columns=classes)

    # Add 'id' column at the start
    submission_df.insert(0, "id", ids)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
