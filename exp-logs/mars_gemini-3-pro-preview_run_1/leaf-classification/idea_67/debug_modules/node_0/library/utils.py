import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, FLOAT_PRECISION, PROB_CLIP_EPSILON


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS hashing.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def normalize_probabilities(probs):
    """
    Normalizes the probability matrix so that each row sums to 1.
    Uses high precision (float64) as defined in config.

    Args:
        probs (array-like): Input probabilities of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Normalized probabilities.
    """
    # Ensure high precision
    probs = np.array(probs, dtype=FLOAT_PRECISION)

    # Compute row sums
    row_sums = probs.sum(axis=1, keepdims=True)

    # Avoid division by zero by replacing zero sums with 1 (though unlikely in softmax)
    row_sums[row_sums == 0] = 1.0

    return probs / row_sums


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the Multi-class Log Loss strictly according to competition metrics.
    Probabilities are normalized and then clipped to [eps, 1-eps].

    Args:
        y_true (array-like): Ground truth labels (indices or strings).
        y_pred (array-like): Predicted probabilities matrix.
        labels (list, optional): List of class labels to index the matrix if y_true are strings.

    Returns:
        float: The calculated log loss.
    """
    # 1. Normalize probabilities (rescaling per metric definition)
    y_pred_norm = normalize_probabilities(y_pred)

    # 2. Clip probabilities
    # Metric: max(min(p, 1-10^-15), 10^-15)
    # sklearn log_loss does this internally if we pass eps, but we do it explicitly
    # to ensure consistency before passing to the metric function if needed,
    # though sklearn's implementation is sufficient with the eps parameter.
    y_pred_clipped = np.clip(y_pred_norm, PROB_CLIP_EPSILON, 1.0 - PROB_CLIP_EPSILON)

    # 3. Calculate Log Loss
    return log_loss(y_true, y_pred_clipped, labels=labels, eps=PROB_CLIP_EPSILON)


def format_submission(test_ids, classes, probs, output_path):
    """
    Formats the predictions into a CSV file for submission.

    Args:
        test_ids (array-like): Array of test image IDs.
        classes (list): List of class names corresponding to the columns of probs.
        probs (array-like): Predicted probability matrix.
        output_path (str): File path to save the submission CSV.
    """
    # Normalize probabilities for submission to ensure valid range [0, 1] and sum to 1
    probs_norm = normalize_probabilities(probs)

    # Create DataFrame
    df = pd.DataFrame(probs_norm, columns=classes)

    # Insert 'id' column at the start
    df.insert(0, "id", test_ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
