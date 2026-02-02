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
    Calculates the Multi-class Log Loss.

    Cite solution_lesson_node_00184: Passing raw softmax probabilities directly
    to log_loss (which handles clipping) avoids numerical noise from redundant
    normalization steps, matching the 'Current Best' strategy.

    Args:
        y_true (array-like): Ground truth labels (indices or strings).
        y_pred (array-like): Predicted probabilities matrix (assumed from softmax).
        labels (list, optional): List of class labels.

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss applies clipping internally using eps.
    # We pass the raw probabilities to minimize floating point operations.
    return log_loss(y_true, y_pred, labels=labels, eps=PROB_CLIP_EPSILON)


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
