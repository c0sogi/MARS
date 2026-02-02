import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config


def set_seed(seed=config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across numpy, random, and os.

    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_STATE.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific rescaling and clipping
    as defined in the competition metric.

    The metric requires:
    1. Rescaling rows to sum to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].

    Args:
        y_true: Ground truth labels (1D array of labels or 2D one-hot array).
        y_pred: Predicted probabilities (2D array).

    Returns:
        float: The calculated log loss.
    """
    # Ensure float64 for precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale: each row divided by row sum
    # The problem statement says probabilities are rescaled prior to being scored.
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid division by zero (though unlikely with valid models)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to avoid extremes of the log function
    # Range: [1e-15, 1 - 1e-15]
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss using sklearn
    # sklearn's log_loss handles the actual log calculation and summation.
    # We pass the pre-processed probabilities.
    return log_loss(y_true, y_pred)


def save_submission(ids, predictions, class_names, filename="submission.csv"):
    """
    Formats and saves the submission file.

    Args:
        ids: Array-like of image IDs.
        predictions: 2D numpy array of probabilities (shape: [n_samples, n_classes]).
        class_names: List of class names corresponding to the columns of predictions.
        filename: Name of the output file.
    """
    # Create DataFrame
    df = pd.DataFrame(predictions, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Construct full path
    filepath = os.path.join(config.SUBMISSION_DIR, filename)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")
