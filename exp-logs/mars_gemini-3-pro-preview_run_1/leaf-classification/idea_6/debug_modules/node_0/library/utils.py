import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python's random, numpy, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss with specific normalization and clipping
    as defined in the task description.

    The metric requires:
    1. Rescaling each row of probabilities to sum to 1.
    2. Clipping probabilities to the range [1e-15, 1 - 1e-15].
    3. Computing the negative log likelihood.

    Args:
        y_true: Array-like of shape (n_samples,) containing true class labels
                (strings or integers) or (n_samples, n_classes) one-hot encoded.
        y_pred: Array-like of shape (n_samples, n_classes) containing predicted probabilities.
        labels: List of class labels to index y_pred columns if y_true are strings/integers.

    Returns:
        float: The calculated log loss value.
    """
    # Ensure y_pred is a numpy array of floats
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale rows to sum to 1
    # This simulates the competition's scoring mechanism
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero by replacing 0 sums with 1 (though 0 sum implies all 0 probs)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities to avoid log(0) extremes
    # Metric requirement: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Calculate Log Loss
    # sklearn.metrics.log_loss handles the summation and label matching
    loss = log_loss(y_true, y_pred, labels=labels, eps=eps)

    return loss


def save_submission(ids, classes, probs, output_path=Config.SUBMISSION_FILE):
    """
    Saves the prediction probabilities to a CSV file in the required format.

    The file will have a header: id, Class1, Class2, ...
    And rows corresponding to each test image.

    Args:
        ids: Array-like of image IDs.
        classes: List of class names corresponding to the columns of probs.
        probs: Array-like of shape (n_samples, n_classes) containing probabilities.
        output_path: Path to save the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=classes)

    # Insert 'id' column at the beginning
    df.insert(0, "id", ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
