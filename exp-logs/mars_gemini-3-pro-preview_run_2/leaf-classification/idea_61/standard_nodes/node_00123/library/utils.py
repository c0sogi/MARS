import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, FLOAT_PRECISION


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clipped_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss with row normalization and clipping
    as specified in the task description.

    Logic:
    1. Cast predictions to float64.
    2. Normalize each row so probabilities sum to 1.
    3. Clip probabilities to [1e-15, 1 - 1e-15].
    4. Compute log loss.

    Args:
        y_true: Ground truth labels. Can be 1D array of class indices/labels
                or 2D binary indicator matrix (one-hot).
        y_pred: Predicted probabilities. 2D array of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure double precision as per requirements
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # 1. Row normalization (rescaled prior to being scored)
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid division by zero (though unlikely in valid output)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clipping to avoid extremes of log function
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1.0 - epsilon)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the actual log calculation and averaging.
    # We pass the pre-processed (normalized and clipped) probabilities.
    return log_loss(y_true, y_pred_clipped)


def save_submission(ids, classes, probs, output_path):
    """
    Saves the predictions to a CSV file in the required submission format.

    Format:
    id,Class_1,Class_2,...
    2,0.1,0.5,...

    Args:
        ids: 1D array-like of image IDs.
        classes: List of class names (strings) corresponding to the columns of probs.
        probs: 2D array of probabilities (n_samples, n_classes).
        output_path: File path to save the CSV.
    """
    # Ensure probs are float64
    probs = np.array(probs, dtype=FLOAT_PRECISION)

    # Create DataFrame with class names as headers
    df = pd.DataFrame(probs, columns=classes)

    # Insert 'id' column at the start
    df.insert(0, "id", ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
