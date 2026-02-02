import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import SUBMISSION_PATH, RANDOM_SEED, FLOAT_PRECISION


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and os environments.
    Also attempts to set torch seeds if the library is available.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific rescaling and clipping
    as defined in the competition metric description.

    Steps:
    1. Rescale rows to sum to 1.
    2. Clip probabilities to [1e-15, 1-1e-15].
    3. Calculate Log Loss.

    Args:
        y_true: Ground truth labels (1D array of labels or 2D one-hot matrix).
        y_pred: Predicted probabilities (2D array).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays with correct precision
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # Rescale rows to sum to 1
    # Add a small epsilon to sum to avoid division by zero if a row is all zeros (unlikely but safe)
    row_sums = y_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    # sklearn.metrics.log_loss handles the actual log calculation and averaging
    score = log_loss(y_true, y_pred)

    return score


def save_submission(ids, probabilities, class_names, output_path=SUBMISSION_PATH):
    """
    Formats and saves the submission file according to the competition requirements.

    Args:
        ids: Array-like of image IDs.
        probabilities: (N, C) array of predicted probabilities.
        class_names: List of class names corresponding to the columns of probabilities.
        output_path: Path to save the CSV file.
    """
    # Ensure probabilities are numpy array with correct precision
    probs = np.array(probabilities, dtype=FLOAT_PRECISION)

    # Create dictionary for DataFrame construction
    data = {"id": ids}

    # Add class columns
    for i, class_name in enumerate(class_names):
        data[class_name] = probs[:, i]

    df = pd.DataFrame(data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
