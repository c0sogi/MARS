import os
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to Config.setup_reproducibility to use the centralized configuration.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.setup_reproducibility(seed)


def compute_metric(y_true, y_pred):
    """
    Computes the macro-averaged Area Under the ROC Curve (AUC) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle NaNs in predictions
    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred, nan=0.0)

    try:
        # Calculate macro-averaged ROC AUC
        # This computes AUC for each class and averages them
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for edge cases (e.g., a class has no positive samples in the subset)
        # Calculate AUC per class and average only the valid ones
        scores = []
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            try:
                # Check if class has both 0 and 1 labels
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                continue

        if scores:
            score = np.mean(scores)
        else:
            # If no classes are valid (extremely rare/impossible on full set), return 0.5
            score = 0.5

    return score


def ensure_dir(path):
    """
    Ensures that the specified directory exists.

    Args:
        path (str): The directory path.
    """
    os.makedirs(path, exist_ok=True)


def get_logger(log_file):
    """
    Creates a simple logging function that writes to both console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        callable: A function `log(msg)` that prints and writes to file.
    """
    ensure_dir(os.path.dirname(log_file))

    # Initialize/Clear the log file
    with open(log_file, "w") as f:
        pass

    def log(msg):
        print(msg)
        # Append to file
        with open(log_file, "a") as f:
            f.write(str(msg) + "\n")

    return log
