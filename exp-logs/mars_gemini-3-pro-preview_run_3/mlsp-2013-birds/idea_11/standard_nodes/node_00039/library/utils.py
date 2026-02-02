import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def parse_label_string(label_str, num_classes=Config.NUM_CLASSES):
    """
    Parses a space-separated string of label indices into a binary vector.
    Handles '?' for test data and NaN/empty values.

    Args:
        label_str (str): String containing space-separated class indices (e.g., "0 4").
                         Can be "?" or NaN.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: Binary vector of shape (num_classes,) with 1.0 at present indices.
    """
    vec = np.zeros(num_classes, dtype=np.float32)

    # Handle non-string inputs (e.g. NaN from pandas)
    if not isinstance(label_str, str):
        return vec

    label_str = label_str.strip()

    # Handle unknown labels (test set) or empty strings
    if label_str == "?" or label_str == "":
        return vec

    try:
        # Split by whitespace and convert to integers
        indices = [int(idx) for idx in label_str.split()]
        for idx in indices:
            if 0 <= idx < num_classes:
                vec[idx] = 1.0
    except ValueError:
        # In case of parsing errors (malformed string), return zero vector
        pass

    return vec


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve.
    Robustly handles cases where specific classes are missing in the batch/fold.
    Cite solution_lesson_node_00038.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Check if inputs are valid
    if y_true is None or y_pred is None:
        return 0.0

    # Ensure numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Only calculate if both positive and negative samples exist for this class
        # Cite solution_lesson_node_00038
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                pass

    # If no classes were valid (extremely rare edge case), return default
    if not auc_scores:
        return 0.5

    return float(np.mean(auc_scores))
