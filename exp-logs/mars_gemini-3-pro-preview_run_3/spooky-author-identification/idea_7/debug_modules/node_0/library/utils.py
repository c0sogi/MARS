import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(y_pred: np.ndarray) -> np.ndarray:
    """
    Clips predicted probabilities to the range [1e-15, 1-1e-15] to avoid
    extremes of the log function, as specified in the competition metric.

    Args:
        y_pred (np.ndarray): The predicted probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    # The metric specifies max(min(p, 1-10^-15), 10^-15)
    return np.clip(y_pred, 1e-15, 1 - 1e-15)


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the multi-class logarithmic loss after normalizing and clipping
    the predictions, mimicking the competition's scoring mechanism.

    Args:
        y_true (np.ndarray): True class indices or one-hot encoded labels.
        y_pred (np.ndarray): Predicted probabilities (raw).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale: each row is divided by the row sum to ensure they sum to 1
    # We use a safe division to handle potential zero sums, though unlikely with softmax
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred_norm = np.divide(
        y_pred, row_sums, out=np.zeros_like(y_pred), where=row_sums != 0
    )

    # Clip probabilities to avoid log(0) and log(1) extremes
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # Calculate log loss
    # We provide labels explicitly to handle cases where a batch might miss a class
    labels = list(range(Config.NUM_CLASSES))

    return log_loss(y_true, y_pred_clipped, labels=labels)
