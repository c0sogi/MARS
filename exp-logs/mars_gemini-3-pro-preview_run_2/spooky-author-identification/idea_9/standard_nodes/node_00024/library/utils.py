import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probs(probs):
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to avoid log(0) errors.

    Args:
        probs (np.ndarray): The array of probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1 - epsilon)


def get_score(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss.

    Performs rescaling of probabilities (row-wise normalization) and clipping
    prior to scoring, consistent with the competition metric.

    Args:
        y_true (array-like): Ground truth labels (indices or one-hot).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss score.
    """
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale probabilities so each row sums to 1
    # Handle potential division by zero if a row sums to 0 (though unlikely)
    row_sums = y_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums

    # Clip probabilities
    y_pred_clipped = clip_probs(y_pred_rescaled)

    # Calculate log loss
    # We explicitly provide labels to ensure correct handling even if a batch is missing a class
    labels = list(range(Config.num_classes))
    return log_loss(y_true, y_pred_clipped, labels=labels)
