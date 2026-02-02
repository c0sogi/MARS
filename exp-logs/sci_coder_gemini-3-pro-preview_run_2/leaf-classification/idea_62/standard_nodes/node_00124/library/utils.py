import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
import library.config as conf


def set_seed(seed=conf.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is not installed or not being used
        pass


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the Multi-class Log Loss with specific normalization and clipping
    rules as defined in the task description.

    Steps:
    1. Rescale: Each row of probabilities is divided by the row sum.
    2. Clip: Probabilities are clipped to [1e-15, 1 - 1e-15].
    3. Score: Calculate log loss.

    Args:
        y_true (array-like): Ground truth labels (1D array of class indices).
        y_pred (array-like): Predicted probabilities (2D array, shape [n_samples, n_classes]).

    Returns:
        float: The calculated log loss.
    """
    # Ensure high precision for metric calculation
    y_pred = np.array(y_pred, dtype=conf.FLOAT_PRECISION)

    # 1. Rescale rows to sum to 1
    # Add a small epsilon to row_sums to avoid division by zero if a row is all zeros
    row_sums = y_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities to avoid log(0) extremes
    # Using constants from config: min=1e-15, max=1-1e-15
    y_pred_clipped = np.clip(y_pred_norm, conf.PROB_CLIP_MIN, conf.PROB_CLIP_MAX)

    # 3. Calculate Log Loss
    # We provide the list of all possible labels to ensure correct handling
    # even if y_true in this batch doesn't contain all classes.
    # Assuming y_true are indices 0..98 corresponding to the columns of y_pred.
    labels = list(range(conf.N_CLASSES))

    score = log_loss(y_true, y_pred_clipped, labels=labels)

    return score
