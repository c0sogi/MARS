import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
import library.config as config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred, classes=None):
    """
    Computes the multi-class log loss with specific rescaling and clipping
    as defined in the task description.

    The metric requires:
    1. Rescaling probabilities so each row sums to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].
    3. Calculating multi-class log loss.

    Args:
        y_true: Ground truth labels (1D array-like). Can be class indices or strings.
        y_pred: Predicted probabilities (2D array-like).
        classes: List of class labels corresponding to the columns of y_pred.
                 Required if y_true contains string labels.

    Returns:
        float: The calculated log loss.
    """
    # Ensure high precision for calculation
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescaling: Each row is divided by the row sum
    # "The submitted probabilities ... are rescaled prior to being scored"
    row_sums = y_pred.sum(axis=1, keepdims=True)

    # Handle edge case where row sum is 0 to avoid NaN (though unlikely in valid models)
    row_sums[row_sums == 0] = 1.0

    y_pred_norm = y_pred / row_sums

    # 2. Clipping: max(min(p, 1-10^-15), 10^-15)
    # Using constants defined in config
    y_pred_clipped = np.clip(y_pred_norm, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX)

    # 3. Compute Log Loss
    # Sklearn handles the mapping of string labels in y_true to the columns via `classes`
    return log_loss(y_true, y_pred_clipped, labels=classes)
